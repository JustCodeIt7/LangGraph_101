import os
import re
import asyncio
from typing import List, Dict, Any, Tuple

import streamlit as st
from dotenv import load_dotenv

# LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from bs4 import BeautifulSoup

# Load environment from .env
load_dotenv(override=False)


# ------------- Crawl4AI helpers -------------
def _html_to_text(html: str) -> str:
    if not html:
        return ''
    soup = BeautifulSoup(html, 'html.parser')
    for el in soup(['script', 'style', 'noscript']):
        el.extract()
    text = soup.get_text(separator='\n')
    text = re.sub(r'\s+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def _collect_pages_from_result(result: Any) -> List[Dict[str, str]]:
    pages: List[Dict[str, str]] = []

    def grab_text_from_obj(obj: Any) -> Tuple[str, str]:
        url = ''
        content = ''

        for k in ('url', 'page_url', 'source', 'link'):
            if hasattr(obj, k) and getattr(obj, k):
                url = getattr(obj, k)
                break
            if isinstance(obj, dict) and k in obj and obj[k]:
                url = obj[k]
                break

        for k in ('markdown', 'text', 'cleaned_text', 'content', 'extracted_content', 'page_content', 'md'):
            if hasattr(obj, k) and getattr(obj, k):
                content = getattr(obj, k)
                break
            if isinstance(obj, dict) and k in obj and obj[k]:
                content = obj[k]
                break

        if not content:
            for k in ('html', 'raw_html'):
                if hasattr(obj, k) and getattr(obj, k):
                    content = _html_to_text(getattr(obj, k))
                    break
                if isinstance(obj, dict) and k in obj and obj[k]:
                    content = _html_to_text(obj[k])
                    break

        return (url or '', (content or '').strip())

    if isinstance(result, list):
        for item in result:
            url, content = grab_text_from_obj(item)
            if content:
                pages.append({'url': url, 'content': content})
        return pages

    for attr in ('pages', 'results', 'docs', 'documents', 'items'):
        if hasattr(result, attr) and getattr(result, attr):
            seq = getattr(result, attr)
            for item in seq:
                url, content = grab_text_from_obj(item)
                if content:
                    pages.append({'url': url, 'content': content})
            return pages

    url, content = grab_text_from_obj(result)
    if content:
        pages.append({'url': url, 'content': content})
    return pages


def crawl_site(
    url: str, max_pages: int = 20, max_depth: int = 2, same_domain: bool = True, timeout: int = 30
) -> List[Dict[str, str]]:
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode

    async def _async_crawl():
        async with AsyncWebCrawler() as crawler:
            cfg = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                page_timeout=timeout * 1000,  # Convert seconds to milliseconds
                exclude_external_links=same_domain,  # If True, only crawl pages from the same domain
                word_count_threshold=10,  # Lower threshold to capture more content
            )
            result = await crawler.arun(url=url, config=cfg)
            return result

    # Run async function in event loop (compatible with Streamlit)
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    result = loop.run_until_complete(_async_crawl())
    return _collect_pages_from_result(result if result else [])


# ------------- Vector store and LLM -------------
def build_vectorstore(pages: List[Dict[str, str]], chunk_size: int = 1000, chunk_overlap: int = 150) -> FAISS:
    docs: List[Document] = []
    for p in pages:
        text = (p.get('content') or '').strip()
        if not text:
            continue
        url = p.get('url') or ''
        docs.append(Document(page_content=text, metadata={'source': url}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vs = FAISS.from_documents(split_docs, embeddings)
    return vs


def build_conversational_chain(vectorstore: FAISS, model: str = 'gpt-4o-mini') -> ConversationalRetrievalChain:
    llm = ChatOpenAI(model=model, temperature=0.2)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 4})
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer',  # Explicitly set output key for memory
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key='answer',  # Explicitly set output key for the chain
    )
    return chain


# ------------- Streamlit UI -------------
st.set_page_config(page_title='Website Chat (Crawl4AI + LangChain)', page_icon='🕷️', layout='wide')
st.title('🕷️ Website Chat with Crawl4AI + LangChain')

with st.sidebar:
    st.header('Configuration')
    max_pages = st.number_input('Max pages', min_value=1, max_value=200, value=20, step=1)
    max_depth = st.number_input('Max depth', min_value=1, max_value=5, value=2, step=1)
    same_domain = st.checkbox('Restrict to same domain', value=True)
    chunk_size = st.slider('Chunk size', min_value=300, max_value=2000, value=1000, step=50)
    chunk_overlap = st.slider('Chunk overlap', min_value=0, max_value=400, value=150, step=10)
    model_name = st.selectbox('Chat model', options=['gpt-4o-mini', 'gpt-4o', 'gpt-4o-mini-2024-07-18'], index=0)

# Session state
st.session_state.setdefault('vectorstore', None)
st.session_state.setdefault('chain', None)
st.session_state.setdefault('pages', [])
st.session_state.setdefault('chat_messages', [])  # List[Tuple[role, content]]

# URL input and crawl action
url = st.text_input('Website URL', placeholder='https://example.com', value='https://www.promptingguide.ai')
col_crawl, col_reset = st.columns([1, 1])
with col_crawl:
    crawl_btn = st.button('Crawl and Index', type='primary')
with col_reset:
    if st.button('Reset'):
        st.session_state['vectorstore'] = None
        st.session_state['chain'] = None
        st.session_state['pages'] = []
        st.session_state['chat_messages'] = []

# Crawl and index (no error handling by request)
if crawl_btn:
    pages = crawl_site(
        url.strip(),
        max_pages=int(max_pages),
        max_depth=int(max_depth),
        same_domain=bool(same_domain),
        timeout=30,
    )
    vs = build_vectorstore(pages, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
    chain = build_conversational_chain(vs, model=model_name)

    st.session_state['vectorstore'] = vs
    st.session_state['chain'] = chain
    st.session_state['pages'] = pages
    st.session_state['chat_messages'] = []
    st.success(f'Crawled and indexed {len(pages)} page(s). You can now chat with the site.')


# Content summary
def summarize_pages(pages: List[Dict[str, str]]) -> str:
    total_chars = sum(len((p.get('content') or '')) for p in pages)
    top_samples = []
    for p in pages[:5]:
        u = p.get('url') or ''
        text = (p.get('content') or '').strip()
        sample = (text[:200] + '…') if len(text) > 200 else text
        top_samples.append(f'- {u}\n  Sample: {sample}')
    summary = f'Pages: {len(pages)}\nTotal characters: {total_chars}\nTop pages:\n' + (
        '\n'.join(top_samples) if top_samples else '  (no samples)'
    )
    return summary


if st.session_state['pages']:
    with st.expander('Crawled Content Summary'):
        st.text(summarize_pages(st.session_state['pages']))

st.divider()
st.subheader('Chat with the Website')

# Render chat history
for role, msg in st.session_state['chat_messages']:
    with st.chat_message(role):
        st.markdown(msg)

# Chat input (no error handling by request)
user_msg = st.chat_input("Ask something about the website's content...")
if user_msg:
    st.session_state['chat_messages'].append(('user', user_msg))
    with st.chat_message('user'):
        st.markdown(user_msg)

    res = st.session_state['chain']({'question': user_msg})
    answer = res.get('answer') or res.get('result') or ''
    with st.chat_message('assistant'):
        st.markdown(answer)
        src_docs = res.get('source_documents') or []
        if src_docs:
            with st.expander('Sources'):
                for i, d in enumerate(src_docs, 1):
                    src = d.metadata.get('source', '(unknown)')
                    st.write(f'{i}. {src}')
    st.session_state['chat_messages'].append(('assistant', answer))
