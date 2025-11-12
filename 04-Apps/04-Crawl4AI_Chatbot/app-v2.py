"""
Streamlit Web Chat App with Crawl4AI and LangChain
A simple app to crawl websites, extract content, and chat with it using RAG.
"""

import streamlit as st
import os
from typing import List, Dict, Set
import asyncio
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# Load environment variables from .env file
load_dotenv()

# Core dependencies
from crawl4ai import AsyncWebCrawler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document


# ==================== Configuration ====================

st.set_page_config(page_title='Web Content Chat', page_icon='🌐', layout='wide')


# ==================== Helper Functions ====================


def validate_url(url: str) -> bool:
    """Validate if the provided string is a valid URL."""
    result = urlparse(url)
    return all([result.scheme, result.netloc])


def get_base_domain(url: str) -> str:
    """Extract base domain from URL."""
    parsed = urlparse(url)
    return f'{parsed.scheme}://{parsed.netloc}'


def extract_links(html_content: str, base_url: str) -> Set[str]:
    """Extract all links from HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    links = set()

    for link in soup.find_all('a', href=True):
        href = link['href']
        # Convert relative URLs to absolute
        absolute_url = urljoin(base_url, href)
        # Only include links from the same domain
        if get_base_domain(absolute_url) == get_base_domain(base_url):
            links.add(absolute_url)

    return links


def extract_title_from_html(html_content: str) -> str:
    """Extract title from HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    title_tag = soup.find('title')
    if title_tag:
        return title_tag.get_text().strip()
    return 'Untitled'


async def crawl_website(url: str, max_pages: int = 5, max_depth: int = 2) -> List[Dict[str, str]]:
    """
    Crawl website and extract text content using Crawl4AI.

    Args:
        url: Starting URL to crawl
        max_pages: Maximum number of pages to crawl
        max_depth: Maximum depth to crawl (0 = only start page, 1 = start + linked pages, etc.)

    Returns:
        List of dictionaries containing url and content
    """
    crawled_data = []
    visited_urls = set()
    to_crawl = [(url, 0)]  # (url, depth)
    base_domain = get_base_domain(url)

    async with AsyncWebCrawler(verbose=False) as crawler:
        while to_crawl and len(crawled_data) < max_pages:
            current_url, current_depth = to_crawl.pop(0)

            # Skip if already visited or depth exceeded
            if current_url in visited_urls or current_depth > max_depth:
                continue

            visited_urls.add(current_url)

            # Crawl the page
            result = await crawler.arun(url=current_url)

            if result.success:
                # Extract title from HTML
                title = extract_title_from_html(result.html) if result.html else 'Untitled'

                # Get content (prefer markdown, fallback to cleaned_html)
                content = result.markdown if result.markdown else (result.cleaned_html if result.cleaned_html else '')

                if content:
                    crawled_data.append(
                        {'url': current_url, 'content': content, 'title': title, 'depth': current_depth}
                    )

                # Extract links if we haven't reached max depth
                if current_depth < max_depth and len(crawled_data) < max_pages and result.html:
                    links = extract_links(result.html, current_url)
                    for link in links:
                        if link not in visited_urls and get_base_domain(link) == base_domain:
                            to_crawl.append((link, current_depth + 1))

    return crawled_data


def create_vector_store(crawled_data: List[Dict[str, str]], openai_api_key: str):
    """
    Process crawled content and create FAISS vector store.

    Args:
        crawled_data: List of crawled content dictionaries
        openai_api_key: OpenAI API key

    Returns:
        FAISS vector store and number of chunks
    """
    # Create documents from crawled data
    documents = []
    for item in crawled_data:
        if item['content'].strip():
            doc = Document(
                page_content=item['content'],
                metadata={'source': item['url'], 'title': item.get('title', 'Untitled'), 'depth': item.get('depth', 0)},
            )
            documents.append(doc)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    splits = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(splits, embeddings)

    return vector_store, len(splits)


def create_conversation_chain(vector_store, openai_api_key: str):
    """
    Create a conversational retrieval chain.

    Args:
        vector_store: FAISS vector store
        openai_api_key: OpenAI API key

    Returns:
        ConversationalRetrievalChain
    """
    # Initialize language model
    llm = ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo', openai_api_key=openai_api_key)

    # Create memory for conversation
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

    # Create conversational chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={'k': 3}),
        memory=memory,
        return_source_documents=True,
    )

    return conversation_chain


# ==================== Session State Initialization ====================

if 'conversation' not in st.session_state:
    st.session_state.conversation = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'crawled_data' not in st.session_state:
    st.session_state.crawled_data = None

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

if 'processing' not in st.session_state:
    st.session_state.processing = False


# ==================== Main UI ====================

st.title('🌐 Web Content Chat Assistant')
st.markdown("""
Chat with any website's content! Enter a URL, and I'll crawl it, extract the content, 
and let you ask questions about it.
""")

# ==================== Sidebar - Configuration ====================

with st.sidebar:
    st.header('⚙️ Configuration')

    # Get API Key from .env file
    openai_api_key = os.getenv('OPENAI_API_KEY', '')

    if openai_api_key:
        st.success('✅ OpenAI API Key loaded from .env')
    else:
        st.error('❌ OPENAI_API_KEY not found in .env file')
        st.info('Create a .env file with: OPENAI_API_KEY=your-key-here')

    st.markdown('---')

    # URL input
    st.subheader('🔗 Website to Crawl')
    website_url = st.text_input(
        'Enter Website URL', placeholder='https://example.com', help='Enter the full URL including https://'
    )

    # Crawl settings
    st.subheader('🕷️ Crawl Settings')

    max_pages = st.number_input(
        'Maximum Pages to Crawl',
        min_value=1,
        max_value=50,
        value=5,
        step=1,
        help='Total number of pages to crawl from the website',
    )

    max_depth = st.number_input(
        'Crawl Depth',
        min_value=0,
        max_value=5,
        value=1,
        step=1,
        help='0 = Only start page, 1 = Start page + direct links, 2 = Two levels deep, etc.',
    )

    st.caption(f'📊 Will crawl up to {max_pages} pages at depth {max_depth}')

    st.markdown('---')

    # Crawl button
    crawl_button = st.button('🕷️ Crawl & Process', use_container_width=True, type='primary')

    # Reset button
    if st.button('🔄 Reset', use_container_width=True):
        st.session_state.conversation = None
        st.session_state.chat_history = []
        st.session_state.crawled_data = None
        st.session_state.vector_store = None
        st.rerun()

    st.markdown('---')

    # Display crawl status
    if st.session_state.crawled_data:
        st.success('✅ Website processed!')
        st.metric('Pages Crawled', len(st.session_state.crawled_data))

        # Show crawled content summary
        with st.expander('📄 Crawled Content Summary'):
            for idx, item in enumerate(st.session_state.crawled_data):
                depth_indicator = '📍' * (item.get('depth', 0) + 1)
                st.markdown(f'{depth_indicator} **{idx + 1}. {item.get("title", "Untitled")}**')
                st.caption(f'Depth: {item.get("depth", 0)} | {item["url"]}')
                content_preview = item['content'][:200] + '...' if len(item['content']) > 200 else item['content']
                st.text(content_preview)
                st.markdown('---')


# ==================== Crawl Processing ====================

if crawl_button:
    if not openai_api_key:
        st.error('❌ Please set OPENAI_API_KEY in your .env file')
    elif not website_url:
        st.error('❌ Please enter a website URL')
    elif not validate_url(website_url):
        st.error('❌ Please enter a valid URL (include https://)')
    else:
        with st.spinner(f'🕷️ Crawling website (up to {max_pages} pages at depth {max_depth})...'):
            # Run async crawl
            crawled_data = asyncio.run(crawl_website(website_url, max_pages, max_depth))
            st.session_state.crawled_data = crawled_data

        if crawled_data:
            with st.spinner('🧠 Processing content and creating embeddings...'):
                # Create vector store
                vector_store, num_chunks = create_vector_store(crawled_data, openai_api_key)
                st.session_state.vector_store = vector_store

                # Create conversation chain
                conversation = create_conversation_chain(vector_store, openai_api_key)
                st.session_state.conversation = conversation

                # Reset chat history for new website
                st.session_state.chat_history = []

            st.success(f'✅ Successfully processed {len(crawled_data)} page(s) into {num_chunks} chunks!')
            st.rerun()
        else:
            st.error('❌ No content extracted from the website')


# ==================== Chat Interface ====================

if st.session_state.conversation:
    st.markdown('---')
    st.subheader('💬 Chat with the Website Content')

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

                # Display sources if available
                if message['role'] == 'assistant' and 'sources' in message:
                    with st.expander('📚 Sources'):
                        for source in message['sources']:
                            st.caption(f'- {source}')

    # Chat input
    user_question = st.chat_input('Ask a question about the website content...')

    if user_question:
        # Add user message to chat history
        st.session_state.chat_history.append({'role': 'user', 'content': user_question})

        # Display user message
        with st.chat_message('user'):
            st.markdown(user_question)

        # Generate response
        with st.chat_message('assistant'):
            with st.spinner('Thinking...'):
                # Get response from conversation chain
                response = st.session_state.conversation({'question': user_question})

                answer = response['answer']
                source_documents = response.get('source_documents', [])

                # Display answer
                st.markdown(answer)

                # Extract and display sources
                sources = list(set([doc.metadata.get('source', 'Unknown') for doc in source_documents]))

                if sources:
                    with st.expander('📚 Sources'):
                        for source in sources:
                            st.caption(f'- {source}')

                # Add assistant message to chat history
                st.session_state.chat_history.append({'role': 'assistant', 'content': answer, 'sources': sources})
