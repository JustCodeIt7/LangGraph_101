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

load_dotenv()

from crawl4ai import AsyncWebCrawler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

################################ Helper Functions ################################


def get_base_domain(url: str) -> str:
    """Parse a URL to extract its scheme and network location (domain)."""
    parsed = urlparse(url)
    return f'{parsed.scheme}://{parsed.netloc}'


def extract_links(html_content: str, base_url: str) -> Set[str]:
    """Extract all internal links from a given HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    links = set()
    base_domain = get_base_domain(base_url)

    # Find all anchor tags with an 'href' attribute
    for link in soup.find_all('a', href=True):
        # Convert relative URLs to absolute URLs
        absolute_url = urljoin(base_url, link['href'])
        # Only include links that belong to the same base domain
        if get_base_domain(absolute_url) == base_domain:
            links.add(absolute_url)

    return links


def extract_title(html_content: str) -> str:
    """Extract the page title from the HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    title_tag = soup.find('title')
    # Return the title text or 'Untitled' if not found
    return title_tag.get_text().strip() if title_tag else 'Untitled'


async def crawl_website(url: str, max_pages: int = 5, max_depth: int = 2) -> List[Dict[str, str]]:
    """Crawl a website asynchronously, respecting depth and page limits."""
    crawled_data = []
    visited_urls = set()
    to_crawl = [(url, 0)]  # A queue of (URL, depth) tuples

    # Use an asynchronous web crawler for efficient fetching
    async with AsyncWebCrawler(verbose=False) as crawler:
        # Continue crawling while there are pages in the queue and page limit is not reached
        while to_crawl and len(crawled_data) < max_pages:
            current_url, current_depth = to_crawl.pop(0)

            # Skip if the URL has been visited or the crawl depth is exceeded
            if current_url in visited_urls or current_depth > max_depth:
                continue

            visited_urls.add(current_url)
            # Fetch the content of the current URL
            result = await crawler.arun(url=current_url)

            # Process the result if the crawl was successful
            if result.success:
                title = extract_title(result.html) if result.html else 'Untitled'
                # Prefer Markdown content, fall back to cleaned HTML
                content = result.markdown if result.markdown else result.cleaned_html

                # Add the extracted data to our list if content exists
                if content:
                    crawled_data.append(
                        {'url': current_url, 'content': content, 'title': title, 'depth': current_depth}
                    )

                # If not at max depth, find new links to add to the queue
                if current_depth < max_depth and result.html:
                    links = extract_links(result.html, current_url)
                    for link in links:
                        # Add new, unvisited links to the crawl queue
                        if link not in visited_urls:
                            to_crawl.append((link, current_depth + 1))

    return crawled_data


def create_vector_store(crawled_data: List[Dict[str, str]], api_key: str):
    """Create a FAISS vector store from the crawled website content."""
    documents = []
    # Convert crawled data into LangChain Document objects
    for item in crawled_data:
        # Ensure content is not just whitespace
        if item['content'].strip():
            doc = Document(
                page_content=item['content'],
                metadata={'source': item['url'], 'title': item['title'], 'depth': item['depth']},
            )
            documents.append(doc)

    # Split documents into smaller chunks for effective embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Generate embeddings for the document chunks
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    # Create the FAISS vector store from the chunks and their embeddings
    vector_store = FAISS.from_documents(splits, embeddings)

    return vector_store, len(splits)


def create_conversation_chain(vector_store, api_key: str):
    """Create a conversational retrieval chain with memory."""
    # Initialize the language model
    llm = ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo', openai_api_key=api_key)
    # Set up memory to retain chat history
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

    # Construct the conversational retrieval chain
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={'k': 3}),  # Retrieve top 3 relevant chunks
        memory=memory,
        return_source_documents=True,
    )


################################ UI Component Functions ################################


def initialize_session_state():
    """Initialize session state variables."""
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'crawled_data' not in st.session_state:
        st.session_state.crawled_data = None


def render_sidebar():
    """Render the sidebar with configuration and controls."""
    with st.sidebar:
        st.header('⚙️ Configuration')

        # Load the OpenAI API key from environment variables
        openai_api_key = os.getenv('OPENAI_API_KEY', '')

        # Display a status message for the API key
        if openai_api_key:
            st.success('✅ OpenAI API Key loaded')
        else:
            st.error('❌ OPENAI_API_KEY not found in .env file')

        st.markdown('---')

        st.subheader('🔗 Website to Crawl')
        # Create a text input for the target website URL
        website_url = st.text_input(
            'Enter Website URL',
            placeholder='https://example.com',
            value='https://www.promptingguide.ai',  # Provide a default example URL
        )

        st.subheader('🕷️ Crawl Settings')
        # Create number inputs for crawl parameters
        max_pages = st.number_input('Maximum Pages', min_value=1, max_value=50, value=5)
        max_depth = st.number_input('Crawl Depth', min_value=0, max_value=5, value=1)

        st.caption(f'📊 Will crawl up to {max_pages} pages at depth {max_depth}')
        st.markdown('---')

        # Create the primary button to initiate the crawl and processing
        crawl_button = st.button('🕷️ Crawl & Process', use_container_width=True, type='primary')

        # Create a button to reset the application state
        if st.button('🔄 Reset', use_container_width=True):
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.session_state.crawled_data = None
            st.rerun()  # Rerun the script to reflect the cleared state

        st.markdown('---')

        # Display crawl results in the sidebar after processing is complete
        if st.session_state.crawled_data:
            st.success('✅ Website processed!')
            st.metric('Pages Crawled', len(st.session_state.crawled_data))

            # Show the list of crawled pages in an expander
            with st.expander('📄 Crawled Pages'):
                # Loop through each crawled item to display its details
                for idx, item in enumerate(st.session_state.crawled_data):
                    depth_indicator = '📍' * (item['depth'] + 1)  # Visualize crawl depth
                    st.markdown(f'{depth_indicator} **{idx + 1}. {item["title"]}**')
                    st.caption(item['url'])
                    st.markdown('---')

    return openai_api_key, website_url, max_pages, max_depth, crawl_button


def handle_crawl_and_process(openai_api_key: str, website_url: str, max_pages: int, max_depth: int):
    """Handle the crawl and processing logic."""
    # Validate that the API key and URL are provided
    if not openai_api_key:
        st.error('❌ Please set OPENAI_API_KEY in .env file')
        return

    if not website_url:
        st.error('❌ Please enter a website URL')
        return

    # Show a spinner while the website is being crawled
    with st.spinner(f'🕷️ Crawling website...'):
        crawled_data = asyncio.run(crawl_website(website_url, max_pages, max_depth))
        st.session_state.crawled_data = crawled_data

    # Proceed only if crawling returned some data
    if crawled_data:
        # Show a spinner while the content is being processed into a vector store
        with st.spinner('🧠 Processing content...'):
            vector_store, num_chunks = create_vector_store(crawled_data, openai_api_key)
            conversation = create_conversation_chain(vector_store, openai_api_key)
            st.session_state.conversation = conversation
            st.session_state.chat_history = []  # Clear previous chat history

        st.success(f'✅ Processed {len(crawled_data)} pages into {num_chunks} chunks!')
        st.rerun()  # Rerun to update the UI and display the chat interface
    else:
        st.error('❌ No content extracted from the website')


def render_chat_interface():
    """Render the chat interface."""
    # Display the chat interface only if the conversation chain is ready
    if not st.session_state.conversation:
        return

    st.markdown('---')
    st.subheader('💬 Chat with the Website Content')

    # Display previous messages from the chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            # If the message is from the assistant, show the sources it used
            if message['role'] == 'assistant' and 'sources' in message:
                with st.expander('📚 Sources'):
                    for source in message['sources']:
                        st.caption(f'- {source}')

    # Capture the user's new question from the chat input
    if user_question := st.chat_input('Ask a question about the website content...'):
        # Add the user's question to the chat history
        st.session_state.chat_history.append({'role': 'user', 'content': user_question})

        # Display the user's question in the chat
        with st.chat_message('user'):
            st.markdown(user_question)

        # Generate and display the assistant's response
        with st.chat_message('assistant'):
            with st.spinner('Thinking...'):
                # Send the question to the conversation chain to get a response
                response = st.session_state.conversation({'question': user_question})
                answer = response['answer']
                source_documents = response.get('source_documents', [])

                st.markdown(answer)

                # Extract unique source URLs from the response documents
                sources = list({doc.metadata.get('source', 'Unknown') for doc in source_documents})

                # Display the sources in an expander if any were found
                if sources:
                    with st.expander('📚 Sources'):
                        for source in sources:
                            st.caption(f'- {source}')

                # Add the assistant's response and sources to the chat history
                st.session_state.chat_history.append({'role': 'assistant', 'content': answer, 'sources': sources})


################################ Main Entry Point ################################


def main():
    """Main entry point for the Streamlit application."""
    # Configure the Streamlit page settings
    st.set_page_config(page_title='Web Content Chat', page_icon='🌐', layout='wide')

    # Set up the main title and description of the app
    st.title('🌐 Web Content Chat Assistant')
    st.markdown("Chat with any website's content! Enter a URL, crawl it, and ask questions.")

    # Initialize session state
    initialize_session_state()

    # Render sidebar and get user inputs
    openai_api_key, website_url, max_pages, max_depth, crawl_button = render_sidebar()

    # Handle crawl and process action
    if crawl_button:
        handle_crawl_and_process(openai_api_key, website_url, max_pages, max_depth)

    # Render chat interface
    render_chat_interface()


################################ Application Entry Point ################################

if __name__ == '__main__':
    main()
