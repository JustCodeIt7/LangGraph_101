from typing import Literal, List, Dict
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain.schema.runnable.config import RunnableConfig
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from dotenv import load_dotenv
import os
from pygooglenews import GoogleNews
import yfinance

load_dotenv()
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL')

MODEL_NAME = 'qwen3:1.7b'


# ----------------------------------
# Tool Definition
# ----------------------------------
@tool
def get_top_news() -> str:
    """Get the top news stories for the current country and language."""
    gn = GoogleNews(lang='en', country='US')
    top_stories = gn.top_news()
    articles = []
    articles.extend(f'• {entry.title}\n  {entry.link}' for entry in top_stories['entries'][:10])
    return '\n\n'.join(articles)


@tool
def get_topic_headlines(topic: str) -> str:
    """Get news headlines for a specific topic.

    Accepted topics: WORLD, NATION, BUSINESS, TECHNOLOGY, ENTERTAINMENT, SCIENCE, SPORTS, HEALTH
    """
    valid_topics = ['WORLD', 'NATION', 'BUSINESS', 'TECHNOLOGY', 'ENTERTAINMENT', 'SCIENCE', 'SPORTS', 'HEALTH']
    topic_upper = topic.upper()

    if topic_upper not in valid_topics:
        return f'Invalid topic. Please use one of: {", ".join(valid_topics)}'

    gn = GoogleNews(lang='en', country='US')
    headlines = gn.topic_headlines(topic_upper)
    articles = []
    articles.extend(f'• {entry.title}\n  {entry.link}' for entry in headlines['entries'][:10])
    return f'Headlines for {topic}:\n\n' + '\n\n'.join(articles)


@tool
def get_geo_headlines(location: str) -> str:
    """Get news headlines for a specific geographic location."""
    gn = GoogleNews(lang='en', country='US')
    geo_news = gn.geo_headlines(location)
    articles = []
    articles.extend(f'• {entry.title}\n  {entry.link}' for entry in geo_news['entries'][:10])
    return f'News for {location}:\n\n' + '\n\n'.join(articles)


@tool
def search_news(query: str, when: str = None) -> str:
    """Search for news articles using a custom query.

    Args:
        query: Search terms. Supports Google search operators like:
               - "exact phrase" for phrase search
               - boeing OR airbus for OR search
               - boeing -airbus to exclude terms
               - intitle:boeing to search in titles
               - allintitle:multiple words to search all words in title
        when: Time range filter (e.g., '1h', '12h', '1d', '7d', '1m')
    """
    gn = GoogleNews(lang='en', country='US')
    search_results = gn.search(query, when=when)

    if not search_results['entries']:
        return f'No news found for query: {query}'

    articles = []
    for entry in search_results['entries'][:10]:  # Limit to top 10
        published = getattr(entry, 'published', 'Unknown date')
        articles.append(f'• {entry.title}\n  Published: {published}\n  {entry.link}')

    time_filter = f' (last {when})' if when else ''
    return f"Search results for '{query}'{time_filter}:\n\n" + '\n\n'.join(articles)


@tool
def search_recent_news(query: str, hours: int = 1) -> str:
    """Search for recent news articles within the last specified hours.

    Args:
        query: Search terms
        hours: Number of hours to look back (1-24)
    """
    if hours < 1 or hours > 24:
        return 'Hours must be between 1 and 24'

    gn = GoogleNews(lang='en', country='US')
    search_results = gn.search(query, when=f'{hours}h')

    if not search_results['entries']:
        return f'No recent news found for query: {query} in the last {hours} hours'

    articles = []
    for entry in search_results['entries'][:10]:
        published = getattr(entry, 'published', 'Unknown date')
        articles.append(f'• {entry.title}\n  Published: {published}\n  {entry.link}')

    return f"Recent news for '{query}' (last {hours} hours):\n\n" + '\n\n'.join(articles)


# ----------------------------------
# LLM Setup
# ----------------------------------
# Define all available tools
all_tools = [get_top_news, get_topic_headlines, get_geo_headlines, search_news, search_recent_news]

# System prompt that includes Al Roker personality
SYSTEM_PROMPT = """You are a helpful news assistant with the enthusiastic and warm personality of Al Roker. 
Respond to user queries about news and current events in Al Roker's characteristic style - upbeat, friendly, 
and engaging. Use tools when needed to fetch current news information, then present the results in your 
distinctive voice."""


def create_llm(model_name: str, temperature: float = 0.1) -> ChatOllama:
    """Initialize and configure a ChatOllama instance with system prompt."""
    llm = ChatOllama(model=model_name, base_url=OLLAMA_BASE_URL, temperature=temperature)
    return llm.bind_tools(all_tools)


# Single LLM instance
base_llm = create_llm(MODEL_NAME)
tool_node = ToolNode(tools=all_tools)


# ----------------------------------
# Graph Node Functions
# ----------------------------------
def should_continue(state: MessagesState) -> Literal['tools', '__end__']:
    """Decide whether to invoke tools or end the conversation."""
    last = state['messages'][-1]
    # Limit the number of tool calls to prevent infinite loops
    tool_call_count = sum(bool(hasattr(msg, 'tool_calls') and msg.tool_calls) for msg in state['messages'])

    return 'tools' if last.tool_calls and tool_call_count < 5 else '__end__'


def call_llm(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    """Invoke the LLM with system prompt and append its response."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state['messages']
    response = base_llm.invoke(messages)
    return {'messages': [response]}


# ----------------------------------
# Build State Graph
# ----------------------------------
def build_state_graph() -> StateGraph:
    builder = StateGraph(MessagesState)
    builder.add_node('agent', call_llm)
    builder.add_node('tools', tool_node)

    builder.add_edge(START, 'agent')
    builder.add_conditional_edges('agent', should_continue)
    builder.add_edge('tools', 'agent')

    # Compile with recursion limit
    return builder.compile(checkpointer=None)


agent_graph = build_state_graph()
# print graph structure for debugging
print(agent_graph.get_graph().draw_ascii())


# ----------------------------------
# Chainlit Event Handler
# ----------------------------------
@cl.on_message
async def on_message(msg: cl.Message):
    """Handle incoming messages and stream the response."""
    thread_id = cl.context.session.id
    history = [HumanMessage(content=msg.content)]
    reply = cl.Message(content='')

    try:
        async for chunk in agent_graph.astream(
            {'messages': history},
            config={'configurable': {'thread_id': thread_id}, 'recursion_limit': 50},
        ):
            # Stream content from agent responses
            if 'agent' in chunk and chunk['agent'].get('messages'):
                agent_message = chunk['agent']['messages'][-1]
                if hasattr(agent_message, 'content') and agent_message.content:
                    await reply.stream_token(agent_message.content)
    except Exception as e:
        await reply.stream_token(f'Error: {str(e)}')

    await reply.send()
