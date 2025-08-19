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
import pandas as pd

load_dotenv()
# OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL')
OLLAMA_BASE_URL = 'http://localhost:11434'
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
MODEL_NAME = 'qwen3:1.7b'
MODEL_NAME = 'llama3.2'
MODEL_NAME = 'openai/gpt-oss-120b'

# ----------------------------------
# Tool Definition
# ----------------------------------

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


# New tool: Fetch latest stock prices as a DataFrame (returned as CSV text for LLM consumption)
@tool
def get_stock_prices(ticker: str, period: str = '1d', interval: str = '1m', rows: int = 20) -> str:
    """Get recent stock price data for a ticker as a DataFrame (CSV string).

    Args:
        ticker: Stock symbol (e.g., AAPL, MSFT)
        period: Data period (e.g., '1d', '5d', '1mo', '3mo', '1y')
        interval: Data interval (e.g., '1m', '5m', '15m', '1h', '1d')
        rows: Number of most recent rows to return
    Returns:
        CSV string representing a pandas DataFrame with timestamp and OHLCV columns.
    """
    try:
        df = yfinance.download(tickers=ticker, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty:
            return f'No price data found for {ticker} (period={period}, interval={interval}).'

        # Keep the last N rows, reset index so Datetime becomes a column
        df = df.tail(max(1, rows)).reset_index()

        # Normalize potential MultiIndex columns (e.g., if dividends/splits present)
        df.columns = [col if isinstance(col, str) else ' '.join(map(str, col)).strip() for col in df.columns]

        # Ensure standard column names exist if available
        # Expected columns often include: Datetime/Date, Open, High, Low, Close, Adj Close, Volume
        csv_text = df.to_csv(index=False)
        header = f'Price data for {ticker.upper()} (period={period}, interval={interval}, rows={len(df)}):\n'
        return header + csv_text
    except Exception as e:
        return f'Error fetching data for {ticker}: {e}'


# ----------------------------------
# LLM Setup
# ----------------------------------
# Define all available tools
all_tools = [get_topic_headlines, search_recent_news, get_stock_prices]

# System prompt that includes Al Roker personality
SYSTEM_PROMPT = """You are a helpful news assistant with the enthusiastic and warm personality of Al Roker. 
Respond to user queries about news and current events in Al Roker's characteristic style - upbeat, friendly, 
and engaging. Use tools when needed to fetch current news information, then present the results in your 
distinctive voice."""


def create_llm(model_name: str, temperature: float = 0.1) -> ChatOllama:
    """Initialize and configure a ChatOllama instance with system prompt."""
    llm = ChatOllama(model=model_name, base_url=OLLAMA_BASE_URL, temperature=temperature, streaming=True)
    # Base LLM
    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
        api_key=openrouter_api_key,
        temperature=0,
    )

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
