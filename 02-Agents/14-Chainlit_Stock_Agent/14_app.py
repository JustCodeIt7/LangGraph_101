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
import json

# New imports for MCP integration
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv()
# OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL')
OLLAMA_BASE_URL = 'http://localhost:11434'
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
MODEL_NAME = 'qwen3:1.7b'
MODEL_NAME = 'llama3.2'
MODEL_NAME = 'openai/gpt-oss-120b'
MODEL_NAME = 'openai/gpt-oss-20b'

# Configure Yahoo Finance MCP server (ensure the server is installed, e.g., `pip install mcp-server-yfinance`)
server_params = StdioServerParameters(
    command='uvx',
    args=['yfmcp@latest'],  # Alternative module names may be: mcp_server_yahoo_finance
    env=None,
)

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


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression using a safe eval. (e.g., 10 * 5)"""
    try:
        # Use a whitelist to prevent arbitrary code execution
        allowed_chars = set('0123456789+-*/.(). ')
        if not expression or any(c not in allowed_chars for c in expression):
            return 'Invalid expression'
        result = eval(expression)
        return f'{expression} = {result}'
    except Exception as e:
        return f'Calculation error: {e}'


# ----------------------------------
# LLM Setup
# ----------------------------------
# Define local (non-MCP) tools
local_tools = [get_topic_headlines, search_recent_news, calculator]

# System prompt that includes Al Roker personality
SYSTEM_PROMPT = """You are a helpful news assistant with the enthusiastic and warm personality of Al Roker. 
Respond to user queries about news and current events in Al Roker's characteristic style - upbeat, friendly, 
and engaging. Use tools when needed to fetch current news information, then present the results in your 
distinctive voice."""


def create_llm(model_name: str, temperature: float = 0.1) -> ChatOllama:
    """Initialize and configure a Chat model (OpenRouter/OpenAI or Ollama) without binding tools."""
    llm = ChatOllama(model=model_name, base_url=OLLAMA_BASE_URL, temperature=temperature, streaming=True)
    # Base LLM via OpenRouter
    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
        api_key=openrouter_api_key,
        temperature=0,
    )
    return llm


# ----------------------------------
# Graph Node Functions
# ----------------------------------


def should_continue(state: MessagesState) -> Literal['tools', '__end__']:
    """Decide whether to invoke tools or end the conversation."""
    last = state['messages'][-1]
    # Limit the number of tool calls to prevent infinite loops
    tool_call_count = sum(bool(hasattr(msg, 'tool_calls') and msg.tool_calls) for msg in state['messages'])

    return 'tools' if last.tool_calls and tool_call_count < 5 else '__end__'


def build_state_graph_with(llm, tools) -> StateGraph:
    """Build a StateGraph using the provided LLM and tools (including MCP tools)."""
    builder = StateGraph(MessagesState)

    def call_llm_node(state: MessagesState) -> Dict[str, List[BaseMessage]]:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state['messages']
        response = llm.bind_tools(tools).invoke(messages)
        return {'messages': [response]}

    tool_node = ToolNode(tools=tools)

    builder.add_node('agent', call_llm_node)
    builder.add_node('tools', tool_node)

    builder.add_edge(START, 'agent')
    builder.add_conditional_edges('agent', should_continue)
    builder.add_edge('tools', 'agent')

    return builder.compile(checkpointer=None)


# ----------------------------------
# Chainlit Event Handler
# ----------------------------------


# Starter Prompts
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label='Get Tesla stock price?',
            message='What is the current stock price of TSLA?',
            icon='https://cdn.simpleicons.org/tesla',
        ),
        cl.Starter(
            label='Summarize and analysis recent Apple News?',
            message='Can you provide a summary and analysis of the latest news articles about Apple?',
            icon='https://cdn.simpleicons.org/apple',
        ),
        cl.Starter(
            label='Summarize Tech Headlines',
            message='Can you provide a summary of the latest headlines in technology?',
            icon='https://api.iconify.design/eos-icons/ai.svg',
            command='code',
        ),
        cl.Starter(
            label='Summarize Recent News?',
            message='Can you provide a summary of the latest news articles?',
            icon='https://attic.sh/si2powwhauur4mlts7mqn2e3syz3',
        ),
        cl.Starter(
            label='Available tools?',
            message='Can you provide a list of available tools and their descriptions?',
            icon='https://attic.sh/dhbw2bdxwayue0zgof33fxk8jkn1',
        ),
        # calc 10 shares of AAPL
        cl.Starter(
            label='Calculate 10 shares of AAPL',
            message='get the current price of AAPL and then calculate how much 10 shares of AAPL would cost at the current price.',
            icon='https://cdn.simpleicons.org/apple',
        ),
        # test calc tool
        cl.Starter(
            label='What is 10 * 5?',
            message='Calculate 10 * 5',
            icon='https://attic.sh/8v2lqhwii3ape4z4l23vg8s3rez9',
        ),
    ]


@cl.on_message
async def on_message(msg: cl.Message):
    """Handle incoming messages and stream the response with Yahoo Finance MCP tools."""
    thread_id = cl.context.session.id
    history = [HumanMessage(content=msg.content)]
    reply = cl.Message(content='')

    try:
        # Start Yahoo Finance MCP server and load its tools for this interaction
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                mcp_tools = await load_mcp_tools(session)

                # Combine MCP tools with local tools
                tools = local_tools + mcp_tools

                # Create LLM and graph bound to the combined tools
                llm = create_llm(MODEL_NAME)
                agent_graph = build_state_graph_with(llm, tools)

                async for chunk in agent_graph.astream(
                    {'messages': history},
                    config={'configurable': {'thread_id': thread_id}, 'recursion_limit': 50},
                ):
                    # Stream content from agent responses
                    if 'agent' in chunk and chunk['agent'].get('messages'):
                        agent_message = chunk['agent']['messages'][-1]
                        if hasattr(agent_message, 'content') and agent_message.content:
                            await reply.stream_token(f'\n\n[Agent message]\n {agent_message.content}')

                        if tool_calls := getattr(agent_message, 'tool_calls', None) or getattr(
                            agent_message, 'additional_kwargs', {}
                        ).get('tool_calls'):
                            for tc in tool_calls:
                                name = tc.get('name') or (tc.get('function') or {}).get('name')
                                raw_args = tc.get('args') or (tc.get('function') or {}).get('arguments')
                                try:
                                    args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or '{}')
                                except Exception:
                                    args = raw_args
                                # Trim long args for readability
                                await reply.stream_token(f'\n\n[Tool call]\n {name} args: {str(args)[:500]}')

                    # Visibility: stream tool results returned by the tools node
                    if 'tools' in chunk and chunk['tools'].get('messages'):
                        tool_msg = chunk['tools']['messages'][-1]
                        tool_name = getattr(tool_msg, 'name', '') or ''
                        if content := getattr(tool_msg, 'content', ''):
                            await reply.stream_token(f'\n[Tool result]\n {tool_name}: {str(content)[:700]}')
    except Exception as e:
        await reply.stream_token(f'Error: {str(e)}')

    await reply.send()
