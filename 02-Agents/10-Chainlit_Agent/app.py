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

MODEL_NAME = 'qwen3:4b'
FINETUNE_MODEL_NAME = 'qwen3:4b'
# get env OLLAMA_BASE_URL from environment variables or config


# ----------------------------------
# Tool Definition
# ----------------------------------
@tool
def get_stock_price(symbol: str) -> str:
    """Fetch the current stock price for a given symbol."""
    # Placeholder implementation; replace with actual stock fetching logic
    stock = yfinance.Ticker(symbol)
    price = stock.history(period='1d')['Close'].iloc[-1]
    return f'The current price of {symbol} is ${price:.2f}'


@tool
def get_top_news(topic: str) -> str:
    """Fetch news headlines for a specific topic."""
    gn = GoogleNews()
    headlines = gn.get_top_headlines(q=topic)
    return '\n'.join([f'{article.title}: {article.link}' for article in headlines.articles])


# ----------------------------------
# LLM Setup
# ----------------------------------
def create_llm(model_name: str, temperature: float = 0.0, tags: List[str] = None) -> ChatOllama:
    """Initialize and configure a ChatOllama instance."""
    llm = ChatOllama(model=model_name, base_url=OLLAMA_BASE_URL, temperature=temperature)
    if tags:
        llm = llm.with_config(tags=tags)
    return llm.bind_tools([get_stock_price, get_top_news])


# Instantiate base and final LLMs
base_llm = create_llm(MODEL_NAME, temperature=0.1)
# Create final LLM without tool binding to avoid callback issues
final_llm = ChatOllama(model=FINETUNE_MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.1)

tool_node = ToolNode(tools=[get_stock_price, get_top_news])


# ----------------------------------
# Graph Node Functions
# ----------------------------------
def should_route(state: MessagesState) -> Literal['tools', 'final']:
    """Decide whether to invoke tools or finish the conversation."""
    last = state['messages'][-1]
    # Limit the number of tool calls to prevent infinite loops
    tool_call_count = sum(1 for msg in state['messages'] if hasattr(msg, 'tool_calls') and msg.tool_calls)

    if last.tool_calls and tool_call_count < 5:  # Max 5 tool calls
        return 'tools'
    return 'final'


def call_base_llm(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    """Invoke the base LLM and append its response."""
    response = base_llm.invoke(state['messages'])
    return {'messages': [response]}


def call_final_llm(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    """Invoke the final LLM to rewrite the last message."""
    last = state['messages'][-1]
    rewritten = final_llm.invoke([
        SystemMessage(content='Rewrite this in the voice of Al Roker.'),
        HumanMessage(content=last.content),
    ])
    rewritten.id = last.id  # preserve original ID
    return {'messages': [rewritten]}


# ----------------------------------
# Build State Graph
# ----------------------------------
def build_state_graph() -> StateGraph:
    builder = StateGraph(MessagesState)
    builder.add_node('agent', call_base_llm)
    builder.add_node('tools', tool_node)
    builder.add_node('final', call_final_llm)

    builder.add_edge(START, 'agent')
    builder.add_conditional_edges('agent', should_route)
    builder.add_edge('tools', 'agent')
    builder.add_edge('final', END)

    # Compile with recursion limit
    g = builder.compile(checkpointer=None)
    return g


agent_graph = build_state_graph()
# print graph structure for debugging
print(agent_graph.get_graph().draw_ascii())


# ----------------------------------
# Chainlit Event Handler
# ----------------------------------
@cl.on_message
async def on_message(msg: cl.Message):
    """Handle incoming messages and stream the final response."""
    thread_id = cl.context.session.id
    history = [HumanMessage(content=msg.content)]
    reply = cl.Message(content='')

    try:
        async for chunk in agent_graph.astream(
            {'messages': history},
            config={'configurable': {'thread_id': thread_id}, 'recursion_limit': 50},
        ):
            if 'final' in chunk and chunk['final'].get('messages'):
                final_message = chunk['final']['messages'][-1]
                if hasattr(final_message, 'content') and final_message.content:
                    await reply.stream_token(final_message.content)
    except Exception as e:
        await reply.stream_token(f'Error: {str(e)}')

    await reply.send()
