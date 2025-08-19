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

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

################################ Config ################################
load_dotenv()
# OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL')
OLLAMA_BASE_URL = 'http://localhost:11434'
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
MODEL_NAME = 'qwen3:1.7b'
MODEL_NAME = 'llama3.2'
MODEL_NAME = 'openai/gpt-oss-120b'
MODEL_NAME = 'openai/gpt-oss-20b'

################################ MCP & Tool Configuration ################################
# Configure the Yahoo Finance MCP server to run as a local subprocess

################################ Local Tool Definitions ################################
@tool
def get_topic_headlines(topic: str) -> str:
    """Get news headlines for a specific topic.

    Accepted topics: WORLD, NATION, BUSINESS, TECHNOLOGY, ENTERTAINMENT, SCIENCE, SPORTS, HEALTH
    """


@tool
def search_recent_news(query: str, hours: int = 1) -> str:
    """Search for recent news articles within the last specified hours.

    Args:
        query: Search terms
        hours: Number of hours to look back (1-24)
    """


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression using a safe eval. (e.g., 10 * 5)"""


################################ LLM & Prompt Configuration ################################
# Define a list of tools that are available locally


def create_llm(model_name: str, temperature: float = 0.1) -> ChatOpenAI:
    """Initialize and configure a Chat model from OpenRouter or Ollama."""
    # Example for Ollama (currently overridden by OpenRouter below)
    # llm = ChatOllama(model=model_name, base_url=OLLAMA_BASE_URL, temperature=temperature, streaming=True)


################################ LangGraph Agent Definition ################################
def should_continue(state: MessagesState) -> Literal['tools', '__end__']:
    """Determine whether to call tools or end the conversation."""


def build_state_graph_with(llm, tools) -> StateGraph:
    """Build a StateGraph with the provided LLM and a combined list of tools."""


################################ Chainlit UI & Event Handlers ################################
# Define starter prompts for the Chainlit UI
@cl.set_starters
async def set_starters():
    """Set the starter prompts for the Chainlit UI."""


@cl.on_message
async def on_message(msg: cl.Message):
    """Handle incoming user messages, run the agent, and stream the response."""
