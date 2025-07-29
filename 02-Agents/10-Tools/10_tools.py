# %%
"""
Example 3: Basic LangGraph Tools - Five Practical Examples
This file demonstrates five essential LangGraph tool patterns:
1. Weather Tool - Simulates API calls with structured data
2. File Operations Tool - Shows file system interactions
3. Stock Price Tool - Real-time financial data with yfinance
4. Math Tools - Chained calculations showing tool interoperability
5. SQLite Database Query Tool - LLM-generated SQL queries with mock data
"""

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import json
import os
from datetime import datetime
from rich import print
from langchain_litellm import ChatLiteLLM
from langchain_ollama import ChatOllama
import sqlite3
from utils import create_mock_database
import pathlib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv('OPENROUTER_API_KEY')

# change working directory to the current file's directory
# pathlib.Path(__file__).parent.resolve()
# os.chdir(pathlib.Path(__file__).parent.resolve())

# # Initialize the LLM using ChatLiteLLM
model_name = 'google/gemini-2.0-flash-001'
# model_name = 'meta-llama/llama-3.2-3b-instruct'
llm = ChatLiteLLM(
    model=f'openrouter/{model_name}',
    temperature=0.1,
    api_base='https://openrouter.ai/api/v1',
    openrouter_api_key=api_key,
)

# %%
# ############### Example 1: Weather Tool with structured response ###############
@tool
def get_weather(location: str) -> str:
    """Get current weather information for a given location.

    Args:
        location: The city name to get weather for
    """


# %%
# Test Example 1: Weather query



# %%
# ############### Example 2: File Operations Tool - File system interactions ###############
@tool
def save_note(filename: str, content: str) -> str:
    """
    Save a text note for the user to a file.

    Args:
        filename: Name of the file to save (without extension)
        content: The text content to save
    """


# %%
# Test Example 2: File operation

# %%
# ############### Example 3: Stock Price Tool - Real-time financial data with yfinance ###############
@tool
def get_stock_price(ticker: str) -> str:
    """Get current stock price and basic information for a given ticker symbol.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, GOOGL, TSLA)
    """


# %%
# Test Example 3: Stock price tool

# %%
# Test all tools together: Stock + Weather + File

# %%
# ############### Example 4: Math Tools - Chained calculations showing tool interoperability ###############
@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number
    """


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together.

    Args:
        a: First number
        b: Second number
    """


# %%
# Test Example 4: Chained math operations

# %%
# Test complex chained calculation


# %%
# ############### Example 5: SQLite Database Query Tool - LLM-generated SQL queries with mock data ###############


@tool
def query_database(sql_query: str) -> str:
    """Execute a SQL query on the company database and return the results as JSON.

    Args:
        sql_query: The SQL query to execute (SELECT statements only for safety)
    """


# Create the mock database
# create_mock_database()
# %%
# Test Example 5: Database query tool
print('=== Example 5: SQLite Database Query Tool - LLM-generated SQL queries with mock data ===')


# %%


# %%
# Test complex database query

# %%
