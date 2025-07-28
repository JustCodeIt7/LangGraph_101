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
pathlib.Path(__file__).parent.resolve()
os.chdir(pathlib.Path(__file__).parent.resolve())

# # Initialize the LLM using ChatLiteLLM
model_name = 'google/gemini-2.0-flash-001'
# model_name = 'meta-llama/llama-3.2-3b-instruct'
llm = ChatLiteLLM(
    model=f'openrouter/{model_name}',
    temperature=0.1,
    api_base='https://openrouter.ai/api/v1',
    # api_key=api_key,
    openrouter_api_key=api_key,
)


# llm = ChatOllama(
#     model='phi4-mini',
#     temperature=0.1,
#     api_base='http://eos.local:11434',
#     streaming=False,
# )
# %%
# ############### Example 1: Weather Tool with structured response ###############
@tool
def get_weather(location: str) -> str:
    """Get current weather information for a given location.

    Args:
        location: The city name to get weather for
    """
    # Simulate weather API response
    weather_data = {
        'location': location.title(),
        'temperature': 72,
        'condition': 'Sunny',
        'humidity': 45,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }
    return json.dumps(weather_data, indent=2)


# %%
# Test Example 1: Weather query
print('=== Example 1: Weather Tool ===')
agent1 = create_react_agent(model=llm, tools=[get_weather])
result1 = agent1.invoke({'messages': [{'role': 'user', 'content': "What's the weather like in Tokyo?"}]})
print(result1['messages'][-1].content)
print()


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
    try:
        filepath = f'notes/{filename}.txt'
        os.makedirs('notes', exist_ok=True)

        with open(filepath, 'w') as f:
            f.write(f'Note saved at {datetime.now()}\n')
            f.write('-' * 40 + '\n')
            f.write(content)

        return f'Note saved successfully to {filepath}'
    except Exception as e:
        return f'Error saving note: {str(e)}'


# %%
# Test Example 2: File operation
print('=== Example 2: File Operations Tool - File system interactions ===')
agent2 = create_react_agent(model=llm, tools=[save_note])
result2 = agent2.invoke({
    'messages': [
        {
            'role': 'user',
            'content': "Save a note called to file 'shopping_list' with my groceries: milk, eggs, bread, and cheese",
        }
    ]
})
print(result2['messages'][-1].content)
print()


# %%
# ############### Example 3: Stock Price Tool - Real-time financial data with yfinance ###############
@tool
def get_stock_price(ticker: str) -> str:
    """Get current stock price and basic information for a given ticker symbol.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, GOOGL, TSLA)
    """
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker.upper())
        info = stock.info

        # Get current price data
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 'N/A')
        previous_close = info.get('previousClose', 'N/A')
        market_cap = info.get('marketCap', 'N/A')
        company_name = info.get('longName', ticker.upper())

        # Calculate change
        if current_price != 'N/A' and previous_close != 'N/A':
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100
        else:
            change = change_percent = 'N/A'

        stock_data = {
            'ticker': ticker.upper(),
            'company': company_name,
            'current_price': f'${current_price:.2f}' if isinstance(current_price, (int, float)) else current_price,
            'change': f'${change:.2f} ({change_percent:+.2f}%)' if isinstance(change, (int, float)) else change,
            'market_cap': f'${market_cap:,}' if isinstance(market_cap, (int, float)) else market_cap,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        }

        return json.dumps(stock_data, indent=2)

    except ImportError:
        return 'Error: yfinance package not installed. Run: pip install yfinance'
    except Exception as e:
        return f'Error fetching stock data: {str(e)}'


# %%
# Test Example 3: Stock price tool
print('=== Example 3: Stock Price Tool - Real-time financial data with yfinance ===')
agent3 = create_react_agent(model=llm, tools=[get_stock_price])
result4 = agent3.invoke({'messages': [{'role': 'user', 'content': "What's the current price of Apple stock?"}]})
print(result4['messages'][-1].content)
print()
# %%
# Test all tools together: Stock + Weather + File
print('=== All Tools Combined ===')
agent3b = create_react_agent(model=llm, tools=[get_stock_price, get_weather, save_note])
result5 = agent3b.invoke({
    'messages': [
        {
            'role': 'user',
            'content': "Get the stock price for TSLA, check the weather in San Francisco, and save both results to a file called 'market_weather_report'",
        }
    ]
})
print(result5['messages'][-1].content)
print()


# %%
# ############### Example 4: Math Tools - Chained calculations showing tool interoperability ###############
@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number
    """
    return a + b


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together.

    Args:
        a: First number
        b: Second number
    """
    return a * b


# %%
# Test Example 4: Chained math operations
print('=== Example 4: Math Tools - Chained calculations showing tool interoperability ===')
agent4 = create_react_agent(model=llm, tools=[add_numbers, multiply_numbers])
q = 'If I add 10 and 20, then multiply the result by 2, what do I get?'
result6 = agent4.invoke({
    'messages': [
        {
            'role': 'user',
            'content': q,
        }
    ]
})
print(result6['messages'][-1].content)
print()
# %%
# Test complex chained calculation
print('=== Example 4: Complex Chained Calculation - Math Tools ===')
agent4 = create_react_agent(model=llm, tools=[get_stock_price, add_numbers, multiply_numbers])
q = "I own 50 shares of TSLA and 30 shares of AAPL. What is the total value of my portfolio? You'll need to get both stock prices and do multiple calculations."
result7 = agent4.invoke({
    'messages': [
        {
            'role': 'user',
            'content': q,
        }
    ]
})
print(result7['messages'][-1].content)
print()


# %%
# ############### Example 5: SQLite Database Query Tool - LLM-generated SQL queries with mock data ###############


@tool
def query_database(sql_query: str) -> str:
    """Execute a SQL query on the company database and return the results as JSON.

    Args:
        sql_query: The SQL query to execute (SELECT statements only for safety)
    """
    try:
        # Security check - only allow SELECT statements
        if not sql_query.strip().upper().startswith('SELECT'):
            return 'Error: Only SELECT queries are allowed for security reasons.'

        conn = sqlite3.connect('company.db')
        cursor = conn.cursor()

        # Execute the query
        cursor.execute(sql_query)

        # Get column names
        columns = [description[0] for description in cursor.description]

        # Get results
        results = cursor.fetchall()
        conn.close()

        # Format as list of dictionaries
        formatted_results = [dict(zip(columns, row)) for row in results]

        return json.dumps({'query': sql_query, 'results': formatted_results, 'count': len(formatted_results)}, indent=2)

    except Exception as e:
        return f'Error executing query: {str(e)}'


# Create the mock database
# create_mock_database()
# %%
# Test Example 5: Database query tool
print('=== Example 5: SQLite Database Query Tool - LLM-generated SQL queries with mock data ===')


agent5 = create_react_agent(model=llm, tools=[query_database])
# Count number of employees
result8 = agent5.invoke({
    'messages': [{'role': 'user', 'content': 'count the number of employees in the company database'}]
})
print('=== Example 5: Count Employees - Database Query ===')
print(result8['messages'][-1].content)
print()


result8 = agent5.invoke({
    'messages': [
        {'role': 'user', 'content': 'Find all employees in the Engineering department with a salary above 90000'}
    ]
})
print(result8['messages'][-1].content)
print()
# %%
# Test complex database query
print('=== Example 5: Complex Database Query - SQL Analysis ===')
agent5b = create_react_agent(model=llm, tools=[query_database])
result9 = agent5b.invoke({
    'messages': [
        {
            'role': 'user',
            'content': "What's the average salary in each department? Show the results sorted by average salary descending.",
        }
    ]
})
print(result9['messages'][-1].content)
# %%
