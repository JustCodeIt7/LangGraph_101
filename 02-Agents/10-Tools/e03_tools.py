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
from langchain.chat_models import init_chat_model
import json
import os
from datetime import datetime
from rich import print
from langchain_litellm import ChatLiteLLM
import sqlite3

# Initialize the LLM
# llm = init_chat_model(model='llama3.2', model_provider='ollama', temperature=0, api_base='http://eos.local:11434')
llm = init_chat_model(model='ollama:phi4-mini', temperature=0, api_base='http://eos.local:11434')

# %%
# Example 1: Weather Tool with structured response
@tool
def get_weather(location: str) -> str:
    """Get current weather information for a given location.
    
    Args:
        location: The city name to get weather for
    """
    # Simulate weather API response
    weather_data = {
        "location": location.title(),
        "temperature": 72,
        "condition": "Sunny",
        "humidity": 45,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    return json.dumps(weather_data, indent=2)

# %%
# Example 2: File operations tool
@tool
def save_note(filename: str, content: str) -> str:
    """Save a text note to a file.
    
    Args:
        filename: Name of the file to save (without extension)
        content: The text content to save
    """
    try:
        filepath = f"notes/{filename}.txt"
        os.makedirs("notes", exist_ok=True)
        
        with open(filepath, "w") as f:
            f.write(f"Note saved at {datetime.now()}\n")
            f.write("-" * 40 + "\n")
            f.write(content)
        
        return f"Note saved successfully to {filepath}"
    except Exception as e:
        return f"Error saving note: {str(e)}"

# %%
# Example 3: Stock price tool using yfinance
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
            "ticker": ticker.upper(),
            "company": company_name,
            "current_price": f"${current_price:.2f}" if isinstance(current_price, (int, float)) else current_price,
            "change": f"${change:.2f} ({change_percent:+.2f}%)" if isinstance(change, (int, float)) else change,
            "market_cap": f"${market_cap:,}" if isinstance(market_cap, (int, float)) else market_cap,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        
        return json.dumps(stock_data, indent=2)
        
    except ImportError:
        return "Error: yfinance package not installed. Run: pip install yfinance"
    except Exception as e:
        return f"Error fetching stock data: {str(e)}"

# %%
# Example 4: Math tools for chained calculations
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


@tool
def calculate_stock_value(shares: float, price_per_share: float) -> str:
    """Calculate the total value of stock shares.

    Args:
        shares: Number of shares owned
        price_per_share: Current price per share
    """
    try:
        total_value = shares * price_per_share
        return f'Total stock value: ${total_value:.2f} ({shares} shares at ${price_per_share:.2f} per share)'
    except Exception as e:
        return f'Error calculating stock value: {str(e)}'

# %%
# Example 5: SQLite Database Query Tool
def create_mock_database():
    """Create a mock SQLite database with fake employee data for testing."""
    conn = sqlite3.connect('company.db')
    cursor = conn.cursor()
    
    # Create employees table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            salary REAL NOT NULL,
            hire_date TEXT NOT NULL
        )
    ''')
    
    # Insert mock data
    employees = [
        (1, 'Alice Johnson', 'Engineering', 95000, '2020-01-15'),
        (2, 'Bob Smith', 'Marketing', 75000, '2019-03-22'),
        (3, 'Carol Davis', 'Engineering', 105000, '2018-07-10'),
        (4, 'David Wilson', 'Sales', 65000, '2021-05-30'),
        (5, 'Eve Brown', 'Marketing', 80000, '2020-11-12'),
        (6, 'Frank Miller', 'Engineering', 92000, '2019-09-05'),
        (7, 'Grace Lee', 'HR', 70000, '2021-02-18'),
        (8, 'Henry Taylor', 'Sales', 72000, '2020-08-24')
    ]
    
    cursor.executemany('INSERT OR REPLACE INTO employees VALUES (?, ?, ?, ?, ?)', employees)
    conn.commit()
    conn.close()

@tool
def query_database(sql_query: str) -> str:
    """Execute a SQL query on the company database and return the results as JSON.
    
    Args:
        sql_query: The SQL query to execute (SELECT statements only for safety)
    """
    try:
        # Security check - only allow SELECT statements
        if not sql_query.strip().upper().startswith('SELECT'):
            return "Error: Only SELECT queries are allowed for security reasons."
        
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
        
        return json.dumps({
            "query": sql_query,
            "results": formatted_results,
            "count": len(formatted_results)
        }, indent=2)
        
    except Exception as e:
        return f"Error executing query: {str(e)}"

# Create the mock database
create_mock_database()

# %%
# Create agent with all tools
tools = [get_weather, save_note, get_stock_price, add_numbers, multiply_numbers, calculate_stock_value, query_database]
agent = create_react_agent(model=llm, tools=tools)
# agent = ChatLiteLLM(model='ollama/llama3.2', api_base='http://eos.local:11434', temperature=0, tools=tools)


# Test Example 1: Weather query
print("=== Example 1: Weather Tool ===")
result1 = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather like in Tokyo?"}]
})
print(result1['messages'][-1].content)
print()

# Test Example 2: File operation
print("=== Example 2: File Operations Tool ===")
result2 = agent.invoke({
    "messages": [{"role": "user", "content": "Save a note called 'shopping' with my grocery list: milk, eggs, bread, and cheese"}]
})
print(result2['messages'][-1].content)
print()

# Test combined usage
print("=== Combined Usage: Weather + File Save ===")
result3 = agent.invoke({
    "messages": [{"role": "user", "content": "Get the weather in Paris and save it to a file called 'paris_weather'"}]
})
print(result3['messages'][-1].content)
print()

# Test Example 3: Stock price tool
print("=== Example 3: Stock Price Tool ===")
result4 = agent.invoke({
    "messages": [{"role": "user", "content": "What's the current price of Apple stock?"}]
})
print(result4['messages'][-1].content)
print()

# Test all tools together
print("=== All Tools Combined ===")
result5 = agent.invoke({
    "messages": [{"role": "user", "content": "Get the stock price for TSLA, check the weather in San Francisco, and save both results to a file called 'market_weather_report'"}]
})
print(result5['messages'][-1].content)
print()

# Test Example 4: Chained math operations
print('=== Example 4: Chained Math Tools ===')
result6 = agent.invoke({
    'messages': [
        {
            'role': 'user',
            'content': 'Calculate the total value of 100 shares of Apple stock. First get the current price, then multiply it by 100.',
        }
    ]
})
print(result6['messages'][-1].content)
print()

# Test complex chained calculation
print('=== Complex Chained Calculation ===')
result7 = agent.invoke({
    'messages': [
        {
            'role': 'user',
            'content': "I own 50 shares of TSLA and 30 shares of AAPL. What is the total value of my portfolio? You'll need to get both stock prices and do multiple calculations.",
        }
    ]
})
print(result7['messages'][-1].content)
print()

# Test Example 5: Database query tool
print("=== Example 5: Database Query Tool ===")
result8 = agent.invoke({
    "messages": [{"role": "user", "content": "Find all employees in the Engineering department with a salary above 90000"}]
})
print(result8['messages'][-1].content)
print()

# Test complex database query
print("=== Complex Database Query ===")
result9 = agent.invoke({
    "messages": [{"role": "user", "content": "What's the average salary in each department? Show the results sorted by average salary descending."}]
})
print(result9['messages'][-1].content)