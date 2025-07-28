"""
Example 3: Basic LangGraph Tools - Three Practical Examples
This file demonstrates three essential LangGraph tool patterns:
1. Weather Tool - Simulates API calls with structured data
2. File Operations Tool - Shows file system interactions
3. Stock Price Tool - Real-time financial data with yfinance
"""

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
import json
import os
from datetime import datetime

# Initialize the LLM
llm = init_chat_model("ollama:llama3.2", temperature=0)

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

# Create agent with all three tools
tools = [get_weather, save_note, get_stock_price]
agent = create_react_agent(model=llm, tools=tools)

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