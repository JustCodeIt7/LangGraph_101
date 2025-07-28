"""
Example 3: Basic LangGraph Tools - Two Practical Examples
This file demonstrates two essential LangGraph tool patterns:
1. Weather Tool - Simulates API calls with structured data
2. File Operations Tool - Shows file system interactions
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

# Create agent with both tools
tools = [get_weather, save_note]
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