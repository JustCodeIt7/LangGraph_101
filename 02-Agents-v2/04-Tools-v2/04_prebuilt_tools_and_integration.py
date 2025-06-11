"""
LangGraph Tools Tutorial - Part 4: Prebuilt Tools and Integration
================================================================

This script demonstrates:
- Using prebuilt tools from model providers
- LangChain tool integrations
- Creating a comprehensive agent with multiple tool types
- Real-world tool composition patterns
"""

import os
from typing import Dict, List, Any
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from datetime import datetime, timedelta
import json
import re

# Set up the model
MODEL_CHOICE = "openai"  # Change to "ollama" to use local model

if MODEL_CHOICE == "openai":
    model = init_chat_model("openai:gpt-4o-mini", temperature=0)
else:
    model = init_chat_model("ollama:llama3.2", temperature=0)

print("=" * 60)
print("LangGraph Tools Tutorial - Part 4: Prebuilt Tools & Integration")
print("=" * 60)

# Example 1: Simulating Prebuilt Tools (Web Search)
print("\n1. Simulated Prebuilt Tools")
print("-" * 30)

# Note: This simulates web search since we don't have actual API access in this demo
@tool("web_search_simulation")
def simulate_web_search(query: str) -> str:
    """Simulate web search results for demonstration purposes."""
    # This simulates what a real web search tool might return
    search_results = {
        "python": "Python is a high-level programming language known for its simplicity and readability. Latest version: Python 3.12",
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs. It's part of the LangChain ecosystem.",
        "ai": "Artificial Intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence.",
        "weather": "Current weather conditions vary by location. Today's forecast shows sunny skies in most regions.",
        "news": "Latest tech news: Major advancements in AI, new programming frameworks released, and cybersecurity updates."
    }
    
    # Find best match
    query_lower = query.lower()
    for key, result in search_results.items():
        if key in query_lower:
            return f"🔍 Search Results for '{query}':\n{result}\n\n[This is a simulated search result]"
    
    return f"🔍 Search Results for '{query}':\nNo specific results found. Try searching for: python, langgraph, ai, weather, or news\n\n[This is a simulated search result]"

# Example 2: Database-like Tools
print("\n2. Database Simulation Tools")
print("-" * 30)

# Simulate a simple database
MOCK_DATABASE = {
    "users": [
        {"id": 1, "name": "Alice Johnson", "email": "alice@example.com", "role": "admin"},
        {"id": 2, "name": "Bob Smith", "email": "bob@example.com", "role": "user"},
        {"id": 3, "name": "Carol Davis", "email": "carol@example.com", "role": "user"},
    ],
    "products": [
        {"id": 1, "name": "Laptop", "price": 999.99, "category": "Electronics"},
        {"id": 2, "name": "Mouse", "price": 29.99, "category": "Electronics"},
        {"id": 3, "name": "Book", "price": 15.99, "category": "Education"},
    ],
    "orders": [
        {"id": 1, "user_id": 1, "product_id": 1, "quantity": 1, "date": "2024-01-15"},
        {"id": 2, "user_id": 2, "product_id": 2, "quantity": 2, "date": "2024-01-16"},
    ]
}

@tool("database_query")
def query_database(table: str, filter_field: str = "", filter_value: str = "") -> str:
    """Query the mock database with optional filtering."""
    if table not in MOCK_DATABASE:
        return f"❌ Table '{table}' not found. Available tables: {list(MOCK_DATABASE.keys())}"
    
    data = MOCK_DATABASE[table]
    
    if filter_field and filter_value:
        filtered_data = []
        for item in data:
            if str(item.get(filter_field, "")).lower() == filter_value.lower():
                filtered_data.append(item)
        data = filtered_data
    
    if not data:
        return f"No records found in table '{table}' with filter {filter_field}={filter_value}"
    
    return f"📊 Query Results from '{table}':\n" + json.dumps(data, indent=2)

@tool("database_stats")
def get_database_statistics() -> str:
    """Get statistics about the database."""
    stats = {}
    for table, data in MOCK_DATABASE.items():
        stats[table] = {
            "count": len(data),
            "columns": list(data[0].keys()) if data else []
        }
    
    return f"📈 Database Statistics:\n" + json.dumps(stats, indent=2)

# Example 3: File System Simulation Tools
print("\n3. File System Simulation Tools")
print("-" * 35)

# Mock file system
MOCK_FILE_SYSTEM = {
    "/home/user/documents/": ["report.pdf", "notes.txt", "presentation.pptx"],
    "/home/user/projects/": ["langgraph_tutorial.py", "data_analysis.ipynb", "README.md"],
    "/home/user/downloads/": ["setup.exe", "image.jpg", "data.csv"],
    "/etc/": ["config.conf", "hosts", "passwd"],
}

@tool("list_directory")
def list_directory(path: str = "/home/user/") -> str:
    """List contents of a directory."""
    # Normalize path
    if not path.endswith("/"):
        path += "/"
    
    if path in MOCK_FILE_SYSTEM:
        files = MOCK_FILE_SYSTEM[path]
        return f"📁 Contents of '{path}':\n" + "\n".join(f"  - {file}" for file in files)
    else:
        available_paths = list(MOCK_FILE_SYSTEM.keys())
        return f"❌ Directory '{path}' not found.\nAvailable directories: {available_paths}"

@tool("find_files")
def find_files(pattern: str) -> str:
    """Find files matching a pattern across the file system."""
    matches = []
    for directory, files in MOCK_FILE_SYSTEM.items():
        for file in files:
            if pattern.lower() in file.lower():
                matches.append(f"{directory}{file}")
    
    if matches:
        return f"🔍 Files matching '{pattern}':\n" + "\n".join(f"  - {match}" for match in matches)
    else:
        return f"❌ No files found matching pattern '{pattern}'"

@tool("file_info")
def get_file_info(filepath: str) -> str:
    """Get information about a specific file."""
    # Extract directory and filename
    parts = filepath.replace("\\", "/").split("/")
    filename = parts[-1]
    directory = "/".join(parts[:-1]) + "/"
    
    if directory in MOCK_FILE_SYSTEM and filename in MOCK_FILE_SYSTEM[directory]:
        # Simulate file info
        extension = filename.split(".")[-1] if "." in filename else "no extension"
        size = len(filename) * 1024  # Mock size calculation
        
        return f"""📄 File Information for '{filepath}':
  - Name: {filename}
  - Directory: {directory}
  - Extension: {extension}
  - Size: {size} bytes (simulated)
  - Type: {get_file_type(extension)}
  - Last Modified: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (simulated)"""
    else:
        return f"❌ File '{filepath}' not found"

def get_file_type(extension: str) -> str:
    """Helper function to determine file type."""
    types = {
        "pdf": "PDF Document",
        "txt": "Text File",
        "py": "Python Script",
        "ipynb": "Jupyter Notebook",
        "md": "Markdown Document",
        "jpg": "JPEG Image",
        "csv": "CSV Data File",
        "exe": "Executable File",
        "conf": "Configuration File"
    }
    return types.get(extension.lower(), "Unknown File Type")

# Example 4: API Integration Tools
print("\n4. API Integration Simulation Tools")
print("-" * 40)

@tool("weather_api")
def get_weather(city: str) -> str:
    """Get weather information for a city (simulated API call)."""
    # Simulate different weather conditions for different cities
    weather_data = {
        "new york": {"temp": 22, "condition": "Partly Cloudy", "humidity": 65},
        "london": {"temp": 15, "condition": "Rainy", "humidity": 80},
        "tokyo": {"temp": 28, "condition": "Sunny", "humidity": 70},
        "paris": {"temp": 18, "condition": "Overcast", "humidity": 75},
    }
    
    city_key = city.lower()
    if city_key in weather_data:
        data = weather_data[city_key]
        return f"""🌤️ Weather in {city.title()}:
  - Temperature: {data['temp']}°C
  - Condition: {data['condition']}
  - Humidity: {data['humidity']}%
  - Last Updated: {datetime.now().strftime('%H:%M')}
  
[This is simulated weather data]"""
    else:
        return f"❌ Weather data not available for '{city}'. Try: New York, London, Tokyo, or Paris"

@tool("news_api")
def get_latest_news(category: str = "technology") -> str:
    """Get latest news headlines by category (simulated API call)."""
    news_data = {
        "technology": [
            "AI Breakthrough: New Language Model Achieves Human-Level Performance",
            "Tech Giants Announce Major Partnership in Quantum Computing",
            "Open Source Framework Revolutionizes Machine Learning Development"
        ],
        "business": [
            "Global Markets Show Strong Growth in Q4",
            "Startup Raises $50M for Sustainable Energy Solutions",
            "E-commerce Platform Reports Record Sales"
        ],
        "science": [
            "Scientists Discover New Exoplanet in Habitable Zone",
            "Medical Breakthrough: Gene Therapy Shows Promise",
            "Climate Research Reveals Surprising Ocean Patterns"
        ]
    }
    
    category_key = category.lower()
    if category_key in news_data:
        headlines = news_data[category_key]
        result = f"📰 Latest {category.title()} News:\n"
        for i, headline in enumerate(headlines, 1):
            result += f"  {i}. {headline}\n"
        result += f"\n[This is simulated news data from {datetime.now().strftime('%Y-%m-%d')}]"
        return result
    else:
        return f"❌ News category '{category}' not available. Try: technology, business, or science"

# Example 5: Comprehensive Agent with All Tool Types
print("\n5. Creating Comprehensive Agent")
print("-" * 35)

# Additional utility tools
@tool("text_analyzer")
def analyze_text(text: str) -> str:
    """Analyze text and provide detailed statistics."""
    words = text.split()
    sentences = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
    paragraphs = len([p for p in text.split('\n\n') if p.strip()])
    
    return f"""📝 Text Analysis Results:
  - Characters: {len(text)}
  - Words: {len(words)}
  - Sentences: {sentences}
  - Paragraphs: {paragraphs}
  - Average words per sentence: {len(words)/max(sentences, 1):.1f}
  - Reading time: {len(words)/200:.1f} minutes (est.)"""

@tool("unit_converter")
def convert_units(value: float, from_unit: str, to_unit: str, unit_type: str) -> str:
    """Convert between different units."""
    conversions = {
        "length": {
            "meter": 1.0,
            "kilometer": 0.001,
            "centimeter": 100.0,
            "inch": 39.3701,
            "foot": 3.28084,
            "yard": 1.09361,
            "mile": 0.000621371
        },
        "weight": {
            "kilogram": 1.0,
            "gram": 1000.0,
            "pound": 2.20462,
            "ounce": 35.274,
            "ton": 0.001
        },
        "temperature": {
            # Special handling needed for temperature
        }
    }
    
    unit_type = unit_type.lower()
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    
    if unit_type == "temperature":
        # Handle temperature conversion separately
        if from_unit == "celsius":
            if to_unit == "fahrenheit":
                result = value * 9/5 + 32
            elif to_unit == "kelvin":
                result = value + 273.15
            else:
                result = value
        elif from_unit == "fahrenheit":
            if to_unit == "celsius":
                result = (value - 32) * 5/9
            elif to_unit == "kelvin":
                result = (value - 32) * 5/9 + 273.15
            else:
                result = value
        else:  # kelvin
            if to_unit == "celsius":
                result = value - 273.15
            elif to_unit == "fahrenheit":
                result = (value - 273.15) * 9/5 + 32
            else:
                result = value
    elif unit_type in conversions:
        conversion_table = conversions[unit_type]
        if from_unit in conversion_table and to_unit in conversion_table:
            # Convert to base unit, then to target unit
            base_value = value / conversion_table[from_unit]
            result = base_value * conversion_table[to_unit]
        else:
            return f"❌ Unknown units for {unit_type}: {from_unit} or {to_unit}"
    else:
        return f"❌ Unknown unit type: {unit_type}. Supported: length, weight, temperature"
    
    return f"🔄 Unit Conversion: {value} {from_unit} = {result:.4f} {to_unit}"

# Create comprehensive agent with all tools
all_tools = [
    # Search and information
    simulate_web_search,
    
    # Database tools
    query_database,
    get_database_statistics,
    
    # File system tools
    list_directory,
    find_files,
    get_file_info,
    
    # API integration tools
    get_weather,
    get_latest_news,
    
    # Utility tools
    text_analyzer,
    unit_converter,
]

comprehensive_agent = create_react_agent(
    model=model,
    tools=all_tools
)

print("Testing comprehensive agent with multiple tool types...")

test_scenarios = [
    "Search for information about LangGraph",
    "Show me all users in the database and get weather for New York",
    "Find all Python files and analyze this text: 'Hello world! This is a test. How are you?'",
    "Convert 100 degrees Fahrenheit to Celsius and get the latest technology news",
]

for i, scenario in enumerate(test_scenarios, 1):
    print(f"\n--- Comprehensive Test {i} ---")
    print(f"Query: {scenario}")
    try:
        response = comprehensive_agent.invoke({
            "messages": [{"role": "user", "content": scenario}]
        })
        print(f"Response: {response['messages'][-1].content}")
    except Exception as e:
        print(f"Error: {e}")

print("\n" + "=" * 60)
print("Part 4 Complete! This covered:")
print("✓ Simulated prebuilt tools (web search)")
print("✓ Database integration tools")
print("✓ File system tools")
print("✓ API integration patterns")
print("✓ Comprehensive agent with multiple tool types")
print("✓ Real-world tool composition patterns")
print("=" * 60)

print("\n" + "=" * 60)
print("🎉 COMPLETE LANGGRAPH TOOLS TUTORIAL FINISHED!")
print("=" * 60)
print("You've learned about:")
print("✅ Basic tool creation and usage")
print("✅ Advanced tool features (state, config, parallel calling)")
print("✅ Error handling strategies")
print("✅ Tool integration patterns")
print("✅ Building comprehensive AI agents")
print("\nNext steps:")
print("• Explore LangChain's tool integrations")
print("• Build your own custom tools")
print("• Combine tools for complex workflows")
print("• Deploy agents in production environments")
print("=" * 60)