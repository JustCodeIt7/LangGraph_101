"""
LangGraph Tools Tutorial - Part 1: Basic Tool Creation and Usage
================================================================

This script demonstrates the fundamentals of creating and using tools in LangGraph,
including simple function tools and customized tools with the @tool decorator.
"""

import os
from typing import Annotated
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from phoenix.otel import register

# configure the Phoenix tracer
tracer_provider = register(
    project_name='langgraph_101',  # Default is 'default'
    auto_instrument=True,  # Auto-instrument your app based on installed OI dependencies
)
# Set up the model - you can switch between OpenAI and Ollama
MODEL_CHOICE = "openai"  # Change to "ollama" to use local model

if MODEL_CHOICE == "openai":
    # Make sure to set your OPENAI_API_KEY environment variable
    model = init_chat_model("openai:gpt-4o-mini", temperature=0)
else:
    # Using Ollama local model
    model = init_chat_model("ollama:llama3.2", temperature=0)

print("=" * 60)
print("LangGraph Tools Tutorial - Part 1: Basic Tools")
print("=" * 60)

# Example 1: Simple Function Tools
print("\n1. Simple Function Tools")
print("-" * 30)

def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

def calculate_area_rectangle(length: float, width: float) -> float:
    """Calculate the area of a rectangle."""
    return length * width

# Create agent with simple function tools
simple_agent = create_react_agent(
    model=model,
    tools=[add_numbers, multiply_numbers, calculate_area_rectangle]
)

print("Testing simple function tools...")
try:
    response = simple_agent.invoke({
        "messages": [{"role": "user", "content": "What is 15 + 27, and what is 8 * 6?"}]
    })
    print(f"Response: {response['messages'][-1].content}")
except Exception as e:
    print(f"Error: {e}")

# Example 2: Using @tool Decorator for More Control
print("\n\n2. Using @tool Decorator")
print("-" * 30)

@tool("temperature_converter", parse_docstring=True)
def convert_temperature(temperature: float, from_unit: str, to_unit: str) -> str:
    """Convert temperature between Celsius, Fahrenheit, and Kelvin.

    Args:
        temperature: The temperature value to convert
        from_unit: Source unit (celsius, fahrenheit, kelvin)
        to_unit: Target unit (celsius, fahrenheit, kelvin)
    """
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    
    # Convert to Celsius first
    if from_unit == "fahrenheit":
        celsius = (temperature - 32) * 5/9
    elif from_unit == "kelvin":
        celsius = temperature - 273.15
    else:  # celsius
        celsius = temperature
    
    # Convert from Celsius to target
    if to_unit == "fahrenheit":
        result = celsius * 9/5 + 32
    elif to_unit == "kelvin":
        result = celsius + 273.15
    else:  # celsius
        result = celsius
    
    return f"{temperature}° {from_unit.title()} = {result:.2f}° {to_unit.title()}"

@tool("string_analyzer")
def analyze_string(text: str) -> dict:
    """Analyze a string and return various statistics about it."""
    return {
        'length': len(text),
        'word_count': len(text.split()),
        'vowel_count': sum(char in 'aeiou' for char in text.lower()),
        'consonant_count': sum(bool(char.isalpha() and char not in 'aeiou') for char in text.lower()),
        'uppercase_count': sum(bool(char.isupper()) for char in text),
        'lowercase_count': sum(bool(char.islower()) for char in text),
    }

# Create agent with decorated tools
decorated_agent = create_react_agent(
    model=model,
    tools=[convert_temperature, analyze_string]
)

print("Testing decorated tools...")
try:
    response = decorated_agent.invoke({
        "messages": [{"role": "user", "content": "Convert 100 degrees Fahrenheit to Celsius and analyze the string 'Hello World'"}]
    })
    print(f"Response: {response['messages'][-1].content}")
except Exception as e:
    print(f"Error: {e}")

# Example 3: Custom Input Schema with Pydantic
print("\n\n3. Custom Input Schema with Pydantic")
print("-" * 40)

class CalculatorInputSchema(BaseModel):
    """Input schema for calculator operations"""
    operation: str = Field(description="The operation to perform: add, subtract, multiply, divide")
    x: float = Field(description="First number")
    y: float = Field(description="Second number")

@tool("advanced_calculator", args_schema=CalculatorInputSchema)
def advanced_calculator(operation: str, x: float, y: float) -> str:
    """Perform advanced calculator operations with proper error handling."""
    operation = operation.lower()
    
    if operation == "add":
        result = x + y
    elif operation == "subtract":
        result = x - y
    elif operation == "multiply":
        result = x * y
    elif operation == "divide":
        if y == 0:
            return "Error: Cannot divide by zero!"
        result = x / y
    else:
        return f"Error: Unknown operation '{operation}'. Supported: add, subtract, multiply, divide"
    
    return f"{x} {operation} {y} = {result}"

class PersonInputSchema(BaseModel):
    """Input schema for person information"""
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age in years")
    occupation: str = Field(description="Person's job or occupation")

@tool("create_person_profile", args_schema=PersonInputSchema)
def create_person_profile(name: str, age: int, occupation: str) -> str:
    """Create a formatted person profile."""
    return f"""
    === PERSON PROFILE ===
    Name: {name}
    Age: {age} years old
    Occupation: {occupation}
    Profile ID: {hash(f"{name}{age}{occupation}") % 10000}
    """

# Create agent with Pydantic schema tools
schema_agent = create_react_agent(
    model=model,
    tools=[advanced_calculator, create_person_profile]
)

print("Testing Pydantic schema tools...")
try:
    response = schema_agent.invoke({
        "messages": [{"role": "user", "content": "Calculate 15.5 divided by 3.2 and create a profile for John Smith, age 30, software engineer"}]
    })
    print(f"Response: {response['messages'][-1].content}")
except Exception as e:
    print(f"Error: {e}")

# Example 4: File and Data Processing Tools
print("\n\n4. File and Data Processing Tools")
print("-" * 40)

@tool("word_frequency_analyzer")
def analyze_word_frequency(text: str, top_n: int = 5) -> dict:
    """Analyze word frequency in a given text and return top N most common words."""
    import re
    from collections import Counter
    
    # Clean and tokenize text
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = Counter(words)
    
    return {
        "total_words": len(words),
        "unique_words": len(word_freq),
        "top_words": dict(word_freq.most_common(top_n))
    }

@tool("list_processor")
def process_number_list(numbers_str: str, operation: str) -> str:
    """Process a comma-separated list of numbers with various operations."""
    try:
        numbers = [float(x.strip()) for x in numbers_str.split(',')]
        
        if operation.lower() == "sum":
            result = sum(numbers)
        elif operation.lower() == "average":
            result = sum(numbers) / len(numbers)
        elif operation.lower() == "max":
            result = max(numbers)
        elif operation.lower() == "min":
            result = min(numbers)
        elif operation.lower() == "count":
            result = len(numbers)
        else:
            return f"Unknown operation: {operation}. Supported: sum, average, max, min, count"
        
        return f"Operation '{operation}' on [{numbers_str}] = {result}"
    except ValueError:
        return "Error: Please provide valid comma-separated numbers"

# Create agent with data processing tools
data_agent = create_react_agent(
    model=model,
    tools=[analyze_word_frequency, process_number_list]
)

print("Testing data processing tools...")
try:
    response = data_agent.invoke({
        "messages": [{"role": "user", "content": "Analyze the word frequency in 'The quick brown fox jumps over the lazy dog' and find the sum of these numbers: 10, 20, 30, 40, 50"}]
    })
    print(f"Response: {response['messages'][-1].content}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("Part 1 Complete! This covered:")
print("✓ Simple function tools")
print("✓ @tool decorator usage")
print("✓ Custom Pydantic schemas")
print("✓ Data processing tools")
print("=" * 60)