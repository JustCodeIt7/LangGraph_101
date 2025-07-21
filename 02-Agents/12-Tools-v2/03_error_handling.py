"""
LangGraph Tools Tutorial - Part 3: Error Handling
==================================================

This script demonstrates different approaches to handling errors in LangGraph tools:
- Default error handling (enabled)
- Disabled error handling
- Custom error handling
- Error recovery strategies
"""

import os
from langgraph.prebuilt import create_react_agent, ToolNode
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model

# Set up the model
MODEL_CHOICE = "openai"  # Change to "ollama" to use local model

if MODEL_CHOICE == "openai":
    model = init_chat_model("openai:gpt-4o-mini", temperature=0)
else:
    model = init_chat_model("ollama:llama3.2", temperature=0)

print("=" * 60)
print("LangGraph Tools Tutorial - Part 3: Error Handling")
print("=" * 60)

# Example 1: Default Error Handling (Enabled)
print("\n1. Default Error Handling (Enabled)")
print("-" * 40)

@tool("divide_numbers")
def divide_numbers(a: float, b: float) -> float:
    """Divide two numbers. Will raise error if dividing by zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero!")
    return a / b

@tool("access_list_item")
def access_list_item(index: int) -> str:
    """Access an item from a predefined list by index."""
    items = ["apple", "banana", "cherry", "date", "elderberry"]
    if index < 0 or index >= len(items):
        raise IndexError(f"Index {index} is out of range. List has {len(items)} items (indices 0-{len(items)-1})")
    return f"Item at index {index}: {items[index]}"

@tool("process_age")
def process_age(age: int) -> str:
    """Process age information with validation."""
    if age < 0:
        raise ValueError("Age cannot be negative!")
    if age > 150:
        raise ValueError("Age seems unrealistic (over 150)")
    if age < 18:
        return f"Minor: {age} years old"
    elif age < 65:
        return f"Adult: {age} years old"
    else:
        return f"Senior: {age} years old"

# Create agent with default error handling (enabled by default)
default_error_agent = create_react_agent(
    model=model,
    tools=[divide_numbers, access_list_item, process_age]
)

print("Testing default error handling...")
test_cases = [
    "Divide 10 by 0",
    "Get the item at index 10 from the list",
    "Process age -5",
    "Divide 20 by 4"  # This should work fine
]

for i, test_case in enumerate(test_cases, 1):
    print(f"\nTest {i}: {test_case}")
    try:
        response = default_error_agent.invoke({
            "messages": [{"role": "user", "content": test_case}]
        })
        print(f"Result: {response['messages'][-1].content}")
    except Exception as e:
        print(f"Exception: {e}")

# Example 2: Disabled Error Handling
print("\n\n2. Disabled Error Handling")
print("-" * 30)

@tool("risky_calculation")
def risky_calculation(x: int, operation: str) -> float:
    """Perform risky calculations that might fail."""
    if operation == "square_root":
        if x < 0:
            raise ValueError("Cannot calculate square root of negative number")
        return x ** 0.5
    elif operation == "inverse":
        if x == 0:
            raise ZeroDivisionError("Cannot calculate inverse of zero")
        return 1 / x
    elif operation == "factorial":
        if x < 0:
            raise ValueError("Factorial not defined for negative numbers")
        if x > 20:
            raise OverflowError("Factorial too large to calculate")
        result = 1
        for i in range(1, x + 1):
            result *= i
        return result
    else:
        raise ValueError(f"Unknown operation: {operation}")

# Create ToolNode with error handling disabled
tool_node_no_errors = ToolNode(
    [risky_calculation],
    handle_tool_errors=False  # Disable error handling
)

# Create agent with disabled error handling
no_error_handling_agent = create_react_agent(
    model=model,
    tools=tool_node_no_errors
)

print("Testing disabled error handling...")
print("Note: This will show actual exceptions being raised")

try:
    response = no_error_handling_agent.invoke({
        "messages": [{"role": "user", "content": "Calculate the square root of -16"}]
    })
    print(f"Result: {response['messages'][-1].content}")
except Exception as e:
    print(f"Caught exception: {type(e).__name__}: {e}")

# Example 3: Custom Error Handling
print("\n\n3. Custom Error Handling")
print("-" * 28)

@tool("file_processor")
def process_file(filename: str, operation: str) -> str:
    """Process a file with various operations."""
    # Simulate file operations that might fail
    valid_files = ["data.txt", "config.json", "results.csv"]
    
    if filename not in valid_files:
        raise FileNotFoundError(f"File '{filename}' not found. Available files: {valid_files}")
    
    if operation == "read":
        return f"Successfully read content from {filename}"
    elif operation == "delete":
        if filename == "config.json":
            raise PermissionError("Cannot delete system configuration file")
        return f"Successfully deleted {filename}"
    elif operation == "backup":
        if filename == "results.csv" and len(filename) > 10:  # Simulate storage full
            raise OSError("Not enough disk space for backup")
        return f"Successfully backed up {filename}"
    else:
        raise ValueError(f"Unknown operation: {operation}")

@tool("network_request")
def make_network_request(url: str) -> str:
    """Make a network request (simulated)."""
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    
    if "timeout" in url:
        raise TimeoutError("Request timed out after 30 seconds")
    
    if "unauthorized" in url:
        raise PermissionError("401 Unauthorized: Invalid credentials")
    
    if "notfound" in url:
        raise FileNotFoundError("404 Not Found: Resource does not exist")
    
    return f"Successfully fetched data from {url}"

# Create different custom error handling strategies
custom_error_messages = {
    "file_processor": "File operation failed. Please check file permissions and availability.",
    "network_request": "Network request failed. Please check your internet connection and try again."
}

# Strategy 1: Simple custom message
tool_node_custom_simple = ToolNode(
    [file_processor, network_request],
    handle_tool_errors="⚠️ Tool execution failed. Please try a different approach or check your inputs."
)

# Strategy 2: Custom function for error handling
def custom_error_handler(error: Exception, tool_name: str) -> str:
    """Custom error handler that provides context-specific messages."""
    error_type = type(error).__name__
    
    if error_type == "FileNotFoundError":
        return f"🔍 File Error: {str(error)}. Please verify the file exists and you have access."
    elif error_type == "PermissionError":
        return f"🔒 Permission Error: {str(error)}. You may need elevated privileges."
    elif error_type == "TimeoutError":
        return f"⏰ Timeout Error: {str(error)}. The operation took too long."
    elif error_type == "ValueError":
        return f"📝 Input Error: {str(error)}. Please check your input parameters."
    else:
        return f"❌ Unexpected error in {tool_name}: {error_type}: {str(error)}"

tool_node_custom_function = ToolNode(
    [file_processor, network_request],
    handle_tool_errors=custom_error_handler
)

# Create agents with different custom error handling
custom_simple_agent = create_react_agent(
    model=model,
    tools=tool_node_custom_simple
)

custom_function_agent = create_react_agent(
    model=model,
    tools=tool_node_custom_function
)

print("Testing custom error handling with simple message...")
try:
    response = custom_simple_agent.invoke({
        "messages": [{"role": "user", "content": "Delete the config.json file"}]
    })
    print(f"Simple custom error result: {response['messages'][-1].content}")
except Exception as e:
    print(f"Exception: {e}")

print("\nTesting custom error handling with function...")
try:
    response = custom_function_agent.invoke({
        "messages": [{"role": "user", "content": "Make a request to https://api.timeout.com/data"}]
    })
    print(f"Function custom error result: {response['messages'][-1].content}")
except Exception as e:
    print(f"Exception: {e}")

# Example 4: Error Recovery Strategies
print("\n\n4. Error Recovery Strategies")
print("-" * 32)

@tool("smart_calculator")
def smart_calculator(expression: str) -> str:
    """Smart calculator that attempts to handle common errors gracefully."""
    try:
        # Remove spaces and convert to lowercase
        expr = expression.replace(" ", "").lower()
        
        # Handle common issues
        if "infinity" in expr or "inf" in expr:
            raise ValueError("Expression results in infinity")
        
        if "/" in expr and "/0" in expr:
            raise ZeroDivisionError("Division by zero detected")
        
        # Evaluate the expression safely
        # Note: In production, use a proper math parser instead of eval
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Expression contains invalid characters")
        
        result = eval(expression)
        
        if abs(result) > 1e10:
            return f"Result is very large: {result:.2e}"
        
        return f"Calculation result: {result}"
        
    except ZeroDivisionError:
        return "⚠️ Error: Division by zero. Try a different expression."
    except ValueError as e:
        return f"⚠️ Error: Invalid expression. {str(e)}"
    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}. Please check your expression."

@tool("robust_data_processor")
def robust_data_processor(data: str, format_type: str) -> str:
    """Process data with built-in error recovery."""
    try:
        if format_type.lower() == "json":
            import json
            # Try to parse as JSON
            parsed = json.loads(data)
            return f"✓ Successfully parsed JSON with {len(parsed)} items"
        elif format_type.lower() == "csv":
            # Simulate CSV processing
            lines = data.split('\n')
            if len(lines) < 2:
                return "⚠️ CSV data seems incomplete, but processed what was available"
            return f"✓ Successfully processed CSV with {len(lines)} rows"
        else:
            return f"⚠️ Unknown format '{format_type}', treated as plain text: {len(data)} characters"
    
    except json.JSONDecodeError:
        return "⚠️ Invalid JSON format, but continuing with partial processing..."
    except Exception as e:
        return f"⚠️ Processing error: {str(e)}, but operation completed with available data"

# Create agent with robust tools
robust_agent = create_react_agent(
    model=model,
    tools=[smart_calculator, robust_data_processor]
)

print("Testing error recovery strategies...")
test_recovery_cases = [
    "Calculate 10 / 0",
    "Calculate 15 + 25 * 2",
    "Process this JSON data: {'name': 'John', 'age': 30} in json format",
    "Process invalid JSON: {broken json} in json format"
]

for i, test_case in enumerate(test_recovery_cases, 1):
    print(f"\nRecovery Test {i}: {test_case}")
    try:
        response = robust_agent.invoke({
            "messages": [{"role": "user", "content": test_case}]
        })
        print(f"Result: {response['messages'][-1].content}")
    except Exception as e:
        print(f"Exception: {e}")

print("\n" + "=" * 60)
print("Part 3 Complete! This covered:")
print("✓ Default error handling (enabled)")
print("✓ Disabled error handling")
print("✓ Custom error messages and handlers")
print("✓ Error recovery strategies")
print("✓ Graceful error handling in tools")
print("=" * 60)