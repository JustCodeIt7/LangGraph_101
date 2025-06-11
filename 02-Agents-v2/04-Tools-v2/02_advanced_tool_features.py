"""
LangGraph Tools Tutorial - Part 2: Advanced Tool Features
==========================================================

This script demonstrates advanced tool features including:
- Hiding arguments from the model (using state and config)
- Disabling parallel tool calling
- Return direct functionality
- Force tool use
"""

import os
from typing import Annotated
from langgraph.prebuilt import create_react_agent, InjectedState, ToolNode
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
import time

# Set up the model
MODEL_CHOICE = "openai"  # Change to "ollama" to use local model

if MODEL_CHOICE == "openai":
    model = init_chat_model("openai:gpt-4o-mini", temperature=0)
else:
    model = init_chat_model("ollama:llama3.2", temperature=0)

print("=" * 60)
print("LangGraph Tools Tutorial - Part 2: Advanced Features")
print("=" * 60)

# Example 1: Hiding Arguments from the Model
print("\n1. Hiding Arguments Using State and Config")
print("-" * 45)

@tool("user_greeting")
def greet_user_with_context(
    greeting_message: str,  # This will be controlled by the LLM
    state: Annotated[AgentState, InjectedState],  # Hidden from LLM - current conversation state
    config: RunnableConfig  # Hidden from LLM - static configuration
) -> str:
    """Greet the user with a personalized message using context from state and config."""
    
    # Extract information from state (conversation history)
    message_count = len(state.get("messages", []))
    
    # Extract information from config (static data passed at invocation)
    user_id = config.get("configurable", {}).get("user_id", "Unknown User")
    session_id = config.get("configurable", {}).get("session_id", "no-session")
    
    return f"""
{greeting_message}
[Context Info]
- User ID: {user_id}
- Session: {session_id}
- Messages in conversation: {message_count}
- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""

@tool("secure_operation")
def perform_secure_operation(
    operation: str,  # LLM controls this
    state: Annotated[AgentState, InjectedState],  # Hidden from LLM
    config: RunnableConfig  # Hidden from LLM
) -> str:
    """Perform a secure operation that requires user authentication context."""
    
    # Check if user is authenticated (from config)
    is_authenticated = config.get("configurable", {}).get("authenticated", False)
    user_role = config.get("configurable", {}).get("user_role", "guest")
    
    if not is_authenticated:
        return "Error: User not authenticated. Please log in first."
    
    if user_role != "admin" and operation.lower() in ["delete", "modify", "admin"]:
        return f"Error: Insufficient permissions. Role '{user_role}' cannot perform '{operation}'"
    
    return f"✓ Successfully performed '{operation}' operation for authenticated {user_role}"

# Create agent with hidden argument tools
hidden_args_agent = create_react_agent(
    model=model,
    tools=[greet_user_with_context, secure_operation]
)

print("Testing tools with hidden arguments...")
try:
    # Note: We pass configuration that tools can access but LLM cannot control
    response = hidden_args_agent.invoke(
        {"messages": [{"role": "user", "content": "Say hello to me and then perform a read operation"}]},
        config={
            "configurable": {
                "user_id": "user123",
                "session_id": "sess_456",
                "authenticated": True,
                "user_role": "admin"
            }
        }
    )
    print(f"Response: {response['messages'][-1].content}")
except Exception as e:
    print(f"Error: {e}")

# Example 2: Disabling Parallel Tool Calling
print("\n\n2. Disabling Parallel Tool Calling")
print("-" * 40)

@tool("slow_calculation")
def slow_calculation(number: int) -> str:
    """Perform a slow calculation that takes time."""
    time.sleep(1)  # Simulate slow operation
    result = number ** 2 + number + 1
    return f"Slow calculation for {number}: {result} (took 1 second)"

@tool("database_query")
def simulate_database_query(table: str) -> str:
    """Simulate a database query that should not run in parallel."""
    time.sleep(0.5)  # Simulate database latency
    return f"Database query result from table '{table}': Found 42 records"

# Create agent with parallel tool calling disabled
tools = [slow_calculation, simulate_database_query]
sequential_agent = create_react_agent(
    model=model.bind_tools(tools, parallel_tool_calls=False),  # Disable parallel execution
    tools=tools
)

print("Testing sequential tool execution...")
start_time = time.time()
try:
    response = sequential_agent.invoke({
        "messages": [{"role": "user", "content": "Calculate the slow calculation for 5 and query the users table"}]
    })
    end_time = time.time()
    print(f"Response: {response['messages'][-1].content}")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
except Exception as e:
    print(f"Error: {e}")

# Example 3: Return Direct Functionality
print("\n\n3. Return Direct Functionality")
print("-" * 35)

@tool(return_direct=True)
def get_current_time() -> str:
    """Get the current time and date. This tool returns results directly."""
    return f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}"

@tool(return_direct=True)
def generate_random_quote() -> str:
    """Generate a random inspirational quote. Returns directly to user."""
    quotes = [
        "The only way to do great work is to love what you do. - Steve Jobs",
        "Innovation distinguishes between a leader and a follower. - Steve Jobs",
        "Life is what happens to you while you're busy making other plans. - John Lennon",
        "The future belongs to those who believe in the beauty of their dreams. - Eleanor Roosevelt",
        "It is during our darkest moments that we must focus to see the light. - Aristotle"
    ]
    import random
    return f"🌟 {random.choice(quotes)}"

# Regular tool for comparison
@tool()
def regular_calculation(x: int, y: int) -> int:
    """Regular calculation tool that allows agent to continue processing."""
    return x * y + 10

# Create agent with return_direct tools
direct_return_agent = create_react_agent(
    model=model,
    tools=[get_current_time, generate_random_quote, regular_calculation]
)

print("Testing return_direct tools...")
try:
    response = direct_return_agent.invoke({
        "messages": [{"role": "user", "content": "What time is it?"}]
    })
    print(f"Direct return response: {response['messages'][-1].content}")
    
    print("\nTesting non-direct tool:")
    response2 = direct_return_agent.invoke({
        "messages": [{"role": "user", "content": "Calculate 7 times 3 plus 10"}]
    })
    print(f"Regular tool response: {response2['messages'][-1].content}")
    
except Exception as e:
    print(f"Error: {e}")

# Example 4: Force Tool Use
print("\n\n4. Force Tool Use")
print("-" * 20)

@tool(return_direct=True)
def mandatory_greeting(user_name: str) -> str:
    """Mandatory greeting tool that must be used."""
    return f"🎉 Welcome {user_name}! You have successfully triggered the mandatory greeting tool!"

@tool()
def optional_calculation(a: int, b: int) -> int:
    """Optional calculation tool."""
    return a + b

# Create agent that forces use of specific tool
forced_tools = [mandatory_greeting, optional_calculation]
forced_agent = create_react_agent(
    model=model.bind_tools(
        forced_tools, 
        tool_choice={"type": "tool", "name": "mandatory_greeting"}  # Force this specific tool
    ),
    tools=forced_tools
)

print("Testing forced tool use...")
try:
    response = forced_agent.invoke({
        "messages": [{"role": "user", "content": "Hi there, I'm Alice and I'd like to do some math"}]
    })
    print(f"Forced tool response: {response['messages'][-1].content}")
except Exception as e:
    print(f"Error: {e}")

# Example with recursion limit to prevent infinite loops
print("\nTesting with recursion limit...")
try:
    # Create agent with recursion limit for safety
    safe_forced_agent = create_react_agent(
        model=model.bind_tools(
            [mandatory_greeting], 
            tool_choice={"type": "tool", "name": "mandatory_greeting"}
        ),
        tools=[mandatory_greeting]
    )
    
    response = safe_forced_agent.invoke(
        {"messages": [{"role": "user", "content": "Hello, I'm Bob"}]},
        {"recursion_limit": 3}  # Limit execution steps
    )
    print(f"Safe forced tool response: {response['messages'][-1].content}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("Part 2 Complete! This covered:")
print("✓ Hiding arguments from model using state/config")
print("✓ Disabling parallel tool calling")
print("✓ Return direct functionality")
print("✓ Forcing tool use with safety measures")
print("=" * 60)