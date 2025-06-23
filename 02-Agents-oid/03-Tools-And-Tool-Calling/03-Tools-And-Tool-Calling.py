#!/usr/bin/env python3
"""
LangGraph Tools and Tool Calling Tutorial

This interactive script demonstrates:
1. Creating custom tools for LangGraph agents
2. Tool calling and execution
3. Agent decision-making with multiple tools
4. Error handling in tool execution
5. Tool result integration into conversation flow

Topics Covered:
- Defining custom tools with @tool decorator
- ToolExecutor and ToolInvocation
- Agent state management with tools
- Conditional routing based on tool results
- Interactive tool demonstration
"""

import json
import operator
import random
from datetime import datetime
from typing import TypedDict, Annotated, List, Union

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolExecutor, ToolInvocation

print("=" * 60)
print("LangGraph Tools and Tool Calling Tutorial")
print("=" * 60)

# ============================================================================
# PART 1: DEFINE CUSTOM TOOLS
# ============================================================================

print("\n🔧 PART 1: Defining Custom Tools")
print("-" * 40)

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"Current date and time: {current_time}"

@tool
def calculate_math(expression: str) -> str:
    """
    Calculate a mathematical expression safely.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 3 * 4")
    
    Returns:
        The result of the calculation as a string
    """
    try:
        # Safe evaluation - only allow basic math operations
        allowed_operators = {'+', '-', '*', '/', '(', ')', ' ', '.'}
        allowed_chars = set('0123456789') | allowed_operators
        
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression. Only numbers and +, -, *, /, (), . are allowed."
        
        result = eval(expression)
        return f"The result of '{expression}' is: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"

@tool
def get_random_fact() -> str:
    """Get a random interesting fact."""
    facts = [
        "Octopuses have three hearts and blue blood.",
        "A group of flamingos is called a 'flamboyance'.",
        "Honey never spoils - archaeologists have found edible honey in ancient Egyptian tombs.",
        "Bananas are berries, but strawberries aren't.",
        "The shortest war in history lasted only 38-45 minutes.",
        "A shrimp's heart is in its head.",
        "Polar bears have black skin under their white fur.",
        "The human brain uses about 20% of the body's total energy."
    ]
    fact = random.choice(facts)
    return f"Random fact: {fact}"

@tool
def weather_simulator(location: str) -> str:
    """
    Simulate weather information for a given location.
    
    Args:
        location: The location to get weather for
    
    Returns:
        Simulated weather information
    """
    weather_conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "windy"]
    temperature = random.randint(-10, 35)  # Celsius
    condition = random.choice(weather_conditions)
    
    return f"Weather in {location}: {condition.title()}, {temperature}°C"

@tool
def text_analyzer(text: str) -> str:
    """
    Analyze text and provide statistics.
    
    Args:
        text: The text to analyze
    
    Returns:
        Text analysis results
    """
    if not text.strip():
        return "Error: No text provided for analysis."
    
    words = text.split()
    characters = len(text)
    characters_no_spaces = len(text.replace(" ", ""))
    sentences = text.count('.') + text.count('!') + text.count('?')
    
    return f"""Text Analysis Results:
- Characters (with spaces): {characters}
- Characters (without spaces): {characters_no_spaces}
- Words: {len(words)}
- Estimated sentences: {sentences}
- Average word length: {characters_no_spaces / len(words):.1f} characters"""

# Create list of all tools
tools = [get_current_time, calculate_math, get_random_fact, weather_simulator, text_analyzer]

# Create tool executor
tool_executor = ToolExecutor(tools)

print(f"✅ Created {len(tools)} custom tools:")
for tool in tools:
    print(f"   - {tool.name}: {tool.description}")

# ============================================================================
# PART 2: DEFINE AGENT STATE
# ============================================================================

print("\n📊 PART 2: Agent State Definition")
print("-" * 40)

class ToolAgentState(TypedDict):
    messages: Annotated[List, add_messages]
    user_name: str
    tools_used: List[str]
    total_tool_calls: int

print("✅ Defined ToolAgentState with:")
print("   - messages: Conversation history")
print("   - user_name: Current user's name")
print("   - tools_used: List of tools used in session")
print("   - total_tool_calls: Counter for tool usage")

# ============================================================================
# PART 3: DEFINE AGENT NODES
# ============================================================================

print("\n🤖 PART 3: Agent Node Functions")
print("-" * 40)

def initialize_state(state: ToolAgentState):
    """Initialize or check the agent state."""
    print("   🔄 Initializing agent state...")
    
    if not state.get("user_name"):
        state["user_name"] = "User"
    if not state.get("tools_used"):
        state["tools_used"] = []
    if state.get("total_tool_calls") is None:
        state["total_tool_calls"] = 0
    
    return state

def call_llm_with_tools(state: ToolAgentState):
    """Call the LLM with tool capabilities."""
    print("   🧠 Calling LLM with tool access...")
    
    messages = state["messages"]
    
    # Create system message with tool information
    system_msg = f"""You are a helpful AI assistant with access to various tools. 
Current user: {state.get('user_name', 'User')}
Tools used in this session: {', '.join(state.get('tools_used', [])) or 'None yet'}
Total tool calls made: {state.get('total_tool_calls', 0)}

Available tools:
{chr(10).join([f"- {tool.name}: {tool.description}" for tool in tools])}

When you need to use a tool, call it with the appropriate parameters. 
Be helpful and use tools when they would provide valuable information to answer the user's question."""

    # Prepare messages with system context
    full_messages = [SystemMessage(content=system_msg)] + messages
    
    # Create LLM with tools
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.3
    ).bind_tools(tools)
    
    # Get response from LLM
    response = llm.invoke(full_messages)
    print(f"   💭 LLM Response: {response.content}")
    
    if response.tool_calls:
        print(f"   🔧 LLM requested {len(response.tool_calls)} tool call(s)")
        for tool_call in response.tool_calls:
            print(f"      - {tool_call['name']} with args: {tool_call['args']}")
    
    return {"messages": [response]}

def execute_tools(state: ToolAgentState):
    """Execute any tool calls requested by the LLM."""
    print("   ⚙️  Executing requested tools...")
    
    messages = state["messages"]
    last_message = messages[-1]
    
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        print("   ℹ️  No tools to execute")
        return state
    
    tool_messages = []
    tools_used = list(state.get("tools_used", []))
    total_calls = state.get("total_tool_calls", 0)
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        print(f"   🔧 Executing {tool_name} with args: {tool_args}")
        
        try:
            # Execute the tool
            tool_invocation = ToolInvocation(
                tool=tool_name,
                tool_input=tool_args
            )
            tool_result = tool_executor.invoke(tool_invocation)
            
            print(f"   ✅ Tool result: {tool_result}")
            
            # Create tool message
            tool_message = ToolMessage(
                content=tool_result,
                tool_call_id=tool_call["id"]
            )
            tool_messages.append(tool_message)
            
            # Track tool usage
            if tool_name not in tools_used:
                tools_used.append(tool_name)
            total_calls += 1
            
        except Exception as e:
            print(f"   ❌ Tool execution failed: {e}")
            error_message = ToolMessage(
                content=f"Error executing {tool_name}: {str(e)}",
                tool_call_id=tool_call["id"]
            )
            tool_messages.append(error_message)
    
    return {
        "messages": tool_messages,
        "tools_used": tools_used,
        "total_tool_calls": total_calls
    }

def generate_final_response(state: ToolAgentState):
    """Generate final response after tool execution."""
    print("   📝 Generating final response...")
    
    messages = state["messages"]
    
    # Create LLM without tools for final response
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.3
    )
    
    # Add system message for final response
    system_msg = f"""Based on the conversation and any tool results, provide a helpful and comprehensive response to the user.
User: {state.get('user_name', 'User')}
Session stats - Tools used: {len(state.get('tools_used', []))}, Total calls: {state.get('total_tool_calls', 0)}"""
    
    full_messages = [SystemMessage(content=system_msg)] + messages
    
    response = llm.invoke(full_messages)
    print(f"   💬 Final response: {response.content}")
    
    return {"messages": [response]}

# ============================================================================
# PART 4: ROUTING LOGIC
# ============================================================================

print("\n🔀 PART 4: Routing Logic")
print("-" * 40)

def should_continue(state: ToolAgentState):
    """Determine if we should continue to tool execution or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if the last message has tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print("   🔄 Routing to tool execution")
        return "execute_tools"
    else:
        print("   🏁 Routing to end")
        return "end"

def should_respond(state: ToolAgentState):
    """Determine if we should generate a final response."""
    print("   📤 Routing to final response")
    return "final_response"

print("✅ Routing functions defined:")
print("   - should_continue: Routes to tool execution or end")
print("   - should_respond: Routes to final response generation")

# ============================================================================
# PART 5: BUILD THE GRAPH
# ============================================================================

print("\n🏗️  PART 5: Building the LangGraph")
print("-" * 40)

# Create the state graph
workflow = StateGraph(ToolAgentState)

# Add nodes
workflow.add_node("initialize", initialize_state)
workflow.add_node("agent", call_llm_with_tools)
workflow.add_node("execute_tools", execute_tools)
workflow.add_node("final_response", generate_final_response)

# Define the flow
workflow.set_entry_point("initialize")
workflow.add_edge("initialize", "agent")

# Conditional routing from agent
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "execute_tools": "execute_tools",
        "end": END
    }
)

# After tool execution, generate final response
workflow.add_conditional_edges(
    "execute_tools",
    should_respond,
    {
        "final_response": "final_response"
    }
)

# End after final response
workflow.add_edge("final_response", END)

# Compile the graph
tool_agent = workflow.compile()

print("✅ Graph compiled successfully!")
print("   Flow: initialize → agent → [tools?] → [final_response] → END")

# ============================================================================
# PART 6: DEMONSTRATION FUNCTIONS
# ============================================================================

print("\n🎭 PART 6: Demonstration Functions")
print("-" * 40)

def demonstrate_single_tool():
    """Demonstrate a single tool execution."""
    print("\n🔍 Single Tool Demonstration")
    print("=" * 40)
    
    # Test each tool individually
    demo_inputs = [
        ("get_current_time", {}),
        ("calculate_math", {"expression": "25 * 4 + 10"}),
        ("get_random_fact", {}),
        ("weather_simulator", {"location": "Paris"}),
        ("text_analyzer", {"text": "LangGraph is amazing for building AI agents!"})
    ]
    
    for tool_name, args in demo_inputs:
        print(f"\n🔧 Testing {tool_name}:")
        try:
            tool_invocation = ToolInvocation(tool=tool_name, tool_input=args)
            result = tool_executor.invoke(tool_invocation)
            print(f"   ✅ Result: {result}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

def run_agent_example(user_input: str, user_name: str = "Demo User"):
    """Run a single agent interaction example."""
    print(f"\n🤖 Agent Example: '{user_input}'")
    print("=" * 60)
    
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "user_name": user_name,
        "tools_used": [],
        "total_tool_calls": 0
    }
    
    print(f"Input: {user_input}")
    print("\n--- Graph Execution ---")
    
    final_state = None
    for event in tool_agent.stream(initial_state, stream_mode="values"):
        final_state = event
    
    if final_state and "messages" in final_state:
        last_message = final_state["messages"][-1]
        if hasattr(last_message, 'content'):
            print(f"\n🎯 Final Response: {last_message.content}")
        
        print(f"\n📊 Session Stats:")
        print(f"   - Tools used: {final_state.get('tools_used', [])}")
        print(f"   - Total tool calls: {final_state.get('total_tool_calls', 0)}")

# ============================================================================
# PART 7: INTERACTIVE MAIN FUNCTION
# ============================================================================

def interactive_mode():
    """Run the agent in interactive mode."""
    print("\n🎮 Interactive Mode")
    print("=" * 40)
    
    user_name = input("What's your name? ").strip() or "User"
    print(f"\nHello {user_name}! I'm your tool-enabled AI assistant.")
    print("I have access to these tools:")
    for tool in tools:
        print(f"  • {tool.name}: {tool.description}")
    
    print("\nAsk me anything! Type 'exit' to quit, 'demo' for examples.")
    
    # Initialize persistent state
    current_state = {
        "messages": [],
        "user_name": user_name,
        "tools_used": [],
        "total_tool_calls": 0
    }
    
    while True:
        user_input = input(f"\n{user_name}: ").strip()
        
        if user_input.lower() == 'exit':
            print(f"\nGoodbye {user_name}! 👋")
            break
        elif user_input.lower() == 'demo':
            demonstrate_examples()
            continue
        elif user_input.lower() == 'stats':
            print(f"\n📊 Current session stats:")
            print(f"   - Tools used: {current_state.get('tools_used', [])}")
            print(f"   - Total tool calls: {current_state.get('total_tool_calls', 0)}")
            print(f"   - Messages in history: {len(current_state.get('messages', []))}")
            continue
        elif not user_input:
            print("Please enter a message or 'exit' to quit.")
            continue
        
        # Add user message to state
        current_state["messages"].append(HumanMessage(content=user_input))
        
        print("\n--- Processing your request ---")
        
        # Run the agent
        try:
            final_state = None
            for event in tool_agent.stream(current_state, stream_mode="values"):
                final_state = event
            
            if final_state and "messages" in final_state:
                # Update persistent state
                current_state.update(final_state)
                
                # Show AI response
                last_message = final_state["messages"][-1]
                if hasattr(last_message, 'content'):
                    print(f"\nAI: {last_message.content}")
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            # Reset messages on error to prevent issues
            current_state["messages"] = []

def demonstrate_examples():
    """Run pre-defined examples to show tool capabilities."""
    print("\n🎪 Running Example Demonstrations")
    print("=" * 50)
    
    examples = [
        "What time is it?",
        "Calculate 15 * 7 + 23",
        "Tell me a random fact",
        "What's the weather like in Tokyo?",
        "Analyze this text: 'LangGraph enables powerful AI agent workflows with tool integration.'"
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n--- Example {i}/{len(examples)} ---")
        run_agent_example(example)
        input("\nPress Enter to continue to next example...")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n🚀 LangGraph Tools Tutorial - Ready!")
    print("\nChoose an option:")
    print("1. Interactive mode (chat with the agent)")
    print("2. Run demonstrations")
    print("3. Test individual tools")
    print("4. Show all examples")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        interactive_mode()
    elif choice == "2":
        demonstrate_examples()
    elif choice == "3":
        demonstrate_single_tool()
    elif choice == "4":
        demonstrate_single_tool()
        demonstrate_examples()
        print("\n" + "="*60)
        print("All demonstrations complete!")
        print("Try interactive mode to experiment with the agent yourself.")
    else:
        print("Starting interactive mode by default...")
        interactive_mode()

    print("\n" + "="*60)
    print("Tutorial complete! Key takeaways:")
    print("• Tools extend agent capabilities")
    print("• LangGraph handles tool routing automatically")
    print("• State management tracks tool usage")
    print("• Error handling ensures robust execution")
    print("• Interactive loops enable dynamic conversations")
    print("="*60)
