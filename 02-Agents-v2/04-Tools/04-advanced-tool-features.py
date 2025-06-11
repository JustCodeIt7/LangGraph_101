# LangGraph Tools Tutorial - Example 4: Advanced Tool Features
# This example demonstrates advanced tool features like return_direct, force tool use, and more

import os
from typing import List, Dict, Any, Annotated
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from rich import print
from rich.console import Console
from rich.panel import Panel

console = Console()

####################################################################
########## Step 1: Tools with Return Direct ########################
####################################################################

@tool(return_direct=True)
def calculate_square_root(number: float) -> str:
    """Calculate the square root of a number and return the result directly.
    
    When return_direct=True, the agent will return this result immediately
    without further processing.
    """
    print(f"[bold blue]Tool called:[/bold blue] Calculating square root of {number}")
    
    if number < 0:
        return f"Error: Cannot calculate square root of negative number {number}"
    
    result = number ** 0.5
    return f"The square root of {number} is {result:.4f}"

####################################################################
########## Step 2: Tools with Hidden Arguments #####################
####################################################################

def search_knowledge_base(
    query: str,
    # This will be populated by the agent state
    state: Annotated[AgentState, InjectedState],
    # This will be populated from the config
    config: RunnableConfig,
) -> str:
    """Search a knowledge base for information.
    
    This tool demonstrates how to access agent state and config
    that are not directly controlled by the LLM.
    """
    print(f"[bold blue]Tool called:[/bold blue] Searching knowledge base for: {query}")
    
    # Access user information from config (would be passed when invoking the agent)
    user_id = config.get("user_id", "unknown")
    
    # Access conversation history from state
    message_count = len(state["messages"]) if "messages" in state else 0
    
    # Simulate knowledge base search
    knowledge = {
        "python": "Python is a high-level, interpreted programming language known for its readability.",
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        "tools": "In LangGraph, tools are functions that can be called by language models to perform actions."
    }
    
    # Get result from knowledge base or default message
    result = knowledge.get(query.lower(), f"No information found for '{query}'")
    
    # Add metadata about the search
    metadata = f"\n\nSearch performed by user {user_id} during a conversation with {message_count} messages."
    
    return result + metadata

####################################################################
########## Step 3: Force Tool Use ##################################
####################################################################

def demonstrate_forced_tool_use():
    """Demonstrate how to force the agent to use a specific tool."""
    console.print(Panel.fit("Demonstrating forced tool use...", title="Forced Tool Use", border_style="green"))
    
    @tool
    def greet(user_name: str) -> str:
        """Greet the user by name."""
        print(f"[bold blue]Tool called:[/bold blue] Greeting {user_name}")
        return f"Hello {user_name}! It's nice to meet you."
    
    # Initialize the chat model
    model = init_chat_model("openai:gpt-4.1-nano", temperature=0)
    tools = [greet]
    
    # Create an agent that forces the use of the greet tool
    agent = create_react_agent(
        model=model.bind_tools(tools, tool_choice={"type": "tool", "name": "greet"}),
        tools=tools
    )
    
    # Run an example
    user_message = "Hi, my name is Alice"
    console.print(f"[bold]User:[/bold] {user_message}\n")
    console.print("[bold]Agent (forced to use greet tool):[/bold]")
    
    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]})
    
    if "messages" in result:
        final_message = result["messages"][-1]
        if hasattr(final_message, "content"):
            console.print(f"{final_message.content}")
        else:
            console.print(f"{final_message}")
    
    console.print("\n" + "-" * 50 + "\n")

####################################################################
########## Step 4: Disable Parallel Tool Calling ###################
####################################################################

def demonstrate_sequential_tool_calls():
    """Demonstrate how to disable parallel tool calling."""
    console.print(Panel.fit("Demonstrating sequential tool calls...", title="Sequential Tools", border_style="green"))
    
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        print(f"[bold blue]Tool called:[/bold blue] Adding {a} + {b}")
        return a + b

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        print(f"[bold blue]Tool called:[/bold blue] Multiplying {a} × {b}")
        return a * b
    
    # Initialize the chat model
    model = init_chat_model("openai:gpt-4.1-nano", temperature=0)
    tools = [add, multiply]
    
    # Create an agent with sequential tool calls
    agent = create_react_agent(
        model=model.bind_tools(tools, parallel_tool_calls=False),
        tools=tools
    )
    
    # Run an example
    user_message = "What's 3 + 5 and 4 * 7?"
    console.print(f"[bold]User:[/bold] {user_message}\n")
    console.print("[bold]Agent (sequential tool calls):[/bold]")
    
    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]})
    
    if "messages" in result:
        final_message = result["messages"][-1]
        if hasattr(final_message, "content"):
            console.print(f"{final_message.content}")
        else:
            console.print(f"{final_message}")
    
    console.print("\n" + "-" * 50 + "\n")

####################################################################
########## Step 5: Create Standard Agent with Advanced Tools #######
####################################################################

def create_tool_agent():
    """Create a LangGraph agent with advanced tools."""
    console.print(Panel.fit("Creating a LangGraph agent with advanced tools...", title="Setup", border_style="green"))
    
    # Create the agent with our advanced tools
    agent = create_react_agent(
        model="ollama:llama3.2",
        tools=[calculate_square_root, search_knowledge_base]
    )
    
    return agent

####################################################################
########## Step 6: Run Examples with the Agent #####################
####################################################################

def run_example(agent, user_message: str, config: Dict[str, Any] = None):
    """Run an example with the agent and display the results."""
    console.print(f"[bold]User:[/bold] {user_message}\n")
    console.print("[bold]Agent:[/bold]")
    
    # Create input for the agent
    agent_input = {"messages": [{"role": "user", "content": user_message}]}
    
    # Default config if none provided
    if config is None:
        config = {"user_id": "demo_user"}
    
    # Invoke the agent with config
    result = agent.invoke(agent_input, config=config)
    
    # Display the final response
    if "messages" in result:
        final_message = result["messages"][-1]
        if hasattr(final_message, "content"):
            console.print(f"{final_message.content}")
        else:
            console.print(f"{final_message}")
    
    console.print("\n" + "-" * 50 + "\n")

####################################################################
########## Step 7: Main Demonstration ##############################
####################################################################

def main():
    """Main function to demonstrate advanced tool features."""
    console.print(Panel.fit(
        "LangGraph Tools Tutorial - Example 4: Advanced Tool Features\n"
        "This example demonstrates advanced tool features like return_direct, force tool use, and more.",
        title="LangGraph Tutorial", 
        border_style="red"
    ))
    
    # Demonstrate return_direct tool
    agent = create_tool_agent()
    
    # Run examples with the standard agent
    console.print(Panel.fit("Return Direct and Hidden Arguments", border_style="yellow"))
    examples = [
        "Calculate the square root of 16",
        "What can you tell me about Python?",
        "I want to learn about LangGraph tools"
    ]
    
    for example in examples:
        # Pass custom config with user information
        config = {"user_id": f"user_{hash(example) % 1000}"}
        run_example(agent, example, config)
    
    # Demonstrate forced tool use
    demonstrate_forced_tool_use()
    
    # Demonstrate sequential tool calls
    demonstrate_sequential_tool_calls()

if __name__ == "__main__":
    main()