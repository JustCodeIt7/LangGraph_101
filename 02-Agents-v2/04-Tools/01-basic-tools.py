# LangGraph Tools Tutorial - Example 1: Basic Tools
# This example demonstrates how to define and use simple tools in LangGraph

import os
from typing import List, Dict, Any
from langgraph.prebuilt import create_react_agent
from rich import print
from rich.console import Console
from rich.panel import Panel

console = Console()

####################################################################
########## Step 1: Define Simple Tools #############################
####################################################################

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    print(f"[bold blue]Tool called:[/bold blue] Multiplying {a} × {b}")
    return a * b

def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    print(f"[bold blue]Tool called:[/bold blue] Getting weather for {location}")
    # In a real application, this would call a weather API
    return f"It's currently 72°F and sunny in {location}."

def calculate_tip(bill_amount: float, tip_percentage: float = 15.0) -> str:
    """Calculate the tip amount for a bill."""
    print(f"[bold blue]Tool called:[/bold blue] Calculating {tip_percentage}% tip on ${bill_amount}")
    tip_amount = bill_amount * (tip_percentage / 100)
    total_amount = bill_amount + tip_amount
    return f"For a ${bill_amount:.2f} bill with {tip_percentage}% tip:\nTip amount: ${tip_amount:.2f}\nTotal amount: ${total_amount:.2f}"

####################################################################
########## Step 2: Create an Agent with Tools ######################
####################################################################

def create_tool_agent():
    """Create a LangGraph agent with the defined tools."""
    console.print(Panel.fit("Creating a LangGraph agent with tools...", title="Setup", border_style="green"))
    
    # Create the agent with our tools
    agent = create_react_agent(
        model="ollama:llama3.2",  # You can also use "openai:gpt-4.1-nano"
        tools=[multiply, get_weather, calculate_tip]
    )
    
    return agent

####################################################################
########## Step 3: Run Examples with the Agent #####################
####################################################################

def run_example(agent, user_message: str):
    """Run an example with the agent and display the results."""
    console.print(f"[bold]User:[/bold] {user_message}\n")
    console.print("[bold]Agent:[/bold]")
    
    # Create input for the agent
    agent_input = {"messages": [{"role": "user", "content": user_message}]}
    
    # Invoke the agent
    result = agent.invoke(agent_input)
    
    # Display the final response
    if "messages" in result:
        final_message = result["messages"][-1]
        if hasattr(final_message, "content"):
            console.print(f"{final_message.content}")
        else:
            console.print(f"{final_message}")
    
    console.print("\n" + "-" * 50 + "\n")

####################################################################
########## Step 4: Main Demonstration ##############################
####################################################################

def main():
    """Main function to demonstrate the agent with tools."""
    console.print(Panel.fit(
        "LangGraph Tools Tutorial - Example 1: Basic Tools\n"
        "This example demonstrates how to define and use simple tools in LangGraph.",
        title="LangGraph Tutorial", 
        border_style="red"
    ))
    
    # Create the agent
    agent = create_tool_agent()
    
    # Run examples
    examples = [
        "What's 25 multiplied by 16?",
        "What's the weather like in San Francisco?",
        "I had dinner that cost $85. Can you calculate a 20% tip?",
        "I need to multiply 13 by 7 and then calculate a 15% tip on the result."
    ]
    
    for example in examples:
        run_example(agent, example)

if __name__ == "__main__":
    main()