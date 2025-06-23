# LangGraph Tools Tutorial - Example 3: Tool Error Handling
# This example demonstrates how to handle errors in tools using different strategies

import os
from typing import List, Dict, Any
from langgraph.prebuilt import create_react_agent, ToolNode
from rich import print
from rich.console import Console
from rich.panel import Panel

console = Console()

####################################################################
########## Step 1: Define Tools with Potential Errors ##############
####################################################################

def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    print(f"[bold blue]Tool called:[/bold blue] Dividing {a} ÷ {b}")
    
    # This will raise a ZeroDivisionError if b is 0
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    
    return a / b

def fetch_user_data(user_id: int) -> Dict[str, Any]:
    """Fetch user data from a database."""
    print(f"[bold blue]Tool called:[/bold blue] Fetching data for user {user_id}")
    
    # Simulate a database of users
    users = {
        1: {"name": "Alice", "email": "alice@example.com"},
        2: {"name": "Bob", "email": "bob@example.com"},
        3: {"name": "Charlie", "email": "charlie@example.com"}
    }
    
    # This will raise a KeyError if the user_id is not found
    if user_id not in users:
        raise KeyError(f"User with ID {user_id} not found")
    
    return users[user_id]

def parse_date(date_string: str) -> str:
    """Parse a date string into a standardized format."""
    print(f"[bold blue]Tool called:[/bold blue] Parsing date: {date_string}")

    import datetime

    # This will raise a ValueError if the date format is invalid
    try:
        # Try different date formats
        for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%B %d, %Y"]:
            try:
                date_obj = datetime.datetime.strptime(date_string, fmt)
                return f"Parsed date: {date_obj.strftime('%Y-%m-%d')}"
            except ValueError:
                continue

        # If we get here, none of the formats worked
        raise ValueError(f"Could not parse date: {date_string}")
    except Exception as e:
        raise ValueError(f'Invalid date format: {date_string}. Please use YYYY-MM-DD format.') from e

####################################################################
########## Step 2: Create Agents with Different Error Handling #####
####################################################################

def create_default_error_handling_agent():
    """Create an agent with default error handling (catch and report errors)."""
    console.print(Panel.fit("Creating an agent with default error handling...", title="Setup", border_style="green"))
    
    # Default behavior: catch errors and report them to the LLM
    agent = create_react_agent(
        model="ollama:llama3.2",
        tools=[divide, fetch_user_data, parse_date]
    )
    
    return agent

def create_disabled_error_handling_agent():
    """Create an agent with disabled error handling (let errors propagate)."""
    console.print(Panel.fit("Creating an agent with disabled error handling...", title="Setup", border_style="green"))
    
    # Create a ToolNode with error handling disabled
    tool_node = ToolNode(
        [divide, fetch_user_data, parse_date],
        handle_tool_errors=False  # Disable error handling
    )
    
    # Create the agent with the custom tool node
    agent = create_react_agent(
        model="ollama:llama3.2",
        tools=tool_node
    )
    
    return agent

def create_custom_error_handling_agent():
    """Create an agent with custom error messages for specific errors."""
    console.print(Panel.fit("Creating an agent with custom error handling...", title="Setup", border_style="green"))
    
    # Create a ToolNode with custom error messages
    tool_node = ToolNode(
        [divide, fetch_user_data, parse_date],
        handle_tool_errors={
            ZeroDivisionError: "You cannot divide by zero. Please try with a non-zero divisor.",
            KeyError: "The requested user ID doesn't exist. Please try with a valid user ID (1, 2, or 3).",
            ValueError: "The date format is invalid. Please use YYYY-MM-DD format."
        }
    )
    
    # Create the agent with the custom tool node
    agent = create_react_agent(
        model="ollama:llama3.2",
        tools=tool_node
    )
    
    return agent

####################################################################
########## Step 3: Run Examples with the Agents ####################
####################################################################

def run_example(agent, user_message: str, agent_type: str):
    """Run an example with the agent and display the results."""
    console.print(f"[bold]Agent Type:[/bold] {agent_type}")
    console.print(f"[bold]User:[/bold] {user_message}\n")
    console.print("[bold]Agent:[/bold]")
    
    # Create input for the agent
    agent_input = {"messages": [{"role": "user", "content": user_message}]}
    
    try:
        # Invoke the agent
        result = agent.invoke(agent_input)
        
        # Display the final response
        if "messages" in result:
            final_message = result["messages"][-1]
            if hasattr(final_message, "content"):
                console.print(f"{final_message.content}")
            else:
                console.print(f"{final_message}")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
    
    console.print("\n" + "-" * 50 + "\n")

####################################################################
########## Step 4: Main Demonstration ##############################
####################################################################

def main():
    """Main function to demonstrate tool error handling."""
    console.print(Panel.fit(
        "LangGraph Tools Tutorial - Example 3: Tool Error Handling\n"
        "This example demonstrates how to handle errors in tools using different strategies.",
        title="LangGraph Tutorial", 
        border_style="red"
    ))
    
    # Create agents with different error handling strategies
    default_agent = create_default_error_handling_agent()
    custom_agent = create_custom_error_handling_agent()
    
    # Examples that will trigger errors
    error_examples = [
        "What is 10 divided by 0?",
        "Can you fetch the data for user with ID 5?",
        "Parse the date: 2023/13/45"
    ]
    
    # Run examples with default error handling
    console.print(Panel.fit("Default Error Handling", border_style="yellow"))
    for example in error_examples:
        run_example(default_agent, example, "Default Error Handling")
    
    # Run examples with custom error handling
    console.print(Panel.fit("Custom Error Handling", border_style="yellow"))
    for example in error_examples:
        run_example(custom_agent, example, "Custom Error Handling")
    
    # Note: We're not demonstrating the disabled error handling agent
    # because it would crash the program when errors occur

if __name__ == "__main__":
    main()