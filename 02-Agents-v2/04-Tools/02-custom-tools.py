# LangGraph Tools Tutorial - Example 2: Customizing Tools
# This example demonstrates how to customize tools using decorators and Pydantic schemas

import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from rich import print
from rich.console import Console
from rich.panel import Panel

console = Console()

####################################################################
########## Step 1: Define Tools Using Decorators ###################
####################################################################

# Method 1: Using the @tool decorator with parse_docstring=True
@tool("multiply_tool", parse_docstring=True)
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.
    
    Args:
        a: First operand
        b: Second operand
    """
    print(f"[bold blue]Tool called:[/bold blue] Multiplying {a} × {b}")
    return a * b

####################################################################
########## Step 2: Define Tools with Custom Schemas ################
####################################################################

# Method 2: Using Pydantic schemas for more control over input validation
class WeatherInputSchema(BaseModel):
    """Schema for weather tool inputs"""
    location: str = Field(description="The city or location to get weather for")
    units: str = Field(description="Temperature units (celsius/fahrenheit)", default="fahrenheit")

@tool("weather_tool", args_schema=WeatherInputSchema)
def get_weather(location: str, units: str = "fahrenheit") -> str:
    """Get the current weather for a location."""
    print(f"[bold blue]Tool called:[/bold blue] Getting weather for {location} in {units}")
    
    # In a real application, this would call a weather API
    temp = 72 if units.lower() == "fahrenheit" else 22
    unit_symbol = "°F" if units.lower() == "fahrenheit" else "°C"
    
    return f"It's currently {temp}{unit_symbol} and sunny in {location}."

####################################################################
########## Step 3: Define Tools with Complex Validation ############
####################################################################

class TipCalculatorSchema(BaseModel):
    """Schema for tip calculator inputs with validation"""
    bill_amount: float = Field(
        description="The total bill amount in dollars", 
        gt=0  # Ensure bill amount is greater than 0
    )
    tip_percentage: float = Field(
        description="The tip percentage to apply", 
        default=15.0,
        ge=0, le=100  # Ensure tip percentage is between 0 and 100
    )

@tool("tip_calculator", args_schema=TipCalculatorSchema)
def calculate_tip(bill_amount: float, tip_percentage: float = 15.0) -> str:
    """Calculate the tip amount for a bill."""
    print(f"[bold blue]Tool called:[/bold blue] Calculating {tip_percentage}% tip on ${bill_amount}")
    
    tip_amount = bill_amount * (tip_percentage / 100)
    total_amount = bill_amount + tip_amount
    
    return f"For a ${bill_amount:.2f} bill with {tip_percentage}% tip:\nTip amount: ${tip_amount:.2f}\nTotal amount: ${total_amount:.2f}"

####################################################################
########## Step 4: Create an Agent with Custom Tools ###############
####################################################################

def create_tool_agent():
    """Create a LangGraph agent with the custom tools."""
    console.print(Panel.fit("Creating a LangGraph agent with custom tools...", title="Setup", border_style="green"))
    
    # Create the agent with our custom tools
    agent = create_react_agent(
        model="openai:gpt-4.1-nano",  # You can also use "ollama:llama3.2"
        tools=[multiply, get_weather, calculate_tip]
    )
    
    return agent

####################################################################
########## Step 5: Run Examples with the Agent #####################
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
########## Step 6: Main Demonstration ##############################
####################################################################

def main():
    """Main function to demonstrate the agent with custom tools."""
    console.print(Panel.fit(
        "LangGraph Tools Tutorial - Example 2: Customizing Tools\n"
        "This example demonstrates how to customize tools using decorators and Pydantic schemas.",
        title="LangGraph Tutorial", 
        border_style="red"
    ))
    
    # Create the agent
    agent = create_tool_agent()
    
    # Run examples
    examples = [
        "What's 42 multiplied by 56?",
        "What's the weather like in Tokyo in celsius?",
        "I had dinner that cost $120. Can you calculate a 18% tip?",
        "I need to calculate a 25% tip on a $85 bill."
    ]
    
    for example in examples:
        run_example(agent, example)

if __name__ == "__main__":
    main()