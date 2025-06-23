# LangGraph Streaming Example 3: Tool Updates Streaming
# This example demonstrates how to stream updates from tools as they are executed

#%%
import os
import time
from langgraph.prebuilt import create_react_agent
from langgraph.config import get_stream_writer
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

console = Console()

#%%
####################################################################
########## Step 1: Create a Weather Tool with Streaming ############
####################################################################

def get_weather(city: str) -> str:
    """
    Get weather for a given city with streaming updates.
    
    This tool demonstrates how to use get_stream_writer to emit
    custom streaming updates during tool execution.
    """
    # Get the stream writer to emit custom updates
    writer = get_stream_writer()
    
    console.print(f"[bold blue]Tool called:[/bold blue] Fetching weather for {city}")
    
    # Simulate a process with multiple steps
    writer(f"Initializing weather lookup for {city}...")
    time.sleep(0.5)
    
    writer(f"Connecting to weather service for {city}...")
    time.sleep(0.5)
    
    writer(f"Retrieving current conditions for {city}...")
    time.sleep(0.5)
    
    writer(f"Analyzing weather patterns for {city}...")
    time.sleep(0.5)
    
    writer(f"Preparing weather report for {city}...")
    time.sleep(0.5)
    
    # Return the final result
    return f"It's always sunny in {city} with a high of 72°F."

#%%
####################################################################
########## Step 2: Create a Basic Agent ###########################
####################################################################

# Create a basic agent with a model and tools
console.print(Panel.fit("Creating a LangGraph agent...", title="Setup", border_style="green"))

agent = create_react_agent(
    model="ollama:llama3.2",  # You can replace with your preferred model
    tools=[get_weather]
)

#%%
####################################################################
########## Step 3: Tool Updates Streaming #########################
####################################################################

console.print(Panel.fit(
    "Tool Updates Streaming (stream_mode='custom')\n"
    "This mode streams custom updates from tools during execution.",
    title="Example 3", border_style="yellow"
))

def demonstrate_tool_updates_streaming():
    """
    Demonstrate tool updates streaming using stream_mode='custom'.
    
    This streams custom updates from tools as they are executed,
    allowing for real-time feedback during long-running operations.
    """
    user_message = "What's the weather in San Francisco and New York?"
    console.print(f"[bold]User:[/bold] {user_message}\n")
    console.print("[bold]Tool updates (streaming):[/bold]")
    
    # Create input for the agent
    stream_input = {"messages": [{"role": "user", "content": user_message}]}
    
    # Stream custom updates from tools
    for i, chunk in enumerate(agent.stream(stream_input, stream_mode="custom")):
        console.print(f"[bold magenta]Custom Update {i+1}:[/bold magenta] [cyan]{chunk}[/cyan]")

#%%
####################################################################
########## Step 4: Run the Example ################################
####################################################################

if __name__ == "__main__":
    console.print(Panel.fit("LangGraph Streaming Tutorial - Example 3: Tool Updates", title="LangGraph Tutorial", border_style="red"))
    demonstrate_tool_updates_streaming()