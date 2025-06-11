# LangGraph Streaming Example 4: Multiple Streaming Modes
# This example demonstrates how to stream multiple types of data at the same time

#%%
import os
import time
from langgraph.prebuilt import create_react_agent
from langgraph.config import get_stream_writer
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

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
    
    writer(f"Retrieving current conditions for {city}...")
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
########## Step 3: Multiple Streaming Modes #######################
####################################################################

console.print(Panel.fit(
    "Multiple Streaming Modes (stream_mode=['updates', 'messages', 'custom'])\n"
    "This mode combines all streaming types for comprehensive real-time updates.",
    title="Example 4", border_style="yellow"
))

def demonstrate_multiple_streaming_modes():
    """
    Demonstrate multiple streaming modes using stream_mode=['updates', 'messages', 'custom'].
    
    This combines all streaming types:
    - 'updates': Agent progress after each node
    - 'messages': LLM tokens as they're generated
    - 'custom': Custom updates from tools
    
    This provides comprehensive real-time feedback during agent execution.
    """
    user_message = "What's the weather in San Francisco and New York?"
    console.print(f"[bold]User:[/bold] {user_message}\n")
    console.print("[bold]Multiple streaming modes:[/bold]")
    
    # Create input for the agent
    stream_input = {"messages": [{"role": "user", "content": user_message}]}
    
    # Stream with multiple modes
    for stream_mode, chunk in agent.stream(
        stream_input, 
        stream_mode=["updates", "messages", "custom"]
    ):
        # Format based on the type of stream
        if stream_mode == "updates":
            console.print(Panel(
                Text(f"Agent Update", style="bold white"),
                subtitle=str(chunk)[:100] + "...",
                border_style="green"
            ))
        
        elif stream_mode == "messages":
            token, metadata = chunk
            if token:
                node_name = metadata.get("node_name", "unknown") if metadata else "unknown"
                console.print(Panel(
                    Text(f"LLM Token from {node_name}", style="bold white"),
                    subtitle=token,
                    border_style="blue"
                ))
        
        elif stream_mode == "custom":
            console.print(Panel(
                Text("Tool Update", style="bold white"),
                subtitle=str(chunk),
                border_style="magenta"
            ))

#%%
####################################################################
########## Step 4: Run the Example ################################
####################################################################

if __name__ == "__main__":
    console.print(Panel.fit("LangGraph Streaming Tutorial - Example 4: Multiple Streaming Modes", title="LangGraph Tutorial", border_style="red"))
    demonstrate_multiple_streaming_modes()