# LangGraph Streaming Example 1: Agent Progress Streaming
# This example demonstrates how to stream agent progress updates

#%%
import os
from langgraph.prebuilt import create_react_agent
from rich import print
from rich.console import Console
from rich.panel import Panel

console = Console()

#%%
####################################################################
########## Step 1: Create a Weather Tool ###########################
####################################################################

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    print(f"[bold blue]Tool called:[/bold blue] Fetching weather for {city}")
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
########## Step 3: Agent Progress Streaming #######################
####################################################################

console.print(Panel.fit(
    "Agent Progress Streaming (stream_mode='updates')\n"
    "This mode emits an event after every agent step.",
    title="Example 1", border_style="yellow"
))

def demonstrate_agent_progress_streaming():
    """
    Demonstrate agent progress streaming using stream_mode='updates'.
    
    This emits an event after every agent step:
    - LLM node: AI message with tool call requests
    - Tool node: Tool message with execution result
    - LLM node: Final AI response
    """
    user_message = "What's the weather in San Francisco and New York?"
    console.print(f"[bold]User:[/bold] {user_message}\n")
    console.print("[bold]Agent progress (streaming):[/bold]")
    
    # Create input for the agent
    stream_input = {"messages": [{"role": "user", "content": user_message}]}
    
    # Stream agent progress updates
    for i, chunk in enumerate(agent.stream(stream_input, stream_mode="updates")):
        console.print(f"[bold cyan]Update {i+1}:[/bold cyan]")
        
        # Pretty print the streaming updates
        for node_name, node_data in chunk.items():
            if "messages" in node_data:
                message = node_data["messages"][-1]
                if hasattr(message, "content") and message.content:
                    console.print(f"  [bold green]{node_name}:[/bold green] {message.content[:150]}...")
        
        console.print()  # Add a blank line between updates

#%%
####################################################################
########## Step 4: Run the Example ################################
####################################################################

if __name__ == "__main__":
    console.print(Panel.fit("LangGraph Streaming Tutorial - Example 1: Agent Progress", title="LangGraph Tutorial", border_style="red"))
    demonstrate_agent_progress_streaming()