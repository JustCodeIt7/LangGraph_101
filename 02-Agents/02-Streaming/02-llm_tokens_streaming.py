# LangGraph Streaming Example 2: LLM Tokens Streaming
# This example demonstrates how to stream tokens as they are produced by the LLM

#%%
import os
from langgraph.prebuilt import create_react_agent
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.text import Text

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
########## Step 3: LLM Tokens Streaming ###########################
####################################################################

console.print(Panel.fit(
    "LLM Tokens Streaming (stream_mode='messages')\n"
    "This mode streams tokens as they are produced by the language model.",
    title="Example 2", border_style="yellow"
))

def demonstrate_llm_tokens_streaming():
    """
    Demonstrate LLM tokens streaming using stream_mode='messages'.
    
    This streams tokens as they are produced by the language model,
    allowing for a more interactive and responsive user experience.
    """
    user_message = "What's the weather in San Francisco and New York?"
    console.print(f"[bold]User:[/bold] {user_message}\n")
    console.print("[bold]LLM tokens (streaming):[/bold]")
    
    # Create input for the agent
    stream_input = {"messages": [{"role": "user", "content": user_message}]}
    
    # Variables to track the current node and accumulated text
    current_node = None
    accumulated_text = ""
    
    # Stream LLM tokens
    for token, metadata in agent.stream(stream_input, stream_mode="messages"):
        # Extract node information from metadata
        if metadata and "run_id" in metadata:
            node_name = metadata.get("node_name", "unknown")
            
            # If we've switched to a new node, print the accumulated text and reset
            if current_node and current_node != node_name:
                console.print(f"[bold green]{current_node}:[/bold green] {accumulated_text}")
                accumulated_text = ""
            
            current_node = node_name
        
        # Accumulate tokens
        if token:
            accumulated_text += token
            
            # Print the token with a special highlight
            console.print(f"[bold cyan]Token:[/bold cyan] [yellow]{token}[/yellow]")
    
    # Print any remaining accumulated text
    if accumulated_text:
        console.print(f"[bold green]{current_node}:[/bold green] {accumulated_text}")

#%%
####################################################################
########## Step 4: Run the Example ################################
####################################################################

if __name__ == "__main__":
    console.print(Panel.fit("LangGraph Streaming Tutorial - Example 2: LLM Tokens", title="LangGraph Tutorial", border_style="red"))
    demonstrate_llm_tokens_streaming()
# %%
