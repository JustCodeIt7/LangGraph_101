# LangGraph Streaming Example 5: Asynchronous Streaming
# This example demonstrates how to use asynchronous streaming in LangGraph

#%%
import os
import asyncio
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
########## Step 3: Asynchronous Streaming #########################
####################################################################

console.print(Panel.fit(
    "Asynchronous Streaming with astream()\n"
    "This demonstrates how to use async/await with streaming in LangGraph.",
    title="Example 5", border_style="yellow"
))

async def demonstrate_async_agent_progress():
    """
    Demonstrate asynchronous agent progress streaming using astream().
    
    This uses the async/await pattern for streaming agent updates,
    which is useful in asynchronous applications like web servers.
    """
    user_message = "What's the weather in San Francisco and New York?"
    console.print(f"[bold]User:[/bold] {user_message}\n")
    console.print("[bold]Async agent progress (streaming):[/bold]")
    
    # Create input for the agent
    stream_input = {"messages": [{"role": "user", "content": user_message}]}
    
    # Stream agent progress updates asynchronously
    i = 0
    async for chunk in agent.astream(stream_input, stream_mode="updates"):
        i += 1
        console.print(f"[bold cyan]Async Update {i}:[/bold cyan]")
        
        # Pretty print the streaming updates
        for node_name, node_data in chunk.items():
            if "messages" in node_data:
                message = node_data["messages"][-1]
                if hasattr(message, "content") and message.content:
                    console.print(f"  [bold green]{node_name}:[/bold green] {message.content[:150]}...")
        
        console.print()  # Add a blank line between updates

async def demonstrate_async_llm_tokens():
    """
    Demonstrate asynchronous LLM token streaming using astream().
    
    This uses the async/await pattern for streaming LLM tokens,
    which is useful in asynchronous applications.
    """
    user_message = "What's the weather in San Francisco?"
    console.print(f"[bold]User:[/bold] {user_message}\n")
    console.print("[bold]Async LLM tokens (streaming):[/bold]")
    
    # Create input for the agent
    stream_input = {"messages": [{"role": "user", "content": user_message}]}
    
    # Stream LLM tokens asynchronously
    async for token, metadata in agent.astream(stream_input, stream_mode="messages"):
        if token:
            node_name = metadata.get("node_name", "unknown") if metadata else "unknown"
            console.print(f"[bold blue]Token from {node_name}:[/bold blue] [yellow]{token}[/yellow]")

async def demonstrate_async_custom_updates():
    """
    Demonstrate asynchronous custom updates streaming using astream().
    
    This uses the async/await pattern for streaming custom tool updates,
    which is useful in asynchronous applications.
    """
    user_message = "What's the weather in San Francisco?"
    console.print(f"[bold]User:[/bold] {user_message}\n")
    console.print("[bold]Async custom updates (streaming):[/bold]")
    
    # Create input for the agent
    stream_input = {"messages": [{"role": "user", "content": user_message}]}
    
    # Stream custom updates asynchronously
    i = 0
    async for chunk in agent.astream(stream_input, stream_mode="custom"):
        i += 1
        console.print(f"[bold magenta]Async Custom Update {i}:[/bold magenta] [cyan]{chunk}[/cyan]")

async def demonstrate_async_multiple_modes():
    """
    Demonstrate asynchronous streaming with multiple modes.
    
    This uses the async/await pattern for streaming multiple types of data,
    which is useful in asynchronous applications.
    """
    user_message = "What's the weather in San Francisco?"
    console.print(f"[bold]User:[/bold] {user_message}\n")
    console.print("[bold]Async multiple modes (streaming):[/bold]")

    # Create input for the agent
    stream_input = {'messages': [{'role': 'user', 'content': user_message}]}

    # Stream with multiple modes asynchronously
    async for stream_mode, chunk in agent.astream(
        stream_input, 
        stream_mode=["updates", "messages", "custom"]
    ):
        # Format based on the type of stream
        if stream_mode == 'custom':
            console.print(
                Panel(Text('Async Tool Update', style='bold white'), subtitle=str(chunk), border_style='magenta')
            )
        elif stream_mode == "messages":
            token, metadata = chunk
            if token:
                node_name = metadata.get("node_name", "unknown") if metadata else "unknown"
                console.print(Panel(
                    Text(f"Async LLM Token from {node_name}", style="bold white"),
                    subtitle=token,
                    border_style="blue"
                ))

        elif stream_mode == 'updates':
            console.print(
                Panel(
                    Text('Async Agent Update', style='bold white'),
                    subtitle=f'{str(chunk)[:100]}...',
                    border_style='green',
                )
            )

#%%
####################################################################
########## Step 4: Run the Example ################################
####################################################################

async def main():
    """Main async function to run all examples."""
    console.print(Panel.fit("LangGraph Streaming Tutorial - Example 5: Async Streaming", title="LangGraph Tutorial", border_style="red"))
    
    # Choose one of the async examples to run
    # Uncomment the one you want to run
    
    await demonstrate_async_agent_progress()
    # await demonstrate_async_llm_tokens()
    # await demonstrate_async_custom_updates()
    # await demonstrate_async_multiple_modes()

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())