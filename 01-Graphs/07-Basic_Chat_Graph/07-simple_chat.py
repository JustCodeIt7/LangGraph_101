"""
LangGraph Terminal Chat Application

This script implements a terminal-based chat application using LangGraph with Ollama integration.
It uses the llama3.2 model and includes both looping logic for continuous conversation and
conditional logic based on a parameter.
"""

import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_litellm import ChatLiteLLM
from rich import print
from rich.console import Console
from rich.prompt import Prompt
import sys

# Define the state structure
class ChatState(TypedDict):
    messages: List[Dict[str, str]]
    current_response: str
    exit_requested: bool
    verbose_mode: bool

# Initialize the console for rich output
console = Console()

def create_llm():
    """Create a LiteLLM instance using Ollama with llama3.2 model."""
    console.print("🤖 Initializing LLM with Ollama (llama3.2)...", style="bold blue")
    
    llm = ChatLiteLLM(
        model="ollama/llama3.2",  # Using llama3.2 model as specified
        api_base="http://localhost:11434",  # Default Ollama local server
        temperature=0.7,
        max_tokens=1000,
    )
    
    console.print("✅ LLM initialized successfully!", style="bold green")
    return llm

def initialize_state(state: ChatState) -> ChatState:
    """Initialize the chat state."""
    state["messages"] = []
    state["current_response"] = ""
    state["exit_requested"] = False
    state["verbose_mode"] = False
    return state

def process_user_input(state: ChatState) -> ChatState:
    """Process user input and add it to the messages."""
    try:
        # Get user input
        user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit", "bye"]:
            state["exit_requested"] = True
            return state
            
        # Check for verbose mode toggle
        if user_input.lower() == "verbose":
            state["verbose_mode"] = not state["verbose_mode"]
            console.print(f"[bold yellow]Verbose mode {'enabled' if state['verbose_mode'] else 'disabled'}[/bold yellow]")
            return state
            
        # Add user message to the conversation history
        state["messages"].append({"role": "user", "content": user_input})
        
        return state
    except KeyboardInterrupt:
        state["exit_requested"] = True
        return state

def generate_ai_response(state: ChatState, llm: ChatLiteLLM) -> ChatState:
    """Generate AI response using the LLM."""
    if state["exit_requested"] or not state["messages"]:
        return state
        
    try:
        # Display thinking indicator
        console.print("[bold yellow]Thinking...[/bold yellow]")
        
        # Generate response from LLM
        response = llm.invoke(state["messages"])
        
        # Extract the response content
        ai_message = response.content
        
        # Add AI message to the conversation history
        state["messages"].append({"role": "assistant", "content": ai_message})
        state["current_response"] = ai_message
        
        # Display the response
        console.print(f"\n[bold green]Assistant[/bold green]: {ai_message}")
        
        # Display verbose information if enabled
        if state["verbose_mode"]:
            console.print("\n[bold magenta]Debug Info:[/bold magenta]")
            console.print(f"Message count: {len(state['messages'])}")
            console.print(f"Last user message: {state['messages'][-2]['content'][:50]}...")
        
        return state
    except Exception as e:
        console.print(f"[bold red]Error generating response: {str(e)}[/bold red]")
        return state

def should_continue(state: ChatState) -> str:
    """Conditional logic to determine whether to continue the conversation or end."""
    if state["exit_requested"]:
        console.print("[bold yellow]Exiting chat...[/bold yellow]")
        return "end"
    else:
        return "continue"

def main():
    """Main function to run the terminal chat application."""
    console.print("[bold]Welcome to LangGraph Terminal Chat![/bold]", style="bold blue")
    console.print("Type 'exit', 'quit', or 'bye' to end the conversation.", style="dim")
    console.print("Type 'verbose' to toggle verbose mode.", style="dim")
    
    # Create the LLM
    llm = create_llm()
    
    # Build the graph
    graph = StateGraph(ChatState)
    
    # Add nodes
    graph.add_node("initialize", initialize_state)
    graph.add_node("user_input", process_user_input)
    graph.add_node("ai_response", lambda state: generate_ai_response(state, llm))
    
    # Add edges
    graph.add_edge("initialize", "user_input")
    graph.add_edge("user_input", "ai_response")
    
    # Add conditional edge for looping or ending
    graph.add_conditional_edges(
        "ai_response",
        should_continue,
        {
            "continue": "user_input",  # Loop back for more conversation
            "end": END,  # End the conversation
        },
    )
    
    # Set the entry point
    graph.set_entry_point("initialize")
    
    # Compile the graph
    app = graph.compile()
    
    try:
        # Run the graph
        app.invoke({})
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Chat interrupted by user. Goodbye![/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]An error occurred: {str(e)}[/bold red]")
    
    console.print("\n[bold blue]Thank you for using LangGraph Terminal Chat![/bold blue]")

if __name__ == "__main__":
    main()
