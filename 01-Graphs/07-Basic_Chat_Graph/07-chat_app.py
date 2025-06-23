#%%
# ===========================================
# IMPORTS AND DEPENDENCIES
# ==========================================
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_litellm import ChatLiteLLM
from rich.console import Console
from rich.prompt import Prompt
import os
import subprocess
import tempfile
from pathlib import Path
from IPython.display import Image, display
#%%
# =============================================
# STATE DEFINITIONS AND DATA STRUCTURES
# =============================================

# Define the state structure
class ChatState(TypedDict):
    messages: List[Dict[str, str]]
    current_response: str
    exit_requested: bool
    verbose_mode: bool
    command_processed: bool  # New field to track command processing


# Initialize the console for rich output
console = Console()

#%%
# ===============================================
# LLM INITIALIZATION AND CONFIGURATION
# ==============================================
def create_llm(model='ollama/qwen3:0.6b') -> ChatLiteLLM:
    """Create a LiteLLM instance using Ollama with llama3.2 model."""
    console.print("🤖 Initializing LLM with Ollama (llama3.2)...", style="bold blue")

    llm = ChatLiteLLM(
        model= model,
        api_base="http://localhost:11434",  # Default Ollama local server
        temperature=0.7,
        max_tokens=1000,
    )

    console.print("✅ LLM initialized successfully!", style="bold green")
    return llm

#%%
# =============================================
# STATE MANAGEMENT FUNCTIONS
# ============================================
def initialize_state(state: ChatState) -> ChatState:
    """Initialize the chat state."""
    state["messages"] = []
    state["current_response"] = ""
    state["exit_requested"] = False
    state["verbose_mode"] = False
    state["command_processed"] = False  # Initialize new field
    return state

#%%
# ===========================================
# USER INPUT PROCESSING
# ==========================================

def process_user_input(state: ChatState) -> ChatState:
    """Process user input and add it to the messages."""
    # Get user input
    user_input = Prompt.ask('\n[bold cyan]You[/bold cyan]')

    # Check for exit command
    if user_input.lower() in ['/exit', '/quit', '/bye']:
        state['exit_requested'] = True
        return state

    # Check for verbose mode toggle
    if user_input.lower() == 'verbose':
        state['verbose_mode'] = not state['verbose_mode']
        console.print(f'[bold yellow]Verbose mode {"enabled" if state["verbose_mode"] else "disabled"}[/bold yellow]')
        return state

    # Add user message to the conversation history
    state['messages'].append({'role': 'user', 'content': user_input})

    return state


#%%
# =========================================
# AI RESPONSE GENERATION
# ========================================
def generate_ai_response(state: ChatState, llm: ChatLiteLLM) -> ChatState:
    """Generate AI response using the LLM."""
    # Skip AI response if exit requested, command was processed, or no messages
    if (state["exit_requested"] or
        state["command_processed"] or
        not state["messages"] or
        len(state["messages"]) == 0):
        return state

    # Also skip if the last message is not from user
    if state["messages"] and state["messages"][-1].get("role") != "user":
        return state


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
        if len(state['messages']) >= 2:
            console.print(f"Last user message: {state['messages'][-2]['content'][:50]}...")
        console.print(f"Current response length: {len(ai_message)} characters")

    return state


#%%
# =======================================
# CONVERSATION FLOW CONTROL
# ======================================

def should_continue(state: ChatState) -> str:
    """Conditional logic to determine whether to continue the conversation or end."""
    if not state['exit_requested']:
        # If a command was processed, go back to user input without AI response
        return "continue_input"
    console.print('[bold yellow]Exiting chat...[/bold yellow]')
    return 'end'

def decide_after_user_input(state: ChatState) -> str:
    """
    Decide what to do after processing user input.

    Returns:
        "end" - User wants to exit
        "continue_input" - User entered a command, stay in input loop
        "ai_response" - User entered a message, generate AI response
    """
    if state['exit_requested']:
        return 'end'
    elif state['command_processed']:
        return 'continue_input'
    else:
        return 'ai_response'



#%%
# =====================================================
# MAIN APPLICATION LOGIC AND GRAPH CONFIGURATION
# ====================================================

def main():
    """Main function to run the terminal chat application."""
    # ========================================
    # Application Initialization
    # ========================================
    console.print('[bold]Welcome to LangGraph Terminal Chat![/bold]', style='bold blue')
    console.print("Type 'exit', 'quit', or 'bye' to end the conversation.", style='dim')
    console.print("Type 'verbose' to toggle verbose mode.", style='dim')

    # Create the LLM
    llm = create_llm()

    # ========================================
    # Graph Construction
    # ========================================
    # Build the graph
    graph = StateGraph(ChatState)

    # Add nodes
    graph.add_node("initialize", initialize_state)
    graph.add_node("user_input", process_user_input)
    graph.add_node("ai_response", lambda state: generate_ai_response(state, llm))

    # Add edges
    graph.add_edge("initialize", "user_input")

    # Add conditional edge after user input
    graph.add_conditional_edges(
        "user_input",
        decide_after_user_input,
        {
            "ai_response": "ai_response",
            "continue_input": "user_input",
            "end": END,
        },
    )
    # Add conditional edge for looping or ending after AI response
    graph.add_conditional_edges(
        "ai_response",
        should_continue,
        {
            # "continue": "user_input",  # Loop back for more conversation
            "continue_input": "user_input",  # Direct back to input (shouldn't happen here)
            "end": END,  # End the conversation
        },
    )

    # Set the entry point
    graph.set_entry_point("initialize")
    app = graph.compile()
    # app.get_graph().draw_mermaid_png()
    with open('chat_graph.mmd', 'w') as f:
        f.write(app.get_graph().draw_mermaid())

    # save the graph image
    with open('chat_graph.png', 'wb') as f:
        f.write(app.get_graph().draw_mermaid_png())


    # ========================================
    # Application Execution
    # ========================================
    app.invoke({})



#%%
# ===================================================
# APPLICATION ENTRY POINT
# ==================================================
if __name__ == "__main__":
    main()
