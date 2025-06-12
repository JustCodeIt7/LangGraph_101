#%%
# ===========================================
# IMPORTS AND DEPENDENCIES
# ==========================================
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_litellm import ChatLiteLLM
from rich.console import Console
from rich.prompt import Prompt
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


# Initialize the console for rich output
console = Console()

#%%
# ===============================================
# LLM INITIALIZATION AND CONFIGURATION
# ==============================================

def create_llm(model='qwen3:0.6b') -> ChatLiteLLM:
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
create_llm()

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
    return state

#%%
# ===========================================
# USER INPUT PROCESSING
# ==========================================

def process_user_input(state: ChatState) -> ChatState:
    """Process user input and add it to the messages."""
    try:
        # Get user input
        user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")

        # Check for exit command
        if user_input.lower() in ["/exit", "/quit", "/bye"]:
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


#%%
# =========================================
# AI RESPONSE GENERATION
# ========================================
def generate_ai_response(state: ChatState, llm: ChatLiteLLM) -> ChatState:
    """Generate AI response using the LLM."""

    # Check if exit has been requested or if there are no messages
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
#%%
# =======================================
# CONVERSATION FLOW CONTROL
# ======================================

def should_continue(state: ChatState) -> str:
    """Conditional logic to determine whether to continue the conversation or end."""
    if state["exit_requested"]:
        console.print("[bold yellow]Exiting chat...[/bold yellow]")
        return "end"
    else:
        return "continue"

#%%
# =====================================================
# MAIN APPLICATION LOGIC AND GRAPH CONFIGURATION
# ====================================================

def main():
    """Main function to run the terminal chat application."""
    # ========================================
    # Application Initialization
    # ========================================
    console.print("[bold]Welcome to LangGraph Terminal Chat![/bold]", style="bold blue")
    console.print("Type 'exit', 'quit', or 'bye' to end the conversation.", style="dim")
    console.print("Type 'verbose' to toggle verbose mode.", style="dim")

    # ========================================
    # Graph Construction
    # ========================================

    # ========================================
    # Application Execution
    # ========================================

#%%
# ===================================================
# APPLICATION ENTRY POINT
# ==================================================
if __name__ == "__main__":
    main()
#%%