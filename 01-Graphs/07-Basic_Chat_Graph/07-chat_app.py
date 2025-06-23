# %%
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

# %%
# =============================================
# STATE DEFINITIONS AND DATA STRUCTURES
# =============================================
# Initialize the console for rich output
console = Console()

# Define the state structure
class ChatState(TypedDict):
    """State structure for the chat application."""
    messages: List[Dict[str, str]]
    current_response: str
    exit_requested: bool
    verbose_mode: bool
    command_processed: bool  # New field to track command processing

# %%
# ===============================================
# LLM INITIALIZATION AND CONFIGURATION
# ==============================================
def create_llm(model='ollama/qwen3:0.6b') -> ChatLiteLLM:
    """Create a LiteLLM instance using Ollama """
    console.print('🤖 Initializing LLM with Ollama (llama3.2)...', style='bold blue')

    llm = ChatLiteLLM(
        model=model,
        api_base='http://localhost:11434',  # Default Ollama local server
        temperature=0.7,
        max_tokens=1000,
    )

    console.print('✅ LLM initialized successfully!', style='bold green')
    return llm

# %%
# =============================================
# STATE MANAGEMENT FUNCTIONS
# ============================================
def initialize_state(state: ChatState) -> ChatState:
    """Initialize the chat state."""

# %%
# ===========================================
# USER INPUT PROCESSING
# ==========================================
def process_user_input(state: ChatState) -> ChatState:
    """Process user input and add it to the messages."""


# %%
# =========================================
# AI RESPONSE GENERATION
# ========================================
def generate_ai_response(state: ChatState, llm: ChatLiteLLM) -> ChatState:
    """Generate AI response using the LLM."""


# %%
# =======================================
# CONVERSATION FLOW CONTROL
# ======================================
def should_continue(state: ChatState) -> str:
    """Conditional logic to determine whether to continue the conversation or end."""


def decide_after_user_input(state: ChatState) -> str:
    """Decide what to do after processing user input"""


# %%
# =====================================================
# MAIN APPLICATION LOGIC AND GRAPH CONFIGURATION
# ====================================================
def main():
    """Main function to run the terminal chat application."""

    # ========================================
    # Application Initialization
    # ========================================
    console.print('[bold]Welcome to LangGraph cli Chat![/bold]', style='bold blue')
    console.print("Type '/exit', '/quit', or '/bye' to end chat.", style='dim')
    console.print("Type 'verbose' to toggle verbose mode.", style='dim')

    # Create the LLM
    llm = create_llm(model='ollama/qwen3:0.6b')

# %%
# ===================================================
# APPLICATION ENTRY POINT
# ==================================================
if __name__ == '__main__':
    main()
