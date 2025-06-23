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


# Define the state structure
class ChatState(TypedDict):
    messages: List[Dict[str, str]]
    current_response: str
    exit_requested: bool
    verbose_mode: bool
    command_processed: bool  # New field to track command processing


# Initialize the console for rich output
console = Console()


# %%
# ===============================================
# LLM INITIALIZATION AND CONFIGURATION
# ==============================================
def create_llm(model='ollama/qwen3:0.6b') -> ChatLiteLLM:
    """Create a LiteLLM instance using Ollama with llama3.2 model."""


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


# %%
# ===================================================
# APPLICATION ENTRY POINT
# ==================================================
if __name__ == '__main__':
    main()
