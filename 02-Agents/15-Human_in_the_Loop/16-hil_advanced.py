import uuid
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from rich import print
################################ Configuration & Setup ################################

# Initialize the Ollama large language model
llm = ChatOllama(model='phi4-mini', temperature=0.7)


# Define the dictionary structure for the graph's state
class PromptState(TypedDict):
    """
    Represents the state of the prompt processing graph.
    """


################################ Core Logic & Helper Functions ################################


def call_llm(prompt: str, system_message: str = '') -> str:
    """
    Call Ollama LLM with a system message and a user prompt.
    """


################################ Graph Nodes ################################


def improve_prompt_node(state: PromptState) -> PromptState:
    """
    Use the LLM to refine and improve the user's original prompt.
    """


def human_review_node(state: PromptState) -> PromptState:
    """
    Pause the graph's execution to allow for human review and editing.
    """


def answer_prompt_node(state: PromptState) -> PromptState:
    """
    Use the LLM to generate a final answer based on the approved prompt.
    """


################################ Graph Definition & Compilation ################################


def create_app():
    """Create and compile the LangGraph application."""


################################ Application Execution ################################


def main():
    """Run an interactive console session for the Human-in-the-Loop app."""  # Define the main entry point for the script


if __name__ == '__main__':
    print('🚀 Advanced Human-in-the-Loop LangGraph App with Ollama')
    main()
