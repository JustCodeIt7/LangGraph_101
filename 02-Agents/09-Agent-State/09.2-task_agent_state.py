# %%
########################## Imports and Setup ##########################
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from rich import print
from typing import TypedDict, Annotated, Sequence
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama, OllamaEmbeddings
# %%
# --- Example 2: State with Task Tracking ---
# Implements task tracking, retries, and completion status.


########################## State Definition ##########################
# Defines the agent's state for task management.
class TaskAgentState(TypedDict):
    """State for managing a task with retries and completion status."""


########################## Node Definitions ##########################
# Node to initialize the task.
def init_task_node(state: TaskAgentState) -> TaskAgentState:
    """Initializes task_id, retries, and sets is_complete to False."""


# Node to process the task, including a retry mechanism.
def process_node(state: TaskAgentState) -> TaskAgentState:
    """Simulates task processing with up to 2 retries before completion."""


########################## Conditional Edge Function ##########################
# Conditional edge function to decide next step based on task completion.
def check_complete(state: TaskAgentState) -> str:
    """Returns 'END' if the task is complete, otherwise 'process' for retry."""


# %%
################ Building the State Graph for Task Management ################
# Build the graph.


# %%
########################## Memory and Compilation ##########################
# Add memory for persisting state across runs.


# %%
########################## Visualization ##########################


# %%
########################## Execution and Output ##########################

# %%
