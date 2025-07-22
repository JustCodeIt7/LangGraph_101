# %%
from typing import TypedDict, Annotated, Sequence, Dict, Any
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from rich import print
from langchain_ollama import ChatOllama, OllamaEmbeddings

# %%
################ Initialize LLM ################
llm = ChatOllama(model='llama3.2', base_url='http://localhost:11434', temperature=0.1)

################ Complex State with Nested Data ################
# Handles nested subtasks and summarizes them.


# %%
################ Define State Schemas ################
# Defines the state for a single subtask.
class SubTaskState(TypedDict):
    """State for managing a subtask."""


# Defines the overall state for the agent.
class ComplexAgentState(TypedDict):
    """State for managing complex tasks with subtasks."""


# %%
################ Define Node Functions ################
# Node to simulate adding and processing a subtask.
def subtask_node(state: ComplexAgentState) -> ComplexAgentState:
    """Adds a new subtask and a message indicating the action."""


# Node to summarize completed subtasks.
def summarize_node(state: ComplexAgentState) -> ComplexAgentState:
    """Generates an overall summary from the completed subtasks."""


# %%
################ Building the State Graph for Task Management ################
# Build the graph.

# %%
# Add nodes to the graph.
################ Displaying and Executing the Complex State Graph ################
# Display the graph.


# %%

# %%

