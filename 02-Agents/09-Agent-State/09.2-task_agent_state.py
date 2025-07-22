# %%
########################## Imports and Setup ##########################
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from rich import print
from typing import TypedDict, Annotated, Sequence  # noqa: F811
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama, OllamaEmbeddings


# %%
# --- Example 1: State with Task Tracking ---
# Implements task tracking, retries, and completion status.
########################## State Definition ##########################
# Defines the agent's state for task management.
class TaskAgentState(TypedDict):
    """State for managing a task with retries and completion status."""

    # ANNOTATED EXPLANATION:
    # Annotated[Type, metadata] allows you to add metadata to type hints
    # Here we're telling LangGraph HOW to handle state updates for this field
    messages: Annotated[Sequence[BaseMessage], operator.add]  # Chat message history.
    # - Type: Sequence[BaseMessage] (a sequence of chat messages)
    # - Metadata: operator.add (when updating this field, ADD/APPEND new messages to existing ones)
    # - Without Annotated: new messages would REPLACE old ones
    # - With operator.add: new messages get APPENDED to the existing sequence

    task_id: str  # Unique identifier for the task.
    retries: Annotated[int, operator.add]  # Number of retries for the task.
    is_complete: bool  # Flag indicating if the task is complete.


# %%
########################## Node Definitions ##########################
# Node to initialize the task.
def init_task_node(state: TaskAgentState) -> TaskAgentState:
    """Initializes task_id, retries, and sets is_complete to False."""
    llm = ChatOllama(model='llama3.2')
    return {'task_id': 'task_123', 'retries': 0, 'is_complete': False, 'messages': [llm.invoke('Task initialized.')]}


# Node to process the task, including a retry mechanism.
def process_node(state: TaskAgentState, num: int = 2) -> TaskAgentState:
    """Simulates task processing with up to 2 retries before completion."""
    llm = ChatOllama(model='llama3.2', base_url='http://localhost:11434', temperature=0.1)
    print(f'Retries: {state["retries"]}')
    if state['retries'] < num:
        # Simulate failure and increment retries.
        return {
            'retries': state['retries'] + 1,
            'messages': [llm.invoke(f'Processing... Retry {state["retries"] + 1}')],
        }
    else:
        # Task completes after retries.
        return {'is_complete': True, 'messages': [llm.invoke('Task completed!')]}


########################## Conditional Edge Function ######################
# Conditional edge function to decide next step based on task completion.
def check_complete(state: TaskAgentState) -> str:
    """Returns 'END' if the task is complete, otherwise 'process' for retry."""
    return END if state['is_complete'] else 'process'


# %%
################ Building the State Graph for Task Management ################
workflow = StateGraph(state_schema=TaskAgentState)

# Add nodes.
workflow.add_node('init', init_task_node)
workflow.add_node('process', process_node)

# Define edges (transitions).
workflow.add_edge('init', 'process')
edge_map = {'process': 'process', END: END}
workflow.add_conditional_edges('process', check_complete, edge_map)
# Set the entry point of the graph.
workflow.set_entry_point('init')

# %%
########################## Memory and Compilation ##########################
# Add memory for persisting state across runs.
workflow.checkpointer = InMemorySaver()

# Compile the graph.
task_graph = workflow.compile()
# %%

# %%
########################## Visualization ##########################
# display the graph
diagram = task_graph.get_graph().draw_mermaid_png()
with open('g02_diagram.png', 'wb') as f:
    f.write(diagram)

# %%
########################## Execution and Output ##########################
initial_state = {'messages': [HumanMessage(content='Start task')]}
print('\nExample 1 Output - Task Status:')
# print(task_graph.invoke(initial_state))
print(task_graph.invoke(initial_state, debug=True))
# %%
