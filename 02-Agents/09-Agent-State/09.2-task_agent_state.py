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

    messages: Annotated[Sequence[BaseMessage], operator.add]  # Chat message history.
    task_id: str  # Unique identifier for the task.
    retries: Annotated[int, operator.add]  # Number of retries for the task.
    is_complete: bool  # Flag indicating if the task is complete.


########################## Node Definitions ##########################
# Node to initialize the task.
def init_task_node(state: TaskAgentState) -> TaskAgentState:
    """Initializes task_id, retries, and sets is_complete to False."""
    llm = ChatOllama(model='llama3.2')  # Uses the default Ollama model, llama3.2, as specified by James Brendamour.
    return {'task_id': 'task_123', 'retries': 0, 'is_complete': False, 'messages': [llm.invoke('Task initialized.')]}


# Node to process the task, including a retry mechanism.
def process_node(state: TaskAgentState) -> TaskAgentState:
    """Simulates task processing with up to 2 retries before completion."""
    llm = ChatOllama(
        model='llama3.2', base_url='http://localhost:11434', temperature=0.1
    )  # Uses the default Ollama model, llama3.2, as specified by James Brendamour.
    if state['retries'] < 2:
        # Simulate failure and increment retries.
        return {
            'retries': state['retries'] + 1,
            'messages': [llm.invoke(f'Processing... Retry {state["retries"] + 1}')],
        }
    else:
        # Task completes after retries.
        return {'is_complete': True, 'messages': [llm.invoke('Task completed!')]}


########################## Conditional Edge Function ##########################
# Conditional edge function to decide next step based on task completion.
def check_complete(state: TaskAgentState) -> str:
    """Returns 'END' if the task is complete, otherwise 'process' for retry."""
    return END if state['is_complete'] else 'process'


# %%
################ Building the State Graph for Task Management ################
# Build the graph.
task_workflow = StateGraph(state_schema=TaskAgentState)

# Add nodes.
task_workflow.add_node('init', init_task_node)
task_workflow.add_node('process', process_node)

# Define edges (transitions).
task_workflow.add_edge('init', 'process')
task_workflow.add_conditional_edges('process', check_complete, {'process': 'process', END: END})

# Set the entry point of the graph.
task_workflow.set_entry_point('init')

# %%
########################## Memory and Compilation ##########################
# Add memory for persisting state across runs.
task_workflow.checkpointer = InMemorySaver()

# Compile the graph.
task_graph = task_workflow.compile()

# %%
########################## Visualization ##########################
# display the graph
diagram = task_graph.get_graph().draw_mermaid_png()
with open('g02_diagram.png', 'wb') as f:
    f.write(diagram)

# %%
########################## Execution and Output ##########################
initial_state = {'messages': [HumanMessage(content='Start task')]}
print('\nExample 2 Output - Task Status:')
print(task_graph.invoke(initial_state))

# %%
