# %%
import os
from typing import Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver
from IPython.display import Image, display
from rich import print


# %%
################################ 1. State Definition ################################
# This is the shared state that all nodes in our graph will have access to and can modify.
class WorkflowState(TypedDict):
    task: str
    # The user's decision ('approve' or 'reject') will be stored here after the interrupt.
    user_decision: str
    # The final status of our workflow.
    status: str


# %%
################################ 2. Node Definitions ################################
# Each function below represents a "node" or a step in our graph.


def get_approval(state: WorkflowState):
    """
    This node uses 'interrupt' to pause the graph.
    It waits for a human to provide a decision before the graph can proceed.
    """
    print('--- ⏸️ PAUSING FOR APPROVAL ---')
    print(f"Task: '{state['task']}'")
    # The 'interrupt' function stops execution here. The string passed to it
    # is a message to the user about what input is expected.
    # When the graph is resumed, the value provided will be the return value of this function.
    decision = interrupt("Please enter 'approve' or 'reject' to continue.")
    print(f"--- ▶️ RESUMING WITH DECISION: '{decision}' ---")
    return {'user_decision': decision}


def router(state: WorkflowState) -> Command:
    """
    This node uses 'Command' to dynamically route the graph's execution.
    Based on the user's decision, it decides which node to run next.
    """
    print('--- 🔀 ROUTING ---')
    decision = state.get('user_decision', '').strip().lower()

    if decision == 'approve':
        print("Decision: ✅ Approved -> Routing to 'complete_task'")
        # Command(goto=...) tells LangGraph which node to execute next.
        return Command(goto='complete_task')
    else:
        print("Decision: ❌ Rejected -> Routing to 'cancel_task'")
        return Command(goto='cancel_task')


def complete_task(state: WorkflowState):
    """A final node for when the task is approved."""
    print('--- 🎉 TASK COMPLETED ---')
    return {'status': 'done'}


def cancel_task(state: WorkflowState):
    """A final node for when the task is rejected."""
    print('--- 🗑️ TASK CANCELED ---')
    return {'status': 'canceled'}


# %%
################################ 3. Graph Construction ################################

# Initialize an in-memory checkpointer. This is required for 'interrupt' to work,
# as it needs to save the graph's state when it pauses.
memory = InMemorySaver()

# Create the graph builder.
builder = StateGraph(WorkflowState)

# Add the functions as nodes to the graph.
builder.add_node('get_approval', get_approval)
builder.add_node('router', router)
builder.add_node('complete_task', complete_task)
builder.add_node('cancel_task', cancel_task)

# Define the graph's structure (its edges).
builder.add_edge(START, 'get_approval')
builder.add_edge('get_approval', 'router')

# NOTE: We do NOT need to add conditional edges from the 'router' node.
# The 'Command' object returned by the router handles the routing dynamically.

builder.add_edge('complete_task', END)
builder.add_edge('cancel_task', END)

# Compile the graph, enabling the checkpointer.
graph = builder.compile(checkpointer=memory)

# The complete_task and cancel_task nodes appear disconnected in the visualization because their connection to the graph is dynamic, not static. The routing decision is made at runtime by the router node using the Command object.

# You can visualize the graph structure.
# Notice the router doesn't have explicit paths leading out of it.
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
    # print(graph.get_graph().draw_mermaid())
except Exception as e:
    print(f'Could not display graph: {e}')

# %%
########################## 4. Graph Execution & Interaction ##########################

# --- Run 1: Approve the task ---
print('\n' + '=' * 50 + '\n🚀 STARTING RUN 1: APPROVAL\n' + '=' * 50)

# A 'thread_id' is needed to track the state of a single run.
thread = {'configurable': {'thread_id': 'run-1'}}
initial_task = {'task': 'Deploy new feature to production'}

# Start the graph. It will run until it hits the 'interrupt' in the 'get_approval' node.
# We use 'stream' to see the events as they happen.
for event in graph.stream(initial_task, thread, stream_mode='values'):
    print(f'\n[STREAM EVENT]:\n{event}\n')

# At this point, the graph is paused. Let's resume it with the user's decision.
# We send a Command object with the 'resume' payload.
print("\n... Resuming Run 1 with 'approve' ...\n")
for event in graph.stream(Command(resume='approve'), thread, stream_mode='values'):
    print(f'\n[STREAM EVENT]:\n{event}\n')

# %%
# --- Run 2: Reject the task ---
print('\n' + '=' * 50 + '\n🚀 STARTING RUN 2: REJECTION\n' + '=' * 50)

# Use a new thread_id for the second, independent run.
thread2 = {'configurable': {'thread_id': 'run-2'}}

# Start the second run.
for event in graph.stream(initial_task, thread2, stream_mode='values'):
    print(f'\n[STREAM EVENT]:\n{event}\n')

# Resume the second run, but this time with a 'reject' decision.
print("\n... Resuming Run 2 with 'reject' ...\n")
for event in graph.stream(Command(resume='reject'), thread2, stream_mode='values'):
    print(f'\n[STREAM EVENT]:\n{event}\n')

# %%
