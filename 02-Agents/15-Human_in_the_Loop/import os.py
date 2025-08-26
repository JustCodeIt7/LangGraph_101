import os
from typing import Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver
from IPython.display import Image, display
from rich import print

# State Definition
class WorkflowState(TypedDict):
    task: str
    user_decision: str
    status: str

# Node Definitions
def get_approval(state: WorkflowState):
    """
    This node uses 'interrupt' to pause the graph.
    """
    print('--- ⏸️ PAUSING FOR APPROVAL ---')
    print(f"Task: '{state['task']}'")
    # The interrupt function pauses execution here
    decision = interrupt("Please enter 'approve' or 'reject' to continue.")
    print(f"--- ▶️ RESUMING WITH DECISION: '{decision}' ---")
    return {'user_decision': decision}

def router(state: WorkflowState) -> Command:
    """
    This node uses 'Command' to dynamically route the graph's execution.
    """
    print('--- 🔀 ROUTING ---')
    decision = state.get('user_decision', '').strip().lower()
    
    if decision == 'approve':
        print("Decision: ✅ Approved -> Routing to 'complete_task'")
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

# Graph Construction
memory = InMemorySaver()
builder = StateGraph(WorkflowState)

# Add nodes
builder.add_node('get_approval', get_approval)
builder.add_node('router', router)
builder.add_node('complete_task', complete_task)
builder.add_node('cancel_task', cancel_task)

# Add edges
builder.add_edge(START, 'get_approval')
builder.add_edge('get_approval', 'router')
# NOTE: No conditional edges needed when using Command(goto=...)
builder.add_edge('complete_task', END)
builder.add_edge('cancel_task', END)

# Compile the graph
graph = builder.compile(checkpointer=memory)

# Execution
print('\n' + '=' * 50 + '\n🚀 STARTING RUN 1: APPROVAL\n' + '=' * 50)

thread = {'configurable': {'thread_id': 'run-1'}}
initial_task = {'task': 'Deploy new feature to production'}

# Start the graph
for event in graph.stream(initial_task, thread, stream_mode='values'):
    print(f'\n[STREAM EVENT]:\n{event}\n')

# CORRECT: Resume with Command(resume=...)
print("\n... Resuming Run 1 with 'approve' ...\n")
for event in graph.stream(Command(resume='approve'), thread, stream_mode='values'):
    print(f'\n[STREAM EVENT]:\n{event}\n')
