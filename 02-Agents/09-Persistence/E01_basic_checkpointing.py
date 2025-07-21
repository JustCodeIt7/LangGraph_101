# %%
import uuid
from typing import Annotated, TypedDict, List
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from rich import print

# from langchain.vectorstores.chroma import ChromaStore
# %%
# --- Example 1: Easiest - Basic Checkpointing with InMemorySaver ---
# This demonstrates creating a simple graph, invoking it with a thread,
# and retrieving checkpoints/state history.
print('\n=== Example 1: Basic Checkpointing ===')


# Define the state schema
class SimpleState(TypedDict):
    counter: Annotated[int, add]  # Reducer to accumulate values
    messages: Annotated[List[str], add]


# %%
# Define nodes
def increment_counter(state: SimpleState):
    return {'counter': 1, 'messages': ['Incremented counter']}


def double_counter(state: SimpleState):
    return {'counter': state['counter'] * 2, 'messages': ['Doubled counter']}


# %%
# Build the graph
workflow1 = StateGraph(SimpleState)
workflow1.add_node('increment', increment_counter)
workflow1.add_node('double', double_counter)
workflow1.add_edge(START, 'increment')
workflow1.add_edge('increment', 'double')
workflow1.add_edge('double', END)

# %%
# Compile with checkpointer
checkpointer1 = InMemorySaver()
graph1 = workflow1.compile(checkpointer=checkpointer1)
# %%
# Generate and save diagram for graph1
diagram1 = graph1.get_graph().draw_mermaid_png()
with open('g01_diagram.png', 'wb') as f:
    f.write(diagram1)
# %%
print('Saved g01 diagram to g01_diagram.png')
# Invoke with a thread_id
config1 = {'configurable': {'thread_id': 'example1'}}
input_state = {'counter': 0, 'messages': []}
result1 = graph1.invoke(input_state, config1)
print('Final State after Invoke:', result1)
# %%
# Get latest state snapshot
latest_state = graph1.get_state(config1)
print('Latest State Snapshot:', latest_state.values)
# %%
# Get full state history (list of snapshots, most recent first)
history = list(graph1.get_state_history(config1))
print('State History (most recent first):')
for snapshot in history:
    print(f'  - Values: {snapshot.values}, Next: {snapshot.next}')
