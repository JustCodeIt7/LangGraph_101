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

# --- Example 2: Medium - Replaying and Updating State ---
# This builds on Example 1. We replay from a checkpoint (time travel)
# and update state manually (forking and editing).
print('\n=== Example 2: Replaying and Updating State ===')
# Reuse the same graph and checkpointer from Example 1 (state persists in memory)
# Get a specific checkpoint_id from history (e.g., after increment but before double)
# From Example 1's history, let's assume we want to replay after 'increment'
# (In a real run, you'd inspect history to get the ID; here we simulate by invoking again if needed)
# For demo, re-invoke to ensure checkpoints
checkpointer1 = InMemorySaver()
workflow1 = StateGraph(TypedDict('SimpleState', {'counter': Annotated[int, add], 'messages': Annotated[List[str], add]}))
workflow1.add_node('increment', lambda state: {'counter': 1, 'messages': ['Incremented counter']})
workflow1.add_node('double', lambda state: {'counter': state['counter'] * 2, 'messages': ['Doubled counter']})
workflow1.add_edge(START, 'increment')
workflow1.add_edge('increment', 'double')
workflow1.add_edge('double', END)
graph1 = workflow1.compile(checkpointer=checkpointer1)
graph1.invoke({'counter': 0, 'messages': []}, {'configurable': {'thread_id': 'example1'}})  # Re-run to populate if needed
# Fetch history again and pick the checkpoint after 'increment' (second most recent)
history = list(graph1.get_state_history({'configurable': {'thread_id': 'example1'}}))
checkpoint_after_increment = history[1].config['configurable']['checkpoint_id']  # Second in list
print("Checkpoint ID after 'increment':", checkpoint_after_increment)
# Replay from that checkpoint (forks and executes only after the checkpoint)
replay_config = {'configurable': {'thread_id': 'example1', 'checkpoint_id': checkpoint_after_increment}}
replay_result = graph1.invoke(None, replay_config)  # None input since state is loaded from checkpoint
print("Replayed State (only 'double' executes):", replay_result)
# Now, update state manually (fork and edit as if from a node)
# Update 'counter' (overwrites since no reducer) and 'messages' (appends due to reducer)
update_values = {'counter': 100, 'messages': ['Manual update']}
updated_config = graph1.update_state({'configurable': {'thread_id': 'example1'}}, update_values, as_node='double')
print('Config after Update:', updated_config)
# Get the new state after update
updated_state = graph1.get_state(updated_config)
print('Updated State Snapshot:', updated_state.values)