# This script teaches LangGraph persistence concepts through 3 examples.
# %%
import uuid
from typing import Annotated, TypedDict, List
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables import RunnableConfig  # Added import
from langgraph.store.base import BaseStore  # Added import
from rich import print
# %%
# --- Example 1: Easiest - Basic Checkpointing with InMemorySaver ---
# This demonstrates creating a simple graph, invoking it with a thread,
# and retrieving checkpoints/state history.

print('\n=== Example 1: Basic Checkpointing ===')


# Define the state schema
class SimpleState(TypedDict):
    counter: Annotated[int, add]  # Reducer to accumulate values
    messages: Annotated[List[str], add]


# Define nodes
def increment_counter(state: SimpleState):
    return {'counter': 1, 'messages': ['Incremented counter']}


def double_counter(state: SimpleState):
    return {'counter': state['counter'] * 2, 'messages': ['Doubled counter']}


# Build the graph
workflow1 = StateGraph(SimpleState)
workflow1.add_node('increment', increment_counter)
workflow1.add_node('double', double_counter)
workflow1.add_edge(START, 'increment')
workflow1.add_edge('increment', 'double')
workflow1.add_edge('double', END)

# Compile with checkpointer
checkpointer1 = InMemorySaver()
graph1 = workflow1.compile(checkpointer=checkpointer1)

# Generate and save diagram for graph1
diagram1 = graph1.get_graph().draw_mermaid_png()
with open('graph1_diagram.png', 'wb') as f:
    f.write(diagram1)
print('Saved graph1 diagram to graph1_diagram.png')

# Invoke with a thread_id
config1 = {'configurable': {'thread_id': 'example1'}}
input_state = {'counter': 0, 'messages': []}
result1 = graph1.invoke(input_state, config1)
print('Final State after Invoke:', result1)

# Get latest state snapshot
latest_state = graph1.get_state(config1)
print('Latest State Snapshot:', latest_state.values)

# Get full state history (list of snapshots, most recent first)
history = list(graph1.get_state_history(config1))
print('State History (most recent first):')
for snapshot in history:
    print(f'  - Values: {snapshot.values}, Next: {snapshot.next}')
# %%
# --- Example 2: Medium - Replaying and Updating State ---
# This builds on Example 1. We replay from a checkpoint (time travel)
# and update state manually (forking and editing).

print('\n=== Example 2: Replaying and Updating State ===')

# Reuse the same graph and checkpointer from Example 1 (state persists in memory)

# Get a specific checkpoint_id from history (e.g., after increment but before double)
# From Example 1's history, let's assume we want to replay after 'increment'
# (In a real run, you'd inspect history to get the ID; here we simulate by invoking again if needed)
# For demo, re-invoke to ensure checkpoints
graph1.invoke({'counter': 0, 'messages': []}, config1)  # Re-run to populate if needed

# Fetch history again and pick the checkpoint after 'increment' (second most recent)
history = list(graph1.get_state_history(config1))
checkpoint_after_increment = history[1].config['configurable']['checkpoint_id']  # Second in list
print("Checkpoint ID after 'increment':", checkpoint_after_increment)

# Replay from that checkpoint (forks and executes only after the checkpoint)
replay_config = {'configurable': {'thread_id': 'example1', 'checkpoint_id': checkpoint_after_increment}}
replay_result = graph1.invoke(None, replay_config)  # None input since state is loaded from checkpoint
print("Replayed State (only 'double' executes):", replay_result)

# Now, update state manually (fork and edit as if from a node)
# Update 'counter' (overwrites since no reducer) and 'messages' (appends due to reducer)
update_values = {'counter': 100, 'messages': ['Manual update']}
updated_config = graph1.update_state(config1, update_values, as_node='double')
print('Config after Update:', updated_config)

# Get the new state after update
updated_state = graph1.get_state(updated_config)
print('Updated State Snapshot:', updated_state.values)
# %%
# --- Example 3: Intermediate - Memory Store Across Threads ---
# This demonstrates using InMemoryStore for cross-thread persistence (e.g., user memories)
# integrated with checkpoints for thread-specific state.

print('\n=== Example 3: Memory Store Across Threads ===')


# Define state schema with messages
class MemoryState(TypedDict):
    messages: Annotated[List[str], add]


# Define a node that updates and uses memory store
# Updated signature: Added RunnableConfig and made store keyword-only with BaseStore
def chat_node(state: MemoryState, config: RunnableConfig, *, store: BaseStore):
    user_id = config['configurable'].get('user_id', 'default_user')
    namespace = (user_id, 'memories')

    # Search for relevant memories (simple exact match for demo; could use semantic search)
    query = state['messages'][-1] if state['messages'] else 'default'
    memories = store.search(namespace, query=query, limit=2)
    memory_info = '\n'.join([mem.value.get('info', '') for mem in memories])

    # Create a new memory
    new_memory_id = str(uuid.uuid4())
    new_memory = {'info': f'User said: {query}. Stored at {new_memory_id}'}
    store.put(namespace, new_memory_id, new_memory)

    # Return state update
    return {'messages': [f'Chat response. Memories: {memory_info}']}


# Build the graph
workflow3 = StateGraph(MemoryState)
workflow3.add_node('chat', chat_node)
workflow3.add_edge(START, 'chat')
workflow3.add_edge('chat', END)

# Compile with checkpointer and store
checkpointer3 = InMemorySaver()
store3 = InMemoryStore()  # Could add embedding config for semantic search
graph3 = workflow3.compile(checkpointer=checkpointer3, store=store3)

# Generate and save diagram for graph3
diagram3 = graph3.get_graph().draw_mermaid_png()
with open('graph3_diagram.png', 'wb') as f:
    f.write(diagram3)
print('Saved graph3 diagram to graph3_diagram.png')

# Invoke on thread 1 with user_id
config_thread1 = {'configurable': {'thread_id': 'thread1', 'user_id': 'user123'}}
result_thread1 = graph3.invoke({'messages': ['Hello, remember me?']}, config_thread1)
print('Thread 1 Result:', result_thread1)

# Invoke on a NEW thread with SAME user_id (memories persist across threads)
config_thread2 = {'configurable': {'thread_id': 'thread2', 'user_id': 'user123'}}
result_thread2 = graph3.invoke({'messages': ["What's my previous message?"]}, config_thread2)
print('Thread 2 Result (memories carried over):', result_thread2)

# Manually check stored memories for the user
memories = store3.search(('user123', 'memories'))
print('All Memories for user123:')
for mem in memories:
    print(f'  - {mem.dict()}')

# %%
