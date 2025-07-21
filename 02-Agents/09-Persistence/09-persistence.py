# persistence_examples.py
# Three examples demonstrating persistence in LangGraph

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
import uuid


# Define the state schema
class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]


# Node definitions


def node_a(state: State) -> dict:
    # Set foo to 1 and append to bar
    return {'foo': 1, 'bar': ['a']}


def node_b(state: State) -> dict:
    # Set foo to 2 and append to bar
    return {'foo': 2, 'bar': ['b']}


# Compile graph with in-memory checkpointer
checkpointer = InMemorySaver()
graph = StateGraph(State)
graph.add_node(node_a)
graph.add_node(node_b)
graph.add_edge(START, 'node_a')
graph.add_edge('node_a', 'node_b')
graph.add_edge('node_b', END)
graph = graph.compile(checkpointer=checkpointer)

# Example 1: Basic Checkpointing and State History
print('--- Example 1: Basic Checkpointing and State History ---')
thread_id = 'thread-1'
config = {'configurable': {'thread_id': thread_id}}
# Invoke the graph
graph.invoke({'foo': 0, 'bar': []}, config)
# Retrieve and print the full history of state snapshots
history = list(graph.get_state_history(config))
for idx, snapshot in enumerate(reversed(history)):
    print(f'Checkpoint {idx}: values={snapshot.values}, next={snapshot.next}')

# Example 2: Replay / Time Travel
print('\n--- Example 2: Replay / Time Travel ---')
# Choose the checkpoint_id of the state after node_a (step 1)
mid_checkpoint = history[-2].config['configurable']['checkpoint_id']
print('Replaying from checkpoint_id:', mid_checkpoint)
# Replay from that checkpoint
replay_config = {'configurable': {'thread_id': thread_id, 'checkpoint_id': mid_checkpoint}}
graph.invoke(None, replay_config)
# Verify that only node_b runs after replay
new_state = graph.get_state({'configurable': {'thread_id': thread_id}})
print('State after replay:', new_state.values)

# Example 3: Update State / Forking
print('\n--- Example 3: Update State / Forking ---')
# Overwrite foo and append to bar via update_state
update_values = {'foo': 99, 'bar': ['x']}
print('Updating state with:', update_values)
graph.update_state(config, update_values)
# Get the updated state
updated_snapshot = graph.get_state(config)
print('State after update_state:', updated_snapshot.values)
