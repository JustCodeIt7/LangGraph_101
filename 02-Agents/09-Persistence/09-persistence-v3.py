import operator
import os
import uuid
from typing import Annotated, TypedDict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Set up the model and memory
# The checkpointer is the core of persistence. It saves the state of the graph at each step.
# We're using SqliteSaver to persist to a local file, making it easy to inspect.
memory = SqliteSaver.from_conn_string('tutorial_persistence.sqlite')


# Define the state for our graph.
# `messages` is a list of messages that will be accumulated over time.
# The `add_messages` function is a special reducer that appends new messages
# to the existing list, rather than overwriting it.
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# Define the graph nodes
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)


def chatbot(state: State):
    """A simple chatbot node that responds to the last message."""
    return {'messages': [llm.invoke(state['messages'])]}


# --- Main Execution ---


def run_example(name, graph, inputs, thread_id):
    """Helper function to run and print an example."""
    print('--------------------------------------------------')
    print(f'## 🚀 Running Example: {name} for Thread ID: {thread_id}')
    print('--------------------------------------------------')

    config = {'configurable': {'thread_id': thread_id}}

    # The first invocation creates the thread and runs the graph
    print(f'\n▶️  Invoking graph with input: {inputs[0]}')
    graph.invoke(inputs[0], config)

    # Subsequent invocations continue the same thread
    for i in range(1, len(inputs)):
        print(f'\n▶️  Invoking graph with input: {inputs[i]}')
        graph.invoke(inputs[i], config)

    # Get the final state of the thread
    final_state = graph.get_state(config)
    print('\n✅ Final State:')
    for message in final_state.values['messages']:
        print(f'  - {message.type.upper()}: {message.content}')

    # Get the full history of the thread
    # Get the full history of the thread
    print('\n🔍 State History:')
    history = list(graph.get_state_history(config))
    print(f'  - Found {len(history)} checkpoints.')
    # Print the last 3 checkpoints for brevity
    for i, step in enumerate(history[-3:]):       
        print(f'\n  --- Checkpoint {len(history) - 3 + i} ---')
        print(f'    Next node: {step.next}')
        print('    Values:')
        for msg in step.values['messages']:
            print(f'      - {msg.type.upper()}: {msg.content}')

    return config, history

if __name__ == '__main__':
    # Clean up previous runs
    if os.path.exists('tutorial_persistence.sqlite'):
        os.remove('tutorial_persistence.sqlite')

    # The SqliteSaver must be used as a context manager
    with SqliteSaver.from_conn_string('tutorial_persistence.sqlite') as memory:
        # Define the graph structure
        workflow = StateGraph(State)
        workflow.add_node('chatbot', chatbot)
        workflow.add_edge(START, 'chatbot')
        workflow.add_edge('chatbot', END)

        # Compile the graph with the checkpointer INSIDE the 'with' block
        graph = workflow.compile(checkpointer=memory)

        # ==============================================================================
        # == Example 1: Basic Persistence & State Inspection
        # ==============================================================================
        ex1_thread_id = 'thread-1'
        ex1_inputs = [{'messages': [HumanMessage(content='Hi there! What is LangGraph?')]}]
        run_example('Basic Persistence', graph, ex1_inputs, ex1_thread_id)

        # ==============================================================================
        # == Example 2: Continuing a Conversation (Memory)
        # ==============================================================================
        ex2_thread_id = 'thread-2'
        ex2_inputs = [
            {'messages': [HumanMessage(content='My name is James.')]},
            {'messages': [HumanMessage(content='What is my name?')]},
        ]
        run_example('Conversation Memory', graph, ex2_inputs, ex2_thread_id)

        # ==============================================================================
        # == Example 3: Time Travel and State Modification
        # ==============================================================================
        print('--------------------------------------------------')
        print(f'## 🚀 Running Example: Time Travel for Thread ID: {ex2_thread_id}')
        print('--------------------------------------------------')

        # We'll create a new thread to make the time travel example self-contained
        time_travel_thread_id = 'thread-for-time-travel'
        config_ex3, history_ex3 = run_example('Setup for Time Travel', graph, ex2_inputs, time_travel_thread_id)

        # The history is ordered from most recent to oldest.
        # Checkpoint history_ex3[2] is after the first human message and AI response.
        time_travel_checkpoint = history_ex3[2]
        checkpoint_id_to_fork = time_travel_checkpoint.config['configurable']['checkpoint_id']

        print(f'\n🕰️  Traveling back to checkpoint where state was:')
        for message in time_travel_checkpoint.values['messages']:
            print(f'  - {message.type.upper()}: {message.content}')

        fork_config = {'configurable': {'thread_id': time_travel_thread_id}}

        print('\n✍️  Updating state at that checkpoint with a new message.')
        graph.update_state(
            fork_config, {'messages': [HumanMessage(content='Actually, my name is Gemini.')]}, as_node='chatbot'
        )

        print("\n▶️  Invoking graph again. The AI should now use the name 'Gemini'.")
        final_result = graph.invoke({'messages': [HumanMessage(content='What is my name?')]}, fork_config)

        print('\n✅ Final State of Forked Thread:')
        for message in final_result['messages']:
            print(f'  - {message.type.upper()}: {message.content}')