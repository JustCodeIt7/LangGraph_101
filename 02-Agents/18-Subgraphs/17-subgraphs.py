# main.py
import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# Optional: Set up LangSmith for tracing (highly recommended for development)
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGSMITH_API_KEY"


# --- 1. Define the Subgraph ---
# This graph has its own state and logic, completely separate from the parent.
# Its job is to take a name and create a detailed greeting.


class SubgraphState(TypedDict):
    """
    The state for our subgraph. It only knows about the 'name' and the 'greeting' it's building.
    """

    name: str
    greeting: str


def add_salutation(state: SubgraphState) -> dict:
    """
    First node in the subgraph: adds a salutation.
    """
    print('>> Subgraph Node: add_salutation')
    name = state['name']
    return {'greeting': f'Hello, {name}!'}


def add_inquiry(state: SubgraphState) -> dict:
    """
    Second node in the subgraph: adds a friendly question.
    """
    print('>> Subgraph Node: add_inquiry')
    greeting = state['greeting']
    return {'greeting': f'{greeting} How are you today?'}


# Create the subgraph builder
subgraph_builder = StateGraph(SubgraphState)

# Add nodes to the subgraph
subgraph_builder.add_node('salutation', add_salutation)
subgraph_builder.add_node('inquiry', add_inquiry)

# Define the workflow within the subgraph
subgraph_builder.add_edge(START, 'salutation')
subgraph_builder.add_edge('salutation', 'inquiry')
subgraph_builder.add_edge('inquiry', END)

# Compile the subgraph into a runnable graph
subgraph = subgraph_builder.compile()


# --- 2. Define the Parent Graph ---
# This is the main workflow that will invoke our subgraph.


class ParentState(TypedDict):
    """
    The state for the parent graph. It contains the final 'full_message'.
    """

    name: str
    full_message: str


def prepare_input(state: ParentState) -> dict:
    """
    First node in the parent graph: prepares the data.
    """
    print('-> Parent Node: prepare_input')
    return {'full_message': f'Preparing message for {state["name"]}...'}


def run_subgraph(state: ParentState) -> dict:
    """
    This is the key node that calls our subgraph.
    It acts as a bridge, mapping the parent state to the subgraph's input
    and the subgraph's output back to the parent's state.
    """
    print('-> Parent Node: run_subgraph (invoking subgraph now)')

    # The input for the subgraph must match SubgraphState
    subgraph_input = {'name': state['name']}

    # Invoke the subgraph and get the final state
    subgraph_output = subgraph.invoke(subgraph_input)

    # The output from the subgraph is in the 'greeting' key of its state
    final_greeting = subgraph_output['greeting']

    print('-> Parent Node: run_subgraph (subgraph finished)')

    # Return the result to update the parent state
    return {'full_message': final_greeting}


def add_closing(state: ParentState) -> dict:
    """
    Final node in the parent graph: adds a closing remark.
    """
    print('-> Parent Node: add_closing')
    message = state['full_message']
    return {'full_message': f'{message} Have a great day!'}


# Create the parent graph builder
parent_builder = StateGraph(ParentState)

# Add nodes to the parent graph
parent_builder.add_node('prepare', prepare_input)
parent_builder.add_node('generate_greeting', run_subgraph)
parent_builder.add_node('close', add_closing)

# Define the main workflow
parent_builder.add_edge(START, 'prepare')
parent_builder.add_edge('prepare', 'generate_greeting')
parent_builder.add_edge('generate_greeting', 'close')
parent_builder.add_edge('close', END)

# Compile the parent graph
main_graph = parent_builder.compile()


# --- 3. Run the Graph ---
if __name__ == '__main__':
    # The initial input must match the ParentState
    initial_input = {'name': 'Alice'}
    print(f'🚀 Starting graph with input: {initial_input}\n')

    # stream() lets us see the output of each node as it executes
    for event in main_graph.stream(initial_input):
        for node, output in event.items():
            print(f"✅ Output from '{node}':")
            print(f'   {output}')
            print('--------------------')

    # You can also use invoke() to get just the final result
    final_state = main_graph.invoke(initial_input)
    print(f'\n✨ Final Result:\n{final_state["full_message"]}')
