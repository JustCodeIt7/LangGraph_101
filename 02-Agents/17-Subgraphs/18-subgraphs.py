# %%
import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from rich import print
# %%
################################ Tracing Configuration ################################

# Configure LangSmith for tracing during development and debugging.
# Uncomment and replace "YOUR_LANGSMITH_API_KEY" to enable.
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGSMITH_API_KEY"
# %%
################################ Subgraph Definition ################################


# Define the state for the subgraph.
# This state is isolated from the parent graph's state.
class SubgraphState(TypedDict):
    """
    Define the state schema for the subgraph.
    It tracks the name used for the greeting and the evolving greeting message.
    """

    name: str
    greeting: str


def add_salutation(state: SubgraphState) -> dict:
    """
    Generate an initial greeting for the provided name.
    """
    print('>> Subgraph Node: add_salutation')
    name = state['name']
    # Return the updated greeting to the subgraph state
    return {'greeting': f'Hello, {name}!'}


def add_inquiry(state: SubgraphState) -> dict:
    """
    Append a friendly inquiry to the existing greeting.
    """
    print('>> Subgraph Node: add_inquiry')
    greeting = state['greeting']
    # Update the greeting with an additional phrase
    return {'greeting': f'{greeting} How are you today?'}


# %%
# Initialize the StateGraph for the subgraph.
subgraph_builder = StateGraph(SubgraphState)

# Define the nodes within the subgraph.
subgraph_builder.add_node('salutation', add_salutation)
subgraph_builder.add_node('inquiry', add_inquiry)

# Define the workflow path within the subgraph.
subgraph_builder.add_edge(START, 'salutation')
subgraph_builder.add_edge('salutation', 'inquiry')
subgraph_builder.add_edge('inquiry', END)

# Compile the subgraph into a runnable object.
subgraph = subgraph_builder.compile()
# %%
print('Subgraph structure:')
# Visualize the compiled subgraph structure.
try:
    from IPython.display import Image, display

    display(Image(subgraph.get_graph().draw_mermaid_png()))
except Exception:
    pass
# %%
################################ Parent Graph Definition ################################


# Define the state for the parent graph.
# This state manages the overall message flow and includes the name input.
class ParentState(TypedDict):
    """
    Define the state schema for the parent graph.
    It holds the initial name and the final composite message.
    """

    name: str
    full_message: str


# %%
def prepare_input(state: ParentState) -> dict:
    """
    Initialize the full message with a preparatory statement.
    """
    print('-> Parent Node: prepare_input')
    # Set an initial message based on the input name
    return {'full_message': f'Preparing message for {state["name"]}...'}


def run_subgraph(state: ParentState) -> dict:
    """
    Execute the predefined subgraph and incorporate its output into the parent state.
    """
    print('-> Parent Node: run_subgraph (invoking subgraph now)')

    # Map parent state variables to the subgraph's expected input state.
    subgraph_input = {'name': state['name']}

    # Invoke the compiled subgraph with its specific input.
    subgraph_output = subgraph.invoke(subgraph_input)

    # Extract the final greeting from the subgraph's output state.
    final_greeting = subgraph_output['greeting']

    print('-> Parent Node: run_subgraph (subgraph finished)')

    # Update the parent's full_message with the subgraph's result.
    return {'full_message': final_greeting}


def add_closing(state: ParentState) -> dict:
    """
    Append a concluding remark to the generated message.
    """
    print('-> Parent Node: add_closing')
    message = state['full_message']
    # Concatenate the closing phrase to the existing message
    return {'full_message': f'{message} Have a great day!'}


# %%

# Initialize the StateGraph for the parent graph.
parent_builder = StateGraph(ParentState)

# Define the nodes within the parent graph.
parent_builder.add_node('prepare', prepare_input)
parent_builder.add_node('generate_greeting', run_subgraph)
parent_builder.add_node('close', add_closing)

# Define the main workflow path for the parent graph.
parent_builder.add_edge(START, 'prepare')
parent_builder.add_edge('prepare', 'generate_greeting')
parent_builder.add_edge('generate_greeting', 'close')
parent_builder.add_edge('close', END)

# Compile the parent graph into a runnable object.
main_graph = parent_builder.compile()
# %%
print('Parent graph structure:')
# Visualize the compiled parent graph structure.
try:
    from IPython.display import Image, display

    display(Image(main_graph.get_graph().draw_mermaid_png()))
except Exception:
    pass

# %%
################################ Graph Execution ################################

# Define the initial input for the main graph.
initial_input = {'name': 'Alice'}
print(f'🚀 Starting graph with input: {initial_input}\n')

# Stream events to observe intermediate node outputs.
# Iterate through each event to print node outputs as they occur.
for event in main_graph.stream(initial_input, debug=True):
    for node, output in event.items():
        print(f"✅ Output from '{node}':")
        print(f'   {output}')
        print('--------------------')

# Invoke the graph to get the final state directly.
final_state = main_graph.invoke(initial_input)
print(f'\n✨ Final Result:\n{final_state["full_message"]}')

# %%
