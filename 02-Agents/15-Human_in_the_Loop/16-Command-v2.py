# %%
import os
from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START
from langgraph.types import Command
from IPython.display import display, Image
from rich import print
# %%
################################ State & Node Definitions ################################


# Define the structured state that will be passed between nodes in the graph.
class AppState(TypedDict):
    user_input: str
    result: str
    status: str


# %%
# Route the workflow based on the content of the user input.
def router(state: AppState) -> Command[Literal['handle_positive', 'handle_negative']]:
    print('---' + 'ROUTER' + '---')
    # Check for a keyword in the input to determine the execution path.
    if 'positive' in state['user_input'].lower():
        print('Input is positive, routing to handle_positive.')
        # Use Command to dynamically direct the graph to the next node.
        return Command(goto='handle_positive')
    else:
        print('Input is not positive, routing to handle_negative.')
        return Command(goto='handle_negative')


# Handle the "positive" branch of the graph's logic.
def handle_positive(state: AppState) -> Command[Literal['finish']]:
    print('---' + 'POSITIVE HANDLER' + '---')
    result = 'Processed the positive case successfully.'
    # Update the shared state and specify the next node to execute.
    return Command(
        update={'result': result, 'status': 'processed'},
        goto='finish',
    )


# Handle the "negative" branch of the graph's logic.
def handle_negative(state: AppState) -> Command[Literal['finish']]:
    print('---' + 'NEGATIVE HANDLER' + '---')
    result = 'Processed the negative case successfully.'
    # Update the shared state and specify the next node to execute.
    return Command(
        update={'result': result, 'status': 'processed'},
        goto='finish',
    )


# Define the terminal node that concludes the graph's execution.
def finish(state: AppState):
    print('---' + 'FINISH' + '---')
    # Perform a final state update.
    return {'status': 'completed'}


# %%
################################ Graph Construction ################################

# Instantiate the state graph with our defined state schema.
builder = StateGraph(AppState)

# Set the router as the entry point for the graph.
builder.add_edge(START, 'router')

# Register all the functions as nodes in the graph.
builder.add_node(router)
builder.add_node(handle_positive)
builder.add_node(handle_negative)
builder.add_node(finish)

# Compile the graph into a runnable object.
# No other edges are needed since Command handles the dynamic routing.
graph = builder.compile()

# Attempt to generate and display a visual representation of the graph.
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f'Could not display graph: {e}')
# %%
################################ Graph Execution ################################

# --- Run the graph with different inputs ---

# Execute the graph with an input that triggers the positive path.
print('=== Example 1: Positive Input ===')
positive_input = {'user_input': 'This is a positive statement.'}
result1 = graph.invoke(positive_input)
print(f'Final State: {result1}')
# %%
# Execute the graph with an input that triggers the negative path.
print('=== Example 2: Negative Input ===')
negative_input = {'user_input': 'This is a different statement.'}
result2 = graph.invoke(negative_input)
print(f'Final State: {result2}')

# %%
