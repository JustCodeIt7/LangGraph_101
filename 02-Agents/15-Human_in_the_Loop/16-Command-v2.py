import os
from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START
from langgraph.types import Command
from IPython.display import display, Image

# Define a simple state for our graph
class AppState(TypedDict):
    user_input: str
    result: str
    status: str

# Define the nodes for the graph

def router(state: AppState) -> Command[Literal["handle_positive", "handle_negative"]]:
    """
    A router node that directs the flow based on user input.
    It uses Command to specify the next node to execute.
    """
    print("---" + "ROUTER" + "---")
    if "positive" in state["user_input"].lower():
        print("Input is positive, routing to handle_positive.")
        return Command(goto="handle_positive")
    else:
        print("Input is not positive, routing to handle_negative.")
        return Command(goto="handle_negative")

def handle_positive(state: AppState) -> Command[Literal["finish"]]:
    """
    This node handles the "positive" case.
    It updates the state and then uses Command to proceed to the finish node.
    """
    print("---" + "POSITIVE HANDLER" + "---")
    result = "Processed the positive case successfully."
    return Command(
        update={"result": result, "status": "processed"},
        goto="finish",
    )

def handle_negative(state: AppState) -> Command[Literal["finish"]]:
    """
    This node handles the "negative" case.
    It updates the state and then uses Command to proceed to the finish node.
    """
    print("---" + "NEGATIVE HANDLER" + "---")
    result = "Processed the negative case successfully."
    return Command(
        update={"result": result, "status": "processed"},
        goto="finish",
    )

def finish(state: AppState):
    """
    The final node in the graph. It sets the status to "completed".
    """
    print("---" + "FINISH" + "---")
    return {"status": "completed"}

# Build the graph
builder = StateGraph(AppState)

# The START edge now points to our router
builder.add_edge(START, "router")

# Add the nodes to the graph
builder.add_node(router)
builder.add_node(handle_positive)
builder.add_node(handle_negative)
builder.add_node(finish)

# Compile the graph
# Note: No other edges are needed because Command handles the routing.
graph = builder.compile()

# Visualize the graph
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"Could not display graph: {e}")


# --- Run the graph with different inputs ---

# Example 1: Positive input
print("=== Example 1: Positive Input ===")
positive_input = {"user_input": "This is a positive statement."}
result1 = graph.invoke(positive_input)
print(f"Final State: {result1}")

# Example 2: Negative input
print("=== Example 2: Negative Input ===")
negative_input = {"user_input": "This is a different statement."}
result2 = graph.invoke(negative_input)
print(f"Final State: {result2}")