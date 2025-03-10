from IPython.core.display import Image
from langgraph.graph import StateGraph
from typing import TypedDict


# Define the state schema for the graph.
# Talking Point: The state schema defines the data structure passed between nodes.
class GraphState(TypedDict):
    messages: list[dict]


# Define the first node function that processes the state.
def process_node(state: GraphState) -> GraphState:
    # Process the state by appending a processed message.
    new_message = {"content": "Processed message", "status": "processed"}
    state["messages"].append(new_message)
    # Talking Point: This node demonstrates the initial processing step.
    return state


# Define an additional node function that finalizes the state.
def finalize_node(state: GraphState) -> GraphState:
    # Further process the state by appending a finalized message.
    final_message = {"content": "Finalized message", "status": "finalized"}
    state["messages"].append(final_message)
    # Talking Point: This node shows how subsequent steps can extend the workflow.
    return state


# Create the StateGraph instance with the defined state schema.
# Talking Point: The graph orchestrates the sequence of node executions.
graph = StateGraph(GraphState)

# Add both nodes to the graph.
graph.add_node("process", process_node)
graph.add_node("finalize", finalize_node)

# Add an edge to connect the "process" node to the "finalize" node.
# Talking Point: Edges define the flow of execution between nodes.
graph.add_edge("process", "finalize")

# Set the entry point to "process" and the finish point to "finalize".
# Talking Point: Clearly defined start and end points ensure a predictable workflow.
graph.set_entry_point("process")
graph.set_finish_point("finalize")

# Compile the graph into an executable application.
app = graph.compile()
# display the graph structure with Ascii art.
print(app.get_graph().draw_ascii())
# Display the graph structure as a PNG image.
img = app.get_graph().draw_mermaid_png()
# save the image to a file.
with open("graph.png", "wb") as f:
    f.write(img)


print(app.get_graph().draw_mermaid())


# Run the graph with an initial state.
initial_state = {"messages": []}
result = app.invoke(initial_state)

# Output the final state to observe the accumulated changes.
print("Final state after graph execution:", result)
