# --- Video 3: Your First LangGraph ---
# Demonstrates building a basic linear graph with nodes and edges.

# Required imports
import os # Used to load environment variables
import time # Used to add delays for demonstration purposes
from typing import TypedDict, List # For defining the state structure
import operator # Used later for state accumulation (though not heavily in this simple example)

from dotenv import load_dotenv # Used to load environment variables from a .env file

from langgraph.graph import StateGraph, END # Core components for building the graph

# --- 1. Define the State ---
# Use TypedDict to define the structure of data that flows through the graph.
# This state will be passed to each node and edges can be conditioned on it.
class WorkflowState(TypedDict):
    """
    Represents the shared state of our simple workflow.
    """
    input_message: str # The initial message to process
    processing_log: List[str] # A log to track steps taken

# --- 2. Define Node Functions ---
# Nodes are functions or runnables that perform actions.
# They receive the current state and return a dictionary with updates to the state.

def start_node(state: WorkflowState) -> dict:
    """
    The first node in the workflow. Logs the start.
    Input: The current workflow state.
    Output: A dictionary with updates for the 'processing_log'.
    """
    print("--- Executing Start Node ---")
    initial_input = state.get("input_message", "No input provided")
    print(f"Input received: {initial_input}")

    # Get current log or initialize if it doesn't exist
    current_log = state.get("processing_log", [])
    
    # Create the update for the state (we are overwriting here for simplicity,
    # accumulation will be shown properly in later videos)
    log_update = ["Workflow started."]
    
    # Simulate some work
    time.sleep(1) 
    
    # Return the dictionary mapping state keys to their new values
    return {"processing_log": log_update}

def processing_step_node(state: WorkflowState) -> dict:
    """
    A node representing an intermediate processing step. Logs its execution.
    Input: The current workflow state.
    Output: A dictionary with updates for the 'processing_log'.
    """
    print("--- Executing Processing Step Node ---")
    
    # Get current log (it should exist after start_node)
    current_log = state.get("processing_log", [])
    
    # Create the update for the state (appending to the previous log)
    log_update = current_log + ["Processing step executed."] 
    
    # Simulate some work
    time.sleep(1)
    
    # Return the update
    return {"processing_log": log_update}

# --- 3. Build the Graph ---

# Instantiate the StateGraph with the defined state structure
graph_builder = StateGraph(WorkflowState)

# --- 4. Add Nodes ---
# Add each function as a node to the graph with a unique string name.
graph_builder.add_node("start", start_node)
graph_builder.add_node("processing_step", processing_step_node)

# --- 5. Set the Entry Point ---
# Define the node where the graph execution should begin.
graph_builder.set_entry_point("start")

# --- 6. Add Edges ---
# Connect the nodes to define the flow of execution.
# For a simple linear graph: start -> processing_step -> END

# Edge from 'start' node to 'processing_step' node
graph_builder.add_edge("start", "processing_step")

# Edge from 'processing_step' node to the special END node.
# END signifies that this path of the workflow is complete.
graph_builder.add_edge("processing_step", END)


# --- Compile and Run the Graph (Example Usage) ---

# Compile the graph into a runnable application
app = graph_builder.compile()

# Define the initial input for the graph
# This dictionary must match the keys expected by the entry point node
# and the overall state definition.
initial_input = {"input_message": "Hello, LangGraph Basics!", "processing_log": []}

print("Invoking the graph...")
# Run the graph with the initial input
# .invoke() runs the graph synchronously and returns the final state.
final_state = app.invoke(initial_input)

print("\n--- Workflow Finished ---")
print("Final State:")
# The final state contains the accumulated results after reaching END.
print(final_state)

# You can visualize the graph structure (requires graphviz)
# try:
#     img_data = app.get_graph().draw_png()
#     with open("video3_graph.png", "wb") as f:
#         f.write(img_data)
#     print("\nGraph visualization saved to video3_graph.png")
# except Exception as e:
#     print(f"\nCould not generate graph visualization: {e} (Graphviz might be needed)")