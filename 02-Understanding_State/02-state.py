# --- Imports ---
import operator
from typing import TypedDict, List, Annotated, Dict, Any
from pydantic import BaseModel # Optional, for Pydantic schema example
from langgraph.graph import StateGraph, END
from rich import print

# --- Introduction: Role of State ---
# The 'state' in LangGraph is a central dictionary-like object that is passed
# between nodes in the graph. Each node can read from the state and return
# updates to it. This allows information to flow and be modified throughout
# the workflow.

# --- 1. Defining State Schema ---

# LangGraph requires you to define the *shape* of your state. This helps
# ensure consistency and allows LangGraph to manage updates correctly.
# You can use standard Python TypedDict or Pydantic BaseModels.

# Method 1: Using TypedDict (Standard Library)
# Good for simple structures, no external dependencies needed beyond Python.
class GraphStateTypedDict(TypedDict):
    """
    Represents the state of our graph using TypedDict.
    """
    input_text: str  # The initial input text
    processed_text: str | None # Text after processing by node_a
    # Use Annotated to specify how updates to this list should be handled.
    # operator.add for lists means 'extend' or 'concatenate'.
    log_entries: Annotated[List[str], operator.add]
    final_result: str | None # The final result after node_b

# Method 2: Using Pydantic BaseModel (Optional)
# Provides data validation and more complex type hints if needed.
# Requires installing pydantic: pip install pydantic
# class GraphStatePydantic(BaseModel):
#     """
#     Represents the state of our graph using Pydantic.
#     Provides validation.
#     """
#     input_text: str
#     processed_text: str | None = None # Default value example
#     # Pydantic automatically handles Annotated for accumulation with operator.add
#     log_entries: Annotated[List[str], operator.add] = [] # Default list
#     final_result: str | None = None


# --- 2. How Nodes Update State ---

# Nodes are Python functions (or LangChain Runnables) that receive the
# current state as input and return a dictionary containing *only the keys*
# they want to update and their new values.

def node_a_processor(state: GraphStateTypedDict) -> Dict[str, Any]:
    """
    A simple node that processes the input text and adds a log entry.
    Input: The current graph state.
    Output: A dictionary with keys to update in the state.
    """
    print("--- Inside Node A ---")
    current_input = state['input_text']
    print(f"Received input: {current_input}")

    # Perform some processing
    processed = f"Processed: {current_input.upper()}"
    log = f"Node A processed '{current_input}'."

    print(f"Returning updates: processed_text='{processed}', log_entries=['{log}']")
    # Return ONLY the fields to be updated
    return {
        "processed_text": processed,
        "log_entries": [log] # Note: This is a list containing one new entry
                             # Because 'log_entries' is Annotated with operator.add,
                             # LangGraph will append this list to the existing list.
    }

def node_b_finalizer(state: GraphStateTypedDict) -> Dict[str, Any]:
    """
    A node that uses the processed text to create a final result and adds a log.
    Demonstrates reading from state updated by a previous node.
    """
    print("--- Inside Node B ---")
    processed_text = state['processed_text'] # Read value set by node_a
    print(f"Received processed text: {processed_text}")

    if processed_text is None:
        final = "Error: No processed text found."
        log = "Node B failed: processed_text was None."
    else:
        final = f"Final Result: {processed_text}!!"
        log = f"Node B finalized result using '{processed_text}'."

    print(f"Returning updates: final_result='{final}', log_entries=['{log}']")
    # Return updates
    return {
        "final_result": final,
        "log_entries": [log] # This will also be appended to the existing log_entries
    }

# --- 3. Update Mechanisms: Overwrite vs. Accumulation ---

# By default, when a node returns a value for a key already in the state,
# the old value is overwritten.
# Example: If node_a returned {"input_text": "new value"}, the original input_text
#          would be replaced.

# Accumulation: We used `typing.Annotated[List[str], operator.add]` for `log_entries`.
# This tells LangGraph *how* to merge the update returned by the node with the
# existing value in the state. `operator.add` for lists means concatenation/extend.
# So, instead of overwriting `log_entries`, new entries returned by nodes are added.
# Other operators can be used for different accumulation logic (e.g., summing numbers).

# --- Build and Run the Graph ---

# Define the graph using the chosen state definition
workflow = StateGraph(GraphStateTypedDict)
# workflow = StateGraph(GraphStatePydantic) # If using Pydantic

# Add the nodes
workflow.add_node("node_a", node_a_processor)
workflow.add_node("node_b", node_b_finalizer)

# Define the edges (control flow)
workflow.set_entry_point("node_a") # Start with node_a
workflow.add_edge("node_a", "node_b") # Go from node_a to node_b
workflow.add_edge("node_b", END) # End after node_b

# Compile the graph into a runnable application
app = workflow.compile()

print("\n--- Graph Structure ---")
print(app.get_graph().draw_ascii())

# --- Execute and Observe State Changes ---

initial_state = {
    "input_text": "hello world",
    "log_entries": ["Initial log: Graph started."] # Start with an initial log
}

print("\n--- Running Graph ---")
print(f"Initial State: {initial_state}\n")

# Use stream to see state changes after each step
for step in app.stream(initial_state):
    step_name, step_state = list(step.items())[0]
    print(f"--- After Step: {step_name} ---")
    print(f"Current State: {step_state}\n")


print("\n--- Final State (from invoke) ---")
# You can also use invoke to just get the final result
final_state = app.invoke(initial_state)
print(final_state)

# --- Key Takeaways ---
# * State is defined with TypedDict or Pydantic.
# * Nodes read the current state and return *updates* as dictionaries.
# * By default, updates overwrite existing values.
# * Use `typing.Annotated[<type>, <reducer_function>]` (like operator.add)
#   in the state definition to accumulate values instead of overwriting.
# * The state allows data to flow and be modified across different steps
#   in your graph.