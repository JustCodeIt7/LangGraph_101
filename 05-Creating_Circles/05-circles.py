# --- Imports ---
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
import operator

# --- State Definition ---
# Define the state for our looping graph.
# 'counter' will track the number of iterations.
# 'max_loops' will define when to stop.
# 'log' will accumulate messages from each step.
class LoopState(TypedDict):
    counter: int
    max_loops: int
    log: Annotated[list[str], operator.add]

# --- Node Functions ---
# Define the functions representing graph nodes.

def process_step(state: LoopState) -> dict:
    """
    Represents a step within the loop.
    Increments the counter and logs the action.
    """
    current_count = state['counter']
    print(f"--- Processing Step {current_count + 1} ---")
    message = f"Executing step {current_count + 1}"
    print(message)
    # Increment the counter in the returned state update
    return {"counter": current_count + 1, "log": [message]}

# --- Conditional Edge Logic ---
# Define the function to decide whether to continue the loop or exit.

def should_continue(state: LoopState) -> Literal["continue_loop", "exit_loop"]:
    """
    Checks if the loop should continue based on the counter.
    """
    current_count = state['counter']
    max_loops = state['max_loops']
    print(f"--- Condition Check (Count: {current_count}, Max: {max_loops}) ---")
    if current_count < max_loops:
        print("Decision: Continue Loop")
        return "continue_loop"
    else:
        print("Decision: Exit Loop")
        return "exit_loop"

# --- Graph Definition ---
# Instantiate the graph and define its structure, including the cycle.

workflow = StateGraph(LoopState)

# Add nodes
workflow.add_node("process", process_step)
# Note: The condition check itself doesn't need a separate node if it only routes.
# We use add_conditional_edges directly after the 'process' node.

# Set the entry point
workflow.set_entry_point("process")

# Add the conditional edge for looping/exiting
# After the 'process' node, check the 'should_continue' condition.
workflow.add_conditional_edges(
    "process",      # Check state after this node runs
    should_continue, # Function to determine the next step
    {
        # If 'should_continue' returns "continue_loop", go back to "process"
        "continue_loop": "process",
        # If 'should_continue' returns "exit_loop", go to the predefined END node
        "exit_loop": END
    }
)
# This creates the cycle: process -> condition -> process (if continue)
# And the exit path: process -> condition -> END (if exit)

# --- Compile the Graph ---
app = workflow.compile()

# --- Run the Graph ---

print("\n--- Running the Loop (Max 3 loops) ---")
initial_state = {"counter": 0, "max_loops": 3, "log": []}
# Use the 'config' dictionary to set the recursion limit.
# This prevents infinite loops if the exit condition is never met.
config = {"recursion_limit": 10} # Set a safety limit higher than expected loops

final_state = app.invoke(initial_state, config=config)
print("\n--- Final State (After Loop) ---")
print(final_state)
# Expected Output: Logs for steps 1, 2, 3, then condition checks, finally exiting.
# Final state should have counter = 3.

print("\n" + "="*30 + "\n")

print("--- Running the Loop (Max 5 loops, Recursion Limit 4) ---")
initial_state_limit = {"counter": 0, "max_loops": 5, "log": []}
# Set recursion limit lower than max_loops to see it trigger
config_limit = {"recursion_limit": 4}

try:
    # This invoke call is expected to raise a RecursionError
    app.invoke(initial_state_limit, config=config_limit)
except RecursionError as e:
    print(f"\n--- Caught Expected Error ---")
    print(f"Successfully caught RecursionError: {e}")
    print("This demonstrates the recursion_limit preventing infinite execution.")
# Expected Output: Logs for steps 1, 2, 3, 4, then a RecursionError.

# --- Visualization (Optional) ---
try:
    print("\n--- Graph Structure (ASCII) ---")
    print(app.get_graph().print_ascii())

    # # Generate PNG (requires graphviz and pillow)
    # from PIL import Image
    # img_bytes = app.get_graph().draw_png()
    # with open("cycle_graph.png", "wb") as f:
    #     f.write(img_bytes)
    # print("\nGraph image saved to cycle_graph.png")

except ImportError:
    print("\nInstall pygraphviz and pillow to generate graph images: pip install pygraphviz pillow")
except Exception as e:
    print(f"\nCould not generate graph visualization: {e}")
