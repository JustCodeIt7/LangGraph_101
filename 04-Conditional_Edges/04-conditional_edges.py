from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
import operator

# --- State Definition ---
# Define the structure of the data that will flow through the graph.
# 'choice' will determine the path taken.
class AgentState(TypedDict):
    choice: str
    # Optional: Add other state variables as needed
    log: Annotated[list[str], operator.add] # Example of accumulating state

# --- Node Functions ---
# Define the functions that represent the nodes in the graph.
# Each function takes the current state and returns a dictionary
# with updates to the state.

def start_node(state: AgentState) -> dict:
    """
    The entry point node. It logs the initial choice.
    """
    print("--- Starting Node ---")
    print(f"Initial choice: {state['choice']}")
    return {"log": ["Started with choice: " + state['choice']]}

def option_a_node(state: AgentState) -> dict:
    """
    Node executed if the choice is 'A'.
    """
    print("--- Option A Node ---")
    # Simulate some work specific to option A
    result = "Processed Option A"
    print(result)
    return {"log": [result]}

def option_b_node(state: AgentState) -> dict:
    """
    Node executed if the choice is 'B'.
    """
    print("--- Option B Node ---")
    # Simulate some work specific to option B
    result = "Processed Option B"
    print(result)
    return {"log": [result]}

# --- Conditional Edge Logic ---
# Define the function that determines which path to take based on the state.
# It must return the name of the edge (which corresponds to a key in the path_map).

def should_follow_path(state: AgentState) -> Literal["path_a", "path_b", "__error__"]:
    """
    Checks the 'choice' field in the state and returns the corresponding path name.
    """
    print("--- Condition Check ---")
    choice = state.get("choice", "").upper() # Get choice, default to empty string, convert to upper
    print(f"Evaluating choice: {choice}")
    if choice == "A":
        return "path_a"
    elif choice == "B":
        return "path_b"
    else:
        print("Invalid choice detected!")
        # It's good practice to handle unexpected states.
        # You could raise an error or route to a specific error-handling node.
        # Returning a specific string allows mapping to an error node or END.
        return "__error__" # Or route to an error handling node if defined

# --- Graph Definition ---
# Instantiate the graph and define its structure.

workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("start", start_node)
workflow.add_node("option_a", option_a_node)
workflow.add_node("option_b", option_b_node)
# Optional: Add an error handling node
# workflow.add_node("error_node", error_handling_function)

# Set the entry point
workflow.set_entry_point("start")

# Add conditional edges
# After the 'start' node, call 'should_follow_path' to decide the next step.
# Map the return values ('path_a', 'path_b') to the destination nodes.
workflow.add_conditional_edges(
    "start",  # The node whose state is checked after execution
    should_follow_path,  # The function that decides the path
    {
        # "condition_function_return_value": "destination_node_name"
        "path_a": "option_a",
        "path_b": "option_b",
        "__error__": END # If choice is invalid, end the graph
        # Or map "__error__" to "error_node" if you added one
    }
)

# Add regular edges from the option nodes to the end
workflow.add_edge("option_a", END)
workflow.add_edge("option_b", END)

# --- Compile the Graph ---
# Compile the graph into a runnable application.
app = workflow.compile()

# --- Run the Graph ---

print("\n--- Running with Choice 'A' ---")
initial_state_a = {"choice": "A", "log": []}
final_state_a = app.invoke(initial_state_a)
print("--- Final State (A) ---")
print(final_state_a)
# Expected Output: Logs from start_node, condition check, option_a_node

print("\n" + "="*30 + "\n") # Separator

print("--- Running with Choice 'B' ---")
initial_state_b = {"choice": "B", "log": []}
final_state_b = app.invoke(initial_state_b)
print("--- Final State (B) ---")
print(final_state_b)
# Expected Output: Logs from start_node, condition check, option_b_node

print("\n" + "="*30 + "\n") # Separator

print("--- Running with Invalid Choice 'C' ---")
initial_state_c = {"choice": "C", "log": []}
final_state_c = app.invoke(initial_state_c)
print("--- Final State (C) ---")
print(final_state_c)
# Expected Output: Logs from start_node, condition check (invalid), graph ends

# --- Visualization (Optional) ---
# You can print an ASCII representation or generate an image
try:
    # Print ASCII representation
    print("\n--- Graph Structure (ASCII) ---")
    print(app.get_graph().print_ascii())

    # # Generate a PNG image (requires graphviz and Python packages: pip install pygraphviz pillow)
    # from PIL import Image
    # img_bytes = app.get_graph().draw_png()
    # with open("conditional_graph.png", "wb") as f:
    #     f.write(img_bytes)
    # print("\nGraph image saved to conditional_graph.png")

except ImportError:
    print("\nInstall pygraphviz and pillow to generate graph images: pip install pygraphviz pillow")
except Exception as e:
    print(f"\nCould not generate graph visualization: {e}")

