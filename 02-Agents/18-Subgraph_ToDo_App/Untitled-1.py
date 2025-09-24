import operator
import os
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from rich import print

################################ Configuration & Setup ################################

load_dotenv()

# Set up the LLM to use a local Ollama model.
llm = ChatOllama(model="llama3.2", temperature=0.0) # Use a low temperature for deterministic responses

################################ Application State Definition ################################

# Define the state object that will be passed between graph nodes.
class TodoState(TypedDict):
    """Represents the application's state."""
    todos: Annotated[List[str], operator.add] # Accumulate new todos into the list across graph runs
    reminder: str
    new_todo: str
    is_complex: bool # A flag for conditional routing in the enrichment subgraph

################################ Enrichment Subgraph Nodes ################################

def analyze_task_complexity(state: TodoState):
    """Analyzes if a task is simple or complex using an LLM."""
    print("--- SUBGRAPH 1: ANALYZING TASK ---")
    task = state["new_todo"]

    # Prepare a prompt to instruct the LLM to classify the task.
    prompt = [
        SystemMessage(
            "You are a task analyzer. Classify the task as 'SIMPLE' or 'COMPLEX'. "
            "A complex task requires multiple steps. A simple task is a single action."
        ),
        HumanMessage(f"Task: '{task}'"),
    ]
    # Invoke the LLM to get the classification.
    response = llm.invoke(prompt)

    # Set the complexity flag based on the LLM's response.
    complexity = "COMPLEX" in response.content.upper()
    print(f"Task '{task}' is {'COMPLEX' if complexity else 'SIMPLE'}.")
    return {"is_complex": complexity}


def breakdown_complex_task(state: TodoState):
    """Breaks down a complex task into sub-tasks using an LLM."""
    print("--- SUBGRAPH 1: BREAKING DOWN TASK ---")
    task = state["new_todo"]

    # Prepare a prompt instructing the LLM to act as a project manager.
    prompt = [
        SystemMessage(
            "You are a project manager. Break down the complex task into a main task with 2-3 sub-tasks."
        ),
        HumanMessage(f"Task: '{task}'"),
    ]
    response = llm.invoke(prompt)

    # Add the entire formatted response as a single new todo item.
    return {"todos": [response.content]}


def pass_through_simple_task(state: TodoState):
    """Passes a simple task directly to the todo list without modification."""
    print("--- SUBGRAPH 1: ADDING SIMPLE TASK ---")
    # Add the original task directly to the list of todos.
    return {"todos": [state["new_todo"]]}


def route_on_complexity(state: TodoState):
    """Determines the next node based on the task's complexity flag."""
    # Route to the 'breakdown' node if complex, otherwise to the 'simple' node.
    return "breakdown" if state["is_complex"] else "simple"

################################ Reminder Subgraph Node ################################

def prioritize_tasks_llm(state: TodoState):
    """Generates a prioritized list of all current tasks using an LLM."""
    print("\n--- SUBGRAPH 2: PRIORITIZING TASKS ---")

    # Prevent an LLM call if there are no tasks to prioritize.
    if not state["todos"]:
        return {"reminder": "No tasks to prioritize."}

    # Format the list of todos into a string for the prompt.
    task_list_str = "\n".join(f"- {task}" for task in state["todos"])

    # Prepare a prompt instructing the LLM to prioritize the task list.
    prompt = [
        SystemMessage(
            "You are a helpful assistant. Prioritize the following tasks."
        ),
        HumanMessage(f"Tasks:\n{task_list_str}"),
    ]
    response = llm.invoke(prompt)
    
    # Update the state with the prioritized list as a reminder.
    return {"reminder": response.content}

################################ Final Output Node ################################

def display_final_state(state: TodoState):
    """Displays the final list of tasks and the generated reminder."""
    print("\n--- FINAL TO-DO LIST & REMINDER ---")
    print("Tasks:")
    # Iterate through and display the accumulated tasks.
    for i, task in enumerate(state["todos"]):
        print(f"{i+1}. {task.strip()}")
    print(f"\nReminder:\n{state['reminder']}")
    print("------------------------------------")
    # This node modifies no state, so it returns an empty dictionary.
    return {}

################################ Graph & Subgraph Construction ################################

# Define the enrichment subgraph with conditional routing.
enrichment_graph = StateGraph(TodoState)
enrichment_graph.add_node("analyze", analyze_task_complexity)
enrichment_graph.add_node("breakdown", breakdown_complex_task)
enrichment_graph.add_node("simple", pass_through_simple_task)
enrichment_graph.add_edge(START, "analyze")
# Add conditional logic to route based on the output of the 'analyze' node.
enrichment_graph.add_conditional_edges(
    "analyze", # Start the conditional edge from the 'analyze' node.
    route_on_complexity, # Use this function to decide the next path.
    {"breakdown": "breakdown", "simple": "simple"}, # Map function output to destination nodes.
)
enrichment_graph.add_edge("breakdown", END)
enrichment_graph.add_edge("simple", END)
# Compile the subgraph into a runnable application.
enrichment_app = enrichment_graph.compile()

# Define the reminder subgraph as a simple, single-step graph.
reminder_graph = StateGraph(TodoState)
reminder_graph.add_node("prioritize", prioritize_tasks_llm)
reminder_graph.add_edge(START, "prioritize")
reminder_graph.add_edge("prioritize", END)
# Compile the subgraph.
reminder_app = reminder_graph.compile()

# Define the main graph that orchestrates the subgraphs.
main_graph = StateGraph(TodoState)
main_graph.add_node("enrich_task", enrichment_app)
main_graph.add_node("generate_reminder", reminder_app)
main_graph.add_node("display_results", display_final_state)
main_graph.add_edge(START, "enrich_task")
main_graph.add_edge("enrich_task", "generate_reminder")
main_graph.add_edge("generate_reminder", "display_results")
main_graph.add_edge("display_results", END)
# Compile the main graph into the final runnable application.
app = main_graph.compile()

################################ Graph Visualization ################################

# Attempt to draw the graph structures to the console for visualization.
try:
    print("--- Enrichment Subgraph ---")
    print(enrichment_app.get_graph().draw_ascii())
    print("\n--- Reminder Subgraph ---")
    print(reminder_app.get_graph().draw_ascii())
    print("\n--- Main Graph ---")
    print(app.get_graph().draw_ascii())
# Gracefully handle cases where graphviz is not installed.
except Exception:
    print("Graphviz not installed. Skipping graph visualization.")

################################ Application Execution ################################

# Initialize an empty state for the first run of the graph.
current_state = {"todos": [], "reminder": "", "new_todo": "", "is_complex": False}

# Define a list of tasks to be processed by the graph.
tasks_to_add = [
    "Schedule dentist appointment", # Expected to be simple
    "Plan weekend trip", # Expected to be complex
    "Buy groceries", # Expected to be simple
]

# Process each task sequentially, accumulating the results in the state object.
for task in tasks_to_add:
    print(f"\n\n>>> PROCESSING NEW TASK: {task} <<<")
    inputs = {"new_todo": task}
    # Invoke the graph, passing the new task and the accumulated state from previous runs.
    current_state = app.invoke(inputs, {"recursion_limit": 100}, state=current_state)
