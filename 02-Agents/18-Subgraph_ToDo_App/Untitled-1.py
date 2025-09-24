import operator
import os
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from rich import print

# Load environment variables from .env file
load_dotenv()
# Set up the LLM, assuming Ollama is running. Update model as needed.
# Use your desired model, e.g., llama3.2, gemma2
llm = ChatOllama(model="llama3.2", temperature=0.0)


# --- State Definition ---
class TodoState(TypedDict):
    """Represents the application's state."""
    # Use operator.add to accumulate items into the list across graph runs.
    todos: Annotated[List[str], operator.add]
    reminder: str
    new_todo: str
    is_complex: bool  # Flag for conditional routing in the enrichment subgraph


# --- Enrichment Subgraph Nodes ---
def analyze_task_complexity(state: TodoState):
    """Analyzes if a task is simple or complex using an LLM."""
    print("--- SUBGRAPH 1: ANALYZING TASK ---")
    task = state["new_todo"]
    prompt = [
        SystemMessage(
            "You are a task analyzer. Classify the task as 'SIMPLE' or 'COMPLEX'. "
            "A complex task requires multiple steps. A simple task is a single action."
        ),
        HumanMessage(f"Task: '{task}'"),
    ]
    response = llm.invoke(prompt)
    complexity = "COMPLEX" in response.content.upper()
    print(f"Task '{task}' is {'COMPLEX' if complexity else 'SIMPLE'}.")
    return {"is_complex": complexity}


def breakdown_complex_task(state: TodoState):
    """Breaks down a complex task into sub-tasks using an LLM."""
    print("--- SUBGRAPH 1: BREAKING DOWN TASK ---")
    task = state["new_todo"]
    prompt = [
        SystemMessage(
            "You are a project manager. Break down the complex task into a main task with 2-3 sub-tasks."
        ),
        HumanMessage(f"Task: '{task}'"),
    ]
    response = llm.invoke(prompt)
    # The entire formatted response is added as a single todo item.
    return {"todos": [response.content]}


def pass_through_simple_task(state: TodoState):
    """Passes a simple task directly to the todo list."""
    print("--- SUBGRAPH 1: ADDING SIMPLE TASK ---")
    return {"todos": [state["new_todo"]]}


def route_on_complexity(state: TodoState):
    """Conditional routing based on task complexity."""
    return "breakdown" if state["is_complex"] else "simple"


# --- Reminder Subgraph Nodes ---
def prioritize_tasks_llm(state: TodoState):
    """Generates a prioritized list of tasks using an LLM."""
    print("\n--- SUBGRAPH 2: PRIORITIZING TASKS ---")
    if not state["todos"]:
        return {"reminder": "No tasks to prioritize."}

    task_list_str = "\n".join(f"- {task}" for task in state["todos"])
    prompt = [
        SystemMessage(
            "You are a helpful assistant. Prioritize the following tasks."
        ),
        HumanMessage(f"Tasks:\n{task_list_str}"),
    ]
    response = llm.invoke(prompt)
    return {"reminder": response.content}


# --- Final Display Node ---
def display_final_state(state: TodoState):
    """Displays the final list of tasks and the generated reminder."""
    print("\n--- FINAL TO-DO LIST & REMINDER ---")
    print("Tasks:")
    for i, task in enumerate(state["todos"]):
        print(f"{i+1}. {task.strip()}")
    print(f"\nReminder:\n{state['reminder']}")
    print("------------------------------------")
    return {}


# --- Graph Construction ---

# 1. Enrichment Subgraph (with conditional logic)
enrichment_graph = StateGraph(TodoState)
enrichment_graph.add_node("analyze", analyze_task_complexity)
enrichment_graph.add_node("breakdown", breakdown_complex_task)
enrichment_graph.add_node("simple", pass_through_simple_task)
enrichment_graph.add_edge(START, "analyze")
enrichment_graph.add_conditional_edges(
    "analyze", route_on_complexity, {"breakdown": "breakdown", "simple": "simple"}
)
enrichment_graph.add_edge("breakdown", END)
enrichment_graph.add_edge("simple", END)
enrichment_app = enrichment_graph.compile()

# 2. Reminder Subgraph (simple chain)
reminder_graph = StateGraph(TodoState)
reminder_graph.add_node("prioritize", prioritize_tasks_llm)
reminder_graph.add_edge(START, "prioritize")
reminder_graph.add_edge("prioritize", END)
reminder_app = reminder_graph.compile()

# 3. Main Graph (integrates the two subgraphs)
main_graph = StateGraph(TodoState)
main_graph.add_node("enrich_task", enrichment_app)
main_graph.add_node("generate_reminder", reminder_app)
main_graph.add_node("display_results", display_final_state)
main_graph.add_edge(START, "enrich_task")
main_graph.add_edge("enrich_task", "generate_reminder")
main_graph.add_edge("generate_reminder", "display_results")
main_graph.add_edge("display_results", END)
app = main_graph.compile()


# --- Graph Visualization (Optional) ---
try:
    print("--- Enrichment Subgraph ---")
    print(enrichment_app.get_graph().draw_ascii())
    print("\n--- Main Graph ---")
    print(app.get_graph().draw_ascii())
except Exception:
    print("Graphviz not installed. Skipping graph visualization.")


# --- Running the Application ---
# Initialize an empty state.
current_state = {"todos": [], "reminder": "", "new_todo": "", "is_complex": False}

# Execute the graph sequentially, accumulating tasks in the state.
tasks_to_add = [
    "Schedule dentist appointment",  # Simple
    "Plan weekend trip",  # Complex
    "Buy groceries",  # Simple
]

for task in tasks_to_add:
    print(f"\n\n>>> PROCESSING NEW TASK: {task} <<<")
    inputs = {"new_todo": task}
    current_state = app.invoke(inputs, state=current_state)