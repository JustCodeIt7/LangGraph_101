# %%
import operator
import os
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from rich import print
# %%
################################ Configuration & Setup ################################

load_dotenv()

# Set up the LLM, assuming Ollama is running locally.
# Update the model name to your preferred model (e.g., llama3.2, gemma2).
llm = ChatOllama(model="llama3.2", temperature=0.0)
# %%
################################ Application State Definition ################################

# Define the state object that will be passed between nodes in the graph.
class TodoState(TypedDict):
    """Represents the application's state."""
    todos: Annotated[List[str], operator.add] # Accumulate items into the list across graph runs
    reminder: str
    new_todo: str
    is_complex: bool # Flag for conditional routing in the enrichment subgraph
# %%
################################ Enrichment Subgraph Nodes ################################

def analyze_task_complexity(state: TodoState):
    """Analyzes if a task is simple or complex using an LLM."""
    print("--- SUBGRAPH 1: ANALYZING TASK ---")
    task = state["new_todo"]

    # Define the prompt to instruct the LLM to classify the task.
    prompt = [
        SystemMessage(
            "You are a task analyzer. Classify the task as 'SIMPLE' or 'COMPLEX'. "
            "A complex task requires multiple steps. A simple task is a single action."
        ),
        HumanMessage(f"Task: '{task}'"),
    ]
    response = llm.invoke(prompt)

    # Check the LLM's response to set the complexity flag.
    complexity = "COMPLEX" in response.content.upper()
    print(f"Task '{task}' is {'COMPLEX' if complexity else 'SIMPLE'}.")
    return {"is_complex": complexity}


def breakdown_complex_task(state: TodoState):
    """Breaks down a complex task into sub-tasks using an LLM."""
    print("--- SUBGRAPH 1: BREAKING DOWN TASK ---")
    task = state["new_todo"]

    # Define the prompt to instruct the LLM to act as a project manager.
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
    return {"todos": [state["new_todo"]]}


def route_on_complexity(state: TodoState):
    """Determines the next node based on the task's complexity flag."""
    return "breakdown" if state["is_complex"] else "simple"
# %%
################################ Reminder Subgraph Node ################################

def prioritize_tasks_llm(state: TodoState):
    """Generates a prioritized list of all current tasks using an LLM."""
    print("\n--- SUBGRAPH 2: PRIORITIZING TASKS ---")

    # Avoid calling the LLM if there are no tasks to prioritize.
    if not state["todos"]:
        return {"reminder": "No tasks to prioritize."}

    task_list_str = "\n".join(f"- {task}" for task in state["todos"])

    # Define the prompt to instruct the LLM to prioritize the task list.
    prompt = [
        SystemMessage(
            "You are a helpful assistant. Prioritize the following tasks."
        ),
        HumanMessage(f"Tasks:\n{task_list_str}"),
    ]
    response = llm.invoke(prompt)
    return {"reminder": response.content}
# %%
################################ Final Output Node ################################

def display_final_state(state: TodoState):
    """Displays the final list of tasks and the generated reminder in the console."""
    print("\n--- FINAL TO-DO LIST & REMINDER ---")
    print("Tasks:")
    # Iterate through the accumulated tasks and display them.
    for i, task in enumerate(state["todos"]):
        print(f"{i+1}. {task.strip()}")
    print(f"\nReminder:\n{state['reminder']}")
    print("------------------------------------")
    return {}
# %%
################################ Graph & Subgraph Construction ################################

# 1. Define and compile the enrichment subgraph with conditional routing.
enrichment_graph = StateGraph(TodoState)
enrichment_graph.add_node("analyze", analyze_task_complexity)
enrichment_graph.add_node("breakdown", breakdown_complex_task)
enrichment_graph.add_node("simple", pass_through_simple_task)
enrichment_graph.add_edge(START, "analyze")
enrichment_graph.add_conditional_edges(
    "analyze", # Source node
    route_on_complexity, # Function to determine the route
    {"breakdown": "breakdown", "simple": "simple"} # Map return values to destination nodes
)
enrichment_graph.add_edge("breakdown", END)
enrichment_graph.add_edge("simple", END)
enrichment_app = enrichment_graph.compile()
print(enrichment_app.get_graph().draw_ascii())

# %%
# 2. Define and compile the reminder subgraph as a simple, single-step graph.
reminder_graph = StateGraph(TodoState)
reminder_graph.add_node("prioritize", prioritize_tasks_llm)
reminder_graph.add_edge(START, "prioritize")
reminder_graph.add_edge("prioritize", END)
reminder_app = reminder_graph.compile()
print(reminder_app.get_graph().draw_ascii())
# %%
# 3. Define the main graph that orchestrates the subgraphs.
main_graph = StateGraph(TodoState)
main_graph.add_node("enrich_task", enrichment_app)
main_graph.add_node("generate_reminder", reminder_app)
main_graph.add_node("display_results", display_final_state)
main_graph.add_edge(START, "enrich_task")
main_graph.add_edge("enrich_task", "generate_reminder")
main_graph.add_edge("generate_reminder", "display_results")
main_graph.add_edge("display_results", END)
app = main_graph.compile()
print(app.get_graph().draw_ascii())
# %%
################################ Graph Visualization ################################

# Attempt to draw the graph structures to the console for visualization.
try:
    print("--- Enrichment Subgraph ---")
    print(enrichment_app.get_graph().draw_ascii())
    print("\n--- Main Graph ---")
    print(app.get_graph().draw_ascii())
# Gracefully handle cases where graphviz is not installed.
except Exception:
    print("Graphviz not installed. Skipping graph visualization.")

################################ Application Execution ################################
# %%
# Initialize an empty state for the first run.
current_state = {"todos": [], "reminder": "", "new_todo": "", "is_complex": False}

# Define a list of tasks to be processed by the graph.
tasks_to_add = [
    "Schedule dentist appointment", # Expected to be simple
    # "Plan weekend trip", # Expected to be complex
    # "Buy groceries", # Expected to be simple
]

# Sequentially process each task, accumulating the results in the state object.
for task in tasks_to_add:
    print(f"\n\n>>> PROCESSING NEW TASK: {task} <<<")
    inputs = {"new_todo": task}
    # Invoke the graph with the new task and the accumulated state.
    current_state = app.invoke(inputs, {"recursion_limit": 100}, state=current_state)

# %%
