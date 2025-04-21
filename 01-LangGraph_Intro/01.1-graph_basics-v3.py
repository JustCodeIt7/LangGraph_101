"""
LangGraph Basics – Video 3 demo
Build a three‑step linear graph and run it once.
"""

from typing import List, TypedDict
from langgraph.graph import StateGraph, END


# -------------------------------------------------
# 1. Define the shared state for the workflow
# -------------------------------------------------
class WorkflowState(TypedDict):
    user_input: str
    steps: List[str]


# -------------------------------------------------
# 2. Node functions (each returns a dict of updates)
# -------------------------------------------------
def start(state: WorkflowState) -> dict:
    """Record that we've started processing."""
    print(f"👋  Received: {state['user_input']}")
    return {"steps": ["start"]}


def step_one(state: WorkflowState) -> dict:
    print("🔧  Running step 1 …")
    return {"steps": state["steps"] + ["step 1"]}


def step_two(state: WorkflowState) -> dict:
    print("✅  Running step 2 …")
    return {"steps": state["steps"] + ["step 2"]}


# -------------------------------------------------
# 3. Build the graph structure
# -------------------------------------------------
builder = StateGraph(WorkflowState)

builder.add_node("start", start)
builder.add_node("step_one", step_one)
builder.add_node("step_two", step_two)

# Sequential edges: start ➜ step_one ➜ step_two ➜ END
builder.add_edge("start", "step_one")
builder.add_edge("step_one", "step_two")
builder.add_edge("step_two", END)

builder.set_entry_point("start")

# -------------------------------------------------
# 4. Compile and run once for a quick demo
# -------------------------------------------------
if __name__ == "__main__":
    app = builder.compile()

    initial_state = {"user_input": "Hello LangGraph!", "steps": []}
    final_state = app.invoke(initial_state)
    print(app.get_graph().draw_ascii())
    print("\n🎉  Final state:")
    print(final_state)
