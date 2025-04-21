"""
LangGraph Basics
Build a three‑step linear graph and run it once.
"""

from typing import List, TypedDict
from langgraph.graph import StateGraph, END
from rich import print


# =========================================================
# 1️⃣ Define the shared state for the workflow
# ---------------------------------------------------------
class WorkflowState(TypedDict):
    """What data travels through the graph?"""
    user_input: str  # the text we receive from the user / UI
    steps: List[str]  # a running log of which nodes have executed


# =========================================================
# 2️⃣ Node functions – each returns a **dict of state updates**
# ---------------------------------------------------------
def start(state: WorkflowState) -> dict:
    """👋 First node – greet the user and initialize the `steps` list."""
    # Show what we received (handy for debugging & teaching)
    print(f"\n👋  Received: {state['user_input']}")
    # Return a *partial* state update – here we overwrite / create `steps`
    return {"steps": ["start"]}


def step_one(state: WorkflowState) -> dict:
    """🔧 Second node – pretend to do some processing."""
    print("\n🔧  Running step 1...")
    # We append our name to the running `steps` list
    return {"steps": state["steps"] + ["step 1"]}


def step_two(state: WorkflowState) -> dict:
    """✅ Third & final node – finish the workflow."""
    print("\n✅  Running step 2 ...")
    return {"steps": state["steps"] + ["step 2"]}


# =========================================================
# 3️⃣ Build the graph structure
# ---------------------------------------------------------
builder = StateGraph(WorkflowState)  # tell LangGraph what our state looks like
# -- Add nodes (name, callable) ---------------------------------------------
builder.add_node("start", start)
builder.add_node("step_one", step_one)
builder.add_node("step_two", step_two)

# -- Connect them with sequential edges -------------------------------------
#    start ➜ step_one ➜ step_two ➜ END
builder.add_edge("start", "step_one")
builder.add_edge("step_one", "step_two")
builder.add_edge("step_two", END)
# Which node should kick things off?
builder.set_entry_point("start")
# =========================================================
# 4️⃣ Compile & run once for a quick demo
# ---------------------------------------------------------

if __name__ == "__main__":
    # 4a. Compile
    app = builder.compile()

    # 4b. Prepare the initial state
    initial_state: WorkflowState = {
        "user_input": "Hello LangGraph!",
        "steps": [],
    }

    # 4c. Invoke (synchronous) – returns the *final* state dict
    final_state = app.invoke(initial_state)

    # 4d. Pretty‑print results for the audience
    print("\n--- Graph Structure ---")
    print(app.get_graph().draw_ascii())
    print("\n🎉  Final state:")
    print(final_state)
