"""
LangGraph Basics – Video 3 (Ollama edition)
Build a three‑step linear graph, call the llama3.2 model once, and show the final state.
"""

import os
from typing import List, TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from rich import print

# -------------------------------------------------
# 0.  LLM setup (Ollama llama3.2)
# -------------------------------------------------
# Point to a running Ollama server – default localhost
llm = ChatOllama(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    model="llama3.2",
)


# -------------------------------------------------
# 1. Shared workflow state
# -------------------------------------------------
class WorkflowState(TypedDict, total=False):  # total=False → keys may be added later
    user_input: str
    steps: List[str]
    llm_reply: str  # set in step_one


# -------------------------------------------------
# 2. Node functions
# -------------------------------------------------
def start(state: WorkflowState) -> dict:
    """Record that we've started processing."""
    print(f"\n👋  Received: {state['user_input']}")
    return {"steps": ["start"]}


def step_one(state: WorkflowState) -> dict:
    """Call the LLM once and store its reply in the state."""
    print("\n🔧  Running step 1 – calling llama3.2 …")
    reply = llm.invoke(state["user_input"]).content
    print(f"🤖  LLM reply: {reply}")
    return {
        "steps": state["steps"] + ["step 1"],
        "llm_reply": reply,
    }


def step_two(state: WorkflowState) -> dict:
    """Demonstrate access to the LLM output produced in step 1."""
    print("\n✅  Running step 2 – previous LLM reply available:")
    print(state.get("llm_reply"))
    return {"steps": state["steps"] + ["step 2"]}


# -------------------------------------------------
# 3. Build the graph
# -------------------------------------------------
builder = StateGraph(WorkflowState)

builder.add_node("start", start)
builder.add_node("step_one", step_one)
builder.add_node("step_two", step_two)

builder.add_edge("start", "step_one")
builder.add_edge("step_one", "step_two")
builder.add_edge("step_two", END)

builder.set_entry_point("start")

# -------------------------------------------------
# 4. Compile & quick demo
# -------------------------------------------------
if __name__ == "__main__":
    app = builder.compile()

    initial_state = {
        "user_input": "Hello LangGraph! How would you describe yourself in one sentence?",
        "steps": [],
    }
    final_state = app.invoke(initial_state)
    print("\n--- Graph Structure ---")
    print(app.get_graph().draw_ascii())
    
    print("\n🎉  Final state:")
    print(final_state)
