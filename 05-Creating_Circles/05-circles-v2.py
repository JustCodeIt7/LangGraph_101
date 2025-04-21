"""
LangGraph 101 – Video 5: Creating Cycles
========================================
Illustrates how to build a *looping* workflow in LangGraph.  The graph
repeatedly prompts the user to guess a secret word until they succeed or
exceed a maximum number of attempts.

The structure mirrors a minimal **ReAct**‑style loop:

``agent``  ➜  ``judge``  ──┐
         ▲                │ (continue)
         └─────────────────┘
``judge``  ─(done)──▶  END

> 🎥  Talking‑points
> ------------------
> • Cycles are created by adding an *edge back* to a previous node.
> • Use a **condition node** (``judge``) to decide whether to exit or
>   continue the loop.
> • Protect against infinite loops with the ``recursion_limit`` option
>   in ``config``.
> • Make sure the condition function eventually returns a path to ``END``.
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import operator
from rich import print

SECRET_WORD = "langgraph"      # ✅ Word the user must guess
MAX_ATTEMPTS = 5               # 🔒 Safety guard

# ---------------------------------------------------------------------
# 1️⃣  Shared State definition
# ---------------------------------------------------------------------
# • ``attempts``  – how many guesses the user has made.
# • ``history``   – accumulate conversation text.
# • ``is_correct``– flag set to True when the guess matches SECRET_WORD.

class WorkflowState(TypedDict):
    attempts: int
    history: Annotated[list[str], operator.add]
    is_correct: bool


# ---------------------------------------------------------------------
# 2️⃣  Nodes
# ---------------------------------------------------------------------

def agent(state: WorkflowState):
    """Prompt user for a guess and record it."""
    guess = input("🤖  Guess the secret word → ").strip().lower()
    return {
        "attempts": state.get("attempts", 0) + 1,
        "history": [f"👤 You guessed: {guess}"],
        "is_correct": guess == SECRET_WORD,
    }


def judge(state: WorkflowState):
    """Assess the latest guess and tell the user if they are correct."""
    if state["is_correct"]:
        msg = "🎉 Correct!"
    elif state["attempts"] >= MAX_ATTEMPTS:
        msg = f"🚫 Out of attempts. The word was '{SECRET_WORD}'."
    else:
        msg = "❌ Incorrect – try again."
    return {"history": [msg]}


# ---------------------------------------------------------------------
# 3️⃣  Condition function for branching
# ---------------------------------------------------------------------

def continue_or_end(state: WorkflowState) -> str:
    """Return path name based on correctness / attempt count."""
    if state["is_correct"] or state["attempts"] >= MAX_ATTEMPTS:
        return "done"
    return "continue"


# ---------------------------------------------------------------------
# 4️⃣  Build the graph with a *cycle*
# ---------------------------------------------------------------------

graph = StateGraph(WorkflowState)

# -- Nodes -------------------------------------------------------------
graph.add_node("agent", agent)
graph.add_node("judge", judge)

# -- Entry point -------------------------------------------------------
graph.set_entry_point("agent")

# -- Sequential edge ---------------------------------------------------
graph.add_edge("agent", "judge")

# -- Conditional branching (judge decides) -----------------------------
graph.add_conditional_edges(
    source="judge",
    condition=continue_or_end,
    path_map={
        "continue": "agent",  # 🔄 Loop back to agent
        "done": END,          # ✅ Stop the graph
    },
)

# ---------------------------------------------------------------------
# 5️⃣  Compile & run with a recursion limit
# ---------------------------------------------------------------------
# The recursion_limit is an *absolute cap* on the number of node visits
# for a single invocation.  We set it slightly above MAX_ATTEMPTS × 2 to
# account for both agent and judge visits.

app = graph.compile()

if __name__ == "__main__":

    final_state = app.invoke(
        {"attempts": 0, "history": [], "is_correct": False},
        config={"recursion_limit": MAX_ATTEMPTS * 2 + 2}
    )

    print("\n🧾  History:")
    for line in final_state["history"]:
        print(line)

    print(f"\nTotal attempts: {final_state['attempts']}")
