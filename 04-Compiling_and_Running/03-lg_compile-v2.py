"""
Video 3 – Running Your LangGraph: Compile, Invoke, and Stream
─────────────────────────────────────────────────────────────
This file assumes you already know how to *build* a graph (see Videos 1‑2).
We re‑use a tiny three‑node workflow so we can highlight the runtime API.

Execution modes covered
───────────────────────
• invoke()     – single synchronous run, returns final state
• stream()     – generator that yields (node_name, state_diff) tuples in real‑time
• batch()      – run many inputs concurrently, returns list of final states
• ainvoke()    – async version of invoke
• astream()    – async version of stream
• abatch()     – async version of batch

We also show how to pass a `config` dictionary:
    {"configurable": {"thread_id": "vid‑3‑demo"}, "recursion_limit": 50}
"""

from __future__ import annotations

import asyncio
from typing import List, TypedDict

from langgraph.graph import StateGraph, END

# ─────────────────────────────────────────────────────────────
# 1.  Tiny demo graph – three linear steps
#    (reuse the same nodes we’ve built in previous videos)
# ─────────────────────────────────────────────────────────────
class WFState(TypedDict):
    user_input: str
    steps: List[str]


def start(state: WFState) -> dict:
    print(f"👋  start: {state['user_input']}")
    return {"steps": ["start"]}


def step_one(state: WFState) -> dict:
    print("🔧  step_one")
    return {"steps": state["steps"] + ["step_one"]}


def step_two(state: WFState) -> dict:
    print("✅  step_two")
    return {"steps": state["steps"] + ["step_two"]}


builder = StateGraph(WFState)
builder.add_node("start", start)
builder.add_node("step_one", step_one)
builder.add_node("step_two", step_two)

builder.add_edge("start", "step_one")
builder.add_edge("step_one", "step_two")
builder.add_edge("step_two", END)
builder.set_entry_point("start")

# ─────────────────────────────────────────────────────────────
# 2.  Compile once – creates an executable “app”
#    • Validation: state types & edge consistency
#    • Optimization: internal DAG → efficient runtime
# ─────────────────────────────────────────────────────────────
app = builder.compile()

# A handy config we’ll reuse in every call
CONFIG = {
    "configurable": {"thread_id": "vid‑3‑demo"},
    "recursion_limit": 50,
}

# Single input we’ll keep re‑using
INITIAL = {"user_input": "Hello LangGraph ☕", "steps": []}


# ─────────────────────────────────────────────────────────────
# 3.  Demonstration helpers – one function per execution mode
# ─────────────────────────────────────────────────────────────
def demo_invoke() -> None:
    print("\n📦  invoke() – sync, single run")
    final = app.invoke(INITIAL, config=CONFIG)
    print("🎉  final_state ->", final)


def demo_stream() -> None:
    print("\n📺  stream() – yields intermediate chunks")
    for node, diff in app.stream(INITIAL, config=CONFIG):
        print(f"  • {node} -> {diff}")
    print("✅  stream complete")


def demo_batch() -> None:
    print("\n🚀  batch() – process a list of inputs concurrently")
    inputs = [
        {**INITIAL, "user_input": f"Item #{i}"} for i in range(1, 4)
    ]
    finals = app.batch(inputs, config=CONFIG)
    for i, fs in enumerate(finals, start=1):
        print(f"  • Result {i}: {fs}")


async def demo_async() -> None:
    print("\n⚡  async versions – ainvoke / astream / abatch")

    # 1) ainvoke
    final = await app.ainvoke(INITIAL, config=CONFIG)
    print("  • ainvoke ->", final)

    # 2) astream
    print("  • astream:")
    async for node, diff in app.astream(INITIAL, config=CONFIG):
        print(f"      {node} -> {diff}")

    # 3) abatch
    inputs = [
        {**INITIAL, "user_input": f"Async #{i}"} for i in range(1, 4)
    ]
    finals = await app.abatch(inputs, config=CONFIG)
    for i, fs in enumerate(finals, start=1):
        print(f"      Result {i}: {fs}")


# ─────────────────────────────────────────────────────────────
# 4.  Run the showcase when executed as a script
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_invoke()
    demo_stream()
    demo_batch()
    asyncio.run(demo_async())

    print("\n🖼️  ASCII graph layout:")
    print(app.get_graph().draw_ascii())
