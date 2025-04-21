"""
LangGraph Basics – Video 3 (Ollama edition)
Build a three‑step linear graph, call the llama3.2 model once, and show the final state.
"""

import os
from typing import List, TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from rich import print
from dotenv import load_dotenv
from os import getenv

load_dotenv()
MODEL_NAME = 'deepseek-r1:1.5b'


# -------------------------------------------------
# 1. Shared workflow state
# -------------------------------------------------
class WorkflowState(TypedDict, total=False):  # total=False → keys may be added later
    """ What data flows through the graph? """


# -------------------------------------------------
# 2. Node functions
# -------------------------------------------------
def start(state: WorkflowState) -> dict:
    """Record that we've started processing."""
    print(f"\n👋  Received: {state['user_input']}")
    return {"steps": ["start"]}


def step_one(state: WorkflowState) -> dict:
    """Call the LLM once and store its reply in the state."""


def step_two(state: WorkflowState) -> dict:
    """Demonstrate access to the LLM output produced in step 1."""


# -------------------------------------------------
# 3. Build the graph
# -------------------------------------------------
builder = StateGraph(WorkflowState)

# -------------------------------------------------
# 4. Compile & quick demo
# -------------------------------------------------
if __name__ == "__main__":
    app = builder.compile()
    print(f'Using model: {MODEL_NAME}')

    initial_state = {
        "user_input": "Hello LangGraph! How would you describe yourself in one sentence?",
        "steps": [],
    }
    print("\n--- Graph Structure ---")
    print(app.get_graph().draw_ascii())
    final_state = app.invoke(initial_state)

    print("\n🎉  Final state:")
    print(final_state)
