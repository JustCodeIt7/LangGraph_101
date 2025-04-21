"""
02-LangGraph Connecting LLM Models
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
# -------------------------------------------------
# 0.  LLM setup (Ollama llama3.2)
# -------------------------------------------------
# MODEL_NAME = 'deepseek-r1:1.5b'
# MODEL_NAME = 'llama3.2:1b'
# llm = ChatOllama(
#     base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
#     model=MODEL_NAME
# )


# -------------------------------------------------
# 0. LLM setup OpenRouter
# -------------------------------------------------
# MODEL_NAME = 'google/gemma-3-27b-it:free'
# llm = ChatOpenAI(
#     api_key=getenv("OPENROUTER_API_KEY"),
#     base_url='https://openrouter.ai/api/v1',
#     model=MODEL_NAME,
# )
# -------------------------------------------------
# 0. LLM setup OpenAI
# -------------------------------------------------
MODEL_NAME = 'gpt-4o-mini'
API_KEY = getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model=MODEL_NAME, api_key=API_KEY,
                 temperature=0.0)



# -------------------------------------------------
# 1. Shared workflow state
# -------------------------------------------------
class WorkflowState(TypedDict, total=False):  # total=False → keys may be added later
    user_input: str
    steps: List[str]
    llm_reply: str

# -------------------------------------------------
# 2. Node functions
# -------------------------------------------------
def start(state: WorkflowState) -> dict:
    """Record that we've started processing."""
    print(f"\n👋  Received: {state['user_input']}")
    return {"steps": ["start"]}


def step_one(state: WorkflowState) -> dict:
    """Call the LLM once and store its reply in the state."""
    print(f"\n🔧  Running step 1 – calling {MODEL_NAME} ...")
    reply = llm.invoke(state["user_input"]).content
    # print(f"🤖  LLM reply: {reply}")
    return {
        "steps": state["steps"] + ["step 1"],
        "llm_reply": reply,
    }

def step_two(state: WorkflowState) -> dict:
    """Demonstrate access to the LLM output produced in step 1."""
    print("\n✅  Running step 2 – previous LLM reply available:")
    print(state.get("llm_reply"))
    return {"steps": state["steps"] + ["step 2"]}

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

    print(f'Using model: {MODEL_NAME}')

    initial_state: WorkflowState = {
        "user_input": "Hello LangGraph! How would you describe yourself in one sentence?",
        "steps": [],
    }
    print("\n--- Graph Structure ---")
    print(app.get_graph().draw_ascii())

    final_state = app.invoke(initial_state)

    print("\n🎉  Final state:")
    print(final_state)
