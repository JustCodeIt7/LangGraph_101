"""
LangGraph – Video 2 demo
Understanding State: schema definitions, overwrite vs. accumulate.
Run:  python state_demo.py
"""

from typing import TypedDict, List, Annotated
from pydantic import BaseModel, Field
import operator
from langgraph.graph import StateGraph, END


# ------------------------------------------------------------------
# 1️⃣  TypedDict + Annotated (automatic accumulation)
# ------------------------------------------------------------------
class TDChatState(TypedDict):
    user_msg: str
    # ✨ List will be extended (operator.add) instead of overwritten
    messages: Annotated[List[str], operator.add]
    last_reply: str | None


def td_capture_user(state: TDChatState) -> dict:
    """Append the incoming user_msg to messages (accumulate)."""
    return {"messages": [f"User: {state['user_msg']}"]}


def td_bot_reply(state: TDChatState) -> dict:
    reply = f"Echo ➜ {state['user_msg']}"
    return {
        "messages": [f"Bot:  {reply}"],     # will be appended
        "last_reply": reply                 # overwrites each turn
    }


def build_td_graph() -> StateGraph:
    g = StateGraph(TDChatState)
    g.add_node("user", td_capture_user)
    g.add_node("bot", td_bot_reply)
    g.add_edge("user", "bot")
    g.add_edge("bot", END)
    g.set_entry_point("user")
    return g


# ------------------------------------------------------------------
# 2️⃣  Pydantic BaseModel (manual accumulation)
# ------------------------------------------------------------------
class PDChatState(BaseModel):
    user_msg: str
    messages: List[str] = Field(default_factory=list)  # manual merge
    last_reply: str | None = None


def pd_capture_user(state: PDChatState) -> dict:
    new = state.messages + [f"User: {state.user_msg}"]
    return {"messages": new}


def pd_bot_reply(state: PDChatState) -> dict:
    reply = f"Echo ➜ {state.user_msg}"
    new = state.messages + [f"Bot:  {reply}"]
    return {"messages": new, "last_reply": reply}


def build_pd_graph() -> StateGraph:
    g = StateGraph(PDChatState)
    g.add_node("user", pd_capture_user)
    g.add_node("bot", pd_bot_reply)
    g.add_edge("user", "bot")
    g.add_edge("bot", END)
    g.set_entry_point("user")
    return g


# ------------------------------------------------------------------
# 3️⃣  Quick demo runner
# ------------------------------------------------------------------
def run_once(app, msg: str):
    initial = {"user_msg": msg, "messages": [], "last_reply": None}
    final = app.invoke(initial)
    print("messages:", final["messages"])
    print("last_reply:", final["last_reply"])
    print("-" * 40)


if __name__ == "__main__":
    print("\n=== TypedDict + Annotated (auto‑accumulate) ===")
    run_once(build_td_graph().compile(), "Hello!")

    print("\n=== Pydantic BaseModel (manual accumulate) ===")
    run_once(build_pd_graph().compile(), "Hello!")
