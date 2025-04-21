"""
LangGraph 101 – Video 6: Tool Calling with ToolNode
===================================================
A minimal demo showing how to integrate external **Tools** into a
LangGraph workflow using the built‑in ``ToolNode``.
The workflow:

1. ``collect_input``  – prompt the user.
2. ``llm``            – a toy LLM function decides whether it needs a
   tool (square a number or fetch the current time) and emits a *tool
   call* in its message.
3. Conditional edge:
   * If a tool call is present → ``tool_node``
   * Otherwise                 → ``final_answer``
4. ``tool_node`` executes the call and appends a ``ToolMessage``.
5. ``final_answer`` reads the result and replies to the user → END.
"""

from __future__ import annotations
from typing import TypedDict, Annotated, List, Dict, Any

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from langchain.tools import tool, BaseTool
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

import operator
import datetime as _dt
import re
from rich import print

# ---------------------------------------------------------------------
# 1️⃣  Define LangChain *Tools*
# ---------------------------------------------------------------------

@tool
def square(n: int) -> str:
    """Return the square of an integer n."""
    return str(n * n)


@tool
def utc_now() -> str:
    """Return the current UTC time as an ISO‑8601 string."""
    return _dt.datetime.utcnow().isoformat()


TOOLS: List[BaseTool] = [square, utc_now]


# ---------------------------------------------------------------------
# 2️⃣  State schema – we only store the conversation ``messages`` list
# ---------------------------------------------------------------------

class WorkflowState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


# ---------------------------------------------------------------------
# 3️⃣  Helper – detect if last assistant message includes a tool call
# ---------------------------------------------------------------------

def last_message_has_tool_call(state: WorkflowState) -> str:
    last = state["messages"][-1]
    return "tool" if isinstance(last, AIMessage) and last.tool_calls else "done"


# ---------------------------------------------------------------------
# 4️⃣  Node definitions
# ---------------------------------------------------------------------


def collect_input(state: WorkflowState):
    user_text = input("💬  Ask me something → ")
    return {
        "messages": [HumanMessage(content=user_text)]
    }


def llm(state: WorkflowState):
    user_msg = state["messages"][-1].content.lower()

    num_match = re.search(r"(-?\d+)", user_msg)
    if num_match:
        n = int(num_match.group(1))
        assistant = AIMessage(
            content="",
            tool_calls=[
                {"name": "square", "arguments": {"n": n}, "id": "tool_square"}
            ],
        )
    elif "time" in user_msg:
        assistant = AIMessage(
            content="",
            tool_calls=[
                {"name": "utc_now", "arguments": {}, "id": "tool_time"}
            ],
        )
    else:
        assistant = AIMessage(content="Sorry, I can only square numbers or tell the time.")

    return {"messages": [assistant]}


def final_answer(state: WorkflowState):
    tool_msg = next(
        (m for m in reversed(state["messages"]) if isinstance(m, ToolMessage)),
        None,
    )
    if tool_msg:
        response_text = f"Result ➜ {tool_msg.content}"
    else:
        last_ai_msg = state["messages"][-1]
        response_text = last_ai_msg.content

    return {"messages": [AIMessage(content=response_text)]}


# ---------------------------------------------------------------------
# 5️⃣  Build the graph
# ---------------------------------------------------------------------

graph = StateGraph(WorkflowState)

graph.add_node("collect_input", collect_input)

graph.add_node("llm", llm)

graph.add_node("final_answer", final_answer)

graph.set_entry_point("collect_input")

graph.add_edge("collect_input", "llm")

# ToolNode automatically handles execution & adds ToolMessage to state

tool_node = ToolNode(TOOLS)
graph.add_node("tool_node", tool_node)

# Conditional routing based on whether llm emitted a tool call

graph.add_conditional_edges(
    source="llm",
    path=last_message_has_tool_call,
    path_map={
        "tool": "tool_node",
        "done": "final_answer",
    },
)

# After running the tool, go to final_answer

graph.add_edge("tool_node", "final_answer")

graph.add_edge("final_answer", END)


# ---------------------------------------------------------------------
# 6️⃣  Compile and run
# ---------------------------------------------------------------------

app = graph.compile()

if __name__ == "__main__":
    state = app.invoke({"messages": []})

    print("\n📜  Conversation history:")
    for msg in state["messages"]:
        role = msg.type.upper()
        content = msg.content
        if isinstance(msg, AIMessage) and msg.tool_calls:
            role = "ASSISTANT (tool call)"
            content = f"Tool Calls: {msg.tool_calls}"
        elif isinstance(msg, ToolMessage):
            role = "TOOL"
        print(f"[{role}] {content}")
