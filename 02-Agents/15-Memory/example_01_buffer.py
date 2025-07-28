"""
example_01_buffer.py  ────────────────────────────────────────────────
A one‑file demo of LangGraph’s *short‑term* memory using the built‑in
`InMemorySaver`. Perfect for quick prototypes or unit‑tests.

Run:
    python example_01_buffer.py
"""

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState, START
from rich import print
from langchain_ollama import ChatOllama
from langchain_litellm import ChatLiteLLM
# llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
llm = ChatLiteLLM(model_name='ollama/llama3.2', temperature=0)

# ── ❶ Graph node ────────────────────────────────────────────────────
def chat(state: MessagesState):
    """Echo user input while preserving history in RAM."""
    reply = llm.invoke(state['messages'])
    return {'messages': [reply]}


# ── ❷ Build graph ──────────────────────────────────────────────────
builder = StateGraph(MessagesState)
builder.add_node(chat)
builder.add_edge(START, 'chat')
graph = builder.compile(checkpointer=InMemorySaver())

# ── ❸ CLI loop ─────────────────────────────────────────────────────
if __name__ == '__main__':
    thread_cfg = {'configurable': {'thread_id': 'demo'}}
    while True:
        user = input('\nYou ➜ ')
        for event in graph.stream({'messages': user}, thread_cfg, stream_mode='messages'):
            print('Bot ➜', event)
        # YT: pause here to show how the state grows after each turn
