"""
Basic LangGraph subagent (subgraph) demo.

What this shows:
- Parent graph ("Orchestrator") hands off to a child subgraph ("Researcher").
- Parent and subgraph share a single state key: `messages`.
- We stream updates with subgraphs=True to show subgraph internals in the console.

Run:
    python subagent_demo.py
"""

from typing import List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START


# 1) Shared state schema (parent <-> subgraph communicate via `messages`)
class ChatState(TypedDict):
    messages: List[str]


# 2) Define the subgraph (our "Researcher" subagent)
#
# The Researcher has two internal steps:
#   - researcher_think: adds an analysis line
#   - researcher_answer: adds a final answer line
#
# Note: We're not calling an LLM here to keep the example simple and runnable
# without credentials. You can replace the body of these functions with your LLM calls.
def researcher_think(state: ChatState) -> ChatState:
    # Grab the user's question from messages (simplistic parsing)
    user_question = next(
        (m.split('User: ', 1)[1] for m in reversed(state['messages']) if m.startswith('User: ')),
        'unknown question',
    )
    thought = f"Researcher: Let me break that down – '{user_question}'."
    return {'messages': state['messages'] + [thought]}


def researcher_answer(state: ChatState) -> ChatState:
    # Imagine using the prior "think" step to inform the answer
    answer = 'Researcher: Short answer: Subgraphs let you nest workflows inside nodes.'
    return {'messages': state['messages'] + [answer]}


# Build the subgraph
researcher_builder = StateGraph(ChatState)
researcher_builder.add_node('researcher_think', researcher_think)
researcher_builder.add_node('researcher_answer', researcher_answer)
researcher_builder.add_edge(START, 'researcher_think')
researcher_builder.add_edge('researcher_think', 'researcher_answer')
researcher_subgraph = researcher_builder.compile()


# 3) Define the parent graph (the "Orchestrator")
def orchestrator_intro(state: ChatState) -> ChatState:
    intro = 'Orchestrator: Handing this off to the Researcher subagent...'
    return {'messages': state['messages'] + [intro]}


def orchestrator_wrapup(state: ChatState) -> ChatState:
    # Pull the last researcher message for a quick wrap
    last_researcher_msg = next(
        (m for m in reversed(state['messages']) if m.startswith('Researcher: ')),
        'Researcher: (no output)',
    )
    wrap = f'Orchestrator: Thanks, summarizing – {last_researcher_msg.replace("Researcher: ", "")}'
    return {'messages': state['messages'] + [wrap]}


# Build the parent graph
builder = StateGraph(ChatState)
builder.add_node('intro', orchestrator_intro)
# Add the compiled subgraph as a node in the parent graph
builder.add_node('research_agent', researcher_subgraph)
builder.add_node('wrapup', orchestrator_wrapup)

builder.add_edge(START, 'intro')
builder.add_edge('intro', 'research_agent')
builder.add_edge('research_agent', 'wrapup')

graph = builder.compile()


if __name__ == '__main__':
    # Initial state from the "user"
    initial_state: ChatState = {'messages': ['User: How do subgraphs (subagents) work in LangGraph?']}

    print('\n--- Streaming updates (including subgraph internals) ---')
    for chunk in graph.stream(
        initial_state,
        subgraphs=True,  # include subgraph updates in the stream
        stream_mode='updates',  # print only state updates
    ):
        print(chunk)

    print('\n--- Final state ---')
    final = graph.invoke(initial_state)
    for m in final['messages']:
        print(m)
