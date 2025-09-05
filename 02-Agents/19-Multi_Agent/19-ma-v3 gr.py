# =====================================
# Selected Examples for Multi-Agent LangGraph Architectures (Corrected Imports)
# =====================================
# This file contains the three selected examples, one for each architecture.
# Each section is self-contained and can be run independently.
# Requirements: pip install langgraph langchain langchain-ollama
# Corrected: MessagesState is imported from langgraph.graph, not langchain_core.messages

# =====================================
# 1. Supervisor Architecture (SECTION 2 with ChatOllama)
# =====================================
from typing import TypedDict
from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(model='llama3.2')


class State(MessagesState):
    result: str


def last_user(state: State) -> str:
    for msg in reversed(state['messages']):
        if getattr(msg, 'type', getattr(msg, 'role', '')) in ('human', 'user'):
            return msg.content if hasattr(msg, 'content') else msg.get('content', '')
    return ''


def supervisor(state: State):
    if state.get('result'):
        return Command(goto='finish', update={})
    q = last_user(state).lower()
    if any(op in q for op in ['+', '-', '*', '/', 'solve', 'calc']):
        return Command(goto='math', update={})
    return Command(goto='research', update={})


def research_agent(state: State):
    q = last_user(state)
    resp = llm.invoke([HumanMessage(content=f'Answer briefly and factually: {q}')])
    return Command(goto='supervisor', update={'result': resp.content})


def math_agent(state: State):
    q = last_user(state)
    prompt = f'Solve the math expression or question. Reply with the final answer only:\n{q}'
    resp = llm.invoke([HumanMessage(content=prompt)])
    return Command(goto='supervisor', update={'result': resp.content.strip()})


def finish(state: State):
    return Command(goto=END, update={'messages': [AIMessage(content=state.get('result', 'No result'))]})


builder = StateGraph(State)
builder.add_node('supervisor', supervisor)
builder.add_node('research', research_agent)
builder.add_node('math', math_agent)
builder.add_node('finish', finish)
builder.add_edge(START, 'supervisor')
graph_supervisor = builder.compile()

# Demo usage (uncomment to run)
# out = graph_supervisor.invoke({'messages': [HumanMessage("What's the capital of Japan?")]})
# for m in out['messages']:
#     m.pretty_print()
# out = graph_supervisor.invoke({'messages': [HumanMessage('Compute 12*(3+4)')]})
# for m in out['messages']:
#     m.pretty_print()

# =====================================
# 2. Network Architecture (SECTION 3 - Agents routing with iteration limit)
# =====================================
from typing import Literal, TypedDict
from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage


class NetworkState(MessagesState):
    """State for network of collaborating agents"""

    iteration_count: int = 0


# Network Agents - each can route to any other
def analyst_agent(state: NetworkState) -> Command[Literal['researcher_agent', 'strategist_agent', '__end__']]:
    """Analyst agent that processes data and routes accordingly"""
    last_message = state['messages'][-1].content
    response = f"Analyst: I've reviewed '{last_message}'. The data suggests we need deeper investigation."
    new_state = {'messages': [AIMessage(content=response)], 'iteration_count': state.get('iteration_count', 0) + 1}
    if state.get('iteration_count', 0) >= 3:
        return Command(goto='__end__', update=new_state)
    elif 'strategy' in last_message.lower():
        return Command(goto='strategist_agent', update=new_state)
    else:
        return Command(goto='researcher_agent', update=new_state)


def researcher_agent(state: NetworkState) -> Command[Literal['analyst_agent', 'strategist_agent', '__end__']]:
    """Researcher agent that gathers information"""
    last_message = state['messages'][-1].content
    response = f"Researcher: Based on my investigation of '{last_message}', I found relevant background information."
    new_state = {'messages': [AIMessage(content=response)], 'iteration_count': state.get('iteration_count', 0) + 1}
    if state.get('iteration_count', 0) >= 3:
        return Command(goto='__end__', update=new_state)
    elif 'analysis' in last_message.lower():
        return Command(goto='analyst_agent', update=new_state)
    else:
        return Command(goto='strategist_agent', update=new_state)


def strategist_agent(state: NetworkState) -> Command[Literal['analyst_agent', 'researcher_agent', '__end__']]:
    """Strategist agent that develops plans"""
    last_message = state['messages'][-1].content
    response = (
        f"Strategist: For '{last_message}', I recommend a comprehensive approach combining multiple perspectives."
    )
    new_state = {'messages': [AIMessage(content=response)], 'iteration_count': state.get('iteration_count', 0) + 1}
    if state.get('iteration_count', 0) >= 2:
        final_response = 'Strategist: Final recommendation compiled. All agents have contributed to the solution.'
        return Command(goto='__end__', update={'messages': [AIMessage(content=final_response)]})
    else:
        return Command(goto='analyst_agent', update=new_state)


# Build the network graph
def create_network_graph():
    workflow = StateGraph(NetworkState)
    workflow.add_node('analyst_agent', analyst_agent)
    workflow.add_node('researcher_agent', researcher_agent)
    workflow.add_node('strategist_agent', strategist_agent)
    workflow.set_entry_point('analyst_agent')
    return workflow.compile()


# Usage example (uncomment to run)
# if __name__ == '__main__':
#     graph_network = create_network_graph()
#     result = graph_network.invoke(
#         {'messages': [HumanMessage(content='We need to develop a strategy for market expansion')], 'iteration_count': 0}
#     )
#     print('Network Collaboration Result:')
#     for i, message in enumerate(result['messages']):
#         print(f'{i + 1}. {message.content}')
#         print()

# =====================================
# 3. Hierarchical Architecture (SECTION 6 - Hierarchical with Subgraphs)
# =====================================
from typing import TypedDict
from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

llm = ChatOllama(model='llama3.2')
embedding = OllamaEmbeddings(model='nomic-embed-text')


class TopState(MessagesState):
    data: str
    notes: str


def last_user(s: TopState) -> str:
    for m in reversed(s['messages']):
        if getattr(m, 'type', getattr(m, 'role', '')) in ('human', 'user'):
            return m.content if hasattr(m, 'content') else m.get('content', '')
    return ''


DOCS = [
    'Python was created by Guido van Rossum.',
    'The capital of Japan is Tokyo.',
    'LangGraph helps build multi-agent workflows.',
]
V_DOCS = [embedding.embed_query(d) for d in DOCS]


def gather(state: TopState):
    q = last_user(state)
    qv = embedding.embed_query(q)
    idx = max(range(len(DOCS)), key=lambda i: sum(a * b for a, b in zip(V_DOCS[i], qv)))
    return {'data': DOCS[idx]}


def synthesize(state: TopState):
    prompt = f'Using this note, answer the user briefly:\nNote: {state.get("data", "")}\nUser: {last_user(state)}'
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {'notes': resp.content}


def compute(state: TopState):
    q = last_user(state)
    resp = llm.invoke([HumanMessage(content=f'Analyze and give a score 0-1 with a reason for: {q}')])
    return {'data': resp.content}


def summarize(state: TopState):
    resp = llm.invoke([HumanMessage(content=f'Summarize to one sentence: {state.get("data", "")}')])
    return {'notes': resp.content}


# Research subgraph
r = StateGraph(TopState)
r.add_node('gather', gather)
r.add_node('synth', synthesize)
r.add_edge(START, 'gather')
r.add_edge('gather', 'synth')
r.add_edge('synth', END)
research_team = r.compile()

# Analysis subgraph
a = StateGraph(TopState)
a.add_node('compute', compute)
a.add_node('summary', summarize)
a.add_edge(START, 'compute')
a.add_edge('compute', 'summary')
a.add_edge('summary', END)
analysis_team = a.compile()


def supervisor(state: TopState):
    q = last_user(state).lower()
    if state.get('notes'):
        return Command(goto='final', update={})
    if any(op in q for op in ['+', '-', '*', '/', 'calc', 'solve']):
        return Command(goto='analysis_team', update={})
    return Command(goto='research_team', update={})


def final_node(state: TopState):
    return Command(goto=END, update={'messages': [AIMessage(content=state.get('notes', 'No notes'))]})


b = StateGraph(TopState)
b.add_node('supervisor', supervisor)
b.add_node('research_team', research_team)
b.add_node('analysis_team', analysis_team)
b.add_node('final', final_node)
b.add_edge(START, 'supervisor')
b.add_edge('research_team', 'final')
b.add_edge('analysis_team', 'final')
graph_hierarchical = b.compile()

# Demo usage (uncomment to run)
if __name__ == '__main__':
    out = graph_hierarchical.invoke({'messages': [HumanMessage('Who created Python?')]})
    for m in out['messages']:
        m.pretty_print()
    out = graph_hierarchical.invoke({'messages': [HumanMessage('Quickly estimate 20*(5-2).')]})
    for m in out['messages']:
        m.pretty_print()
