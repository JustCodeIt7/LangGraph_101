# Deep Research Agent using LangGraph

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from tavily import TavilyClient


class ResearchState(TypedDict, total=False):
    """State for the research agent."""

    topic: str
    search_results: List[Dict[str, Any]]
    summary: str
    iteration: int
    next_query: Optional[str]
    report: str


TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')


@tool
def tavily_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Use Tavily to search the web."""
    if not TAVILY_API_KEY:
        raise ValueError('TAVILY_API_KEY not set')

    client = TavilyClient(api_key=TAVILY_API_KEY)
    response = client.search(query=query, max_results=max_results)
    return response.get('results', [])


llm = ChatOllama(model='qwen', base_url=OLLAMA_BASE_URL)


def start(state: ResearchState) -> ResearchState:
    return {**state, 'search_results': [], 'summary': '', 'iteration': 0}


def search(state: ResearchState) -> ResearchState:
    query = state.get('next_query') or state['topic']
    results = tavily_search.invoke({'query': query, 'max_results': 5})
    all_results = state.get('search_results', []) + results
    return {**state, 'search_results': all_results}


def summarize(state: ResearchState) -> ResearchState:
    context = '\n\n'.join(r.get('content', '') for r in state['search_results'])
    prompt = f"Summarize the following information about {state['topic']} and suggest one follow-up search query if needed. Return 'None' if not needed.\n\n{context}"
    resp = llm.invoke([HumanMessage(content=prompt)])
    text = resp.content.strip()
    follow_up = None
    if 'FOLLOWUP:' in text:
        parts = text.split('FOLLOWUP:', 1)
        text = parts[0].strip()
        follow_up = parts[1].strip() or None
    return {
        **state,
        'summary': text,
        'next_query': follow_up,
        'iteration': state.get('iteration', 0) + 1,
    }


def should_continue(state: ResearchState) -> str:
    if state.get('iteration', 0) >= 2 or not state.get('next_query'):
        return 'report'
    return 'search'


def generate_report(state: ResearchState) -> ResearchState:
    prompt = (
        f'Write a detailed research report about {state["topic"]} using the following summary:\n\n{state["summary"]}'
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    report = resp.content.strip()
    return {**state, 'report': report}


def create_graph() -> StateGraph:
    g = StateGraph(ResearchState)
    g.add_node('start', start)
    g.add_node('search', search)
    g.add_node('summarize', summarize)
    g.add_node('report', generate_report)

    g.set_entry_point('start')
    g.add_edge('start', 'search')
    g.add_edge('search', 'summarize')
    g.add_conditional_edges('summarize', should_continue, {'search': 'search', 'report': 'report'})
    g.add_edge('report', END)
    return g


def run(topic: str) -> str:
    graph = create_graph().compile()
    state = {'topic': topic}
    final = graph.invoke(state)
    return final.get('report', '')


__all__ = ['run', 'create_graph', 'ResearchState']
