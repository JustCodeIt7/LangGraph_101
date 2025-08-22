#!/usr/bin/env python3
"""
Deep Research Agent using LangGraph, LangChain, Ollama, and Brave Search
A simple implementation for tutorial purposes - under 400 lines
"""

import asyncio
import json
import os
import requests
from typing import Dict, List, TypedDict, Annotated
from dataclasses import dataclass

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    research_query: str
    search_results: List[Dict]
    analysis: str
    final_report: str


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


class BraveSearchTool:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
    
    def search(self, query: str, count: int = 5) -> List[SearchResult]:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }
        
        params = {
            "q": query,
            "count": count,
            "search_lang": "en",
            "country": "US",
            "safesearch": "moderate",
            "freshness": "pd"
        }
        
        try:
            response = requests.get(self.base_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for result in data.get("web", {}).get("results", []):
                results.append(SearchResult(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    snippet=result.get("description", "")
                ))
            
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []


class DeepResearchAgent:
    def __init__(self, brave_api_key: str, model_name: str = "llama3.2"):
        self.brave_search = BraveSearchTool(brave_api_key)
        self.llm = ChatOllama(model=model_name, temperature=0.1)
        self.graph = self._build_graph()
    
    @tool
    def web_search(self, query: str) -> str:
        """Search the web for information using Brave Search API"""
        results = self.brave_search.search(query)
        if not results:
            return "No search results found."
        
        formatted_results = []
        for i, result in enumerate(results[:5], 1):
            formatted_results.append(
                f"{i}. **{result.title}**\n"
                f"   URL: {result.url}\n"
                f"   Summary: {result.snippet}\n"
            )
        
        return "\n".join(formatted_results)
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        workflow.add_node("planner", self._planning_node)
        workflow.add_node("researcher", self._research_node)
        workflow.add_node("analyzer", self._analysis_node)
        workflow.add_node("synthesizer", self._synthesis_node)
        
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "researcher")
        workflow.add_edge("researcher", "analyzer")
        workflow.add_edge("analyzer", "synthesizer")
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile()
    
    def _planning_node(self, state: AgentState) -> AgentState:
        messages = [
            SystemMessage(content="""You are a research planner. Given a user query, break it down into 
            specific, searchable research questions. Return 3-5 focused search queries that will help 
            gather comprehensive information on the topic. Format as a numbered list."""),
            HumanMessage(content=f"Research topic: {state['research_query']}")
        ]
        
        response = self.llm.invoke(messages)
        
        state["messages"].extend([
            HumanMessage(content=f"Planning research for: {state['research_query']}"),
            AIMessage(content=f"Research plan:\n{response.content}")
        ])
        
        return state
    
    def _research_node(self, state: AgentState) -> AgentState:
        search_queries = self._extract_search_queries(state["messages"][-1].content)
        all_results = []
        
        for query in search_queries:
            results = self.brave_search.search(query.strip())
            all_results.extend(results)
        
        state["search_results"] = [
            {
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet
            } for r in all_results
        ]
        
        formatted_results = self._format_search_results(all_results)
        
        state["messages"].append(
            AIMessage(content=f"Gathered research data:\n{formatted_results}")
        )
        
        return state
    
    def _analysis_node(self, state: AgentState) -> AgentState:
        search_content = self._format_search_results([
            SearchResult(r["title"], r["url"], r["snippet"]) 
            for r in state["search_results"]
        ])
        
        messages = [
            SystemMessage(content="""You are a research analyst. Analyze the provided search results 
            and identify key themes, important facts, different perspectives, and potential gaps in 
            information. Provide a structured analysis."""),
            HumanMessage(content=f"Original query: {state['research_query']}\n\nSearch results:\n{search_content}")
        ]
        
        response = self.llm.invoke(messages)
        state["analysis"] = response.content
        
        state["messages"].append(
            AIMessage(content=f"Analysis completed:\n{response.content}")
        )
        
        return state
    
    def _synthesis_node(self, state: AgentState) -> AgentState:
        messages = [
            SystemMessage(content="""You are a research synthesizer. Create a comprehensive, 
            well-structured final report based on the research query, search results, and analysis. 
            Include an executive summary, key findings, different perspectives, and conclusions. 
            Make it informative and easy to read."""),
            HumanMessage(content=f"""
            Original query: {state['research_query']}
            
            Analysis: {state['analysis']}
            
            Search results: {self._format_search_results([
                SearchResult(r["title"], r["url"], r["snippet"]) 
                for r in state["search_results"]
            ])}
            """)
        ]
        
        response = self.llm.invoke(messages)
        state["final_report"] = response.content
        
        state["messages"].append(
            AIMessage(content=f"Final research report:\n{response.content}")
        )
        
        return state
    
    def _extract_search_queries(self, plan_content: str) -> List[str]:
        lines = plan_content.split('\n')
        queries = []
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                clean_query = line.lstrip('0123456789.-• ').strip()
                if clean_query:
                    queries.append(clean_query)
        return queries[:5]
    
    def _format_search_results(self, results: List[SearchResult]) -> str:
        if not results:
            return "No results found."
        
        formatted = []
        for i, result in enumerate(results[:10], 1):
            formatted.append(
                f"{i}. **{result.title}**\n"
                f"   Source: {result.url}\n"
                f"   Summary: {result.snippet}\n"
            )
        
        return "\n".join(formatted)
    
    async def research(self, query: str) -> str:
        initial_state = AgentState(
            messages=[],
            research_query=query,
            search_results=[],
            analysis="",
            final_report=""
        )
        
        print(f"🔍 Starting deep research on: {query}")
        print("=" * 60)
        
        final_state = await self.graph.ainvoke(initial_state)
        
        return final_state["final_report"]


async def main():
    print("🤖 Deep Research Agent")
    print("=" * 50)
    
    brave_api_key = os.getenv("BRAVE_API_KEY")
    if not brave_api_key:
        print("❌ Error: BRAVE_API_KEY environment variable not set")
        print("Please get your API key from: https://brave.com/search/api/")
        return
    
    try:
        agent = DeepResearchAgent(brave_api_key)
        print("✅ Agent initialized successfully")
        print("\nℹ️  Make sure Ollama is running with llama3.2 model")
        print("   Run: ollama pull llama3.2")
        print()
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        return
    
    while True:
        try:
            query = input("\n🔍 Enter your research topic (or 'quit' to exit): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                print("Please enter a valid research topic.")
                continue
            
            print(f"\n🚀 Researching: {query}")
            print("-" * 50)
            
            result = await agent.research(query)
            
            print("\n📊 RESEARCH COMPLETE")
            print("=" * 60)
            print(result)
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Research interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error during research: {e}")
            continue


if __name__ == "__main__":
    asyncio.run(main())