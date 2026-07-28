"""
LangChain + LangGraph Deep Research Agent (Local)

A tutorial deep-research agent that runs entirely on-device:
  1. Breaks down a complex query into sub-questions (planning)
  2. Searches the web for each sub-question using DuckDuckGo
  3. Analyzes and synthesizes findings into a comprehensive report

Usage:
    python deep_research_agent_local.py "Your research question here" [--model llama3.2]

Requirements (local):
    pip install langchain-ollama langgraph langchain-community rich python-dotenv
    - Ollama running locally with a model pulled (e.g., `ollama pull llama3.2`)

Learning objectives:
  - Build a multi-step research workflow with LangGraph create_react_agent using local LLM
  - Use DuckDuckGoSearchRun as a web search tool
  - Structure agent tools for planning, searching, and synthesis without cloud APIs
"""

import argparse
from typing import List

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()


# ── 1. Define tools for the research agent ───────────────────────────────────

@tool
def search_web(query: str) -> List[str]:
    """Search the web and return top result snippets."""
    console.print(f"[cyan]Searching[/cyan] for: {query}")
    results = DuckDuckGoSearchRun().invoke({"input": query})
    # Split into individual result lines for cleaner processing
    return [r.strip() for r in results.split("\n") if r.strip()]


@tool
def summarize_findings(findings: str) -> str:
    """Synthesize research findings into a structured summary."""
    console.print("[cyan]Synthesizing[/cyan] final report ...")
    # The LLM will call this tool with accumulated findings; we return as-is
    # and let the agent format it into the final response.
    return findings


# ── 2. Build the LangGraph ReAct research agent (local Ollama) ───────────────

def build_agent(model_name: str = "ollama:llama3.2"):
    """Create a LangGraph ReAct agent for deep research using a local Ollama model."""
    from langchain_ollama import ChatOllama

    # Strip the "ollama:" prefix if present (ChatOllama expects bare model name)
    clean_name = model_name.removeprefix("ollama:")
    llm = ChatOllama(model=clean_name, temperature=0.3)
    tools = [search_web, summarize_findings]
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=(
            "You are a deep research agent. For any complex query:\n"
            "1. Break it into 2-3 specific sub-questions.\n"
            "2. Use search_web to find information for each sub-question.\n"
            "3. Analyze the results and identify key insights.\n"
            "4. Call summarize_findings with all findings, then write a final report."
        ),
    )


# ── 3. Run the research agent ───────────────────────────────────────────────

def run_research(query: str, model_name: str = "ollama:llama3.2") -> None:
    """Run the deep research agent on a query and print results."""
    console.print(Panel.fit(f"[bold blue]Research Query:[/bold blue]\n{query}", border_style="blue"))

    agent = build_agent(model_name)
    result = agent.invoke({"messages": [("user", query)]})

    # Extract the final response from the last message
    final_msg = result["messages"][-1].content
    console.print(Panel.fit(final_msg, title="[bold green]Research Report[/bold green]", border_style="green"))


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain + LangGraph Deep Research Agent (Local)")
    parser.add_argument("query", nargs="+", help="Research question to investigate")
    parser.add_argument(
        "--model", default="ollama:llama3.2",
        help="Ollama model name (default: ollama:llama3.2)",
    )
    args = parser.parse_args()

    query = " ".join(args.query)
    run_research(query, model_name=args.model)


if __name__ == "__main__":
    main()
