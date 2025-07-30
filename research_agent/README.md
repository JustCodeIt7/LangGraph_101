# Deep Research Agent

This directory contains a minimal example of a deep research agent built with [LangGraph](https://github.com/langchain-ai/langgraph).

The agent performs iterative web searches using the Tavily API, summarizes the gathered information with an LLM, and produces a concise research report.

## Files

- `agent.py` – implementation of the research workflow.

## Usage

```python
from research_agent.agent import run

report = run("impact of renewable energy")
print(report)
```

Set the `TAVILY_API_KEY` environment variable and ensure an Ollama server is running (default `http://localhost:11434`).
