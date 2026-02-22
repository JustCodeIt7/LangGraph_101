# Deep Research Agent

A minimal example of a deep research agent built with [LangGraph](https://github.com/langchain-ai/langgraph).

## Overview

This research agent performs iterative web searches using the Tavily API, summarizes gathered information with an LLM, and produces concise research reports.

## Features

- **Iterative Research**: Conducts multiple search iterations to gather comprehensive information
- **LLM-Powered Summarization**: Uses large language models to analyze and summarize findings
- **Topic Refinement**: Automatically refines research topics for better results
- **State Management**: LangGraph manages the workflow state across all nodes

## Files

- `agent.py` – Implementation of the research workflow with graph-based execution

## Architecture

The agent uses a StateGraph with the following key components:

1. **State Definition** - TypedDict containing:
   - `topic`: The research topic
   - `search_results`: List of gathered information
   - `summary`: Final synthesized report

2. **Nodes**:
   - `initiate_node` - Refines and validates the research topic
   - `research_node` - Performs web searches using Tavily API
   - `summarize_node` - Uses LLM to analyze and summarize findings

3. **Edges** - Defines flow control between nodes based on completion criteria

## Usage

```python
from research_agent.agent import run

# Run the agent with a topic
report = run("impact of renewable energy")
print(report)
```

## Environment Variables

- `TAVILY_API_KEY` - API key for Tavily search service
- `OLLAMA_BASE_URL` - Ollama server URL (default: `http://localhost:11434`)
- `MODEL_NAME` - LLM model to use (default: `llama3.2`)

## Dependencies

Install required packages:
```bash
pip install langgraph langchain langchain-community tavily-python
```

## How It Works

1. User provides a research topic
2. Agent refines the topic using LLM judgment
3. Performs web searches via Tavily API
4. Gathers and analyzes results with LLM
5. Produces final synthesized report

The graph ensures each step builds on previous results, allowing for iterative refinement of the research.