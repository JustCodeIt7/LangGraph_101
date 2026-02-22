# Deep Research Examples

This module contains various implementations of deep research agents using LangGraph, demonstrating different approaches to automated research and information synthesis.

## Overview

The examples showcase multiple ways to build research agents with:
- Web search integration (Brave Search API)
- Local LLM processing (Ollama)
- Multi-step research workflows
- Different LLM providers (OpenAI, Anthropic Claude, Grok)

## Projects

### 01_DeepResearch/

Basic implementation of a deep research agent.

**Files:**
- `research_agent.py` - Core agent implementation
- `research_agent_enhanced.py` - Enhanced version with additional features
- `requirements.txt` - Python dependencies

**Features:**
- Multi-step research workflow using LangGraph
- Web search integration
- LLM-powered analysis and synthesis

### DR_Agent,py/

Advanced deep research agent implementations.

**Files:**
- `dr_agent.py` - Main implementation (uses Ollama + Brave Search)
- `dr_agent-grok.py` - Version using Grok as the LLM provider
- `README.md` - Project documentation

**Features:**
- Async workflow with LangGraph
- Modular design for easy understanding
- Clear separation of concerns
- Under 400 lines total per file

## Architecture

Each research agent follows a similar pattern:

1. **Planning Node**: Breaks down the query into specific research questions
2. **Research Node**: Searches the web using search APIs (Brave/Tavily)
3. **Analysis Node**: Analyzes search results for key themes and insights  
4. **Synthesis Node**: Creates a comprehensive final report

## Usage

```bash
# Run basic research agent
cd DeepResearch_Examples/01_DeepResearch
python research_agent.py

# Run enhanced version
python research_agent_enhanced.py

# Run DR Agent with Ollama
cd ../DR_Agent,py
python dr_agent.py

# Run DR Agent with Grok
python dr_agent-grok.py
```

## Environment Variables

Required for different implementations:

- `BRAVE_API_KEY` - For Brave Search API (dr_agent.py)
- `OPENAI_API_KEY` - For OpenAI models
- `ANTHROPIC_API_KEY` - For Claude models  
- `GROK_API_KEY` - For Grok models
- `TAVILY_API_KEY` - For Tavily search

## Dependencies

Install all required packages:
```bash
pip install -r requirements.txt
```

Core dependencies across examples:
- langgraph
- langchain
- langchain-community
- brave-search
- tavily-python