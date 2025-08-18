# Deep Research Agent with LangGraph

This project implements a deep research agent using LangGraph that can conduct research on any topic provided by the user through a chat interface.

## Features

- Chat-based interface for researching topics
- Multi-step research process using LangGraph
- Web search simulation (can be connected to real APIs)
- Research summarization with LLM-powered analysis
- State management through graph nodes
- Topic refinement using LLMs

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key in the `.env` file:
```bash
OPENAI_API_KEY=your-actual-api-key
```

## Usage

Run the basic research agent:
```bash
python research_agent.py
```

Or run the enhanced research agent with LLM-powered summarization:
```bash
python research_agent_enhanced.py
```

Then simply type any research topic when prompted, and the agent will:
1. Initiate research on your topic (and refine it if needed)
2. Conduct web searches and gather information
3. Summarize the findings with comprehensive analysis

Type 'quit' to exit the chat interface.

## How It Works

The agent uses LangGraph to manage a research workflow with the following steps:

1. **Initiate**: Identifies and refines the research topic from user input
2. **Research**: Conducts web searches and gathers detailed information
3. **Summarize**: Creates a comprehensive summary of the research findings using LLM analysis

Each step is implemented as a node in the graph, with the state being passed between nodes.

## Customization

To connect to real APIs:
1. Replace the dummy `search_web` tool with actual API calls (e.g., SerpAPI, Google Custom Search)
2. Replace the `get_article_content` tool with web scraping functionality
3. Add more sophisticated research tools for different data sources