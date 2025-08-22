# Deep Research Agent

A simple deep research agent built with LangGraph, LangChain, Ollama, and Brave Search API.

## Features

- 🔍 Multi-step research workflow using LangGraph
- 🌐 Web search integration with Brave Search API
- 🤖 Local LLM processing with Ollama
- 📊 Structured analysis and synthesis
- 💬 Simple CLI interface

## Setup

1. **Install Ollama and pull the model:**
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull llama3.2
   ```

2. **Get Brave Search API key:**
   - Go to https://brave.com/search/api/
   - Sign up and get your API key

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variable:**
   ```bash
   export BRAVE_API_KEY="your_api_key_here"
   ```

## Usage

```bash
python dr_agent.py
```

Then enter your research topic when prompted.

## How it Works

1. **Planning:** Breaks down your query into specific research questions
2. **Research:** Searches the web using Brave Search API
3. **Analysis:** Analyzes search results for key themes and insights
4. **Synthesis:** Creates a comprehensive final report

## Example Queries

- "Latest developments in AI safety research"
- "Climate change impact on agriculture in 2024"
- "Quantum computing breakthrough implications"
- "Remote work productivity trends"

## Code Structure

- Under 400 lines total
- Clear separation of concerns
- Async workflow with LangGraph
- Modular design for easy understanding