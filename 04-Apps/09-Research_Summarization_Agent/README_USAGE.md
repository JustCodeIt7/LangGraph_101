# Research & Summarization Agent - Usage Guide

## Overview

The Research & Summarization Agent performs research by scraping web pages and scholarly papers. It can synthesize information from multiple sources, include citations, and generate concise reports or executive summaries.

## Features

- Web scraping for information gathering
- Scholarly paper search and extraction
- Information synthesis with citations
- Multiple output formats (concise report, executive summary, bullet points)
- LangGraph for agent orchestration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LangGraph_101.git
cd LangGraph_101/03-Apps/09-Research_Summarization_Agent
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Ollama (for local LLM):
   - Install Ollama from [https://ollama.ai/](https://ollama.ai/)
   - Pull the llama3.2 model:
   ```bash
   ollama pull llama3.2
   ```

## Usage

### Running the Agent

To run the agent in interactive mode:

```bash
python research_summarization_agent.py
```

This will start a conversation loop where you can enter research requests.

### Example Requests

- "Research the impact of artificial intelligence on healthcare"
- "Find scholarly articles about climate change and provide an executive summary"
- "Give me bullet points about renewable energy from web sources"

### Specifying Research Scope and Output Format

You can specify the research scope and output format in your request:

- Research scope: "web", "scholarly", or "all" (default)
- Output format: "concise_report" (default), "executive_summary", or "bullet_points"

Examples:
- "Research quantum computing from scholarly sources and give me an executive summary"
- "Find information about renewable energy from web sources and provide bullet points"

### Testing the Agent

To run the test script:

```bash
python test_agent.py
```

This will run the agent with several test cases and display the results.

## Implementation Notes

The current implementation uses mock functions for web scraping and scholarly paper search. In a production environment, these would be replaced with actual implementations using:

- BeautifulSoup or Scrapy for web scraping
- Scholarly APIs like Semantic Scholar, CORE, or PubMed for academic papers

## Customization

### Adding New Output Formats

To add a new output format:

1. Add the format to the `OutputFormat` enum in `research_summarization_agent.py`
2. Update the `generate_report_tool` function to handle the new format

### Implementing Real Web Scraping

To implement real web scraping:

1. Replace the `mock_web_search` and `mock_web_scrape` functions with actual implementations
2. Update the `identify_web_sources_tool` and `scrape_web_pages_tool` functions accordingly

### Implementing Real Scholarly Paper Search

To implement real scholarly paper search:

1. Uncomment and install the relevant packages in `requirements.txt`
2. Replace the `mock_scholarly_search` and `mock_extract_paper_text` functions with actual implementations
3. Update the `search_scholarly_papers_tool` and `extract_text_from_papers_tool` functions accordingly

## Troubleshooting

- If you encounter issues with the LLM, ensure Ollama is running and the llama3.2 model is available
- For web scraping issues, check your internet connection and consider implementing rate limiting to avoid being blocked

## License

This project is licensed under the MIT License - see the LICENSE file for details.