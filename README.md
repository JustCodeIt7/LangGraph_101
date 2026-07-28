# LangGraph 101: A Comprehensive Tutorial Series

Welcome to **LangGraph 101**, a hands-on tutorial series designed to help you learn and master [LangGraph](https://github.com/langchain-ai/langgraph), a powerful library for building stateful, multi-actor applications with Large Language Models (LLMs).

This repository is an educational sandbox — examples are script/notebook oriented, self-contained per concept folder, and prioritize readability over abstraction. There is no monolithic package; each directory teaches one idea end-to-end.

---

## 📚 About This Repository

A structured learning path from basic concepts to advanced applications:

| Section | Focus |
|---|---|
| `01-Graphs/` | Core graph primitives — state, routing, loops, chat graphs |
| `02-Agents/` | Agent patterns — tools, streaming, memory, persistence, MCP, multi-agent |
| `03-Agent_Design_Patterns/` | RAG agents, orchestration, parallel/multi-agent routers |
| `04-Apps/` | End-to-end applied agents (finance, travel, support, legal, research) |
| `research_agent/` | Importable Python module-style agent example |
| `DeepResearch_Examples/` | Advanced deep-research agent implementations |
| `Docling_File_Processer/` | Document processing with DocLing |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.12+** (configured in `langgraph.json`)
- **[Ollama](https://ollama.ai/)** for running local models (`llama3.2`, `qwen3`, `deepseek-r1`, etc.)
- Basic understanding of LLMs and Python
- API keys as needed: OpenAI, Anthropic, Tavily (for web search), LangSmith

### Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/LangGraph_101.git
cd LangGraph_101

# Install dependencies (pip)
pip install -r requirements.txt

# Or use uv for faster package management
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration
```

### Quick Start: Run Your First Graph

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class WorkflowState(TypedDict):
    user_input: str
    steps: list[str]

def start_node(state: WorkflowState) -> dict:
    return {"steps": ["started"]}

builder = StateGraph(WorkflowState)
builder.add_node("start", start_node)
builder.add_edge(START, "start")
builder.add_edge("start", END)

app = builder.compile()
result = app.invoke({"user_input": "hello", "steps": []})
print(result["steps"])  # ['started']
```

---

## 📖 Repository Structure

### `01-Graphs/` — Core Graph Primitives

| Directory | Topic | Key Files |
|---|---|---|
| `03-Graph_Basics` | Fundamental graph structures and state management | `03-Graph_Basics.ipynb`, `03-Graph_Basics.py` |
| `04-Multiple_Inputs` | Working with various input types within graphs | `04-Multiple_Inputs.ipynb` |
| `05-Conditional_Routing` | Decision logic and conditional edges | `05-Conditional_Routing-dev.py`, `.ipynb` |
| `06-Looping_Logic` | Loops, recursion, and iterative patterns | `06-Looping_Logic-dev.py`, `.ipynb` |
| `07-Basic_Chat_Graph` | Complete chat application with LLM integration | `07-chat_app.py` |

### `02-Agents/` — Agent Patterns & Implementations

| Directory | Topic | Key Files |
|---|---|---|
| `08-Running_Agents` | Basic agent setup and execution | `01-running_agents-*.py` |
| `09-Agent-State` | Managing complex agent state (basic, task, complex) | `09.1-basic_agent_state.py`, etc. |
| `09-Persistence` | Checkpointing, replaying state, memory stores | `E01_basic_checkpointing.py`, etc. |
| `10-Streaming` | Agent progress, LLM tokens, custom updates streaming | `01-agent_progress_streaming.py`, etc. |
| `10-Tools` | Tool integration patterns and examples | `10_tools.py`, `utils.py` |
| `12-Tools-v2` | Advanced tool usage, error handling | `01_basic_tools.py`, `03_error_handling.py` |
| `13-MCP` | Model Context Protocol tools & custom servers | `mcp_example.py`, `math_server.py`, `weather_server.py` |
| `14-Chainlit_Stock_Agent` | Stock analysis agent with Chainlit UI | `14_app.py` |
| `15-Human_in_the_Loop` | Interrupts, commands, HITL workflows | `16-Interrupts_Command.py`, `.ipynb` |
| `15-Memory` | Agent memory implementations and patterns | `15-memory*.py` |
| `17-Subgraphs` | Subgraph functionality for modular design | `18-subgraphs.py`, `basic_subgraph_example.py` |
| `18-Subgraph_ToDo_App` | Advanced subgraph todo application | `18-subgraph_advanced_todo.py` |
| `19-Multi_Agent` | Multi-agent coordination, supervisor patterns | `19-ma-v3.py`, `multi_agent_prebuilt.ipynb` |

### `03-Agent_Design_Patterns/` — Architectural Patterns

| Directory | Topic | Key Files |
|---|---|---|
| `20-Multi_Agent_Router` | Multi-agent routing and dispatch | `20-routing_agents.ipynb` |
| `21-Multi-Agent_Parallel` | Parallel agent execution patterns | `20-parallel_agents.py`, `.ipynb` |
| `22-Orchestration_Agent` | Report generation orchestration | `22-orchestrate_report_generator.py` |
| `23-Evaluator-Optimizer` | Evaluator-optimizer workflow pattern | `23-Eval_optimizer.py` |
| `24-Custom_RAG` | Custom RAG implementation | `rag_agent.py`, `app.py` |
| `25-RAG_Agent` | Retrieval-Augmented Generation agent | `rag_agent.py`, `example_usage.py` |
| `26-Advanced-Agent-Examples-v2/21-Research_Agent` | Research agent with deep capabilities | `research_agent.py` |
| `27-SQL_Agent` | SQL-based agent for database queries | `sql_agent.py` |

### `04-Apps/` — End-to-End Applications

| Directory | Application | Key Files |
|---|---|---|
| `01-LangGraph_Stock_Agent` | Financial data analysis & stock insights | `main.py`, `stock_app_chat.py` |
| `02-Finance_Investment_Agent` | Investment advisory system | `finance_investment_agent.py` |
| `03-Customer_Support_Agent` | Customer service automation | `customer_support_agent.py` |
| `04-Crawl4AI_Chatbot` | Web crawling chatbot with Crawl4AI | `app.py`, `app-v3.py` |
| `06-Developer_Assistant_Agent` | Code assistance and review | `app.py` |
| `07-Legal_Document_Analyzer_Agent` | Legal document processing & analysis | `legal_analyzer.py`, `doc_analyzer.py` |
| `08-Content_Creation_Pipeline_Agent` | Content generation workflow pipeline | `app.py` |
| `09-Research_Summarization_Agent` | Academic research assistant | `research_summarization_agent.py` |
| `12-DeepResearch_Agent` | Advanced deep-research agent with Chainlit UI | `chainlit_app.py`, `research_agent-v2.py` |
| `14-Invoice_Parser` | Invoice parsing and data extraction | `demo_ollama.py`, `.ipynb` |

### Other Key Directories

| Directory | Description |
|---|---|
| `research_agent/` | Importable Python module-style agent (`agent.py`) with `__init__.py` |
| `DeepResearch_Examples/` | Advanced deep-research implementations (DR_Agent, 01_DeepResearch) |
| `Docling_File_Processer/` | Document processing pipeline using DocLing |
| `04-LangGraph_Studio/01-Studio_Basics` | LangGraph Studio setup and development (`src/agent.py`) |

---

## 💡 Quick Start Examples

### Basic Graph Structure

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class WorkflowState(TypedDict):
    user_input: str
    steps: list[str]

def start_node(state: WorkflowState) -> dict:
    return {"steps": ["started"]}

builder = StateGraph(WorkflowState)
builder.add_node("start", start_node)
builder.add_edge(START, "start")
builder.add_edge("start", END)

app = builder.compile()
```

### Chat Application with LLM (Ollama)

See the complete example in [`01-Graphs/07-Basic_Chat_Graph/`](01-Graphs/07-Basic_Chat_Graph/).

```python
from langchain_openai import ChatOpenAI
# or for local Ollama:
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="qwen3:0.6b",
    base_url="http://localhost:11434",
    temperature=0.7,
)
```

### Agent with Tools (MCP Example)

See [`02-Agents/13-MCP/`](02-Agents/13-MCP/) for a complete MCP integration example including custom tool servers (`math_server.py`, `weather_server.py`).

---

## 🔧 Supported LLM Providers

This repository demonstrates integration with various providers:

| Provider | Usage | Example Models |
|---|---|---|
| **Ollama** (local) | Local model inference via Ollama server | `llama3.2`, `qwen3`, `deepseek-r1` |
| **OpenAI** | Cloud API access to GPT models | `gpt-4o-mini`, `gpt-4` |
| **Anthropic** | Claude model integration | `claude-3-sonnet`, etc. |
| **LiteLLM** | Unified interface for multiple providers | Any LiteLLM-supported provider |

---

## 📋 Dependencies

Key dependencies (see [`requirements.txt`](requirements.txt) for the full list):

```text
langgraph          # Core graph framework
langchain-core     # LangChain foundation
langchain-openai   # OpenAI integration
langchain-ollama   # Ollama local model support
litellm            # Multi-provider LLM interface
rich               # Enhanced terminal output
yfinance           # Financial data for stock agents
python-dotenv      # Environment variable management
```

---

## 🎯 Learning Path

We recommend this progression:

1. **Start with Graphs** — Work through [`01-Graphs/`](01-Graphs/) to master state, routing, and loops
2. **Build Agents** — Progress to [`02-Agents/`](02-Agents/) for tools, streaming, memory, and MCP integration
3. **Learn Patterns** — Study [`03-Agent_Design_Patterns/`](03-Agent_Design_Patterns/) for RAG, orchestration, and multi-agent architectures
4. **Apply Knowledge** — Build real apps using the examples in [`04-Apps/`](04-Apps/)

---

## 🛠️ Development Setup

```bash
# Install dependencies (preferred)
uv sync

# Or with pip
pip install -r requirements.txt

# Format code
make format

# Run linters (ruff + mypy)
make lint

# Run tests (if available in a subdirectory)
python -m pytest path/to/tests/

# Use with Jupyter notebooks
jupyter notebook 01-Graphs/03-Graph_Basics/03-Graph_Basics.ipynb
```

### Project Configuration Files

| File | Purpose |
|---|---|
| `requirements.txt` | Python dependencies (pip) |
| `pyproject.toml` | Build system and project metadata |
| `uv.lock` | Locked dependency versions for uv |
| `ruff.toml` | Linting rules (line-length 120, single quotes) |
| `langgraph.json` | LangGraph Studio configuration |
| `.env.example` | Template for environment variables |

---

## 📊 Visualization

Many examples include graph visualizations (`.png`, `.svg`, `.mermaid`) in their directories. These help understand the flow and structure of each LangGraph application. Look for files like `graph.png`, `output.png`, or `*.md` documentation alongside code.

---

## 🤝 Contributing

Contributions are welcome! Please submit pull requests or open issues for:

- New example applications
- Bug fixes and improvements
- Documentation enhancements
- Additional LLM provider integrations

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---