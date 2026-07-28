# AGENTS.md

This guide is for agentic coding assistants working in the LangGraph_101 repository. This is an educational sandbox — examples are script/notebook oriented, self-contained per concept folder, and prioritize readability over abstraction. There is no monolithic package; each directory teaches one idea end-to-end.

## Project Structure Overview

```
LangGraph_101/
├── 01-Graphs/                    # Core graph primitives (state, routing, loops)
│   ├── 03-Graph_Basics/          # Fundamental graph structures & state management
│   ├── 04-Multiple_Inputs/       # Working with various input types
│   ├── 05-Conditional_Routing/   # Decision logic and conditional edges
│   ├── 06-Looping_Logic/         # Loops, recursion, iterative patterns
│   └── 07-Basic_Chat_Graph/      # Complete chat application with LLM
├── 02-Agents/                    # Agent patterns & implementations
│   ├── 08-Running_Agents/        # Basic agent setup and execution
│   ├── 09-Agent-State/           # Complex state management (basic, task, complex)
│   ├── 09-Persistence/           # Checkpointing, replaying state, memory stores
│   ├── 10-Streaming/             # Agent progress, LLM tokens, custom updates
│   ├── 10-Tools/ & 12-Tools-v2/  # Tool integration (basic and advanced)
│   ├── 13-MCP/                   # Model Context Protocol tools & servers
│   ├── 14-Chainlit_Stock_Agent/  # Stock analysis agent with Chainlit UI
│   ├── 15-Human_in_the_Loop/     # Interrupts, commands, HITL workflows
│   ├── 15-Memory/                # Agent memory implementations
│   ├── 17-Subgraphs/             # Subgraph functionality for modular design
│   └── 19-Multi_Agent/           # Multi-agent coordination & supervisor patterns
├── 03-Agent_Design_Patterns/     # Architectural patterns (RAG, orchestration)
│   ├── 20-Multi_Agent_Router/    # Multi-agent routing and dispatch
│   ├── 21-Multi-Agent_Parallel/  # Parallel agent execution
│   ├── 22-Orchestration_Agent/   # Report generation orchestration
│   ├── 23-Evaluator-Optimizer/   # Evaluator-optimizer workflow pattern
│   ├── 24-Custom_RAG/            # Custom RAG implementation
│   ├── 25-RAG_Agent/             # Retrieval-Augmented Generation agent
│   └── 27-SQL_Agent/             # SQL-based database query agent
├── 04-Apps/                      # End-to-end applied agents (finance, travel, etc.)
├── research_agent/               # Importable Python module-style agent
├── DeepResearch_Examples/        # Advanced deep-research implementations
└── Docling_File_Processer/       # Document processing with DocLing
```

## Python Environment

This project uses the conda env **`py312`** (Python 3.12). Activate before running anything:

```bash
conda activate py312
```

Or invoke the interpreter directly without activation:

```bash
/Users/james/miniconda3/envs/py312/bin/python <script.py>
```

All `python`, `pip`, `make`, and `pytest` commands below assume this env is active.

## Build/Lint/Test Commands

### Formatting & Linting

```bash
# Format code (ruff format + import sorting)
make format

# Run linters: ruff check, ruff format --diff, mypy --strict
make lint

# Spell check
make spell_check

# Fix spelling errors in place
make spell_fix
```

### Testing

**Note:** There is no top-level `tests/` directory. Tests exist only within specific subdirectories (e.g., `02-Agents/09-Agent-State/20-autoread/tests/`, `archive/01-LangGraph_Basics/tests/`). The Makefile targets reference paths that may not exist at the repo root — run tests from their respective directories:

```bash
# Run a specific test file (from its directory)
python -m pytest path/to/test_file.py

# Run a single test function
python -m pytest path/to/test_file.py::test_function_name

# Watch mode for development (requires pytest-watch)
make test_watch TEST_FILE=path/to/tests/
```

### Running Examples

Most examples are standalone scripts or notebooks. Run them directly:

```bash
# Run a Python script example
python 01-Graphs/07-Basic_Chat_Graph/07-chat_app.py

# Open a notebook in Jupyter
jupyter notebook 02-Agents/13-MCP/mcp_example_workflow.ipynb

# Run the importable research agent module
python -m research_agent.agent
```

## Project Preferences

### Agent Type & LLM Providers

- **Primary framework:** LangGraph for graph/agent development, LangChain for LLM integration
- **Local models (preferred):** Ollama — `ChatOllama`, `OllamaEmbeddings`
- **Cloud providers:** OpenAI (`ChatOpenAI`), Anthropic (`ChatAnthropic`)

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

################# Model Initialization ################

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
MODEL_NAME = 'llama3.2'

# Initialize the LLM client with local Ollama settings
model = ChatOllama(
    model=MODEL_NAME,
    temperature=0.2,  # Lower temperature for more consistent evaluations
    base_url=OLLAMA_BASE_URL,
)

embeddings = OllamaEmbeddings(
    model='nomic-embed-text:latest',
    base_url=OLLAMA_BASE_URL,
)
```

### Environment Variables

Load environment variables with `load_dotenv()` at the top of scripts. Use `.env` for local config — never commit it. Reference env vars via `os.getenv("API_KEY")`.

Key environment variables (see [`.env.example`](.env.example)):

| Variable | Purpose |
|---|---|
| `LANGSMITH_PROJECT` | LangSmith tracing project name |
| `OPENAI_API_KEY` | OpenAI API access |
| `ANTHROPIC_API_KEY` | Anthropic Claude API access |
| `TAVILY_API_KEY` | Web search for research agents |

## Code Style Guidelines

### Imports

- Order: standard library → third-party → local
- No unused imports (enforced by ruff, though F401 is currently ignored in `ruff.toml`)
- Import specific items, not modules: `from typing import TypedDict, List`
- Use `from dotenv import load_dotenv` for environment variables

### Formatting

- Line length: 120 characters (configured in `ruff.toml`)
- Quote style: single quotes (`quote-style = "single"` in ruff config)
- Format with `ruff format` before committing
- Run `ruff check --select I` for import sorting

### Type Hints

- Always type function parameters and returns
- Use `TypedDict` for graph states
- Use `Annotated` for state reducers: `messages: Annotated[list, add_messages]`
- Prefer concrete types over `Any` when possible

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph
from langchain_core.messages import add_messages

class MyState(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str

def example_function(state: MyState) -> dict:
    return {"user_input": state["user_input"]}
```

### Naming Conventions

- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Graph states: `TypedDict` with descriptive field names

### Error Handling

- Use specific exceptions (e.g., `GraphRecursionError`, `requests.RequestException`)
- Wrap API calls in try-except blocks
- Return error strings or add to state['errors'] list
- Don't let exceptions crash agents silently

```python
try:
    response = requests.get(url)
    response.raise_for_status()
except requests.RequestException as e:
    return f"Error: {str(e)}"
```

### Graph Patterns

- Use `StateGraph(state_schema)` for graphs with typed state
- Define states as `TypedDict` with `Annotated[list, add_messages]` for messages
- Node functions accept state and return a dict to merge into state (or the full state)
- Use `add_conditional_edges` for routing logic based on node output

### LangGraph Patterns

- Initialize models: `ChatOllama(model="qwen3", base_url=..., temperature=0.2)` or `ChatOpenAI(model="gpt-4o-mini")`
- Tools: Use `@tool` decorator with descriptive docstrings
- State management: Use checkpointers for persistence (`MemorySaver`, etc.)
- Streaming: Use `.stream()` with modes: `'updates'`, `'messages'`, `'custom'`
- Recursion limit: Configure via `with_config(recursion_limit=N)` or pass to `.invoke()`

### Comments & Documentation

- Docstrings for all functions and classes (Google style)
- Comments for complex logic only — no inline comments for obvious code
- Keep docstrings concise but informative
- Top-of-file docstring summarizing the learning objective when creating new examples

### Testing

- Use pytest for tests (`test_*.py` or `*_test.py`)
- Async tests: `async def test_*()` with `asyncio.run()`
- Mock external services (LLMs, APIs) in unit tests
- Tests are colocated within specific subdirectories — there is no top-level test suite

### Dependencies

Core dependencies (see [`requirements.txt`](requirements.txt)):

| Package | Purpose |
|---|---|
| `langgraph` | Core graph framework |
| `langchain-core` | LangChain foundation |
| `langchain-openai` | OpenAI integration |
| `langchain-ollama` | Ollama local model support |
| `litellm` | Multi-provider LLM interface |
| `rich` | Enhanced terminal output |
| `yfinance` | Financial data for stock agents |

### File Layout Conventions

When adding new examples:

1. Create a clearly numbered directory that fits the progression (e.g., `02-Agents/XX-Topic_Name/`)
2. Each concept folder is self-contained — code, optional output/visualization
3. Provide a top-of-file docstring summarizing the learning objective
4. Show minimal runnable path: define state → nodes → build graph → invoke once
5. If multi-agent: demonstrate at least two specializations + a router
6. If tool use: clearly isolate tool definition vs model invocation

### Visualization

Many examples include `.png`, `.svg`, or `.mermaid` files for graph visualization. These are generated outputs — keep them alongside the code they visualize.