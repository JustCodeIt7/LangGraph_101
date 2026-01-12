# AGENTS.md

This guide is for agentic coding assistants working in the LangGraph_101 repository.

## Build/Lint/Test Commands

```bash
# Format code
make format

# Run linters (check formatting, imports, type checking)
make lint

# Run all unit tests
make test
# or: python -m pytest tests/unit_tests

# Run a single test file
make test TEST_FILE=path/to/test_file.py
# or: python -m pytest path/to/test_file.py

# Run a single test function
python -m pytest path/to/test_file.py::test_function_name

# Run integration tests
make integration_tests

# Run tests in watch mode
make test_watch
```

## Code Style Guidelines

### Imports
- Order: standard library → third-party → local
- No unused imports (enforced by ruff)
- Import specific items, not modules: `from typing import TypedDict, List`
- Use `from dotenv import load_dotenv` for environment variables

### Formatting
- Line length: 120 characters
- Quote style: single quotes
- Format with `ruff format` before committing
- Run `ruff check --select I` for import sorting

### Type Hints
- Always type function parameters and returns
- Use `TypedDict` for graph states
- Use `Annotated` for state reducers: `messages: Annotated[list, add_messages]`
- Prefer concrete types over `Any` when possible

```python
def example_function(state: MyState, config: RunnableConfig) -> MyState:
    return state
```

### Naming Conventions
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Graph states: `TypedDict` with descriptive field names

### Error Handling
- Use specific exceptions: `GraphRecursionError`, `requests.RequestException`
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
- Use `StateGraph(state_schema)` for graphs
- Define states as `TypedDict` with `Annotated[list, add_messages]` for messages
- Use `create_react_agent` for simple agents
- Node functions accept state and return state (or dict to merge)
- Use `add_conditional_edges` for routing logic

### LangGraph Patterns
- Initialize models: `ChatOpenAI(model="gpt-4o-mini", temperature=0.1)`
- Tools: Use `@tool` decorator with descriptive docstrings
- State management: Use checkpointers for persistence
- Streaming: Use `stream()` with modes: 'updates', 'messages', 'custom'
- Recursion limit: Configure via `with_config(recursion_limit=N)`

### Comments
- Docstrings for all functions and classes (Google style)
- Comments for complex logic only
- No inline comments for obvious code
- Keep docstrings concise

### Testing
- Use pytest for tests
- Test files: `test_*.py` or `*_test.py`
- Async tests: `async def test_*()` with `asyncio.run()`
- Mock external services (LLMs, APIs) in unit tests

### Environment
- Load environment variables with `load_dotenv()`
- Use `.env` for local config
- Never commit `.env` files
- Reference env vars: `os.getenv("API_KEY")`

### Dependencies
- Core: `langgraph>=0.4.7`, `langchain-core~=0.3.59`
- LLM providers: `langchain-openai`, `langchain-ollama`, `langchain_litellm`
- Utilities: `rich` for terminal output, `pydantic` for schemas
