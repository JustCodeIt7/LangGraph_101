# LangGraph Tools Tutorial

This tutorial series covers comprehensive usage of tools in LangGraph, from basic concepts to advanced integration patterns.

## Files Overview

### 01_basic_tools.py

- Simple function tools
- Using the @tool decorator
- Custom Pydantic schemas
- Data processing tools

### 02_advanced_tool_features.py

- Hiding arguments from the model using state and config
- Disabling parallel tool calling
- Return direct functionality
- Force tool use with safety measures

### 03_error_handling.py

- Default error handling (enabled)
- Disabled error handling
- Custom error messages and handlers
- Error recovery strategies

### 04_prebuilt_tools_and_integration.py

- Simulated prebuilt tools
- Database integration patterns
- File system tools
- API integration examples
- Comprehensive agent composition

## Custom Tool Definitions

### 1. Define Simple Math Tool

Use `create_react_agent` to convert a vanilla Python function into a callable tool, for example, a `multiply(a, b)` function. ([langchain-ai.github.io][1])

### 2. Customize Tools with `@tool` Decorator

Demonstrate using `@tool` from `langchain_core.tools` to name the tool, parse docstrings, and define a Pydantic `args_schema` for structured inputs. ([langchain-ai.github.io][1])

## Advanced Tool Configuration

### 3. Hide Arguments from the Model

Show how to inject state or config parameters (e.g., user ID, session info) that are not exposed to the LLM by leveraging `InjectedState` and `RunnableConfig`. ([langchain-ai.github.io][1])

### 4. Disable Parallel Tool Calling

Illustrate disabling parallel execution of multiple tool calls via `model.bind_tools(..., parallel_tool_calls=False)` to ensure sequential invocation. ([langchain-ai.github.io][1])

### 5. Handle Tool Errors Gracefully

Use `ToolNode(handle_tool_errors=...)` to catch exceptions inside tools and return custom error messages or strategies back to the agent. ([langchain-ai.github.io][1])

### 6. Return Tool Results Directly

Configure tools with `return_direct=True` so that their output is returned immediately, terminating the agent loop after a single tool call. ([langchain-ai.github.io][1])

### 7. Force Tool Use

Demonstrate forcing the agent to use a specific tool via `model.bind_tools(tools, tool_choice={...})`, useful for testing or deterministic flows. ([langchain-ai.github.io][1])

## Prebuilt Tool Integrations

### 8. Web Search Preview Tool

Leverage the `web_search_preview` prebuilt tool (e.g., from OpenAI) to conduct news or web searches within an agent workflow. ([langchain-ai.github.io][1])

### 9. Python REPL Tool

Integrate the Python REPL tool to allow dynamic code execution and data inspection within the graph. ([langchain-ai.github.io][2])

### 10. SQL Database Tool

Connect to an SQL database (e.g., Postgres, MySQL) using the prebuilt SQL tool integration for querying and updating records. ([langchain-ai.github.io][2])

[1]: https://langchain-ai.github.io/langgraph/agents/tools/ "Tools"
[2]: https://langchain-ai.github.io/langgraph/concepts/tools/ "Overview"

### Core Building Blocks

1. **StateGraph & MessagesState** - [Essential foundation](https://www.gettingstarted.ai/langgraph-tutorial-with-example/) for managing conversation state and data flow between nodes
2. **Basic Chatbot with Memory** - [Simple conversational agent](https://www.analyticsvidhya.com/blog/2025/05/langgraph-tutorial-for-beginners/) that maintains chat history and context

### Agent Architectures

3. **Multi-Agent Supervisor** - [Supervisor architecture](https://www.analyticsvidhya.com/blog/2025/05/langgraph-tutorial-for-beginners/) where one agent controls and routes tasks to specialized agents
4. **Tool-Calling Agents** - [Agents that can use external tools](https://www.scalablepath.com/machine-learning/langgraph) like web search, APIs, or databases

### Advanced Features

5. **RAG System with LangGraph** - [Retrieval-Augmented Generation](https://medium.com/@nikhilpurao1998/how-to-guide-for-langgraph-3856d49896aa) combining vector search with conversational AI
6. **Human-in-the-Loop Integration** - [Interactive workflows](https://www.analyticsvidhya.com/blog/2025/05/langgraph-tutorial-for-beginners/) where humans can review and approve agent actions
7. **Conditional Logic & Branching** - [Dynamic decision-making](https://realpython.com/langgraph-python/) with conditional edges based on agent outputs

### Production Features

8. **Persistence & Checkpointing** - [State management](https://www.analyticsvidhya.com/blog/2025/05/langgraph-tutorial-for-beginners/) for resuming interrupted workflows and error recovery
9. **Streaming Outputs** - [Real-time response streaming](https://www.analyticsvidhya.com/blog/2025/05/langgraph-tutorial-for-beginners/) for better user experience
10. **Resume Enhancement Agent** - [Practical application](https://www.gettingstarted.ai/langgraph-tutorial-with-example/) that analyzes job descriptions and improves resumes accordingly
