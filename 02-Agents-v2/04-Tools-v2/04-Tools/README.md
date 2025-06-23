# LangGraph Tools Tutorial

This directory contains example scripts for a YouTube tutorial on using tools with LangGraph. Each script demonstrates different aspects of tool usage in LangGraph.

## Examples Overview

### 1. Basic Tools (`01-basic-tools.py`)
This example demonstrates how to define and use simple tools in LangGraph. It covers:
- Defining basic tool functions
- Creating an agent with tools
- Running the agent with different queries that use tools

### 2. Customizing Tools (`02-custom-tools.py`)
This example shows how to customize tools using decorators and Pydantic schemas. It covers:
- Using the `@tool` decorator with `parse_docstring=True`
- Creating custom input schemas with Pydantic
- Adding validation to tool inputs
- Creating an agent with custom tools

### 3. Tool Error Handling (`03-tool-error-handling.py`)
This example illustrates different strategies for handling errors in tools. It covers:
- Creating tools that can raise exceptions
- Default error handling behavior
- Custom error handling with specific messages for different error types
- Disabling error handling

### 4. Advanced Tool Features (`04-advanced-tool-features.py`)
This example demonstrates advanced tool features. It covers:
- Using `return_direct=True` to return tool results immediately
- Accessing hidden arguments (state and config) in tools
- Forcing tool use with `tool_choice`
- Disabling parallel tool calling with `parallel_tool_calls=False`

## Running the Examples

Each example can be run independently. Make sure you have the required dependencies installed:

```bash
pip install langgraph langchain langchain-core rich
```

To run an example:

```bash
python 01-basic-tools.py
```

## Model Configuration

These examples use either:
- `ollama:llama3.2` - Requires Ollama to be installed and the llama3.2 model to be available
- `openai:gpt-4.1-nano` - Requires an OpenAI API key to be set in your environment

You can modify the model used in each example by changing the `model` parameter in the `create_react_agent` function.

## Additional Resources

For more information on LangGraph tools, refer to the official documentation:
- [LangGraph Tools Documentation](https://langchain-ai.github.io/langgraph/agents/tools/)
- [LangChain Tools Documentation](https://python.langchain.com/docs/concepts/tools/)