# Basic Chat Graph in LangGraph

This section demonstrates how to build a basic chatbot using LangGraph with state management for conversation history.

## Overview

A chat application requires:
- Maintaining message history
- Processing user input through LLM calls
- Managing conversation flow
- Handling tool execution when needed

## Key Concepts

### Message State Management

Use `Annotated` with reducer functions to manage message lists:

```python
from typing import TypedDict, Annotated
from operator import add

class ChatState(TypedDict):
    messages: Annotated[list, add]  # Append new messages

def process_message(state: ChatState) -> ChatState:
    """Add processed response to messages."""
    return {"messages": ["Assistant: Response here"]}
```

### Using LangChain Tools

Define tools that the LLM can call:

```python
from langchain_core.tools import tool

@tool
def search_database(query: str) -> str:
    """Search the database for information."""
    # Tool implementation
    return result
```

## Files in This Directory

| File | Description |
|------|-------------|
| `07-chat_app.py` | Basic chat application |
| `chat_app_v2.py` | Enhanced version with more features |
| `test_07-chat_app-v2.py` | Test suite for v2 |
| `chat_graph.mmd` | Mermaid diagram definition |

### Generated Visualizations

- `chat_graph.png` - Graph structure visualization
- `mermaid-diagram.png/.svg` - Alternative visualizations

## Code Example: Basic Chat App

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

# Define state with message history
class ChatState(TypedDict):
    messages: Annotated[list, add]

def chat_node(state: ChatState) -> ChatState:
    """Process user message and generate response."""
    # Get the last user message
    user_message = state["messages"][-1]
    
    # Call LLM (using Ollama in this project)
    response = model.invoke(user_message.content)
    
    # Add assistant's response to messages
    return {"messages": [response]}

# Build the graph
graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.set_entry_point("chat")
graph.add_edge("chat", END)

app = graph.compile()

# Run the chat
result = app.invoke({
    "messages": [HumanMessage(content="Hello!")]
})
```

## Code Example: Enhanced Chat with Tools

```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

# Create agent with tools
agent = create_react_agent(model, [calculator])

def chat_with_tools(state: ChatState) -> ChatState:
    """Chat with tool execution capability."""
    result = agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}
```

## Graph Structure

```
User Input → [Process] → LLM Response → End
           or
           [Tool Call] → Process Result → LLM Response → End
```

The graph can:
1. Process user messages directly
2. Invoke tools when needed
3. Loop back for additional interactions

## Running the Code

```bash
# Run basic chat app
python 07-chat_app.py

# Run enhanced version with tools
python chat_app_v2.py

# Run tests
python -m pytest test_07-chat_app-v2.py
```

## Best Practices

1. **Use message types**: `HumanMessage`, `AIMessage`, `SystemMessage`
2. **Persist state**: Use checkpointers for conversation persistence
3. **Handle tool errors gracefully**: Return error messages to the LLM
4. **Limit history**: Trim old messages to prevent context overflow

## State Patterns

### Simple Message List

```python
messages: Annotated[list, add]
```

### With Conversation Metadata

```python
class ChatState(TypedDict):
    messages: Annotated[list, add]
    session_id: str
    user_preferences: dict
```

### With Tool Call History

```python
class AgentState(TypedDict):
    messages: Annotated[list, add]
    tool_calls: list  # Track which tools were called
    tool_results: dict
```

## Dependencies

- `langgraph>=0.4.7`
- `langchain-core~=0.3.59`
- `langchain-ollama` (for local LLM)
- `langchain_core.messages`

See also: [Looping Logic](../06-Looping_Logic/README.md) for iteration patterns.