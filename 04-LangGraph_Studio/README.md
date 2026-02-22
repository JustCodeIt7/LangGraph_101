# LangGraph Studio

This section covers the LangGraph Studio integration, which provides a visual interface for building and debugging LangGraph applications.

## Overview

LangGraph Studio is a web-based IDE designed specifically for developing, testing, and deploying LangGraph agents. It provides:

- **Visual Graph Editor**: Build graphs with drag-and-drop nodes
- **Interactive Testing**: Test your agent in real-time
- **Debugging Tools**: Inspect state at each step
- **Deployment Ready**: Export configurations for production

## Structure

```
04-LangGraph_Studio/
├── 01-Studio_Basics/          # Basic setup and configuration
│   ├── langgraph.json         # Graph definition file
│   └── src/
│       └── agent.py           # Agent implementation
```

## Getting Started

### Configuration File (langgraph.json)

The `langgraph.json` file defines your graph structure:

```json
{
  "dependencies": ["src/agent.py"],
  "graphs": {
    "agent": "agent.graph"
  }
}
```

### Running the Studio

1. Install LangGraph CLI:
   ```bash
   pip install langgraph-cli
   ```

2. Start the studio:
   ```bash
   langgraph studio
   ```

3. Open your browser to the provided URL (typically http://localhost:4050)

## Key Concepts

- **Node**: A function that processes state and returns updates
- **Edge**: Connections between nodes defining flow control
- **State Schema**: Defines what data flows through the graph
- **Checkpointer**: Enables conversation persistence

## Example Agent Structure

```python
from langgraph.graph import StateGraph
from typing import TypedDict

class GraphState(TypedDict):
    messages: list

def node_function(state: GraphState) -> GraphState:
    # Process state
    return {"messages": [...]}  # or state update

# Build the graph
graph = StateGraph(GraphState)
graph.add_node("node_name", node_function)
graph.set_entry_point("node_name")
```

## Features

- **Visual Debugging**: Step through execution and inspect state
- **Test Chat**: Interact with your agent in real-time
- **Checkpoints**: Save and restore conversation states
- **Streaming**: See token-by-token generation