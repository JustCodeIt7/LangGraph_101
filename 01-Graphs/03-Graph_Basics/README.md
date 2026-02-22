# LangGraph Basics

This section covers the fundamental concepts of building graphs in LangGraph, including state management, node creation, and basic graph construction.

## Overview

LangGraph is a library for building stateful, multi-step applications with Large Language Models (LLMs). This module introduces the core building blocks:

- **State**: The data that flows through your graph
- **Nodes**: Individual steps in your workflow
- **Edges**: Connections between nodes defining the flow

## Key Concepts

### State Definition

States are defined using `TypedDict` to provide type safety and clear schema definitions:

```python
from typing import TypedDict

class GraphState(TypedDict):
    messages: list[str]
    result: str
```

### Node Functions

Nodes are Python functions that:
- Accept the current state as input
- Return updated state (or a dict to merge updates)
- Are connected to form a directed graph

```python
def process_node(state: GraphState) -> GraphState:
    # Process the state
    return {"result": "processed"}
```

### Building the Graph

Create a `StateGraph` and add nodes:

```python
from langgraph.graph import StateGraph

graph = StateGraph(GraphState)
graph.add_node("process", process_node)
graph.set_entry_point("process")
graph.set_finish_point("process")
```

## Files in This Directory

| File | Description |
|------|-------------|
| `03-Graph_Basics.ipynb` | Jupyter notebook with interactive examples |
| `03-Graph_Basics-dev.ipynb` | Development version of the notebook |
| `output/` | Generated diagrams and visualizations |

## Running the Code

### Using Jupyter Notebook

```bash
jupyter notebook 03-Graph_Basics.ipynb
```

### Key Patterns Demonstrated

1. **Simple Node Execution**: Single node that processes input
2. **Multiple Inputs**: Passing multiple parameters to nodes
3. **Conditional Routing**: Decision-making based on state
4. **Sequential Graphs**: Multiple nodes connected in sequence

## Graph Visualization

The `output/` directory contains generated visualizations:
- `.mmd` files: Mermaid diagram definitions
- `.png` files: Rendered graph images
- `.svg` files: Vector graphics versions

## Dependencies

- `langgraph>=0.4.7`
- `langchain-core~=0.3.59`
- `jupyter` for notebook execution

## Further Reading

See the main [README.md](../README.md) for project-wide documentation.