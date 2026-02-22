# Multiple Inputs in LangGraph

This section demonstrates how to handle multiple input parameters when building LangGraph applications.

## Overview

In real-world applications, nodes often need access to multiple pieces of data beyond what's stored in the state. This module shows different approaches for passing additional inputs to node functions.

## Key Concepts

### Using Graph Inputs

LangGraph allows you to pass initial inputs when invoking the graph:

```python
# Invoke with multiple inputs
result = graph.invoke({
    "user_input": "Hello",
    "context": {"session_id": "123"}
})
```

### State with Multiple Fields

Define a state with multiple fields to hold different types of data:

```python
from typing import TypedDict

class MultiInputState(TypedDict):
    user_input: str
    context: dict
    history: list[str]
```

### Node Functions with Additional Parameters

Nodes can access additional data through the `config` parameter:

```python
from langchain_core.runnables import RunnableConfig

def process_node(state: MultiInputState, config: RunnableConfig) -> MultiInputState:
    # Access configuration data
    session_id = config.get("configurable", {}).get("session_id")
    return {"result": f"Processed {state['user_input']}"}
```

## Files in This Directory

| File | Description |
|------|-------------|
| `04-Multiple_Inputs.ipynb` | Jupyter notebook with examples |
| `04-Multiple_Inputs.py` | Standalone Python implementation |
| `output/` | Generated diagrams |

### Code Example

```python
from langgraph.graph import StateGraph, END

# Define state with multiple fields
class InputState(TypedDict):
    name: str
    age: int
    preferences: dict

def process_name(state: InputState) -> InputState:
    """Process the user's name."""
    return {"processed": True}

def process_age(state: InputState) -> InputState:
    """Process the user's age."""
    return {"age_processed": True}

# Build graph with multiple inputs
graph = StateGraph(InputState)
graph.add_node("process_name", process_name)
graph.add_node("process_age", process_age)

# Set up sequential flow
graph.set_entry_point("process_name")
graph.add_edge("process_name", "process_age")
graph.add_edge("process_age", END)

# Compile the graph
app = graph.compile()

# Invoke with multiple inputs
result = app.invoke({
    "name": "John",
    "age": 30,
    "preferences": {"theme": "dark"}
})
```

## Graph Visualization

The `output/` directory contains:
- `04-Multiple_Inputs.mmd` - Mermaid diagram definition
- `04-Multiple_Inputs.png` - Rendered visualization

## Running the Code

```bash
# Run the Python script directly
python 04-Multiple_Inputs.py

# Or use Jupyter notebook
jupyter notebook 04-Multiple_Inputs.ipynb
```

## Best Practices

1. **Use TypedDict** for type-safe state definitions
2. **Keep related data together** in the state structure
3. **Pass configuration** through `RunnableConfig` for runtime settings
4. **Initialize default values** to avoid KeyError issues

## Dependencies

- `langgraph>=0.4.7`
- `langchain-core~=0.3.59`

See also: [LangGraph Basics](../03-Graph_Basics/README.md) for foundational concepts.