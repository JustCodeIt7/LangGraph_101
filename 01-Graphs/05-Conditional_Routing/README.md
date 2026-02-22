# Conditional Routing in LangGraph

This section demonstrates how to implement conditional logic in LangGraph, allowing your graph to make decisions and route to different nodes based on the current state.

## Overview

Conditional routing enables dynamic flow control in your graphs. Instead of a fixed path, the graph can evaluate conditions and choose different next steps based on runtime data.

## Key Concepts

### Conditional Edge Functions

A conditional edge function takes the current state and returns a string representing the name of the next node:

```python
def route_logic(state: GraphState) -> str:
    """Determine which node to visit next."""
    if state.get("needs_more_info"):
        return "ask_question"
    return "finish"
```

### Using add_conditional_edges

Connect conditional logic to your graph:

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(GraphState)
graph.add_node("process", process_node)
graph.add_node("ask_question", question_node)
graph.add_node("finish", finish_node)

# Add conditional edge from 'process'
graph.add_conditional_edges(
    "process",
    route_logic,
    {
        "ask_question": "ask_question",
        "finish": "finish"
    }
)
```

## Files in This Directory

| File | Description |
|------|-------------|
| `05-Conditional_Routing.ipynb` | Main notebook with examples |
| `05-Conditional_Routing-v2.ipynb` | Alternative version |
| `05-Conditional_Routing.py` | Standalone Python implementation |
| `flowchart TD.mmd` | Mermaid flowchart definition |

## Code Examples

### Example 1: Text Routing

Route based on text content:

```python
def route_based_on_input(state: InputState) -> str:
    """Route to different processing paths based on input."""
    text = state.get("text", "").lower()
    
    if "help" in text or "?" in text:
        return "help_node"
    elif "complaint" in text:
        return "support_node"
    else:
        return "general_node"

graph.add_conditional_edges("process_input", route_based_on_input, {
    "help_node": "help_handler",
    "support_node": "support_handler", 
    "general_node": "general_handler"
})
```

### Example 2: State-Based Routing

Route based on processing results:

```python
def should_escalate(state: AnalysisState) -> str:
    """Determine if we need to escalate."""
    confidence = state.get("confidence", 0)
    
    if confidence < 0.5:
        return "human_review"
    elif confidence > 0.9:
        return "auto_approve"
    else:
        return "additional_checks"

graph.add_conditional_edges("analyze", should_escalate, {
    "human_review": "review_queue",
    "auto_approve": "approval_node",
    "additional_checks": "verification_node"
})
```

## Graph Visualization

The `output/` directory contains generated diagrams showing the conditional flow:
- Conditional paths are visualized with diamond-shaped decision nodes
- Each branch represents a possible outcome

## Running the Code

```bash
# Run Python implementation
python 05-Conditional_Routing.py

# Or use Jupyter notebook
jupyter notebook 05-Conditional_Routing.ipynb
```

## Best Practices

1. **Define all possible routes** in the mapping dictionary
2. **Return meaningful node names** that match your graph structure
3. **Handle edge cases** - ensure every state returns a valid route
4. **Use descriptive routing function names** for debugging

## Common Patterns

### Retry Logic

```python
def should_retry(state: RetryState) -> str:
    attempts = state.get("attempts", 0)
    max_attempts = 5
    
    if attempts >= max_attempts:
        return "fail"
    elif state.get("success"):
        return "success"
    else:
        return "retry"
```

### Validation Gates

```python
def validate_and_route(state: ValidatedState) -> str:
    errors = state.get("validation_errors", [])
    
    if errors:
        return "error_handler"
    return "continue_processing"
```

## Dependencies

- `langgraph>=0.4.7`
- `langchain-core~=0.3.59`

See also: [Multiple Inputs](../04-Multiple_Inputs/README.md) for input handling.