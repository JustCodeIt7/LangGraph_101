# Looping Logic in LangGraph

This section demonstrates how to implement loops and iteration patterns in LangGraph, enabling your agents to repeat operations until a condition is met.

## Overview

Loops are essential for building agents that need to:
- Retry failed operations
- Process items iteratively
- Continue until a goal is achieved
- Implement iterative refinement

## Key Concepts

### Loop Pattern

Create loops by connecting nodes back to earlier nodes in the graph:

```python
from langgraph.graph import StateGraph, END

def should_continue(state: LoopState) -> str:
    """Determine if we should continue looping."""
    if state.get("completed"):
        return "end"
    return "continue"

# Build a loop by adding an edge back to a previous node
graph.add_edge("process", "check_status")
graph.add_conditional_edges("check_status", should_continue, {
    "end": END,
    "continue": "process"  # Loop back!
})
```

### State Tracking for Loops

Track iteration count and completion status:

```python
class LoopState(TypedDict):
    items: list[str]
    current_index: int
    completed: bool
    attempts: int

def process_item(state: LoopState) -> LoopState:
    """Process one item per iteration."""
    idx = state["current_index"]
    
    if idx >= len(state["items"]):
        return {"completed": True}
    
    # Process current item
    result = process(state["items"][idx])
    
    return {
        "current_index": idx + 1,
        "attempts": state.get("attempts", 0) + 1
    }
```

## Files in This Directory

| File | Description |
|------|-------------|
| `06-Looping_Logic.ipynb` | Main notebook with examples |
| `06-Looping_Logic.py` | Standalone Python implementation |
| `diagram.md` | Diagram notes |
| `output/` | Generated visualizations |

### Output Visualizations

- `06.1-Looping_Logic.mmd/.png` - Basic loop pattern
- `06.2-Looping_Logic.mmd/.png` - Advanced looping with conditions

## Code Examples

### Example 1: Retry Pattern

```python
def should_retry(state: RetryState) -> str:
    """Check if we should retry."""
    if state.get("success"):
        return "end"
    
    attempts = state.get("attempts", 0)
    max_attempts = 3
    
    if attempts >= max_attempts:
        return "fail"
    
    return "retry"

graph.add_conditional_edges("attempt_operation", should_retry, {
    "end": END,
    "retry": "attempt_operation",
    "fail": "error_handler"
})
```

### Example 2: Collection Pattern

```python
def should_continue_collection(state: CollectionState) -> str:
    """Check if we need more items."""
    collected = state.get("collected_count", 0)
    target = state.get("target_count", 5)
    
    if collected >= target:
        return "finish"
    return "collect_more"

graph.add_edge("collect_item", "check_collection")
graph.add_conditional_edges("check_collection", should_continue_collection, {
    "finish": END,
    "collect_more": "collect_item"
})
```

### Example 3: Iterative Refinement

```python
def should_refine(state: RefineState) -> str:
    """Determine if more refinement is needed."""
    quality = state.get("quality_score", 0)
    
    if quality >= 0.9:
        return "done"
    elif state.get("iterations", 0) >= 10:
        return "done"  # Max iterations reached
    return "refine"

graph.add_edge("generate", "evaluate")
graph.add_conditional_edges("evaluate", should_refine, {
    "done": END,
    "refine": "generate"
})
```

## Graph Visualization

The `output/` directory contains:
- Loop flow diagrams showing back edges
- State transitions during iteration
- Visual representation of termination conditions

## Running the Code

```bash
# Run Python implementation
python 06-Looping_Logic.py

# Or use Jupyter notebook
jupyter notebook 06-Looping_Logic.ipynb
```

## Best Practices

1. **Always have an exit condition** to prevent infinite loops
2. **Track iteration count** in state to enforce limits
3. **Use meaningful state flags** like `completed` or `success`
4. **Set recursion limits**: `graph.compile(config={"recursion_limit": 100})`

## Common Patterns

### Maximum Attempts

```python
MAX_ATTEMPTS = 5

def check_attempts(state: State) -> str:
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "give_up"
    return "continue"
```

### Convergence Check

```python
def has_converged(state: State) -> str:
    """Check if solution is stable."""
    current = state.get("current_output")
    previous = state.get("previous_output")
    
    if abs(current - previous) < 0.01:
        return "end"
    return "continue"
```

## Dependencies

- `langgraph>=0.4.7`
- `langchain-core~=0.3.59`

See also: [Conditional Routing](../05-Conditional_Routing/README.md) for decision logic.