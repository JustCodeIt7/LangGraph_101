# LangGraph Human-in-the-Loop (HITL) Examples

This directory provides three distinct examples of implementing Human-in-the-Loop (HITL) workflows using LangGraph. Each example is self-contained and demonstrates a different pattern for integrating human oversight and intervention into agentic systems.

## 🎯 Overview

Human-in-the-Loop (HITL) workflows are essential for building trustworthy AI systems that maintain human oversight and control. These examples showcase different patterns for integrating human decision-making into automated agent workflows using LangGraph's state management and conditional routing capabilities.

## ⚙️ HITL Patterns Demonstrated

### 1. `example1_basic_approval.py`
**Pattern**: Simple Yes/No Approval
**Use Case**: Sensitive actions requiring explicit permission

**Features**:
- Binary approval workflow (approve/reject)
- State persistence during human interaction
- Clear action descriptions for review
- Graceful handling of rejections
- Multiple action types (email, purchase, file operations)

**Example Scenarios**:
- Email campaign approval
- Purchase authorization
- File deletion confirmation
- Database modification approval

**Key Components**:
- `BasicApprovalState`: Simple state with approval status
- `propose_action()`: Presents action for review
- `request_human_approval()`: Captures human decision
- `execute_action()`: Performs approved actions

---

### 2. `example2_content_editing.py`
**Pattern**: Iterative Review and Editing
**Use Case**: Content creation with human refinement

**Features**:
- Multi-turn review and editing cycles
- Human content modification capabilities
- Iterative refinement with AI assistance
- Version history tracking
- Flexible approval process

**Example Scenarios**:
- Blog post creation and editing
- Email draft refinement
- Social media content approval
- Document review workflows

**Key Components**:
- `ContentEditingState`: Rich state with content history
- `generate_initial_content()`: AI creates first draft
- `request_human_review()`: Multi-option review interface
- `revise_content()`: AI revisions based on feedback

**Review Options**:
1. **Approve**: Accept content as-is
2. **Edit**: Direct human modification
3. **Revise**: Request AI revision with feedback
4. **Reject**: Start over

---

### 3. `example3_decision_tree.py`
**Pattern**: Multi-Step Decision Tree
**Use Case**: Complex workflows with multiple decision points

**Features**:
- Sequential decision points
- Dynamic branching based on choices
- Context-aware option presentation
- Rollback to previous decisions
- Complex state management

**Example Scenarios**:
- Project planning workflows
- Hiring process management
- Product launch decisions
- Strategic planning sequences

**Key Components**:
- `DecisionTreeState`: Complex state with decision history
- `initialize_scenario()`: Sets up workflow context
- `present_decision_options()`: Context-aware option display
- `get_human_decision()`: Captures choices with rollback
- `determine_next_step()`: Dynamic workflow routing

**Decision Scenarios**:
1. **Project Planning**: Budget → Technology → Team Structure
2. **Hiring Process**: Screening → Offer Strategy → Approval
3. **Product Launch**: Timing → Marketing → Execution

## 🚀 Getting Started

### Prerequisites

```bash
pip install langgraph langchain-core
```

### Running the Examples

Each example is self-contained and can be run independently:

```bash
# Basic Approval Pattern
python example1_basic_approval.py

# Content Editing Pattern
python example2_content_editing.py

# Decision Tree Pattern
python example3_decision_tree.py
```

### Integration Patterns

#### Web Application Integration
```python
# Example: Flask integration
from flask import request, jsonify
from langgraph.checkpoint.memory import MemorySaver

app = create_basic_approval_graph()
memory = MemorySaver()

@app.route('/request-approval', methods=['POST'])
def request_approval():
    state = request.json
    config = {"configurable": {"thread_id": state.get("user_id")}}
    result = app.invoke(state, config)
    return jsonify(result)
```

#### WebSocket Real-time Updates
```python
import asyncio
import websockets

async def hitl_websocket(websocket, path):
    async for message in websocket:
        # Process HITL decision
        state = json.loads(message)
        result = await app.ainvoke(state, config)
        await websocket.send(json.dumps(result))
```

#### REST API Integration
```python
# Example: FastAPI integration
from fastapi import FastAPI
from pydantic import BaseModel

class ApprovalRequest(BaseModel):
    action_description: str
    action_type: str
    parameters: dict

@app.post("/approve")
async def approve_action(request: ApprovalRequest):
    state = request.dict()
    result = await app.ainvoke(state, config)
    return result
```

## 🏗️ Architecture Patterns

### State Management
All examples use TypedDict for state management with:
- **Immutable updates**: State is never mutated directly
- **Type safety**: Clear type definitions for all state fields
- **Persistence**: Memory checkpointers for workflow continuity

### Error Handling
Robust error handling patterns:
- **Input validation**: Check for valid user inputs
- **Graceful degradation**: Handle invalid choices gracefully
- **Retry mechanisms**: Allow users to retry failed operations

### Conditional Routing
Smart workflow routing based on:
- **Human decisions**: Route based on approval/rejection
- **State conditions**: Check iteration limits, completion status
- **Context awareness**: Adapt options based on current state

## 🔧 Customization Guide

### Adding New Decision Types

1. **Extend State Definition**:
```python
class CustomState(TypedDict):
    # Add your custom fields
    custom_field: str
    custom_options: List[dict]
```

2. **Create Decision Node**:
```python
def custom_decision_node(state: CustomState) -> CustomState:
    # Your custom decision logic
    return updated_state
```

3. **Add to Workflow**:
```python
workflow.add_node("custom_decision", custom_decision_node)
workflow.add_edge("previous_node", "custom_decision")
```

### Custom Approval Logic

```python
def custom_approval_logic(state: YourState) -> Literal["approve", "reject", "modify"]:
    # Implement your approval criteria
    if meets_criteria(state):
        return "approve"
    elif needs_modification(state):
        return "modify"
    else:
        return "reject"
```

### Adding Persistence

```python
from langgraph.checkpoint.postgres import PostgresSaver

# Use PostgreSQL for persistence
checkpointer = PostgresSaver.from_conn_string("postgresql://...")
app = workflow.compile(checkpointer=checkpointer)
```

## 📊 Monitoring and Analytics

### Decision Tracking
```python
def track_decision(state: dict, decision: str):
    # Log decision for analytics
    logger.info(f"Decision made: {decision}", extra={
        "user_id": state.get("user_id"),
        "workflow_type": state.get("scenario_type"),
        "timestamp": datetime.now().isoformat()
    })
```

### Performance Metrics
- **Approval rates**: Track approval vs rejection ratios
- **Iteration counts**: Monitor review cycles in editing workflows
- **Decision timing**: Measure time between decisions
- **Rollback frequency**: Track rollback usage in decision trees

## 🔒 Security Considerations

### Input Validation
```python
def validate_user_input(user_input: str) -> bool:
    # Implement input sanitization
    if not user_input or len(user_input) > MAX_INPUT_LENGTH:
        return False
    return True
```

### Access Control
```python
def check_user_permissions(user_id: str, action_type: str) -> bool:
    # Implement role-based access control
    user_role = get_user_role(user_id)
    return action_type in ROLE_PERMISSIONS.get(user_role, [])
```

### Audit Logging
```python
def audit_log(state: dict, action: str, result: str):
    audit_logger.info(f"HITL Action: {action}", extra={
        "user_id": state.get("user_id"),
        "action": action,
        "result": result,
        "timestamp": datetime.now().isoformat(),
        "ip_address": get_client_ip()
    })
```

## 🧪 Testing Strategies

### Unit Testing
```python
def test_approval_workflow():
    state = create_test_state()
    result = approval_node(state)
    assert result["human_approval"] is not None
```

### Integration Testing
```python
async def test_full_workflow():
    app = create_approval_graph()
    config = {"configurable": {"thread_id": "test-123"}}
    
    # Simulate full workflow
    result = await app.ainvoke(test_state, config)
    assert result["workflow_complete"] is True
```

### Mock Human Input
```python
def mock_human_input(responses: List[str]):
    """Mock human input for testing"""
    def mock_input(prompt):
        return responses.pop(0) if responses else "approve"
    return mock_input
```

## 📚 Best Practices

### 1. Clear Communication
- Provide clear, contextual information for human decisions
- Use descriptive action descriptions
- Display relevant parameters and implications

### 2. Graceful Error Handling
- Validate all user inputs
- Provide helpful error messages
- Allow users to retry failed operations

### 3. State Management
- Keep state immutable
- Use type hints for clarity
- Implement proper state validation

### 4. User Experience
- Minimize decision fatigue with clear options
- Provide rollback capabilities where appropriate
- Show progress and context throughout workflows

### 5. Performance
- Use appropriate checkpointers for your scale
- Implement timeouts for human decisions
- Cache expensive operations

## 🤝 Contributing

When adding new HITL patterns:

1. **Follow naming conventions**: `exampleN_pattern_name.py`
2. **Include comprehensive documentation**: Docstrings and comments
3. **Add demo functions**: Show real-world usage scenarios
4. **Update this README**: Document new patterns and features

## 📄 License

These examples are part of the LangGraph_101 tutorial series and follow the same license as the main repository.