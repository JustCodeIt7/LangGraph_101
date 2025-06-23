# LangGraph Agent State Examples - Progressive Learning

This directory contains 4 examples that demonstrate LangGraph agent state management, progressing from basic concepts to advanced multi-agent coordination systems.

## Learning Path

### Start Here: Original Example
- **[`02-Agent-State.py`](02-Agent-State.py)** - The original enhanced agent state example
  - Basic state with `messages`, `user_name`, and `turn_count`
  - Simple node structure with initialization, greeting, and LLM response
  - Demonstrates fundamental state persistence across conversation turns

### Example 1: Basic Agent State (Simplified)
- **[`example1_basic_agent_state.py`](example1_basic_agent_state.py)** - Minimal state management
  - **Concepts**: TypedDict basics, simple state updates, turn counting
  - **State**: `messages` + `turn_count` only
  - **Nodes**: Turn counter → LLM response
  - **Best for**: Understanding core LangGraph state concepts

### Example 2: Intermediate Agent State
- **[`example2_intermediate_agent_state.py`](example2_intermediate_agent_state.py)** - Context-aware behavior
  - **Concepts**: Multi-field state, conditional logic, user profiling
  - **State**: User mood, topics, response styles, help tracking
  - **Nodes**: User analyzer → Style adapter → Turn tracker → Context-aware LLM
  - **Best for**: Learning adaptive behavior and context management

### Example 3: Advanced Agent State
- **[`example3_advanced_agent_state.py`](example3_advanced_agent_state.py)** - Multi-agent coordination
  - **Concepts**: Complex routing, specialist agents, task management, learning
  - **State**: Intent classification, task queues, agent coordination, learning patterns
  - **Nodes**: Intent classifier → Agent router → Task manager → Specialist LLM → Learning → Finalizer
  - **Best for**: Understanding sophisticated agent systems

## Progressive Complexity

### 🟢 Basic Level (Examples 1 & Original)
**What you'll learn:**
- How to define agent state with TypedDict
- Basic state initialization and updates
- Simple node-to-node state passing
- Conversation history management

**Key patterns:**
```python
class BasicState(TypedDict):
    messages: Annotated[List, add_messages]
    turn_count: int

def simple_node(state: BasicState):
    # Read state
    count = state["turn_count"]
    # Update and return
    return {"turn_count": count + 1}
```

### 🟡 Intermediate Level (Example 2)
**What you'll learn:**
- Multiple state fields with different data types
- Conditional behavior based on state values
- User profiling and preference tracking
- Context-aware response generation

**Key patterns:**
```python
# Complex state with multiple concerns
class IntermediateState(TypedDict):
    messages: Annotated[List, add_messages]
    user_mood: str
    conversation_topic: Optional[str]
    response_style: str
    help_count: int

# Conditional logic based on state
def style_adapter(state):
    mood = state["user_mood"]
    if mood == "confused":
        return {"response_style": "helpful"}
    elif mood == "positive":
        return {"response_style": "enthusiastic"}
```

### 🔴 Advanced Level (Example 3)
**What you'll learn:**
- Multi-agent coordination patterns
- Complex conditional routing
- Task queue and priority management
- Learning and adaptation systems
- Error handling and recovery

**Key patterns:**
```python
# Complex nested state structures
class AdvancedState(TypedDict):
    messages: Annotated[List, add_messages]
    active_tasks: Dict[str, Task]
    current_agent: AgentRole
    learned_patterns: Dict[str, int]
    # ... many more fields

# Conditional routing
def route_logic(state):
    confidence = state["confidence_score"]
    if confidence < 0.3:
        return "error_handler"
    else:
        return "specialist_agent"

workflow.add_conditional_edges("classifier", route_logic, {
    "error_handler": "error_handler",
    "specialist_agent": "specialist"
})
```

## Key Concepts by Example

| Concept | Basic | Intermediate | Advanced |
|---------|-------|--------------|----------|
| State Definition | ✅ Simple | ✅ Multi-field | ✅ Complex nested |
| Node Communication | ✅ Linear | ✅ Sequential | ✅ Conditional routing |
| User Profiling | ❌ | ✅ Mood/topics | ✅ Full profiling |
| Adaptation | ❌ | ✅ Style changes | ✅ Learning system |
| Task Management | ❌ | ❌ | ✅ Priority queues |
| Multi-Agent | ❌ | ❌ | ✅ Role-based agents |
| Error Handling | ❌ | ❌ | ✅ Recovery systems |

## Running the Examples

### Prerequisites
```bash
pip install langchain langchain-openai langgraph python-dotenv
export OPENAI_API_KEY="your-api-key-here"
```

### Execution Order (Recommended)
```bash
# 1. Start with the original
python 02-Agent-State.py

# 2. Understand basics
python example1_basic_agent_state.py

# 3. Learn intermediate concepts
python example2_intermediate_agent_state.py

# 4. Explore advanced features
python example3_advanced_agent_state.py
```

## Example Comparison

### State Complexity
- **Basic**: 2 fields (messages, turn_count)
- **Intermediate**: 6 fields (adds mood, topic, style, help_count)
- **Advanced**: 15+ fields (tasks, agents, learning, errors, etc.)

### Node Count
- **Basic**: 2 nodes (turn_counter → llm)
- **Intermediate**: 4 nodes (analyzer → adapter → tracker → llm)
- **Advanced**: 6+ nodes with conditional routing

### Interaction Features
- **Basic**: Simple Q&A with turn counting
- **Intermediate**: Mood detection, style adaptation, topic tracking
- **Advanced**: Intent classification, multi-agent routing, task management

## Learning Outcomes

After working through these examples, you'll understand:

1. **State Design Principles**
   - How to structure state for different use cases
   - When to use simple vs. complex state structures
   - Type safety with TypedDict

2. **Node Development Patterns**
   - Reading and updating state effectively
   - Partial state updates vs. full state replacement
   - Node specialization and responsibility

3. **Graph Architecture**
   - Linear vs. conditional routing
   - Error handling and recovery patterns
   - Multi-agent coordination strategies

4. **Advanced Techniques**
   - Learning and adaptation systems
   - Task management and prioritization
   - Complex decision trees

## Next Steps

After mastering these examples:
- Explore state persistence across sessions
- Add external tool integration
- Implement real-time streaming
- Build custom MCP servers
- Create domain-specific agents

## Tips for Success

1. **Start Simple**: Begin with the basic example even if you're experienced
2. **Understand State Flow**: Trace how state moves through nodes
3. **Experiment**: Modify examples to see how changes affect behavior
4. **Read Comments**: Each example has detailed explanations
5. **Build Incrementally**: Use patterns from simpler examples in complex ones

This progressive approach ensures you build a solid foundation before tackling advanced LangGraph concepts.