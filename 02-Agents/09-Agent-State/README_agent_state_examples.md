# LangGraph Agent State Examples

This directory contains three comprehensive examples demonstrating different patterns and approaches to managing agent state in LangGraph applications.

## Overview

Each example showcases a different aspect of agent state management:

1. **Task Tracking Agent State** - Managing multi-step workflows and progress tracking
2. **Context-Aware Agent State** - Maintaining conversation context and user preferences  
3. **Collaborative Agent State** - Coordinating multiple agents with shared knowledge

## Example 1: Task Tracking Agent State

**File**: [`example1_task_tracking_agent_state.py`](example1_task_tracking_agent_state.py)

### Purpose
Demonstrates how to manage complex task workflows with progress tracking, subtask management, and completion status.

### Key State Components
```python
class TaskTrackingAgentState(TypedDict):
    messages: Annotated[List, add_messages]
    current_task_id: Optional[str]
    tasks: Dict[str, Task]  # Task ID -> Task
    task_queue: List[str]  # Priority-ordered task IDs
    overall_progress: float
    active_goals: List[str]
    session_stats: Dict[str, int]
```

### Features
- **Task Creation**: Automatically detects and creates tasks from user input
- **Progress Tracking**: Calculates completion percentages and tracks status
- **Priority Queue**: Orders tasks by priority for optimal workflow
- **Status Management**: Tracks tasks through pending → in_progress → completed states
- **Session Statistics**: Maintains counts of created and completed tasks

### Usage
```bash
python example1_task_tracking_agent_state.py
```

Commands:
- `new task [description]` - Creates a new task
- `complete` - Marks current task as completed
- `status` - Shows all tasks and progress
- `exit` - Exits the application

### Key Learning Points
- How to manage complex nested state structures
- Progress calculation and status tracking
- Task queue management with priority ordering
- State updates across multiple node executions

---

## Example 2: Context-Aware Agent State

**File**: [`example2_context_aware_agent_state.py`](example2_context_aware_agent_state.py)

### Purpose
Shows how to maintain rich conversation context, learn user preferences, and adapt communication style dynamically.

### Key State Components
```python
class ContextAwareAgentState(TypedDict):
    messages: Annotated[List, add_messages]
    user_profile: UserProfile
    conversation_context: ConversationContext
    emotional_state: EmotionalState
    learned_preferences: Dict[str, any]
    session_memory: Dict[str, any]
    long_term_insights: List[str]
    adaptation_history: List[Dict[str, any]]
```

### Features
- **Profile Learning**: Automatically detects communication style preferences
- **Topic Tracking**: Maintains conversation context and topic depth
- **Emotional Awareness**: Detects and responds to user emotional state
- **Adaptive Responses**: Adjusts communication style based on learned preferences
- **Interest Detection**: Identifies and remembers user interests
- **Expertise Assessment**: Adapts explanations to user's technical level

### Usage
```bash
python example2_context_aware_agent_state.py
```

Commands:
- Normal conversation - Agent learns and adapts automatically
- `profile` - Shows what the agent has learned about you
- `exit` - Exits the application

### Key Learning Points
- Dynamic user profiling and preference learning
- Contextual conversation management
- Emotional state detection and adaptation
- Multi-faceted state updates (profile, context, emotions)
- Adaptive system prompt generation

---

## Example 3: Collaborative Agent State

**File**: [`example3_collaborative_agent_state.py`](example3_collaborative_agent_state.py)

### Purpose
Demonstrates multi-agent coordination with shared knowledge, work delegation, and consensus building.

### Key State Components
```python
class CollaborativeAgentState(TypedDict):
    messages: Annotated[List, add_messages]
    active_agents: Dict[AgentRole, AgentInfo]
    current_coordinator: AgentRole
    shared_workspace: Dict[str, any]
    knowledge_base: SharedKnowledge
    active_tasks: Dict[str, CollaborativeTask]
    collaboration_queue: List[str]
    consensus_items: List[Dict[str, any]]
    meeting_notes: List[str]
    project_timeline: List[Dict[str, any]]
```

### Features
- **Multi-Agent Coordination**: Manages 6 specialized agent roles
- **Intelligent Task Assignment**: Automatically assigns relevant agents based on request type
- **Shared Knowledge Base**: Maintains collective insights and decisions
- **Consensus Building**: Synthesizes multiple agent perspectives
- **Work Delegation**: Distributes tasks based on agent expertise and workload
- **Collaboration Tracking**: Records meeting notes and collaboration history

### Agent Roles
- **Coordinator**: Manages workflow and task assignment
- **Researcher**: Gathers information and data
- **Analyst**: Processes data and identifies patterns
- **Creative**: Provides innovative solutions and alternatives
- **Technical**: Assesses feasibility and implementation
- **Critic**: Reviews work and identifies improvements

### Usage
```bash
python example3_collaborative_agent_state.py
```

Commands:
- Normal requests - System automatically assigns relevant agents
- `status` - Shows current collaboration status
- `exit` - Exits the application

### Key Learning Points
- Multi-agent state coordination
- Dynamic agent assignment based on task requirements
- Shared knowledge management
- Consensus building from multiple perspectives
- Complex workflow orchestration

---

## Common Patterns Across Examples

### 1. State Structure Design
All examples use TypedDict for type safety and clear state definitions:
```python
class MyAgentState(TypedDict):
    messages: Annotated[List, add_messages]  # Always include messages
    # ... other state components
```

### 2. Node Specialization
Each example demonstrates specialized nodes:
- **Analysis Nodes**: Process input and update state
- **Tracking Nodes**: Maintain progress and context
- **Response Nodes**: Generate contextually appropriate responses

### 3. State Update Patterns
```python
def my_node(state: MyAgentState):
    # Read current state
    current_value = state["some_field"]
    
    # Process and update
    updated_value = process(current_value)
    
    # Return partial state update
    return {"some_field": updated_value}
```

### 4. Graph Flow Design
All examples follow similar patterns:
```python
workflow.set_entry_point("analyzer")
workflow.add_edge("analyzer", "processor") 
workflow.add_edge("processor", "responder")
workflow.add_edge("responder", END)
```

## Running the Examples

### Prerequisites
```bash
pip install langchain langchain-openai langgraph
```

### Environment Setup
Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Individual Execution
Each example can be run independently:
```bash
# Task tracking
python example1_task_tracking_agent_state.py

# Context awareness  
python example2_context_aware_agent_state.py

# Multi-agent collaboration
python example3_collaborative_agent_state.py
```

## Key Takeaways

1. **State Design**: Plan your state structure carefully - it's the foundation of your agent's capabilities
2. **Node Specialization**: Create focused nodes that handle specific aspects of state management
3. **Partial Updates**: Return only the state fields that need updating from each node
4. **Type Safety**: Use TypedDict for clear contracts and better development experience
5. **Context Preservation**: Maintain conversation history and context across interactions
6. **Scalability**: Design state structures that can grow with your application's complexity

## Next Steps

- Experiment with combining patterns from different examples
- Add persistence to maintain state across sessions
- Implement more sophisticated NLP for better context understanding
- Add error handling and state validation
- Explore streaming capabilities with complex state management

These examples provide a solid foundation for building sophisticated LangGraph applications with rich state management capabilities.