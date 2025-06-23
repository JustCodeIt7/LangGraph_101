"""
Example 3: Advanced Agent State
==============================

This example demonstrates advanced state management with multiple specialized
agents, complex decision trees, state persistence, and sophisticated
workflow orchestration with conditional routing.

Key Concepts:
- Multi-agent coordination with role-based specialization
- Complex state with nested structures and enums
- Conditional routing based on state values
- Task queue and priority management
- State-driven workflow decisions
- Memory and learning capabilities
- Advanced error handling and recovery
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List, Dict, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from datetime import datetime
from enum import Enum

# 1. Define Advanced State with Complex Data Structures
class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

class AgentRole(Enum):
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    CRITIC = "critic"
    RESEARCHER = "researcher"

class UserIntent(Enum):
    QUESTION = "question"
    TASK_REQUEST = "task_request"
    COMPLAINT = "complaint"
    COMPLIMENT = "compliment"
    UNCLEAR = "unclear"

class Task(TypedDict):
    id: str
    description: str
    priority: TaskPriority
    assigned_agent: AgentRole
    status: str
    created_at: str
    estimated_complexity: int

class AdvancedAgentState(TypedDict):
    # Core conversation
    messages: Annotated[List, add_messages]
    
    # User profiling
    user_name: str
    expertise_level: str  # beginner, intermediate, expert
    interaction_history: List[Dict]
    user_satisfaction: float  # 0.0 to 1.0
    
    # Conversation management
    turn_count: int
    current_intent: UserIntent
    confidence_score: float
    
    # Task management
    active_tasks: Dict[str, Task]
    completed_tasks: List[str]
    failed_tasks: List[str]
    
    # Multi-agent coordination
    current_agent: AgentRole
    agent_recommendations: Dict[AgentRole, str]
    collaboration_needed: bool
    
    # Learning and adaptation
    learned_patterns: Dict[str, int]
    successful_strategies: List[str]
    user_preferences: Dict[str, any]
    
    # System state
    system_mode: str  # normal, debug, learning
    error_count: int
    last_error: Optional[str]

# 2. Advanced Node Functions

def intent_classifier_node(state: AdvancedAgentState):
    """
    Advanced intent classification with confidence scoring.
    Determines user intent and routes to appropriate specialist.
    """
    print("---CLASSIFYING USER INTENT---")
    
    messages = state["messages"]
    if not messages:
        return state
    
    latest_message = messages[-1]
    if not isinstance(latest_message, HumanMessage):
        return state
    
    user_text = latest_message.content.lower()
    
    # Intent classification with confidence
    intent_scores = {}
    
    # Question indicators
    question_words = ["what", "how", "why", "when", "where", "can you", "?"]
    question_score = sum(1 for word in question_words if word in user_text) / len(question_words)
    intent_scores[UserIntent.QUESTION] = question_score
    
    # Task request indicators
    task_words = ["please", "help me", "i need", "can you do", "create", "make"]
    task_score = sum(1 for word in task_words if word in user_text) / len(task_words)
    intent_scores[UserIntent.TASK_REQUEST] = task_score
    
    # Complaint indicators
    complaint_words = ["problem", "wrong", "doesn't work", "frustrated", "bad"]
    complaint_score = sum(1 for word in complaint_words if word in user_text) / len(complaint_words)
    intent_scores[UserIntent.COMPLAINT] = complaint_score
    
    # Compliment indicators
    compliment_words = ["great", "thank you", "awesome", "perfect", "excellent"]
    compliment_score = sum(1 for word in compliment_words if word in user_text) / len(compliment_words)
    intent_scores[UserIntent.COMPLIMENT] = compliment_score
    
    # Determine best intent
    best_intent = max(intent_scores, key=intent_scores.get)
    confidence = intent_scores[best_intent]
    
    if confidence < 0.1:
        best_intent = UserIntent.UNCLEAR
        confidence = 0.0
    
    print(f"Intent: {best_intent.value}, Confidence: {confidence:.2f}")
    
    return {
        "current_intent": best_intent,
        "confidence_score": confidence
    }

def agent_router_node(state: AdvancedAgentState):
    """
    Routes to appropriate specialist agent based on intent and context.
    Demonstrates complex conditional logic.
    """
    print("---ROUTING TO SPECIALIST AGENT---")
    
    intent = state.get("current_intent", UserIntent.UNCLEAR)
    confidence = state.get("confidence_score", 0.0)
    turn_count = state.get("turn_count", 0)
    error_count = state.get("error_count", 0)
    
    # Complex routing logic
    if intent == UserIntent.TASK_REQUEST and confidence > 0.3:
        agent = AgentRole.SPECIALIST
        collaboration_needed = confidence < 0.7  # Low confidence needs collaboration
    elif intent == UserIntent.QUESTION:
        if turn_count < 3:
            agent = AgentRole.COORDINATOR  # New users get coordinator
        else:
            agent = AgentRole.RESEARCHER   # Experienced users get researcher
        collaboration_needed = False
    elif intent == UserIntent.COMPLAINT or error_count > 2:
        agent = AgentRole.CRITIC
        collaboration_needed = True  # Always collaborate on complaints
    elif intent == UserIntent.COMPLIMENT:
        agent = AgentRole.COORDINATOR
        collaboration_needed = False
    else:
        agent = AgentRole.COORDINATOR  # Default fallback
        collaboration_needed = True
    
    print(f"Routed to: {agent.value}, Collaboration needed: {collaboration_needed}")
    
    return {
        "current_agent": agent,
        "collaboration_needed": collaboration_needed
    }

def task_manager_node(state: AdvancedAgentState):
    """
    Manages task creation, prioritization, and assignment.
    Shows complex state manipulation with nested structures.
    """
    print("---MANAGING TASKS---")
    
    intent = state.get("current_intent", UserIntent.UNCLEAR)
    
    if intent == UserIntent.TASK_REQUEST:
        messages = state["messages"]
        latest_message = messages[-1]
        
        # Create new task
        task_id = f"task_{len(state['active_tasks']) + 1}"
        
        # Estimate complexity (simple heuristic)
        text_length = len(latest_message.content)
        complexity = min(10, max(1, text_length // 20))
        
        # Determine priority based on keywords
        urgent_words = ["urgent", "asap", "immediately", "emergency"]
        high_words = ["important", "priority", "soon", "quickly"]
        
        user_text = latest_message.content.lower()
        if any(word in user_text for word in urgent_words):
            priority = TaskPriority.URGENT
        elif any(word in user_text for word in high_words):
            priority = TaskPriority.HIGH
        else:
            priority = TaskPriority.MEDIUM
        
        new_task: Task = {
            "id": task_id,
            "description": latest_message.content,
            "priority": priority,
            "assigned_agent": state.get("current_agent", AgentRole.COORDINATOR),
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "estimated_complexity": complexity
        }
        
        updated_tasks = state["active_tasks"].copy()
        updated_tasks[task_id] = new_task
        
        print(f"Created task {task_id}: Priority {priority.name}, Complexity {complexity}")
        
        return {"active_tasks": updated_tasks}
    
    return state

def specialist_llm_node(state: AdvancedAgentState):
    """
    Generates responses from specialist agents with role-specific behavior.
    """
    print("---SPECIALIST AGENT RESPONDING---")
    
    messages = state["messages"]
    current_agent = state.get("current_agent", AgentRole.COORDINATOR)
    user_name = state.get("user_name", "User")
    expertise_level = state.get("expertise_level", "intermediate")
    intent = state.get("current_intent", UserIntent.UNCLEAR)
    
    # Role-specific prompts
    role_prompts = {
        AgentRole.COORDINATOR: f"""You are the Coordinator Agent. You manage overall workflow and provide general assistance.
        User: {user_name} (Level: {expertise_level})
        Intent: {intent.value}
        
        Be organized, clear, and helpful in coordinating the response.""",
        
        AgentRole.SPECIALIST: f"""You are the Specialist Agent with deep technical expertise.
        User: {user_name} (Level: {expertise_level})
        Intent: {intent.value}
        
        Provide detailed, technical responses appropriate to the user's expertise level.""",
        
        AgentRole.RESEARCHER: f"""You are the Research Agent focused on finding and providing information.
        User: {user_name} (Level: {expertise_level})
        Intent: {intent.value}
        
        Provide well-researched, factual responses with sources when relevant.""",
        
        AgentRole.CRITIC: f"""You are the Critic Agent focused on quality and problem-solving.
        User: {user_name} (Level: {expertise_level})
        Intent: {intent.value}
        
        Be constructive, identify issues, and provide solutions."""
    }
    
    system_prompt = role_prompts.get(current_agent, role_prompts[AgentRole.COORDINATOR])
    
    # Add collaboration context if needed
    if state.get("collaboration_needed", False):
        system_prompt += "\n\nNOTE: This response may need collaboration with other agents. Consider mentioning if additional expertise would be helpful."
    
    context_messages = [SystemMessage(content=system_prompt)] + messages
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    response = llm.invoke(context_messages)
    
    print(f"Response from {current_agent.value} agent")
    
    return {"messages": [AIMessage(content=response.content)]}

def learning_node(state: AdvancedAgentState):
    """
    Updates learning patterns and user preferences based on interaction.
    """
    print("---UPDATING LEARNING PATTERNS---")
    
    intent = state.get("current_intent", UserIntent.UNCLEAR)
    confidence = state.get("confidence_score", 0.0)
    current_agent = state.get("current_agent", AgentRole.COORDINATOR)
    
    # Update learned patterns
    updated_patterns = state["learned_patterns"].copy()
    pattern_key = f"{intent.value}_{current_agent.value}"
    updated_patterns[pattern_key] = updated_patterns.get(pattern_key, 0) + 1
    
    # Track successful strategies (simple heuristic)
    updated_strategies = state["successful_strategies"].copy()
    if confidence > 0.7:
        strategy = f"High confidence {intent.value} -> {current_agent.value}"
        if strategy not in updated_strategies:
            updated_strategies.append(strategy)
    
    # Update user satisfaction (placeholder logic)
    current_satisfaction = state.get("user_satisfaction", 0.5)
    if intent == UserIntent.COMPLIMENT:
        new_satisfaction = min(1.0, current_satisfaction + 0.1)
    elif intent == UserIntent.COMPLAINT:
        new_satisfaction = max(0.0, current_satisfaction - 0.1)
    else:
        new_satisfaction = current_satisfaction
    
    print(f"Learning updated: Pattern {pattern_key}, Satisfaction: {new_satisfaction:.2f}")
    
    return {
        "learned_patterns": updated_patterns,
        "successful_strategies": updated_strategies,
        "user_satisfaction": new_satisfaction
    }

def turn_finalizer_node(state: AdvancedAgentState):
    """
    Finalizes the turn by updating counters and interaction history.
    """
    print("---FINALIZING TURN---")
    
    current_count = state.get("turn_count", 0)
    new_count = current_count + 1
    
    # Add to interaction history
    interaction_record = {
        "turn": new_count,
        "intent": state.get("current_intent", UserIntent.UNCLEAR).value,
        "agent": state.get("current_agent", AgentRole.COORDINATOR).value,
        "confidence": state.get("confidence_score", 0.0),
        "timestamp": datetime.now().isoformat()
    }
    
    updated_history = state["interaction_history"].copy()
    updated_history.append(interaction_record)
    
    # Keep only last 50 interactions
    if len(updated_history) > 50:
        updated_history = updated_history[-50:]
    
    return {
        "turn_count": new_count,
        "interaction_history": updated_history
    }

# 3. Build Advanced Graph with Conditional Routing
def route_after_classification(state: AdvancedAgentState):
    """Route based on intent classification confidence"""
    confidence = state.get("confidence_score", 0.0)
    if confidence < 0.3:
        return "error_handler"
    else:
        return "agent_router"

def route_after_agent_selection(state: AdvancedAgentState):
    """Route based on whether we need task management"""
    intent = state.get("current_intent", UserIntent.UNCLEAR)
    if intent == UserIntent.TASK_REQUEST:
        return "task_manager"
    else:
        return "specialist_llm"

workflow = StateGraph(AdvancedAgentState)

# Add all nodes
workflow.add_node("intent_classifier", intent_classifier_node)
workflow.add_node("agent_router", agent_router_node)
workflow.add_node("task_manager", task_manager_node)
workflow.add_node("specialist_llm", specialist_llm_node)
workflow.add_node("learning", learning_node)
workflow.add_node("turn_finalizer", turn_finalizer_node)

# Error handler node
def error_handler_node(state: AdvancedAgentState):
    print("---HANDLING LOW CONFIDENCE/ERROR---")
    error_count = state.get("error_count", 0) + 1
    return {
        "current_agent": AgentRole.COORDINATOR,
        "error_count": error_count,
        "last_error": "Low confidence in intent classification"
    }

workflow.add_node("error_handler", error_handler_node)

# Define complex routing
workflow.set_entry_point("intent_classifier")
workflow.add_conditional_edges(
    "intent_classifier",
    route_after_classification,
    {
        "agent_router": "agent_router",
        "error_handler": "error_handler"
    }
)
workflow.add_edge("error_handler", "specialist_llm")
workflow.add_conditional_edges(
    "agent_router", 
    route_after_agent_selection,
    {
        "task_manager": "task_manager",
        "specialist_llm": "specialist_llm"
    }
)
workflow.add_edge("task_manager", "specialist_llm")
workflow.add_edge("specialist_llm", "learning")
workflow.add_edge("learning", "turn_finalizer")
workflow.add_edge("turn_finalizer", END)

# 4. Compile Advanced Graph
advanced_agent = workflow.compile()

# 5. Advanced Interactive Loop
if __name__ == "__main__":
    print("Advanced Multi-Agent State System")
    print("=" * 40)
    print("This system features:")
    print("• Intent classification with confidence scoring")
    print("• Multi-agent routing (Coordinator, Specialist, Researcher, Critic)")
    print("• Task management with priority queues")
    print("• Learning and adaptation capabilities")
    print("• Complex conditional routing")
    print("• Error handling and recovery\n")
    
    # Get user info
    user_name = input("What's your name? ") or "User"
    expertise = input("Your expertise level (beginner/intermediate/expert): ") or "intermediate"
    
    # Initialize complex state
    current_state = {
        "messages": [],
        "user_name": user_name,
        "expertise_level": expertise,
        "interaction_history": [],
        "user_satisfaction": 0.5,
        "turn_count": 0,
        "current_intent": UserIntent.UNCLEAR,
        "confidence_score": 0.0,
        "active_tasks": {},
        "completed_tasks": [],
        "failed_tasks": [],
        "current_agent": AgentRole.COORDINATOR,
        "agent_recommendations": {},
        "collaboration_needed": False,
        "learned_patterns": {},
        "successful_strategies": [],
        "user_preferences": {},
        "system_mode": "normal",
        "error_count": 0,
        "last_error": None
    }
    
    print(f"\nWelcome {user_name}! Your advanced assistant is ready.")
    print("Available commands:")
    print("• 'status' - View system status and learning progress")
    print("• 'tasks' - View active and completed tasks")
    print("• 'agents' - See agent performance and routing history")
    print("• 'debug' - Toggle debug mode")
    print("• 'exit' - Quit the system\n")
    
    while True:
        user_input = input(f"\nTurn {current_state['turn_count'] + 1} - {user_name}: ")
        
        if user_input.lower() == "exit":
            print(f"Goodbye {user_name}! Thank you for helping me learn.")
            break
        
        if user_input.lower() == "status":
            print(f"\n--- SYSTEM STATUS ---")
            print(f"Satisfaction Score: {current_state['user_satisfaction']:.2f}/1.0")
            print(f"Total Interactions: {len(current_state['interaction_history'])}")
            print(f"Learned Patterns: {len(current_state['learned_patterns'])}")
            print(f"Successful Strategies: {len(current_state['successful_strategies'])}")
            print(f"Error Count: {current_state['error_count']}")
            print(f"Active Tasks: {len(current_state['active_tasks'])}")
            continue
        
        if user_input.lower() == "tasks":
            print(f"\n--- TASK STATUS ---")
            if current_state['active_tasks']:
                for task_id, task in current_state['active_tasks'].items():
                    print(f"• {task_id}: {task['description'][:50]}... (Priority: {task['priority'].name})")
            else:
                print("No active tasks")
            continue
        
        if user_input.lower() == "agents":
            print(f"\n--- AGENT ROUTING HISTORY ---")
            recent_interactions = current_state['interaction_history'][-5:]
            for interaction in recent_interactions:
                print(f"Turn {interaction['turn']}: {interaction['intent']} -> {interaction['agent']} (Confidence: {interaction['confidence']:.2f})")
            continue
        
        if user_input.lower() == "debug":
            current_mode = current_state.get('system_mode', 'normal')
            new_mode = 'debug' if current_mode == 'normal' else 'normal'
            current_state['system_mode'] = new_mode
            print(f"Debug mode: {new_mode}")
            continue
        
        if not user_input.strip():
            print("Please enter some text.")
            continue
        
        # Add user message
        current_state["messages"] = [HumanMessage(content=user_input)]
        
        if current_state.get('system_mode') == 'debug':
            print("\n--- DEBUG: Processing Advanced Workflow ---")
        
        # Run advanced workflow
        result = advanced_agent.invoke(current_state)
        
        # Update state
        current_state.update(result)
        
        # Display response with agent info
        if result["messages"]:
            ai_message = result["messages"][-1]
            if isinstance(ai_message, AIMessage):
                agent_name = current_state.get("current_agent", AgentRole.COORDINATOR).value
                intent = current_state.get("current_intent", UserIntent.UNCLEAR).value
                confidence = current_state.get("confidence_score", 0.0)
                
                print(f"\n[{agent_name.upper()}] (Intent: {intent}, Confidence: {confidence:.2f})")
                print(f"{ai_message.content}")
    
    print(f"\n--- SESSION SUMMARY ---")
    print(f"Total turns: {current_state['turn_count']}")
    print(f"Final satisfaction: {current_state['user_satisfaction']:.2f}")
    print(f"Patterns learned: {len(current_state['learned_patterns'])}")
    print(f"Tasks created: {len(current_state['active_tasks']) + len(current_state['completed_tasks'])}")
    print(f"Most used agents: {[k for k, v in current_state['learned_patterns'].items() if v > 1]}")