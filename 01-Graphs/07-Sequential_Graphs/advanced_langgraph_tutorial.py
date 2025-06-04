# %% [markdown]
# # Comprehensive LangGraph Tutorial
# 
# ## Introduction
# 
# This notebook demonstrates advanced LangGraph concepts including:
# - Sequential and parallel nodes
# - Conditional routing
# - Loops and cycles
# - State management
# - Error handling
# - Multi-agent workflows
# 
# ## Environment Setup

# %%
from typing import TypedDict, Literal, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, SystemMessage
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from rich import print
from IPython.display import Image, display
from litellm import completion
import random
import json

# Load environment variables
load_dotenv("../.env")

# %% [markdown]
# ## 1. Basic State Definition with Message Handling

# %%
class TutorialState(TypedDict):
    # User information
    name: str
    user_type: Literal["beginner", "intermediate", "advanced"]
    
    # Conversation management
    messages: Annotated[list[AnyMessage], add_messages]
    current_step: int
    max_attempts: int
    attempt_count: int
    
    # Tutorial progress
    topics_covered: list[str]
    quiz_score: int
    needs_help: bool
    
    # Routing decisions
    next_action: Literal["continue", "quiz", "help", "review", "complete"]

# %% [markdown]
# ## 2. Node Functions - Each Demonstrating Different Concepts

# %%
def welcome_node(state: TutorialState) -> dict:
    """
    Entry point node - demonstrates basic state initialization
    """
    welcome_msg = AIMessage(
        content=f"Welcome to the LangGraph Tutorial, {state['name']}! "
        f"I see you're a {state['user_type']} level learner. Let's get started!"
    )
    
    return {
        "messages": [welcome_msg],
        "current_step": 1,
        "attempt_count": 0,
        "topics_covered": [],
        "quiz_score": 0,
        "needs_help": False,
        "next_action": "continue"
    }

def content_delivery_node(state: TutorialState) -> dict:
    """
    Content delivery node - demonstrates sequential processing
    """
    topics = {
        1: "Understanding Nodes and Edges",
        2: "State Management in LangGraph", 
        3: "Conditional Routing",
        4: "Implementing Loops and Cycles",
        5: "Error Handling and Recovery"
    }
    
    current_topic = topics.get(state["current_step"], "Advanced Concepts")
    
    content_msg = AIMessage(
        content=f"📚 Lesson {state['current_step']}: {current_topic}\n\n"
        f"This is where we would deliver the content for {current_topic}. "
        f"The content would be tailored for {state['user_type']} level learners."
    )
    
    updated_topics = state.get("topics_covered", []) + [current_topic]
    
    return {
        "messages": [content_msg],
        "topics_covered": updated_topics,
        "next_action": "quiz"
    }

def quiz_node(state: TutorialState) -> dict:
    """
    Quiz node - demonstrates conditional logic and scoring
    """
    # Simulate quiz question based on current step
    questions = {
        1: "What connects nodes in a LangGraph?",
        2: "What is the purpose of TypedDict in state management?",
        3: "How do you implement conditional routing?",
        4: "What enables loops in LangGraph?",
        5: "How do you handle errors in node functions?"
    }
    
    question = questions.get(state["current_step"], "What is LangGraph?")
    
    # Simulate random quiz performance
    correct = random.choice([True, False, True])  # 66% success rate
    
    if correct:
        score_increase = 10
        feedback = "✅ Correct! Well done!"
        next_action = "continue"
    else:
        score_increase = 0
        feedback = "❌ Not quite right. Let's review this topic."
        next_action = "help" if state["attempt_count"] >= 1 else "quiz"
    
    quiz_msg = AIMessage(
        content=f"🔍 Quiz: {question}\n{feedback}"
    )
    
    return {
        "messages": [quiz_msg],
        "quiz_score": state.get("quiz_score", 0) + score_increase,
        "attempt_count": state.get("attempt_count", 0) + 1,
        "next_action": next_action,
        "needs_help": not correct and state.get("attempt_count", 0) >= 1
    }

def help_node(state: TutorialState) -> dict:
    """
    Help node - demonstrates recovery and personalized assistance
    """
    help_content = {
        "beginner": "Let me break this down into simpler steps...",
        "intermediate": "Here's a more detailed explanation with examples...",
        "advanced": "Let's dive into the technical implementation details..."
    }
    
    help_msg = AIMessage(
        content=f"🆘 Help Mode Activated!\n\n"
        f"{help_content[state['user_type']]}\n\n"
        f"Topic: {state['topics_covered'][-1] if state['topics_covered'] else 'Getting Started'}\n"
        f"Don't worry, everyone learns at their own pace!"
    )
    
    return {
        "messages": [help_msg],
        "needs_help": False,
        "attempt_count": 0,  # Reset attempts after help
        "next_action": "quiz"  # Give another chance
    }

def review_node(state: TutorialState) -> dict:
    """
    Review node - demonstrates state aggregation and summary
    """
    topics_list = "\n".join([f"- {topic}" for topic in state.get("topics_covered", [])])
    
    review_msg = AIMessage(
        content=f"📋 Progress Review for {state['name']}\n\n"
        f"Level: {state['user_type']}\n"
        f"Current Step: {state['current_step']}\n"
        f"Quiz Score: {state.get('quiz_score', 0)}/50\n\n"
        f"Topics Covered:\n{topics_list}\n\n"
        f"Ready to continue? You're doing great!"
    )
    
    return {
        "messages": [review_msg],
        "next_action": "continue"
    }

def completion_node(state: TutorialState) -> dict:
    """
    Completion node - demonstrates final state handling
    """
    final_score = state.get("quiz_score", 0)
    performance = "Excellent" if final_score >= 40 else "Good" if final_score >= 30 else "Keep Learning"
    
    completion_msg = AIMessage(
        content=f"🎉 Congratulations {state['name']}!\n\n"
        f"You've completed the LangGraph tutorial!\n"
        f"Final Score: {final_score}/50 ({performance})\n"
        f"Topics Mastered: {len(state.get('topics_covered', []))}\n\n"
        f"You're now ready to build your own LangGraph applications!"
    )
    
    return {
        "messages": [completion_msg],
        "next_action": "complete"
    }

# %% [markdown]
# ## 3. Conditional Routing Functions

# %%
def route_next_action(state: TutorialState) -> Literal["content", "quiz", "help", "review", "complete"]:
    """
    Main routing function - demonstrates conditional logic
    """
    action = state.get("next_action", "continue")
    
    if action == "complete":
        return "complete"
    elif action == "help":
        return "help"
    elif action == "review":
        return "review"
    elif action == "quiz":
        return "quiz"
    else:  # continue
        return "content"

def should_continue_tutorial(state: TutorialState) -> Literal["continue", "end"]:
    """
    Continuation check - demonstrates loop control
    """
    current_step = state.get("current_step", 1)
    max_steps = 5
    
    # Check if tutorial should continue
    if current_step >= max_steps:
        return "end"
    
    # Check if user needs too much help (adaptive learning)
    if state.get("attempt_count", 0) > 3:
        return "end"  # Might need different approach
    
    return "continue"

def increment_step(state: TutorialState) -> dict:
    """
    Step increment node - demonstrates simple state updates
    """
    return {
        "current_step": state.get("current_step", 1) + 1,
        "attempt_count": 0,  # Reset for new topic
        "next_action": "continue"
    }

# %% [markdown]
# ## 4. Graph Construction - Demonstrating All LangGraph Features

# %%
def create_tutorial_graph():
    """
    Creates the complete tutorial graph demonstrating:
    - Sequential flow
    - Conditional routing  
    - Loops and cycles
    - Parallel processing potential
    - Error recovery
    """
    
    # Initialize the graph
    graph = StateGraph(TutorialState)
    
    # Add all nodes
    graph.add_node("welcome", welcome_node)
    graph.add_node("content", content_delivery_node)
    graph.add_node("quiz", quiz_node)
    graph.add_node("help", help_node)
    graph.add_node("review", review_node)
    graph.add_node("increment", increment_step)
    graph.add_node("complete", completion_node)
    
    # Entry point
    graph.add_edge(START, "welcome")
    
    # Main flow with conditional routing
    graph.add_conditional_edges(
        "welcome",
        route_next_action,
        {
            "content": "content",
            "quiz": "quiz", 
            "help": "help",
            "review": "review",
            "complete": "complete"
        }
    )
    
    # Content to quiz flow
    graph.add_conditional_edges(
        "content",
        route_next_action,
        {
            "quiz": "quiz",
            "review": "review",
            "complete": "complete"
        }
    )
    
    # Quiz routing - demonstrates complex conditional logic
    graph.add_conditional_edges(
        "quiz",
        route_next_action,
        {
            "continue": "increment",  # Passed quiz, move to next topic
            "quiz": "quiz",           # Retry quiz (loop)
            "help": "help",           # Need assistance
            "review": "review",       # Review progress
            "complete": "complete"    # Finished
        }
    )
    
    # Help recovery flow
    graph.add_conditional_edges(
        "help",
        route_next_action,
        {
            "quiz": "quiz",           # Try quiz again after help
            "content": "content",     # Re-deliver content
            "review": "review"        # Check overall progress
        }
    )
    
    # Review flow
    graph.add_conditional_edges(
        "review", 
        route_next_action,
        {
            "continue": "increment",
            "content": "content",
            "complete": "complete"
        }
    )
    
    # Step increment with continuation check (loop control)
    graph.add_conditional_edges(
        "increment",
        should_continue_tutorial,
        {
            "continue": "content",    # Continue to next lesson
            "end": "complete"         # Tutorial finished
        }
    )
    
    # End points
    graph.add_edge("complete", END)
    
    return graph.compile()

# %% [markdown]
# ## 5. Graph Visualization and Execution

# %%
# Create and visualize the graph
tutorial_app = create_tutorial_graph()

# Visualize the graph
try:
    display(Image(tutorial_app.get_graph().draw_mermaid_png()))
    # Save the graph visualization
    with open("../output/enhanced_langgraph_tutorial.png", "wb") as f:
        f.write(tutorial_app.get_graph().draw_mermaid_png())
except Exception as e:
    print(f"Could not generate graph visualization: {e}")

# %% [markdown]
# ## 6. Example Execution - Multiple User Types

# %%
def run_tutorial_example(name: str, user_type: Literal["beginner", "intermediate", "advanced"]):
    """
    Run tutorial example for different user types
    """
    print(f"\n{'='*50}")
    print(f"Running tutorial for {name} ({user_type} level)")
    print(f"{'='*50}")
    
    initial_state = TutorialState(
        name=name,
        user_type=user_type,
        messages=[],
        current_step=1,
        max_attempts=3,
        attempt_count=0,
        topics_covered=[],
        quiz_score=0,
        needs_help=False,
        next_action="continue"
    )
    
    # Run the tutorial
    result = tutorial_app.invoke(initial_state)
    
    # Print conversation
    print(f"\nFinal conversation for {name}:")
    for msg in result["messages"][-3:]:  # Show last 3 messages
        print(f"{msg.type}: {msg.content}\n")
    
    print(f"Final Score: {result.get('quiz_score', 0)}")
    print(f"Topics Covered: {len(result.get('topics_covered', []))}")
    
    return result

# Run examples for different user types
beginner_result = run_tutorial_example("Alice", "beginner")
intermediate_result = run_tutorial_example("Bob", "intermediate") 
advanced_result = run_tutorial_example("Carol", "advanced")

# %% [markdown]
# ## 7. Streaming Example - Real-time Processing

# %%
def stream_tutorial_example():
    """
    Demonstrate streaming capabilities
    """
    print("\n" + "="*50)
    print("Streaming Tutorial Example")
    print("="*50)
    
    initial_state = TutorialState(
        name="David",
        user_type="intermediate",
        messages=[],
        current_step=1,
        max_attempts=3,
        attempt_count=0,
        topics_covered=[],
        quiz_score=0,
        needs_help=False,
        next_action="continue"
    )
    
    # Stream the execution
    step_count = 0
    for step in tutorial_app.stream(initial_state):
        step_count += 1
        print(f"\n--- Step {step_count} ---")
        for node_name, node_output in step.items():
            print(f"Node: {node_name}")
            if "messages" in node_output and node_output["messages"]:
                latest_msg = node_output["messages"][-1]
                print(f"Output: {latest_msg.content[:100]}...")
        
        # Limit output for demo
        if step_count >= 5:
            break

# Run streaming example
stream_tutorial_example()

# %% [markdown]
# ## 8. Key Learning Points
# 
# This enhanced tutorial demonstrates:
# 
# ### Node Concepts
# - **Simple Nodes**: `welcome_node`, `completion_node`
# - **Processing Nodes**: `content_delivery_node`, `quiz_node`
# - **Recovery Nodes**: `help_node`, `review_node`
# - **Control Nodes**: `increment_step`
# 
# ### Edge Types
# - **Simple Edges**: START → welcome, complete → END
# - **Conditional Edges**: All routing decisions
# - **Self-loops**: quiz → quiz (retry mechanism)
# 
# ### Advanced Features
# - **State Management**: Complex state with multiple data types
# - **Conditional Routing**: Dynamic flow based on state
# - **Loops and Cycles**: Tutorial progression with retries
# - **Error Recovery**: Help system and adaptive learning
# - **Streaming**: Real-time step-by-step execution
# 
# ### Graph Patterns
# - **Sequential Flow**: welcome → content → quiz → increment
# - **Conditional Branching**: Multiple paths from each node
# - **Cycle Detection**: Loop control with `should_continue_tutorial`
# - **State Persistence**: Information carried through entire flow
# 
# This structure provides a comprehensive foundation for teaching LangGraph concepts!

# %%

