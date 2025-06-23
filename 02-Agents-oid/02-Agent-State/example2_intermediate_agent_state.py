"""
Example 2: Intermediate Agent State
==================================

This example builds on the basic state by adding more sophisticated
state management including user preferences, conversation context,
and conditional logic based on state values.

Key Concepts:
- Multiple state fields with different data types
- Conditional node execution based on state
- User preference tracking and adaptation
- Context-aware responses
- State-driven decision making
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 1. Define Intermediate Agent State
class IntermediateAgentState(TypedDict):
    messages: Annotated[List, add_messages]
    user_name: str
    turn_count: int
    user_mood: str                    # New: track user's mood
    conversation_topic: Optional[str] # New: current topic
    response_style: str              # New: formal, casual, friendly
    help_count: int                  # New: how many times user asked for help

# 2. Define Intermediate Nodes
def user_analyzer_node(state: IntermediateAgentState):
    """
    Analyzes the user's latest message to understand mood and context.
    Shows how to read messages and update multiple state fields.
    """
    print("---ANALYZING USER INPUT---")
    
    messages = state["messages"]
    if not messages:
        return state
    
    latest_message = messages[-1]
    if isinstance(latest_message, HumanMessage):
        user_text = latest_message.content.lower()
        
        # Simple mood detection
        if any(word in user_text for word in ["happy", "great", "awesome", "fantastic"]):
            mood = "positive"
        elif any(word in user_text for word in ["sad", "bad", "terrible", "awful"]):
            mood = "negative" 
        elif any(word in user_text for word in ["help", "confused", "don't understand"]):
            mood = "confused"
        else:
            mood = "neutral"
        
        # Topic detection
        topic = None
        if any(word in user_text for word in ["weather", "temperature", "rain", "sunny"]):
            topic = "weather"
        elif any(word in user_text for word in ["work", "job", "career", "office"]):
            topic = "work"
        elif any(word in user_text for word in ["food", "eat", "hungry", "restaurant"]):
            topic = "food"
        elif any(word in user_text for word in ["help", "how", "what", "explain"]):
            topic = "assistance"
        
        # Update help counter
        help_count = state.get("help_count", 0)
        if "help" in user_text:
            help_count += 1
        
        print(f"Detected mood: {mood}, topic: {topic}, help requests: {help_count}")
        
        return {
            "user_mood": mood,
            "conversation_topic": topic,
            "help_count": help_count
        }
    
    return state

def style_adapter_node(state: IntermediateAgentState):
    """
    Adapts response style based on user mood and interaction history.
    Demonstrates conditional logic based on state values.
    """
    print("---ADAPTING RESPONSE STYLE---")
    
    mood = state.get("user_mood", "neutral")
    help_count = state.get("help_count", 0)
    turn_count = state.get("turn_count", 0)
    
    # Determine response style based on state
    if mood == "confused" or help_count > 2:
        style = "helpful"
    elif mood == "positive":
        style = "enthusiastic"
    elif mood == "negative":
        style = "supportive"
    elif turn_count < 3:
        style = "friendly"
    else:
        style = "casual"
    
    print(f"Adapted style: {style} (based on mood: {mood}, help: {help_count}, turns: {turn_count})")
    
    return {"response_style": style}

def context_aware_llm_node(state: IntermediateAgentState):
    """
    Generates responses that are aware of conversation context and user state.
    Shows how to use multiple state fields to create adaptive behavior.
    """
    print("---GENERATING CONTEXT-AWARE RESPONSE---")
    
    messages = state["messages"]
    user_name = state.get("user_name", "User")
    mood = state.get("user_mood", "neutral")
    topic = state.get("conversation_topic")
    style = state.get("response_style", "friendly")
    turn_count = state.get("turn_count", 0)
    help_count = state.get("help_count", 0)
    
    # Create context-aware system prompt
    style_instructions = {
        "helpful": "Be extra patient and provide detailed, clear explanations.",
        "enthusiastic": "Match their positive energy with enthusiasm and excitement.",
        "supportive": "Be empathetic and supportive, offering encouragement.",
        "friendly": "Be warm and approachable with a friendly tone.",
        "casual": "Use a relaxed, conversational style."
    }
    
    system_prompt = f"""You are talking to {user_name}. This is turn {turn_count} of your conversation.

Context:
- User's current mood: {mood}
- Current topic: {topic or 'general conversation'}
- Response style: {style}
- Times user asked for help: {help_count}

Instructions: {style_instructions.get(style, 'Be natural and helpful.')}

Remember this context when responding and adapt your tone accordingly."""
    
    # Add system message with context
    context_messages = [SystemMessage(content=system_prompt)] + messages
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    response = llm.invoke(context_messages)
    
    print(f"Context-aware response generated (Style: {style})")
    
    return {"messages": [AIMessage(content=response.content)]}

def turn_tracker_node(state: IntermediateAgentState):
    """
    Updates turn count and performs turn-based logic.
    """
    print("---TRACKING CONVERSATION TURNS---")
    
    current_count = state.get("turn_count", 0)
    new_count = current_count + 1
    
    # Special logic for certain turns
    if new_count == 5:
        print("Milestone: 5 turns reached! User seems engaged.")
    elif new_count == 10:
        print("Milestone: 10 turns! Long conversation detected.")
    
    return {"turn_count": new_count}

# 3. Build Intermediate Graph
workflow = StateGraph(IntermediateAgentState)

# Add all nodes
workflow.add_node("analyzer", user_analyzer_node)
workflow.add_node("style_adapter", style_adapter_node)
workflow.add_node("turn_tracker", turn_tracker_node)
workflow.add_node("context_llm", context_aware_llm_node)

# Define the flow with multiple processing steps
workflow.set_entry_point("analyzer")
workflow.add_edge("analyzer", "style_adapter")
workflow.add_edge("style_adapter", "turn_tracker") 
workflow.add_edge("turn_tracker", "context_llm")
workflow.add_edge("context_llm", END)

# 4. Compile Graph
intermediate_agent = workflow.compile()

# 5. Enhanced Interactive Loop
if __name__ == "__main__":
    print("Intermediate Agent State Example")
    print("=" * 35)
    print("This agent adapts its behavior based on:")
    print("• Your mood and tone")
    print("• Conversation topics")
    print("• How often you ask for help")
    print("• Turn count and interaction history\n")
    
    # Get user name
    user_name = input("What's your name? ") or "User"
    
    # Initialize with intermediate state
    current_state = {
        "messages": [],
        "user_name": user_name,
        "turn_count": 0,
        "user_mood": "neutral",
        "conversation_topic": None,
        "response_style": "friendly",
        "help_count": 0
    }
    
    print(f"\nHi {user_name}! I'm your adaptive assistant.")
    print("Type 'status' to see what I know about our conversation.")
    print("Type 'exit' to quit.\n")
    
    while True:
        # Show current turn in prompt
        user_input = input(f"Turn {current_state['turn_count'] + 1} - {user_name}: ")
        
        if user_input.lower() == "exit":
            print(f"Goodbye {user_name}!")
            break
        
        if user_input.lower() == "status":
            print(f"\n--- CONVERSATION STATUS ---")
            print(f"Name: {current_state['user_name']}")
            print(f"Turn: {current_state['turn_count']}")
            print(f"Current mood: {current_state['user_mood']}")
            print(f"Topic: {current_state['conversation_topic'] or 'None'}")
            print(f"Response style: {current_state['response_style']}")
            print(f"Help requests: {current_state['help_count']}")
            print(f"Messages in history: {len(current_state['messages'])}")
            continue
            
        if not user_input.strip():
            print("Please enter some text.")
            continue
        
        # Add user message
        current_state["messages"] = [HumanMessage(content=user_input)]
        
        print("\n--- Processing (Multiple Analysis Steps) ---")
        
        # Run through the intermediate workflow
        result = intermediate_agent.invoke(current_state)
        
        # Update state with all results
        current_state.update(result)
        
        # Display AI response
        if result["messages"]:
            ai_message = result["messages"][-1]
            if isinstance(ai_message, AIMessage):
                print(f"\nAI ({current_state['response_style']} style): {ai_message.content}")
    
    print(f"\n--- CONVERSATION SUMMARY ---")
    print(f"Total turns: {current_state['turn_count']}")
    print(f"Final mood: {current_state['user_mood']}")
    print(f"Last topic: {current_state['conversation_topic']}")
    print(f"Help requests: {current_state['help_count']}")
    print(f"Final response style: {current_state['response_style']}")