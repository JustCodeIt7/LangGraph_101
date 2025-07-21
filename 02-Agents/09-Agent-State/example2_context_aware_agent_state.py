"""
Example 2: Context-Aware Agent State
====================================

This example demonstrates how to use LangGraph agent state to maintain rich
conversation context, user preferences, and adaptive behavior based on interaction history.

Key Features:
- User profile and preferences tracking
- Conversation context and topic management
- Emotional state awareness
- Adaptive communication style
- Memory of past interactions and learned preferences
"""

import contextlib
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List, Dict, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from datetime import datetime
from enum import Enum

class CommunicationStyle(Enum):
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    FRIENDLY = "friendly"
    CONCISE = "concise"

class EmotionalState(Enum):
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    FRUSTRATED = "frustrated"
    CONFUSED = "confused"
    EXCITED = "excited"

class UserProfile(TypedDict):
    name: str
    preferred_style: CommunicationStyle
    expertise_level: str  # "beginner", "intermediate", "expert"
    interests: List[str]
    time_zone: Optional[str]
    language_preference: str
    interaction_count: int

class ConversationContext(TypedDict):
    current_topic: str
    topic_depth: int  # How deep we've gone into current topic
    related_topics: List[str]
    context_keywords: List[str]
    conversation_flow: List[str]  # Track how conversation has evolved

class ContextAwareAgentState(TypedDict):
    messages: Annotated[List, add_messages]
    user_profile: UserProfile
    conversation_context: ConversationContext
    emotional_state: EmotionalState
    learned_preferences: Dict[str, any]  # Dynamic preferences learned over time
    session_memory: Dict[str, any]  # Short-term memory for current session
    long_term_insights: List[str]  # Patterns learned about user over time
    adaptation_history: List[Dict[str, any]]  # Track how agent has adapted

def profile_analyzer_node(state: ContextAwareAgentState):
    """
    Analyzes user messages to update profile and detect preferences.
    """
    print("---ANALYZING USER PROFILE---")

    messages = state['messages']
    if not messages:
        return state

    latest_message = messages[-1]
    if not isinstance(latest_message, HumanMessage):
        return state

    user_input = latest_message.content.lower()

    # Update interaction count
    updated_profile = state["user_profile"].copy()
    updated_profile["interaction_count"] += 1

    # Detect communication style preference
    if any(word in user_input for word in ["please", "thank you", "could you"]):
        if updated_profile["preferred_style"] != CommunicationStyle.FORMAL:
            updated_profile["preferred_style"] = CommunicationStyle.FORMAL
            print('Detected formal communication preference')
    elif any(word in user_input for word in ["yo", "hey", "sup", "cool", "awesome"]):
        if updated_profile["preferred_style"] != CommunicationStyle.CASUAL:
            updated_profile["preferred_style"] = CommunicationStyle.CASUAL
            print('Detected casual communication preference')

    # Detect expertise level
    technical_terms = ["algorithm", "api", "database", "framework", "architecture", "optimization"]
    if any(term in user_input for term in technical_terms) and updated_profile['expertise_level'] == 'beginner':
        updated_profile['expertise_level'] = 'intermediate'
        print('Updated expertise level to intermediate')

    # Extract interests/topics
    interest_keywords = ["interested in", "like", "love", "enjoy", "passion", "hobby"]
    for keyword in interest_keywords:
        if keyword in user_input:
            # Simple interest extraction (in real implementation, use NLP)
            words = user_input.split()
            with contextlib.suppress(ValueError):
                idx = words.index(keyword.split()[-1])
                if idx + 1 < len(words):
                    potential_interest = words[idx + 1]
                    if potential_interest not in updated_profile["interests"]:
                        updated_profile["interests"].append(potential_interest)
                        print(f'Added interest: {potential_interest}')
    return {"user_profile": updated_profile}

def context_tracker_node(state: ContextAwareAgentState):
    """
    Tracks conversation context and emotional state.
    """
    print("---TRACKING CONVERSATION CONTEXT---")
    
    messages = state["messages"]
    if not messages:
        return state
    
    latest_message = messages[-1]
    if not isinstance(latest_message, HumanMessage):
        return state
    
    user_input = latest_message.content.lower()
    
    # Update conversation context
    updated_context = state["conversation_context"].copy()
    
    # Simple topic detection (in real implementation, use topic modeling)
    topics = {
        "programming": ["code", "programming", "development", "software", "algorithm"],
        "career": ["job", "career", "work", "interview", "salary", "promotion"],
        "learning": ["learn", "study", "course", "tutorial", "education", "skill"],
        "personal": ["personal", "life", "family", "relationship", "health"],
        "technology": ["technology", "tech", "ai", "machine learning", "data science"]
    }
    
    detected_topic = None
    for topic, keywords in topics.items():
        if any(keyword in user_input for keyword in keywords):
            detected_topic = topic
            break
    
    if detected_topic:
        if updated_context["current_topic"] != detected_topic:
            # Topic change
            updated_context["topic_depth"] = 1
            updated_context["current_topic"] = detected_topic
            print(f"Topic changed to: {detected_topic}")
        else:
            # Same topic, going deeper
            updated_context["topic_depth"] += 1
            print(f"Deepening {detected_topic} discussion (depth: {updated_context['topic_depth']})")
    
    # Detect emotional state
    emotional_indicators = {
        EmotionalState.FRUSTRATED: ["frustrated", "annoying", "difficult", "hate", "angry"],
        EmotionalState.EXCITED: ["excited", "amazing", "awesome", "love", "fantastic"],
        EmotionalState.CONFUSED: ["confused", "don't understand", "unclear", "help"],
        EmotionalState.POSITIVE: ["good", "great", "thanks", "helpful", "perfect"]
    }
    
    detected_emotion = EmotionalState.NEUTRAL
    for emotion, indicators in emotional_indicators.items():
        if any(indicator in user_input for indicator in indicators):
            detected_emotion = emotion
            break
    
    print(f"Detected emotional state: {detected_emotion.value}")
    
    return {
        "conversation_context": updated_context,
        "emotional_state": detected_emotion
    }

def adaptive_response_node(state: ContextAwareAgentState):
    """
    Generates contextually aware responses adapted to user profile and state.
    """
    print("---GENERATING ADAPTIVE RESPONSE---")
    
    messages = state["messages"]
    user_profile = state["user_profile"]
    context = state["conversation_context"]
    emotional_state = state["emotional_state"]
    
    # Build adaptive system prompt
    style_instructions = {
        CommunicationStyle.FORMAL: "Use formal, respectful language with proper grammar.",
        CommunicationStyle.CASUAL: "Use casual, friendly language with informal expressions.",
        CommunicationStyle.TECHNICAL: "Use precise technical terminology and detailed explanations.",
        CommunicationStyle.FRIENDLY: "Be warm, encouraging, and personable in your responses.",
        CommunicationStyle.CONCISE: "Keep responses brief and to the point."
    }
    
    expertise_instructions = {
        "beginner": "Explain concepts simply, avoid jargon, provide examples.",
        "intermediate": "Provide detailed explanations with some technical depth.",
        "expert": "Use technical language freely, focus on advanced concepts."
    }
    
    emotional_adaptations = {
        EmotionalState.FRUSTRATED: "Be patient and supportive. Offer step-by-step help.",
        EmotionalState.EXCITED: "Match their enthusiasm and provide engaging information.",
        EmotionalState.CONFUSED: "Clarify concepts clearly and ask if they need more explanation.",
        EmotionalState.POSITIVE: "Maintain the positive tone and continue being helpful."
    }
    
    system_prompt = f"""
    You are an adaptive AI assistant talking to {user_profile['name']}.
    
    User Profile:
    - Communication Style: {user_profile['preferred_style'].value}
    - Expertise Level: {user_profile['expertise_level']}
    - Interests: {', '.join(user_profile['interests'])}
    - Interaction Count: {user_profile['interaction_count']}
    
    Current Context:
    - Topic: {context['current_topic']}
    - Topic Depth: {context['topic_depth']}
    - Emotional State: {emotional_state.value}
    
    Adaptation Instructions:
    - Style: {style_instructions.get(user_profile['preferred_style'], 'Be natural and helpful.')}
    - Expertise: {expertise_instructions.get(user_profile['expertise_level'], 'Adapt to user level.')}
    - Emotional: {emotional_adaptations.get(emotional_state, 'Respond naturally.')}
    
    Remember their interests and adapt your examples and explanations accordingly.
    Build on the conversation context and acknowledge their emotional state appropriately.
    """
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Create adaptive message context
    context_messages = [SystemMessage(content=system_prompt)] + messages
    
    response = llm.invoke(context_messages)
    print(f"Adaptive Response Generated (Style: {user_profile['preferred_style'].value})")
    
    # Track adaptation
    adaptation_record = {
        "timestamp": datetime.now().isoformat(),
        "style_used": user_profile['preferred_style'].value,
        "expertise_level": user_profile['expertise_level'],
        "emotional_state": emotional_state.value,
        "topic": context['current_topic']
    }
    
    updated_adaptation_history = state["adaptation_history"].copy()
    updated_adaptation_history.append(adaptation_record)
    
    return {
        "messages": [AIMessage(content=response.content)],
        "adaptation_history": updated_adaptation_history
    }

# Build the graph
workflow = StateGraph(ContextAwareAgentState)

workflow.add_node("profiler", profile_analyzer_node)
workflow.add_node("context_tracker", context_tracker_node)
workflow.add_node("adaptive_responder", adaptive_response_node)

# Set up the flow
workflow.set_entry_point("profiler")
workflow.add_edge("profiler", "context_tracker")
workflow.add_edge("context_tracker", "adaptive_responder")
workflow.add_edge("adaptive_responder", END)

# Compile the graph
context_aware_agent = workflow.compile()

if __name__ == "__main__":
    print("Starting Context-Aware Agent...")
    
    # Get initial user info
    user_name = input("What's your name? ") or "User"
    
    # Initialize state
    initial_state: ContextAwareAgentState = {
        "messages": [],
        "user_profile": {
            "name": user_name,
            "preferred_style": CommunicationStyle.FRIENDLY,
            "expertise_level": "intermediate",
            "interests": [],
            "time_zone": None,
            "language_preference": "english",
            "interaction_count": 0
        },
        "conversation_context": {
            "current_topic": "general",
            "topic_depth": 0,
            "related_topics": [],
            "context_keywords": [],
            "conversation_flow": []
        },
        "emotional_state": EmotionalState.NEUTRAL,
        "learned_preferences": {},
        "session_memory": {},
        "long_term_insights": [],
        "adaptation_history": []
    }
    
    current_state = initial_state
    
    print(f"\nHello {user_name}! I'm your context-aware assistant.")
    print("I'll adapt to your communication style and preferences as we chat.")
    print("Type 'profile' to see what I've learned about you, or 'exit' to quit.\n")
    
    while True:
        user_input = input(f"{user_name}: ")
        if user_input.lower() == "exit":
            break
        
        if user_input.lower() == "profile":
            profile = current_state["user_profile"]
            context = current_state["conversation_context"]
            print(f"\n--- YOUR PROFILE ---")
            print(f"Name: {profile['name']}")
            print(f"Communication Style: {profile['preferred_style'].value}")
            print(f"Expertise Level: {profile['expertise_level']}")
            print(f"Interests: {', '.join(profile['interests']) or 'None detected yet'}")
            print(f"Interactions: {profile['interaction_count']}")
            print(f"Current Topic: {context['current_topic']}")
            print(f"Emotional State: {current_state['emotional_state'].value}")
            continue
        
        # Add user message to state
        current_state["messages"] = [HumanMessage(content=user_input)]
        
        # Run the graph
        result = context_aware_agent.invoke(current_state)
        
        # Update current state with results
        current_state.update(result)
        
        # Display AI response
        if "messages" in result and result["messages"]:
            ai_message = result["messages"][-1]
            if isinstance(ai_message, AIMessage):
                print(f"Assistant: {ai_message.content}")
    
    print(f"\n--- SESSION SUMMARY ---")
    profile = current_state["user_profile"]
    print(f"Total Interactions: {profile['interaction_count']}")
    print(f"Learned Interests: {', '.join(profile['interests'])}")
    print(f"Final Communication Style: {profile['preferred_style'].value}")
    print(f"Adaptations Made: {len(current_state['adaptation_history'])}")