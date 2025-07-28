#!/usr/bin/env python3
"""
LangGraph Memory Examples - YouTube Tutorial
============================================

This module demonstrates three different memory patterns in LangGraph using local SQLite databases.
Perfect for YouTube tutorials showing practical memory management without external dependencies.

Examples:
1. Basic SQLite Checkpointing with Conversation Memory
2. Advanced Memory with Custom State and Context Retrieval  
3. Multi-Thread Memory Management with Session Isolation

Requirements:
- langgraph
- langchain-core
- langchain-openai (or langchain-ollama for local models)
- sqlite3 (built-in)

Author: LangGraph Tutorial Series
"""

import os
import sqlite3
import json
import uuid
from datetime import datetime
from typing import Annotated, TypedDict, List, Dict, Any, Optional
from operator import add

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.store.sqlite import SqliteStore

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
# Alternative: from langchain_ollama import ChatOllama


def cleanup_databases():
    """Clean up existing database files for fresh starts."""
    db_files = [
        'conversation_memory.db',
        'advanced_memory.db', 
        'multi_thread_memory.db',
        'context_store.db'
    ]
    
    for db_file in db_files:
        if os.path.exists(db_file):
            os.remove(db_file)
            print(f"🗑️  Cleaned up {db_file}")


# =============================================================================
# EXAMPLE 1: Basic SQLite Checkpointing with Conversation Memory
# =============================================================================

class ConversationState(TypedDict):
    """State schema for basic conversation with message history."""
    messages: Annotated[List[BaseMessage], add_messages]
    user_name: Optional[str]
    topic_count: Annotated[int, add]


def run_example_1_basic_conversation_memory():
    """
    Example 1: Basic SQLite Checkpointing with Conversation Memory
    
    This example demonstrates:
    - SQLite persistence for conversation history
    - Message accumulation across sessions
    - Basic state management with user context
    - Checkpoint retrieval and state history
    """
    print("\n" + "="*60)
    print("📚 EXAMPLE 1: Basic SQLite Conversation Memory")
    print("="*60)
    
    # Initialize LLM (choose one)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    # llm = ChatOllama(model="llama3.2", temperature=0.7)  # Alternative local model
    
    def chatbot_with_memory(state: ConversationState) -> Dict[str, Any]:
        """Chatbot node that maintains conversation context."""
        response = llm.invoke(state["messages"])
        
        # Extract user name from first message if not set
        user_name = state.get("user_name")
        if not user_name and state["messages"]:
            first_message = state["messages"][0]
            if isinstance(first_message, HumanMessage):
                content = first_message.content.lower()
                if "my name is" in content:
                    name_part = content.split("my name is")[-1].strip()
                    user_name = name_part.split()[0].capitalize()
        
        return {
            "messages": [response],
            "user_name": user_name,
            "topic_count": 1  # Increment for each interaction
        }
    
    # Build the graph
    workflow = StateGraph(ConversationState)
    workflow.add_node("chatbot", chatbot_with_memory)
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", END)
    
    # Set up SQLite persistence
    with SqliteSaver.from_conn_string("conversation_memory.db") as checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)
        
        # Thread configuration
        thread_config = {"configurable": {"thread_id": "conversation-1"}}
        
        print("🤖 Starting conversation with memory...")
        
        # First interaction - introduce user
        print("\n--- First Message ---")
        result1 = graph.invoke({
            "messages": [HumanMessage(content="Hi! My name is Alice and I love programming.")],
            "user_name": None,
            "topic_count": 0
        }, thread_config)
        
        print(f"AI: {result1['messages'][-1].content}")
        print(f"User Name Extracted: {result1.get('user_name', 'Not detected')}")
        print(f"Topic Count: {result1['topic_count']}")
        
        # Second interaction - test memory
        print("\n--- Second Message ---")
        result2 = graph.invoke({
            "messages": [HumanMessage(content="What's my name and what do I love?")]
        }, thread_config)
        
        print(f"AI: {result2['messages'][-1].content}")
        print(f"Topic Count: {result2['topic_count']}")
        
        # Third interaction - continue conversation
        print("\n--- Third Message ---")
        result3 = graph.invoke({
            "messages": [HumanMessage(content="Can you recommend a Python library for me?")]
        }, thread_config)
        
        print(f"AI: {result3['messages'][-1].content}")
        print(f"Topic Count: {result3['topic_count']}")
        
        # Display conversation history
        print("\n📜 Full Conversation History:")
        final_state = graph.get_state(thread_config)
        for i, msg in enumerate(final_state.values["messages"]):
            speaker = "👤 Human" if isinstance(msg, HumanMessage) else "🤖 AI"
            print(f"{i+1}. {speaker}: {msg.content[:100]}...")
        
        # Show checkpoint history
        print(f"\n🔍 Total Checkpoints: {len(list(graph.get_state_history(thread_config)))}")
        print(f"Final User Name: {final_state.values.get('user_name', 'Unknown')}")
        print(f"Final Topic Count: {final_state.values['topic_count']}")


# =============================================================================
# EXAMPLE 2: Advanced Memory with Custom State and Context Retrieval
# =============================================================================

class AdvancedMemoryState(TypedDict):
    """Advanced state schema with rich context and metadata."""
    messages: Annotated[List[BaseMessage], add_messages]
    user_profile: Dict[str, Any]
    context_summaries: Annotated[List[str], add]
    conversation_metadata: Dict[str, Any]
    current_intent: Optional[str]


class ContextualMemoryManager:
    """Manages contextual memory with SQLite storage for summaries and user data."""
    
    def __init__(self, db_path: str = "context_store.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for context storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    summary TEXT,
                    keywords TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                )
            """)
            conn.commit()
    
    def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]):
        """Update user profile in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO user_profiles (user_id, profile_data, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (user_id, json.dumps(profile_data)))
            conn.commit()
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Retrieve user profile from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT profile_data FROM user_profiles WHERE user_id = ?",
                (user_id,)
            )
            result = cursor.fetchone()
            return json.loads(result[0]) if result else {}
    
    def add_conversation_summary(self, user_id: str, summary: str, keywords: List[str]):
        """Add conversation summary to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO conversation_summaries (user_id, summary, keywords)
                VALUES (?, ?, ?)
            """, (user_id, summary, json.dumps(keywords)))
            conn.commit()
    
    def get_relevant_summaries(self, user_id: str, keywords: List[str], limit: int = 3) -> List[str]:
        """Retrieve relevant conversation summaries based on keywords."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT summary FROM conversation_summaries 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (user_id, limit))
            return [row[0] for row in cursor.fetchall()]


def run_example_2_advanced_memory_context():
    """
    Example 2: Advanced Memory with Custom State and Context Retrieval
    
    This example demonstrates:
    - Custom SQLite store for user profiles and conversation summaries
    - Intent detection and context-aware responses
    - Rich state management with metadata
    - Context retrieval based on conversation patterns
    """
    print("\n" + "="*60)
    print("🧠 EXAMPLE 2: Advanced Memory with Context Retrieval")
    print("="*60)
    
    # Initialize components
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    memory_manager = ContextualMemoryManager()
    
    def analyze_intent(message: str) -> str:
        """Simple intent analysis (in production, use NLU)."""
        message_lower = message.lower()
        if any(word in message_lower for word in ["recommend", "suggest", "advice"]):
            return "seeking_recommendation"
        elif any(word in message_lower for word in ["learn", "how to", "tutorial"]):
            return "learning_request"
        elif any(word in message_lower for word in ["problem", "issue", "error", "help"]):
            return "troubleshooting"
        else:
            return "general_conversation"
    
    def extract_keywords(message: str) -> List[str]:
        """Extract keywords from message (simplified)."""
        # In production, use proper NLP libraries like spaCy or NLTK
        stop_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were"}
        words = message.lower().split()
        return [word for word in words if len(word) > 3 and word not in stop_words]
    
    def advanced_chatbot(state: AdvancedMemoryState) -> Dict[str, Any]:
        """Advanced chatbot with context awareness and profile management."""
        current_message = state["messages"][-1]
        user_id = state["conversation_metadata"].get("user_id", "unknown")
        
        # Analyze current message
        intent = analyze_intent(current_message.content)
        keywords = extract_keywords(current_message.content)
        
        # Retrieve user profile and relevant context
        user_profile = memory_manager.get_user_profile(user_id)
        relevant_summaries = memory_manager.get_relevant_summaries(user_id, keywords)
        
        # Build context-aware prompt
        context_info = f"""
        User Profile: {json.dumps(user_profile, indent=2)}
        Current Intent: {intent}
        Relevant Past Conversations: {relevant_summaries}
        Keywords: {keywords}
        """
        
        # Create enhanced messages with context
        enhanced_messages = state["messages"] + [
            AIMessage(content=f"[CONTEXT]: {context_info}")
        ]
        
        response = llm.invoke(enhanced_messages[:-1])  # Don't include context in actual conversation
        
        # Update user profile based on conversation
        if not user_profile and "name" in current_message.content.lower():
            # Extract name (simplified)
            content = current_message.content.lower()
            if "my name is" in content:
                name = content.split("my name is")[-1].strip().split()[0].capitalize()
                user_profile["name"] = name
                user_profile["first_interaction"] = datetime.now().isoformat()
        
        # Add interests based on keywords
        if "interests" not in user_profile:
            user_profile["interests"] = []
        user_profile["interests"].extend(keywords[:3])  # Add top 3 keywords as interests
        user_profile["interests"] = list(set(user_profile["interests"]))[:10]  # Limit to 10 unique
        
        # Update profile in database
        memory_manager.update_user_profile(user_id, user_profile)
        
        # Create conversation summary
        summary = f"Intent: {intent}, Keywords: {keywords}, Response type: {type(response).__name__}"
        memory_manager.add_conversation_summary(user_id, summary, keywords)
        
        return {
            "messages": [response],
            "user_profile": user_profile,
            "context_summaries": [summary],
            "conversation_metadata": {
                **state["conversation_metadata"],
                "last_intent": intent,
                "keywords": keywords
            },
            "current_intent": intent
        }
    
    # Build the graph
    workflow = StateGraph(AdvancedMemoryState)
    workflow.add_node("advanced_chatbot", advanced_chatbot)
    workflow.add_edge(START, "advanced_chatbot")
    workflow.add_edge("advanced_chatbot", END)
    
    # Set up persistence
    with SqliteSaver.from_conn_string("advanced_memory.db") as checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)
        
        thread_config = {"configurable": {"thread_id": "advanced-user-1"}}
        user_id = "user_alice_2024"
        
        print("🧠 Starting advanced conversation with contextual memory...")
        
        # First interaction - profile building
        print("\n--- Building User Profile ---")
        result1 = graph.invoke({
            "messages": [HumanMessage(content="Hi! My name is Alice and I'm interested in machine learning and Python programming.")],
            "user_profile": {},
            "context_summaries": [],
            "conversation_metadata": {"user_id": user_id, "session_start": datetime.now().isoformat()},
            "current_intent": None
        }, thread_config)
        
        print(f"AI: {result1['messages'][-1].content}")
        print(f"Detected Intent: {result1['current_intent']}")
        print(f"User Profile: {json.dumps(result1['user_profile'], indent=2)}")
        
        # Second interaction - seeking recommendation
        print("\n--- Seeking Recommendation ---")
        result2 = graph.invoke({
            "messages": [HumanMessage(content="Can you recommend some good machine learning libraries for beginners?")]
        }, thread_config)
        
        print(f"AI: {result2['messages'][-1].content}")
        print(f"Detected Intent: {result2['current_intent']}")
        
        # Third interaction - learning request
        print("\n--- Learning Request ---")
        result3 = graph.invoke({
            "messages": [HumanMessage(content="How do I get started with neural networks in Python?")]
        }, thread_config)
        
        print(f"AI: {result3['messages'][-1].content}")
        print(f"Detected Intent: {result3['current_intent']}")
        
        # Display final state
        final_state = graph.get_state(thread_config)
        print(f"\n📊 Final User Profile: {json.dumps(final_state.values['user_profile'], indent=2)}")
        print(f"💭 Context Summaries: {len(final_state.values['context_summaries'])}")
        print(f"🎯 Last Intent: {final_state.values['current_intent']}")


# =============================================================================
# EXAMPLE 3: Multi-Thread Memory Management with Session Isolation
# =============================================================================

class MultiThreadState(TypedDict):
    """State schema for multi-thread conversations with session isolation."""
    messages: Annotated[List[BaseMessage], add_messages]
    session_info: Dict[str, Any]
    user_preferences: Dict[str, Any]
    interaction_count: Annotated[int, add]


class SessionManager:
    """Manages multiple conversation sessions with isolated memory."""
    
    def __init__(self, db_path: str = "multi_thread_memory.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database for session management."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    session_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS global_user_data (
                    user_id TEXT PRIMARY KEY,
                    preferences TEXT,
                    total_sessions INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def create_session(self, user_id: str, session_type: str = "general") -> str:
        """Create a new session for a user."""
        session_id = f"{user_id}_{session_type}_{uuid.uuid4().hex[:8]}"
        session_data = {
            "type": session_type,
            "created_at": datetime.now().isoformat(),
            "messages_count": 0
        }
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO sessions (session_id, user_id, session_data)
                VALUES (?, ?, ?)
            """, (session_id, user_id, json.dumps(session_data)))
            
            # Update user session count
            conn.execute("""
                INSERT OR REPLACE INTO global_user_data (user_id, total_sessions, updated_at)
                VALUES (?, COALESCE((SELECT total_sessions FROM global_user_data WHERE user_id = ?), 0) + 1, CURRENT_TIMESTAMP)
            """, (user_id, user_id))
            conn.commit()
        
        return session_id
    
    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all sessions for a user."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT session_id, session_data FROM sessions 
                WHERE user_id = ? 
                ORDER BY last_activity DESC
            """, (user_id,))
            
            sessions = []
            for row in cursor.fetchall():
                session_data = json.loads(row[1])
                session_data["session_id"] = row[0]
                sessions.append(session_data)
            return sessions
    
    def update_session_activity(self, session_id: str):
        """Update last activity timestamp for a session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE sessions 
                SET last_activity = CURRENT_TIMESTAMP 
                WHERE session_id = ?
            """, (session_id,))
            conn.commit()


def run_example_3_multi_thread_sessions():
    """
    Example 3: Multi-Thread Memory Management with Session Isolation
    
    This example demonstrates:
    - Multiple isolated conversation threads per user
    - Session-specific memory that doesn't bleed between conversations
    - Global user preferences that persist across sessions
    - Cross-session analytics and insights
    """
    print("\n" + "="*60)
    print("🔀 EXAMPLE 3: Multi-Thread Session Management")
    print("="*60)
    
    # Initialize components
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    session_manager = SessionManager()
    
    def session_aware_chatbot(state: MultiThreadState) -> Dict[str, Any]:
        """Chatbot that maintains session-specific context."""
        session_info = state["session_info"]
        session_id = session_info.get("session_id")
        
        # Update session activity
        if session_id:
            session_manager.update_session_activity(session_id)
        
        # Build context with session info
        session_context = f"""
        Session Type: {session_info.get('type', 'general')}
        Session Messages: {state['interaction_count']}
        User Preferences: {state['user_preferences']}
        """
        
        current_message = state["messages"][-1]
        
        # Add session context to conversation (for AI understanding)
        contextual_messages = [
            AIMessage(content=f"[SESSION_CONTEXT]: {session_context}"),
            *state["messages"]
        ]
        
        response = llm.invoke(contextual_messages[1:])  # Exclude context from actual conversation
        
        # Update user preferences based on conversation
        preferences = state["user_preferences"].copy()
        
        # Simple preference extraction (in production, use more sophisticated NLP)
        content_lower = current_message.content.lower()
        if "prefer" in content_lower or "like" in content_lower:
            # Extract preferences (simplified)
            if "technical" in content_lower:
                preferences["communication_style"] = "technical"
            elif "simple" in content_lower or "easy" in content_lower:
                preferences["communication_style"] = "simple"
        
        return {
            "messages": [response],
            "session_info": {
                **session_info,
                "last_message_at": datetime.now().isoformat()
            },
            "user_preferences": preferences,
            "interaction_count": 1
        }
    
    # Build the graph
    workflow = StateGraph(MultiThreadState)
    workflow.add_node("session_chatbot", session_aware_chatbot)
    workflow.add_edge(START, "session_chatbot")
    workflow.add_edge("session_chatbot", END)
    
    # Set up persistence with separate databases for each session
    user_id = "alice_multi"
    
    print("🔀 Demonstrating multiple isolated conversation sessions...\n")
    
    # SESSION 1: Technical Discussion
    print("--- SESSION 1: Technical Programming Discussion ---")
    session1_id = session_manager.create_session(user_id, "technical")
    
    with SqliteSaver.from_conn_string("session1_memory.db") as checkpointer1:
        graph1 = workflow.compile(checkpointer=checkpointer1)
        thread_config1 = {"configurable": {"thread_id": session1_id}}
        
        result1a = graph1.invoke({
            "messages": [HumanMessage(content="I prefer technical explanations. Can you explain how neural networks work?")],
            "session_info": {"session_id": session1_id, "type": "technical"},
            "user_preferences": {"communication_style": "balanced"},
            "interaction_count": 0
        }, thread_config1)
        
        print(f"🤖 AI (Technical): {result1a['messages'][-1].content[:150]}...")
        print(f"Session Type: {result1a['session_info']['type']}")
        print(f"Preferences: {result1a['user_preferences']}")
        
        result1b = graph1.invoke({
            "messages": [HumanMessage(content="What about backpropagation algorithms?")]
        }, thread_config1)
        
        print(f"🤖 AI (Technical): {result1b['messages'][-1].content[:150]}...")
        print(f"Interaction Count: {result1b['interaction_count']}")
    
    # SESSION 2: Casual Learning
    print("\n--- SESSION 2: Casual Learning Discussion ---")
    session2_id = session_manager.create_session(user_id, "casual")
    
    with SqliteSaver.from_conn_string("session2_memory.db") as checkpointer2:
        graph2 = workflow.compile(checkpointer=checkpointer2)
        thread_config2 = {"configurable": {"thread_id": session2_id}}
        
        result2a = graph2.invoke({
            "messages": [HumanMessage(content="I like simple explanations. What's machine learning in easy terms?")],
            "session_info": {"session_id": session2_id, "type": "casual"},
            "user_preferences": {"communication_style": "balanced"},
            "interaction_count": 0
        }, thread_config2)
        
        print(f"🤖 AI (Casual): {result2a['messages'][-1].content[:150]}...")
        print(f"Session Type: {result2a['session_info']['type']}")
        print(f"Preferences: {result2a['user_preferences']}")
        
        result2b = graph2.invoke({
            "messages": [HumanMessage(content="Can you give me a fun example?")]
        }, thread_config2)
        
        print(f"🤖 AI (Casual): {result2b['messages'][-1].content[:150]}...")
        print(f"Interaction Count: {result2b['interaction_count']}")
    
    # SESSION 3: Problem Solving
    print("\n--- SESSION 3: Problem Solving Session ---")
    session3_id = session_manager.create_session(user_id, "problem_solving")
    
    with SqliteSaver.from_conn_string("session3_memory.db") as checkpointer3:
        graph3 = workflow.compile(checkpointer=checkpointer3)
        thread_config3 = {"configurable": {"thread_id": session3_id}}
        
        result3a = graph3.invoke({
            "messages": [HumanMessage(content="I'm having trouble with my Python code. Can you help debug it?")],
            "session_info": {"session_id": session3_id, "type": "problem_solving"},
            "user_preferences": {"communication_style": "balanced"},
            "interaction_count": 0
        }, thread_config3)
        
        print(f"🤖 AI (Problem Solving): {result3a['messages'][-1].content[:150]}...")
        print(f"Session Type: {result3a['session_info']['type']}")
    
    # Display session analytics
    print("\n📊 USER SESSION ANALYTICS:")
    user_sessions = session_manager.get_user_sessions(user_id)
    print(f"Total Sessions: {len(user_sessions)}")
    
    for session in user_sessions:
        print(f"  - {session['session_id']}: {session['type']} (Created: {session['created_at'][:19]})")
    
    print(f"\n✨ Each session maintains isolated memory while sharing user preferences!")


# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main():
    """Run all memory examples with proper database cleanup."""
    print("🚀 LangGraph Memory Examples - YouTube Tutorial")
    print("=" * 60)
    print("This tutorial demonstrates three different memory patterns:")
    print("1. Basic SQLite Checkpointing with Conversation Memory")
    print("2. Advanced Memory with Custom State and Context Retrieval")
    print("3. Multi-Thread Memory Management with Session Isolation")
    print("=" * 60)
    
    # Clean up any existing databases
    cleanup_databases()
    
    try:
        # Run Example 1: Basic Conversation Memory
        run_example_1_basic_conversation_memory()
        
        # Run Example 2: Advanced Memory with Context
        run_example_2_advanced_memory_context()
        
        # Run Example 3: Multi-Thread Sessions
        run_example_3_multi_thread_sessions()
        
        print("\n" + "="*60)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("📁 Database files created:")
        print("  - conversation_memory.db (Example 1)")
        print("  - advanced_memory.db (Example 2)") 
        print("  - context_store.db (Example 2 - Context storage)")
        print("  - multi_thread_memory.db (Example 3 - Session management)")
        print("  - session1_memory.db, session2_memory.db, session3_memory.db (Example 3)")
        print("\n🎥 Perfect for YouTube tutorial demonstration!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure you have all required dependencies installed:")
        print("  pip install langgraph langchain-core langchain-openai")


if __name__ == "__main__":
    main()