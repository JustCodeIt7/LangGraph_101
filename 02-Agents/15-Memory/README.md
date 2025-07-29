# LangGraph Memory Examples - YouTube Tutorial

This directory contains three comprehensive examples demonstrating different memory patterns in LangGraph using local SQLite databases. These examples are designed for YouTube tutorials and require no external database dependencies.

## 📋 Examples Overview

### 1. Basic SQLite Checkpointing with Conversation Memory
**File**: `15-memory.py` - Example 1
- **Demonstrates**: Basic conversation persistence with SQLite
- **Features**:
  - Message history accumulation across sessions
  - User context extraction and persistence
  - Basic state management with counters
  - Checkpoint retrieval and state history

### 2. Advanced Memory with Custom State and Context Retrieval
**File**: `15-memory.py` - Example 2
- **Demonstrates**: Advanced contextual memory management
- **Features**:
  - Custom SQLite store for user profiles and conversation summaries
  - Intent detection and context-aware responses
  - Rich state management with metadata
  - Context retrieval based on conversation patterns
  - Keyword extraction and semantic storage

### 3. Multi-Thread Memory Management with Session Isolation
**File**: `15-memory.py` - Example 3
- **Demonstrates**: Multiple isolated conversation sessions
- **Features**:
  - Session-specific memory that doesn't bleed between conversations
  - Global user preferences that persist across sessions
  - Cross-session analytics and insights
  - Separate SQLite databases for complete isolation

## 🛠️ Requirements

Install the required dependencies:

```bash
pip install langgraph langchain-core langchain-openai
```

**Alternative for local models:**
```bash
pip install langgraph langchain-core langchain-ollama
```

## 🚀 Usage

### Run All Examples
```bash
cd 02-Agents/15-Memory
python 15-memory.py
```

### Run Individual Examples
You can also import and run specific examples:

```python
from 15-memory import (
    run_example_1_basic_conversation_memory,
    run_example_2_advanced_memory_context,
    run_example_3_multi_thread_sessions
)

# Run individual examples
run_example_1_basic_conversation_memory()
run_example_2_advanced_memory_context()
run_example_3_multi_thread_sessions()
```

## 📊 Database Files Created

After running the examples, you'll find these SQLite database files:

- `conversation_memory.db` - Example 1: Basic conversation history
- `advanced_memory.db` - Example 2: Advanced state management
- `context_store.db` - Example 2: Context and user profile storage
- `multi_thread_memory.db` - Example 3: Session management
- `session1_memory.db`, `session2_memory.db`, `session3_memory.db` - Example 3: Individual session databases

## 🎯 Key Learning Points

### Example 1: Basic Memory
- How to use `SqliteSaver` for persistent conversations
- Message accumulation with `add_messages`
- State retrieval and checkpoint history
- Basic user context extraction

### Example 2: Advanced Memory
- Custom SQLite stores for complex data
- Context-aware conversation management
- Intent detection and keyword extraction
- User profile building and persistence

### Example 3: Multi-Thread Sessions
- Session isolation and management
- Cross-session user preferences
- Multiple database management
- Session analytics and insights

## 🔧 Configuration

### LLM Provider Setup

**OpenAI (Default):**
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
```

**Local Ollama Alternative:**
```python
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.2", temperature=0.7)
```

Make sure to set your `OPENAI_API_KEY` environment variable if using OpenAI.

## 📚 Code Structure

```
15-memory.py
├── ConversationState          # Basic conversation state schema
├── AdvancedMemoryState        # Advanced state with rich context
├── MultiThreadState           # Multi-session state management
├── ContextualMemoryManager    # Advanced memory management class
├── SessionManager            # Multi-session management class
└── Main execution functions   # Individual example runners
```

## 🎥 YouTube Tutorial Features

These examples are specifically designed for YouTube tutorials:

1. **Self-contained**: No external databases required
2. **Progressive complexity**: From basic to advanced patterns
3. **Visual output**: Clear console output showing memory persistence
4. **Database inspection**: Easy to examine created SQLite files
5. **Modular design**: Each example can be run independently

## 🔍 Debugging and Inspection

You can inspect the created SQLite databases using any SQLite browser or command line:

```bash
# Example: Inspect conversation memory
sqlite3 conversation_memory.db
.tables
.schema
SELECT * FROM checkpoint_blobs LIMIT 5;
```

## 🤝 Contributing

Feel free to extend these examples with additional memory patterns:
- Vector store integration
- Redis-based caching
- Custom memory retrieval strategies
- Memory compression techniques

## 📖 Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Memory Patterns](https://langchain-ai.github.io/langgraph/concepts/memory/)
- [SQLite Documentation](https://docs.python.org/3/library/sqlite3.html)
