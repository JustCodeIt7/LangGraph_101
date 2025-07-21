#!/usr/bin/env python3
"""
Example usage of the RAG Agent
This script demonstrates how to use the RAG Agent programmatically
"""

import os
import sys
from pathlib import Path

# Add current directory to path to import rag_agent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_agent import RAGAgent

def main():
    """Demonstrate RAG Agent usage"""
    print("🚀 RAG Agent Example Usage")
    print("=" * 40)
    
    # Initialize the RAG Agent
    try:
        agent = RAGAgent(model_name="llama3.2", persist_directory="./example_db")
        print("✅ RAG Agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize RAG Agent: {e}")
        print("Make sure Ollama is running and llama3.2 model is available")
        return
    
    # Example 1: Load a single document
    print("\n📄 Example 1: Loading a single document")
    
    # Create a sample markdown document for testing
    sample_doc_path = "sample_document.md"
    sample_content = """# Sample Document

## Introduction
This is a sample document for testing the RAG Agent. It contains information about artificial intelligence and machine learning.

## Machine Learning Basics
Machine learning is a subset of artificial intelligence that focuses on developing algorithms that can learn from data. There are three main types:

1. **Supervised Learning**: Uses labeled data to train models
2. **Unsupervised Learning**: Finds patterns in unlabeled data  
3. **Reinforcement Learning**: Learns through interaction with environment

## Deep Learning
Deep learning is a subset of machine learning that uses neural networks with multiple layers. It has been particularly successful in:

- Image recognition
- Natural language processing
- Speech recognition
- Game playing (like AlphaGo)

## Applications
AI and ML are used in many fields:
- Healthcare: Disease diagnosis, drug discovery
- Finance: Fraud detection, algorithmic trading
- Transportation: Autonomous vehicles
- Technology: Recommendation systems, virtual assistants

## Conclusion
The field of AI and ML continues to evolve rapidly, with new breakthroughs happening regularly.
"""
    
    with open(sample_doc_path, 'w') as f:
        f.write(sample_content)
    
    # Load the document
    if agent.load_documents(sample_doc_path):
        print(f"✅ Successfully loaded: {sample_doc_path}")
        print(f"📊 Document chunks in database: {agent.get_document_count()}")
    else:
        print(f"❌ Failed to load: {sample_doc_path}")
        return
    
    # Example 2: Ask questions about the document
    print("\n💬 Example 2: Asking questions")
    
    questions = [
        "What are the three main types of machine learning?",
        "What is deep learning and what is it good for?",
        "How is AI used in healthcare?",
        "What's the difference between supervised and unsupervised learning?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n❓ Question {i}: {question}")
        try:
            response = agent.chat(question)
            print(f"🤖 Response: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Example 3: Demonstrate chat history
    print("\n📚 Example 3: Chat history context")
    print("Asking a follow-up question that references previous context...")
    
    follow_up_question = "Can you give me more details about the first type you mentioned?"
    print(f"\n❓ Follow-up: {follow_up_question}")
    try:
        response = agent.chat(follow_up_question)
        print(f"🤖 Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 4: Clear history and ask again
    print("\n🧹 Example 4: Clearing chat history")
    agent.clear_history()
    print("Chat history cleared!")
    
    # Same follow-up question should now lack context
    print(f"\n❓ Same question after clearing history: {follow_up_question}")
    try:
        response = agent.chat(follow_up_question)
        print(f"🤖 Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 5: Load additional documents
    print("\n📁 Example 5: Loading additional documents")
    
    # Create another sample document
    additional_doc_path = "additional_document.md"
    additional_content = """# Programming Languages

## Python
Python is a high-level programming language known for its simplicity and readability. It's widely used in:
- Data science and machine learning
- Web development
- Automation and scripting
- Scientific computing

## JavaScript
JavaScript is the language of the web. It's used for:
- Frontend web development
- Backend development (Node.js)
- Mobile app development
- Desktop applications

## SQL
SQL (Structured Query Language) is used for managing relational databases. Key operations include:
- SELECT: Retrieve data
- INSERT: Add new data
- UPDATE: Modify existing data
- DELETE: Remove data

## Best Practices
- Write clean, readable code
- Use version control (Git)
- Test your code
- Document your work
- Follow coding standards
"""
    
    with open(additional_doc_path, 'w') as f:
        f.write(additional_content)
    
    if agent.load_documents(additional_doc_path):
        print(f"✅ Successfully loaded additional document: {additional_doc_path}")
        print(f"📊 Total document chunks in database: {agent.get_document_count()}")
    else:
        print(f"❌ Failed to load: {additional_doc_path}")
    
    # Ask questions about the new content
    print("\n❓ Question about new content: What programming languages are mentioned?")
    try:
        response = agent.chat("What programming languages are mentioned and what are they used for?")
        print(f"🤖 Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Ask a question that combines information from both documents
    print("\n❓ Cross-document question: How might Python be used in machine learning?")
    try:
        response = agent.chat("How might Python be used in machine learning based on the information provided?")
        print(f"🤖 Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Cleanup
    print("\n🧹 Cleaning up example files...")
    try:
        os.remove(sample_doc_path)
        os.remove(additional_doc_path)
        print("✅ Example files cleaned up")
    except Exception as e:
        print(f"⚠️  Could not clean up files: {e}")
    
    print("\n🎉 Example completed!")
    print("\nTo run the interactive CLI:")
    print("python rag_agent.py --load /path/to/your/documents")

if __name__ == "__main__":
    main()