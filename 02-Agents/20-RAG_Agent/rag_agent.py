#!/usr/bin/env python3
"""
RAG Agent with LangGraph - Chat with Documents
A CLI application that allows chatting with PDF and markdown files using:
- Ollama Llama3.2 as the LLM
- ChromaDB as the vector database
- LangGraph for agent orchestration
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, TypedDict
import logging

# Core dependencies
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# LangGraph imports
from langgraph.graph import Graph, StateGraph
from langgraph.graph.message import MessageGraph
from langgraph.prebuilt import ToolExecutor
from langgraph.checkpoint.sqlite import SqliteSaver

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGState(TypedDict):
    """State for the RAG agent"""
    query: str
    documents: List[Document]
    retrieved_docs: List[Document]
    response: str
    chat_history: List[Dict[str, str]]
    vector_store: Optional[Any]
    error: Optional[str]

class RAGAgent:
    """RAG Agent using LangGraph for document chat functionality"""
    
    def __init__(self, model_name: str = "llama3.2", persist_directory: str = "./chroma_db"):
        """
        Initialize the RAG Agent
        
        Args:
            model_name: Ollama model name (default: llama3.2)
            persist_directory: Directory to persist ChromaDB
        """
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.vector_store = None
        self.chat_history = []
        
        # Initialize LLM and embeddings
        try:
            self.llm = Ollama(model=model_name, temperature=0.7)
            self.embeddings = OllamaEmbeddings(model=model_name)
            logger.info(f"Initialized Ollama with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            raise
        
        # Text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Create the RAG graph
        self.graph = self._create_rag_graph()
    
    def _create_rag_graph(self) -> StateGraph:
        """Create the LangGraph workflow for RAG"""
        
        # Define the workflow
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("load_documents", self._load_documents_node)
        workflow.add_node("retrieve_documents", self._retrieve_documents_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Define edges
        workflow.set_entry_point("load_documents")
        workflow.add_edge("load_documents", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "generate_response")
        workflow.add_edge("generate_response", "__end__")
        workflow.add_edge("handle_error", "__end__")
        
        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "load_documents",
            self._check_for_errors,
            {
                "error": "handle_error",
                "continue": "retrieve_documents"
            }
        )
        
        return workflow.compile()
    
    def _check_for_errors(self, state: RAGState) -> str:
        """Check if there are any errors in the state"""
        if state.get("error"):
            return "error"
        return "continue"
    
    def _load_documents_node(self, state: RAGState) -> RAGState:
        """Load and process documents if vector store is not initialized"""
        try:
            if self.vector_store is None:
                logger.info("Vector store not initialized. Please load documents first.")
                state["error"] = "No documents loaded. Please load documents using load_documents() method."
            else:
                state["vector_store"] = self.vector_store
        except Exception as e:
            logger.error(f"Error in load_documents_node: {e}")
            state["error"] = str(e)
        
        return state
    
    def _retrieve_documents_node(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents based on the query"""
        try:
            if state["vector_store"] and state["query"]:
                # Retrieve relevant documents
                retriever = state["vector_store"].as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                )
                retrieved_docs = retriever.get_relevant_documents(state["query"])
                state["retrieved_docs"] = retrieved_docs
                logger.info(f"Retrieved {len(retrieved_docs)} relevant documents")
            else:
                state["retrieved_docs"] = []
        except Exception as e:
            logger.error(f"Error in retrieve_documents_node: {e}")
            state["error"] = str(e)
        
        return state
    
    def _generate_response_node(self, state: RAGState) -> RAGState:
        """Generate response using LLM and retrieved documents"""
        try:
            # Create prompt template
            prompt_template = ChatPromptTemplate.from_template("""
            You are a helpful AI assistant that answers questions based on the provided context.
            Use the following context to answer the user's question. If you cannot find the answer
            in the context, say so clearly.
            
            Context:
            {context}
            
            Chat History:
            {chat_history}
            
            Question: {question}
            
            Answer:
            """)
            
            # Prepare context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in state.get("retrieved_docs", [])])
            
            # Prepare chat history
            chat_history = "\n".join([
                f"Human: {entry['human']}\nAssistant: {entry['assistant']}"
                for entry in state.get("chat_history", [])[-3:]  # Last 3 exchanges
            ])
            
            # Create the chain
            chain = (
                {
                    "context": lambda x: context,
                    "chat_history": lambda x: chat_history,
                    "question": RunnablePassthrough()
                }
                | prompt_template
                | self.llm
                | StrOutputParser()
            )
            
            # Generate response
            response = chain.invoke(state["query"])
            state["response"] = response
            
            # Update chat history
            if "chat_history" not in state:
                state["chat_history"] = []
            state["chat_history"].append({
                "human": state["query"],
                "assistant": response
            })
            
            logger.info("Generated response successfully")
            
        except Exception as e:
            logger.error(f"Error in generate_response_node: {e}")
            state["error"] = str(e)
        
        return state
    
    def _handle_error_node(self, state: RAGState) -> RAGState:
        """Handle errors gracefully"""
        error_msg = state.get("error", "Unknown error occurred")
        state["response"] = f"Sorry, I encountered an error: {error_msg}"
        logger.error(f"Error handled: {error_msg}")
        return state
    
    def load_documents(self, path: str) -> bool:
        """
        Load documents from a file or directory
        
        Args:
            path: Path to file or directory
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            documents = []
            path_obj = Path(path)
            
            if path_obj.is_file():
                documents.extend(self._load_single_file(path))
            elif path_obj.is_dir():
                documents.extend(self._load_directory(path))
            else:
                logger.error(f"Path does not exist: {path}")
                return False
            
            if not documents:
                logger.warning("No documents were loaded")
                return False
            
            # Split documents into chunks
            logger.info(f"Splitting {len(documents)} documents into chunks...")
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(split_docs)} document chunks")
            
            # Create or update vector store
            if os.path.exists(self.persist_directory):
                # Load existing vector store
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                # Add new documents
                self.vector_store.add_documents(split_docs)
                logger.info("Added documents to existing vector store")
            else:
                # Create new vector store
                self.vector_store = Chroma.from_documents(
                    documents=split_docs,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
                logger.info("Created new vector store")
            
            # Persist the vector store
            self.vector_store.persist()
            logger.info(f"Successfully loaded and indexed {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return False
    
    def _load_single_file(self, file_path: str) -> List[Document]:
        """Load a single file"""
        documents = []
        file_path = Path(file_path)
        
        try:
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
                documents = loader.load()
                logger.info(f"Loaded PDF: {file_path}")
            elif file_path.suffix.lower() in ['.md', '.markdown']:
                loader = UnstructuredMarkdownLoader(str(file_path))
                documents = loader.load()
                logger.info(f"Loaded Markdown: {file_path}")
            else:
                logger.warning(f"Unsupported file type: {file_path}")
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
        
        return documents
    
    def _load_directory(self, dir_path: str) -> List[Document]:
        """Load all supported files from a directory"""
        documents = []
        dir_path = Path(dir_path)
        
        # Supported file extensions
        supported_extensions = ['.pdf', '.md', '.markdown']
        
        for file_path in dir_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                documents.extend(self._load_single_file(str(file_path)))
        
        logger.info(f"Loaded {len(documents)} documents from directory: {dir_path}")
        return documents
    
    def chat(self, query: str) -> str:
        """
        Chat with the loaded documents
        
        Args:
            query: User's question
            
        Returns:
            str: AI response
        """
        try:
            # Create initial state
            initial_state = RAGState(
                query=query,
                documents=[],
                retrieved_docs=[],
                response="",
                chat_history=self.chat_history,
                vector_store=self.vector_store,
                error=None
            )
            
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            # Update chat history
            if result.get("chat_history"):
                self.chat_history = result["chat_history"]
            
            return result.get("response", "Sorry, I couldn't generate a response.")
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []
        logger.info("Chat history cleared")
    
    def get_document_count(self) -> int:
        """Get the number of documents in the vector store"""
        if self.vector_store:
            try:
                return self.vector_store._collection.count()
            except:
                return 0
        return 0

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="RAG Agent - Chat with your documents")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name (default: llama3.2)")
    parser.add_argument("--db-path", default="./chroma_db", help="ChromaDB persistence directory")
    parser.add_argument("--load", help="Path to document or directory to load")
    parser.add_argument("--query", help="Single query mode")
    
    args = parser.parse_args()
    
    print("🤖 RAG Agent - Chat with your documents")
    print("=" * 50)
    
    # Initialize the agent
    try:
        agent = RAGAgent(model_name=args.model, persist_directory=args.db_path)
        print(f"✅ Initialized RAG Agent with model: {args.model}")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        print("Make sure Ollama is running and the model is available.")
        sys.exit(1)
    
    # Load documents if specified
    if args.load:
        print(f"\n📚 Loading documents from: {args.load}")
        if agent.load_documents(args.load):
            doc_count = agent.get_document_count()
            print(f"✅ Successfully loaded documents! ({doc_count} chunks in database)")
        else:
            print("❌ Failed to load documents")
            sys.exit(1)
    elif os.path.exists(args.db_path):
        # Try to load existing vector store
        try:
            agent.vector_store = Chroma(
                persist_directory=args.db_path,
                embedding_function=agent.embeddings
            )
            doc_count = agent.get_document_count()
            print(f"📖 Loaded existing document database ({doc_count} chunks)")
        except Exception as e:
            print(f"⚠️  Could not load existing database: {e}")
    
    # Single query mode
    if args.query:
        if not agent.vector_store:
            print("❌ No documents loaded. Use --load to load documents first.")
            sys.exit(1)
        
        print(f"\n❓ Query: {args.query}")
        response = agent.chat(args.query)
        print(f"🤖 Response: {response}")
        sys.exit(0)
    
    # Interactive chat mode
    if not agent.vector_store:
        print("\n⚠️  No documents loaded. Use 'load <path>' to load documents.")
    
    print("\n💬 Interactive Chat Mode")
    print("Commands:")
    print("  - Type your question to chat")
    print("  - 'load <path>' to load documents")
    print("  - 'clear' to clear chat history")
    print("  - 'quit' to exit")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'clear':
                agent.clear_history()
                print("🧹 Chat history cleared!")
                continue
            
            elif user_input.lower().startswith('load '):
                path = user_input[5:].strip()
                print(f"📚 Loading documents from: {path}")
                if agent.load_documents(path):
                    doc_count = agent.get_document_count()
                    print(f"✅ Successfully loaded documents! ({doc_count} chunks in database)")
                else:
                    print("❌ Failed to load documents")
                continue
            
            # Chat with documents
            if not agent.vector_store:
                print("❌ No documents loaded. Use 'load <path>' to load documents first.")
                continue
            
            print("🤖 Thinking...")
            response = agent.chat(user_input)
            print(f"🤖 Assistant: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()