"""
Vector store management using ChromaDB
"""

import chromadb
from chromadb.config import Settings
from typing import List, Optional
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import tempfile
import os

class VectorStoreManager:
    """Manage ChromaDB vector store operations"""
    
    def __init__(self, collection_name: str = "default", persist_directory: str = None):
        self.collection_name = collection_name
        
        # Set up persist directory
        if persist_directory is None:
            self.persist_directory = os.path.join(os.path.expanduser("~"), ".langchat", "chromadb")
        else:
            self.persist_directory = persist_directory
        
        # Create directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize embeddings (using Ollama's embedding model)
        self.embeddings = OllamaEmbeddings(model="llama3.2")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Initialize vector store
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize the Chroma vector store"""
        try:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store"""
        if not documents:
            return []
        
        try:
            # Add documents to vector store
            ids = self.vector_store.add_documents(documents)
            
            # Persist the changes
            self.vector_store.persist()
            
            return ids
        except Exception as e:
            print(f"Error adding documents: {e}")
            return []
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        if not self.vector_store:
            return []
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Search for similar documents with similarity scores"""
        if not self.vector_store:
            return []
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            print(f"Error during similarity search with score: {e}")
            return []
    
    def delete_collection(self):
        """Delete the current collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self._initialize_vector_store()
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    def list_collections(self) -> List[str]:
        """List all available collections"""
        try:
            collections = self.client.list_collections()
            return [collection.name for collection in collections]
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection"""
        try:
            collection = self.client.get_collection(name=self.collection_name)
            return collection.count()
        except Exception as e:
            print(f"Error getting collection count: {e}")
            return 0