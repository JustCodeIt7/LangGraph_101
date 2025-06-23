"""
Configuration settings for the application
"""

import os
from pathlib import Path

class Config:
    """Application configuration"""
    
    # Default settings
    DEFAULT_MODEL = "llama3.2"
    DEFAULT_COLLECTION = "default"
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200
    DEFAULT_SEARCH_K = 5
    
    # Paths
    HOME_DIR = Path.home()
    APP_DIR = HOME_DIR / ".langchat"
    CHROMADB_DIR = APP_DIR / "chromadb"
    
    # Ollama settings
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        cls.APP_DIR.mkdir(exist_ok=True)
        cls.CHROMADB_DIR.mkdir(exist_ok=True)