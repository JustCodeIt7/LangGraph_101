"""
Document processing for PDF and Markdown files
"""

import os
from pathlib import Path
from typing import List, Union
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import PyPDF2
from io import StringIO

class DocumentProcessor:
    """Process documents from various sources"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Supported file extensions
        self.supported_extensions = {'.pdf', '.md', '.txt', '.markdown'}
    
    def process_path(self, path: Union[str, Path]) -> List[Document]:
        """Process a file or directory path"""
        path = Path(path)
        documents = []
        
        if path.is_file():
            if path.suffix.lower() in self.supported_extensions:
                docs = self._process_file(path)
                documents.extend(docs)
        elif path.is_dir():
            for file_path in path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                    docs = self._process_file(file_path)
                    documents.extend(docs)
        
        return documents
    
    def _process_file(self, file_path: Path) -> List[Document]:
        """Process a single file"""
        try:
            if file_path.suffix.lower() == '.pdf':
                return self._process_pdf(file_path)
            elif file_path.suffix.lower() in {'.md', '.markdown', '.txt'}:
                return self._process_text_file(file_path)
            else:
                return []
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []
    
    def _process_pdf(self, file_path: Path) -> List[Document]:
        """Process PDF file"""
        documents = []
        
        try:
            # Use PyPDF2 for better handling
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
                if text.strip():
                    # Create document with metadata
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": str(file_path),
                            "file_type": "pdf",
                            "total_pages": len(pdf_reader.pages)
                        }
                    )
                    
                    # Split into chunks
                    chunks = self.text_splitter.split_documents([doc])
                    documents.extend(chunks)
        
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
        
        return documents
    
    def _process_text_file(self, file_path: Path) -> List[Document]:
        """Process text-based files (Markdown, TXT)"""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                if content.strip():
                    # Create document with metadata
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": str(file_path),
                            "file_type": file_path.suffix.lower()[1:],  # Remove the dot
                            "file_size": len(content)
                        }
                    )
                    
                    # Split into chunks
                    chunks = self.text_splitter.split_documents([doc])
                    documents.extend(chunks)
        
        except Exception as e:
            print(f"Error processing text file {file_path}: {e}")
        
        return documents
    
    def get_supported_extensions(self) -> set:
        """Get supported file extensions"""
        return self.supported_extensions.copy()