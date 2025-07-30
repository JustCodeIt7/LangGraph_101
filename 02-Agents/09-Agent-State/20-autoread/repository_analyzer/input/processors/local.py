"""Local path processor."""

import os
from pathlib import Path
from typing import Optional
from .base import BaseProcessor
from ..handler import ProcessedInput, InputType
from ..config import InputConfig
from ..exceptions import InputValidationError


class LocalPathProcessor(BaseProcessor):
    """Processor for local file system paths."""
    
    def can_process(self, source: str) -> bool:
        """Check if this processor can handle the given source.
        
        Args:
            source: Input source to check
            
        Returns:
            True if this processor can handle the source, False otherwise
        """
        # Check if it looks like a local path
        if source.startswith("/") or source.startswith("./") or source.startswith("../"):
            return True
        
        # Check if it's a valid local path
        if os.path.exists(source):
            return True
        
        return False
    
    def process(self, source: str) -> ProcessedInput:
        """Process the local path input source.
        
        Args:
            source: Local path to process
            
        Returns:
            ProcessedInput object with processing results
        """
        # Validate the local path
        if not os.path.exists(source):
            raise InputValidationError(f"Local path does not exist: {source}")
        
        if not os.access(source, os.R_OK):
            raise InputValidationError(f"No read permission for local path: {source}")
        
        # Normalize the path
        normalized_path = os.path.abspath(source)
        
        # Create processed input result
        processed = ProcessedInput(
            source=source,
            input_type=InputType.LOCAL_PATH,
            local_path=normalized_path,
            is_temporary=False,
            auth_used=False,
            provider=None,
            cleanup_callback=None,
            metadata={
                'original_path': source,
                'file_count': self._count_files(normalized_path) if os.path.isdir(normalized_path) else 1,
                'is_directory': os.path.isdir(normalized_path)
            }
        )
        
        return processed
    
    def _count_files(self, path: str) -> int:
        """Count files in a directory.
        
        Args:
            path: Directory path to count files in
            
        Returns:
            Number of files in the directory
        """
        try:
            if os.path.isfile(path):
                return 1
            elif os.path.isdir(path):
                count = 0
                for root, dirs, files in os.walk(path):
                    count += len(files)
                return count
            return 0
        except Exception:
            return 0