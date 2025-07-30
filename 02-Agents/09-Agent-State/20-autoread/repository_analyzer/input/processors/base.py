"""Base processor interface for input handling."""

from abc import ABC, abstractmethod
from typing import Optional
from ..handler import ProcessedInput
from ..config import InputConfig


class BaseProcessor(ABC):
    """Base interface for input processors."""
    
    def __init__(self, config: InputConfig):
        """Initialize the processor with configuration.
        
        Args:
            config: Input configuration
        """
        self.config = config
    
    @abstractmethod
    def can_process(self, source: str) -> bool:
        """Check if this processor can handle the given source.
        
        Args:
            source: Input source to check
            
        Returns:
            True if this processor can handle the source, False otherwise
        """
        pass
    
    @abstractmethod
    def process(self, source: str) -> ProcessedInput:
        """Process the input source.
        
        Args:
            source: Input source to process
            
        Returns:
            ProcessedInput object with processing results
        """
        pass