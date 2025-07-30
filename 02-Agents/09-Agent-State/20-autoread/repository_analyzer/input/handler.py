"""Main InputHandler class for unified input processing."""

from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import os
import tempfile
import shutil

from .config import InputConfig
from .validator import InputValidator, ValidationResult
from .classifier import InputClassifier, ClassificationResult, InputType
from .auth.manager import AuthManager, AuthConfig
from .utils.temp_manager import TempDirectoryManager
from .utils.url_parser import URLParser
from .exceptions import InputHandlerError, InputValidationError, InputAuthenticationError
from ..core.exceptions import RepositoryNotFoundError


@dataclass
class ProcessedInput:
    """Result of input processing."""
    source: str
    input_type: InputType
    local_path: str
    is_temporary: bool
    auth_used: bool
    provider: Optional[str]
    cleanup_callback: Optional[callable]
    metadata: Dict[str, Any]


class InputHandler:
    """Unified input processing system for repository sources."""
    
    def __init__(self, config: Optional[InputConfig] = None):
        """Initialize the InputHandler.
        
        Args:
            config: Input configuration, uses default if None
        """
        self.config = config or InputConfig()
        self.validator = InputValidator(self.config)
        self.classifier = InputClassifier()
        self.auth_manager = AuthManager(AuthConfig())
        self.temp_manager = TempDirectoryManager(self.config.temp_dir)
        self.url_parser = URLParser()
        
        # Track processed inputs for cleanup
        self._processed_inputs = []
    
    def process(self, source: str) -> ProcessedInput:
        """Process and validate input source.
        
        Args:
            source: Input source (local path or URL)
            
        Returns:
            ProcessedInput object with processing results
            
        Raises:
            InputValidationError: If input validation fails
            InputAuthenticationError: If authentication fails
            InputHandlerError: If processing fails for other reasons
        """
        try:
            # 1. Classify input
            classification = self.classifier.classify(source)
            if classification.input_type == InputType.UNKNOWN:
                raise InputValidationError(f"Unable to classify input source: {source}")
            
            # 2. Validate input
            validation = self.validator.validate(source)
            
            # 3. Process based on input type
            if classification.input_type == InputType.LOCAL_PATH:
                return self._process_local_path(source, validation)
            else:
                return self._process_remote_source(source, classification, validation)
                
        except Exception as e:
            if isinstance(e, InputHandlerError):
                raise
            else:
                raise InputHandlerError(f"Failed to process input {source}: {e}") from e
    
    def _process_local_path(self, source: str, validation: ValidationResult) -> ProcessedInput:
        """Process local path input."""
        # For local paths, we can use them directly
        processed = ProcessedInput(
            source=source,
            input_type=InputType.LOCAL_PATH,
            local_path=validation.normalized_source,
            is_temporary=False,
            auth_used=False,
            provider=None,
            cleanup_callback=None,
            metadata=validation.metadata or {}
        )
        
        self._processed_inputs.append(processed)
        return processed
    
    def _process_remote_source(self, source: str, classification: ClassificationResult, 
                              validation: ValidationResult) -> ProcessedInput:
        """Process remote source (Git repository)."""
        # 1. Apply authentication if needed
        auth_used = False
        auth_source = source
        
        if self.config.enable_authentication and classification.provider:
            auth_source = self.auth_manager.authenticate_url(source, classification.provider)
            auth_used = (auth_source != source)
        
        # 2. Create temporary directory for cloning
        temp_dir = self.temp_manager.create_temp_directory("repo_input_")
        
        # 3. Clone repository (simplified - in real implementation this would use Git)
        try:
            # In a real implementation, this would actually clone the repository
            # For now, we'll simulate by creating a basic structure
            repo_path = Path(temp_dir)
            
            # Create a basic repo structure for demonstration
            (repo_path / ".git").mkdir(exist_ok=True)
            (repo_path / "README.md").write_text(f"# Repository from {source}")
            
            processed = ProcessedInput(
                source=source,
                input_type=classification.input_type,
                local_path=str(repo_path),
                is_temporary=True,
                auth_used=auth_used,
                provider=classification.provider,
                cleanup_callback=lambda: self.temp_manager.cleanup_directory(temp_dir),
                metadata=validation.metadata or {}
            )
            
            self._processed_inputs.append(processed)
            return processed
            
        except Exception as e:
            # Clean up on failure
            try:
                self.temp_manager.cleanup_directory(temp_dir)
            except Exception:
                pass  # Ignore cleanup errors
            raise InputHandlerError(f"Failed to process remote source {source}: {e}") from e
    
    def cleanup(self, processed_input: ProcessedInput) -> bool:
        """Clean up temporary resources.
        
        Args:
            processed_input: ProcessedInput to clean up
            
        Returns:
            True if cleanup was successful, False otherwise
        """
        try:
            if processed_input.is_temporary and processed_input.cleanup_callback:
                return processed_input.cleanup_callback()
            return True
        except Exception:
            return False
    
    def cleanup_all(self) -> None:
        """Clean up all tracked processed inputs."""
        for processed in self._processed_inputs:
            try:
                self.cleanup(processed)
            except Exception:
                pass  # Continue with other cleanups
        self._processed_inputs.clear()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        if self.config.auto_cleanup:
            self.cleanup_all()