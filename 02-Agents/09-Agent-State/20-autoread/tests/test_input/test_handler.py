"""Tests for the InputHandler class."""

import os
import tempfile
from pathlib import Path
import pytest

from repository_analyzer.input.handler import InputHandler, ProcessedInput
from repository_analyzer.input.config import InputConfig
from repository_analyzer.input.exceptions import InputValidationError, InputAuthenticationError


class TestInputHandler:
    """Test cases for the InputHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp(prefix="input_handler_test_")
        self.config = InputConfig(
            temp_dir=self.temp_dir,
            timeout=30,
            auto_cleanup=False
        )
        self.input_handler = InputHandler(self.config)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.input_handler.cleanup_all()
        # Clean up temp directory
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_process_local_path_valid(self):
        """Test processing a valid local path."""
        # Create a temporary file for testing
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("test content")
        
        processed = self.input_handler.process(str(test_file))
        
        assert isinstance(processed, ProcessedInput)
        assert processed.source == str(test_file)
        assert processed.local_path == str(test_file)
        assert processed.input_type.name == "LOCAL_PATH"
        assert not processed.is_temporary
        assert not processed.auth_used
        assert processed.provider is None
    
    def test_process_local_path_nonexistent(self):
        """Test processing a non-existent local path."""
        with pytest.raises(InputValidationError):
            self.input_handler.process("/nonexistent/path/that/does/not/exist")
    
    def test_process_github_url(self):
        """Test processing a GitHub URL (mocked)."""
        # This test would normally require mocking the Git operations
        # For now, we'll test the classification part
        github_url = "https://github.com/example/repo.git"
        
        # Test classification
        classification = self.input_handler.classifier.classify(github_url)
        assert classification.input_type.name == "GITHUB_URL"
        assert classification.provider == "github"
    
    def test_cleanup_all(self):
        """Test cleanup of all processed inputs."""
        # Process a few inputs
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("test content")
        
        processed = self.input_handler.process(str(test_file))
        assert len(self.input_handler._processed_inputs) == 1
        
        # Clean up all
        self.input_handler.cleanup_all()
        assert len(self.input_handler._processed_inputs) == 0
    
    def test_context_manager(self):
        """Test InputHandler as context manager."""
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("test content")
        
        with InputHandler(self.config) as handler:
            processed = handler.process(str(test_file))
            assert len(handler._processed_inputs) == 1
        
        # After context exit, inputs should be cleaned up if auto_cleanup is True
        # In our test config, auto_cleanup is False, so we need to check manually