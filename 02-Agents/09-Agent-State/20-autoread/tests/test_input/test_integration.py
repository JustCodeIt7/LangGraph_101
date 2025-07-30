"""Integration tests for the InputHandler system."""

import os
import tempfile
from pathlib import Path
import pytest

from repository_analyzer.input.handler import InputHandler
from repository_analyzer.input.config import InputConfig
from repository_analyzer.core.analyzer import RepositoryAnalyzer
from repository_analyzer.core.config import AnalysisConfig


class TestInputHandlerIntegration:
    """Integration tests for the InputHandler system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp(prefix="integration_test_")
        self.input_config = InputConfig(
            temp_dir=self.temp_dir,
            timeout=30,
            auto_cleanup=False
        )
        self.input_handler = InputHandler(self.input_config)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.input_handler.cleanup_all()
        # Clean up temp directory
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_local_path_processing(self):
        """Test end-to-end processing of a local path."""
        # Create a test repository structure
        repo_dir = Path(self.temp_dir) / "test_repo"
        repo_dir.mkdir()
        
        # Create some test files
        (repo_dir / "README.md").write_text("# Test Repository\n\nThis is a test repository.")
        (repo_dir / "main.py").write_text("# Test Python file\nprint('Hello, World!')\n")
        (repo_dir / "requirements.txt").write_text("requests==2.25.1\n")
        
        # Process the local path
        processed_input = self.input_handler.process(str(repo_dir))
        
        # Verify the processing results
        assert processed_input.source == str(repo_dir)
        assert processed_input.local_path == str(repo_dir)
        assert not processed_input.is_temporary
        assert not processed_input.auth_used
        
        # Verify the directory exists and has content
        assert os.path.exists(processed_input.local_path)
        assert os.path.exists(os.path.join(processed_input.local_path, "README.md"))
        assert os.path.exists(os.path.join(processed_input.local_path, "main.py"))
    
    def test_integration_with_repository_analyzer(self):
        """Test integration with RepositoryAnalyzer."""
        # Create a test repository structure
        repo_dir = Path(self.temp_dir) / "test_repo"
        repo_dir.mkdir()
        
        # Create some test files
        (repo_dir / "README.md").write_text("# Test Repository\n\nThis is a test repository.")
        (repo_dir / "main.py").write_text("# Test Python file\nprint('Hello, World!')\n")
        (repo_dir / "requirements.txt").write_text("requests==2.25.1\n")
        
        # Configure RepositoryAnalyzer with the same temp directory
        analysis_config = AnalysisConfig(temp_dir=self.temp_dir)
        analyzer = RepositoryAnalyzer(analysis_config)
        
        # Analyze the repository using the input handler
        structure = analyzer.analyze(str(repo_dir))
        
        # Verify the analysis results
        assert structure.source == str(repo_dir)
        assert structure.metadata.name == "test_repo"
        assert len(structure.files) > 0
        assert len(structure.directories) > 0
        
        # Verify specific files are detected
        file_names = [f.name for f in structure.files.values()]
        assert "README.md" in file_names
        assert "main.py" in file_names
        assert "requirements.txt" in file_names
    
    def test_context_manager_integration(self):
        """Test InputHandler context manager integration."""
        # Create a test repository structure
        repo_dir = Path(self.temp_dir) / "test_repo"
        repo_dir.mkdir()
        (repo_dir / "README.md").write_text("# Test Repository")
        
        # Use InputHandler as context manager
        with InputHandler(self.input_config) as handler:
            processed_input = handler.process(str(repo_dir))
            assert len(handler._processed_inputs) == 1
            
            # The processed input should be valid
            assert processed_input.source == str(repo_dir)
            assert processed_input.local_path == str(repo_dir)
        
        # After context exit, inputs should be cleaned up if auto_cleanup was True
        # In our test config, auto_cleanup is False, so we check manually
        # Note: In this specific test, there's no cleanup needed for local paths
    
    def test_error_handling_integration(self):
        """Test error handling integration."""
        # Test processing a non-existent path
        with pytest.raises(Exception):  # Should raise some kind of validation error
            self.input_handler.process("/nonexistent/path/that/does/not/exist")
        
        # Test processing an invalid URL
        with pytest.raises(Exception):  # Should raise some kind of validation error
            self.input_handler.process("invalid://url/format")
    
    def test_configuration_integration(self):
        """Test configuration integration."""
        # Create a custom configuration
        custom_config = InputConfig(
            temp_dir=self.temp_dir,
            timeout=60,
            max_path_length=2000,
            enable_path_traversal_check=True
        )
        
        # Create handler with custom config
        custom_handler = InputHandler(custom_config)
        
        # Create a test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("test content")
        
        # Process with custom configuration
        processed_input = custom_handler.process(str(test_file))
        
        # Verify processing worked with custom config
        assert processed_input.source == str(test_file)
        assert processed_input.local_path == str(test_file)