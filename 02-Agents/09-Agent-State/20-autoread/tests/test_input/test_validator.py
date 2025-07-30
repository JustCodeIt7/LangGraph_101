"""Tests for the InputValidator class."""

import os
import tempfile
from pathlib import Path
import pytest

from repository_analyzer.input.validator import InputValidator, ValidationResult
from repository_analyzer.input.config import InputConfig
from repository_analyzer.input.exceptions import InputValidationError
from repository_analyzer.input.classifier import InputType


class TestInputValidator:
    """Test cases for the InputValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp(prefix="validator_test_")
        self.config = InputConfig()
        self.validator = InputValidator(self.config)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temp directory
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_validate_local_path_valid(self):
        """Test validation of a valid local path."""
        # Create a temporary file for testing
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("test content")
        
        result = self.validator.validate(str(test_file))
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert result.input_type == InputType.LOCAL_PATH
        assert result.normalized_source == str(test_file.resolve())
    
    def test_validate_local_path_nonexistent(self):
        """Test validation of a non-existent local path."""
        with pytest.raises(InputValidationError):
            self.validator.validate("/nonexistent/path/that/does/not/exist")
    
    def test_validate_local_path_no_permission(self):
        """Test validation of a path with no read permission."""
        # This test might be skipped on some systems where permission changes don't work as expected
        test_file = Path(self.temp_dir) / "no_read.txt"
        test_file.write_text("test content")
        
        # Try to test permission error, but skip if we can't change permissions
        try:
            test_file.chmod(0o000)  # Remove all permissions
            with pytest.raises(InputValidationError):
                self.validator.validate(str(test_file))
        except (PermissionError, OSError):
            # Skip this test if we can't change permissions
            pytest.skip("Cannot change file permissions on this system")
        finally:
            # Restore permissions so cleanup can work
            try:
                test_file.chmod(0o644)
            except (PermissionError, OSError):
                pass
    
    def test_validate_github_url_valid(self):
        """Test validation of a valid GitHub URL."""
        valid_urls = [
            "https://github.com/user/repo.git",
            "https://github.com/user/repo",
            "git@github.com:user/repo.git"
        ]
        
        for url in valid_urls:
            result = self.validator.validate(url)
            assert isinstance(result, ValidationResult)
            assert result.is_valid
            # GitHub URLs should be classified as GitHub URL type
            assert result.input_type in [InputType.GITHUB_URL, InputType.GENERIC_GIT_URL]
    
    def test_validate_invalid_url_scheme(self):
        """Test validation of URL with invalid scheme."""
        with pytest.raises(InputValidationError):
            self.validator.validate("ftp://example.com/repo.git")
    
    def test_validate_malformed_url(self):
        """Test validation of malformed URL."""
        with pytest.raises(InputValidationError):
            self.validator.validate("not a url at all")
    
    def test_validate_path_traversal_prevention(self):
        """Test path traversal prevention."""
        # Create a config with path traversal check enabled
        config = InputConfig(enable_path_traversal_check=True)
        validator = InputValidator(config)
        
        # Test path traversal attempt
        with pytest.raises(InputValidationError):
            validator.validate("../../etc/passwd")
    
    def test_validate_url_without_host(self):
        """Test validation of URL without host."""
        with pytest.raises(InputValidationError):
            self.validator.validate("https:///repo.git")