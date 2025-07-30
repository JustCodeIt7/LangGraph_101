"""Tests for configuration system."""

import pytest
import os
from repository_analyzer.core.config import AnalysisConfig


def test_analysis_config_default_values():
    """Test default values of AnalysisConfig."""
    config = AnalysisConfig()
    
    assert config.max_depth == 10
    assert config.respect_gitignore is True
    assert ".git" in config.ignore_patterns
    assert config.include_hidden is False
    assert config.analyze_imports is True
    assert config.detect_frameworks is True
    assert config.map_relationships is True
    assert config.max_file_size == 10 * 1024 * 1024  # 10MB


def test_analysis_config_custom_values():
    """Test custom values of AnalysisConfig."""
    config = AnalysisConfig(
        max_depth=5,
        respect_gitignore=False,
        ignore_patterns=[".git", "node_modules"],
        include_hidden=True,
        analyze_imports=False,
        detect_frameworks=False,
        map_relationships=False,
        max_file_size=1024 * 1024  # 1MB
    )
    
    assert config.max_depth == 5
    assert config.respect_gitignore is False
    assert "node_modules" in config.ignore_patterns
    assert config.include_hidden is True
    assert config.analyze_imports is False
    assert config.detect_frameworks is False
    assert config.map_relationships is False
    assert config.max_file_size == 1024 * 1024


def test_analysis_config_get_temp_dir():
    """Test get_temp_dir method."""
    config = AnalysisConfig()
    temp_dir = config.get_temp_dir()
    
    assert temp_dir.is_absolute()
    assert str(temp_dir) == config.temp_dir


def test_analysis_config_should_ignore_path():
    """Test should_ignore_path method."""
    config = AnalysisConfig()
    
    # Test ignored patterns
    assert config.should_ignore_path("/path/to/.git/file") is True
    assert config.should_ignore_path("/path/to/node_modules/package") is True
    assert config.should_ignore_path("/path/to/__pycache__/file.pyc") is True
    
    # Test hidden files when not including them
    assert config.should_ignore_path("/path/to/.hidden_file") is True
    
    # Test normal files
    assert config.should_ignore_path("/path/to/normal_file.py") is False


def test_analysis_config_with_custom_ignore_patterns():
    """Test AnalysisConfig with custom ignore patterns."""
    config = AnalysisConfig(ignore_patterns=["custom_ignore"])
    
    assert config.should_ignore_path("/path/to/custom_ignore/file") is True


def test_analysis_config_include_hidden():
    """Test AnalysisConfig with include_hidden=True."""
    config = AnalysisConfig(include_hidden=True)
    
    # Hidden files should not be ignored when include_hidden=True
    assert config.should_ignore_path("/path/to/.hidden_file") is False


def test_analysis_config_environment_variables(monkeypatch):
    """Test AnalysisConfig with environment variables."""
    # Set environment variables
    monkeypatch.setenv("REPO_ANALYZER_TEMP_DIR", "/custom/temp")
    monkeypatch.setenv("GITHUB_TOKEN", "test_token")
    
    config = AnalysisConfig()
    
    assert config.temp_dir == "/custom/temp"
    assert config.git_auth_token == "test_token"