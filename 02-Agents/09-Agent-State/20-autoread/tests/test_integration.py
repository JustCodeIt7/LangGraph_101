"""Integration tests for repository analyzer."""

import pytest
import os
from pathlib import Path

from repository_analyzer import RepositoryAnalyzer
from repository_analyzer.core.config import AnalysisConfig


def test_analyzer_with_sample_repository():
    """Test analyzer with a sample repository."""
    # Get the path to our sample repository
    sample_repo_path = Path(__file__).parent / "fixtures" / "sample_repos" / "simple_python_project"
    
    # Check that the sample repository exists
    assert sample_repo_path.exists(), f"Sample repository not found at {sample_repo_path}"
    
    # Create a configuration for testing
    config = AnalysisConfig(
        max_depth=10,
        respect_gitignore=True,
        analyze_imports=True,
        detect_frameworks=True,
        map_relationships=True,
        max_file_size=1024 * 1024  # 1MB
    )
    
    # Create analyzer
    analyzer = RepositoryAnalyzer(config)
    
    try:
        # Analyze the sample repository
        structure = analyzer.analyze(str(sample_repo_path))
        
        # Verify basic structure
        assert structure is not None
        assert structure.root_path == str(sample_repo_path)
        assert structure.source == str(sample_repo_path)
        
        # Verify directories were detected
        assert len(structure.directories) > 0
        assert "src" in structure.directories
        assert "tests" in structure.directories
        
        # Verify files were detected
        assert len(structure.files) > 0
        
        # Check for main.py
        assert "src/main.py" in structure.files
        main_file = structure.files["src/main.py"]
        assert main_file.name == "main.py"
        assert main_file.extension == ".py"
        assert main_file.language == "Python"
        
        # Check for utils.py
        assert "src/utils.py" in structure.files
        utils_file = structure.files["src/utils.py"]
        assert utils_file.name == "utils.py"
        assert utils_file.extension == ".py"
        assert utils_file.language == "Python"
        
        # Check for test files
        assert "tests/test_utils.py" in structure.files
        test_file = structure.files["tests/test_utils.py"]
        assert test_file.type.value == "test"
        
        # Verify metadata
        assert structure.metadata is not None
        assert structure.metadata.primary_language == "Python"
        
    finally:
        # Clean up
        analyzer.cleanup()


def test_analyzer_file_type_detection():
    """Test that file type detection works correctly."""
    sample_repo_path = Path(__file__).parent / "fixtures" / "sample_repos" / "simple_python_project"
    assert sample_repo_path.exists()
    
    config = AnalysisConfig(max_depth=5)
    analyzer = RepositoryAnalyzer(config)
    
    try:
        structure = analyzer.analyze(str(sample_repo_path))
        
        # Check that README.md is detected as documentation
        readme_files = [f for f in structure.files.values() if f.name.lower() == "readme.md"]
        assert len(readme_files) > 0
        readme_file = readme_files[0]
        assert readme_file.type.value == "doc"
        
        # Check that requirements.txt is detected as configuration
        req_files = [f for f in structure.files.values() if f.name.lower() == "requirements.txt"]
        assert len(req_files) > 0
        req_file = req_files[0]
        assert req_file.type.value == "config"
        
    finally:
        analyzer.cleanup()


def test_analyzer_gitignore_respect():
    """Test that .gitignore files are respected."""
    sample_repo_path = Path(__file__).parent / "fixtures" / "sample_repos" / "simple_python_project"
    assert sample_repo_path.exists()
    
    config = AnalysisConfig(
        max_depth=5,
        respect_gitignore=True
    )
    analyzer = RepositoryAnalyzer(config)
    
    try:
        structure = analyzer.analyze(str(sample_repo_path))
        
        # Check that common ignored files are not in the structure
        ignored_files = [f for f in structure.files.keys() if "__pycache__" in f]
        assert len(ignored_files) == 0
        
    finally:
        analyzer.cleanup()


def test_analyzer_configuration_parsing():
    """Test that configuration files are parsed correctly."""
    sample_repo_path = Path(__file__).parent / "fixtures" / "sample_repos" / "simple_python_project"
    assert sample_repo_path.exists()
    
    config = AnalysisConfig(max_depth=5)
    analyzer = RepositoryAnalyzer(config)
    
    try:
        structure = analyzer.analyze(str(sample_repo_path))
        
        # Check that metadata includes configuration files
        assert len(structure.metadata.configuration_files) >= 1
        
    finally:
        analyzer.cleanup()