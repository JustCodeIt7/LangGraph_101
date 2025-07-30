"""Pytest configuration and fixtures."""

import pytest
import tempfile
import os
from pathlib import Path
from repository_analyzer.core.config import AnalysisConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """Create a sample AnalysisConfig for tests."""
    return AnalysisConfig(
        max_depth=5,
        respect_gitignore=True,
        ignore_patterns=[".git", "__pycache__"],
        include_hidden=False,
        analyze_imports=True,
        detect_frameworks=True,
        map_relationships=True,
        temp_dir=tempfile.gettempdir(),
        max_file_size=1024 * 1024  # 1MB
    )


@pytest.fixture
def sample_repository_structure():
    """Create a sample repository structure for tests."""
    return {
        "files": {
            "README.md": {
                "type": "doc",
                "language": None
            },
            "src/main.py": {
                "type": "source",
                "language": "Python"
            },
            "src/utils.py": {
                "type": "source",
                "language": "Python"
            },
            "tests/test_main.py": {
                "type": "test",
                "language": "Python"
            },
            "config/settings.json": {
                "type": "config",
                "language": None
            }
        },
        "directories": {
            "src": "source",
            "tests": "tests",
            "config": "config"
        }
    }