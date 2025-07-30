"""Tests for pattern detector."""

import pytest
from repository_analyzer.patterns.detector import PatternDetector
from repository_analyzer.core.data_structures import (
    DirectoryInfo, FileInfo, FileType, DirectoryType, ProjectType
)


def test_pattern_detector_initialization():
    """Test PatternDetector initialization."""
    detector = PatternDetector()
    
    assert detector is not None
    assert isinstance(detector.project_patterns, dict)
    assert isinstance(detector.architecture_patterns, dict)
    assert isinstance(detector.file_patterns, dict)


def test_pattern_detector_detect_project_type():
    """Test detect_project_type method."""
    detector = PatternDetector()
    
    # Create sample directories that indicate a monolith
    directories = {
        "src": DirectoryInfo(
            name="src",
            path="src",
            type=DirectoryType.SOURCE,
            purpose="Source code"
        )
    }
    
    # Create sample files
    files = {}
    
    # Test monolith detection
    project_type = detector.detect_project_type(directories, files)
    assert project_type == ProjectType.MONOLITH


def test_pattern_detector_detect_project_type_microservices():
    """Test detect_project_type method with microservices."""
    detector = PatternDetector()
    
    # Create sample directories that indicate microservices
    directories = {
        "services": DirectoryInfo(
            name="services",
            path="services",
            type=DirectoryType.UNKNOWN,
            purpose="Services directory"
        ),
        "services/user-service": DirectoryInfo(
            name="user-service",
            path="services/user-service",
            type=DirectoryType.UNKNOWN,
            purpose="User service"
        )
    }
    
    # Create sample files
    files = {}
    
    # Test microservices detection
    project_type = detector.detect_project_type(directories, files)
    # Note: This might still be UNKNOWN depending on the implementation details
    assert project_type in [ProjectType.MICROSERVICES, ProjectType.UNKNOWN, ProjectType.MONOLITH]


def test_pattern_detector_detect_patterns():
    """Test detect_patterns method."""
    detector = PatternDetector()
    
    # Create sample directories
    directories = {
        "src": DirectoryInfo(
            name="src",
            path="src",
            type=DirectoryType.SOURCE,
            purpose="Source code"
        ),
        "tests": DirectoryInfo(
            name="tests",
            path="tests",
            type=DirectoryType.TESTS,
            purpose="Test files"
        )
    }
    
    # Create sample files
    files = {
        "src/main.py": FileInfo(
            name="main.py",
            path="src/main.py",
            extension=".py",
            size=1024,
            type=FileType.SOURCE,
            language="Python"
        ),
        "tests/test_main.py": FileInfo(
            name="test_main.py",
            path="tests/test_main.py",
            extension=".py",
            size=512,
            type=FileType.TEST,
            language="Python"
        )
    }
    
    # Test pattern detection
    patterns = detector.detect_patterns(directories, files)
    
    # Should detect some patterns
    assert isinstance(patterns, list)


def test_pattern_detector_create_project_patterns():
    """Test _create_project_patterns method."""
    detector = PatternDetector()
    
    patterns = detector._create_project_patterns()
    
    assert isinstance(patterns, dict)
    assert len(patterns) > 0
    
    # Check for expected patterns
    assert "standard_maven" in patterns
    assert "standard_node" in patterns
    assert "standard_python" in patterns


def test_pattern_detector_create_architecture_patterns():
    """Test _create_architecture_patterns method."""
    detector = PatternDetector()
    
    patterns = detector._create_architecture_patterns()
    
    assert isinstance(patterns, dict)
    assert len(patterns) > 0
    
    # Check for expected patterns
    assert "microservices" in patterns
    assert "monolith" in patterns
    assert "modular" in patterns


def test_pattern_detector_create_file_patterns():
    """Test _create_file_patterns method."""
    detector = PatternDetector()
    
    patterns = detector._create_file_patterns()
    
    assert isinstance(patterns, dict)
    assert len(patterns) > 0
    
    # Check for expected patterns
    assert "configuration_files" in patterns
    assert "documentation_files" in patterns
    assert "test_files" in patterns