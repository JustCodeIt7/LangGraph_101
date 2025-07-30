"""Tests for framework detector."""

import pytest
from repository_analyzer.patterns.frameworks import FrameworkDetector
from repository_analyzer.core.data_structures import (
    DirectoryInfo, FileInfo, FileType, DirectoryType
)


def test_framework_detector_initialization():
    """Test FrameworkDetector initialization."""
    detector = FrameworkDetector()
    
    assert detector is not None
    assert isinstance(detector.framework_signatures, dict)
    assert isinstance(detector.language_frameworks, dict)


def test_framework_detector_detect_frameworks():
    """Test detect_frameworks method."""
    detector = FrameworkDetector()
    
    # Create sample files with React indicators
    files = {
        "package.json": FileInfo(
            name="package.json",
            path="package.json",
            extension=".json",
            size=512,
            type=FileType.CONFIG,
            language=None
        ),
        "src/App.js": FileInfo(
            name="App.js",
            path="src/App.js",
            extension=".js",
            size=1024,
            type=FileType.SOURCE,
            language="JavaScript"
        )
    }
    
    # Create sample directories
    directories = {
        "src": DirectoryInfo(
            name="src",
            path="src",
            type=DirectoryType.SOURCE,
            purpose="Source code"
        )
    }
    
    # Test framework detection
    frameworks = detector.detect_frameworks(files, directories)
    
    # Should return a list (might be empty depending on implementation)
    assert isinstance(frameworks, list)


def test_framework_detector_create_framework_signatures():
    """Test _create_framework_signatures method."""
    detector = FrameworkDetector()
    
    signatures = detector._create_framework_signatures()
    
    assert isinstance(signatures, dict)
    assert len(signatures) > 0
    
    # Check for expected frameworks
    assert "React" in signatures
    assert "Vue.js" in signatures
    assert "Django" in signatures
    assert "Spring Boot" in signatures


def test_framework_detector_create_language_frameworks():
    """Test _create_language_frameworks method."""
    detector = FrameworkDetector()
    
    frameworks = detector._create_language_frameworks()
    
    assert isinstance(frameworks, dict)
    assert len(frameworks) > 0
    
    # Check for expected languages
    assert "Python" in frameworks
    assert "JavaScript" in frameworks
    assert "Java" in frameworks


def test_framework_detector_detect_frameworks_in_config():
    """Test _detect_frameworks_in_config method."""
    detector = FrameworkDetector()
    
    # Create a sample FileInfo for package.json
    file_info = FileInfo(
        name="package.json",
        path="package.json",
        extension=".json",
        size=512,
        type=FileType.CONFIG,
        language=None
    )
    
    # Test with React in package.json content
    # This would normally read the actual file content
    frameworks = detector._detect_frameworks_in_config("package.json", file_info)
    
    # Should return a list of tuples
    assert isinstance(frameworks, list)


def test_framework_detector_detect_frameworks_in_source():
    """Test _detect_frameworks_in_source method."""
    detector = FrameworkDetector()
    
    # Create a sample FileInfo for a JavaScript file
    file_info = FileInfo(
        name="App.js",
        path="src/App.js",
        extension=".js",
        size=1024,
        type=FileType.SOURCE,
        language="JavaScript"
    )
    
    # Test framework detection in source file
    frameworks = detector._detect_frameworks_in_source("src/App.js", file_info)
    
    # Should return a list of tuples
    assert isinstance(frameworks, list)


def test_framework_detector_detect_frameworks_in_structure():
    """Test _detect_frameworks_in_structure method."""
    detector = FrameworkDetector()
    
    # Create sample directories
    directories = {
        "components": DirectoryInfo(
            name="components",
            path="components",
            type=DirectoryType.UNKNOWN,
            purpose="Components directory"
        )
    }
    
    # Test framework detection in structure
    frameworks = detector._detect_frameworks_in_structure(directories)
    
    # Should return a list of tuples
    assert isinstance(frameworks, list)