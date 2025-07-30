"""Tests for core data structures."""

import pytest
from repository_analyzer.core.data_structures import (
    RepositoryStructure, DirectoryInfo, FileInfo, 
    Framework, Pattern, Relationship, RepositoryMetadata,
    DirectoryType, FileType, ProjectType
)


def test_directory_info_creation():
    """Test creation of DirectoryInfo objects."""
    directory = DirectoryInfo(
        name="src",
        path="src",
        type=DirectoryType.SOURCE,
        purpose="Source code directory",
        children=["main.py", "utils.py"],
        file_count=2,
        patterns=["python_source"]
    )
    
    assert directory.name == "src"
    assert directory.path == "src"
    assert directory.type == DirectoryType.SOURCE
    assert directory.purpose == "Source code directory"
    assert len(directory.children) == 2
    assert directory.file_count == 2
    assert "python_source" in directory.patterns


def test_file_info_creation():
    """Test creation of FileInfo objects."""
    file_info = FileInfo(
        name="main.py",
        path="src/main.py",
        extension=".py",
        size=1024,
        type=FileType.SOURCE,
        language="Python",
        framework_markers=["flask"],
        imports=["os", "sys"]
    )
    
    assert file_info.name == "main.py"
    assert file_info.path == "src/main.py"
    assert file_info.extension == ".py"
    assert file_info.size == 1024
    assert file_info.type == FileType.SOURCE
    assert file_info.language == "Python"
    assert "flask" in file_info.framework_markers
    assert len(file_info.imports) == 2


def test_repository_metadata_creation():
    """Test creation of RepositoryMetadata objects."""
    metadata = RepositoryMetadata(
        name="test-repo",
        description="A test repository",
        primary_language="Python",
        languages=["Python", "JavaScript"],
        frameworks=["Flask", "React"],
        architecture_type="monolith",
        complexity_score=0.5,
        documentation_coverage=0.8,
        test_coverage_estimate=0.7,
        entry_points=["main.py"],
        configuration_files=["config/settings.json"]
    )
    
    assert metadata.name == "test-repo"
    assert metadata.primary_language == "Python"
    assert len(metadata.languages) == 2
    assert "Flask" in metadata.frameworks
    assert metadata.architecture_type == "monolith"
    assert metadata.complexity_score == 0.5


def test_repository_structure_creation():
    """Test creation of RepositoryStructure objects."""
    # Create sample directories
    directories = {
        "src": DirectoryInfo(
            name="src",
            path="src",
            type=DirectoryType.SOURCE,
            purpose="Source code"
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
        )
    }
    
    # Create sample frameworks
    frameworks = [
        Framework(
            name="Flask",
            version="2.0.0",
            confidence=0.9
        )
    ]
    
    # Create sample patterns
    patterns = [
        Pattern(
            name="standard_python",
            type="project_structure",
            confidence=0.8
        )
    ]
    
    # Create sample relationships
    relationships = [
        Relationship(
            source="src/main.py",
            target="src/utils.py",
            type="import",
            strength=0.9
        )
    ]
    
    # Create metadata
    metadata = RepositoryMetadata(
        name="test-repo",
        primary_language="Python"
    )
    
    # Create repository structure
    structure = RepositoryStructure(
        source="https://github.com/user/test-repo",
        root_path="/tmp/test-repo",
        project_type=ProjectType.MONOLITH,
        frameworks=frameworks,
        directories=directories,
        files=files,
        patterns=patterns,
        relationships=relationships,
        metadata=metadata
    )
    
    assert structure.source == "https://github.com/user/test-repo"
    assert structure.root_path == "/tmp/test-repo"
    assert structure.project_type == ProjectType.MONOLITH
    assert len(structure.frameworks) == 1
    assert len(structure.directories) == 1
    assert len(structure.files) == 1
    assert len(structure.patterns) == 1
    assert len(structure.relationships) == 1
    assert structure.metadata.name == "test-repo"