"""Data structures for repository analysis."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pathlib import Path


class DirectoryType(Enum):
    """Type of directory in a repository."""
    SOURCE = "source"
    CONFIG = "config"
    DOCS = "docs"
    TESTS = "tests"
    ASSETS = "assets"
    SCRIPTS = "scripts"
    BUILD = "build"
    TEMP = "temp"
    UNKNOWN = "unknown"


class FileType(Enum):
    """Type of file in a repository."""
    SOURCE = "source"
    CONFIG = "config"
    DOC = "doc"
    TEST = "test"
    ASSET = "asset"
    SCRIPT = "script"
    BUILD = "build"
    TEMP = "temp"
    UNKNOWN = "unknown"


class ProjectType(Enum):
    """Type of project architecture."""
    MONOLITH = "monolith"
    MICROSERVICES = "microservices"
    LIBRARY = "library"
    PLUGIN = "plugin"
    UNKNOWN = "unknown"


@dataclass
class DirectoryInfo:
    """Information about a directory in a repository."""
    name: str
    path: str
    type: DirectoryType
    purpose: str
    children: List[str] = field(default_factory=list)
    file_count: int = 0
    patterns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileInfo:
    """Information about a file in a repository."""
    name: str
    path: str
    extension: str
    size: int
    type: FileType
    language: Optional[str] = None
    framework_markers: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Framework:
    """Information about a detected framework."""
    name: str
    version: Optional[str] = None
    confidence: float = 0.0
    files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Pattern:
    """Detected pattern in the repository."""
    name: str
    type: str
    confidence: float
    files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    """Relationship between components in the repository."""
    source: str
    target: str
    type: str
    strength: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RepositoryMetadata:
    """Metadata about the repository."""
    name: str = ""
    description: str = ""
    primary_language: str = ""
    languages: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    architecture_type: str = "unknown"
    complexity_score: float = 0.0
    documentation_coverage: float = 0.0
    test_coverage_estimate: float = 0.0
    entry_points: List[str] = field(default_factory=list)
    configuration_files: List[str] = field(default_factory=list)
    last_commit: Optional[str] = None
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RepositoryStructure:
    """Complete representation of a repository structure."""
    source: str
    root_path: str
    project_type: ProjectType
    frameworks: List[Framework] = field(default_factory=list)
    directories: Dict[str, DirectoryInfo] = field(default_factory=dict)
    files: Dict[str, FileInfo] = field(default_factory=dict)
    patterns: List[Pattern] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    metadata: RepositoryMetadata = field(default_factory=RepositoryMetadata)