# Repository Analyzer Component - Project Structure

## Recommended Directory Layout

```
repository_analyzer/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── data_structures.py      # RepositoryStructure, DirectoryInfo, FileInfo
│   ├── config.py              # AnalysisConfig, settings management
│   └── exceptions.py          # Custom exception classes
├── git/
│   ├── __init__.py
│   ├── cloner.py             # GitCloner class
│   ├── auth.py               # Authentication handling
│   └── utils.py              # Git utility functions
├── scanner/
│   ├── __init__.py
│   ├── filesystem.py         # FileSystemScanner class
│   ├── cataloger.py          # File cataloging and metadata
│   └── filters.py            # Ignore patterns and filtering
├── patterns/
│   ├── __init__.py
│   ├── detector.py           # PatternDetector base class
│   ├── general.py            # General directory/file patterns
│   ├── frameworks.py         # FrameworkAnalyzer class
│   └── languages.py          # Language-specific patterns
├── analysis/
│   ├── __init__.py
│   ├── relationships.py      # RelationshipMapper class
│   ├── imports.py            # Import analysis for source files
│   ├── config_parser.py      # Configuration file analysis
│   └── metrics.py            # Analysis metrics and scoring
├── langgraph/
│   ├── __init__.py
│   ├── nodes.py              # LangGraph node implementations
│   ├── state.py              # State management and schemas
│   └── workflows.py          # Complete analysis workflows
├── utils/
│   ├── __init__.py
│   ├── file_utils.py         # File operation utilities
│   ├── path_utils.py         # Path manipulation helpers
│   └── logging_utils.py      # Logging configuration
└── tests/
    ├── __init__.py
    ├── conftest.py           # Pytest configuration and fixtures
    ├── test_core/
    │   ├── test_data_structures.py
    │   └── test_config.py
    ├── test_git/
    │   ├── test_cloner.py
    │   └── test_auth.py
    ├── test_scanner/
    │   ├── test_filesystem.py
    │   └── test_cataloger.py
    ├── test_patterns/
    │   ├── test_detector.py
    │   ├── test_frameworks.py
    │   └── test_languages.py
    ├── test_analysis/
    │   ├── test_relationships.py
    │   └── test_imports.py
    ├── test_langgraph/
    │   ├── test_nodes.py
    │   └── test_workflows.py
    └── fixtures/
        ├── sample_repos/     # Sample repository structures
        ├── config_files/     # Sample configuration files
        └── test_data/        # Test data files
```

## Key Files and Their Responsibilities

### Core Module
- **data_structures.py**: All dataclass definitions for repository representation
- **config.py**: Configuration management and validation
- **exceptions.py**: Custom exception hierarchy

### Git Module
- **cloner.py**: Repository cloning, URL parsing, temporary storage management
- **auth.py**: GitHub authentication, token management, SSH key handling
- **utils.py**: Git-specific utility functions

### Scanner Module
- **filesystem.py**: Directory traversal, file discovery, metadata extraction
- **cataloger.py**: File categorization, type detection, content analysis
- **filters.py**: .gitignore parsing, custom ignore patterns

### Patterns Module
- **detector.py**: Base pattern detection interface
- **general.py**: Common project structure patterns (src/, docs/, tests/)
- **frameworks.py**: Framework-specific detection logic
- **languages.py**: Programming language patterns and conventions

### Analysis Module
- **relationships.py**: Component relationship mapping and dependency analysis
- **imports.py**: Source code import analysis
- **config_parser.py**: Configuration file parsing and analysis
- **metrics.py**: Repository complexity and quality metrics

### LangGraph Module
- **nodes.py**: LangGraph node implementations for the agent
- **state.py**: Agent state management and type definitions
- **workflows.py**: Complete analysis workflow orchestration

## Entry Points

### Main Analyzer Class
```python
# repository_analyzer/__init__.py
from .core.analyzer import RepositoryAnalyzer

class RepositoryAnalyzer:
    """Main entry point for repository analysis"""
    
    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        
    def analyze(self, source: str) -> RepositoryStructure:
        """Analyze a repository from URL or local path"""
        pass
        
    def analyze_async(self, source: str) -> Awaitable[RepositoryStructure]:
        """Async version for LangGraph integration"""
        pass
```

### LangGraph Integration
```python
# repository_analyzer/langgraph/__init__.py
from .nodes import RepositoryAnalyzerNode
from .workflows import create_analysis_workflow

# For easy LangGraph integration
def get_analyzer_node(config: AnalysisConfig = None) -> RepositoryAnalyzerNode:
    return RepositoryAnalyzerNode(config)
```

## Configuration Files

### pyproject.toml
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "repository-analyzer"
version = "0.1.0"
description = "Repository structure analysis component for LangGraph agents"
dependencies = [
    "langgraph>=0.1.0",
    "gitpython>=3.1.0",
    "pathlib2>=2.3.0",
    "dataclasses-json>=0.6.0",
    "pydantic>=2.0.0",
    "typing-extensions>=4.0.0"
]

[project.optional-dependencies]
full = [
    "pygithub>=1.59.0",
    "tree-sitter>=0.20.0",
    "toml>=0.10.0",
    "pyyaml>=6.0.0",
    "chardet>=5.0.0"
]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-mock>=3.10.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0"
]
```

### requirements.txt (for simple pip install)
```
langgraph>=0.1.0
gitpython>=3.1.0
pathlib2>=2.3.0
dataclasses-json>=0.6.0
pydantic>=2.0.0
typing-extensions>=4.0.0
```

## Development Setup Files

### .gitignore
```
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

.pytest_cache/
.coverage
htmlcov/
.mypy_cache/
.dmypy.json
dmypy.json

# Temporary directories for cloned repos
temp_repos/
.temp/
```

### Makefile
```makefile
.PHONY: install test lint format clean

install:
	pip install -e .[dev]

test:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=repository_analyzer --cov-report=html

lint:
	mypy repository_analyzer/
	black --check repository_analyzer/ tests/
	isort --check-only repository_analyzer/ tests/

format:
	black repository_analyzer/ tests/
	isort repository_analyzer/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
```

This project structure provides a clean, modular architecture that separates concerns while maintaining easy integration with LangGraph workflows.