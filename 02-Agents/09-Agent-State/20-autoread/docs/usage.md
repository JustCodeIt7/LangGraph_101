# Repository Analyzer Usage Guide

## Installation

Install the repository analyzer using pip:

```bash
pip install repository-analyzer
```

Or install from source:

```bash
git clone https://github.com/example/repository-analyzer.git
cd repository-analyzer
pip install -e .
```

## Basic Usage

### Analyzing a Repository

```python
from repository_analyzer import RepositoryAnalyzer

# Create analyzer instance
analyzer = RepositoryAnalyzer()

# Analyze a GitHub repository
structure = analyzer.analyze("https://github.com/user/repo")

# Or analyze a local repository
structure = analyzer.analyze("/path/to/local/repo")

# Access analysis results
print(f"Project: {structure.metadata.name}")
print(f"Primary Language: {structure.metadata.primary_language}")
print(f"Frameworks: {structure.metadata.frameworks}")
print(f"Architecture Type: {structure.metadata.architecture_type}")
```

### Configuration

The analyzer can be configured with various options:

```python
from repository_analyzer import AnalysisConfig

config = AnalysisConfig(
    max_depth=10,
    respect_gitignore=True,  # Parse and apply .gitignore rules by default
    ignore_patterns=[".git", "__pycache__", "node_modules"],
    include_hidden=False,
    analyze_imports=True,
    detect_frameworks=True,
    map_relationships=True,
    temp_dir="/tmp/repo_analyzer",
    git_auth_token="your_github_token",
    max_file_size=10 * 1024 * 1024  # 10MB limit
)

analyzer = RepositoryAnalyzer(config)
```

## README Generation

The repository analyzer includes functionality to automatically generate professional README files from repository analysis results.

### Basic Usage

```python
from repository_analyzer.readme_gen.generator import READMEGenerator

# After analyzing a repository
generator = READMEGenerator()
readme_content = generator.generate_readme(structure)

# Save the generated README
with open("generated_README.md", "w") as f:
    f.write(readme_content)
```

### LangGraph Integration

For LangGraph workflows, README generation is automatically included:

```python
from repository_analyzer.langgraph import create_analysis_workflow

# Create a complete workflow with README generation
workflow = create_analysis_workflow()

# Run the workflow
result = workflow.invoke({"repository_url": "https://github.com/user/repo"})

# Access the generated README
generated_readme = result["generated_readme"]
```

### Customization

The README generator uses templates tailored to different project types:

1. **Default Template**: Generic template for any project type
2. **Python Library**: Specialized for Python libraries and packages
3. **Web Application**: For web applications (React, Vue, Angular, etc.)
4. **API Service**: For API services (FastAPI, Flask, Express, etc.)
5. **Data Science**: For data science projects (Pandas, NumPy, etc.)

Templates are automatically selected based on the frameworks and technologies detected in the repository.

## LangGraph Integration

### Using the Repository Analyzer Node

```python
from repository_analyzer.langgraph import get_analyzer_node
from repository_analyzer import AnalysisConfig

# Create analyzer node with custom config
config = AnalysisConfig(
    max_depth=8,
    analyze_imports=True,
    detect_frameworks=True
)

analyzer_node = get_analyzer_node(config)

# Use in LangGraph workflow
# workflow.add_node("repository_analysis", analyzer_node)
```

### Creating a Complete Analysis Workflow

```python
from repository_analyzer.langgraph import create_analysis_workflow

# Create a complete workflow
workflow = create_analysis_workflow()

# Run the workflow
# result = workflow.invoke({"repository_url": "https://github.com/user/repo"})
```

## Analysis Results

The repository analyzer provides detailed information about the repository structure:

### Repository Structure Object

```python
from repository_analyzer.core.data_structures import RepositoryStructure

# The structure object contains all analysis results
structure: RepositoryStructure = analyzer.analyze("https://github.com/user/repo")

# Access directories
for path, directory in structure.directories.items():
    print(f"Directory: {directory.name} ({directory.type.value})")

# Access files
for path, file in structure.files.items():
    print(f"File: {file.name} ({file.language})")

# Access frameworks
for framework in structure.frameworks:
    print(f"Framework: {framework.name} (confidence: {framework.confidence})")

# Access patterns
for pattern in structure.patterns:
    print(f"Pattern: {pattern.name} (type: {pattern.type})")

# Access relationships
for relationship in structure.relationships:
    print(f"Relationship: {relationship.source} -> {relationship.target}")
```

### Repository Metadata

```python
# Access repository metadata
metadata = structure.metadata

print(f"Project Name: {metadata.name}")
print(f"Primary Language: {metadata.primary_language}")
print(f"Languages: {metadata.languages}")
print(f"Frameworks: {metadata.frameworks}")
print(f"Architecture Type: {metadata.architecture_type}")
print(f"Complexity Score: {metadata.complexity_score}")
print(f"Documentation Coverage: {metadata.documentation_coverage}")
print(f"Test Coverage Estimate: {metadata.test_coverage_estimate}")
print(f"Entry Points: {metadata.entry_points}")
print(f"Configuration Files: {metadata.configuration_files}")
```

## Advanced Usage

### Custom Ignore Patterns

```python
config = AnalysisConfig(
    ignore_patterns=[
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".tox",
        ".venv",
        "venv",
        "node_modules",
        "build",
        "dist",
        "custom_ignore_pattern"
    ]
)
```

### Authentication for Private Repositories

```python
config = AnalysisConfig(
    git_auth_token="your_github_personal_access_token"
)

analyzer = RepositoryAnalyzer(config)
structure = analyzer.analyze("https://github.com/user/private-repo")
```

### Performance Optimization

```python
config = AnalysisConfig(
    max_depth=5,  # Limit directory traversal depth
    max_file_size=5 * 1024 * 1024,  # 5MB file size limit
    parallel_processing=True,  # Enable parallel processing
    max_workers=8  # Number of worker threads
)
```

## Error Handling

The repository analyzer provides comprehensive error handling:

```python
from repository_analyzer.core.exceptions import (
    RepositoryNotFoundError,
    AuthenticationError,
    AnalysisError
)

try:
    structure = analyzer.analyze("https://github.com/user/repo")
except RepositoryNotFoundError:
    print("Repository not found")
except AuthenticationError:
    print("Authentication failed")
except AnalysisError as e:
    print(f"Analysis failed: {e}")
```

## Testing

Run the test suite:

```bash
# Run all tests
make test

# Run tests with coverage
make test-coverage

# Run specific test
pytest tests/test_core/test_analyzer.py
```

## Development

### Setup Development Environment

```bash
# Install in development mode
make install

# Run linting
make lint

# Format code
make format

# Clean build artifacts
make clean
```

## API Reference

### RepositoryAnalyzer

Main class for repository analysis.

#### Methods

- `analyze(source: str) -> RepositoryStructure`: Analyze a repository from URL or local path
- `cleanup()`: Clean up temporary resources

### AnalysisConfig

Configuration class for repository analysis.

#### Attributes

- `max_depth`: Maximum directory traversal depth
- `respect_gitignore`: Parse and apply .gitignore rules
- `ignore_patterns`: Additional patterns to ignore
- `include_hidden`: Include hidden files and directories
- `analyze_imports`: Analyze import statements in source files
- `detect_frameworks`: Detect frameworks and technologies
- `map_relationships`: Map relationships between components
- `temp_dir`: Temporary directory for cloned repositories
- `git_auth_token`: GitHub authentication token
- `max_file_size`: Maximum file size to analyze
- `parallel_processing`: Enable parallel processing
- `max_workers`: Number of worker threads for parallel processing

### RepositoryStructure

Complete representation of a repository structure.

#### Attributes

- `source`: Source URL or local path
- `root_path`: Path to repository root
- `project_type`: Project type (monolith, microservices, etc.)
- `frameworks`: Detected frameworks
- `directories`: Directory information
- `files`: File information
- `patterns`: Detected patterns
- `relationships`: Component relationships
- `metadata`: Repository metadata

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.