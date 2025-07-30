# Repository Analyzer

A comprehensive Python library for analyzing GitHub repositories and local file systems to extract structural information, identify patterns, and map relationships between project components. Designed for integration with LangGraph agents for automated repository analysis and README generation workflows.

## Features

- **Unified Input Handling**: Process local paths and remote Git repositories through a single interface
- **Repository Cloning**: Clone GitHub repositories locally for deep analysis with authentication support
- **File System Analysis**: Recursive traversal with intelligent filtering including automatic .gitignore parsing
- **Pattern Detection**: Identify common project structures (src/, lib/, docs/, tests/, config/, etc.)
- **Framework Detection**: Recognize 20+ frameworks including React, Django, Spring Boot, and more
- **Relationship Mapping**: Analyze dependencies and connections between code components
- **README Generation**: Automatically generate professional README files from repository analysis
- **LangGraph Integration**: Ready-to-use nodes for agent workflows
- **Multiple Formats**: Support for JSON, YAML, TOML, and other configuration file formats

## Installation

```bash
pip install repository-analyzer
```

Or install from source:

```bash
git clone https://github.com/example/repository-analyzer.git
cd repository-analyzer
pip install -e .
```

## Quick Start

### Basic Usage

```python
from repository_analyzer import RepositoryAnalyzer

# Analyze a GitHub repository or local path
analyzer = RepositoryAnalyzer()
structure = analyzer.analyze("https://github.com/user/repo")
# or
structure = analyzer.analyze("/path/to/local/repository")

# Access analysis results
print(f"Project: {structure.metadata.name}")
print(f"Language: {structure.metadata.primary_language}")
print(f"Frameworks: {[f.name for f in structure.frameworks]}")
```

### Input Handler System

The new InputHandler system provides unified processing for all input sources:

```python
from repository_analyzer.input.handler import InputHandler
from repository_analyzer.input.config import InputConfig

# Configure InputHandler
config = InputConfig(
    temp_dir="/tmp/repo_analyzer",
    timeout=300,
    enable_authentication=True
)

# Process any supported input type
input_handler = InputHandler(config)
processed_input = input_handler.process("https://github.com/user/repo.git")
# or
processed_input = input_handler.process("/local/path/to/repo")

# Clean up temporary resources
input_handler.cleanup(processed_input)
```

### LangGraph Integration

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

### README Generation

The repository analyzer can automatically generate README files from analysis results:

```python
from repository_analyzer.readme_gen.generator import READMEGenerator

# After analyzing a repository
generator = READMEGenerator()
readme_content = generator.generate_readme(structure)

# Save the generated README
with open("generated_README.md", "w") as f:
    f.write(readme_content)
```

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

## Configuration

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
    git_auth_token="your_github_token"
)
```

## Architecture

The Repository Analyzer consists of several modular components:

1. **Input Handler**: Unified input processing for local paths and remote repositories
2. **Git Module**: Handles repository cloning with authentication
3. **Scanner Module**: File system traversal and cataloging with filtering
4. **Patterns Module**: General and framework-specific pattern detection
5. **Analysis Module**: Relationship mapping and import analysis
6. **README Generation Module**: Automated README creation from analysis results
7. **LangGraph Module**: Integration nodes for agent workflows

## Development

### Setup

```bash
make install
```

### Testing

```bash
make test
```

### Code Quality

```bash
make lint
make format
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.