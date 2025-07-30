# README Generation Module

This module provides functionality for generating professional README.md files from repository analysis results.

## Overview

The README generation module analyzes the structure and content of a repository to automatically create comprehensive documentation. It uses templates tailored to different project types and populates them with information extracted during the analysis phase.

## Features

- **Template System**: Multiple templates for different project types (libraries, web apps, APIs, etc.)
- **Automatic Population**: Templates are automatically populated with analyzed repository information
- **Framework Awareness**: Adapts content based on detected frameworks and technologies
- **Professional Formatting**: Generates well-structured, readable documentation
- **LangGraph Integration**: Seamlessly integrates with existing LangGraph workflows

## Usage

### Basic Usage

```python
from repository_analyzer.readme_gen.generator import READMEGenerator
from repository_analyzer.core.analyzer import RepositoryAnalyzer

# Analyze a repository
analyzer = RepositoryAnalyzer()
structure = analyzer.analyze("https://github.com/user/repo")

# Generate README
generator = READMEGenerator()
readme_content = generator.generate_readme(structure)

# Save to file
with open("README.md", "w") as f:
    f.write(readme_content)
```

### LangGraph Integration

The module includes a LangGraph node for seamless integration:

```python
from repository_analyzer.readme_gen.nodes import get_readme_generator_node

# Create README generator node
readme_node = get_readme_generator_node()

# Add to existing workflow
workflow.add_node("generate_readme", readme_node)
workflow.add_edge("analyze_repository", "generate_readme")
```

## Templates

The module includes several templates for different project types:

1. **Default Template**: Generic template for any project type
2. **Python Library**: Specialized for Python libraries and packages
3. **Web Application**: For web applications (React, Vue, Angular, etc.)
4. **API Service**: For API services (FastAPI, Flask, Express, etc.)
5. **Data Science**: For data science projects (Pandas, NumPy, etc.)

Templates are automatically selected based on the frameworks and technologies detected in the repository.

## Template Variables

Templates use the following variables that are populated with analyzed data:

- `{{project_name}}`: Repository name
- `{{primary_language}}`: Main programming language
- `{{languages}}`: All detected languages
- `{{frameworks}}`: Detected frameworks
- `{{architecture_type}}`: Project architecture type
- `{{complexity_score}}`: Repository complexity score
- `{{documentation_coverage}}`: Documentation coverage percentage
- `{{test_coverage}}`: Test coverage estimate
- `{{framework_details}}`: Detailed framework information
- `{{installation_instructions}}`: Language-specific installation instructions
- `{{usage_examples}}`: Usage examples based on project type
- `{{configuration_details}}`: Configuration file information
- `{{entry_points}}`: Detected entry points

## Customization

To customize the templates or add new ones:

1. Extend the `READMEGenerator` class
2. Add new template methods
3. Update the `_select_template` method to use your new templates

## Integration with Analysis Results

The README generator uses information from the repository analysis:

- **Metadata**: Project name, languages, frameworks
- **Structure**: Files, directories, patterns
- **Configuration**: Configuration files and settings
- **Entry Points**: Main executables and entry points
- **Metrics**: Complexity scores, coverage estimates

This information is used to create accurate and relevant documentation.