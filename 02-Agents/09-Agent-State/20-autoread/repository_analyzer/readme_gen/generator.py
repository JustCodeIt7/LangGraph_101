"""README Generator for repository analysis results."""

from typing import Dict, List, Optional
from pathlib import Path
from ..core.data_structures import RepositoryStructure, RepositoryMetadata, Framework


class READMEGenerator:
    """Generates professional README files from repository analysis results."""
    
    def __init__(self):
        """Initialize the READMEGenerator."""
        self.templates = self._load_templates()
    
    def generate_readme(self, structure: RepositoryStructure) -> str:
        """Generate a README.md content from repository structure analysis.
        
        Args:
            structure: RepositoryStructure object with analysis results
            
        Returns:
            Generated README content as string
        """
        # Determine the appropriate template based on project type
        template = self._select_template(structure)
        
        # Populate template with repository information
        readme_content = self._populate_template(template, structure)
        
        return readme_content
    
    def _load_templates(self) -> Dict[str, str]:
        """Load README templates for different project types.
        
        Returns:
            Dictionary mapping template names to template content
        """
        templates = {
            "default": self._get_default_template(),
            "python_library": self._get_python_library_template(),
            "web_app": self._get_web_app_template(),
            "api_service": self._get_api_service_template(),
            "data_science": self._get_data_science_template()
        }
        return templates
    
    def _select_template(self, structure: RepositoryStructure) -> str:
        """Select the appropriate template based on repository analysis.
        
        Args:
            structure: RepositoryStructure object with analysis results
            
        Returns:
            Template name
        """
        # Check for specific project types based on frameworks and patterns
        frameworks = [f.name for f in structure.frameworks]
        project_type = structure.project_type.value if structure.project_type else ""
        
        # Python library detection
        if structure.metadata.primary_language == "Python":
            if "Django" in frameworks or "Flask" in frameworks or "FastAPI" in frameworks:
                return "web_app"
            elif project_type == "library":
                return "python_library"
            elif any(f in frameworks for f in ["Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch"]):
                return "data_science"
        
        # Web application detection
        if any(f in frameworks for f in ["React", "Vue.js", "Angular", "Express.js", "Next.js"]):
            return "web_app"
        
        # API service detection
        if any(f in frameworks for f in ["FastAPI", "Flask", "Express.js", "Spring Boot"]):
            return "api_service"
        
        # Default to generic template
        return "default"
    
    def _populate_template(self, template: str, structure: RepositoryStructure) -> str:
        """Populate template with repository analysis data.
        
        Args:
            template: Template content
            structure: RepositoryStructure object with analysis results
            
        Returns:
            Populated README content
        """
        # Extract key information
        metadata = structure.metadata
        frameworks = [f.name for f in structure.frameworks]
        
        # Replace template placeholders with actual data
        readme_content = template
        readme_content = readme_content.replace("{{project_name}}", metadata.name or "Project")
        readme_content = readme_content.replace("{{primary_language}}", metadata.primary_language or "Not detected")
        readme_content = readme_content.replace("{{languages}}", ", ".join(metadata.languages) if metadata.languages else "Not detected")
        readme_content = readme_content.replace("{{frameworks}}", ", ".join(frameworks) if frameworks else "Not detected")
        readme_content = readme_content.replace("{{architecture_type}}", metadata.architecture_type or "Not detected")
        
        # Add framework-specific sections
        framework_details = self._generate_framework_details(structure.frameworks)
        readme_content = readme_content.replace("{{framework_details}}", framework_details)
        
        # Add installation instructions
        installation_instructions = self._generate_installation_instructions(structure)
        readme_content = readme_content.replace("{{installation_instructions}}", installation_instructions)
        
        # Add usage examples
        usage_examples = self._generate_usage_examples(structure)
        readme_content = readme_content.replace("{{usage_examples}}", usage_examples)
        
        # Add configuration details
        config_details = self._generate_configuration_details(structure)
        readme_content = readme_content.replace("{{configuration_details}}", config_details)
        
        # Add entry points
        entry_points = "\n".join([f"- `{ep}`" for ep in metadata.entry_points]) if metadata.entry_points else "Not detected"
        readme_content = readme_content.replace("{{entry_points}}", entry_points)
        
        # Add complexity metrics
        readme_content = readme_content.replace("{{complexity_score}}", f"{metadata.complexity_score:.2f}" if metadata.complexity_score else "Not calculated")
        readme_content = readme_content.replace("{{documentation_coverage}}", f"{metadata.documentation_coverage*100:.1f}%" if metadata.documentation_coverage else "Not calculated")
        readme_content = readme_content.replace("{{test_coverage}}", f"{metadata.test_coverage_estimate*100:.1f}%" if metadata.test_coverage_estimate else "Not calculated")
        
        return readme_content
    
    def _generate_framework_details(self, frameworks: List[Framework]) -> str:
        """Generate framework details section.
        
        Args:
            frameworks: List of detected Framework objects
            
        Returns:
            Formatted framework details
        """
        if not frameworks:
            return "No frameworks detected."
        
        details = []
        for framework in frameworks:
            details.append(f"- **{framework.name}** (confidence: {framework.confidence:.2f})")
        
        return "\n".join(details)
    
    def _generate_installation_instructions(self, structure: RepositoryStructure) -> str:
        """Generate installation instructions based on project type.
        
        Args:
            structure: RepositoryStructure object with analysis results
            
        Returns:
            Formatted installation instructions
        """
        metadata = structure.metadata
        primary_language = metadata.primary_language
        
        if primary_language == "Python":
            return """```bash
# Clone the repository
git clone {{repository_url}}
cd {{project_name}}

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```"""
        elif primary_language in ["JavaScript", "TypeScript"]:
            return """```bash
# Clone the repository
git clone {{repository_url}}
cd {{project_name}}

# Install dependencies
npm install
```"""
        elif primary_language == "Java":
            return """```bash
# Clone the repository
git clone {{repository_url}}
cd {{project_name}}

# Build with Maven/Gradle
mvn install
# or
./gradlew build
```"""
        else:
            return """```bash
# Clone the repository
git clone {{repository_url}}
cd {{project_name}}
```"""
    
    def _generate_usage_examples(self, structure: RepositoryStructure) -> str:
        """Generate usage examples based on project type.
        
        Args:
            structure: RepositoryStructure object with analysis results
            
        Returns:
            Formatted usage examples
        """
        frameworks = [f.name for f in structure.frameworks]
        entry_points = structure.metadata.entry_points
        
        if "Django" in frameworks:
            return """```bash
# Run development server
python manage.py runserver

# Create migrations
python manage.py makemigrations
python manage.py migrate
```"""
        elif "Flask" in frameworks or "FastAPI" in frameworks:
            return """```python
from {{project_name}} import app

if __name__ == "__main__":
    app.run(debug=True)
```"""
        elif "React" in frameworks:
            return """```bash
# Start development server
npm start

# Build for production
npm run build
```"""
        elif entry_points:
            entry_points_str = "\n".join([f"python {ep}" for ep in entry_points if ep.endswith('.py')])
            if entry_points_str:
                return f"""```bash
# Run the application
{entry_points_str}
```"""
        
        return "Refer to the entry points section for main executables."
    
    def _generate_configuration_details(self, structure: RepositoryStructure) -> str:
        """Generate configuration details section.
        
        Args:
            structure: RepositoryStructure object with analysis results
            
        Returns:
            Formatted configuration details
        """
        config_files = structure.metadata.configuration_files
        if not config_files:
            return "No configuration files detected."
        
        details = ["Configuration files in this project:"]
        for config_file in config_files[:5]:  # Limit to first 5 for brevity
            details.append(f"- `{config_file}`")
        
        if len(config_files) > 5:
            details.append(f"- ... and {len(config_files) - 5} more")
        
        return "\n".join(details)
    
    def _get_default_template(self) -> str:
        """Get the default README template.
        
        Returns:
            Default template content
        """
        return """# {{project_name}}

A repository analyzed and documented by the Repository Analyzer.

## Project Overview

- **Primary Language**: {{primary_language}}
- **Languages**: {{languages}}
- **Frameworks**: {{frameworks}}
- **Architecture Type**: {{architecture_type}}
- **Complexity Score**: {{complexity_score}}
- **Documentation Coverage**: {{documentation_coverage}}
- **Test Coverage Estimate**: {{test_coverage}}

## Frameworks Detected

{{framework_details}}

## Installation

{{installation_instructions}}

## Usage

{{usage_examples}}

## Configuration

{{configuration_details}}

## Entry Points

{{entry_points}}

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the terms described in [LICENSE](LICENSE).
"""
    
    def _get_python_library_template(self) -> str:
        """Get the Python library README template.
        
        Returns:
            Python library template content
        """
        return """# {{project_name}}

A Python library analyzed and documented by the Repository Analyzer.

## Overview

This is a Python library with the following characteristics:
- **Primary Language**: {{primary_language}}
- **Frameworks**: {{frameworks}}
- **Architecture Type**: {{architecture_type}}

## Installation

{{installation_instructions}}

## Usage

```python
import {{project_name}}

# Example usage
# result = {{project_name}}.some_function()
```

## API Documentation

For detailed API documentation, please refer to the source code and docstrings.

## Frameworks Detected

{{framework_details}}

## Configuration

{{configuration_details}}

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the terms described in [LICENSE](LICENSE).
"""
    
    def _get_web_app_template(self) -> str:
        """Get the web application README template.
        
        Returns:
            Web application template content
        """
        return """# {{project_name}}

A web application analyzed and documented by the Repository Analyzer.

## Project Overview

- **Primary Language**: {{primary_language}}
- **Frameworks**: {{frameworks}}
- **Architecture Type**: {{architecture_type}}

## Getting Started

### Prerequisites

- {{primary_language}} installed
- Package manager (pip, npm, etc.)

### Installation

{{installation_instructions}}

### Running the Application

{{usage_examples}}

## Frameworks Detected

{{framework_details}}

## Configuration

{{configuration_details}}

## Entry Points

{{entry_points}}

## API Endpoints

For API endpoints, please refer to the source code or documentation.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the terms described in [LICENSE](LICENSE).
"""
    
    def _get_api_service_template(self) -> str:
        """Get the API service README template.
        
        Returns:
            API service template content
        """
        return """# {{project_name}} API

An API service analyzed and documented by the Repository Analyzer.

## Overview

This is an API service with the following characteristics:
- **Primary Language**: {{primary_language}}
- **Frameworks**: {{frameworks}}
- **Architecture Type**: {{architecture_type}}

## Getting Started

### Prerequisites

- {{primary_language}} installed
- Package manager (pip, npm, etc.)

### Installation

{{installation_instructions}}

### Running the Service

{{usage_examples}}

## API Documentation

API endpoints and documentation can be found at `/docs` or `/swagger` when the service is running.

## Frameworks Detected

{{framework_details}}

## Configuration

{{configuration_details}}

## Entry Points

{{entry_points}}

## Environment Variables

Key environment variables for this service:
- `PORT` - Port to run the service on
- `DEBUG` - Enable/disable debug mode

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the terms described in [LICENSE](LICENSE).
"""
    
    def _get_data_science_template(self) -> str:
        """Get the data science project README template.
        
        Returns:
            Data science template content
        """
        return """# {{project_name}}

A data science project analyzed and documented by the Repository Analyzer.

## Project Overview

- **Primary Language**: {{primary_language}}
- **Frameworks**: {{frameworks}}
- **Architecture Type**: {{architecture_type}}

## Getting Started

### Prerequisites

- {{primary_language}} installed
- Jupyter Notebook/Lab (optional)
- Required data files (see Data section)

### Installation

{{installation_instructions}}

## Data

This project uses the following data sources:
- [Dataset 1](link_to_dataset)
- [Dataset 2](link_to_dataset)

## Usage

{{usage_examples}}

## Notebooks

This project includes Jupyter notebooks for analysis:
- `notebooks/analysis.ipynb` - Main analysis notebook
- `notebooks/visualization.ipynb` - Data visualization

## Frameworks Detected

{{framework_details}}

## Configuration

{{configuration_details}}

## Results

Key findings from this analysis:
- [Result 1]
- [Result 2]

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the terms described in [LICENSE](LICENSE).
"""