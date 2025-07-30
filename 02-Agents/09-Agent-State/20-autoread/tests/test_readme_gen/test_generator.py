"""Tests for the README generator module."""

import pytest
from repository_analyzer.readme_gen.generator import READMEGenerator
from repository_analyzer.core.data_structures import (
    RepositoryStructure, RepositoryMetadata, FileInfo, DirectoryInfo, Framework
)


class TestREADMEGenerator:
    """Test cases for the READMEGenerator class."""
    
    def test_initialization(self):
        """Test that READMEGenerator initializes correctly."""
        generator = READMEGenerator()
        assert generator is not None
        assert isinstance(generator.templates, dict)
        assert len(generator.templates) > 0
    
    def test_template_loading(self):
        """Test that templates are loaded correctly."""
        generator = READMEGenerator()
        templates = generator.templates
        
        # Check that all expected templates are present
        expected_templates = ["default", "python_library", "web_app", "api_service", "data_science"]
        for template_name in expected_templates:
            assert template_name in templates
            assert isinstance(templates[template_name], str)
            assert len(templates[template_name]) > 0
    
    def test_readme_generation(self):
        """Test README generation with a simple structure."""
        # Create a minimal repository structure for testing
        metadata = RepositoryMetadata()
        metadata.name = "Test Project"
        metadata.primary_language = "Python"
        metadata.languages = ["Python"]
        metadata.frameworks = ["Django"]
        metadata.architecture_type = "web_application"
        metadata.complexity_score = 0.5
        metadata.documentation_coverage = 0.75
        metadata.test_coverage_estimate = 0.6
        
        framework = Framework(name="Django", confidence=0.9, files=[])
        
        structure = RepositoryStructure(
            source="https://github.com/user/test-repo",
            root_path="/tmp/test-repo",
            project_type=None,
            frameworks=[framework],
            directories={},
            files={},
            patterns=[],
            relationships=[],
            metadata=metadata
        )
        
        # Generate README
        generator = READMEGenerator()
        readme_content = generator.generate_readme(structure)
        
        # Verify content
        assert readme_content is not None
        assert isinstance(readme_content, str)
        assert len(readme_content) > 0
        assert "Test Project" in readme_content
        assert "Python" in readme_content
        assert "Django" in readme_content
    
    def test_template_selection(self):
        """Test that the correct template is selected based on project type."""
        generator = READMEGenerator()
        
        # Test Python library detection
        metadata = RepositoryMetadata()
        metadata.primary_language = "Python"
        metadata.frameworks = []
        
        framework_django = Framework(name="Django", confidence=0.9, files=[])
        framework_flask = Framework(name="Flask", confidence=0.8, files=[])
        
        structure = RepositoryStructure(
            source="",
            root_path="",
            project_type=None,
            frameworks=[framework_django],
            directories={},
            files={},
            patterns=[],
            relationships=[],
            metadata=metadata
        )
        
        # Should select web_app template for Django
        template_name = generator._select_template(structure)
        assert template_name == "web_app"
        
        # Test with Flask
        structure.frameworks = [framework_flask]
        template_name = generator._select_template(structure)
        assert template_name == "web_app"
        
        # Test with no frameworks (should default to generic)
        structure.frameworks = []
        template_name = generator._select_template(structure)
        assert template_name == "python_library"  # Python library template for Python projects with no frameworks
    
    def test_framework_details_generation(self):
        """Test generation of framework details section."""
        generator = READMEGenerator()
        
        frameworks = [
            Framework(name="Django", confidence=0.9, files=[]),
            Framework(name="React", confidence=0.8, files=[])
        ]
        
        details = generator._generate_framework_details(frameworks)
        assert "Django" in details
        assert "React" in details
        assert "0.90" in details
        assert "0.80" in details
    
    def test_installation_instructions(self):
        """Test generation of installation instructions."""
        generator = READMEGenerator()
        
        metadata = RepositoryMetadata()
        metadata.primary_language = "Python"
        
        structure = RepositoryStructure(
            source="",
            root_path="",
            project_type=None,
            frameworks=[],
            directories={},
            files={},
            patterns=[],
            relationships=[],
            metadata=metadata
        )
        
        instructions = generator._generate_installation_instructions(structure)
        assert "pip install -r requirements.txt" in instructions
        assert "python -m venv venv" in instructions
    
    def test_configuration_details(self):
        """Test generation of configuration details."""
        generator = READMEGenerator()
        
        metadata = RepositoryMetadata()
        metadata.configuration_files = ["config.yaml", "settings.py"]
        
        structure = RepositoryStructure(
            source="",
            root_path="",
            project_type=None,
            frameworks=[],
            directories={},
            files={},
            patterns=[],
            relationships=[],
            metadata=metadata
        )
        
        config_details = generator._generate_configuration_details(structure)
        assert "config.yaml" in config_details
        assert "settings.py" in config_details


if __name__ == "__main__":
    pytest.main([__file__])