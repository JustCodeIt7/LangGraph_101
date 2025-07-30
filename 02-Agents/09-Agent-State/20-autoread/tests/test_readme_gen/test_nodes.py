"""Tests for the README generation LangGraph nodes."""

import pytest
from repository_analyzer.readme_gen.nodes import READMEGeneratorNode
from repository_analyzer.langgraph.nodes import RepositoryAnalysisState
from repository_analyzer.core.data_structures import (
    RepositoryStructure, RepositoryMetadata, Framework
)


class TestREADMEGeneratorNode:
    """Test cases for the READMEGeneratorNode class."""
    
    def test_initialization(self):
        """Test that READMEGeneratorNode initializes correctly."""
        node = READMEGeneratorNode()
        assert node is not None
        assert hasattr(node, 'generator')
    
    def test_node_execution_with_structure(self):
        """Test executing the node with a repository structure."""
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
        
        # Create initial state
        initial_state: RepositoryAnalysisState = {
            "repository_url": "https://github.com/user/test-repo",
            "local_path": None,
            "analysis_config": None,
            "repository_structure": structure,
            "analysis_summary": None,
            "generated_readme": None,
            "current_step": "analysis_complete",
            "errors": [],
            "warnings": []
        }
        
        # Execute the node
        node = READMEGeneratorNode()
        result_state = node(initial_state)
        
        # Verify results
        assert result_state is not None
        assert result_state["current_step"] == "readme_generation_complete"
        assert "generated_readme" in result_state
        assert result_state["generated_readme"] is not None
        assert isinstance(result_state["generated_readme"], str)
        assert len(result_state["generated_readme"]) > 0
        assert "Test Project" in result_state["generated_readme"]
    
    def test_node_execution_without_structure(self):
        """Test executing the node without a repository structure."""
        # Create initial state without structure
        initial_state: RepositoryAnalysisState = {
            "repository_url": "https://github.com/user/test-repo",
            "local_path": None,
            "analysis_config": None,
            "repository_structure": None,
            "analysis_summary": None,
            "generated_readme": None,
            "current_step": "analysis_complete",
            "errors": [],
            "warnings": []
        }
        
        # Execute the node
        node = READMEGeneratorNode()
        result_state = node(initial_state)
        
        # Verify results
        assert result_state is not None
        assert result_state["current_step"] == "readme_generation_failed"
        assert "No repository structure provided" in str(result_state["errors"])
    
    def test_node_execution_with_exception(self):
        """Test executing the node when an exception occurs."""
        # Create a malformed structure to cause an exception
        structure = RepositoryStructure(
            source="",
            root_path="",
            project_type=None,
            frameworks=None,  # This will cause an exception
            directories=None,
            files=None,
            patterns=None,
            relationships=None,
            metadata=None
        )
        
        # Create initial state
        initial_state: RepositoryAnalysisState = {
            "repository_url": "https://github.com/user/test-repo",
            "local_path": None,
            "analysis_config": None,
            "repository_structure": structure,
            "analysis_summary": None,
            "generated_readme": None,
            "current_step": "analysis_complete",
            "errors": [],
            "warnings": []
        }
        
        # Execute the node
        node = READMEGeneratorNode()
        result_state = node(initial_state)
        
        # Verify results
        assert result_state is not None
        # The node should handle exceptions gracefully
        assert result_state["current_step"] in ["readme_generation_complete", "readme_generation_failed"]


if __name__ == "__main__":
    pytest.main([__file__])