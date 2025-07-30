"""Tests for LangGraph nodes."""

import pytest
from unittest.mock import Mock, patch
from repository_analyzer.langgraph.nodes import (
    RepositoryAnalyzerNode, 
    get_analyzer_node, 
    create_analysis_workflow,
    RepositoryAnalysisState
)
from repository_analyzer.core.data_structures import RepositoryStructure


def test_repository_analyzer_node_initialization():
    """Test RepositoryAnalyzerNode initialization."""
    node = RepositoryAnalyzerNode()
    
    assert node is not None
    assert node.config is None
    assert node.analyzer is not None


def test_repository_analyzer_node_initialization_with_config():
    """Test RepositoryAnalyzerNode initialization with config."""
    from repository_analyzer.core.config import AnalysisConfig
    
    config = AnalysisConfig(max_depth=5)
    node = RepositoryAnalyzerNode(config)
    
    assert node.config == config


def test_get_analyzer_node():
    """Test get_analyzer_node function."""
    node = get_analyzer_node()
    
    assert isinstance(node, RepositoryAnalyzerNode)


def test_get_analyzer_node_with_config():
    """Test get_analyzer_node function with config."""
    from repository_analyzer.core.config import AnalysisConfig
    
    config = AnalysisConfig(max_depth=5)
    node = get_analyzer_node(config)
    
    assert isinstance(node, RepositoryAnalyzerNode)
    assert node.config == config


@patch('repository_analyzer.langgraph.nodes.RepositoryAnalyzer')
def test_repository_analyzer_node_call_success(mock_analyzer):
    """Test RepositoryAnalyzerNode __call__ method with success."""
    # Create node
    node = RepositoryAnalyzerNode()
    
    # Mock analyzer.analyze to return a RepositoryStructure
    mock_structure = Mock(spec=RepositoryStructure)
    mock_structure.metadata = Mock()
    mock_structure.metadata.name = "test-repo"
    mock_structure.metadata.primary_language = "Python"
    mock_structure.metadata.languages = ["Python"]
    mock_structure.metadata.frameworks = ["Flask"]
    mock_structure.metadata.architecture_type = "monolith"
    mock_structure.metadata.complexity_score = 0.5
    mock_structure.metadata.documentation_coverage = 0.8
    mock_structure.metadata.test_coverage_estimate = 0.7
    mock_structure.metadata.entry_points = ["main.py"]
    mock_structure.metadata.configuration_files = ["config.json"]
    mock_structure.files = {}
    mock_structure.directories = {}
    mock_structure.frameworks = []
    mock_structure.patterns = []
    mock_structure.relationships = []
    
    node.analyzer.analyze.return_value = mock_structure
    
    # Create test state
    state = {
        "repository_url": "https://github.com/user/repo",
        "errors": [],
        "current_step": "start"
    }
    
    # Test node execution
    result = node(state)
    
    # Check result
    assert result["current_step"] == "analysis_complete"
    assert "repository_structure" in result
    assert "analysis_summary" in result
    assert len(result["errors"]) == 0


@patch('repository_analyzer.langgraph.nodes.RepositoryAnalyzer')
def test_repository_analyzer_node_call_no_source(mock_analyzer):
    """Test RepositoryAnalyzerNode __call__ method with no source."""
    # Create node
    node = RepositoryAnalyzerNode()
    
    # Create test state with no source
    state = {
        "errors": [],
        "current_step": "start"
    }
    
    # Test node execution
    result = node(state)
    
    # Check result
    assert result["current_step"] == "analysis_failed"
    assert "No repository source provided" in result["errors"]


@patch('repository_analyzer.langgraph.nodes.RepositoryAnalyzer')
def test_repository_analyzer_node_call_exception(mock_analyzer):
    """Test RepositoryAnalyzerNode __call__ method with exception."""
    # Create node
    node = RepositoryAnalyzerNode()
    
    # Mock analyzer.analyze to raise an exception
    node.analyzer.analyze.side_effect = Exception("Test error")
    
    # Create test state
    state = {
        "repository_url": "https://github.com/user/repo",
        "errors": [],
        "current_step": "start"
    }
    
    # Test node execution
    result = node(state)
    
    # Check result
    assert result["current_step"] == "analysis_failed"
    assert "Test error" in result["errors"]


def test_repository_analyzer_node_create_analysis_summary():
    """Test RepositoryAnalyzerNode _create_analysis_summary method."""
    from repository_analyzer.core.data_structures import (
        RepositoryStructure, RepositoryMetadata
    )
    
    # Create node
    node = RepositoryAnalyzerNode()
    
    # Create a mock RepositoryStructure
    structure = Mock(spec=RepositoryStructure)
    structure.metadata = Mock(spec=RepositoryMetadata)
    structure.metadata.name = "test-repo"
    structure.metadata.primary_language = "Python"
    structure.metadata.languages = ["Python", "JavaScript"]
    structure.metadata.frameworks = ["Flask", "React"]
    structure.metadata.architecture_type = "monolith"
    structure.metadata.complexity_score = 0.5
    structure.metadata.documentation_coverage = 0.8
    structure.metadata.test_coverage_estimate = 0.7
    structure.metadata.entry_points = ["main.py"]
    structure.metadata.configuration_files = ["config.json"]
    
    structure.files = {"file1": Mock(), "file2": Mock()}
    structure.directories = {"dir1": Mock(), "dir2": Mock()}
    structure.frameworks = [Mock(), Mock()]
    structure.patterns = [Mock(), Mock(), Mock()]
    structure.relationships = [Mock(), Mock()]
    
    # Test summary creation
    summary = node._create_analysis_summary(structure)
    
    # Check summary contents
    assert isinstance(summary, dict)
    assert summary["project_name"] == "test-repo"
    assert summary["primary_language"] == "Python"
    assert len(summary["languages"]) == 2
    assert len(summary["frameworks"]) == 2
    assert summary["architecture_type"] == "monolith"
    assert summary["complexity_score"] == 0.5
    assert summary["documentation_coverage"] == 0.8
    assert summary["test_coverage_estimate"] == 0.7
    assert len(summary["entry_points"]) == 1
    assert len(summary["configuration_files"]) == 1
    assert summary["total_files"] == 2
    assert summary["total_directories"] == 2
    assert summary["detected_frameworks"] == 2
    assert summary["detected_patterns"] == 3
    assert summary["mapped_relationships"] == 2


def test_repository_analysis_state_typed_dict():
    """Test RepositoryAnalysisState TypedDict."""
    # This is just to ensure the TypedDict is properly defined
    state: RepositoryAnalysisState = {
        "repository_url": "https://github.com/user/repo",
        "local_path": "/local/path",
        "analysis_config": None,
        "repository_structure": None,
        "analysis_summary": None,
        "current_step": "start",
        "errors": [],
        "warnings": []
    }
    
    assert state["current_step"] == "start"


@patch('repository_analyzer.langgraph.nodes.RepositoryAnalyzerNode')
def test_create_analysis_workflow(mock_node_class):
    """Test create_analysis_workflow function."""
    from langgraph import StateGraph
    
    # Mock the node class
    mock_node_instance = Mock()
    mock_node_class.return_value = mock_node_instance
    
    # Test workflow creation
    workflow = create_analysis_workflow()
    
    # Check that we got a workflow
    assert workflow is not None