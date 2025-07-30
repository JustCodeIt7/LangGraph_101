"""Tests for main analyzer."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from repository_analyzer.core.analyzer import RepositoryAnalyzer
from repository_analyzer.core.config import AnalysisConfig
from repository_analyzer.core.data_structures import RepositoryStructure


def test_analyzer_initialization():
    """Test RepositoryAnalyzer initialization."""
    analyzer = RepositoryAnalyzer()
    
    assert analyzer is not None
    assert analyzer.config is not None
    assert analyzer.git_cloner is not None
    assert analyzer.file_scanner is not None


def test_analyzer_initialization_with_config():
    """Test RepositoryAnalyzer initialization with custom config."""
    config = AnalysisConfig(max_depth=5)
    analyzer = RepositoryAnalyzer(config)
    
    assert analyzer.config == config


@patch('repository_analyzer.core.analyzer.GitCloner')
@patch('repository_analyzer.core.analyzer.FileSystemScanner')
@patch('repository_analyzer.core.analyzer.FileCataloger')
@patch('repository_analyzer.core.analyzer.PatternDetector')
@patch('repository_analyzer.core.analyzer.FrameworkDetector')
@patch('repository_analyzer.core.analyzer.RelationshipMapper')
@patch('repository_analyzer.core.analyzer.ImportAnalyzer')
@patch('repository_analyzer.core.analyzer.ConfigFileParser')
def test_analyzer_analyze(mock_config_parser, mock_import_analyzer, mock_relationship_mapper, 
                          mock_framework_detector, mock_pattern_detector, mock_file_cataloger, 
                          mock_file_scanner, mock_git_cloner):
    """Test RepositoryAnalyzer analyze method."""
    # Create analyzer
    analyzer = RepositoryAnalyzer()
    
    # Mock all dependencies
    mock_git_cloner.return_value.clone_repository.return_value = "/tmp/test_repo"
    mock_git_cloner.return_value.cleanup_temp_repository.return_value = True
    
    mock_file_scanner.return_value.scan_repository.return_value = ({}, {})
    
    mock_file_cataloger.return_value.catalog_files.return_value = {}
    mock_file_cataloger.return_value.catalog_directories.return_value = {}
    
    mock_pattern_detector.return_value.detect_patterns.return_value = []
    mock_pattern_detector.return_value.detect_project_type.return_value = "monolith"
    
    mock_framework_detector.return_value.detect_frameworks.return_value = []
    
    mock_relationship_mapper.return_value.map_relationships.return_value = []
    
    mock_import_analyzer.return_value.analyze_imports.return_value = {}
    
    # Test analysis
    with patch.object(analyzer, '_prepare_repository', return_value="/tmp/test_repo"):
        with patch.object(analyzer, '_create_repository_metadata', return_value=Mock()):
            result = analyzer.analyze("https://github.com/user/repo")
            
            # Should return a RepositoryStructure
            assert isinstance(result, RepositoryStructure)


def test_analyzer_cleanup():
    """Test RepositoryAnalyzer cleanup method."""
    analyzer = RepositoryAnalyzer()
    
    # Add a temp directory
    analyzer._temp_dirs.append("/tmp/test_dir")
    
    # Test cleanup
    with patch('repository_analyzer.core.analyzer.os.path.exists', return_value=True):
        with patch('repository_analyzer.core.analyzer.shutil.rmtree') as mock_rmtree:
            analyzer.cleanup()
            
            # Check that cleanup was called
            assert len(analyzer._temp_dirs) == 0


@patch('repository_analyzer.core.analyzer.os.path.exists')
def test_analyzer_prepare_repository_local_path(mock_exists):
    """Test _prepare_repository method with local path."""
    analyzer = RepositoryAnalyzer()
    mock_exists.return_value = True
    
    # Test with existing local path
    result = analyzer._prepare_repository("/path/to/local/repo")
    
    assert result == "/path/to/local/repo"


@patch('repository_analyzer.core.analyzer.os.path.exists')
def test_analyzer_prepare_repository_nonexistent_local_path(mock_exists):
    """Test _prepare_repository method with non-existent local path."""
    from repository_analyzer.core.exceptions import RepositoryNotFoundError
    
    analyzer = RepositoryAnalyzer()
    mock_exists.return_value = False
    
    # Test with non-existent local path
    with pytest.raises(RepositoryNotFoundError):
        analyzer._prepare_repository("/path/to/nonexistent/repo")


def test_analyzer_calculate_complexity_score():
    """Test _calculate_complexity_score method."""
    analyzer = RepositoryAnalyzer()
    
    # Test with empty files and directories
    score = analyzer._calculate_complexity_score({}, {})
    
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_analyzer_calculate_documentation_coverage():
    """Test _calculate_documentation_coverage method."""
    analyzer = RepositoryAnalyzer()
    
    # Test with empty files
    coverage = analyzer._calculate_documentation_coverage({})
    
    assert isinstance(coverage, float)
    assert coverage == 0.0


def test_analyzer_calculate_test_coverage():
    """Test _calculate_test_coverage method."""
    analyzer = RepositoryAnalyzer()
    
    # Test with empty files
    coverage = analyzer._calculate_test_coverage({})
    
    assert isinstance(coverage, float)
    assert coverage == 0.0


def test_analyzer_find_entry_points():
    """Test _find_entry_points method."""
    analyzer = RepositoryAnalyzer()
    
    # Test with empty files
    entry_points = analyzer._find_entry_points({})
    
    assert isinstance(entry_points, list)
    assert len(entry_points) == 0


def test_analyzer_find_configuration_files():
    """Test _find_configuration_files method."""
    analyzer = RepositoryAnalyzer()
    
    # Test with empty files
    config_files = analyzer._find_configuration_files({})
    
    assert isinstance(config_files, list)
    assert len(config_files) == 0