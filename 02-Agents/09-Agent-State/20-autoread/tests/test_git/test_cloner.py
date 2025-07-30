"""Tests for Git cloner module."""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from repository_analyzer.git.cloner import GitCloner
from repository_analyzer.core.config import AnalysisConfig
from repository_analyzer.core.exceptions import RepositoryNotFoundError, AuthenticationError, GitError


def test_git_cloner_initialization():
    """Test GitCloner initialization."""
    config = AnalysisConfig()
    cloner = GitCloner(config)
    
    assert cloner.config == config
    assert cloner.temp_dir == config.get_temp_dir()


def test_git_cloner_cleanup_temp_repository():
    """Test cleanup_temp_repository method."""
    config = AnalysisConfig()
    cloner = GitCloner(config)
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = os.path.join(temp_dir, "test_repo")
        os.makedirs(test_dir)
        
        # Create a test file
        test_file = os.path.join(test_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        
        # Test cleanup
        result = cloner.cleanup_temp_repository(test_dir)
        assert result is True
        assert not os.path.exists(test_dir)


def test_git_cloner_cleanup_non_temp_repository():
    """Test cleanup_temp_repository with non-temp directory."""
    config = AnalysisConfig()
    cloner = GitCloner(config)
    
    # Try to cleanup a non-temp directory (should return False)
    result = cloner.cleanup_temp_repository("/non/temp/path")
    assert result is False


@patch('repository_analyzer.git.cloner.git')
def test_git_cloner_clone_github_repository_success(mock_git):
    """Test successful GitHub repository cloning."""
    config = AnalysisConfig()
    cloner = GitCloner(config)
    
    # Mock git.Repo.clone_from to return a mock repo
    mock_repo = Mock()
    mock_git.Repo.clone_from.return_value = mock_repo
    
    # Test cloning
    with tempfile.TemporaryDirectory() as temp_dir:
        result = cloner._clone_github_repository("https://github.com/user/repo.git", temp_dir)
        
        assert result == temp_dir
        mock_git.Repo.clone_from.assert_called_once_with("https://github.com/user/repo.git", temp_dir)


@patch('repository_analyzer.git.cloner.git')
def test_git_cloner_clone_github_repository_auth_failure(mock_git):
    """Test GitHub repository cloning with authentication failure."""
    config = AnalysisConfig()
    cloner = GitCloner(config)
    
    # Mock git.Repo.clone_from to raise GitCommandError with auth failure
    mock_git.exc.GitCommandError = Exception  # Mock the exception class
    mock_git.Repo.clone_from.side_effect = Exception("Authentication failed")
    
    # Test cloning
    with pytest.raises(AuthenticationError):
        cloner._clone_github_repository("https://github.com/user/private-repo.git")


@patch('repository_analyzer.git.cloner.git')
def test_git_cloner_clone_github_repository_not_found(mock_git):
    """Test GitHub repository cloning with repository not found."""
    config = AnalysisConfig()
    cloner = GitCloner(config)
    
    # Mock git.Repo.clone_from to raise GitCommandError with not found
    mock_git.exc.GitCommandError = Exception  # Mock the exception class
    mock_git.Repo.clone_from.side_effect = Exception("Repository not found")
    
    # Test cloning
    with pytest.raises(GitError):
        cloner._clone_github_repository("https://github.com/user/nonexistent-repo.git")


@patch('repository_analyzer.git.cloner.shutil')
def test_git_cloner_handle_local_repository_success(mock_shutil):
    """Test successful handling of local repository."""
    config = AnalysisConfig()
    cloner = GitCloner(config)
    
    # Mock shutil.copytree to do nothing
    mock_shutil.copytree.return_value = None
    
    # Create a mock source path
    with tempfile.TemporaryDirectory() as temp_dir:
        source_path = os.path.join(temp_dir, "source_repo")
        os.makedirs(source_path)
        
        # Test handling local repository
        with tempfile.TemporaryDirectory() as target_dir:
            result = cloner._handle_local_repository(source_path, target_dir)
            
            assert result is not None
            mock_shutil.copytree.assert_called_once()


def test_git_cloner_handle_local_repository_not_found():
    """Test handling of non-existent local repository."""
    config = AnalysisConfig()
    cloner = GitCloner(config)
    
    # Test with non-existent source path
    with pytest.raises(RepositoryNotFoundError):
        cloner._handle_local_repository("/non/existent/path")


@patch('repository_analyzer.git.cloner.os')
def test_git_cloner_cleanup_temp_repository_exception(mock_os):
    """Test cleanup_temp_repository when an exception occurs."""
    config = AnalysisConfig()
    cloner = GitCloner(config)
    
    # Mock os.path.exists to return True
    mock_os.path.exists.return_value = True
    
    # Mock os.path.join to return a temp path
    mock_os.path.join.return_value = "/tmp/repo_analyzer/test"
    
    # Mock shutil.rmtree to raise an exception
    mock_os.rmdir.side_effect = Exception("Permission denied")
    
    # Test cleanup (should return False when exception occurs)
    result = cloner.cleanup_temp_repository("/tmp/repo_analyzer/test")
    assert result is False