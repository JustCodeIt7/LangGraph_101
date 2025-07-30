"""Tests for the InputClassifier class."""

import pytest

from repository_analyzer.input.classifier import InputClassifier, InputType, ClassificationResult


class TestInputClassifier:
    """Test cases for the InputClassifier class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = InputClassifier()
    
    def test_classify_local_path(self):
        """Test classification of local paths."""
        local_paths = [
            "/home/user/project",
            "./relative/path",
            "../parent/path",
            "relative/path"
        ]
        
        for path in local_paths:
            result = self.classifier.classify(path)
            assert isinstance(result, ClassificationResult)
            assert result.input_type == InputType.LOCAL_PATH
            assert result.confidence == 1.0
    
    def test_classify_github_url(self):
        """Test classification of GitHub URLs."""
        github_urls = [
            "https://github.com/user/repo.git",
            "https://github.com/user/repo",
            "git@github.com:user/repo.git",
            "git@github.com:user/repo"
        ]
        
        for url in github_urls:
            result = self.classifier.classify(url)
            assert isinstance(result, ClassificationResult)
            assert result.input_type == InputType.GITHUB_URL
            assert result.provider == "github"
            assert result.confidence >= 0.8
    
    def test_classify_gitlab_url(self):
        """Test classification of GitLab URLs."""
        gitlab_urls = [
            "https://gitlab.com/user/repo.git",
            "https://gitlab.com/user/repo",
            "git@gitlab.com:user/repo.git"
        ]
        
        for url in gitlab_urls:
            result = self.classifier.classify(url)
            assert isinstance(result, ClassificationResult)
            assert result.input_type == InputType.GITLAB_URL
            assert result.provider == "gitlab"
            assert result.confidence >= 0.8
    
    def test_classify_bitbucket_url(self):
        """Test classification of Bitbucket URLs."""
        bitbucket_urls = [
            "https://bitbucket.org/user/repo.git",
            "https://bitbucket.org/user/repo",
            "git@bitbucket.org:user/repo.git"
        ]
        
        for url in bitbucket_urls:
            result = self.classifier.classify(url)
            assert isinstance(result, ClassificationResult)
            assert result.input_type == InputType.BITBUCKET_URL
            assert result.provider == "bitbucket"
            assert result.confidence >= 0.8
    
    def test_classify_generic_git_url(self):
        """Test classification of generic Git URLs."""
        generic_urls = [
            "https://example.com/repo.git",
            "git@example.com:user/repo.git",
            "ssh://git@example.com/user/repo.git"
        ]
        
        for url in generic_urls:
            result = self.classifier.classify(url)
            # Should be classified as generic Git URL if it contains Git indicators
            assert isinstance(result, ClassificationResult)
            if '.git' in url or url.startswith('git@') or 'git://' in url:
                assert result.input_type == InputType.GENERIC_GIT_URL
                assert result.provider == "generic"
                assert result.confidence >= 0.7
    
    def test_classify_unknown_input(self):
        """Test classification of unknown input types."""
        unknown_inputs = [
            "not a path or url",
            "invalid://format",
            ""  # Empty string
        ]
        
        for input_str in unknown_inputs:
            result = self.classifier.classify(input_str)
            assert isinstance(result, ClassificationResult)
            # Should either be UNKNOWN or LOCAL_PATH (for non-URL strings)
            assert result.input_type in [InputType.UNKNOWN, InputType.LOCAL_PATH]
    
    def test_is_local_path(self):
        """Test the _is_local_path helper method."""
        # Test local paths
        assert self.classifier._is_local_path("/home/user/project")
        assert self.classifier._is_local_path("./relative/path")
        assert self.classifier._is_local_path("../parent/path")
        assert self.classifier._is_local_path("relative/path")
        
        # Test URLs
        assert not self.classifier._is_local_path("https://github.com/user/repo")
        assert not self.classifier._is_local_path("git@github.com:user/repo.git")
    
    def test_is_url(self):
        """Test the _is_url helper method."""
        # Test URLs
        assert self.classifier._is_url("https://github.com/user/repo")
        assert self.classifier._is_url("git@github.com:user/repo.git")
        assert self.classifier._is_url("http://example.com")
        
        # Test non-URLs
        assert not self.classifier._is_url("/home/user/project")
        assert not self.classifier._is_url("./relative/path")
        assert not self.classifier._is_url("not a url")