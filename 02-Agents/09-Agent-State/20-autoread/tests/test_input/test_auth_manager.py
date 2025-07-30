"""Tests for the AuthManager class."""

import os
import tempfile
from pathlib import Path
import pytest

from repository_analyzer.input.auth.manager import AuthManager, AuthConfig
from repository_analyzer.input.auth.credentials import CredentialStore
from repository_analyzer.input.auth.strategies import TokenAuth, SSHAuth


class TestAuthManager:
    """Test cases for the AuthManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.auth_config = AuthConfig()
        self.auth_manager = AuthManager(self.auth_config)
    
    def test_init(self):
        """Test initialization of AuthManager."""
        assert isinstance(self.auth_manager.credential_store, CredentialStore)
        assert isinstance(self.auth_manager.strategies['token'], TokenAuth)
        assert isinstance(self.auth_manager.strategies['ssh'], SSHAuth)
    
    def test_get_auth_for_provider_with_token(self):
        """Test getting authentication for provider with token."""
        # Add a token to the credential store
        self.auth_manager.credential_store.add_token('github', 'test_token_123')
        
        strategy = self.auth_manager.get_auth_for_provider('github', 'https://github.com/user/repo.git')
        assert isinstance(strategy, TokenAuth)
        assert strategy.config.get('token') == 'test_token_123'
        assert strategy.config.get('provider') == 'github'
    
    def test_get_auth_for_provider_with_ssh(self):
        """Test getting authentication for provider with SSH key."""
        # Create a temporary SSH key file for testing
        temp_dir = tempfile.mkdtemp(prefix="ssh_test_")
        ssh_key_path = Path(temp_dir) / "id_rsa"
        ssh_key_path.write_text("fake ssh key content")
        
        # Add SSH key to the credential store
        self.auth_manager.credential_store.add_ssh_key('github', str(ssh_key_path))
        
        strategy = self.auth_manager.get_auth_for_provider('github', 'git@github.com:user/repo.git')
        assert isinstance(strategy, SSHAuth)
        assert strategy.config.get('key_path') == str(ssh_key_path)
        assert strategy.config.get('provider') == 'github'
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_authenticate_url_with_token(self):
        """Test authenticating URL with token."""
        # Add a token to the credential store
        self.auth_manager.credential_store.add_token('github', 'test_token_123')
        
        original_url = 'https://github.com/user/repo.git'
        authenticated_url = self.auth_manager.authenticate_url(original_url, 'github')
        
        # Should have inserted the token into the URL
        assert 'test_token_123' in authenticated_url
    
    def test_authenticate_url_without_credentials(self):
        """Test authenticating URL without credentials."""
        original_url = 'https://github.com/user/repo.git'
        authenticated_url = self.auth_manager.authenticate_url(original_url, 'github')
        
        # Should return the original URL unchanged
        assert authenticated_url == original_url
    
    def test_validate_credentials_valid(self):
        """Test validating valid credentials."""
        # Add a token to the credential store
        self.auth_manager.credential_store.add_token('github', 'test_token_123')
        
        is_valid = self.auth_manager.validate_credentials('github')
        assert is_valid
    
    def test_validate_credentials_invalid(self):
        """Test validating invalid credentials."""
        # Test with provider that has no credentials
        is_valid = self.auth_manager.validate_credentials('nonexistent')
        assert not is_valid
    
    def test_is_ssh_url(self):
        """Test the _is_ssh_url helper method."""
        # Test SSH URLs
        assert self.auth_manager._is_ssh_url('git@github.com:user/repo.git')
        assert self.auth_manager._is_ssh_url('ssh://git@example.com/user/repo.git')
        
        # Test non-SSH URLs
        assert not self.auth_manager._is_ssh_url('https://github.com/user/repo.git')
        assert not self.auth_manager._is_ssh_url('/local/path')