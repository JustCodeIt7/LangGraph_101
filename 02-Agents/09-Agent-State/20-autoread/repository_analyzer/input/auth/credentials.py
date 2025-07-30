"""Credential storage and retrieval."""

from typing import Optional, Dict, List
import os
from pathlib import Path


class CredentialStore:
    """Centralized credential storage and retrieval."""
    
    def __init__(self, config):
        self.config = config
        self._tokens = {}
        self._ssh_keys = {}
        self._load_from_environment()
    
    def _load_from_environment(self):
        """Load credentials from environment variables."""
        # GitHub tokens
        github_token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GITHUB_API_TOKEN')
        if github_token:
            self._tokens['github'] = github_token
        
        # GitLab tokens
        gitlab_token = os.environ.get('GITLAB_TOKEN') or os.environ.get('GITLAB_API_TOKEN')
        if gitlab_token:
            self._tokens['gitlab'] = gitlab_token
        
        # Bitbucket tokens
        bitbucket_token = os.environ.get('BITBUCKET_TOKEN') or os.environ.get('BITBUCKET_API_TOKEN')
        if bitbucket_token:
            self._tokens['bitbucket'] = bitbucket_token
        
        # SSH keys from environment
        ssh_key_path = os.environ.get('SSH_KEY_PATH')
        if ssh_key_path and os.path.exists(ssh_key_path):
            self._ssh_keys['github'] = ssh_key_path
    
    def get_token(self, provider: str) -> Optional[str]:
        """Get authentication token for a Git provider."""
        return self._tokens.get(provider.lower())
    
    def get_ssh_key(self, provider: str) -> Optional[str]:
        """Get SSH key path for a Git provider."""
        return self._ssh_keys.get(provider.lower())
    
    def add_token(self, provider: str, token: str) -> None:
        """Add an authentication token for a Git provider."""
        self._tokens[provider.lower()] = token
    
    def add_ssh_key(self, provider: str, key_path: str) -> None:
        """Add an SSH key for a Git provider."""
        if not os.path.exists(key_path):
            raise ValueError(f"SSH key file not found: {key_path}")
        self._ssh_keys[provider.lower()] = key_path
    
    def get_auth_method(self, provider: str) -> str:
        """Determine the authentication method for a provider."""
        provider_lower = provider.lower()
        if provider_lower in self._tokens:
            return 'token'
        elif provider_lower in self._ssh_keys:
            return 'ssh'
        else:
            return 'none'
    
    def list_providers(self) -> List[str]:
        """List all providers with stored credentials."""
        providers = set()
        providers.update(self._tokens.keys())
        providers.update(self._ssh_keys.keys())
        return list(providers)