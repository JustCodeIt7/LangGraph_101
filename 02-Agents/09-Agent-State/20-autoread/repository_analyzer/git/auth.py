"""Git authentication handling."""

import os
from typing import Optional, Dict, Any
from ..core.exceptions import AuthenticationError


class GitAuthManager:
    """Manages Git authentication for repository cloning."""
    
    def __init__(self):
        """Initialize the authentication manager."""
        self.tokens: Dict[str, str] = {}
        self.ssh_keys: Dict[str, str] = {}
    
    def add_token(self, provider: str, token: str) -> None:
        """Add an authentication token for a Git provider.
        
        Args:
            provider: Git provider (e.g., 'github', 'gitlab', 'bitbucket')
            token: Authentication token
        """
        self.tokens[provider.lower()] = token
    
    def add_ssh_key(self, provider: str, key_path: str) -> None:
        """Add an SSH key for a Git provider.
        
        Args:
            provider: Git provider (e.g., 'github', 'gitlab', 'bitbucket')
            key_path: Path to SSH private key
        """
        if not os.path.exists(key_path):
            raise AuthenticationError(f"SSH key file not found: {key_path}")
        self.ssh_keys[provider.lower()] = key_path
    
    def get_token(self, provider: str) -> Optional[str]:
        """Get authentication token for a Git provider.
        
        Args:
            provider: Git provider
            
        Returns:
            Authentication token or None if not found
        """
        return self.tokens.get(provider.lower())
    
    def get_ssh_key(self, provider: str) -> Optional[str]:
        """Get SSH key path for a Git provider.
        
        Args:
            provider: Git provider
            
        Returns:
            SSH key path or None if not found
        """
        return self.ssh_keys.get(provider.lower())
    
    def get_auth_method(self, provider: str) -> str:
        """Determine the authentication method for a provider.
        
        Args:
            provider: Git provider
            
        Returns:
            Authentication method ('token', 'ssh', or 'none')
        """
        provider_lower = provider.lower()
        if provider_lower in self.tokens:
            return 'token'
        elif provider_lower in self.ssh_keys:
            return 'ssh'
        else:
            return 'none'
    
    def load_from_environment(self) -> None:
        """Load authentication credentials from environment variables."""
        # GitHub token
        github_token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GITHUB_API_TOKEN')
        if github_token:
            self.add_token('github', github_token)
        
        # GitLab token
        gitlab_token = os.environ.get('GITLAB_TOKEN') or os.environ.get('GITLAB_API_TOKEN')
        if gitlab_token:
            self.add_token('gitlab', gitlab_token)
        
        # Bitbucket token
        bitbucket_token = os.environ.get('BITBUCKET_TOKEN') or os.environ.get('BITBUCKET_API_TOKEN')
        if bitbucket_token:
            self.add_token('bitbucket', bitbucket_token)
        
        # SSH keys from environment
        ssh_key_path = os.environ.get('SSH_KEY_PATH')
        if ssh_key_path and os.path.exists(ssh_key_path):
            # Assume GitHub by default for SSH
            self.add_ssh_key('github', ssh_key_path)
    
    def create_authenticated_url(self, url: str) -> str:
        """Create an authenticated URL for a Git repository.
        
        Args:
            url: Original Git repository URL
            
        Returns:
            Authenticated Git repository URL
        """
        # Determine provider from URL
        provider = self._extract_provider_from_url(url)
        
        # Apply authentication if available
        if provider:
            auth_method = self.get_auth_method(provider)
            if auth_method == 'token':
                token = self.get_token(provider)
                if token and url.startswith('https://'):
                    # Insert token in URL
                    domain_start = url.find('://') + 3
                    domain_end = url.find('/', domain_start)
                    if domain_end == -1:
                        domain_end = len(url)
                    domain = url[domain_start:domain_end]
                    url = url.replace(f"https://{domain}", f"https://oauth2:{token}@{domain}")
            # For SSH, no URL modification needed
        
        return url
    
    def _extract_provider_from_url(self, url: str) -> Optional[str]:
        """Extract Git provider from URL.
        
        Args:
            url: Git repository URL
            
        Returns:
            Provider name or None if not recognized
        """
        if 'github.com' in url:
            return 'github'
        elif 'gitlab.com' in url:
            return 'gitlab'
        elif 'bitbucket.org' in url:
            return 'bitbucket'
        else:
            return None