"""Authentication strategies for different providers."""

from typing import Optional, Dict
from abc import ABC, abstractmethod


class AuthStrategy(ABC):
    """Base authentication strategy interface."""
    
    def __init__(self):
        self.provider = None
        self.config = {}
    
    def configure(self, **kwargs):
        """Configure the authentication strategy."""
        self.provider = kwargs.get('provider')
        self.config = kwargs
    
    @abstractmethod
    def apply_auth(self, url: str) -> str:
        """Apply authentication to a URL."""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate that credentials are working."""
        pass


class TokenAuth(AuthStrategy):
    """Token-based authentication strategy."""
    
    def apply_auth(self, url: str) -> str:
        """Apply token authentication to URL."""
        token = self.config.get('token')
        provider = self.provider
        
        if not token or not provider:
            return url
        
        # Apply token to URL based on provider
        if provider == 'github' and url.startswith('https://'):
            if '@' not in url.replace('://', '://dummy@'):  # Check if auth already exists
                return url.replace('https://', f'https://{token}@')
        elif provider in ['gitlab', 'bitbucket'] and url.startswith('https://'):
            if '@' not in url.replace('://', '://dummy@'):
                # Extract domain
                domain_start = url.find('://') + 3
                domain_end = url.find('/', domain_start)
                if domain_end == -1:
                    domain_end = len(url)
                domain = url[domain_start:domain_end]
                return url.replace(f"https://{domain}", f"https://oauth2:{token}@{domain}")
        
        return url
    
    def validate(self) -> bool:
        """Validate token authentication."""
        # In a real implementation, this would test the token against the provider's API
        token = self.config.get('token')
        return bool(token) and len(token) > 0


class SSHAuth(AuthStrategy):
    """SSH key-based authentication strategy."""
    
    def apply_auth(self, url: str) -> str:
        """Apply SSH authentication to URL."""
        key_path = self.config.get('key_path')
        
        # For SSH URLs, no modification is typically needed
        # The key is used by the Git client based on SSH config
        if url.startswith('git@') or 'ssh://' in url:
            return url
        
        # For HTTPS URLs, we might want to convert to SSH
        # This is a simplified implementation
        if url.startswith('https://github.com/'):
            # Extract owner/repo from URL
            path_parts = url.replace('https://github.com/', '').strip('/').split('/')
            if len(path_parts) >= 2:
                owner = path_parts[0]
                repo = path_parts[1].replace('.git', '')
                return f'git@github.com:{owner}/{repo}.git'
        
        return url
    
    def validate(self) -> bool:
        """Validate SSH key authentication."""
        key_path = self.config.get('key_path')
        if not key_path:
            return False
        
        # Check if key file exists
        import os
        return os.path.exists(key_path) and os.access(key_path, os.R_OK)