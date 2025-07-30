"""Authentication management system."""

from typing import Dict, Optional, List
from dataclasses import dataclass
import os
import json
from pathlib import Path
from .credentials import CredentialStore
from .strategies import AuthStrategy, TokenAuth, SSHAuth
from ..exceptions import InputAuthenticationError


@dataclass
class AuthConfig:
    """Authentication configuration."""
    enable_env_vars: bool = True
    enable_config_file: bool = True
    enable_ssh_keys: bool = True
    config_file_path: Optional[str] = None
    ssh_key_paths: List[str] = None
    
    def __post_init__(self):
        if self.ssh_key_paths is None:
            self.ssh_key_paths = [
                "~/.ssh/id_rsa",
                "~/.ssh/id_ed25519",
                "~/.ssh/id_ecdsa"
            ]


class AuthManager:
    """Centralized authentication management."""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.credential_store = CredentialStore(config)
        self.strategies: Dict[str, AuthStrategy] = {
            'token': TokenAuth(),
            'ssh': SSHAuth()
        }
    
    def get_auth_for_provider(self, provider: str, url: str) -> Optional[AuthStrategy]:
        """Get appropriate authentication for a Git provider."""
        # Try token authentication first
        token = self.credential_store.get_token(provider)
        if token:
            strategy = self.strategies['token']
            strategy.configure(token=token, provider=provider)
            return strategy
        
        # Try SSH authentication
        ssh_key = self.credential_store.get_ssh_key(provider)
        if ssh_key and self._is_ssh_url(url):
            strategy = self.strategies['ssh']
            strategy.configure(key_path=ssh_key, provider=provider)
            return strategy
        
        return None
    
    def authenticate_url(self, url: str, provider: str) -> str:
        """Apply authentication to URL if credentials are available."""
        auth_strategy = self.get_auth_for_provider(provider, url)
        if auth_strategy:
            return auth_strategy.apply_auth(url)
        return url
    
    def validate_credentials(self, provider: str) -> bool:
        """Validate stored credentials for a provider."""
        try:
            auth_strategy = self.get_auth_for_provider(provider, "")
            return auth_strategy is not None and auth_strategy.validate()
        except Exception:
            return False
    
    def _is_ssh_url(self, url: str) -> bool:
        """Check if URL is SSH-based."""
        return url.startswith('git@') or url.startswith('ssh://')