"""Configuration management for input handling."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import os
from pathlib import Path


@dataclass
class InputConfig:
    """Comprehensive input handling configuration."""
    
    # Authentication settings
    enable_authentication: bool = True
    auth_timeout: int = 30
    
    # Validation settings
    max_path_length: int = 4096
    allowed_url_schemes: List[str] = field(default_factory=lambda: ['http', 'https', 'git', 'ssh'])
    enable_path_traversal_check: bool = True
    
    # Processing settings
    timeout: int = 300
    retry_attempts: int = 3
    retry_delay: float = 1.0
    max_repo_size: int = 500 * 1024 * 1024  # 500MB
    
    # Temporary directory settings
    temp_dir: Optional[str] = None
    auto_cleanup: bool = True
    cleanup_on_error: bool = True
    
    # Security settings
    mask_credentials_in_logs: bool = True
    validate_ssl_certificates: bool = True
    
    # Provider-specific settings
    provider_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize configuration with environment variables."""
        if self.temp_dir is None:
            self.temp_dir = os.environ.get("REPO_ANALYZER_TEMP_DIR", "/tmp/repo_analyzer")
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for a specific provider."""
        return self.provider_configs.get(provider, {})
    
    def set_provider_config(self, provider: str, config: Dict[str, Any]):
        """Set configuration for a specific provider."""
        self.provider_configs[provider] = config