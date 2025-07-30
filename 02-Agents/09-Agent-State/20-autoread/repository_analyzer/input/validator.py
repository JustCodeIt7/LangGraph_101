"""Input validation framework."""

from typing import List, Optional
from urllib.parse import urlparse
import os
import re
from .exceptions import InputValidationError
from .config import InputConfig
from .classifier import InputType, ClassificationResult


class ValidationResult:
    """Result of input validation."""
    
    def __init__(self, is_valid: bool, input_type: InputType, normalized_source: str, metadata: Optional[dict] = None):
        self.is_valid = is_valid
        self.input_type = input_type
        self.normalized_source = normalized_source
        self.metadata = metadata or {}


class InputValidator:
    """Comprehensive input validation framework."""
    
    # URL validation patterns
    GITHUB_PATTERNS = [
        r'^https://github\.com/[\w\-\.]+/[\w\-\.]+(?:\.git)?/?$',
        r'^git@github\.com:[\w\-\.]+/[\w\-\.]+(?:\.git)?$'
    ]
    
    GITLAB_PATTERNS = [
        r'^https://gitlab\.com/[\w\-\.]+/[\w\-\.]+(?:\.git)?/?$',
        r'^git@gitlab\.com:[\w\-\.]+/[\w\-\.]+(?:\.git)?$'
    ]
    
    def __init__(self, config: InputConfig):
        self.config = config
        self.max_path_length = config.max_path_length
        self.allowed_schemes = config.allowed_url_schemes
    
    def validate(self, source: str) -> ValidationResult:
        """Validate input source with comprehensive checks."""
        if not source or not source.strip():
            raise InputValidationError("Empty input source provided")
        
        source = source.strip()
        
        # Length validation
        if len(source) > self.max_path_length:
            raise InputValidationError(f"Input source too long: {len(source)} > {self.max_path_length}")
        
        # Determine input type and validate accordingly
        if self._is_local_path(source):
            return self._validate_local_path(source)
        elif self._is_url(source):
            return self._validate_url(source)
        else:
            raise InputValidationError(f"Invalid input format: {source}")
    
    def _is_local_path(self, source: str) -> bool:
        """Check if source is a local path."""
        return source.startswith("/") or source.startswith("./") or source.startswith("../") or not self._is_url(source)
    
    def _is_url(self, source: str) -> bool:
        """Check if source is a URL."""
        try:
            parsed = urlparse(source)
            return bool(parsed.scheme) and bool(parsed.netloc)
        except Exception:
            return False
    
    def _validate_local_path(self, path: str) -> ValidationResult:
        """Validate local file system path."""
        # Path traversal prevention
        if self.config.enable_path_traversal_check:
            normalized_path = os.path.normpath(path)
            if '..' in normalized_path:
                raise InputValidationError("Path traversal detected in local path")
        
        # Existence check
        if not os.path.exists(path):
            raise InputValidationError(f"Local path does not exist: {path}")
        
        # Permission check
        if not os.access(path, os.R_OK):
            raise InputValidationError(f"No read permission for local path: {path}")
        
        return ValidationResult(
            is_valid=True,
            input_type=InputType.LOCAL_PATH,
            normalized_source=os.path.abspath(path),
            metadata={'original_path': path}
        )
    
    def _validate_url(self, url: str) -> ValidationResult:
        """Validate URL format and accessibility."""
        parsed = urlparse(url)
        
        # Scheme validation
        if parsed.scheme not in self.allowed_schemes:
            raise InputValidationError(f"Unsupported URL scheme: {parsed.scheme}")
        
        # Host validation
        if not parsed.netloc:
            raise InputValidationError("Invalid URL: missing host")
        
        # Provider-specific validation
        provider = self._detect_provider(url)
        if provider:
            self._validate_provider_url(url, provider)
        
        return ValidationResult(
            is_valid=True,
            input_type=self._classify_url_type(url),
            normalized_source=url,  # In a real implementation, this would be normalized
            metadata={'provider': provider, 'original_url': url}
        )
    
    def _detect_provider(self, url: str) -> Optional[str]:
        """Detect Git provider from URL."""
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        
        if 'github.com' in host:
            return 'github'
        elif 'gitlab.com' in host:
            return 'gitlab'
        elif 'bitbucket.org' in host:
            return 'bitbucket'
        
        return None
    
    def _classify_url_type(self, url: str) -> InputType:
        """Classify URL type."""
        provider = self._detect_provider(url)
        if provider:
            return getattr(InputType, f"{provider.upper()}_URL")
        elif '.git' in url or url.startswith('git@') or url.startswith('git://'):
            return InputType.GENERIC_GIT_URL
        else:
            return InputType.UNKNOWN
    
    def _validate_provider_url(self, url: str, provider: str) -> None:
        """Validate provider-specific URL patterns."""
        # In a real implementation, this would check against specific patterns
        # For now, we'll just ensure it's a valid URL structure
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise InputValidationError(f"Invalid {provider} URL format: {url}")