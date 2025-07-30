# InputHandler Implementation Specification

## File Structure and Module Organization

```
repository_analyzer/
├── input/
│   ├── __init__.py
│   ├── handler.py              # Main InputHandler class
│   ├── validator.py            # Input validation framework
│   ├── classifier.py           # Input source detection and classification
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── manager.py          # AuthManager class
│   │   ├── credentials.py      # Credential storage and retrieval
│   │   ├── strategies.py       # Authentication strategies
│   │   └── providers.py        # Provider-specific auth logic
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── base.py             # Base processor interface
│   │   ├── local.py            # Local path processor
│   │   ├── github.py           # GitHub URL processor
│   │   ├── gitlab.py           # GitLab URL processor
│   │   └── generic.py          # Generic Git URL processor
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── url_parser.py       # URL parsing and normalization
│   │   ├── temp_manager.py     # Temporary directory management
│   │   └── cleanup.py          # Resource cleanup utilities
│   ├── config.py              # InputHandler configuration
│   └── exceptions.py          # Input-specific exceptions
```

## Implementation Details

### 1. Input Validation Framework (`validator.py`)

```python
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import os
import re
from .exceptions import InputValidationError
from .config import InputConfig

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
        self.max_path_length = 4096
        self.allowed_schemes = ['http', 'https', 'git', 'ssh']
    
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
    
    def _validate_local_path(self, path: str) -> ValidationResult:
        """Validate local file system path."""
        # Path traversal prevention
        normalized_path = os.path.normpath(path)
        if '..' in normalized_path:
            raise InputValidationError("Path traversal detected in local path")
        
        # Existence check
        if not os.path.exists(normalized_path):
            raise InputValidationError(f"Local path does not exist: {path}")
        
        # Permission check
        if not os.access(normalized_path, os.R_OK):
            raise InputValidationError(f"No read permission for local path: {path}")
        
        return ValidationResult(
            is_valid=True,
            input_type=InputType.LOCAL_PATH,
            normalized_source=normalized_path,
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
            normalized_source=self._normalize_url(url),
            metadata={'provider': provider, 'original_url': url}
        )
```

### 2. Input Classification System (`classifier.py`)

```python
from enum import Enum
from typing import Optional
from urllib.parse import urlparse
import re

class InputType(Enum):
    LOCAL_PATH = "local_path"
    GITHUB_URL = "github_url"
    GITLAB_URL = "gitlab_url"
    BITBUCKET_URL = "bitbucket_url"
    GENERIC_GIT_URL = "generic_git_url"

class InputClassifier:
    """Intelligent input source classification."""
    
    def __init__(self):
        self.provider_patterns = {
            'github': [
                r'github\.com',
                r'raw\.githubusercontent\.com'
            ],
            'gitlab': [
                r'gitlab\.com',
                r'gitlab\.[\w\-\.]+\.com'
            ],
            'bitbucket': [
                r'bitbucket\.org',
                r'bitbucket\.[\w\-\.]+\.com'
            ]
        }
    
    def classify(self, source: str) -> ClassificationResult:
        """Classify input source type and extract metadata."""
        if self._is_local_path(source):
            return self._classify_local_path(source)
        elif self._is_url(source):
            return self._classify_url(source)
        else:
            return ClassificationResult(
                input_type=InputType.UNKNOWN,
                confidence=0.0,
                provider=None,
                metadata={}
            )
    
    def _classify_url(self, url: str) -> ClassificationResult:
        """Classify URL-based input sources."""
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        
        # Provider detection
        for provider, patterns in self.provider_patterns.items():
            for pattern in patterns:
                if re.search(pattern, host):
                    return ClassificationResult(
                        input_type=getattr(InputType, f"{provider.upper()}_URL"),
                        confidence=0.95,
                        provider=provider,
                        metadata={
                            'host': host,
                            'scheme': parsed.scheme,
                            'path': parsed.path
                        }
                    )
        
        # Generic Git URL detection
        if any(indicator in url.lower() for indicator in ['.git', 'git@', 'git://']):
            return ClassificationResult(
                input_type=InputType.GENERIC_GIT_URL,
                confidence=0.8,
                provider='generic',
                metadata={'host': host, 'scheme': parsed.scheme}
            )
        
        return ClassificationResult(
            input_type=InputType.UNKNOWN,
            confidence=0.0,
            provider=None,
            metadata={'host': host}
        )
```

### 3. Authentication Management System (`auth/manager.py`)

```python
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
```

### 4. Temporary Directory Management (`utils/temp_manager.py`)

```python
import tempfile
import shutil
import weakref
from pathlib import Path
from typing import Optional, Dict, Set
from contextlib import contextmanager
from ..exceptions import ResourceManagementError

class TempDirectoryManager:
    """Advanced temporary directory management with cleanup tracking."""
    
    def __init__(self, base_temp_dir: Optional[str] = None):
        self.base_temp_dir = Path(base_temp_dir) if base_temp_dir else Path(tempfile.gettempdir())
        self.active_dirs: Dict[str, Path] = {}
        self.cleanup_callbacks: Dict[str, callable] = {}
        
        # Use weak references to ensure cleanup on object destruction
        weakref.finalize(self, self._cleanup_all)
    
    def create_temp_directory(self, prefix: str = "repo_analyzer_") -> str:
        """Create a new temporary directory with tracking."""
        temp_dir = tempfile.mkdtemp(prefix=prefix, dir=self.base_temp_dir)
        temp_path = Path(temp_dir)
        
        # Track the directory
        dir_id = f"{prefix}_{id(temp_path)}"
        self.active_dirs[dir_id] = temp_path
        
        return str(temp_path)
    
    def cleanup_directory(self, dir_path: str) -> bool:
        """Clean up a specific temporary directory."""
        try:
            path = Path(dir_path)
            if path.exists() and self._is_managed_temp_dir(path):
                shutil.rmtree(path)
                
                # Remove from tracking
                for dir_id, tracked_path in list(self.active_dirs.items()):
                    if tracked_path == path:
                        del self.active_dirs[dir_id]
                        if dir_id in self.cleanup_callbacks:
                            del self.cleanup_callbacks[dir_id]
                        break
                        
                return True
        except Exception as e:
            raise ResourceManagementError(f"Failed to cleanup directory {dir_path}: {e}")
        
        return False
    
    @contextmanager
    def managed_temp_directory(self, prefix: str = "repo_analyzer_"):
        """Context manager for automatic temp directory cleanup."""
        temp_dir = self.create_temp_directory(prefix)
        try:
            yield temp_dir
        finally:
            self.cleanup_directory(temp_dir)
    
    def _cleanup_all(self):
        """Clean up all tracked temporary directories."""
        for temp_path in list(self.active_dirs.values()):
            try:
                if temp_path.exists():
                    shutil.rmtree(temp_path)
            except Exception:
                pass  # Best effort cleanup
        
        self.active_dirs.clear()
        self.cleanup_callbacks.clear()
```

### 5. Unified Error Handling (`exceptions.py`)

```python
from ..core.exceptions import RepositoryAnalyzerError

class InputHandlerError(RepositoryAnalyzerError):
    """Base exception for InputHandler errors."""
    
    def __init__(self, message: str, input_source: str = None, provider: str = None):
        super().__init__(message)
        self.input_source = input_source
        self.provider = provider

class InputValidationError(InputHandlerError):
    """Input validation failed."""
    
    def __init__(self, message: str, input_source: str = None, validation_type: str = None):
        super().__init__(message, input_source)
        self.validation_type = validation_type

class InputAuthenticationError(InputHandlerError):
    """Authentication failed for input source."""
    
    def __init__(self, message: str, input_source: str = None, provider: str = None, auth_method: str = None):
        super().__init__(message, input_source, provider)
        self.auth_method = auth_method

class InputProcessingError(InputHandlerError):
    """Input processing failed."""
    pass

class ResourceManagementError(InputHandlerError):
    """Resource management operation failed."""
    pass

class URLNormalizationError(InputHandlerError):
    """URL normalization failed."""
    pass
```

### 6. Configuration System (`config.py`)

```python
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
```

## Integration Strategy

### RepositoryAnalyzer Integration

```python
# In repository_analyzer/core/analyzer.py

from ..input.handler import InputHandler
from ..input.config import InputConfig

class RepositoryAnalyzer:
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or DEFAULT_CONFIG
        
        # Initialize InputHandler
        input_config = InputConfig(
            temp_dir=self.config.temp_dir,
            timeout=300,
            auto_cleanup=True
        )
        self.input_handler = InputHandler(input_config)
        
        # Keep other existing components
        self.file_scanner = FileSystemScanner(self.config)
        # ... rest of initialization
    
    def analyze(self, source: str) -> RepositoryStructure:
        """Analyze a repository using the new InputHandler."""
        try:
            # Process input through InputHandler
            processed_input = self.input_handler.process(source)
            
            # Use processed input for analysis
            files, directories = self.file_scanner.scan_repository(processed_input.local_path)
            
            # Continue with existing analysis pipeline...
            
        finally:
            # Cleanup through InputHandler
            if 'processed_input' in locals():
                self.input_handler.cleanup(processed_input)
```

### LangGraph Node Integration

```python
# New InputProcessorNode for LangGraph workflows

class InputProcessorNode:
    """LangGraph node for input processing."""
    
    def __init__(self, input_config: Optional[InputConfig] = None):
        self.input_handler = InputHandler(input_config or InputConfig())
    
    def __call__(self, state: RepositoryAnalysisState) -> RepositoryAnalysisState:
        try:
            source = state.get("repository_url") or state.get("local_path")
            if not source:
                return {
                    **state,
                    "errors": state.get("errors", []) + ["No input source provided"],
                    "current_step": "input_processing_failed"
                }
            
            # Process input
            processed_input = self.input_handler.process(source)
            
            # Update state with processed input
            return {
                **state,
                "processed_input": processed_input,
                "local_path": processed_input.local_path,
                "current_step": "input_processed"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": state.get("errors", []) + [f"Input processing failed: {str(e)}"],
                "current_step": "input_processing_failed"
            }
```

## Testing Strategy

### Unit Test Coverage

```python
# tests/test_input/test_validator.py
class TestInputValidator:
    def test_local_path_validation(self):
        """Test local path validation scenarios."""
        
    def test_url_validation(self):
        """Test URL validation scenarios."""
        
    def test_path_traversal_prevention(self):
        """Test security against path traversal attacks."""
        
    def test_malformed_input_handling(self):
        """Test handling of malformed inputs."""

# tests/test_input/test_auth_manager.py
class TestAuthManager:
    def test_token_authentication(self):
        """Test token-based authentication."""
        
    def test_ssh_authentication(self):
        """Test SSH key authentication."""
        
    def test_credential_precedence(self):
        """Test credential source precedence."""
```

### Integration Test Plan

```python
# tests/test_input/test_integration.py
class TestInputHandlerIntegration:
    def test_end_to_end_github_processing(self):
        """Test complete GitHub URL processing flow."""
        
    def test_end_to_end_local_processing(self):
        """Test complete local path processing flow."""
        
    def test_repository_analyzer_integration(self):
        """Test integration with RepositoryAnalyzer."""
        
    def test_langgraph_node_integration(self):
        """Test integration with LangGraph workflows."""
```

This specification provides detailed implementation guidance for each component of the unified InputHandler system, ensuring a secure, robust, and maintainable solution.