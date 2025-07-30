"""Input source classification system."""

from enum import Enum
from typing import Optional
from urllib.parse import urlparse
import re
from .utils.url_parser import URLParser


class InputType(Enum):
    LOCAL_PATH = "local_path"
    GITHUB_URL = "github_url"
    GITLAB_URL = "gitlab_url"
    BITBUCKET_URL = "bitbucket_url"
    GENERIC_GIT_URL = "generic_git_url"
    UNKNOWN = "unknown"


class ClassificationResult:
    """Result of input classification."""
    
    def __init__(self, input_type: InputType, confidence: float, provider: Optional[str] = None, metadata: Optional[dict] = None):
        self.input_type = input_type
        self.confidence = confidence
        self.provider = provider
        self.metadata = metadata or {}


class InputClassifier:
    """Intelligent input source classification."""
    
    def __init__(self):
        self.url_parser = URLParser()
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
    
    def _classify_local_path(self, path: str) -> ClassificationResult:
        """Classify local path input."""
        return ClassificationResult(
            input_type=InputType.LOCAL_PATH,
            confidence=1.0,
            metadata={'original_path': path}
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