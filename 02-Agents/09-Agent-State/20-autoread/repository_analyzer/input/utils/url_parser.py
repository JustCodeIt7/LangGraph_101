"""URL parsing and normalization utilities."""

from typing import Optional, Dict
from urllib.parse import urlparse, urlunparse
import re
from ..exceptions import URLNormalizationError


class URLParser:
    """URL parsing and normalization utilities."""
    
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
    
    def normalize_url(self, url: str) -> str:
        """Normalize URL for consistent processing."""
        try:
            parsed = urlparse(url)
            
            # Remove trailing slashes for consistency
            path = parsed.path.rstrip('/')
            
            # Reconstruct URL
            normalized = urlunparse((
                parsed.scheme,
                parsed.netloc,
                path,
                parsed.params,
                parsed.query,
                parsed.fragment
            ))
            
            return normalized
        except Exception as e:
            raise URLNormalizationError(f"Failed to normalize URL {url}: {e}")
    
    def detect_provider(self, url: str) -> Optional[str]:
        """Detect Git provider from URL."""
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        
        for provider, patterns in self.provider_patterns.items():
            for pattern in patterns:
                if re.search(pattern, host):
                    return provider
        
        return None
    
    def extract_repo_info(self, url: str) -> Dict[str, str]:
        """Extract repository information from URL."""
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')
        
        info = {
            'provider': self.detect_provider(url),
            'host': parsed.netloc,
            'scheme': parsed.scheme
        }
        
        # Extract owner and repo name from path
        if len(path_parts) >= 2:
            info['owner'] = path_parts[0]
            # Remove .git suffix if present
            repo_name = path_parts[1]
            if repo_name.endswith('.git'):
                repo_name = repo_name[:-4]
            info['repo_name'] = repo_name
        
        return info