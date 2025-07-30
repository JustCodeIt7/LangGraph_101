"""Configuration management for repository analyzer."""

from dataclasses import dataclass, field
from typing import List, Optional
import os
from pathlib import Path


@dataclass
class AnalysisConfig:
    """Configuration for repository analysis."""
    # Analysis depth and scope
    max_depth: int = 10
    respect_gitignore: bool = True  # Parse and apply .gitignore rules by default
    ignore_patterns: List[str] = field(default_factory=lambda: [
        ".git", "__pycache__", ".pytest_cache", ".mypy_cache", 
        ".tox", ".venv", "venv", "node_modules", "build", "dist"
    ])
    include_hidden: bool = False
    analyze_imports: bool = True
    detect_frameworks: bool = True
    map_relationships: bool = True
    
    # Storage and temporary directories
    temp_dir: Optional[str] = None
    git_auth_token: Optional[str] = None
    
    # Performance settings
    max_file_size: int = 10 * 1024 * 1024  # 10MB limit by default
    parallel_processing: bool = True
    max_workers: int = 4
    
    def __post_init__(self):
        """Initialize configuration with environment variables."""
        if self.temp_dir is None:
            self.temp_dir = os.environ.get("REPO_ANALYZER_TEMP_DIR", "/tmp/repo_analyzer")
        
        if self.git_auth_token is None:
            self.git_auth_token = os.environ.get("GITHUB_TOKEN")
    
    def get_temp_dir(self) -> Path:
        """Get the temporary directory as a Path object."""
        return Path(self.temp_dir)
    
    def should_ignore_path(self, path: str) -> bool:
        """Check if a path should be ignored based on patterns."""
        path_obj = Path(path)
        
        # Check if path matches any ignore patterns
        for pattern in self.ignore_patterns:
            if pattern in path:
                return True
        
        # Check if it's a hidden file/directory and we're not including them
        if not self.include_hidden and path_obj.name.startswith("."):
            return True
            
        return False


# Default configuration
DEFAULT_CONFIG = AnalysisConfig()