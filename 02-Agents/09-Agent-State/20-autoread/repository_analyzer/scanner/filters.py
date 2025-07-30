"""File filtering system including .gitignore parsing."""

import os
import fnmatch
from pathlib import Path
from typing import List, Set, Optional
from ..core.config import AnalysisConfig
from ..core.exceptions import AnalysisError


class GitignoreParser:
    """Parser for .gitignore files and filtering system."""
    
    def __init__(self, config: AnalysisConfig):
        """Initialize the GitignoreParser.
        
        Args:
            config: Analysis configuration
        """
        self.config = config
        self.gitignore_rules: List[str] = []
        self.negated_patterns: List[str] = []
    
    def load_gitignore(self, repo_path: str) -> None:
        """Load .gitignore rules from a repository.
        
        Args:
            repo_path: Path to the repository root
            
        Raises:
            AnalysisError: If .gitignore file cannot be read
        """
        gitignore_path = Path(repo_path) / ".gitignore"
        
        if gitignore_path.exists():
            try:
                with open(gitignore_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            if line.startswith('!'):
                                self.negated_patterns.append(line[1:])
                            else:
                                self.gitignore_rules.append(line)
            except Exception as e:
                raise AnalysisError(f"Failed to read .gitignore file: {e}")
    
    def is_ignored(self, file_path: str, repo_path: str) -> bool:
        """Check if a file should be ignored based on .gitignore rules and config.
        
        Args:
            file_path: Path to the file to check
            repo_path: Path to the repository root
            
        Returns:
            True if file should be ignored, False otherwise
        """
        # First check config-based ignore patterns
        if self.config.should_ignore_path(file_path):
            return True
        
        # Convert to relative path from repo root
        try:
            rel_path = Path(file_path).relative_to(repo_path).as_posix()
        except ValueError:
            # file_path is not within repo_path, use as-is
            rel_path = file_path
        
        # Check .gitignore rules
        for pattern in self.gitignore_rules:
            if self._matches_pattern(rel_path, pattern):
                # Check if this pattern is negated
                for neg_pattern in self.negated_patterns:
                    if self._matches_pattern(rel_path, neg_pattern):
                        return False
                return True
        
        return False
    
    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if a file path matches a .gitignore pattern.
        
        Args:
            file_path: Path to the file to check
            pattern: .gitignore pattern to match against
            
        Returns:
            True if the file path matches the pattern, False otherwise
        """
        # Handle absolute paths (starting with /)
        if pattern.startswith('/'):
            pattern = pattern[1:]
            # Match only at the root level
            if '/' not in file_path:
                return fnmatch.fnmatch(file_path, pattern)
            else:
                first_part = file_path.split('/')[0]
                return fnmatch.fnmatch(first_part, pattern)
        
        # Handle directory patterns (ending with /)
        if pattern.endswith('/'):
            pattern = pattern[:-1]
            # Match if the file's directory matches the pattern
            file_dir = os.path.dirname(file_path)
            if file_dir:
                return fnmatch.fnmatch(file_dir, pattern) or \
                       self._path_contains_pattern(file_dir, pattern)
            return False
        
        # Handle patterns with slashes (directory-specific)
        if '/' in pattern:
            return fnmatch.fnmatch(file_path, pattern) or \
                   fnmatch.fnmatch(os.path.basename(file_path), pattern)
        
        # General pattern matching
        return fnmatch.fnmatch(os.path.basename(file_path), pattern) or \
               fnmatch.fnmatch(file_path, pattern) or \
               self._path_contains_pattern(file_path, pattern)
    
    def _path_contains_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if any directory in the path matches the pattern.
        
        Args:
            file_path: Path to check
            pattern: Pattern to match
            
        Returns:
            True if any directory in the path matches the pattern
        """
        path_parts = file_path.split('/')
        for i in range(len(path_parts)):
            sub_path = '/'.join(path_parts[:i+1])
            if fnmatch.fnmatch(sub_path, pattern):
                return True
        return False


class FileFilter:
    """File filtering system combining multiple filtering approaches."""
    
    def __init__(self, config: AnalysisConfig):
        """Initialize the FileFilter.
        
        Args:
            config: Analysis configuration
        """
        self.config = config
        self.gitignore_parser = GitignoreParser(config)
    
    def setup_filters(self, repo_path: str) -> None:
        """Set up all filters for a repository.
        
        Args:
            repo_path: Path to the repository root
        """
        if self.config.respect_gitignore:
            self.gitignore_parser.load_gitignore(repo_path)
    
    def should_include_file(self, file_path: str, repo_path: str) -> bool:
        """Determine if a file should be included in analysis.
        
        Args:
            file_path: Path to the file
            repo_path: Path to the repository root
            
        Returns:
            True if file should be included, False if it should be filtered out
        """
        # Check file size limit
        try:
            if os.path.getsize(file_path) > self.config.max_file_size:
                return False
        except (OSError, FileNotFoundError):
            # If we can't get the size, include it for now
            pass
        
        # Check .gitignore and config-based filters
        if self.gitignore_parser.is_ignored(file_path, repo_path):
            return False
        
        # Check if it's a hidden file and we're not including them
        if not self.config.include_hidden:
            path_obj = Path(file_path)
            if any(part.startswith('.') for part in path_obj.parts):
                return False
        
        return True