"""GitHub URL processor."""

import os
import tempfile
from typing import Optional
from .base import BaseProcessor
from ..handler import ProcessedInput, InputType
from ..config import InputConfig
from ..auth.manager import AuthManager, AuthConfig
from ..utils.temp_manager import TempDirectoryManager
from ..exceptions import InputValidationError, InputAuthenticationError
import git


class GitHubProcessor(BaseProcessor):
    """Processor for GitHub repository URLs."""
    
    def __init__(self, config: InputConfig):
        """Initialize the GitHub processor.
        
        Args:
            config: Input configuration
        """
        super().__init__(config)
        self.auth_manager = AuthManager(AuthConfig())
        self.temp_manager = TempDirectoryManager(config.temp_dir)
    
    def can_process(self, source: str) -> bool:
        """Check if this processor can handle the given source.
        
        Args:
            source: Input source to check
            
        Returns:
            True if this processor can handle the source, False otherwise
        """
        # Check if it's a GitHub URL
        return 'github.com' in source and (source.startswith('https://') or source.startswith('git@'))
    
    def process(self, source: str) -> ProcessedInput:
        """Process the GitHub URL input source.
        
        Args:
            source: GitHub URL to process
            
        Returns:
            ProcessedInput object with processing results
        """
        # Validate GitHub URL format
        if not self._is_valid_github_url(source):
            raise InputValidationError(f"Invalid GitHub URL format: {source}")
        
        # Apply authentication if available
        auth_source = source
        auth_used = False
        
        if self.config.enable_authentication:
            auth_source = self.auth_manager.authenticate_url(source, 'github')
            auth_used = (auth_source != source)
        
        # Create temporary directory for cloning
        temp_dir = self.temp_manager.create_temp_directory("github_repo_")
        
        try:
            # Clone the repository
            repo = git.Repo.clone_from(auth_source, temp_dir)
            
            # Create processed input result
            processed = ProcessedInput(
                source=source,
                input_type=InputType.GITHUB_URL,
                local_path=temp_dir,
                is_temporary=True,
                auth_used=auth_used,
                provider='github',
                cleanup_callback=lambda: self.temp_manager.cleanup_directory(temp_dir),
                metadata={
                    'original_url': source,
                    'auth_url': auth_source,
                    'clone_successful': True,
                    'commit_count': len(list(repo.iter_commits())) if repo else 0,
                    'branch_count': len(repo.branches) if repo else 0
                }
            )
            
            return processed
            
        except git.exc.GitCommandError as e:
            # Clean up on failure
            try:
                self.temp_manager.cleanup_directory(temp_dir)
            except Exception:
                pass
            
            # Handle authentication errors specifically
            if "authentication failed" in str(e).lower() or "403" in str(e):
                raise InputAuthenticationError(f"Authentication failed for GitHub repository: {source}")
            elif "not found" in str(e).lower() or "404" in str(e):
                raise InputValidationError(f"GitHub repository not found: {source}")
            else:
                raise InputValidationError(f"Failed to clone GitHub repository: {e}")
                
        except Exception as e:
            # Clean up on failure
            try:
                self.temp_manager.cleanup_directory(temp_dir)
            except Exception:
                pass
            
            raise InputValidationError(f"Failed to process GitHub repository {source}: {e}")
    
    def _is_valid_github_url(self, url: str) -> bool:
        """Check if URL is a valid GitHub URL.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is valid, False otherwise
        """
        # Basic format validation
        if not url.startswith(('https://github.com/', 'git@github.com:')):
            return False
        
        # Extract owner/repo from URL
        if url.startswith('https://github.com/'):
            path_parts = url.replace('https://github.com/', '').strip('/').split('/')
        else:  # SSH format
            path_parts = url.replace('git@github.com:', '').strip('/').split('/')
        
        # Should have at least owner/repo
        return len(path_parts) >= 2