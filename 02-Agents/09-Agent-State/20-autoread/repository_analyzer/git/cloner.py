"""Git repository cloning functionality."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Union
import git
from ..core.config import AnalysisConfig
from ..core.exceptions import RepositoryNotFoundError, AuthenticationError, GitError


class GitCloner:
    """Handles cloning of Git repositories."""
    
    def __init__(self, config: AnalysisConfig):
        """Initialize the GitCloner with configuration.
        
        Args:
            config: Analysis configuration
        """
        self.config = config
        self.temp_dir = config.get_temp_dir()
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def clone_repository(self, source: str, target_dir: Optional[str] = None) -> str:
        """Clone a repository from a URL or local path.
        
        Args:
            source: GitHub URL or local path to repository
            target_dir: Optional target directory for cloning
            
        Returns:
            Path to the cloned repository
            
        Raises:
            RepositoryNotFoundError: If repository cannot be found
            AuthenticationError: If authentication fails for private repos
            GitError: If cloning fails
        """
        # Handle local paths
        if source.startswith("/") or source.startswith("./") or source.startswith("../"):
            if os.path.exists(source):
                return self._handle_local_repository(source, target_dir)
            else:
                raise RepositoryNotFoundError(f"Local repository not found: {source}")
        
        # Handle GitHub URLs
        if source.startswith("https://github.com/") or source.startswith("git@github.com:"):
            return self._clone_github_repository(source, target_dir)
        
        # Try to treat as a Git URL
        return self._clone_generic_repository(source, target_dir)
    
    def _handle_local_repository(self, source: str, target_dir: Optional[str] = None) -> str:
        """Handle local repository by copying to target location.
        
        Args:
            source: Path to local repository
            target_dir: Optional target directory
            
        Returns:
            Path to the repository copy
        """
        source_path = Path(source)
        if not source_path.exists():
            raise RepositoryNotFoundError(f"Source path does not exist: {source}")
        
        if target_dir is None:
            target_dir = tempfile.mkdtemp(dir=self.temp_dir)
        
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Copy the repository
        if source_path.is_dir():
            shutil.copytree(source_path, target_path / source_path.name, dirs_exist_ok=True)
            return str(target_path / source_path.name)
        else:
            shutil.copy2(source_path, target_path)
            return str(target_path / source_path.name)
    
    def _clone_github_repository(self, url: str, target_dir: Optional[str] = None) -> str:
        """Clone a GitHub repository.
        
        Args:
            url: GitHub repository URL
            target_dir: Optional target directory
            
        Returns:
            Path to the cloned repository
            
        Raises:
            AuthenticationError: If authentication fails
            GitError: If cloning fails
        """
        if target_dir is None:
            target_dir = tempfile.mkdtemp(dir=self.temp_dir)
        
        try:
            # Add authentication token if available
            if self.config.git_auth_token and "github.com" in url:
                if url.startswith("https://"):
                    # Insert token in URL
                    url = url.replace("https://", f"https://{self.config.git_auth_token}@")
                elif url.startswith("git@"):
                    # For SSH, token isn't used directly
                    pass
            
            # Clone the repository
            repo = git.Repo.clone_from(url, target_dir)
            return target_dir
            
        except git.exc.GitCommandError as e:
            if "authentication failed" in str(e).lower() or "403" in str(e):
                raise AuthenticationError(f"Authentication failed for repository: {url}")
            elif "not found" in str(e).lower() or "404" in str(e):
                raise RepositoryNotFoundError(f"Repository not found: {url}")
            else:
                raise GitError(f"Failed to clone repository: {e}")
        except Exception as e:
            raise GitError(f"Unexpected error cloning repository: {e}")
    
    def _clone_generic_repository(self, url: str, target_dir: Optional[str] = None) -> str:
        """Clone a generic Git repository.
        
        Args:
            url: Git repository URL
            target_dir: Optional target directory
            
        Returns:
            Path to the cloned repository
            
        Raises:
            GitError: If cloning fails
        """
        if target_dir is None:
            target_dir = tempfile.mkdtemp(dir=self.temp_dir)
        
        try:
            repo = git.Repo.clone_from(url, target_dir)
            return target_dir
        except git.exc.GitCommandError as e:
            raise GitError(f"Failed to clone repository: {e}")
        except Exception as e:
            raise GitError(f"Unexpected error cloning repository: {e}")
    
    def cleanup_temp_repository(self, repo_path: str) -> bool:
        """Clean up a temporary repository.
        
        Args:
            repo_path: Path to repository to clean up
            
        Returns:
            True if cleanup was successful, False otherwise
        """
        try:
            if os.path.exists(repo_path):
                if str(self.temp_dir) in str(repo_path):
                    shutil.rmtree(repo_path)
                    return True
            return False
        except Exception:
            return False