"""Temporary directory management utilities."""

import tempfile
import shutil
import weakref
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
from ..exceptions import ResourceManagementError


class TempDirectoryManager:
    """Advanced temporary directory management with cleanup tracking."""
    
    def __init__(self, base_temp_dir: Optional[str] = None):
        self.base_temp_dir = Path(base_temp_dir) if base_temp_dir else Path(tempfile.gettempdir())
        self.active_dirs = {}
        self.cleanup_callbacks = {}
        
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
    
    def _is_managed_temp_dir(self, path: Path) -> bool:
        """Check if a path is a managed temporary directory."""
        return str(self.base_temp_dir) in str(path)