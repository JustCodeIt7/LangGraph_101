"""File system scanner for repository analysis."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Callable, Generator
from ..core.config import AnalysisConfig
from ..core.data_structures import FileInfo, DirectoryInfo, FileType, DirectoryType
from ..core.exceptions import AnalysisError
from .filters import FileFilter


class FileSystemScanner:
    """Scans file systems and catalogs files and directories."""
    
    def __init__(self, config: AnalysisConfig):
        """Initialize the FileSystemScanner.
        
        Args:
            config: Analysis configuration
        """
        self.config = config
        self.file_filter = FileFilter(config)
        self._file_type_map = self._create_file_type_map()
        self._directory_type_map = self._create_directory_type_map()
    
    def scan_repository(self, repo_path: str) -> tuple[Dict[str, FileInfo], Dict[str, DirectoryInfo]]:
        """Scan a repository and catalog all files and directories.
        
        Args:
            repo_path: Path to the repository root
            
        Returns:
            Tuple of (files_dict, directories_dict)
            
        Raises:
            AnalysisError: If scanning fails
        """
        try:
            # Set up filters
            self.file_filter.setup_filters(repo_path)
            
            # Scan files and directories
            files = self._scan_files(repo_path)
            directories = self._scan_directories(repo_path, files)
            
            return files, directories
        except Exception as e:
            raise AnalysisError(f"Failed to scan repository: {e}")
    
    def _scan_files(self, repo_path: str) -> Dict[str, FileInfo]:
        """Scan all files in a repository.
        
        Args:
            repo_path: Path to the repository root
            
        Returns:
            Dictionary mapping file paths to FileInfo objects
        """
        files = {}
        repo_path_obj = Path(repo_path)
        
        for file_path in self._walk_repository(repo_path):
            # Check if file should be included
            if not self.file_filter.should_include_file(str(file_path), repo_path):
                continue
            
            try:
                # Get file stats
                stat = file_path.stat()
                
                # Determine file type and language
                file_type, language = self._classify_file(str(file_path.relative_to(repo_path_obj)))
                
                # Create FileInfo object
                file_info = FileInfo(
                    name=file_path.name,
                    path=str(file_path.relative_to(repo_path_obj)),
                    extension=file_path.suffix,
                    size=stat.st_size,
                    type=file_type,
                    language=language
                )
                
                files[str(file_path.relative_to(repo_path_obj))] = file_info
            except Exception:
                # Skip files that cause errors
                continue
        
        return files
    
    def _scan_directories(self, repo_path: str, files: Dict[str, FileInfo]) -> Dict[str, DirectoryInfo]:
        """Scan all directories in a repository.
        
        Args:
            repo_path: Path to the repository root
            files: Dictionary of files from _scan_files
            
        Returns:
            Dictionary mapping directory paths to DirectoryInfo objects
        """
        directories = {}
        repo_path_obj = Path(repo_path)
        
        # Get all unique directory paths from files
        dir_paths = set()
        for file_path in files.keys():
            path_obj = Path(file_path)
            for parent in path_obj.parents:
                if parent != Path('.'):
                    dir_paths.add(str(parent))
        
        # Add root directory
        dir_paths.add('.')
        
        # Create DirectoryInfo for each directory
        for dir_path_str in dir_paths:
            dir_path = Path(dir_path_str)
            full_path = repo_path_obj / dir_path
            
            try:
                # Determine directory type and purpose
                dir_type, purpose = self._classify_directory(dir_path_str)
                
                # Get child files and directories
                children = []
                file_count = 0
                
                # Count files in this directory
                for file_path_key in files.keys():
                    if str(Path(file_path_key).parent) == dir_path_str or \
                       (dir_path_str == '.' and str(Path(file_path_key).parent) == '.'):
                        file_count += 1
                        children.append(file_path_key)
                
                # Add subdirectories
                for other_dir_path in dir_paths:
                    if other_dir_path != dir_path_str and \
                       str(Path(other_dir_path).parent) == dir_path_str:
                        children.append(other_dir_path)
                
                # Create DirectoryInfo object
                dir_info = DirectoryInfo(
                    name=dir_path.name if dir_path.name else repo_path_obj.name,
                    path=dir_path_str,
                    type=dir_type,
                    purpose=purpose,
                    children=children,
                    file_count=file_count
                )
                
                directories[dir_path_str] = dir_info
            except Exception:
                # Skip directories that cause errors
                continue
        
        return directories
    
    def _walk_repository(self, repo_path: str) -> Generator[Path, None, None]:
        """Walk through repository files respecting depth limits and filters.
        
        Args:
            repo_path: Path to the repository root
            
        Yields:
            Path objects for files in the repository
        """
        repo_path_obj = Path(repo_path)
        max_depth = self.config.max_depth
        
        for root, dirs, files in os.walk(repo_path):
            # Calculate current depth
            try:
                current_depth = len(Path(root).relative_to(repo_path_obj).parts)
            except ValueError:
                # root is not within repo_path
                current_depth = len(Path(root).parts)
            
            # Skip if beyond max depth
            if max_depth > 0 and current_depth > max_depth:
                dirs.clear()  # Don't recurse further
                continue
            
            # Convert to Path objects and yield files
            root_path = Path(root)
            for file_name in files:
                yield root_path / file_name
    
    def _classify_file(self, file_path: str) -> tuple[FileType, Optional[str]]:
        """Classify a file based on its path and extension.
        
        Args:
            file_path: Path to the file relative to repository root
            
        Returns:
            Tuple of (FileType, language)
        """
        path_obj = Path(file_path)
        extension = path_obj.suffix.lower()
        
        # Check for specific file types
        for pattern, file_type in self._file_type_map.items():
            if pattern in file_path.lower():
                language = self._determine_language(file_path)
                return file_type, language
        
        # Default classification based on extension
        language = self._determine_language(file_path)
        file_type = FileType.UNKNOWN
        
        # Source code files
        if extension in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs', '.go', '.rs', '.rb', '.php']:
            file_type = FileType.SOURCE
        # Configuration files
        elif extension in ['.json', '.yaml', '.yml', '.xml', '.toml', '.ini', '.cfg']:
            file_type = FileType.CONFIG
        # Documentation files
        elif extension in ['.md', '.rst', '.txt', '.pdf', '.doc', '.docx']:
            file_type = FileType.DOC
        # Test files
        elif 'test' in file_path.lower() or 'spec' in file_path.lower():
            file_type = FileType.TEST
        # Asset files
        elif extension in ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.css', '.scss', '.sass']:
            file_type = FileType.ASSET
        # Script files
        elif extension in ['.sh', '.bat', '.ps1']:
            file_type = FileType.SCRIPT
        
        return file_type, language
    
    def _classify_directory(self, dir_path: str) -> tuple[DirectoryType, str]:
        """Classify a directory based on its path.
        
        Args:
            dir_path: Path to the directory relative to repository root
            
        Returns:
            Tuple of (DirectoryType, purpose description)
        """
        dir_path_lower = dir_path.lower()
        
        # Check for specific directory types
        for pattern, (dir_type, purpose) in self._directory_type_map.items():
            if pattern in dir_path_lower:
                return dir_type, purpose
        
        # Default classification
        return DirectoryType.UNKNOWN, "General directory"
    
    def _determine_language(self, file_path: str) -> Optional[str]:
        """Determine programming language based on file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Language name or None if not applicable
        """
        extension_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.cs': 'C#',
            '.go': 'Go',
            '.rs': 'Rust',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.scala': 'Scala',
            '.dart': 'Dart'
        }
        
        extension = Path(file_path).suffix.lower()
        return extension_map.get(extension)
    
    def _create_file_type_map(self) -> Dict[str, FileType]:
        """Create mapping of file patterns to file types.
        
        Returns:
            Dictionary mapping patterns to FileType enums
        """
        return {
            'test': FileType.TEST,
            'spec': FileType.TEST,
            'readme': FileType.DOC,
            'license': FileType.DOC,
            'changelog': FileType.DOC,
            'contributing': FileType.DOC,
            'makefile': FileType.SCRIPT,
            'dockerfile': FileType.CONFIG,
            'docker-compose': FileType.CONFIG,
            '.gitignore': FileType.CONFIG,
            '.gitlab-ci': FileType.CONFIG,
            '.travis': FileType.CONFIG,
            '.github': FileType.CONFIG,
            '.circleci': FileType.CONFIG
        }
    
    def _create_directory_type_map(self) -> Dict[str, tuple[DirectoryType, str]]:
        """Create mapping of directory patterns to directory types.
        
        Returns:
            Dictionary mapping patterns to (DirectoryType, purpose) tuples
        """
        return {
            'src': (DirectoryType.SOURCE, 'Source code'),
            'source': (DirectoryType.SOURCE, 'Source code'),
            'lib': (DirectoryType.SOURCE, 'Library code'),
            'app': (DirectoryType.SOURCE, 'Application code'),
            'config': (DirectoryType.CONFIG, 'Configuration files'),
            'conf': (DirectoryType.CONFIG, 'Configuration files'),
            'settings': (DirectoryType.CONFIG, 'Settings files'),
            'docs': (DirectoryType.DOCS, 'Documentation'),
            'doc': (DirectoryType.DOCS, 'Documentation'),
            'wiki': (DirectoryType.DOCS, 'Wiki documentation'),
            'test': (DirectoryType.TESTS, 'Test files'),
            'tests': (DirectoryType.TESTS, 'Test files'),
            'spec': (DirectoryType.TESTS, 'Specification files'),
            '__tests__': (DirectoryType.TESTS, 'Test files'),
            'assets': (DirectoryType.ASSETS, 'Asset files'),
            'static': (DirectoryType.ASSETS, 'Static assets'),
            'public': (DirectoryType.ASSETS, 'Public assets'),
            'images': (DirectoryType.ASSETS, 'Image assets'),
            'img': (DirectoryType.ASSETS, 'Image assets'),
            'icons': (DirectoryType.ASSETS, 'Icon assets'),
            'scripts': (DirectoryType.SCRIPTS, 'Scripts'),
            'bin': (DirectoryType.SCRIPTS, 'Binary/scripts'),
            'tools': (DirectoryType.SCRIPTS, 'Tools'),
            'build': (DirectoryType.BUILD, 'Build artifacts'),
            'dist': (DirectoryType.BUILD, 'Distribution files'),
            'target': (DirectoryType.BUILD, 'Build target directory'),
            'out': (DirectoryType.BUILD, 'Output directory'),
            'tmp': (DirectoryType.TEMP, 'Temporary files'),
            'temp': (DirectoryType.TEMP, 'Temporary files'),
            'cache': (DirectoryType.TEMP, 'Cache files')
        }