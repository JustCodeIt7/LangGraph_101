"""Main repository analyzer class."""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from ..core.config import AnalysisConfig, DEFAULT_CONFIG
from ..core.data_structures import RepositoryStructure, RepositoryMetadata, ProjectType, FileInfo, DirectoryInfo, Framework
from ..core.exceptions import RepositoryAnalyzerError, RepositoryNotFoundError
from ..git.cloner import GitCloner
from ..scanner.filesystem import FileSystemScanner
from ..scanner.cataloger import FileCataloger
from ..patterns.detector import PatternDetector
from ..patterns.frameworks import FrameworkDetector
from ..analysis.relationships import RelationshipMapper
from ..analysis.imports import ImportAnalyzer
from ..analysis.config_parser import ConfigFileParser


class RepositoryAnalyzer:
    """Main class for analyzing repository structures."""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Initialize the RepositoryAnalyzer.
        
        Args:
            config: Analysis configuration, uses DEFAULT_CONFIG if None
        """
        self.config = config or DEFAULT_CONFIG
        self.git_cloner = GitCloner(self.config)
        self.file_scanner = FileSystemScanner(self.config)
        self.file_cataloger = FileCataloger()
        self.pattern_detector = PatternDetector()
        self.framework_detector = FrameworkDetector()
        self.relationship_mapper = RelationshipMapper()
        self.import_analyzer = ImportAnalyzer()
        self.config_parser = ConfigFileParser()
        
        # Track temporary directories for cleanup
        self._temp_dirs = []
    
    def analyze(self, source: str) -> RepositoryStructure:
        """Analyze a repository from a URL or local path.
        
        Args:
            source: GitHub URL or local path to repository
            
        Returns:
            RepositoryStructure object with analysis results
            
        Raises:
            RepositoryAnalyzerError: If analysis fails
            RepositoryNotFoundError: If repository cannot be found
        """
        repo_path = None
        is_temp_repo = False
        
        try:
            # Clone or copy repository if needed
            repo_path = self._prepare_repository(source)
            is_temp_repo = repo_path.startswith(str(self.config.get_temp_dir()))
            
            # Scan repository structure
            files, directories = self.file_scanner.scan_repository(repo_path)
            
            # Catalog files and extract metadata
            files = self.file_cataloger.catalog_files(files, directories, repo_path)
            directories = self.file_cataloger.catalog_directories(directories, files)
            
            # Analyze imports if enabled
            if self.config.analyze_imports:
                files = self.import_analyzer.analyze_imports(files)
            
            # Detect patterns and project type
            patterns = self.pattern_detector.detect_patterns(directories, files)
            project_type = self.pattern_detector.detect_project_type(directories, files)
            
            # Detect frameworks if enabled
            frameworks = []
            if self.config.detect_frameworks:
                frameworks = self.framework_detector.detect_frameworks(files, directories)
            
            # Map relationships if enabled
            relationships = []
            if self.config.map_relationships:
                relationships = self.relationship_mapper.map_relationships(files, directories)
            
            # Create repository metadata
            metadata = self._create_repository_metadata(repo_path, files, directories, frameworks)
            
            # Create repository structure object
            structure = RepositoryStructure(
                source=source,
                root_path=repo_path,
                project_type=project_type,
                frameworks=frameworks,
                directories=directories,
                files=files,
                patterns=patterns,
                relationships=relationships,
                metadata=metadata
            )
            
            return structure
            
        finally:
            # Clean up temporary repository if needed
            if is_temp_repo and repo_path and self.config.respect_gitignore:
                self.git_cloner.cleanup_temp_repository(repo_path)
    
    def _prepare_repository(self, source: str) -> str:
        """Prepare repository for analysis by cloning or copying.
        
        Args:
            source: GitHub URL or local path to repository
            
        Returns:
            Path to the repository
            
        Raises:
            RepositoryNotFoundError: If repository cannot be found
        """
        # Handle local paths
        if source.startswith("/") or source.startswith("./") or source.startswith("../"):
            if os.path.exists(source):
                return source
            else:
                raise RepositoryNotFoundError(f"Local repository not found: {source}")
        
        # Handle URLs by cloning
        if source.startswith("http://") or source.startswith("https://") or \
           source.startswith("git@"):
            # Create temporary directory for cloning
            temp_dir = tempfile.mkdtemp(dir=self.config.get_temp_dir())
            self._temp_dirs.append(temp_dir)
            return self.git_cloner.clone_repository(source, temp_dir)
        
        # Try to treat as a local path
        if os.path.exists(source):
            return source
        else:
            raise RepositoryNotFoundError(f"Repository not found: {source}")
    
    def _create_repository_metadata(self, repo_path: str, files: Dict[str, FileInfo], 
                                  directories: Dict[str, DirectoryInfo], 
                                  frameworks: List[Framework]) -> RepositoryMetadata:
        """Create repository metadata.
        
        Args:
            repo_path: Path to the repository
            files: Dictionary of FileInfo objects
            directories: Dictionary of DirectoryInfo objects
            frameworks: List of detected Framework objects
            
        Returns:
            RepositoryMetadata object
        """
        metadata = RepositoryMetadata()
        
        try:
            # Get repository name from path
            repo_path_obj = Path(repo_path)
            metadata.name = repo_path_obj.name
            
            # Determine primary language
            language_counts = {}
            for file_info in files.values():
                if file_info.language:
                    language_counts[file_info.language] = language_counts.get(file_info.language, 0) + 1
            
            if language_counts:
                metadata.primary_language = max(language_counts, key=language_counts.get)
                metadata.languages = list(language_counts.keys())
            
            # Add frameworks
            metadata.frameworks = [framework.name for framework in frameworks]
            
            # Determine architecture type
            project_type = self.pattern_detector.detect_project_type(directories, files)
            metadata.architecture_type = project_type.value
            
            # Calculate complexity score
            metadata.complexity_score = self._calculate_complexity_score(files, directories)
            
            # Calculate documentation coverage
            metadata.documentation_coverage = self._calculate_documentation_coverage(files)
            
            # Calculate test coverage estimate
            metadata.test_coverage_estimate = self._calculate_test_coverage(files)
            
            # Find entry points
            metadata.entry_points = self._find_entry_points(files)
            
            # Find configuration files
            metadata.configuration_files = self._find_configuration_files(files)
            
        except Exception:
            # Silently continue if metadata creation fails
            pass
        
        return metadata
    
    def _calculate_complexity_score(self, files: Dict[str, FileInfo], 
                                   directories: Dict[str, DirectoryInfo]) -> float:
        """Calculate repository complexity score.
        
        Args:
            files: Dictionary of FileInfo objects
            directories: Dictionary of DirectoryInfo objects
            
        Returns:
            Complexity score between 0.0 and 1.0
        """
        # Simple complexity calculation based on file count and directory depth
        file_count = len(files)
        dir_count = len(directories)
        
        # Normalize scores
        file_score = min(file_count / 1000.0, 1.0)  # Cap at 1000 files
        dir_score = min(dir_count / 100.0, 1.0)     # Cap at 100 directories
        
        # Weighted average
        complexity = (file_score * 0.7) + (dir_score * 0.3)
        return min(complexity, 1.0)
    
    def _calculate_documentation_coverage(self, files: Dict[str, FileInfo]) -> float:
        """Calculate documentation coverage.
        
        Args:
            files: Dictionary of FileInfo objects
            
        Returns:
            Documentation coverage between 0.0 and 1.0
        """
        total_files = len(files)
        if total_files == 0:
            return 0.0
        
        doc_files = sum(1 for file_info in files.values() 
                       if file_info.type == file_info.DOC)
        
        return doc_files / total_files if total_files > 0 else 0.0
    
    def _calculate_test_coverage(self, files: Dict[str, FileInfo]) -> float:
        """Calculate test coverage estimate.
        
        Args:
            files: Dictionary of FileInfo objects
            
        Returns:
            Test coverage estimate between 0.0 and 1.0
        """
        total_files = len(files)
        if total_files == 0:
            return 0.0
        
        test_files = sum(1 for file_info in files.values() 
                        if file_info.type == file_info.TEST)
        
        return test_files / total_files if total_files > 0 else 0.0
    
    def _find_entry_points(self, files: Dict[str, FileInfo]) -> List[str]:
        """Find entry points in the repository.
        
        Args:
            files: Dictionary of FileInfo objects
            
        Returns:
            List of entry point file paths
        """
        entry_points = []
        
        # Common entry point file names
        entry_point_names = [
            'main.py', 'app.py', 'index.js', 'server.js', 'application.py',
            'main.java', 'program.cs', 'main.go', 'index.ts', 'app.ts'
        ]
        
        for file_path, file_info in files.items():
            if file_info.name.lower() in entry_point_names:
                entry_points.append(file_path)
        
        return entry_points
    
    def _find_configuration_files(self, files: Dict[str, FileInfo]) -> List[str]:
        """Find configuration files in the repository.
        
        Args:
            files: Dictionary of FileInfo objects
            
        Returns:
            List of configuration file paths
        """
        config_files = []
        
        for file_path, file_info in files.items():
            if file_info.type == file_info.CONFIG:
                config_files.append(file_path)
        
        return config_files
    
    def cleanup(self):
        """Clean up temporary resources."""
        for temp_dir in self._temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir)
            except Exception:
                pass  # Silently continue if cleanup fails
        self._temp_dirs.clear()