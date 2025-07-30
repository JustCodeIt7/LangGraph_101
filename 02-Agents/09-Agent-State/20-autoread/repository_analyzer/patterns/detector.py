"""General pattern detection for repository structures."""

from typing import Dict, List, Set
from ..core.data_structures import DirectoryInfo, FileInfo, Pattern, ProjectType
from ..core.exceptions import PatternDetectionError


class PatternDetector:
    """Detects general patterns in repository structures."""
    
    def __init__(self):
        """Initialize the PatternDetector."""
        self.project_patterns = self._create_project_patterns()
        self.architecture_patterns = self._create_architecture_patterns()
        self.file_patterns = self._create_file_patterns()
    
    def detect_patterns(self, directories: Dict[str, DirectoryInfo], 
                       files: Dict[str, FileInfo]) -> List[Pattern]:
        """Detect patterns in the repository structure.
        
        Args:
            directories: Dictionary of DirectoryInfo objects
            files: Dictionary of FileInfo objects
            
        Returns:
            List of detected Pattern objects
        """
        patterns = []
        
        # Detect project structure patterns
        project_patterns = self._detect_project_patterns(directories, files)
        patterns.extend(project_patterns)
        
        # Detect architecture patterns
        architecture_patterns = self._detect_architecture_patterns(directories, files)
        patterns.extend(architecture_patterns)
        
        # Detect file-based patterns
        file_patterns = self._detect_file_patterns(files)
        patterns.extend(file_patterns)
        
        return patterns
    
    def detect_project_type(self, directories: Dict[str, DirectoryInfo], 
                           files: Dict[str, FileInfo]) -> ProjectType:
        """Determine the project type based on structure.
        
        Args:
            directories: Dictionary of DirectoryInfo objects
            files: Dictionary of FileInfo objects
            
        Returns:
            ProjectType enum value
        """
        # Count occurrences of different patterns
        monolith_indicators = 0
        microservice_indicators = 0
        library_indicators = 0
        
        # Check directory structure
        for dir_path, dir_info in directories.items():
            if 'services' in dir_path or 'microservice' in dir_path:
                microservice_indicators += 1
            elif 'packages' in dir_path or 'modules' in dir_path:
                library_indicators += 1
            elif 'src' in dir_path or 'app' in dir_path:
                monolith_indicators += 1
        
        # Check for service-specific files
        service_files = [f for f in files.keys() if 'service' in f.lower()]
        if len(service_files) > 3:
            microservice_indicators += 1
            
        # Check for package-specific files
        package_files = [f for f in files.keys() if 'package' in f.lower() or 'setup.py' in f or 'pom.xml' in f]
        if len(package_files) > 0:
            library_indicators += 1
            
        # Determine project type based on indicators
        if microservice_indicators > library_indicators and microservice_indicators > monolith_indicators:
            return ProjectType.MICROSERVICES
        elif library_indicators > monolith_indicators:
            return ProjectType.LIBRARY
        elif monolith_indicators > 0:
            return ProjectType.MONOLITH
        else:
            return ProjectType.UNKNOWN
    
    def _detect_project_patterns(self, directories: Dict[str, DirectoryInfo], 
                                files: Dict[str, FileInfo]) -> List[Pattern]:
        """Detect project structure patterns.
        
        Args:
            directories: Dictionary of DirectoryInfo objects
            files: Dictionary of FileInfo objects
            
        Returns:
            List of detected Pattern objects
        """
        patterns = []
        
        # Check for common project structure patterns
        for pattern_name, pattern_info in self.project_patterns.items():
            confidence = self._calculate_pattern_confidence(pattern_info, directories, files)
            if confidence > 0.3:  # Minimum confidence threshold
                pattern_files = self._get_files_matching_pattern(pattern_info, files)
                pattern = Pattern(
                    name=pattern_name,
                    type="project_structure",
                    confidence=confidence,
                    files=pattern_files,
                    metadata=pattern_info.get('metadata', {})
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_architecture_patterns(self, directories: Dict[str, DirectoryInfo], 
                                     files: Dict[str, FileInfo]) -> List[Pattern]:
        """Detect architecture patterns.
        
        Args:
            directories: Dictionary of DirectoryInfo objects
            files: Dictionary of FileInfo objects
            
        Returns:
            List of detected Pattern objects
        """
        patterns = []
        
        # Check for architecture patterns
        for pattern_name, pattern_info in self.architecture_patterns.items():
            confidence = self._calculate_architecture_confidence(pattern_info, directories, files)
            if confidence > 0.3:  # Minimum confidence threshold
                pattern_files = self._get_files_matching_architecture_pattern(pattern_info, files)
                pattern = Pattern(
                    name=pattern_name,
                    type="architecture",
                    confidence=confidence,
                    files=pattern_files,
                    metadata=pattern_info.get('metadata', {})
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_file_patterns(self, files: Dict[str, FileInfo]) -> List[Pattern]:
        """Detect file-based patterns.
        
        Args:
            files: Dictionary of FileInfo objects
            
        Returns:
            List of detected Pattern objects
        """
        patterns = []
        
        # Check for file-based patterns
        for pattern_name, pattern_info in self.file_patterns.items():
            confidence = self._calculate_file_pattern_confidence(pattern_info, files)
            if confidence > 0.3:  # Minimum confidence threshold
                pattern_files = self._get_files_matching_file_pattern(pattern_info, files)
                pattern = Pattern(
                    name=pattern_name,
                    type="file_pattern",
                    confidence=confidence,
                    files=pattern_files,
                    metadata=pattern_info.get('metadata', {})
                )
                patterns.append(pattern)
        
        return patterns
    
    def _calculate_pattern_confidence(self, pattern_info: Dict, directories: Dict[str, DirectoryInfo], 
                                     files: Dict[str, FileInfo]) -> float:
        """Calculate confidence for a project pattern.
        
        Args:
            pattern_info: Pattern information dictionary
            directories: Dictionary of DirectoryInfo objects
            files: Dictionary of FileInfo objects
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        required_dirs = pattern_info.get('required_directories', [])
        optional_dirs = pattern_info.get('optional_directories', [])
        required_files = pattern_info.get('required_files', [])
        
        # Count matches
        dir_matches = 0
        file_matches = 0
        total_checks = len(required_dirs) + len(optional_dirs) + len(required_files)
        
        if total_checks == 0:
            return 0.0
        
        # Check required directories
        for dir_pattern in required_dirs:
            if self._directory_matches_pattern(dir_pattern, directories):
                dir_matches += 1
            else:
                # Required directory missing - lower confidence significantly
                return 0.1
        
        # Check optional directories
        for dir_pattern in optional_dirs:
            if self._directory_matches_pattern(dir_pattern, directories):
                dir_matches += 0.5  # Partial credit for optional dirs
        
        # Check required files
        for file_pattern in required_files:
            if self._file_matches_pattern(file_pattern, files):
                file_matches += 1
            else:
                # Required file missing - lower confidence significantly
                return 0.1
        
        # Calculate confidence
        matched_items = dir_matches + file_matches
        confidence = matched_items / total_checks
        
        return min(confidence, 1.0)
    
    def _calculate_architecture_confidence(self, pattern_info: Dict, directories: Dict[str, DirectoryInfo], 
                                          files: Dict[str, FileInfo]) -> float:
        """Calculate confidence for an architecture pattern.
        
        Args:
            pattern_info: Pattern information dictionary
            directories: Dictionary of DirectoryInfo objects
            files: Dictionary of FileInfo objects
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # For architecture patterns, we look at directory structure and file organization
        indicators = pattern_info.get('indicators', [])
        anti_indicators = pattern_info.get('anti_indicators', [])
        
        positive_matches = 0
        negative_matches = 0
        
        # Check positive indicators
        for indicator in indicators:
            if self._check_architecture_indicator(indicator, directories, files):
                positive_matches += 1
        
        # Check anti-indicators (things that would contradict this pattern)
        for anti_indicator in anti_indicators:
            if self._check_architecture_indicator(anti_indicator, directories, files):
                negative_matches += 1
        
        total_indicators = len(indicators) + len(anti_indicators)
        if total_indicators == 0:
            return 0.0
        
        # Calculate confidence (positive matches minus negative matches)
        net_matches = positive_matches - negative_matches
        confidence = net_matches / len(indicators) if len(indicators) > 0 else 0.0
        
        return max(min(confidence, 1.0), 0.0)
    
    def _calculate_file_pattern_confidence(self, pattern_info: Dict, 
                                           files: Dict[str, FileInfo]) -> float:
        """Calculate confidence for a file pattern.
        
        Args:
            pattern_info: Pattern information dictionary
            files: Dictionary of FileInfo objects
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        patterns = pattern_info.get('patterns', [])
        file_types = pattern_info.get('file_types', [])
        
        matching_files = 0
        total_files = len(files)
        
        if total_files == 0:
            return 0.0
        
        # Check for pattern matches in files
        for file_info in files.values():
            # Check file type matches
            if file_types and file_info.type.value in file_types:
                matching_files += 1
            # Check content patterns
            elif patterns:
                for pattern in patterns:
                    if self._file_content_matches_pattern(pattern, file_info):
                        matching_files += 1
                        break
        
        confidence = matching_files / total_files
        return min(confidence, 1.0)
    
    def _directory_matches_pattern(self, dir_pattern: str, 
                                   directories: Dict[str, DirectoryInfo]) -> bool:
        """Check if a directory matches a pattern.
        
        Args:
            dir_pattern: Directory pattern to match
            directories: Dictionary of DirectoryInfo objects
            
        Returns:
            True if directory matches pattern, False otherwise
        """
        for dir_info in directories.values():
            if dir_pattern.lower() in dir_info.path.lower() or \
               dir_pattern.lower() == dir_info.name.lower():
                return True
        return False
    
    def _file_matches_pattern(self, file_pattern: str, 
                              files: Dict[str, FileInfo]) -> bool:
        """Check if a file matches a pattern.
        
        Args:
            file_pattern: File pattern to match
            files: Dictionary of FileInfo objects
            
        Returns:
            True if file matches pattern, False otherwise
        """
        for file_info in files.values():
            if file_pattern.lower() in file_info.path.lower() or \
               file_pattern.lower() == file_info.name.lower():
                return True
        return False
    
    def _check_architecture_indicator(self, indicator: str, 
                                     directories: Dict[str, DirectoryInfo], 
                                     files: Dict[str, FileInfo]) -> bool:
        """Check if an architecture indicator is present.
        
        Args:
            indicator: Architecture indicator to check
            directories: Dictionary of DirectoryInfo objects
            files: Dictionary of FileInfo objects
            
        Returns:
            True if indicator is present, False otherwise
        """
        # Check in directories
        for dir_info in directories.values():
            if indicator.lower() in dir_info.path.lower() or \
               indicator.lower() in dir_info.name.lower():
                return True
        
        # Check in files
        for file_info in files.values():
            if indicator.lower() in file_info.path.lower() or \
               indicator.lower() in file_info.name.lower():
                return True
        
        return False
    
    def _file_content_matches_pattern(self, pattern: str, 
                                     file_info: FileInfo) -> bool:
        """Check if file content matches a pattern.
        
        Args:
            pattern: Pattern to match in file content
            file_info: FileInfo object
            
        Returns:
            True if content matches pattern, False otherwise
        """
        # This would require reading file content, which we avoid for performance
        # In a real implementation, we might sample content or use metadata
        return pattern.lower() in file_info.name.lower()
    
    def _get_files_matching_pattern(self, pattern_info: Dict, 
                                    files: Dict[str, FileInfo]) -> List[str]:
        """Get files that match a pattern.
        
        Args:
            pattern_info: Pattern information dictionary
            files: Dictionary of FileInfo objects
            
        Returns:
            List of file paths that match the pattern
        """
        matching_files = []
        required_files = pattern_info.get('required_files', [])
        
        for file_path, file_info in files.items():
            for file_pattern in required_files:
                if file_pattern.lower() in file_path.lower() or \
                   file_pattern.lower() == file_info.name.lower():
                    matching_files.append(file_path)
                    break
        
        return matching_files
    
    def _get_files_matching_architecture_pattern(self, pattern_info: Dict, 
                                                files: Dict[str, FileInfo]) -> List[str]:
        """Get files that match an architecture pattern.
        
        Args:
            pattern_info: Pattern information dictionary
            files: Dictionary of FileInfo objects
            
        Returns:
            List of file paths that match the pattern
        """
        matching_files = []
        indicators = pattern_info.get('indicators', [])
        
        for file_path, file_info in files.items():
            for indicator in indicators:
                if indicator.lower() in file_path.lower() or \
                   indicator.lower() in file_info.name.lower():
                    matching_files.append(file_path)
                    break
        
        return matching_files
    
    def _get_files_matching_file_pattern(self, pattern_info: Dict, 
                                        files: Dict[str, FileInfo]) -> List[str]:
        """Get files that match a file pattern.
        
        Args:
            pattern_info: Pattern information dictionary
            files: Dictionary of FileInfo objects
            
        Returns:
            List of file paths that match the pattern
        """
        matching_files = []
        patterns = pattern_info.get('patterns', [])
        file_types = pattern_info.get('file_types', [])
        
        for file_path, file_info in files.items():
            # Check file types
            if file_types and file_info.type.value in file_types:
                matching_files.append(file_path)
            # Check patterns
            elif patterns:
                for pattern in patterns:
                    if pattern.lower() in file_path.lower() or \
                       pattern.lower() in file_info.name.lower():
                        matching_files.append(file_path)
                        break
        
        return matching_files
    
    def _create_project_patterns(self) -> Dict:
        """Create dictionary of project patterns.
        
        Returns:
            Dictionary of project pattern definitions
        """
        return {
            "standard_maven": {
                "required_directories": ["src/main", "src/test"],
                "optional_directories": ["src/main/java", "src/test/java", "src/main/resources"],
                "required_files": ["pom.xml"],
                "metadata": {"framework": "Java/Maven"}
            },
            "standard_gradle": {
                "required_directories": ["src/main", "src/test"],
                "optional_directories": ["src/main/java", "src/test/java", "src/main/resources"],
                "required_files": ["build.gradle", "gradlew"],
                "metadata": {"framework": "Java/Gradle"}
            },
            "standard_node": {
                "required_directories": ["node_modules"],
                "optional_directories": ["src", "test", "tests", "docs"],
                "required_files": ["package.json"],
                "metadata": {"framework": "Node.js"}
            },
            "standard_python": {
                "required_directories": ["venv", "__pycache__"],
                "optional_directories": ["src", "tests", "docs"],
                "required_files": ["requirements.txt", "setup.py"],
                "metadata": {"framework": "Python"}
            },
            "standard_django": {
                "required_directories": ["manage.py", "settings.py"],
                "optional_directories": ["apps", "migrations"],
                "required_files": ["manage.py"],
                "metadata": {"framework": "Django"}
            },
            "standard_react": {
                "required_directories": ["src", "public"],
                "optional_directories": ["components", "pages", "hooks"],
                "required_files": ["package.json"],
                "metadata": {"framework": "React"}
            }
        }
    
    def _create_architecture_patterns(self) -> Dict:
        """Create dictionary of architecture patterns.
        
        Returns:
            Dictionary of architecture pattern definitions
        """
        return {
            "microservices": {
                "indicators": ["services", "service", "api-gateway", "docker-compose"],
                "anti_indicators": ["monolith", "single-app"],
                "metadata": {"type": "Microservices Architecture"}
            },
            "monolith": {
                "indicators": ["app", "application", "main"],
                "anti_indicators": ["services", "distributed"],
                "metadata": {"type": "Monolithic Architecture"}
            },
            "modular": {
                "indicators": ["modules", "packages", "components"],
                "anti_indicators": ["single-file"],
                "metadata": {"type": "Modular Architecture"}
            },
            "layered": {
                "indicators": ["controllers", "services", "repositories", "models"],
                "anti_indicators": ["unstructured"],
                "metadata": {"type": "Layered Architecture"}
            }
        }
    
    def _create_file_patterns(self) -> Dict:
        """Create dictionary of file patterns.
        
        Returns:
            Dictionary of file pattern definitions
        """
        return {
            "configuration_files": {
                "file_types": ["config"],
                "patterns": ["config", "settings", ".yml", ".yaml", ".json", ".toml"],
                "metadata": {"type": "Configuration Files"}
            },
            "documentation_files": {
                "file_types": ["doc"],
                "patterns": ["readme", "license", "contributing", "changelog", ".md"],
                "metadata": {"type": "Documentation Files"}
            },
            "test_files": {
                "file_types": ["test"],
                "patterns": ["test", "spec", "e2e"],
                "metadata": {"type": "Test Files"}
            },
            "build_files": {
                "file_types": ["build"],
                "patterns": ["makefile", "dockerfile", "docker-compose", "webpack", "vite"],
                "metadata": {"type": "Build Files"}
            }
        }