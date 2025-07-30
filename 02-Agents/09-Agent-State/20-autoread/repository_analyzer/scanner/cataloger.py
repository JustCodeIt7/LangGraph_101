"""File cataloger for extracting metadata from files."""

import os
import json
import yaml
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from ..core.data_structures import FileInfo, DirectoryInfo
from ..core.exceptions import AnalysisError


class FileCataloger:
    """Catalogs files and extracts detailed metadata."""
    
    def __init__(self):
        """Initialize the FileCataloger."""
        self.language_parsers = {
            'python': self._parse_python_file,
            'javascript': self._parse_javascript_file,
            'typescript': self._parse_typescript_file,
            'java': self._parse_java_file,
        }
    
    def catalog_files(self, files: Dict[str, FileInfo], directories: Dict[str, DirectoryInfo], 
                     repo_path: str) -> Dict[str, FileInfo]:
        """Catalog files and extract detailed metadata.
        
        Args:
            files: Dictionary of FileInfo objects
            directories: Dictionary of DirectoryInfo objects
            repo_path: Path to the repository root
            
        Returns:
            Updated dictionary of FileInfo objects with metadata
        """
        repo_path_obj = Path(repo_path)
        
        for file_path, file_info in files.items():
            try:
                full_path = repo_path_obj / file_path
                
                # Extract basic metadata
                self._extract_basic_metadata(file_info, full_path)
                
                # Extract language-specific metadata
                if file_info.language:
                    language_key = file_info.language.lower()
                    if language_key in self.language_parsers:
                        self.language_parsers[language_key](file_info, full_path)
                
                # Extract framework markers
                self._extract_framework_markers(file_info, full_path)
                
                # Update file info
                files[file_path] = file_info
            except Exception:
                # Continue with other files if one fails
                continue
        
        return files
    
    def _extract_basic_metadata(self, file_info: FileInfo, file_path: Path) -> None:
        """Extract basic metadata from a file.
        
        Args:
            file_info: FileInfo object to update
            file_path: Path to the file
        """
        try:
            # Get file stats
            stat = file_path.stat()
            file_info.metadata['created'] = stat.st_ctime
            file_info.metadata['modified'] = stat.st_mtime
            file_info.metadata['permissions'] = oct(stat.st_mode)[-3:]
            
            # Extract file content information
            if file_info.type in [FileInfo.SOURCE, FileInfo.CONFIG, FileInfo.DOC]:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    file_info.metadata['lines'] = len(content.splitlines())
                    file_info.metadata['characters'] = len(content)
                    file_info.metadata['words'] = len(content.split())
        except Exception:
            # Silently continue if metadata extraction fails
            pass
    
    def _parse_python_file(self, file_info: FileInfo, file_path: Path) -> None:
        """Parse a Python file and extract imports and other metadata.
        
        Args:
            file_info: FileInfo object to update
            file_path: Path to the Python file
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Extract imports
            imports = []
            
            # Standard import statements
            import_matches = re.findall(r'^import\s+(\w+)', content, re.MULTILINE)
            imports.extend(import_matches)
            
            # From import statements
            from_matches = re.findall(r'^from\s+(\w+)', content, re.MULTILINE)
            imports.extend(from_matches)
            
            file_info.imports = list(set(imports))
            
            # Extract class and function definitions
            classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
            functions = re.findall(r'^def\s+(\w+)', content, re.MULTILINE)
            
            file_info.metadata['classes'] = classes
            file_info.metadata['functions'] = functions
            
            # Extract docstrings for modules
            module_docstring = re.search(r'^["\']{3}(.*?)["\']{3}', content, re.DOTALL)
            if module_docstring:
                file_info.metadata['module_docstring'] = module_docstring.group(1).strip()
                
        except Exception:
            # Silently continue if parsing fails
            pass
    
    def _parse_javascript_file(self, file_info: FileInfo, file_path: Path) -> None:
        """Parse a JavaScript file and extract imports and other metadata.
        
        Args:
            file_info: FileInfo object to update
            file_path: Path to the JavaScript file
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Extract ES6 imports
            es6_imports = re.findall(r'^import.*?from\s+["\'](.+?)["\']', content, re.MULTILINE)
            file_info.imports.extend(es6_imports)
            
            # Extract CommonJS requires
            cjs_requires = re.findall(r'require\(["\'](.+?)["\']\)', content)
            file_info.imports.extend(cjs_requires)
            
            # Remove duplicates
            file_info.imports = list(set(file_info.imports))
            
            # Extract function and class declarations
            functions = re.findall(r'^function\s+(\w+)', content, re.MULTILINE)
            classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
            
            file_info.metadata['functions'] = functions
            file_info.metadata['classes'] = classes
            
        except Exception:
            # Silently continue if parsing fails
            pass
    
    def _parse_typescript_file(self, file_info: FileInfo, file_path: Path) -> None:
        """Parse a TypeScript file and extract imports and other metadata.
        
        Args:
            file_info: FileInfo object to update
            file_path: Path to the TypeScript file
        """
        # TypeScript parsing is similar to JavaScript
        self._parse_javascript_file(file_info, file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Extract TypeScript-specific features
            interfaces = re.findall(r'^interface\s+(\w+)', content, re.MULTILINE)
            types = re.findall(r'^type\s+(\w+)', content, re.MULTILINE)
            
            file_info.metadata['interfaces'] = interfaces
            file_info.metadata['types'] = types
                
        except Exception:
            # Silently continue if parsing fails
            pass
    
    def _parse_java_file(self, file_info: FileInfo, file_path: Path) -> None:
        """Parse a Java file and extract imports and other metadata.
        
        Args:
            file_info: FileInfo object to update
            file_path: Path to the Java file
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Extract imports
            imports = re.findall(r'^import\s+(?:static\s+)?([\w.]+)', content, re.MULTILINE)
            file_info.imports = list(set(imports))
            
            # Extract class declarations
            classes = re.findall(r'^\s*(?:public|protected|private)?\s*(?:abstract\s+)?class\s+(\w+)', 
                               content, re.MULTILINE)
            file_info.metadata['classes'] = classes
            
            # Extract package
            package_match = re.search(r'^package\s+([\w.]+)', content, re.MULTILINE)
            if package_match:
                file_info.metadata['package'] = package_match.group(1)
                
        except Exception:
            # Silently continue if parsing fails
            pass
    
    def _extract_framework_markers(self, file_info: FileInfo, file_path: Path) -> None:
        """Extract framework-specific markers from files.
        
        Args:
            file_info: FileInfo object to update
            file_path: Path to the file
        """
        try:
            # Only process certain file types
            if file_info.type not in [FileInfo.SOURCE, FileInfo.CONFIG, FileInfo.DOC]:
                return
                
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            markers = []
            
            # Check for common framework indicators
            if 'react' in content.lower():
                markers.append('react')
            if 'vue' in content.lower():
                markers.append('vue')
            if 'angular' in content.lower():
                markers.append('angular')
            if 'django' in content.lower():
                markers.append('django')
            if 'flask' in content.lower():
                markers.append('flask')
            if 'spring' in content.lower():
                markers.append('spring')
            if 'express' in content.lower():
                markers.append('express')
                
            # Configuration file specific markers
            if file_info.type == FileInfo.CONFIG:
                if 'package.json' in file_path.name:
                    try:
                        with open(file_path, 'r') as f:
                            package_data = json.load(f)
                            if 'dependencies' in package_data:
                                deps = package_data['dependencies']
                                markers.extend([dep for dep in deps.keys() if dep in [
                                    'react', 'vue', 'angular', '@angular/core',
                                    'express', 'koa', 'fastify',
                                    'next', 'nuxt', 'gatsby'
                                ]])
                    except:
                        pass
                elif 'requirements.txt' in file_path.name:
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            if 'Django' in content:
                                markers.append('django')
                            if 'Flask' in content:
                                markers.append('flask')
                    except:
                        pass
                        
            file_info.framework_markers = list(set(markers))
                
        except Exception:
            # Silently continue if marker extraction fails
            pass
    
    def catalog_directories(self, directories: Dict[str, DirectoryInfo], 
                           files: Dict[str, FileInfo]) -> Dict[str, DirectoryInfo]:
        """Catalog directories and extract detailed metadata.
        
        Args:
            directories: Dictionary of DirectoryInfo objects
            files: Dictionary of FileInfo objects
            
        Returns:
            Updated dictionary of DirectoryInfo objects with metadata
        """
        for dir_path, dir_info in directories.items():
            try:
                # Count files by type
                file_types = {}
                for file_path in dir_info.children:
                    if file_path in files:
                        file_type = files[file_path].type.value
                        file_types[file_type] = file_types.get(file_type, 0) + 1
                
                dir_info.metadata['file_types'] = file_types
                
                # Determine if directory has specific content patterns
                patterns = []
                for file_path in dir_info.children:
                    if file_path in files:
                        # Add framework markers from files
                        patterns.extend(files[file_path].framework_markers)
                        
                        # Check for special file types
                        if files[file_path].name.lower() in ['readme.md', 'readme.txt']:
                            patterns.append('documentation')
                        elif files[file_path].name.lower() in ['dockerfile']:
                            patterns.append('docker')
                        elif files[file_path].name.lower() in ['package.json', 'requirements.txt']:
                            patterns.append('dependencies')
                
                dir_info.patterns = list(set(patterns))
                
                # Update directory info
                directories[dir_path] = dir_info
            except Exception:
                # Continue with other directories if one fails
                continue
        
        return directories