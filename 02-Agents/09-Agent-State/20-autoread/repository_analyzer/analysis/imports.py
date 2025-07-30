"""Import analysis for source code files."""

import re
from typing import Dict, List, Set, Optional
from ..core.data_structures import FileInfo
from ..core.exceptions import AnalysisError


class ImportAnalyzer:
    """Analyzes imports in source code files."""
    
    def __init__(self):
        """Initialize the ImportAnalyzer."""
        self.language_import_patterns = self._create_import_patterns()
    
    def analyze_imports(self, files: Dict[str, FileInfo]) -> Dict[str, FileInfo]:
        """Analyze imports in source code files.
        
        Args:
            files: Dictionary of FileInfo objects
            
        Returns:
            Updated dictionary of FileInfo objects with import analysis
        """
        for file_path, file_info in files.items():
            if file_info.type == file_info.SOURCE and file_info.language:
                try:
                    # Extract imports from the file
                    imports = self._extract_imports(file_path, file_info.language)
                    file_info.imports = imports
                    
                    # Update file info
                    files[file_path] = file_info
                except Exception:
                    # Continue with other files if one fails
                    continue
        
        return files
    
    def _extract_imports(self, file_path: str, language: str) -> List[str]:
        """Extract imports from a source code file.
        
        Args:
            file_path: Path to the source file
            language: Programming language of the file
            
        Returns:
            List of import paths
        """
        imports = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            language_lower = language.lower()
            
            # Use appropriate pattern for the language
            if language_lower in self.language_import_patterns:
                patterns = self.language_import_patterns[language_lower]
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.MULTILINE)
                    imports.extend(matches)
            
            # Remove duplicates
            imports = list(set(imports))
        except Exception:
            # Silently continue if file reading fails
            pass
        
        return imports
    
    def _create_import_patterns(self) -> Dict[str, List[str]]:
        """Create import patterns for different languages.
        
        Returns:
            Dictionary mapping languages to import patterns
        """
        return {
            'python': [
                r'^import\s+([\w.]+)',
                r'^from\s+([\w.]+)\s+import',
                r'^from\s+\.*([\w./]+)\s+import'
            ],
            'javascript': [
                r'^import\s+.*?from\s+["\'](.+?)["\']',
                r'^import\s+["\'](.+?)["\']',
                r'require\(["\'](.+?)["\']\)'
            ],
            'typescript': [
                r'^import\s+.*?from\s+["\'](.+?)["\']',
                r'^import\s+["\'](.+?)["\']',
                r'require\(["\'](.+?)["\']\)'
            ],
            'java': [
                r'^import\s+(?:static\s+)?([\w.]+)',
                r'^import\s+(?:static\s+)?([\w.]+)\.\*;'
            ],
            'c#': [
                r'^using\s+([\w.]+);'
            ],
            'go': [
                r'^import\s*\(?["\'](.+?)["\']\)?',
                r'^import\s*\(\s*.*?["\'](.+?)["\'].*?\s*\)'
            ],
            'rust': [
                r'^use\s+([\w::]+)',
                r'^use\s+([\w::]+)::\{.*?\}'
            ],
            'php': [
                r'^use\s+([\w\\]+)',
                r'^require(?:_once)?\s*\(["\'](.+?)["\']\)',
                r'^include(?:_once)?\s*\(["\'](.+?)["\']\)'
            ]
        }
    
    def get_import_graph(self, files: Dict[str, FileInfo]) -> Dict[str, List[str]]:
        """Create an import graph showing dependencies between files.
        
        Args:
            files: Dictionary of FileInfo objects
            
        Returns:
            Dictionary mapping file paths to lists of imported file paths
        """
        import_graph = {}
        
        # Initialize graph with all files
        for file_path in files.keys():
            import_graph[file_path] = []
        
        # Populate import relationships
        for file_path, file_info in files.items():
            if file_info.imports:
                resolved_imports = []
                for import_path in file_info.imports:
                    resolved_path = self._resolve_import_path(import_path, file_path, files)
                    if resolved_path:
                        resolved_imports.append(resolved_path)
                import_graph[file_path] = resolved_imports
        
        return import_graph
    
    def _resolve_import_path(self, import_path: str, source_file_path: str, 
                           files: Dict[str, FileInfo]) -> Optional[str]:
        """Resolve an import path to a specific file.
        
        Args:
            import_path: The import path to resolve
            source_file_path: Path of the source file containing the import
            files: Dictionary of FileInfo objects
            
        Returns:
            Path to the resolved file, or None if not found
        """
        # Handle relative imports
        if import_path.startswith('.'):
            # Resolve relative path
            source_dir = '/'.join(source_file_path.split('/')[:-1])
            if source_dir:
                # Combine source directory with import path
                target_parts = source_dir.split('/') + import_path.split('/')
                # Remove empty parts and navigate up for '..'
                while '..' in target_parts:
                    idx = target_parts.index('..')
                    if idx > 0:
                        target_parts.pop(idx)  # Remove '..'
                        target_parts.pop(idx-1)  # Remove parent directory
                    else:
                        target_parts.pop(idx)
                
                # Remove '.' parts
                target_parts = [part for part in target_parts if part and part != '.']
                target_path = '/'.join(target_parts)
            else:
                target_path = import_path[1:]  # Remove leading '.'
        else:
            # Handle absolute imports
            target_path = import_path.replace('.', '/')
        
        # Look for matching files
        # Try exact match first
        for file_path in files.keys():
            if file_path == f"{target_path}.py" or file_path.endswith(f"/{target_path}.py"):
                return file_path
        
        # Try directory match
        for file_path in files.keys():
            if file_path.startswith(target_path) or target_path in file_path:
                return file_path
        
        return None