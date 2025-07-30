"""Relationship mapping for repository analysis."""

import re
from typing import Dict, List, Set, Tuple, Optional
from ..core.data_structures import FileInfo, DirectoryInfo, Relationship
from ..core.exceptions import RelationshipMappingError


class RelationshipMapper:
    """Maps relationships between files and directories in a repository."""
    
    def __init__(self):
        """Initialize the RelationshipMapper."""
        self.relationship_types = {
            'import': 'Import dependency',
            'config': 'Configuration reference',
            'inheritance': 'Class inheritance',
            'composition': 'Object composition',
            'function_call': 'Function call',
            'file_reference': 'File reference'
        }
    
    def map_relationships(self, files: Dict[str, FileInfo], 
                         directories: Dict[str, DirectoryInfo]) -> List[Relationship]:
        """Map relationships between components in the repository.
        
        Args:
            files: Dictionary of FileInfo objects
            directories: Dictionary of DirectoryInfo objects
            
        Returns:
            List of Relationship objects
        """
        relationships = []
        
        # Map import relationships
        import_relationships = self._map_import_relationships(files)
        relationships.extend(import_relationships)
        
        # Map configuration relationships
        config_relationships = self._map_config_relationships(files, directories)
        relationships.extend(config_relationships)
        
        # Map directory relationships
        directory_relationships = self._map_directory_relationships(directories)
        relationships.extend(directory_relationships)
        
        # Remove duplicates and return
        return self._deduplicate_relationships(relationships)
    
    def _map_import_relationships(self, files: Dict[str, FileInfo]) -> List[Relationship]:
        """Map import relationships between source files.
        
        Args:
            files: Dictionary of FileInfo objects
            
        Returns:
            List of import Relationship objects
        """
        relationships = []
        
        for source_file_path, source_file in files.items():
            # Only process source files with imports
            if not source_file.imports:
                continue
            
            for import_path in source_file.imports:
                # Try to find the target file for this import
                target_file_path = self._find_import_target(import_path, source_file_path, files)
                
                if target_file_path:
                    relationship = Relationship(
                        source=source_file_path,
                        target=target_file_path,
                        type='import',
                        strength=0.8,  # High confidence for direct imports
                        metadata={'import_path': import_path}
                    )
                    relationships.append(relationship)
        
        return relationships
    
    def _map_config_relationships(self, files: Dict[str, FileInfo], 
                               directories: Dict[str, DirectoryInfo]) -> List[Relationship]:
        """Map configuration relationships.
        
        Args:
            files: Dictionary of FileInfo objects
            directories: Dictionary of DirectoryInfo objects
            
        Returns:
            List of configuration Relationship objects
        """
        relationships = []
        
        # Look for configuration files that reference other files
        for config_file_path, config_file in files.items():
            if config_file.type != config_file.CONFIG:
                continue
            
            # Check if this config file references other files
            referenced_files = self._find_config_references(config_file_path, files)
            for referenced_file in referenced_files:
                relationship = Relationship(
                    source=config_file_path,
                    target=referenced_file,
                    type='config',
                    strength=0.7,  # Medium confidence for config references
                    metadata={'config_type': config_file.name}
                )
                relationships.append(relationship)
        
        return relationships
    
    def _map_directory_relationships(self, directories: Dict[str, DirectoryInfo]) -> List[Relationship]:
        """Map relationships between directories.
        
        Args:
            directories: Dictionary of DirectoryInfo objects
            
        Returns:
            List of directory Relationship objects
        """
        relationships = []
        
        # Map parent-child directory relationships
        for dir_path, dir_info in directories.items():
            if dir_path == '.':
                continue  # Skip root directory
            
            # Parent directory relationship
            parent_path = '/'.join(dir_path.split('/')[:-1]) if '/' in dir_path else '.'
            if parent_path == '':
                parent_path = '.'
            
            relationship = Relationship(
                source=dir_path,
                target=parent_path,
                type='parent',
                strength=1.0,  # Certain relationship
                metadata={'relationship_type': 'directory_hierarchy'}
            )
            relationships.append(relationship)
            
            # Sibling directory relationships
            if parent_path in directories:
                siblings = directories[parent_path].children
                for sibling in siblings:
                    if sibling != dir_path and sibling in directories:
                        relationship = Relationship(
                            source=dir_path,
                            target=sibling,
                            type='sibling',
                            strength=0.3,  # Low confidence for sibling relationships
                            metadata={'relationship_type': 'directory_siblings'}
                        )
                        relationships.append(relationship)
        
        return relationships
    
    def _find_import_target(self, import_path: str, source_file_path: str, 
                           files: Dict[str, FileInfo]) -> Optional[str]:
        """Find the target file for an import statement.
        
        Args:
            import_path: The import path to resolve
            source_file_path: Path of the source file containing the import
            files: Dictionary of FileInfo objects
            
        Returns:
            Path to the target file, or None if not found
        """
        # Handle relative imports
        if import_path.startswith('.'):
            # Resolve relative import path
            source_dir = '/'.join(source_file_path.split('/')[:-1])
            if source_dir:
                # Combine source directory with import path
                target_path_parts = source_dir.split('/') + import_path.split('.')
                # Remove empty parts and navigate up for '..'
                while '..' in target_path_parts:
                    idx = target_path_parts.index('..')
                    if idx > 0:
                        target_path_parts.pop(idx)  # Remove '..'
                        target_path_parts.pop(idx-1)  # Remove parent directory
                    else:
                        target_path_parts.pop(idx)
                
                # Remove '.' parts
                target_path_parts = [part for part in target_path_parts if part != '.']
                
                target_dir = '/'.join(target_path_parts)
                target_file_name = target_path_parts[-1] if target_path_parts else ''
            else:
                target_dir = import_path[1:]  # Remove leading '.'
                target_file_name = target_dir.split('.')[-1]
        else:
            # Handle absolute imports
            target_dir = '/'.join(import_path.split('.')[:-1])
            target_file_name = import_path.split('.')[-1]
        
        # Look for matching files
        for file_path in files.keys():
            # Check if file path matches the import
            if target_file_name and file_path.endswith(f"{target_file_name}.py"):
                return file_path
            elif target_dir and file_path.startswith(target_dir):
                return file_path
        
        return None
    
    def _find_config_references(self, config_file_path: str, 
                               files: Dict[str, FileInfo]) -> List[str]:
        """Find files referenced in a configuration file.
        
        Args:
            config_file_path: Path to the configuration file
            files: Dictionary of FileInfo objects
            
        Returns:
            List of referenced file paths
        """
        referenced_files = []
        
        try:
            # Read config file content
            with open(config_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Look for file path references in the config
            # This is a simplified approach - in practice, this would be more sophisticated
            file_path_pattern = r'["\']([/\w\-\.]+(?:\.[\w]+))["\']'
            matches = re.findall(file_path_pattern, content)
            
            for match in matches:
                # Check if this looks like a file reference
                if '.' in match or '/' in match:
                    # Try to find matching files
                    for file_path in files.keys():
                        if match in file_path or file_path.endswith(match):
                            referenced_files.append(file_path)
                            break
        except Exception:
            # Silently continue if file reading fails
            pass
        
        return referenced_files
    
    def _deduplicate_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """Remove duplicate relationships.
        
        Args:
            relationships: List of Relationship objects
            
        Returns:
            List of unique Relationship objects
        """
        unique_relationships = []
        seen = set()
        
        for relationship in relationships:
            # Create a key to identify unique relationships
            key = (relationship.source, relationship.target, relationship.type)
            if key not in seen:
                seen.add(key)
                unique_relationships.append(relationship)
        
        return unique_relationships
    
    def calculate_relationship_strength(self, relationship: Relationship, 
                                      files: Dict[str, FileInfo]) -> float:
        """Calculate the strength of a relationship.
        
        Args:
            relationship: Relationship object
            files: Dictionary of FileInfo objects
            
        Returns:
            Relationship strength between 0.0 and 1.0
        """
        base_strength = relationship.strength or 0.5
        
        # Adjust strength based on file types and metadata
        source_file = files.get(relationship.source)
        target_file = files.get(relationship.target)
        
        if not source_file or not target_file:
            return base_strength
        
        # Increase strength for same language files
        if source_file.language == target_file.language:
            base_strength = min(base_strength + 0.2, 1.0)
        
        # Increase strength for framework-related files
        if source_file.framework_markers or target_file.framework_markers:
            common_frameworks = set(source_file.framework_markers) & set(target_file.framework_markers)
            if common_frameworks:
                base_strength = min(base_strength + 0.1 * len(common_frameworks), 1.0)
        
        return base_strength