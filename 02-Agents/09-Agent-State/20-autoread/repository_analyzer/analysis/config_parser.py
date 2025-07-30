"""Configuration file parser for various formats."""

import json
import yaml
import toml
import configparser
from pathlib import Path
from typing import Dict, Any, Optional, List
from ..core.exceptions import AnalysisError


class ConfigFileParser:
    """Parses configuration files in various formats."""
    
    def __init__(self):
        """Initialize the ConfigFileParser."""
        self.supported_formats = {
            '.json': self._parse_json,
            '.yaml': self._parse_yaml,
            '.yml': self._parse_yaml,
            '.toml': self._parse_toml,
            '.ini': self._parse_ini,
            '.cfg': self._parse_ini,
            '.conf': self._parse_ini
        }
    
    def parse_config_file(self, file_path: str) -> Dict[str, Any]:
        """Parse a configuration file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            Dictionary containing parsed configuration data
            
        Raises:
            AnalysisError: If parsing fails
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension not in self.supported_formats:
            raise AnalysisError(f"Unsupported configuration file format: {extension}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return self.supported_formats[extension](f)
        except Exception as e:
            raise AnalysisError(f"Failed to parse configuration file {file_path}: {e}")
    
    def _parse_json(self, file_handle) -> Dict[str, Any]:
        """Parse a JSON configuration file.
        
        Args:
            file_handle: File handle to read from
            
        Returns:
            Dictionary containing parsed configuration data
        """
        content = file_handle.read()
        return json.loads(content)
    
    def _parse_yaml(self, file_handle) -> Dict[str, Any]:
        """Parse a YAML configuration file.
        
        Args:
            file_handle: File handle to read from
            
        Returns:
            Dictionary containing parsed configuration data
        """
        return yaml.safe_load(file_handle)
    
    def _parse_toml(self, file_handle) -> Dict[str, Any]:
        """Parse a TOML configuration file.
        
        Args:
            file_handle: File handle to read from
            
        Returns:
            Dictionary containing parsed configuration data
        """
        content = file_handle.read()
        return toml.loads(content)
    
    def _parse_ini(self, file_handle) -> Dict[str, Any]:
        """Parse an INI configuration file.
        
        Args:
            file_handle: File handle to read from
            
        Returns:
            Dictionary containing parsed configuration data
        """
        config = configparser.ConfigParser()
        config.read_file(file_handle)
        
        # Convert to dictionary
        result = {}
        for section in config.sections():
            result[section] = dict(config.items(section))
        
        return result
    
    def get_config_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from a configuration file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            Dictionary containing configuration metadata
        """
        metadata = {
            'file_path': file_path,
            'file_size': 0,
            'sections': [],
            'keys': [],
            'format': None,
            'complexity': 0
        }
        
        try:
            path = Path(file_path)
            metadata['file_size'] = path.stat().st_size
            metadata['format'] = path.suffix.lower()
            
            # Parse the configuration to get structure info
            config_data = self.parse_config_file(file_path)
            
            if isinstance(config_data, dict):
                metadata['sections'] = list(config_data.keys())
                metadata['keys'] = self._extract_keys(config_data)
                metadata['complexity'] = self._calculate_complexity(config_data)
        except Exception:
            # Silently continue if metadata extraction fails
            pass
        
        return metadata
    
    def _extract_keys(self, config_data: Dict[str, Any], prefix: str = '') -> List[str]:
        """Extract all keys from configuration data.
        
        Args:
            config_data: Configuration data dictionary
            prefix: Prefix for nested keys
            
        Returns:
            List of all keys
        """
        keys = []
        
        for key, value in config_data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            keys.append(full_key)
            
            if isinstance(value, dict):
                keys.extend(self._extract_keys(value, full_key))
        
        return keys
    
    def _calculate_complexity(self, config_data: Dict[str, Any]) -> int:
        """Calculate complexity score for configuration data.
        
        Args:
            config_data: Configuration data dictionary
            
        Returns:
            Complexity score
        """
        complexity = 0
        
        def calculate_depth(data, current_depth=0):
            nonlocal complexity
            if isinstance(data, dict):
                complexity += len(data) * (current_depth + 1)
                for value in data.values():
                    calculate_depth(value, current_depth + 1)
            elif isinstance(data, list):
                complexity += len(data) * (current_depth + 1)
                for item in data:
                    calculate_depth(item, current_depth + 1)
            else:
                complexity += current_depth + 1
        
        calculate_depth(config_data)
        return complexity
    
    def find_config_files(self, directory: str) -> List[str]:
        """Find configuration files in a directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of configuration file paths
        """
        config_files = []
        path = Path(directory)
        
        if not path.exists():
            return config_files
        
        # Look for common configuration files
        common_config_names = [
            'config', 'configuration', 'settings', 'appsettings',
            'package', 'requirements', 'pom', 'build',
            'docker-compose', 'dockerfile', 'makefile', 'webpack.config'
        ]
        
        # Search for files with supported extensions
        for file_path in path.rglob('*'):
            if file_path.is_file():
                # Check extension
                if file_path.suffix.lower() in self.supported_formats:
                    config_files.append(str(file_path))
                # Check name patterns
                elif any(name in file_path.name.lower() for name in common_config_names):
                    config_files.append(str(file_path))
        
        return config_files