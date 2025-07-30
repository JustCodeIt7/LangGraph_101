"""Framework-specific detection for repository analysis."""

import json
import yaml
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from ..core.data_structures import FileInfo, DirectoryInfo, Framework
from ..core.exceptions import FrameworkDetectionError


class FrameworkDetector:
    """Detects frameworks and technologies used in repositories."""
    
    def __init__(self):
        """Initialize the FrameworkDetector."""
        self.framework_signatures = self._create_framework_signatures()
        self.language_frameworks = self._create_language_frameworks()
    
    def detect_frameworks(self, files: Dict[str, FileInfo], 
                         directories: Dict[str, DirectoryInfo]) -> List[Framework]:
        """Detect frameworks used in the repository.
        
        Args:
            files: Dictionary of FileInfo objects
            directories: Dictionary of DirectoryInfo objects
            
        Returns:
            List of detected Framework objects
        """
        frameworks = []
        detected_frameworks = set()
        
        # Check configuration files for framework signatures
        for file_path, file_info in files.items():
            if file_info.type == file_info.CONFIG:
                framework_matches = self._detect_frameworks_in_config(file_path, file_info)
                for framework_name, confidence in framework_matches:
                    if framework_name not in detected_frameworks:
                        detected_frameworks.add(framework_name)
                        framework = Framework(
                            name=framework_name,
                            confidence=confidence,
                            files=[file_path]
                        )
                        frameworks.append(framework)
        
        # Check source files for framework signatures
        for file_path, file_info in files.items():
            if file_info.type == file_info.SOURCE and file_info.language:
                framework_matches = self._detect_frameworks_in_source(file_path, file_info)
                for framework_name, confidence in framework_matches:
                    if framework_name not in detected_frameworks:
                        detected_frameworks.add(framework_name)
                        framework = Framework(
                            name=framework_name,
                            confidence=confidence,
                            files=[file_path]
                        )
                        frameworks.append(framework)
        
        # Check directory structure for framework patterns
        framework_matches = self._detect_frameworks_in_structure(directories)
        for framework_name, confidence in framework_matches:
            if framework_name not in detected_frameworks:
                detected_frameworks.add(framework_name)
                framework = Framework(
                    name=framework_name,
                    confidence=confidence,
                    files=[]
                )
                frameworks.append(framework)
        
        return frameworks
    
    def _detect_frameworks_in_config(self, file_path: str, file_info: FileInfo) -> List[Tuple[str, float]]:
        """Detect frameworks based on configuration files.
        
        Args:
            file_path: Path to the configuration file
            file_info: FileInfo object for the file
            
        Returns:
            List of (framework_name, confidence) tuples
        """
        matches = []
        file_name = Path(file_path).name.lower()
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Check package.json for Node.js frameworks
            if file_name == 'package.json':
                try:
                    package_data = json.loads(content)
                    deps = package_data.get('dependencies', {})
                    dev_deps = package_data.get('devDependencies', {})
                    all_deps = {**deps, **dev_deps}
                    
                    # Check for specific frameworks
                    for framework, signature in self.framework_signatures.items():
                        if 'package_indicators' in signature:
                            for indicator in signature['package_indicators']:
                                if indicator in all_deps:
                                    confidence = signature['package_indicators'][indicator]
                                    matches.append((framework, confidence))
                except json.JSONDecodeError:
                    pass
            
            # Check requirements.txt for Python frameworks
            elif file_name == 'requirements.txt':
                lines = content.splitlines()
                for framework, signature in self.framework_signatures.items():
                    if 'requirement_indicators' in signature:
                        for indicator in signature['requirement_indicators']:
                            for line in lines:
                                if indicator in line and not line.strip().startswith('#'):
                                    confidence = signature['requirement_indicators'][indicator]
                                    matches.append((framework, confidence))
            
            # Check pom.xml for Java frameworks
            elif file_name == 'pom.xml':
                for framework, signature in self.framework_signatures.items():
                    if 'xml_indicators' in signature:
                        for indicator in signature['xml_indicators']:
                            if indicator in content:
                                confidence = signature['xml_indicators'][indicator]
                                matches.append((framework, confidence))
            
            # Check for Dockerfile patterns
            elif 'dockerfile' in file_name:
                for framework, signature in self.framework_signatures.items():
                    if 'docker_indicators' in signature:
                        for indicator in signature['docker_indicators']:
                            if indicator in content:
                                confidence = signature['docker_indicators'][indicator]
                                matches.append((framework, confidence))
        
        except Exception:
            # Silently continue if file reading fails
            pass
        
        return matches
    
    def _detect_frameworks_in_source(self, file_path: str, file_info: FileInfo) -> List[Tuple[str, float]]:
        """Detect frameworks based on source code files.
        
        Args:
            file_path: Path to the source file
            file_info: FileInfo object for the file
            
        Returns:
            List of (framework_name, confidence) tuples
        """
        matches = []
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Check for framework-specific imports/requirements
            for framework, signature in self.framework_signatures.items():
                if 'import_indicators' in signature:
                    for indicator in signature['import_indicators']:
                        if indicator in content:
                            confidence = signature['import_indicators'][indicator]
                            matches.append((framework, confidence))
        
        except Exception:
            # Silently continue if file reading fails
            pass
        
        return matches
    
    def _detect_frameworks_in_structure(self, directories: Dict[str, DirectoryInfo]) -> List[Tuple[str, float]]:
        """Detect frameworks based on directory structure.
        
        Args:
            directories: Dictionary of DirectoryInfo objects
            
        Returns:
            List of (framework_name, confidence) tuples
        """
        matches = []
        
        # Check directory names for framework patterns
        for framework, signature in self.framework_signatures.items():
            if 'directory_indicators' in signature:
                for dir_info in directories.values():
                    for indicator in signature['directory_indicators']:
                        if indicator.lower() in dir_info.path.lower():
                            confidence = signature['directory_indicators'][indicator]
                            matches.append((framework, confidence))
        
        return matches
    
    def _create_framework_signatures(self) -> Dict:
        """Create framework detection signatures.
        
        Returns:
            Dictionary of framework signatures
        """
        return {
            "React": {
                "package_indicators": {
                    "react": 0.9,
                    "react-dom": 0.9,
                    "next": 0.8,
                    "gatsby": 0.7,
                    "react-router": 0.8
                },
                "import_indicators": {
                    "import React": 0.9,
                    "from 'react'": 0.9,
                    "require('react')": 0.8
                },
                "directory_indicators": {
                    "components": 0.7,
                    "pages": 0.8,
                    "hooks": 0.6
                },
                "docker_indicators": {
                    "node_modules": 0.8
                }
            },
            "Vue.js": {
                "package_indicators": {
                    "vue": 0.9,
                    "nuxt": 0.8,
                    "@vue/cli": 0.7
                },
                "import_indicators": {
                    "import Vue": 0.9,
                    "from 'vue'": 0.9
                },
                "directory_indicators": {
                    "components": 0.7
                }
            },
            "Angular": {
                "package_indicators": {
                    "@angular/core": 0.9,
                    "@angular/cli": 0.8,
                    "@angular/common": 0.8
                },
                "import_indicators": {
                    "from '@angular": 0.9
                },
                "directory_indicators": {
                    "app": 0.6
                }
            },
            "Django": {
                "requirement_indicators": {
                    "Django==": 0.9,
                    "Django>=": 0.9,
                    "Django~=": 0.9
                },
                "import_indicators": {
                    "from django": 0.9,
                    "import django": 0.9
                },
                "directory_indicators": {
                    "manage.py": 0.9,
                    "settings.py": 0.9,
                    "wsgi.py": 0.8
                }
            },
            "Flask": {
                "requirement_indicators": {
                    "Flask==": 0.9,
                    "Flask>=": 0.9
                },
                "import_indicators": {
                    "from flask": 0.9,
                    "import flask": 0.9
                }
            },
            "FastAPI": {
                "requirement_indicators": {
                    "fastapi==": 0.9,
                    "fastapi>=": 0.9
                },
                "import_indicators": {
                    "from fastapi": 0.9,
                    "import fastapi": 0.9
                }
            },
            "Spring Boot": {
                "xml_indicators": {
                    "spring-boot-starter": 0.9,
                    "org.springframework": 0.8
                },
                "import_indicators": {
                    "import org.springframework": 0.9
                },
                "directory_indicators": {
                    "src/main/java": 0.8
                }
            },
            "Express.js": {
                "package_indicators": {
                    "express": 0.9
                },
                "import_indicators": {
                    "from 'express'": 0.9,
                    "require('express')": 0.9
                }
            },
            "Ruby on Rails": {
                "import_indicators": {
                    "require 'rails'": 0.9,
                    "gem 'rails'": 0.9
                },
                "directory_indicators": {
                    "app/controllers": 0.9,
                    "app/models": 0.9,
                    "app/views": 0.9
                }
            },
            "Laravel": {
                "import_indicators": {
                    "use Illuminate": 0.9
                },
                "directory_indicators": {
                    "app/Http": 0.8,
                    "resources/views": 0.8
                }
            }
        }
    
    def _create_language_frameworks(self) -> Dict:
        """Create mapping of languages to frameworks.
        
        Returns:
            Dictionary mapping languages to frameworks
        """
        return {
            "Python": ["Django", "Flask", "FastAPI", "Pyramid", "Bottle"],
            "JavaScript": ["React", "Vue.js", "Angular", "Express.js", "Next.js", "Nuxt.js"],
            "TypeScript": ["React", "Vue.js", "Angular", "Express.js", "NestJS"],
            "Java": ["Spring Boot", "Hibernate", "Struts", "Play Framework"],
            "C#": ["ASP.NET", "Entity Framework", ".NET Core"],
            "PHP": ["Laravel", "Symfony", "CodeIgniter", "Yii"],
            "Ruby": ["Ruby on Rails", "Sinatra", "Hanami"],
            "Go": ["Gin", "Echo", "Fiber", "Beego"],
            "Rust": ["Actix", "Rocket", "Warp", "Axum"]
        }