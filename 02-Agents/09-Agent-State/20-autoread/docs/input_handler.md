# InputHandler System Documentation

## Overview

The InputHandler system is a unified input processing framework designed to handle various repository sources including local file paths and remote Git repositories. It provides centralized validation, authentication, and processing capabilities that replace the scattered input handling logic previously found throughout the codebase.

## Architecture

The InputHandler system follows a modular architecture with the following key components:

```
InputHandler
├── InputValidator      # Input validation framework
├── InputClassifier     # Input source classification
├── AuthManager        # Authentication management
├── TempDirectoryManager # Temporary directory management
└── URLParser          # URL parsing and normalization
```

## Key Features

### 1. Unified Input Processing
The InputHandler class serves as the single entry point for all input processing operations, providing a consistent interface regardless of the input source type.

### 2. Comprehensive Input Validation
The system validates all inputs against security and format requirements, including:
- Path traversal prevention for local paths
- URL format validation for remote sources
- Size limits and length restrictions
- Permission checks for local files

### 3. Multi-Provider Authentication
Support for multiple Git providers with flexible authentication methods:
- GitHub, GitLab, Bitbucket support
- Token-based authentication
- SSH key authentication
- Environment variable integration

### 4. Temporary Resource Management
Advanced temporary directory management with automatic cleanup and resource tracking.

### 5. Error Handling
Comprehensive exception hierarchy with specific error types for different failure scenarios.

## Usage Examples

### Basic Usage

```python
from repository_analyzer.input.handler import InputHandler
from repository_analyzer.input.config import InputConfig

# Configure InputHandler
config = InputConfig(
    temp_dir="/tmp/repo_analyzer",
    timeout=300
)

# Process input
input_handler = InputHandler(config)
processed_input = input_handler.process("https://github.com/user/repo.git")

# Use processed input
print(f"Local path: {processed_input.local_path}")
print(f"Is temporary: {processed_input.is_temporary}")

# Clean up when done
input_handler.cleanup(processed_input)
```

### Context Manager Usage

```python
from repository_analyzer.input.handler import InputHandler

# Use as context manager for automatic cleanup
with InputHandler(config) as handler:
    processed_input = handler.process("/path/to/local/repo")
    # Process the repository...
# Automatic cleanup happens here
```

### Integration with RepositoryAnalyzer

```python
from repository_analyzer.core.analyzer import RepositoryAnalyzer
from repository_analyzer.core.config import AnalysisConfig

# RepositoryAnalyzer now uses InputHandler internally
analyzer_config = AnalysisConfig()
analyzer = RepositoryAnalyzer(analyzer_config)

# Process any supported input type
structure = analyzer.analyze("https://github.com/user/repo.git")
# or
structure = analyzer.analyze("/local/path/to/repo")
```

## Configuration Options

The InputHandler system is highly configurable through the InputConfig class:

```python
from repository_analyzer.input.config import InputConfig

config = InputConfig(
    # Authentication settings
    enable_authentication=True,
    auth_timeout=30,
    
    # Validation settings
    max_path_length=4096,
    allowed_url_schemes=['http', 'https', 'git', 'ssh'],
    enable_path_traversal_check=True,
    
    # Processing settings
    timeout=300,
    retry_attempts=3,
    max_repo_size=500 * 1024 * 1024,  # 500MB
    
    # Temporary directory settings
    temp_dir="/tmp/repo_analyzer",
    auto_cleanup=True,
    
    # Security settings
    mask_credentials_in_logs=True,
    validate_ssl_certificates=True
)
```

## Security Features

### Path Traversal Prevention
The system validates all local paths to prevent directory traversal attacks.

### Credential Security
Authentication credentials are masked in logs and error messages.

### URL Validation
All URLs are validated against strict format requirements to prevent malicious input.

### SSL Certificate Validation
HTTPS connections are validated against system certificates by default.

## Error Handling

The InputHandler system provides a comprehensive exception hierarchy:

```python
from repository_analyzer.input.exceptions import (
    InputHandlerError,
    InputValidationError,
    InputAuthenticationError,
    InputProcessingError,
    ResourceManagementError,
    URLNormalizationError
)

try:
    processed_input = input_handler.process(source)
except InputValidationError as e:
    # Handle validation errors
    print(f"Validation failed: {e}")
except InputAuthenticationError as e:
    # Handle authentication errors
    print(f"Authentication failed: {e}")
except InputHandlerError as e:
    # Handle other input processing errors
    print(f"Processing failed: {e}")
```

## Integration with LangGraph

The InputHandler integrates seamlessly with LangGraph workflows through the InputProcessorNode:

```python
from repository_analyzer.input.nodes import InputProcessorNode

# Create input processor node
input_node = InputProcessorNode(input_config)

# Use in LangGraph workflow
workflow.add_node("process_input", input_node)
```

## Migration from Previous System

The InputHandler system replaces the following components:
- `GitCloner._prepare_repository()` method
- Scattered input validation logic in `RepositoryAnalyzer`
- Manual temporary directory management

The migration maintains backward compatibility while providing enhanced functionality.

## Testing

The InputHandler system includes comprehensive test coverage:

```bash
# Run input handler tests
pytest tests/test_input/
```

Test categories include:
- Unit tests for individual components
- Integration tests for end-to-end workflows
- Security tests for validation and authentication
- Performance tests for resource management

## Performance Considerations

### Caching
The system implements caching for repeated operations to improve performance.

### Resource Pooling
Temporary directories are managed through a pooling system to reduce overhead.

### Parallel Processing
The system is designed to support concurrent input processing operations.

## Future Enhancements

Planned improvements include:
- Archive file support (ZIP, TAR)
- Direct API integration for metadata retrieval
- Incremental repository updates
- Advanced caching mechanisms
- Multi-source input support