"""Custom exceptions for input handling."""

from ..core.exceptions import RepositoryAnalyzerError


class InputHandlerError(RepositoryAnalyzerError):
    """Base exception for InputHandler errors."""
    
    def __init__(self, message: str, input_source: str = None, provider: str = None):
        super().__init__(message)
        self.input_source = input_source
        self.provider = provider


class InputValidationError(InputHandlerError):
    """Input validation failed."""
    
    def __init__(self, message: str, input_source: str = None, validation_type: str = None):
        super().__init__(message, input_source)
        self.validation_type = validation_type


class InputAuthenticationError(InputHandlerError):
    """Authentication failed for input source."""
    
    def __init__(self, message: str, input_source: str = None, provider: str = None, auth_method: str = None):
        super().__init__(message, input_source, provider)
        self.auth_method = auth_method


class InputProcessingError(InputHandlerError):
    """Input processing failed."""
    pass


class ResourceManagementError(InputHandlerError):
    """Resource management operation failed."""
    pass


class URLNormalizationError(InputHandlerError):
    """URL normalization failed."""
    pass