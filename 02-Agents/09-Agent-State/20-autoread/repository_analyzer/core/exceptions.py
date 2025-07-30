"""Custom exceptions for repository analyzer."""

class RepositoryAnalyzerError(Exception):
    """Base exception for repository analyzer errors."""
    pass


class RepositoryNotFoundError(RepositoryAnalyzerError):
    """Raised when a repository cannot be found or accessed."""
    pass


class AuthenticationError(RepositoryAnalyzerError):
    """Raised when authentication fails for private repositories."""
    pass


class AnalysisError(RepositoryAnalyzerError):
    """Raised when analysis fails."""
    pass


class ConfigurationError(RepositoryAnalyzerError):
    """Raised when configuration is invalid."""
    pass


class GitError(RepositoryAnalyzerError):
    """Raised when Git operations fail."""
    pass


class FileSizeLimitError(RepositoryAnalyzerError):
    """Raised when a file exceeds the size limit."""
    pass


class UnsupportedFileTypeError(RepositoryAnalyzerError):
    """Raised when a file type is not supported."""
    pass


class PatternDetectionError(RepositoryAnalyzerError):
    """Raised when pattern detection fails."""
    pass


class FrameworkDetectionError(RepositoryAnalyzerError):
    """Raised when framework detection fails."""
    pass


class RelationshipMappingError(RepositoryAnalyzerError):
    """Raised when relationship mapping fails."""
    pass


class ValidationError(RepositoryAnalyzerError):
    """Raised when validation fails."""
    pass