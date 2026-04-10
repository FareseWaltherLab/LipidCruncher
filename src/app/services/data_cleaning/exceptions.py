"""
Typed exceptions for data cleaning operations.

These inherit from the shared service exception hierarchy so callers
can catch either the specific cleaning exception or the broader
service-level category.
"""

from ..exceptions import (
    ServiceError,
    ConfigurationError as _ServiceConfigurationError,
    EmptyDataError as _ServiceEmptyDataError,
)


class DataCleaningError(ServiceError):
    """Base exception for data cleaning operations."""
    pass


class ConfigurationError(DataCleaningError, _ServiceConfigurationError):
    """Raised when user-provided configuration causes cleaning to fail.

    Examples: overly strict grade filters, score thresholds that eliminate
    all data, invalid filter settings. These are recoverable by adjusting
    configuration.

    Inherits from both DataCleaningError and services.ConfigurationError
    so it can be caught by either.
    """
    pass


class EmptyDataError(DataCleaningError, _ServiceEmptyDataError):
    """Raised when input data is empty or becomes empty after cleaning.

    Examples: empty upload, all rows filtered out, no valid lipid species.

    Inherits from both DataCleaningError and services.EmptyDataError
    so it can be caught by either.
    """
    pass