"""
Typed exceptions for data cleaning operations.

Replaces generic ValueError usage with specific exception types
so callers can distinguish configuration errors from data errors
without fragile keyword-based string matching.
"""


class DataCleaningError(Exception):
    """Base exception for data cleaning operations."""
    pass


class ConfigurationError(DataCleaningError):
    """
    Raised when user-provided configuration causes cleaning to fail.

    Examples: overly strict grade filters, score thresholds that eliminate
    all data, invalid filter settings. These are recoverable by adjusting
    configuration.
    """
    pass


class EmptyDataError(DataCleaningError):
    """
    Raised when input data is empty or becomes empty after cleaning.

    Examples: empty upload, all rows filtered out, no valid lipid species.
    """
    pass