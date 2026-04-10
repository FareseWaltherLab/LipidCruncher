"""
Shared exception hierarchy for all service-layer operations.

Provides typed exceptions so callers can distinguish user-fixable
configuration errors from data errors and programming bugs, without
fragile string matching on error messages.

Hierarchy:
    ServiceError
    ├── ConfigurationError   (user-fixable settings)
    ├── EmptyDataError       (empty/missing input)
    └── ValidationError      (data fails validation)
"""


class ServiceError(ValueError):
    """Base exception for all service-layer operations.

    Inherits from ValueError for backward compatibility — existing code
    that catches ValueError will continue to work while callers can also
    catch the more specific ServiceError subtypes.

    Subclasses represent different failure categories so callers can
    handle them with targeted except clauses.
    """
    pass


class ConfigurationError(ServiceError):
    """Raised when user-provided configuration causes an operation to fail.

    Examples: overly strict grade filters, score thresholds that eliminate
    all data, invalid filter settings, missing normalization mappings.
    These are recoverable by adjusting configuration.
    """
    pass


class EmptyDataError(ServiceError):
    """Raised when input data is empty or becomes empty after processing.

    Examples: empty upload, all rows filtered out, no valid lipid species.
    """
    pass


class ValidationError(ServiceError):
    """Raised when data fails structural or content validation.

    Examples: missing required columns, wrong column types, insufficient
    samples for a statistical test.
    """
    pass