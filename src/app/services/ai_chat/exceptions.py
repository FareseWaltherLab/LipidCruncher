"""
Exception hierarchy for AI chat services.

Extends the shared ServiceError hierarchy with AI-specific error types.

Hierarchy:
    ServiceError (from services.exceptions)
    └── AIChatError (base for all AI chat errors)
        ├── AIClientError    (API communication failures)
        └── SandboxError     (code execution failures)
"""
from ..exceptions import ServiceError


class AIChatError(ServiceError):
    """Base exception for all AI chat service operations.

    Inherits from ServiceError so callers can catch at either the
    AI-chat level or the broader service level.
    """
    pass


class AIClientError(AIChatError):
    """Raised when the Anthropic API call fails.

    Examples: network errors, authentication failures, rate limits,
    invalid API responses.
    """
    pass


class SandboxError(AIChatError):
    """Raised when sandboxed code execution fails.

    Examples: forbidden operations detected during AST validation,
    execution timeout, runtime errors in user code.
    """
    pass

