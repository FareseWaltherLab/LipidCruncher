"""
Unit tests for friendly AI-client error mapping (_friendly_error).

Verifies that low-level Anthropic SDK exceptions are translated into
user-facing AIClientError messages, and that unrecognized exceptions keep
their original detail for debugging.
"""
import anthropic

from app.services.ai_chat.ai_client import _friendly_error
from app.services.ai_chat.exceptions import AIClientError


def _instance(cls, **attrs):
    """Build an instance of an Anthropic exception class without invoking its
    real __init__ (which requires request/response objects). isinstance checks
    depend on the type, not on __init__ having run."""
    obj = cls.__new__(cls)
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


def test_authentication_error_mentions_api_key():
    err = _friendly_error(_instance(anthropic.AuthenticationError))
    assert isinstance(err, AIClientError)
    assert "api key" in str(err).lower()


def test_rate_limit_error_is_actionable():
    err = _friendly_error(_instance(anthropic.RateLimitError))
    assert isinstance(err, AIClientError)
    assert "busy" in str(err).lower() or "rate limit" in str(err).lower()


def test_timeout_error_message():
    err = _friendly_error(_instance(anthropic.APITimeoutError))
    assert "too long" in str(err).lower()


def test_connection_error_message():
    err = _friendly_error(_instance(anthropic.APIConnectionError))
    text = str(err).lower()
    assert "reach" in text or "connection" in text


def test_server_error_is_temporary():
    err = _friendly_error(_instance(anthropic.APIStatusError, status_code=503))
    assert "temporarily unavailable" in str(err).lower()


def test_client_status_error_shows_status_code():
    err = _friendly_error(_instance(anthropic.APIStatusError, status_code=400))
    assert "http 400" in str(err).lower()


def test_unknown_exception_preserves_detail():
    err = _friendly_error(ValueError("boom"))
    assert isinstance(err, AIClientError)
    assert "boom" in str(err)


def test_context_prefix_on_unknown_exception():
    err = _friendly_error(ValueError("boom"), context="streaming")
    assert str(err).startswith("streaming:")
