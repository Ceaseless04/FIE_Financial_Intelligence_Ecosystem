"""Error hierarchy shared by every FIE service.

Errors carry a stable machine-readable ``code``, an HTTP status for API
translation, and a ``retryable`` flag that resilience helpers key on. Nothing
here is domain-specific: financial semantics belong to the product packages.
"""

from __future__ import annotations

from typing import Any


class FIEError(Exception):
    """Base class for every error raised inside the ecosystem.

    Subclasses set ``code``, ``http_status`` and ``retryable`` as class
    attributes so that callers can branch on the class rather than on message
    text, and so API layers can translate without a lookup table.
    """

    code: str = "fie_error"
    http_status: int = 500
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}
        if cause is not None:
            self.__cause__ = cause

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the wire shape used by API error responses."""
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


# --------------------------------------------------------------------------
# Configuration and programming errors
# --------------------------------------------------------------------------


class ConfigurationError(FIEError):
    """Invalid, missing, or unsafe configuration detected at startup."""

    code = "configuration_error"
    http_status = 500


class ValidationError(FIEError):
    """Input failed validation before reaching business logic."""

    code = "validation_error"
    http_status = 422


# --------------------------------------------------------------------------
# Access control
# --------------------------------------------------------------------------


class AuthenticationError(FIEError):
    """Caller identity could not be established."""

    code = "authentication_error"
    http_status = 401


class AuthorizationError(FIEError):
    """Caller is known but lacks permission for the requested action."""

    code = "authorization_error"
    http_status = 403


# --------------------------------------------------------------------------
# Resource state
# --------------------------------------------------------------------------


class NotFoundError(FIEError):
    """Requested resource does not exist."""

    code = "not_found"
    http_status = 404


class ConflictError(FIEError):
    """Request conflicts with current resource state."""

    code = "conflict"
    http_status = 409


# --------------------------------------------------------------------------
# Transient / infrastructure failures
# --------------------------------------------------------------------------


class RateLimitError(FIEError):
    """Caller exceeded a rate limit."""

    code = "rate_limited"
    http_status = 429
    retryable = True

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        details: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, details=details, cause=cause)
        self.retry_after_seconds = retry_after_seconds
        if retry_after_seconds is not None:
            self.details.setdefault("retry_after_seconds", retry_after_seconds)


class OperationTimeoutError(FIEError):
    """An operation exceeded its deadline."""

    code = "timeout"
    http_status = 504
    retryable = True


class ExternalServiceError(FIEError):
    """A dependency outside this service failed."""

    code = "external_service_error"
    http_status = 502
    retryable = True


class CircuitOpenError(FIEError):
    """A circuit breaker rejected the call without attempting it."""

    code = "circuit_open"
    http_status = 503
    retryable = True


# --------------------------------------------------------------------------
# AI provider failures
# --------------------------------------------------------------------------


class AIProviderError(ExternalServiceError):
    """An AI provider call failed."""

    code = "ai_provider_error"


class AIProviderTimeoutError(AIProviderError):
    """An AI provider call exceeded its deadline."""

    code = "ai_provider_timeout"
    http_status = 504


class AIProviderRateLimitError(AIProviderError):
    """An AI provider rejected the call for rate limiting."""

    code = "ai_provider_rate_limited"
    http_status = 429


class AIProviderRefusalError(AIProviderError):
    """The model declined the request on safety grounds.

    Not retryable against the same model: the Claude API returns HTTP 200 with
    ``stop_reason == "refusal"``, so retrying the identical request produces the
    identical refusal. Route to a fallback provider instead.
    """

    code = "ai_provider_refusal"
    http_status = 422
    retryable = False


class AIProviderResponseError(AIProviderError):
    """The provider returned a response that failed schema or grounding checks."""

    code = "ai_provider_response_error"
    http_status = 502
    retryable = False


# --------------------------------------------------------------------------
# Data layer failures
# --------------------------------------------------------------------------


class DatabaseError(ExternalServiceError):
    """A database operation failed."""

    code = "database_error"


class EventBusError(ExternalServiceError):
    """An event bus publish or consume operation failed."""

    code = "event_bus_error"


__all__ = [
    "AIProviderError",
    "AIProviderRateLimitError",
    "AIProviderRefusalError",
    "AIProviderResponseError",
    "AIProviderTimeoutError",
    "AuthenticationError",
    "AuthorizationError",
    "CircuitOpenError",
    "ConfigurationError",
    "ConflictError",
    "DatabaseError",
    "EventBusError",
    "ExternalServiceError",
    "FIEError",
    "NotFoundError",
    "OperationTimeoutError",
    "RateLimitError",
    "ValidationError",
]
