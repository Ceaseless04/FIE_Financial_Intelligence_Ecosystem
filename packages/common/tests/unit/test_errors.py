"""Unit tests for the shared error hierarchy."""

from __future__ import annotations

import pytest

from fie_common.errors import (
    AIProviderError,
    AIProviderRateLimitError,
    AIProviderRefusalError,
    AIProviderTimeoutError,
    AuthenticationError,
    AuthorizationError,
    CircuitOpenError,
    ConfigurationError,
    ConflictError,
    DatabaseError,
    EventBusError,
    ExternalServiceError,
    FIEError,
    NotFoundError,
    OperationTimeoutError,
    RateLimitError,
    ValidationError,
)

pytestmark = pytest.mark.unit


class TestErrorContract:
    def test_base_error_carries_code_message_and_details(self) -> None:
        error = FIEError("boom", details={"key": "value"})

        assert error.message == "boom"
        assert error.code == "fie_error"
        assert error.details == {"key": "value"}
        assert str(error) == "boom"

    def test_to_dict_is_the_api_wire_shape(self) -> None:
        error = ValidationError("bad input", details={"field": "ticker"})

        assert error.to_dict() == {
            "code": "validation_error",
            "message": "bad input",
            "retryable": False,
            "details": {"field": "ticker"},
        }

    def test_cause_is_chained_for_traceback_preservation(self) -> None:
        original = ValueError("root cause")
        error = DatabaseError("query failed", cause=original)

        assert error.__cause__ is original

    def test_repr_includes_code_and_message(self) -> None:
        assert "not_found" in repr(NotFoundError("missing"))

    def test_details_default_to_an_empty_dict_not_none(self) -> None:
        assert FIEError("x").details == {}


class TestErrorClassification:
    @pytest.mark.parametrize(
        ("error_class", "expected_status"),
        [
            (ValidationError, 422),
            (AuthenticationError, 401),
            (AuthorizationError, 403),
            (NotFoundError, 404),
            (ConflictError, 409),
            (RateLimitError, 429),
            (ConfigurationError, 500),
            (ExternalServiceError, 502),
            (CircuitOpenError, 503),
            (OperationTimeoutError, 504),
        ],
    )
    def test_http_status_mapping(self, error_class: type[FIEError], expected_status: int) -> None:
        assert error_class("message").http_status == expected_status

    @pytest.mark.parametrize(
        "error_class",
        [RateLimitError, OperationTimeoutError, ExternalServiceError, CircuitOpenError],
    )
    def test_transient_failures_are_retryable(self, error_class: type[FIEError]) -> None:
        assert error_class("message").retryable is True

    @pytest.mark.parametrize(
        "error_class",
        [ValidationError, AuthenticationError, AuthorizationError, NotFoundError, ConflictError],
    )
    def test_caller_errors_are_not_retryable(self, error_class: type[FIEError]) -> None:
        assert error_class("message").retryable is False

    def test_refusal_is_not_retryable_despite_being_an_ai_provider_error(self) -> None:
        # A refusal is deterministic: the same request produces the same
        # refusal, so retrying wastes budget. Routing must switch providers.
        error = AIProviderRefusalError("declined")

        assert isinstance(error, AIProviderError)
        assert error.retryable is False

    def test_ai_provider_errors_inherit_external_service_semantics(self) -> None:
        assert issubclass(AIProviderError, ExternalServiceError)
        assert AIProviderTimeoutError("t").retryable is True
        assert AIProviderRateLimitError("r").retryable is True

    def test_database_and_event_bus_errors_are_external_and_retryable(self) -> None:
        assert issubclass(DatabaseError, ExternalServiceError)
        assert issubclass(EventBusError, ExternalServiceError)
        assert DatabaseError("d").retryable is True


class TestRateLimitError:
    def test_retry_after_is_exposed_and_recorded_in_details(self) -> None:
        error = RateLimitError("slow down", retry_after_seconds=2.5)

        assert error.retry_after_seconds == 2.5
        assert error.details["retry_after_seconds"] == 2.5

    def test_retry_after_is_optional(self) -> None:
        error = RateLimitError("slow down")

        assert error.retry_after_seconds is None
        assert "retry_after_seconds" not in error.details

    def test_explicit_details_are_preserved_alongside_retry_after(self) -> None:
        error = RateLimitError("slow", retry_after_seconds=1.0, details={"provider": "claude"})

        assert error.details["provider"] == "claude"
        assert error.details["retry_after_seconds"] == 1.0
