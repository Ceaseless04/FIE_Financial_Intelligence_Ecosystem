"""Unit tests for configuration and production guardrails."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from fie_common.config import (
    CoreSettings,
    Environment,
    LogLevel,
    SecretStr,
    require_production_secret,
    secret_value,
)
from fie_common.errors import ConfigurationError

pytestmark = pytest.mark.unit


class TestEnvironment:
    def test_only_production_is_production(self) -> None:
        assert Environment.PRODUCTION.is_production is True
        assert Environment.DEVELOPMENT.is_production is False
        assert Environment.TEST.is_production is False

    def test_non_production_environments_tolerate_weak_secrets(self) -> None:
        assert Environment.DEVELOPMENT.allows_weak_secrets is True
        assert Environment.TEST.allows_weak_secrets is True
        assert Environment.PRODUCTION.allows_weak_secrets is False


class TestCoreSettings:
    def test_defaults_are_development_safe(self) -> None:
        settings = CoreSettings()

        assert settings.environment is Environment.DEVELOPMENT
        assert settings.log_level is LogLevel.INFO
        assert settings.debug is False

    def test_reads_from_environment_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIE_SERVICE_NAME", "atlas-api")
        monkeypatch.setenv("FIE_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("FIE_ENVIRONMENT", "production")

        settings = CoreSettings()

        assert settings.service_name == "atlas-api"
        assert settings.log_level is LogLevel.DEBUG
        assert settings.environment is Environment.PRODUCTION

    def test_debug_is_rejected_in_production(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIE_ENVIRONMENT", "production")
        monkeypatch.setenv("FIE_DEBUG", "true")

        with pytest.raises(PydanticValidationError, match="debug must be disabled in production"):
            CoreSettings()

    def test_debug_is_allowed_outside_production(self) -> None:
        assert CoreSettings(environment=Environment.DEVELOPMENT, debug=True).debug is True

    def test_settings_are_immutable(self) -> None:
        settings = CoreSettings()

        with pytest.raises(PydanticValidationError):
            settings.service_name = "mutated"  # type: ignore[misc]

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_service_name_is_rejected(self, blank: str) -> None:
        with pytest.raises(PydanticValidationError):
            CoreSettings(service_name=blank)

    @pytest.mark.parametrize("timeout", [0.0, -1.0])
    def test_non_positive_timeout_is_rejected(self, timeout: float) -> None:
        with pytest.raises(PydanticValidationError):
            CoreSettings(request_timeout_seconds=timeout)

    def test_retry_attempts_must_be_at_least_one(self) -> None:
        with pytest.raises(PydanticValidationError):
            CoreSettings(max_retry_attempts=0)

    def test_unknown_environment_keys_are_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Several settings classes share one .env file, so unrelated keys must
        # not break construction.
        monkeypatch.setenv("FIE_SOMETHING_UNRELATED", "value")

        assert CoreSettings().service_name == "fie-service"


class TestSecretValue:
    def test_unwraps_secret_str(self) -> None:
        assert secret_value(SecretStr("s3cret")) == "s3cret"

    def test_passes_through_plain_strings(self) -> None:
        assert secret_value("plain") == "plain"

    def test_none_stays_none(self) -> None:
        assert secret_value(None) is None


class TestProductionSecretGuardrail:
    @pytest.mark.parametrize("environment", [Environment.DEVELOPMENT, Environment.TEST])
    def test_weak_secrets_are_permitted_outside_production(self, environment: Environment) -> None:
        require_production_secret("change-me", name="TEST_SECRET", environment=environment)

    def test_missing_secret_is_rejected_in_production(self) -> None:
        with pytest.raises(ConfigurationError, match="must be set in production"):
            require_production_secret(None, name="TEST_SECRET", environment=Environment.PRODUCTION)

    @pytest.mark.parametrize(
        "placeholder", ["change-me", "CHANGEME", "password", "postgres", "your-api-key", ""]
    )
    def test_known_placeholders_are_rejected_in_production(self, placeholder: str) -> None:
        with pytest.raises(ConfigurationError):
            require_production_secret(
                placeholder, name="TEST_SECRET", environment=Environment.PRODUCTION
            )

    @pytest.mark.parametrize(
        "secret",
        [
            "insecure-development-secret-do-not-use",
            "fie-local-development-password-value",
            "my-example-secret-that-is-long-enough-x",
            "your-production-key-goes-here-really",
        ],
    )
    def test_development_markers_are_rejected_even_when_long_enough(self, secret: str) -> None:
        # Length alone is not sufficient: the platform's own defaults clear the
        # length check while remaining publicly known strings.
        assert len(secret) >= 32

        with pytest.raises(ConfigurationError, match="development marker"):
            require_production_secret(
                secret, name="TEST_SECRET", environment=Environment.PRODUCTION
            )

    def test_marker_detection_is_reported_in_details(self) -> None:
        with pytest.raises(ConfigurationError) as exc_info:
            require_production_secret(
                "insecure-development-secret-do-not-use",
                name="FIE_AUTH_JWT_SECRET",
                environment=Environment.PRODUCTION,
            )

        assert exc_info.value.details["marker"] == "insecure"

    def test_short_secrets_are_rejected_in_production(self) -> None:
        with pytest.raises(ConfigurationError, match="at least 32 characters"):
            require_production_secret(
                "aB3xQ9zL7kW2",  # no development markers, simply too short
                name="TEST_SECRET",
                environment=Environment.PRODUCTION,
            )

    def test_strong_secret_passes_in_production(self) -> None:
        require_production_secret(
            SecretStr("k" * 48), name="TEST_SECRET", environment=Environment.PRODUCTION
        )

    def test_min_length_is_configurable_per_setting(self) -> None:
        require_production_secret(
            "sk-ant-api03-abcdefghij",
            name="TEST_SECRET",
            environment=Environment.PRODUCTION,
            min_length=20,
        )

    def test_error_details_name_the_offending_setting(self) -> None:
        with pytest.raises(ConfigurationError) as exc_info:
            require_production_secret(
                "", name="FIE_AUTH_JWT_SECRET", environment=Environment.PRODUCTION
            )

        assert exc_info.value.details["setting"] == "FIE_AUTH_JWT_SECRET"
