"""Configuration primitives.

Every setting resolves from environment variables or a secrets manager — never
from a literal in source. Concrete settings classes live beside the package
they configure (``fie_database.DatabaseSettings``, ``fie_ai.AISettings``); this
module supplies the shared base class, the environment enum, and the
production guardrails those classes are validated against.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from fie_common.errors import ConfigurationError

# Values that are fine in a dev compose file and must never reach production.
_PLACEHOLDER_SECRETS: frozenset[str] = frozenset(
    {
        "",
        "change-me",
        "changeme",
        "placeholder",
        "secret",
        "password",
        "postgres",
        "test",
        "dev",
        "development",
        "localdev",
        "insecure",
        "dummy",
        "example",
        "your-api-key",
        "your-secret-key",
    }
)

# Substrings that mark a value as a development default regardless of its
# length. Exact-match alone is not enough: the platform's own defaults are long
# enough to clear the length check while still being publicly known strings.
_INSECURE_MARKERS: tuple[str, ...] = (
    "insecure",
    "change-me",
    "changeme",
    "placeholder",
    "do-not-use",
    "donotuse",
    "example",
    "dummy",
    "your-",
    "local-development",
    "localdevelopment",
    "development-secret",
    "test-secret",
    "sample",
)

_MIN_PRODUCTION_SECRET_LENGTH = 32


class Environment(StrEnum):
    """Deployment environment.

    Drives the production guardrails: development and test tolerate weak
    defaults so the stack boots with no setup, production does not.
    """

    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"

    @property
    def is_production(self) -> bool:
        return self is Environment.PRODUCTION

    @property
    def allows_weak_secrets(self) -> bool:
        return self is not Environment.PRODUCTION


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class FIEBaseSettings(BaseSettings):
    """Base for all settings classes.

    ``extra="ignore"`` lets several settings classes share one ``.env`` file
    without each having to declare the others' keys. ``frozen=True`` makes a
    loaded configuration immutable for the process lifetime.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        validate_default=True,
    )


class CoreSettings(FIEBaseSettings):
    """Settings every FIE service shares, read from ``FIE_``-prefixed vars."""

    model_config = SettingsConfigDict(
        env_prefix="FIE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        validate_default=True,
    )

    environment: Environment = Environment.DEVELOPMENT
    service_name: str = "fie-service"
    service_version: str = "0.1.0"
    log_level: LogLevel = LogLevel.INFO
    debug: bool = False
    json_logs: bool = True

    # Resilience defaults; individual clients may override.
    request_timeout_seconds: float = 30.0
    max_retry_attempts: int = 3

    @field_validator("service_name")
    @classmethod
    def _service_name_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("service_name must not be blank")
        return value

    @field_validator("request_timeout_seconds")
    @classmethod
    def _timeout_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        return value

    @field_validator("max_retry_attempts")
    @classmethod
    def _attempts_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_retry_attempts must be at least 1")
        return value

    def model_post_init(self, __context: Any) -> None:
        if self.environment.is_production and self.debug:
            raise ValueError("debug must be disabled in production")


def secret_value(secret: SecretStr | str | None) -> str | None:
    """Unwrap a ``SecretStr`` (or plain string) to its underlying value."""
    if secret is None:
        return None
    if isinstance(secret, SecretStr):
        return secret.get_secret_value()
    return secret


def require_production_secret(
    secret: SecretStr | str | None,
    *,
    name: str,
    environment: Environment,
    min_length: int = _MIN_PRODUCTION_SECRET_LENGTH,
) -> None:
    """Reject placeholder or weak secrets when running in production.

    Called from settings validators so a misconfigured deployment fails at
    startup rather than at the first authenticated request. Outside production
    this is a no-op, which keeps the dev compose stack zero-setup.

    Raises:
        ConfigurationError: if ``environment`` is production and the secret is
            missing, a known placeholder, or shorter than ``min_length``.
    """
    if not environment.is_production:
        return

    raw = secret_value(secret)
    if raw is None or not raw.strip():
        raise ConfigurationError(
            f"{name} must be set in production",
            details={"setting": name, "environment": str(environment)},
        )
    normalized = raw.strip().lower()
    if normalized in _PLACEHOLDER_SECRETS:
        raise ConfigurationError(
            f"{name} is set to a known placeholder value and cannot be used in production",
            details={"setting": name, "environment": str(environment)},
        )
    matched_marker = next((marker for marker in _INSECURE_MARKERS if marker in normalized), None)
    if matched_marker is not None:
        raise ConfigurationError(
            f"{name} contains the development marker {matched_marker!r} "
            "and cannot be used in production",
            details={
                "setting": name,
                "environment": str(environment),
                "marker": matched_marker,
            },
        )
    if len(raw) < min_length:
        raise ConfigurationError(
            f"{name} must be at least {min_length} characters in production",
            details={"setting": name, "min_length": min_length, "actual_length": len(raw)},
        )


__all__ = [
    "CoreSettings",
    "Environment",
    "FIEBaseSettings",
    "LogLevel",
    "SecretStr",
    "require_production_secret",
    "secret_value",
]
