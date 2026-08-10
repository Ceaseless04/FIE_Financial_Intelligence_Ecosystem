"""JWT issuance and verification.

Access and refresh tokens are separated by a ``typ`` claim that is checked on
every decode. Without that check a long-lived refresh token would be accepted
as an access token — a common and serious flaw.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

import jwt
from pydantic import Field
from pydantic_settings import SettingsConfigDict

from fie_auth.rbac import Principal, Role
from fie_common.config import (
    Environment,
    FIEBaseSettings,
    SecretStr,
    require_production_secret,
    secret_value,
)
from fie_common.errors import AuthenticationError, ConfigurationError
from fie_common.utils import new_id, utc_now
from fie_schemas.base import FrozenModel


class TokenType(StrEnum):
    ACCESS = "access"
    REFRESH = "refresh"


class AuthSettings(FIEBaseSettings):
    """Authentication configuration (``FIE_AUTH_*``)."""

    model_config = SettingsConfigDict(
        env_prefix="FIE_AUTH_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    jwt_secret: SecretStr = SecretStr("insecure-development-secret-do-not-use")
    jwt_algorithm: str = "HS256"
    issuer: str = "fie-platform"
    audience: str = "fie-services"
    access_token_ttl_seconds: int = Field(default=900, ge=60)
    refresh_token_ttl_seconds: int = Field(default=1_209_600, ge=300)
    #: Tolerance for clock skew between services when validating exp/nbf.
    leeway_seconds: int = Field(default=10, ge=0, le=300)

    def validate_for(self, environment: Environment) -> None:
        require_production_secret(
            self.jwt_secret, name="FIE_AUTH_JWT_SECRET", environment=environment
        )
        if environment.is_production and self.jwt_algorithm == "none":
            raise ConfigurationError("the 'none' JWT algorithm is never permitted")


class TokenClaims(FrozenModel):
    """Verified JWT claims."""

    sub: str
    iss: str
    aud: str
    exp: datetime
    iat: datetime
    jti: str
    typ: TokenType
    tenant_id: str | None = None
    roles: list[Role] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    is_service_account: bool = False

    def to_principal(self) -> Principal:
        return Principal(
            subject=self.sub,
            tenant_id=self.tenant_id,
            roles=self.roles,
            permissions=self.permissions,
            is_service_account=self.is_service_account,
        )


def create_token(
    principal: Principal,
    settings: AuthSettings,
    *,
    token_type: TokenType = TokenType.ACCESS,
    issued_at: datetime | None = None,
    ttl_seconds: int | None = None,
) -> str:
    """Issue a signed JWT for ``principal``."""
    now = issued_at or utc_now()
    ttl = ttl_seconds or (
        settings.access_token_ttl_seconds
        if token_type is TokenType.ACCESS
        else settings.refresh_token_ttl_seconds
    )
    payload: dict[str, Any] = {
        "sub": principal.subject,
        "iss": settings.issuer,
        "aud": settings.audience,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=ttl)).timestamp()),
        "jti": new_id("jti"),
        "typ": str(token_type),
        "tenant_id": principal.tenant_id,
        "roles": [str(role) for role in principal.roles],
        "permissions": list(principal.permissions),
        "is_service_account": principal.is_service_account,
    }
    secret = secret_value(settings.jwt_secret)
    if not secret:
        raise ConfigurationError("FIE_AUTH_JWT_SECRET is not configured")
    return jwt.encode(payload, secret, algorithm=settings.jwt_algorithm)


def decode_token(
    token: str,
    settings: AuthSettings,
    *,
    expected_type: TokenType | None = TokenType.ACCESS,
) -> TokenClaims:
    """Verify a JWT and return its claims.

    Signature, expiry, issuer, and audience are all verified. ``expected_type``
    additionally pins the token kind; pass ``None`` only where either kind is
    genuinely acceptable.

    Raises:
        AuthenticationError: if the token is invalid, expired, or of the wrong kind.
    """
    secret = secret_value(settings.jwt_secret)
    if not secret:
        raise ConfigurationError("FIE_AUTH_JWT_SECRET is not configured")

    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=[settings.jwt_algorithm],
            issuer=settings.issuer,
            audience=settings.audience,
            leeway=settings.leeway_seconds,
            options={"require": ["exp", "iat", "sub", "iss", "aud"]},
        )
    except jwt.ExpiredSignatureError as error:
        raise AuthenticationError("token has expired", details={"reason": "expired"}) from error
    except jwt.InvalidAudienceError as error:
        raise AuthenticationError(
            "token audience is not accepted by this service",
            details={"reason": "invalid_audience"},
        ) from error
    except jwt.InvalidIssuerError as error:
        raise AuthenticationError(
            "token issuer is not trusted", details={"reason": "invalid_issuer"}
        ) from error
    except jwt.InvalidTokenError as error:
        raise AuthenticationError(
            f"token is invalid: {error}", details={"reason": "invalid_token"}
        ) from error

    try:
        claims = TokenClaims(
            sub=payload["sub"],
            iss=payload["iss"],
            aud=payload["aud"],
            exp=datetime.fromtimestamp(payload["exp"], tz=utc_now().tzinfo),
            iat=datetime.fromtimestamp(payload["iat"], tz=utc_now().tzinfo),
            jti=payload.get("jti", ""),
            typ=TokenType(payload.get("typ", TokenType.ACCESS)),
            tenant_id=payload.get("tenant_id"),
            roles=[Role(role) for role in payload.get("roles", [])],
            permissions=list(payload.get("permissions", [])),
            is_service_account=bool(payload.get("is_service_account", False)),
        )
    except (KeyError, ValueError) as error:
        raise AuthenticationError(
            f"token claims are malformed: {error}", details={"reason": "malformed_claims"}
        ) from error

    if expected_type is not None and claims.typ is not expected_type:
        raise AuthenticationError(
            f"expected a {expected_type} token but received a {claims.typ} token",
            details={"reason": "wrong_token_type", "expected": str(expected_type)},
        )
    return claims


def authenticate(
    token: str, settings: AuthSettings, *, expected_type: TokenType = TokenType.ACCESS
) -> Principal:
    """Verify a token and return the principal it represents."""
    return decode_token(token, settings, expected_type=expected_type).to_principal()


def parse_bearer_header(header: str | None) -> str:
    """Extract the token from an ``Authorization: Bearer <token>`` header.

    Raises:
        AuthenticationError: if the header is missing or malformed.
    """
    if not header:
        raise AuthenticationError(
            "missing Authorization header", details={"reason": "missing_credentials"}
        )
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthenticationError(
            "Authorization header must use the Bearer scheme",
            details={"reason": "invalid_scheme"},
        )
    return token.strip()


__all__ = [
    "AuthSettings",
    "TokenClaims",
    "TokenType",
    "authenticate",
    "create_token",
    "decode_token",
    "parse_bearer_header",
]
