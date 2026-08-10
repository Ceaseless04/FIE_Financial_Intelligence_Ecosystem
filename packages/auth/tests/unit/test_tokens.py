"""Unit tests for JWT issuance and verification."""

from __future__ import annotations

from datetime import timedelta

import jwt
import pytest

from fie_auth.rbac import Principal, Role
from fie_auth.tokens import (
    AuthSettings,
    TokenType,
    authenticate,
    create_token,
    decode_token,
    parse_bearer_header,
)
from fie_common.config import Environment, SecretStr
from fie_common.errors import AuthenticationError, ConfigurationError
from fie_common.utils import utc_now

pytestmark = pytest.mark.unit


@pytest.fixture
def settings() -> AuthSettings:
    return AuthSettings(
        jwt_secret=SecretStr("a-test-signing-secret-of-sufficient-length-1234"),
        issuer="fie-test",
        audience="fie-test-services",
        access_token_ttl_seconds=900,
        refresh_token_ttl_seconds=86_400,
    )


@pytest.fixture
def principal() -> Principal:
    return Principal(
        subject="user_42",
        tenant_id="tenant_a",
        roles=[Role.ANALYST],
        permissions=["atlas:research:write"],
    )


class TestAuthSettingsGuardrails:
    def test_default_secret_is_rejected_in_production(self) -> None:
        # The shipped default is long enough to clear the length check, so the
        # marker rule is what stops it reaching production.
        with pytest.raises(ConfigurationError, match="development marker"):
            AuthSettings().validate_for(Environment.PRODUCTION)

    def test_default_secret_is_fine_in_development(self) -> None:
        AuthSettings().validate_for(Environment.DEVELOPMENT)

    def test_strong_secret_passes_in_production(self) -> None:
        AuthSettings(jwt_secret=SecretStr("x" * 48)).validate_for(Environment.PRODUCTION)

    def test_none_algorithm_is_never_permitted(self) -> None:
        settings = AuthSettings(jwt_secret=SecretStr("x" * 48), jwt_algorithm="none")

        with pytest.raises(ConfigurationError, match="'none' JWT algorithm"):
            settings.validate_for(Environment.PRODUCTION)


class TestRoundTrip:
    def test_access_token_round_trips(self, settings: AuthSettings, principal: Principal) -> None:
        token = create_token(principal, settings)
        claims = decode_token(token, settings)

        assert claims.sub == "user_42"
        assert claims.tenant_id == "tenant_a"
        assert claims.roles == [Role.ANALYST]
        assert claims.permissions == ["atlas:research:write"]
        assert claims.typ is TokenType.ACCESS

    def test_claims_convert_back_to_a_principal(
        self, settings: AuthSettings, principal: Principal
    ) -> None:
        restored = decode_token(create_token(principal, settings), settings).to_principal()

        assert restored.subject == principal.subject
        assert restored.has_permission("atlas:research:write") is True

    def test_authenticate_returns_the_principal_directly(
        self, settings: AuthSettings, principal: Principal
    ) -> None:
        assert authenticate(create_token(principal, settings), settings).subject == "user_42"

    def test_each_token_has_a_unique_jti(
        self, settings: AuthSettings, principal: Principal
    ) -> None:
        first = decode_token(create_token(principal, settings), settings)
        second = decode_token(create_token(principal, settings), settings)

        assert first.jti != second.jti

    def test_service_account_flag_survives(self, settings: AuthSettings) -> None:
        service = Principal(subject="svc_1", roles=[Role.SERVICE], is_service_account=True)
        claims = decode_token(create_token(service, settings), settings)

        assert claims.is_service_account is True


class TestTokenTypeSeparation:
    def test_refresh_token_is_rejected_where_an_access_token_is_required(
        self, settings: AuthSettings, principal: Principal
    ) -> None:
        # Without this check a long-lived refresh token would be accepted as an
        # access token, defeating short access-token lifetimes entirely.
        refresh = create_token(principal, settings, token_type=TokenType.REFRESH)

        with pytest.raises(AuthenticationError, match="expected a access token"):
            decode_token(refresh, settings, expected_type=TokenType.ACCESS)

    def test_refresh_token_is_accepted_where_expected(
        self, settings: AuthSettings, principal: Principal
    ) -> None:
        refresh = create_token(principal, settings, token_type=TokenType.REFRESH)
        claims = decode_token(refresh, settings, expected_type=TokenType.REFRESH)

        assert claims.typ is TokenType.REFRESH

    def test_type_check_can_be_disabled_explicitly(
        self, settings: AuthSettings, principal: Principal
    ) -> None:
        refresh = create_token(principal, settings, token_type=TokenType.REFRESH)

        assert decode_token(refresh, settings, expected_type=None).typ is TokenType.REFRESH

    def test_refresh_tokens_live_longer_than_access_tokens(
        self, settings: AuthSettings, principal: Principal
    ) -> None:
        access = decode_token(create_token(principal, settings), settings)
        refresh = decode_token(
            create_token(principal, settings, token_type=TokenType.REFRESH),
            settings,
            expected_type=TokenType.REFRESH,
        )

        assert refresh.exp > access.exp


class TestVerificationFailures:
    def test_expired_token_is_rejected(self, settings: AuthSettings, principal: Principal) -> None:
        expired = create_token(
            principal, settings, issued_at=utc_now() - timedelta(hours=2), ttl_seconds=60
        )

        with pytest.raises(AuthenticationError, match="expired"):
            decode_token(expired, settings)

    def test_token_signed_with_another_secret_is_rejected(
        self, settings: AuthSettings, principal: Principal
    ) -> None:
        other = AuthSettings(
            jwt_secret=SecretStr("a-different-secret-of-sufficient-length-9876"),
            issuer=settings.issuer,
            audience=settings.audience,
        )
        token = create_token(principal, other)

        with pytest.raises(AuthenticationError, match="invalid"):
            decode_token(token, settings)

    def test_wrong_issuer_is_rejected(self, settings: AuthSettings, principal: Principal) -> None:
        foreign = AuthSettings(
            jwt_secret=settings.jwt_secret, issuer="somebody-else", audience=settings.audience
        )
        token = create_token(principal, foreign)

        with pytest.raises(AuthenticationError, match="issuer is not trusted"):
            decode_token(token, settings)

    def test_wrong_audience_is_rejected(self, settings: AuthSettings, principal: Principal) -> None:
        foreign = AuthSettings(
            jwt_secret=settings.jwt_secret, issuer=settings.issuer, audience="another-service"
        )
        token = create_token(principal, foreign)

        with pytest.raises(AuthenticationError, match="audience is not accepted"):
            decode_token(token, settings)

    def test_garbage_token_is_rejected(self, settings: AuthSettings) -> None:
        with pytest.raises(AuthenticationError, match="invalid"):
            decode_token("not.a.jwt", settings)

    def test_unsigned_token_is_rejected(self, settings: AuthSettings, principal: Principal) -> None:
        # The classic "alg: none" forgery must not be accepted.
        forged = jwt.encode(
            {
                "sub": "attacker",
                "iss": settings.issuer,
                "aud": settings.audience,
                "iat": int(utc_now().timestamp()),
                "exp": int((utc_now() + timedelta(hours=1)).timestamp()),
                "typ": "access",
            },
            key="",
            algorithm="none",
        )

        with pytest.raises(AuthenticationError):
            decode_token(forged, settings)

    def test_token_missing_required_claims_is_rejected(self, settings: AuthSettings) -> None:
        incomplete = jwt.encode(
            {"sub": "user_1"},
            settings.jwt_secret.get_secret_value(),
            algorithm=settings.jwt_algorithm,
        )

        with pytest.raises(AuthenticationError):
            decode_token(incomplete, settings)

    def test_malformed_role_claim_is_rejected(self, settings: AuthSettings) -> None:
        token = jwt.encode(
            {
                "sub": "user_1",
                "iss": settings.issuer,
                "aud": settings.audience,
                "iat": int(utc_now().timestamp()),
                "exp": int((utc_now() + timedelta(hours=1)).timestamp()),
                "typ": "access",
                "roles": ["not-a-real-role"],
            },
            settings.jwt_secret.get_secret_value(),
            algorithm=settings.jwt_algorithm,
        )

        with pytest.raises(AuthenticationError, match="malformed"):
            decode_token(token, settings)

    def test_clock_skew_leeway_is_applied(self, principal: Principal) -> None:
        settings = AuthSettings(
            jwt_secret=SecretStr("a-test-signing-secret-of-sufficient-length-1234"),
            leeway_seconds=120,
        )
        just_expired = create_token(
            principal, settings, issued_at=utc_now() - timedelta(seconds=90), ttl_seconds=30
        )

        assert decode_token(just_expired, settings).sub == "user_42"


class TestBearerHeader:
    def test_extracts_the_token(self) -> None:
        assert parse_bearer_header("Bearer abc.def.ghi") == "abc.def.ghi"

    def test_scheme_is_case_insensitive(self) -> None:
        assert parse_bearer_header("bearer abc.def.ghi") == "abc.def.ghi"

    def test_missing_header_is_rejected(self) -> None:
        with pytest.raises(AuthenticationError, match="missing Authorization header"):
            parse_bearer_header(None)

    @pytest.mark.parametrize("header", ["Basic abc", "Bearer", "Bearer   ", "abc.def.ghi"])
    def test_malformed_headers_are_rejected(self, header: str) -> None:
        with pytest.raises(AuthenticationError):
            parse_bearer_header(header)
