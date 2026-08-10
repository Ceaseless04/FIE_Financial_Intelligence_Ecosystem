"""fie_auth — authentication (JWT) and authorization (RBAC)."""

from fie_auth.passwords import (
    MAX_PASSWORD_BYTES,
    MIN_PASSWORD_LENGTH,
    hash_password,
    needs_rehash,
    validate_password_policy,
    verify_password,
)
from fie_auth.rbac import (
    ANONYMOUS,
    ROLE_PERMISSIONS,
    Principal,
    Product,
    Role,
    permission_matches,
    validate_permission,
)
from fie_auth.tokens import (
    AuthSettings,
    TokenClaims,
    TokenType,
    authenticate,
    create_token,
    decode_token,
    parse_bearer_header,
)

__version__ = "0.1.0"

__all__ = [
    "ANONYMOUS",
    "MAX_PASSWORD_BYTES",
    "MIN_PASSWORD_LENGTH",
    "ROLE_PERMISSIONS",
    "AuthSettings",
    "Principal",
    "Product",
    "Role",
    "TokenClaims",
    "TokenType",
    "__version__",
    "authenticate",
    "create_token",
    "decode_token",
    "hash_password",
    "needs_rehash",
    "parse_bearer_header",
    "permission_matches",
    "validate_password_policy",
    "validate_permission",
    "verify_password",
]
