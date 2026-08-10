"""Password hashing.

bcrypt silently truncates input beyond 72 bytes, which means two different
long passwords sharing a 72-byte prefix would verify against each other. This
module rejects over-length input explicitly instead of inheriting that
behaviour.
"""

from __future__ import annotations

import bcrypt

from fie_common.errors import ValidationError

#: bcrypt's hard input limit. Longer input is rejected, never truncated.
MAX_PASSWORD_BYTES = 72
MIN_PASSWORD_LENGTH = 12
DEFAULT_ROUNDS = 12


def validate_password_policy(password: str) -> None:
    """Enforce length bounds before hashing.

    Raises:
        ValidationError: if the password is too short or exceeds bcrypt's limit.
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValidationError(
            f"password must be at least {MIN_PASSWORD_LENGTH} characters",
            details={"min_length": MIN_PASSWORD_LENGTH},
        )
    encoded_length = len(password.encode("utf-8"))
    if encoded_length > MAX_PASSWORD_BYTES:
        raise ValidationError(
            f"password must not exceed {MAX_PASSWORD_BYTES} bytes when UTF-8 encoded",
            details={"max_bytes": MAX_PASSWORD_BYTES, "actual_bytes": encoded_length},
        )


def hash_password(password: str, *, rounds: int = DEFAULT_ROUNDS) -> str:
    """Hash a password with a per-password random salt."""
    validate_password_policy(password)
    digest = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=rounds))
    return digest.decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Constant-time password verification.

    Returns ``False`` rather than raising on a malformed hash, so a corrupted
    stored value cannot be distinguished from a wrong password by timing or by
    error type.
    """
    if not password or not hashed:
        return False
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def needs_rehash(hashed: str, *, rounds: int = DEFAULT_ROUNDS) -> bool:
    """True when a stored hash uses a weaker cost factor than the current policy."""
    try:
        parts = hashed.split("$")
        return int(parts[2]) < rounds
    except (IndexError, ValueError):
        return True


__all__ = [
    "DEFAULT_ROUNDS",
    "MAX_PASSWORD_BYTES",
    "MIN_PASSWORD_LENGTH",
    "hash_password",
    "needs_rehash",
    "validate_password_policy",
    "verify_password",
]
