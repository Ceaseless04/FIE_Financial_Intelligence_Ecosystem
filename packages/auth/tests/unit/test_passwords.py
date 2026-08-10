"""Unit tests for password hashing and policy."""

from __future__ import annotations

import pytest

from fie_auth.passwords import (
    MAX_PASSWORD_BYTES,
    MIN_PASSWORD_LENGTH,
    hash_password,
    needs_rehash,
    validate_password_policy,
    verify_password,
)
from fie_common.errors import ValidationError

pytestmark = pytest.mark.unit

VALID_PASSWORD = "correct-horse-battery-staple"
# Low cost factor keeps the suite fast; production uses the default of 12.
FAST_ROUNDS = 4


class TestPasswordPolicy:
    def test_accepts_a_compliant_password(self) -> None:
        validate_password_policy(VALID_PASSWORD)

    def test_rejects_a_short_password(self) -> None:
        with pytest.raises(ValidationError, match=f"at least {MIN_PASSWORD_LENGTH}"):
            validate_password_policy("short")

    def test_rejects_a_password_over_the_bcrypt_limit(self) -> None:
        with pytest.raises(ValidationError, match=f"not exceed {MAX_PASSWORD_BYTES} bytes"):
            validate_password_policy("a" * (MAX_PASSWORD_BYTES + 1))

    def test_length_is_measured_in_utf8_bytes_not_characters(self) -> None:
        # Multi-byte characters hit bcrypt's byte limit well before the
        # character count suggests, so the check must be byte-based.
        multibyte = "é" * 40  # 80 bytes

        with pytest.raises(ValidationError):
            validate_password_policy(multibyte)

    def test_accepts_a_password_exactly_at_the_byte_limit(self) -> None:
        validate_password_policy("a" * MAX_PASSWORD_BYTES)


class TestHashAndVerify:
    def test_hash_then_verify_succeeds(self) -> None:
        hashed = hash_password(VALID_PASSWORD, rounds=FAST_ROUNDS)

        assert verify_password(VALID_PASSWORD, hashed) is True

    def test_wrong_password_fails(self) -> None:
        hashed = hash_password(VALID_PASSWORD, rounds=FAST_ROUNDS)

        assert verify_password("wrong-horse-battery-staple", hashed) is False

    def test_hashes_are_salted_per_password(self) -> None:
        first = hash_password(VALID_PASSWORD, rounds=FAST_ROUNDS)
        second = hash_password(VALID_PASSWORD, rounds=FAST_ROUNDS)

        assert first != second
        assert verify_password(VALID_PASSWORD, first)
        assert verify_password(VALID_PASSWORD, second)

    def test_hash_does_not_contain_the_plaintext(self) -> None:
        assert VALID_PASSWORD not in hash_password(VALID_PASSWORD, rounds=FAST_ROUNDS)

    def test_hashing_enforces_the_policy(self) -> None:
        with pytest.raises(ValidationError):
            hash_password("short", rounds=FAST_ROUNDS)

    def test_long_passwords_sharing_a_prefix_do_not_collide(self) -> None:
        # bcrypt truncates at 72 bytes. Without an explicit length check, these
        # two distinct passwords would verify against each other.
        base = "a" * MAX_PASSWORD_BYTES
        hashed = hash_password(base, rounds=FAST_ROUNDS)

        assert verify_password(base + "DIFFERENT", hashed) is False

    def test_verify_rejects_empty_input(self) -> None:
        hashed = hash_password(VALID_PASSWORD, rounds=FAST_ROUNDS)

        assert verify_password("", hashed) is False
        assert verify_password(VALID_PASSWORD, "") is False

    def test_verify_returns_false_for_a_malformed_hash(self) -> None:
        # A corrupted stored value must be indistinguishable from a wrong
        # password — no exception type to probe.
        assert verify_password(VALID_PASSWORD, "not-a-bcrypt-hash") is False


class TestNeedsRehash:
    def test_true_when_the_cost_factor_is_below_policy(self) -> None:
        hashed = hash_password(VALID_PASSWORD, rounds=FAST_ROUNDS)

        assert needs_rehash(hashed, rounds=12) is True

    def test_false_when_the_cost_factor_meets_policy(self) -> None:
        hashed = hash_password(VALID_PASSWORD, rounds=FAST_ROUNDS)

        assert needs_rehash(hashed, rounds=FAST_ROUNDS) is False

    def test_true_for_an_unparseable_hash(self) -> None:
        assert needs_rehash("garbage") is True
