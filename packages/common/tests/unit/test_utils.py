"""Unit tests for shared utilities."""

from __future__ import annotations

from datetime import UTC

import pytest

from fie_common.utils import (
    REDACTED,
    chunked,
    coalesce,
    content_hash,
    new_id,
    redact_mapping,
    redact_secret,
    stable_json,
    utc_now,
)

pytestmark = pytest.mark.unit


class TestUtcNow:
    def test_is_timezone_aware_and_utc(self) -> None:
        now = utc_now()

        assert now.tzinfo is not None
        assert now.utcoffset() == UTC.utcoffset(None)


class TestNewId:
    def test_ids_are_unique(self) -> None:
        assert len({new_id() for _ in range(1000)}) == 1000

    def test_prefix_is_applied(self) -> None:
        assert new_id("evt").startswith("evt_")

    def test_empty_prefix_produces_a_bare_id(self) -> None:
        assert "_" not in new_id()


class TestRedactSecret:
    def test_keeps_a_suffix_for_correlation(self) -> None:
        assert redact_secret("sk-ant-api03-abcdef1234") == f"{REDACTED}1234"

    def test_short_values_are_fully_redacted(self) -> None:
        assert redact_secret("abc") == REDACTED

    def test_empty_and_none_are_fully_redacted(self) -> None:
        assert redact_secret("") == REDACTED
        assert redact_secret(None) == REDACTED

    def test_keep_zero_redacts_entirely(self) -> None:
        assert redact_secret("longsecretvalue", keep=0) == REDACTED


class TestRedactMapping:
    def test_redacts_known_sensitive_keys(self) -> None:
        result = redact_mapping({"user": "alice", "password": "hunter2"})

        assert result == {"user": "alice", "password": REDACTED}

    def test_is_case_insensitive(self) -> None:
        result = redact_mapping({"API_KEY": "secret", "Authorization": "Bearer x"})

        assert result["API_KEY"] == REDACTED
        assert result["Authorization"] == REDACTED

    def test_redacts_nested_structures(self) -> None:
        result = redact_mapping(
            {"config": {"database": {"database_url": "postgres://u:p@h/db", "host": "h"}}}
        )

        assert result["config"]["database"]["database_url"] == REDACTED
        assert result["config"]["database"]["host"] == "h"

    def test_redacts_inside_lists(self) -> None:
        result = redact_mapping({"items": [{"token": "abc"}, {"name": "ok"}]})

        assert result["items"][0]["token"] == REDACTED
        assert result["items"][1]["name"] == "ok"

    def test_depth_limit_prevents_runaway_recursion(self) -> None:
        deep: dict[str, object] = {"password": "leaked"}
        for _ in range(10):
            deep = {"nested": deep}

        # No exception; the guard simply stops descending.
        redact_mapping(deep, max_depth=3)

    def test_custom_sensitive_keys_are_supported(self) -> None:
        result = redact_mapping({"ssn": "123"}, sensitive_keys=["ssn"])

        assert result["ssn"] == REDACTED

    def test_input_mapping_is_not_mutated(self) -> None:
        original = {"password": "hunter2"}
        redact_mapping(original)

        assert original["password"] == "hunter2"


class TestStableJson:
    def test_key_order_does_not_affect_output(self) -> None:
        # Unstable serialization silently destroys prompt-cache hit rates.
        assert stable_json({"b": 1, "a": 2}) == stable_json({"a": 2, "b": 1})

    def test_output_is_compact(self) -> None:
        assert stable_json({"a": 1, "b": 2}) == '{"a":1,"b":2}'

    def test_non_serializable_values_fall_back_to_str(self) -> None:
        assert "2024" in stable_json({"when": utc_now().replace(year=2024)})


class TestContentHash:
    def test_is_stable_across_key_orderings(self) -> None:
        assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})

    def test_differs_for_different_content(self) -> None:
        assert content_hash({"a": 1}) != content_hash({"a": 2})

    def test_is_a_sha256_hex_digest(self) -> None:
        digest = content_hash({"a": 1})

        assert len(digest) == 64
        assert all(char in "0123456789abcdef" for char in digest)


class TestChunked:
    def test_splits_into_even_batches(self) -> None:
        assert list(chunked([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]

    def test_final_batch_may_be_short(self) -> None:
        assert list(chunked([1, 2, 3], 2)) == [[1, 2], [3]]

    def test_empty_input_yields_nothing(self) -> None:
        assert list(chunked([], 3)) == []

    def test_size_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            list(chunked([1, 2], 0))


class TestCoalesce:
    def test_returns_the_first_non_none_value(self) -> None:
        assert coalesce(None, None, "third", "fourth") == "third"

    def test_all_none_returns_none(self) -> None:
        assert coalesce(None, None) is None

    def test_falsy_but_non_none_values_are_returned(self) -> None:
        assert coalesce(None, 0) == 0
