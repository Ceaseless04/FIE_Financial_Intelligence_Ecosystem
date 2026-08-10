"""Domain-agnostic assertions used across the Phase 1 test suites."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from fie_common.utils import REDACTED
from fie_schemas.provenance import AssertionKind, Attributed, Provenance


def assert_no_secrets(payload: Any, secrets: Iterable[str]) -> None:
    """Fail if any secret's literal value survives into ``payload``.

    Applied to log records, error bodies, and span attributes — the three
    places a credential most often escapes.
    """
    serialized = json.dumps(payload, default=str)
    leaked = [secret for secret in secrets if secret and secret in serialized]
    assert not leaked, f"secret material leaked into payload: {leaked}"


def assert_redacted(record: Mapping[str, Any], *keys: str) -> None:
    """Assert the given keys were replaced with the redaction marker."""
    for key in keys:
        assert key in record, f"expected key {key!r} to be present"
        assert record[key] == REDACTED, f"expected {key!r} to be redacted, got {record[key]!r}"


def assert_cited(provenance: Provenance) -> None:
    """Assert a value traces back to primary evidence or a computation."""
    assert provenance.is_verifiable, (
        f"provenance of kind {provenance.kind} has neither sources nor a computation"
    )


def assert_not_presented_as_fact(attributed: Attributed[Any]) -> None:
    """Assert model-authored content is labelled as generated, never as fact."""
    assert attributed.provenance.kind is not AssertionKind.FACT or not (
        attributed.provenance.model
    ), "model-authored content must not be labelled as a FACT"


def assert_assumptions_explicit(provenance: Provenance) -> None:
    """Assert an estimate states the assumptions it rests on."""
    assert provenance.kind is AssertionKind.ESTIMATE, f"expected an ESTIMATE, got {provenance.kind}"
    assert provenance.assumptions, "an estimate must state its assumptions"


def assert_deterministic(func: Any, *args: Any, runs: int = 5, **kwargs: Any) -> None:
    """Assert a calculation returns an identical result across repeated calls.

    Financial calculations must be deterministic; this is the guard that keeps
    a model call from being slipped into a computation path later.
    """
    first = func(*args, **kwargs)
    for _ in range(runs - 1):
        assert func(*args, **kwargs) == first, "function produced non-deterministic output"


__all__ = [
    "assert_assumptions_explicit",
    "assert_cited",
    "assert_deterministic",
    "assert_no_secrets",
    "assert_not_presented_as_fact",
    "assert_redacted",
]
