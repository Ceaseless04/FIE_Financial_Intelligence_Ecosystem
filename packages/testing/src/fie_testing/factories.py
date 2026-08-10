"""Test data factories with sensible defaults and targeted overrides."""

from __future__ import annotations

from typing import Any

from fie_ai.contracts import CompletionRequest, Message, Role
from fie_auth.rbac import Principal
from fie_auth.rbac import Role as AuthRole
from fie_events.schemas import DomainEvent
from fie_schemas.provenance import (
    Provenance,
    SourceReference,
    SourceType,
)


def make_completion_request(
    prompt: str = "What is in this filing?",
    *,
    system: str | None = None,
    **overrides: Any,
) -> CompletionRequest:
    fields: dict[str, Any] = {
        "messages": [Message(role=Role.USER, content=prompt)],
        "system": system,
        "max_tokens": 1024,
    }
    fields.update(overrides)
    return CompletionRequest(**fields)


def make_principal(
    subject: str = "user_test",
    *,
    roles: list[AuthRole] | None = None,
    permissions: list[str] | None = None,
    tenant_id: str | None = "tenant_test",
    **overrides: Any,
) -> Principal:
    fields: dict[str, Any] = {
        "subject": subject,
        "tenant_id": tenant_id,
        "roles": roles if roles is not None else [AuthRole.ANALYST],
        "permissions": permissions or [],
    }
    fields.update(overrides)
    return Principal(**fields)


def make_event(
    event_type: str = "platform.test.occurred",
    *,
    payload: dict[str, Any] | None = None,
    source_app: str = "test",
    **overrides: Any,
) -> DomainEvent:
    fields: dict[str, Any] = {
        "event_type": event_type,
        "source_app": source_app,
        "payload": payload or {"value": 1},
        "correlation_id": "corr_test",
        "tenant_id": "tenant_test",
    }
    fields.update(overrides)
    return DomainEvent(**fields)


def make_source(
    source_id: str = "src_test",
    *,
    source_type: SourceType = SourceType.SEC_FILING,
    **overrides: Any,
) -> SourceReference:
    fields: dict[str, Any] = {
        "source_id": source_id,
        "source_type": source_type,
        "title": "Test Source",
        "excerpt": "Verbatim supporting text.",
    }
    fields.update(overrides)
    return SourceReference(**fields)


def make_fact_provenance(*source_ids: str) -> Provenance:
    """Provenance for a sourced fact, with one reference per id."""
    ids = source_ids or ("src_test",)
    return Provenance.fact(*(make_source(source_id) for source_id in ids))


__all__ = [
    "make_completion_request",
    "make_event",
    "make_fact_provenance",
    "make_principal",
    "make_source",
]
