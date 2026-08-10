"""Unit tests for ambient request context propagation."""

from __future__ import annotations

import asyncio

import pytest

from fie_observability.context import (
    RequestContext,
    bind_context,
    current_context,
    request_context,
    reset_context,
    set_context,
)

pytestmark = pytest.mark.unit


class TestRequestContext:
    def test_to_dict_omits_unset_fields(self) -> None:
        context = RequestContext(correlation_id="corr_1", tenant_id="tenant_a")

        assert context.to_dict() == {"correlation_id": "corr_1", "tenant_id": "tenant_a"}

    def test_to_dict_includes_every_set_field(self) -> None:
        context = RequestContext(
            correlation_id="corr_1",
            request_id="req_1",
            tenant_id="t",
            principal_id="p",
            service_name="atlas",
            source_app="atlas",
        )

        assert set(context.to_dict()) == {
            "correlation_id",
            "request_id",
            "tenant_id",
            "principal_id",
            "service_name",
            "source_app",
        }

    def test_is_immutable(self) -> None:
        context = RequestContext(correlation_id="corr_1")

        with pytest.raises(AttributeError):
            context.correlation_id = "changed"  # type: ignore[misc]


class TestBinding:
    def test_default_context_is_empty(self) -> None:
        assert current_context().correlation_id == ""

    def test_bind_generates_a_correlation_id_when_absent(self) -> None:
        token = bind_context(tenant_id="tenant_a")
        try:
            context = current_context()

            assert context.correlation_id.startswith("corr_")
            assert context.tenant_id == "tenant_a"
        finally:
            reset_context(token)

    def test_bind_preserves_an_existing_correlation_id(self) -> None:
        outer = bind_context(correlation_id="corr_fixed")
        try:
            inner = bind_context(tenant_id="tenant_a")
            try:
                assert current_context().correlation_id == "corr_fixed"
            finally:
                reset_context(inner)
        finally:
            reset_context(outer)

    def test_reset_restores_the_previous_context(self) -> None:
        token = bind_context(correlation_id="corr_1", tenant_id="tenant_a")
        reset_context(token)

        assert current_context().correlation_id == ""

    def test_set_context_replaces_wholesale(self) -> None:
        token = set_context(RequestContext(correlation_id="corr_x", request_id="req_x"))
        try:
            assert current_context().request_id == "req_x"
        finally:
            reset_context(token)


class TestRequestContextManager:
    def test_scopes_fields_to_the_block(self) -> None:
        with request_context(correlation_id="corr_1", service_name="atlas") as context:
            assert context.correlation_id == "corr_1"
            assert current_context().service_name == "atlas"

        assert current_context().correlation_id == ""

    def test_restores_context_even_when_the_block_raises(self) -> None:
        with pytest.raises(RuntimeError):
            with request_context(correlation_id="corr_1"):
                raise RuntimeError("boom")

        assert current_context().correlation_id == ""

    def test_nested_blocks_merge_then_unwind(self) -> None:
        with request_context(correlation_id="corr_1", tenant_id="tenant_a"):
            with request_context(principal_id="user_1"):
                context = current_context()

                assert context.correlation_id == "corr_1"
                assert context.tenant_id == "tenant_a"
                assert context.principal_id == "user_1"

            assert current_context().principal_id is None


class TestConcurrencyIsolation:
    async def test_context_does_not_leak_between_concurrent_tasks(self) -> None:
        observed: dict[str, str] = {}

        async def worker(name: str) -> None:
            with request_context(correlation_id=f"corr_{name}", tenant_id=name):
                await asyncio.sleep(0.01)
                observed[name] = current_context().correlation_id

        await asyncio.gather(worker("a"), worker("b"), worker("c"))

        assert observed == {"a": "corr_a", "b": "corr_b", "c": "corr_c"}
