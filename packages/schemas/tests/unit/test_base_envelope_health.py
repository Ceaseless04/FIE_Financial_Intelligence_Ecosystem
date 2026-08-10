"""Unit tests for base models, response envelopes, and health contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fie_common.errors import NotFoundError
from fie_common.errors import ValidationError as FIEValidationError
from fie_schemas.base import (
    FIEModel,
    FrozenModel,
    IdentifiedModel,
    IdentifiedTimestampedModel,
    TimestampedModel,
)
from fie_schemas.envelope import (
    ErrorDetail,
    Page,
    PageInfo,
    PageRequest,
    ResponseEnvelope,
)
from fie_schemas.health import ComponentHealth, HealthReport, HealthStatus

pytestmark = pytest.mark.unit


class Sample(FIEModel):
    name: str
    count: int = 0


class TestFIEModel:
    def test_unknown_fields_are_rejected(self) -> None:
        # A stray field crossing a service boundary is a contract violation;
        # silently dropping it is how integrations rot.
        with pytest.raises(ValidationError):
            Sample(name="x", unexpected="value")  # type: ignore[call-arg]

    def test_assignment_is_validated(self) -> None:
        sample = Sample(name="x")

        with pytest.raises(ValidationError):
            sample.count = "not-an-int"  # type: ignore[assignment]

    def test_strings_are_stripped(self) -> None:
        assert Sample(name="  padded  ").name == "padded"

    def test_to_json_dict_omits_none_and_uses_primitives(self) -> None:
        assert Sample(name="x", count=3).to_json_dict() == {"name": "x", "count": 3}


class TestFrozenModel:
    class Frozen(FrozenModel):
        value: int

    def test_is_immutable(self) -> None:
        frozen = self.Frozen(value=1)

        with pytest.raises(ValidationError):
            frozen.value = 2  # type: ignore[misc]

    def test_is_hashable(self) -> None:
        assert len({self.Frozen(value=1), self.Frozen(value=1)}) == 1


class TestIdentifiedAndTimestamped:
    def test_identifier_is_generated(self) -> None:
        class Thing(IdentifiedModel):
            pass

        assert Thing().id != Thing().id

    def test_timestamps_are_populated(self) -> None:
        class Thing(TimestampedModel):
            pass

        thing = Thing()
        assert thing.created_at is not None
        assert thing.updated_at is not None

    def test_touch_advances_updated_at(self) -> None:
        class Thing(TimestampedModel):
            pass

        thing = Thing()
        original = thing.updated_at
        thing.touch()

        assert thing.updated_at >= original

    def test_combined_mixin_has_identity_and_timestamps(self) -> None:
        class Thing(IdentifiedTimestampedModel):
            pass

        thing = Thing()
        assert thing.id
        assert thing.created_at


class TestErrorDetail:
    def test_from_error_copies_the_error_contract(self) -> None:
        detail = ErrorDetail.from_error(
            NotFoundError("company not found", details={"ticker": "ACME"}),
            correlation_id="corr_1",
        )

        assert detail.code == "not_found"
        assert detail.retryable is False
        assert detail.details["ticker"] == "ACME"
        assert detail.correlation_id == "corr_1"

    def test_retryable_flag_is_carried_through(self) -> None:
        from fie_common.errors import RateLimitError

        assert ErrorDetail.from_error(RateLimitError("slow")).retryable is True


class TestResponseEnvelope:
    def test_ok_wraps_data(self) -> None:
        envelope = ResponseEnvelope[Sample].ok(Sample(name="x"), correlation_id="corr_1")

        assert envelope.success is True
        assert envelope.data is not None
        assert envelope.error is None

    def test_failure_wraps_an_error(self) -> None:
        envelope = ResponseEnvelope[Sample].failure(FIEValidationError("bad"))

        assert envelope.success is False
        assert envelope.error is not None
        assert envelope.error.code == "validation_error"

    def test_success_with_an_error_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not carry an error"):
            ResponseEnvelope[Sample](success=True, error=ErrorDetail(code="x", message="y"))

    def test_failure_without_an_error_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must carry an error"):
            ResponseEnvelope[Sample](success=False)

    def test_failure_accepts_a_prebuilt_detail(self) -> None:
        detail = ErrorDetail(code="custom", message="custom failure")
        envelope = ResponseEnvelope[Sample].failure(detail)

        assert envelope.error is not None
        assert envelope.error.code == "custom"


class TestPagination:
    def test_page_request_bounds_are_enforced(self) -> None:
        with pytest.raises(ValidationError):
            PageRequest(limit=0)
        with pytest.raises(ValidationError):
            PageRequest(limit=501)
        with pytest.raises(ValidationError):
            PageRequest(offset=-1)

    def test_has_more_is_derived_from_total_when_known(self) -> None:
        page = Page[int].of([1, 2, 3], limit=3, offset=0, total=10)

        assert page.page_info.has_more is True

    def test_has_more_is_false_on_the_last_page(self) -> None:
        page = Page[int].of([9, 10], limit=3, offset=8, total=10)

        assert page.page_info.has_more is False

    def test_full_page_without_a_total_assumes_more(self) -> None:
        page = Page[int].of([1, 2, 3], limit=3)

        assert page.page_info.has_more is True

    def test_partial_page_without_a_total_assumes_no_more(self) -> None:
        page = Page[int].of([1, 2], limit=3)

        assert page.page_info.has_more is False

    def test_cursor_is_carried_through(self) -> None:
        page = Page[int].of([1], limit=1, next_cursor="cursor_abc")

        assert page.page_info.next_cursor == "cursor_abc"

    def test_page_info_rejects_a_negative_total(self) -> None:
        with pytest.raises(ValidationError):
            PageInfo(limit=10, total=-1)


class TestHealth:
    def test_degraded_still_serves_traffic(self) -> None:
        assert HealthStatus.DEGRADED.is_serving is True
        assert HealthStatus.HEALTHY.is_serving is True
        assert HealthStatus.UNHEALTHY.is_serving is False

    def test_all_healthy_components_yield_a_healthy_service(self) -> None:
        report = HealthReport.from_components(
            service="atlas",
            version="0.1.0",
            environment="test",
            components=[
                ComponentHealth(name="postgres", status=HealthStatus.HEALTHY),
                ComponentHealth(name="redis", status=HealthStatus.HEALTHY),
            ],
        )

        assert report.status is HealthStatus.HEALTHY
        assert report.is_ready is True

    def test_a_failed_required_component_makes_the_service_unhealthy(self) -> None:
        report = HealthReport.from_components(
            service="atlas",
            version="0.1.0",
            environment="test",
            components=[
                ComponentHealth(name="postgres", status=HealthStatus.UNHEALTHY, required=True)
            ],
        )

        assert report.status is HealthStatus.UNHEALTHY
        assert report.is_ready is False

    def test_a_failed_optional_component_only_degrades_the_service(self) -> None:
        # Graceful degradation: a Neo4j outage must not take Atlas offline if
        # Atlas can still serve filings.
        report = HealthReport.from_components(
            service="atlas",
            version="0.1.0",
            environment="test",
            components=[
                ComponentHealth(name="postgres", status=HealthStatus.HEALTHY, required=True),
                ComponentHealth(name="neo4j", status=HealthStatus.UNHEALTHY, required=False),
            ],
        )

        assert report.status is HealthStatus.DEGRADED
        assert report.is_ready is True

    def test_a_degraded_component_degrades_the_service(self) -> None:
        report = HealthReport.from_components(
            service="atlas",
            version="0.1.0",
            environment="test",
            components=[ComponentHealth(name="redis", status=HealthStatus.DEGRADED)],
        )

        assert report.status is HealthStatus.DEGRADED

    def test_no_components_is_healthy(self) -> None:
        report = HealthReport.from_components(
            service="atlas", version="0.1.0", environment="test", components=[]
        )

        assert report.status is HealthStatus.HEALTHY
        assert report.is_ready is True

    def test_component_requires_a_name(self) -> None:
        with pytest.raises(ValidationError):
            ComponentHealth(name="", status=HealthStatus.HEALTHY)

    def test_negative_latency_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ComponentHealth(name="redis", status=HealthStatus.HEALTHY, latency_ms=-1)
