"""Unit tests for the domain event envelope and the in-memory bus."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from fie_common.errors import ValidationError
from fie_events.bus import (
    InMemoryEventBus,
    Subscription,
    dead_letter_stream,
    stream_for,
)
from fie_events.schemas import DomainEvent

pytestmark = pytest.mark.unit


def event(event_type: str = "marketmind.company.ingested", **overrides: object) -> DomainEvent:
    fields: dict[str, object] = {
        "event_type": event_type,
        "source_app": "marketmind",
        "payload": {"ticker": "ACME"},
        "correlation_id": "corr_1",
        "tenant_id": "tenant_a",
    }
    fields.update(overrides)
    return DomainEvent(**fields)  # type: ignore[arg-type]


class TestDomainEvent:
    def test_identity_and_timestamp_are_generated(self) -> None:
        created = event()

        assert created.event_id.startswith("evt_")
        assert created.occurred_at is not None

    @pytest.mark.parametrize(
        "event_type",
        ["marketmind.company.ingested", "atlas.report.generated", "platform.health.checked"],
    )
    def test_well_formed_event_types_are_accepted(self, event_type: str) -> None:
        assert event(event_type).event_type == event_type

    @pytest.mark.parametrize("event_type", ["nodots", "bad..type", "trailing.", ".leading"])
    def test_malformed_event_types_are_rejected(self, event_type: str) -> None:
        with pytest.raises(PydanticValidationError):
            event(event_type)

    def test_product_is_the_leading_segment(self) -> None:
        assert event("sentinel.risk.detected").product == "sentinel"

    def test_events_are_immutable(self) -> None:
        created = event()

        with pytest.raises(PydanticValidationError):
            created.payload = {}  # type: ignore[misc]

    def test_version_defaults_to_one(self) -> None:
        assert event().version == 1

    def test_version_must_be_positive(self) -> None:
        with pytest.raises(PydanticValidationError):
            event(version=0)


class TestWireFormat:
    def test_round_trips_through_the_wire_format(self) -> None:
        original = event()

        restored = DomainEvent.from_wire(original.to_wire())

        assert restored.event_id == original.event_id
        assert restored.event_type == original.event_type
        assert restored.payload == original.payload
        assert restored.correlation_id == original.correlation_id

    def test_wire_fields_are_all_strings(self) -> None:
        # Redis stream entries are flat string maps.
        assert all(isinstance(value, str) for value in event().to_wire().values())

    def test_absent_optional_fields_become_empty_strings(self) -> None:
        wire = event(correlation_id=None, tenant_id=None).to_wire()

        assert wire["correlation_id"] == ""
        assert wire["tenant_id"] == ""

    def test_empty_strings_decode_back_to_none(self) -> None:
        restored = DomainEvent.from_wire(event(correlation_id=None).to_wire())

        assert restored.correlation_id is None

    def test_a_missing_required_field_is_rejected(self) -> None:
        wire = event().to_wire()
        del wire["event_type"]

        with pytest.raises(ValidationError, match="malformed event"):
            DomainEvent.from_wire(wire)

    def test_an_unparseable_payload_is_rejected(self) -> None:
        wire = event().to_wire()
        wire["payload"] = "{not json"

        with pytest.raises(ValidationError):
            DomainEvent.from_wire(wire)

    def test_a_bad_timestamp_is_rejected(self) -> None:
        wire = event().to_wire()
        wire["occurred_at"] = "not-a-timestamp"

        with pytest.raises(ValidationError):
            DomainEvent.from_wire(wire)

    def test_nested_payloads_survive_the_round_trip(self) -> None:
        original = event(payload={"metrics": {"revenue": [1, 2, 3]}, "flag": True})

        assert DomainEvent.from_wire(original.to_wire()).payload == original.payload


class TestCausationChain:
    def test_a_derived_event_inherits_the_correlation_id(self) -> None:
        origin = event()

        derived = origin.caused(
            "sentinel.risk.detected", {"severity": "high"}, source_app="sentinel"
        )

        assert derived.correlation_id == origin.correlation_id
        assert derived.causation_id == origin.event_id
        assert derived.tenant_id == origin.tenant_id

    def test_a_derived_event_has_its_own_identity(self) -> None:
        origin = event()
        derived = origin.caused("atlas.report.requested", {}, source_app="atlas")

        assert derived.event_id != origin.event_id


class TestStreamNaming:
    def test_streams_are_per_product(self) -> None:
        assert stream_for("marketmind.company.ingested") == "fie.events.marketmind"
        assert stream_for("atlas.report.generated") == "fie.events.atlas"

    def test_all_events_of_a_product_share_one_stream(self) -> None:
        assert stream_for("atlas.report.generated") == stream_for("atlas.filing.parsed")

    def test_dead_letter_stream_is_derived(self) -> None:
        assert dead_letter_stream("fie.events.atlas") == "fie.events.atlas.dlq"


class TestSubscriptionFiltering:
    def test_no_filter_accepts_everything(self) -> None:
        subscription = Subscription(
            stream="s",
            group="g",
            consumer="c",
            handler=_noop,  # type: ignore[arg-type]
        )

        assert subscription.accepts(event("atlas.report.generated")) is True

    def test_prefix_filters_are_applied(self) -> None:
        subscription = Subscription(
            stream="s",
            group="g",
            consumer="c",
            handler=_noop,  # type: ignore[arg-type]
            event_types=("atlas.report",),
        )

        assert subscription.accepts(event("atlas.report.generated")) is True
        assert subscription.accepts(event("atlas.filing.parsed")) is False


async def _noop(_event: DomainEvent) -> None:
    return None


class TestInMemoryEventBus:
    async def test_publish_records_the_event(self) -> None:
        bus = InMemoryEventBus()

        message_id = await bus.publish(event())

        assert message_id
        assert len(bus.published) == 1

    async def test_a_subscriber_receives_matching_events(self) -> None:
        bus = InMemoryEventBus()
        received: list[DomainEvent] = []

        async def handler(received_event: DomainEvent) -> None:
            received.append(received_event)

        await bus.subscribe(
            Subscription(
                stream="fie.events.marketmind",
                group="g",
                consumer="c",
                handler=handler,
            )
        )
        await bus.publish(event())

        assert len(received) == 1

    async def test_events_on_other_streams_are_not_delivered(self) -> None:
        bus = InMemoryEventBus()
        received: list[DomainEvent] = []

        async def handler(received_event: DomainEvent) -> None:
            received.append(received_event)

        await bus.subscribe(
            Subscription(stream="fie.events.atlas", group="g", consumer="c", handler=handler)
        )
        await bus.publish(event("marketmind.company.ingested"))

        assert received == []

    async def test_event_type_filters_are_honored(self) -> None:
        bus = InMemoryEventBus()
        received: list[DomainEvent] = []

        async def handler(received_event: DomainEvent) -> None:
            received.append(received_event)

        await bus.subscribe(
            Subscription(
                stream="fie.events.atlas",
                group="g",
                consumer="c",
                handler=handler,
                event_types=("atlas.report",),
            )
        )
        await bus.publish(event("atlas.report.generated"))
        await bus.publish(event("atlas.filing.parsed"))

        assert len(received) == 1

    async def test_a_failing_handler_does_not_break_the_publisher(self) -> None:
        bus = InMemoryEventBus()

        async def failing(_event: DomainEvent) -> None:
            raise RuntimeError("handler exploded")

        await bus.subscribe(
            Subscription(stream="fie.events.marketmind", group="g", consumer="c", handler=failing)
        )

        await bus.publish(event())

        assert len(bus.dead_lettered) == 1

    async def test_one_failing_handler_does_not_starve_the_others(self) -> None:
        bus = InMemoryEventBus()
        received: list[str] = []

        async def failing(_event: DomainEvent) -> None:
            raise RuntimeError("boom")

        async def working(received_event: DomainEvent) -> None:
            received.append(received_event.event_id)

        for handler in (failing, working):
            await bus.subscribe(
                Subscription(
                    stream="fie.events.marketmind",
                    group=f"g-{handler.__name__}",
                    consumer="c",
                    handler=handler,
                )
            )

        await bus.publish(event())

        assert len(received) == 1

    async def test_test_helpers_expose_published_events(self) -> None:
        bus = InMemoryEventBus()
        await bus.publish(event("atlas.report.generated"))
        await bus.publish(event("atlas.filing.parsed"))

        assert len(bus.events_of_type("atlas.report")) == 1
        assert bus.counts_by_type["atlas.filing.parsed"] == 1

    async def test_clear_resets_recorded_state(self) -> None:
        bus = InMemoryEventBus()
        await bus.publish(event())
        bus.clear()

        assert bus.published == []

    async def test_aclose_removes_subscriptions(self) -> None:
        bus = InMemoryEventBus()
        await bus.subscribe(
            Subscription(stream="s", group="g", consumer="c", handler=_noop)  # type: ignore[arg-type]
        )
        await bus.aclose()
        await bus.publish(event())

        assert len(bus.published) == 1
