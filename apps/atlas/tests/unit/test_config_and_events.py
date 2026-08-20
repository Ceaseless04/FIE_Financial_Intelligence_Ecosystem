"""Configuration guardrails and the event contract.

Configuration is checked at startup rather than at the first valuation, and the
event types are asserted here because five other products will subscribe to them
— a rename is a breaking change to somebody else's build.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from atlas.config import AtlasSettings
from atlas.domain.periods import FiscalPeriod, PeriodKind, period_key
from atlas.events import (
    PUBLISHED_EVENT_TYPES,
    REPORT_PUBLISHED,
    REPORT_WITHHELD,
    ReportPayload,
    StatementsRejectedPayload,
    ValuationProducedPayload,
    build_event,
)
from fie_common.config import Environment
from fie_common.errors import ConfigurationError
from fie_schemas.envelope import ErrorDetail, json_safe

pytestmark = pytest.mark.unit


class TestSettings:
    def test_defaults_are_development_safe(self) -> None:
        settings = AtlasSettings()
        assert settings.api_host == "127.0.0.1"
        settings.validate_for(Environment.DEVELOPMENT)

    def test_a_discount_rate_that_does_not_clear_growth_is_refused(self) -> None:
        """A terminal value from a narrow spread diverges toward infinity."""
        settings = AtlasSettings(
            default_discount_rate=Decimal("0.03"), default_terminal_growth=Decimal("0.028")
        )
        with pytest.raises(ConfigurationError, match="at least"):
            settings.validate_for(Environment.DEVELOPMENT)

    def test_localhost_marketmind_is_refused_in_production(self) -> None:
        settings = AtlasSettings(marketmind_token="service-token")
        with pytest.raises(ConfigurationError, match="localhost"):
            settings.validate_for(Environment.PRODUCTION)

    def test_a_missing_service_token_is_refused_in_production(self) -> None:
        """MarketMind rejects unauthenticated callers, so every lookup would
        fail — silently, because context failures degrade rather than raise."""
        settings = AtlasSettings(marketmind_base_url="http://marketmind.internal:8001")
        with pytest.raises(ConfigurationError, match="ATLAS_MARKETMIND_TOKEN"):
            settings.validate_for(Environment.PRODUCTION)

    def test_disabling_marketmind_removes_its_requirements(self) -> None:
        settings = AtlasSettings(marketmind_enabled=False)
        settings.validate_for(Environment.PRODUCTION)


class TestPeriodKey:
    def test_an_annual_period_carries_an_explicit_zero(self) -> None:
        """NULL would not conflict with NULL in a unique constraint."""
        assert FiscalPeriod.annual(2025, date(2025, 12, 31)).key == "annual:2025:0"

    def test_a_quarter_is_distinct_from_its_year(self) -> None:
        annual = FiscalPeriod.annual(2025, date(2025, 12, 31))
        quarter = FiscalPeriod.quarterly(2025, 4, date(2025, 12, 31))
        assert annual.key != quarter.key

    def test_lookup_by_parts_matches_lookup_by_object(self) -> None:
        """One definition, so a query and a write cannot disagree about a row."""
        period = FiscalPeriod.quarterly(2025, 3, date(2025, 9, 30))
        assert period_key(PeriodKind.QUARTERLY, 2025, 3) == period.key


class TestEventContract:
    def test_every_published_type_is_namespaced(self) -> None:
        assert all(event_type.startswith("atlas.") for event_type in PUBLISHED_EVENT_TYPES)

    def test_types_are_unique(self) -> None:
        assert len(set(PUBLISHED_EVENT_TYPES)) == len(PUBLISHED_EVENT_TYPES)

    def test_publication_and_withholding_are_different_types(self) -> None:
        """A consumer must opt into the ungrounded case rather than miss a flag."""
        assert REPORT_PUBLISHED != REPORT_WITHHELD
        assert REPORT_PUBLISHED in PUBLISHED_EVENT_TYPES
        assert REPORT_WITHHELD in PUBLISHED_EVENT_TYPES

    def test_a_payload_serializes_to_primitives(self) -> None:
        event = build_event(
            REPORT_WITHHELD,
            ReportPayload(
                report_id="rpt-1",
                entity_id="ent-acme",
                period_label="FY2025",
                numerically_grounded=False,
                citations_grounded=True,
                unsupported_figures=["31.2%"],
            ),
        )

        assert event.source_app == "atlas"
        assert event.payload["unsupported_figures"] == ["31.2%"]
        assert event.payload["numerically_grounded"] is False

    def test_a_valuation_payload_carries_figures_as_strings(self) -> None:
        """A JSON number is a double; an event is not a place to lose precision."""
        payload = ValuationProducedPayload(
            entity_id="ent-acme",
            period_label="FY2025",
            enterprise_value="1613590000",
            currency="USD",
            assumptions={"discount_rate": "0.10"},
        )
        event = build_event("atlas.valuation.produced", payload)

        assert isinstance(event.payload["enterprise_value"], str)

    def test_a_rejection_says_whether_figures_contradicted_each_other(self) -> None:
        """ "Nothing was found" and "the numbers disagreed" are different facts."""
        payload = StatementsRejectedPayload(
            filing_id="fil-1",
            entity_id="ent-acme",
            period_label="FY2025",
            reasons=["balance sheet does not balance"],
            failed_validation=True,
        )
        assert payload.failed_validation


class TestErrorDetailSerialization:
    def test_a_raised_exception_in_details_does_not_break_the_body(self) -> None:
        """Pydantic puts the raised ValueError *object* into a validation error's
        ``ctx``. An error handler that cannot serialize its own body turns a
        client's malformed request into a 500 and hides what was wrong."""
        detail = ErrorDetail(
            code="validation_error",
            message="request validation failed",
            details={"errors": [{"loc": ("body", "rate"), "ctx": {"error": ValueError("bad")}}]},
        )

        rendered = detail.model_dump(mode="json")

        assert rendered["details"]["errors"][0]["ctx"]["error"] == "bad"

    def test_json_native_values_are_left_alone(self) -> None:
        assert json_safe({"a": 1, "b": [True, None, "x"], "c": 1.5}) == {
            "a": 1,
            "b": [True, None, "x"],
            "c": 1.5,
        }

    def test_a_decimal_becomes_its_string(self) -> None:
        assert json_safe(Decimal("412600000.5")) == "412600000.5"
