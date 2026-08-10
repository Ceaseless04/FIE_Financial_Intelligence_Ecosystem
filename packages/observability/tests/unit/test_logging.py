"""Unit tests for structured logging and its redaction guarantee."""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
import structlog

from fie_common.config import CoreSettings, LogLevel
from fie_common.utils import REDACTED
from fie_observability.context import request_context
from fie_observability.logging import configure_logging, get_logger, reset_logging

pytestmark = pytest.mark.unit


@pytest.fixture
def captured_logs() -> Iterator[list[dict[str, object]]]:
    """Capture records after the full processor chain has run."""
    records: list[dict[str, object]] = []

    reset_logging()
    configure_logging(CoreSettings(log_level=LogLevel.DEBUG), json_logs=True, force=True)

    # Swap only the final renderer so redaction and context processors still run.
    processors = list(structlog.get_config()["processors"])

    def capture(_logger: object, _method: str, event_dict: dict[str, object]) -> str:
        records.append(dict(event_dict))
        return json.dumps(event_dict, default=str)

    structlog.configure(processors=[*processors[:-1], capture])
    yield records
    reset_logging()


class TestConfiguration:
    def test_configure_is_idempotent(self) -> None:
        reset_logging()
        configure_logging()
        configure_logging()  # must not raise or duplicate handlers

        assert get_logger("test") is not None
        reset_logging()

    def test_force_allows_reconfiguration(self) -> None:
        reset_logging()
        configure_logging(json_logs=True)
        configure_logging(json_logs=False, force=True)

        assert get_logger("test") is not None
        reset_logging()

    def test_get_logger_configures_on_first_use(self) -> None:
        reset_logging()

        assert get_logger("lazy") is not None
        reset_logging()

    def test_console_renderer_is_selectable(self) -> None:
        reset_logging()
        configure_logging(json_logs=False, force=True)

        assert get_logger("console") is not None
        reset_logging()


class TestRedaction:
    def test_top_level_secrets_are_redacted(self, captured_logs: list[dict[str, object]]) -> None:
        get_logger("test").info("auth_attempt", user="alice", password="hunter2")

        assert captured_logs[-1]["password"] == REDACTED
        assert captured_logs[-1]["user"] == "alice"

    def test_nested_secrets_are_redacted(self, captured_logs: list[dict[str, object]]) -> None:
        get_logger("test").info("startup", config={"db": {"database_url": "postgres://u:p@h/d"}})

        config = captured_logs[-1]["config"]
        assert isinstance(config, dict)
        assert config["db"]["database_url"] == REDACTED  # type: ignore[index]

    def test_secrets_inside_lists_are_redacted(
        self, captured_logs: list[dict[str, object]]
    ) -> None:
        get_logger("test").info("providers", items=[{"api_key": "sk-ant-123"}, {"name": "ok"}])

        items = captured_logs[-1]["items"]
        assert isinstance(items, list)
        assert items[0]["api_key"] == REDACTED
        assert items[1]["name"] == "ok"

    def test_api_key_variants_are_all_caught(self, captured_logs: list[dict[str, object]]) -> None:
        get_logger("test").info(
            "call",
            api_key="a",
            apikey="b",
            anthropic_api_key="c",
            authorization="Bearer d",
            token="e",
        )

        record = captured_logs[-1]
        for key in ("api_key", "apikey", "anthropic_api_key", "authorization", "token"):
            assert record[key] == REDACTED, f"{key} was not redacted"

    def test_redaction_is_case_insensitive(self, captured_logs: list[dict[str, object]]) -> None:
        get_logger("test").info("call", API_KEY="sk-secret", Password="p")

        assert captured_logs[-1]["API_KEY"] == REDACTED
        assert captured_logs[-1]["Password"] == REDACTED


class TestRecordEnrichment:
    def test_standard_fields_are_present(self, captured_logs: list[dict[str, object]]) -> None:
        get_logger("test").info("something_happened")

        record = captured_logs[-1]
        assert record["event"] == "something_happened"
        assert record["level"] == "info"
        assert "timestamp" in record

    def test_ambient_request_context_is_merged(
        self, captured_logs: list[dict[str, object]]
    ) -> None:
        with request_context(correlation_id="corr_42", tenant_id="tenant_a"):
            get_logger("test").info("in_request")

        record = captured_logs[-1]
        assert record["correlation_id"] == "corr_42"
        assert record["tenant_id"] == "tenant_a"

    def test_explicit_fields_win_over_ambient_context(
        self, captured_logs: list[dict[str, object]]
    ) -> None:
        with request_context(correlation_id="corr_ambient"):
            get_logger("test").info("override", correlation_id="corr_explicit")

        assert captured_logs[-1]["correlation_id"] == "corr_explicit"

    def test_exception_info_is_formatted(self, captured_logs: list[dict[str, object]]) -> None:
        try:
            raise ValueError("boom")
        except ValueError:
            get_logger("test").exception("handler_failed")

        assert "ValueError" in str(captured_logs[-1].get("exception", ""))

    def test_records_are_json_serializable(self, captured_logs: list[dict[str, object]]) -> None:
        get_logger("test").info("event", count=1, nested={"a": [1, 2]})

        json.dumps(captured_logs[-1], default=str)
