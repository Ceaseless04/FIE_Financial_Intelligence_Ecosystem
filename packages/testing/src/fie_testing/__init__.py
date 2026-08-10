"""fie_testing — shared fakes, factories, probes, and assertions."""

from fie_testing.assertions import (
    assert_assumptions_explicit,
    assert_cited,
    assert_deterministic,
    assert_no_secrets,
    assert_not_presented_as_fact,
    assert_redacted,
)
from fie_testing.factories import (
    make_completion_request,
    make_event,
    make_fact_provenance,
    make_principal,
    make_source,
)
from fie_testing.fake_redis import FakeRedis
from fie_testing.fakes import (
    FakeAIProvider,
    FakeAnthropicClient,
    FakeMessage,
    FakeStopDetails,
    FakeTextBlock,
    FakeThinkingBlock,
    FakeUsage,
    make_refusal_message,
    make_text_message,
)
from fie_testing.infra import (
    ServiceEndpoint,
    has_anthropic_key,
    neo4j_endpoint,
    ollama_endpoint,
    postgres_endpoint,
    redis_endpoint,
    require_anthropic_key,
    require_service,
)

__version__ = "0.1.0"

__all__ = [
    "FakeAIProvider",
    "FakeAnthropicClient",
    "FakeMessage",
    "FakeRedis",
    "FakeStopDetails",
    "FakeTextBlock",
    "FakeThinkingBlock",
    "FakeUsage",
    "ServiceEndpoint",
    "__version__",
    "assert_assumptions_explicit",
    "assert_cited",
    "assert_deterministic",
    "assert_no_secrets",
    "assert_not_presented_as_fact",
    "assert_redacted",
    "has_anthropic_key",
    "make_completion_request",
    "make_event",
    "make_fact_provenance",
    "make_principal",
    "make_refusal_message",
    "make_source",
    "make_text_message",
    "neo4j_endpoint",
    "ollama_endpoint",
    "postgres_endpoint",
    "redis_endpoint",
    "require_anthropic_key",
    "require_service",
]
