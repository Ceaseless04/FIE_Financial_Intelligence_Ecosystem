# Phase 1 — Shared Platform

**Status: complete.** Phase 2 (MarketMind) may begin.

## Objective

Build the infrastructure every product depends on, with no financial domain
logic anywhere in it.

## Delivered

| Package | Contents |
|---|---|
| `fie_common` | Environment/settings base, error hierarchy with retryability and HTTP mapping, retry + circuit breaker + timeout, redaction and stable-serialization helpers |
| `fie_observability` | structlog JSON logging with mandatory redaction, request-context propagation, OpenTelemetry tracing, platform + AI token metrics |
| `fie_schemas` | Strict Pydantic base models, response envelopes and pagination, health/readiness contracts, **provenance and attribution** |
| `fie_ai` | `AIProvider` abstraction, Claude / Ollama / vLLM providers, registry and fallback router, structured-output parsing and grounding checks |
| `fie_auth` | JWT issue/verify with access-vs-refresh separation, bcrypt password hashing with explicit length policy, RBAC with wildcard permissions and tenant isolation |
| `fie_database` | Async SQLAlchemy engine and scoped sessions, Alembic setup, Redis client with JSON helpers and a safe distributed lock, Neo4j client with a read-path write guard |
| `fie_events` | Domain event envelope with causation chaining, Redis Streams bus with consumer groups, idempotent handling, dead-lettering and stale-message reclaim |
| `fie_testing` | Behaviourally faithful fakes (including an Anthropic SDK double and a functional in-memory Redis), factories, grounding/redaction assertions, infrastructure probes |

Also delivered: `docker-compose.dev.yml` and `docker-compose.prod.yml`, the
platform Dockerfile, OTel collector configurations, the CI quality gate, and the
`.env.example` template.

## Gate results

Verified on 2026-08-10 against live Docker infrastructure (PostgreSQL 16 with
pgvector, Redis 7, Neo4j 5, Ollama):

```
649 passed, 5 skipped
Coverage: 95.73%   (gate: 90%)
ruff check:        All checks passed
ruff format:       74 files already formatted
mypy --strict:     no issues in 48 source files
```

The 5 skips are the live-Claude provider tests, which require
`FIE_CLAUDE_API_KEY`. They are intentionally not part of the default gate so a
normal CI run costs nothing; run them before a release.

Integration tests ran against real infrastructure, not mocks — 35 of them,
covering transaction rollback, distributed locking, Cypher parameter binding,
consumer-group reclaim, dead-lettering, idempotent redelivery, and a real model
completion through the provider abstraction.

## Constraint compliance

> "No financial domain logic is allowed in Phase 1."

Held. No package contains money types, ratios, valuation, forecasting, or risk
models. `fie_schemas.provenance` names financial *source kinds* (SEC filing,
earnings call) as enum members, which is document taxonomy rather than financial
logic — nothing computes or interprets a financial value.

Token accounting is recorded in `fie_observability.metrics` but deliberately not
priced; pricing is FinOps domain logic and belongs to Phase 7.

## Decisions worth carrying forward

**Provenance is a type, not a convention.** Four products carry grounding
constraints. Encoding them as validation means a violation fails at construction
time rather than surviving into a research report.

**Refusal is not failure.** The router treats a model refusal as deterministic —
it skips retry and switches provider — while transport failures retry first.
Conflating them wastes budget or downgrades unnecessarily.

**The provider layer absorbs vendor reality.** Callers may set `temperature`;
the Claude provider drops it because current models reject it with a 400. The
neutral contract stays usable across providers instead of degrading to the
intersection of all of them.

**Caller errors don't trip circuit breakers.** A 400 says nothing about
dependency health.

**Fakes are behaviourally faithful.** The Anthropic double reproduces refusal
shape, cache-token fields, and thinking blocks so the real translation logic is
exercised. The Redis double implements consumer-group and delivery-count
semantics so dead-letter and reclaim paths are genuinely covered.

## Issues found and fixed during the phase

Recorded because each was caught by running something rather than reading it:

1. **Production guardrail was defeatable by length.** The default JWT secret
   (`insecure-development-secret-do-not-use`, 38 chars) passed the 32-character
   minimum. Added substring marker detection.
2. **`postgres:16-alpine` does not ship pgvector.** The bootstrap script failed
   and the container exited. Switched dev and prod to `pgvector/pgvector:pg16`.
   Only starting the stack surfaced this.
3. **`httpx.AsyncClient` construction cost ~570ms.** It eagerly builds an SSL
   context and parses the CA bundle, paid per client even for plain HTTP. Added
   a shared cached context (`fie_ai.http`); construction dropped to ~1ms.
4. **`reclaim_idle_ms` floor of 1000ms was arbitrary.** Lowered to 100ms; the
   conservative 60s default is unchanged.
5. **Test doubles inherited production retry.** A scripted error was silently
   retried and the next queued response returned. `FakeAIProvider` now defaults
   to a single attempt; retry is tested explicitly where it belongs.

## Phase 2 entry criteria

Satisfied:

- [x] All Phase 1 unit tests pass
- [x] All Phase 1 integration tests pass against real infrastructure
- [x] Coverage ≥ 90% (95.73%)
- [x] Lint, format, and strict type checking clean
- [x] No financial domain logic in shared packages
- [x] Development and production environments defined and separated
- [x] CI quality gate defined

## What Phase 2 will need from Phase 1

MarketMind builds on: `Neo4jClient` for the graph, `PostgresDatabase` for
document and embedding storage (pgvector is installed by the bootstrap script),
`AIRouter` for extraction and GraphRAG synthesis, `SourceReference` /
`Provenance` for citations on every extracted entity, and `DomainEvent` +
`RedisStreamEventBus` to publish `marketmind.*` events that Atlas and Sentinel
subscribe to.

One thing Phase 2 must add that Phase 1 deliberately left out: the graph schema
and entity-resolution logic are domain concerns and belong in
`apps/marketmind`, not in `packages/database`.
