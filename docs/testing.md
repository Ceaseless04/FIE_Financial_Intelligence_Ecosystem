# Testing strategy

## Layout

Every package carries its own tests:

```
packages/<name>/tests/
├── unit/          fast, isolated, no external dependencies
└── integration/   real Postgres / Redis / Neo4j / model server
```

Markers: `unit`, `integration`, `api`, `evaluation`, `e2e`.

```bash
pytest packages -m unit           # seconds, no infrastructure
pytest packages -m integration    # requires the dev stack
pytest packages --cov            # full suite + 90% coverage gate
```

## Integration tests use real infrastructure

Mocks prove that code calls what you expected. They cannot prove that a
transaction actually rolls back, that a Redis consumer group actually redelivers
after a crash, or that Cypher parameter binding actually prevents injection.
Those are the properties worth testing, so integration tests run against real
services.

When infrastructure is not running, they skip with an actionable reason:

```
SKIPPED — postgres is not reachable at localhost:5432 —
start it with `docker compose -f docker-compose.dev.yml up -d postgres`
```

CI sets `FIE_REQUIRE_INFRA=1`, which converts skips into failures. Without that,
a container that failed to start would produce a green build full of silent
skips — the worst possible outcome, because it looks like coverage.

## Testing AI behaviour

**Never assert on exact model text.** Model output varies across runs, versions,
and providers; a test that pins prose is a test that will be deleted the first
time it fails for no reason.

Four techniques replace it:

### 1. Structured output

The model fills a schema; validation is deterministic; tests assert on typed
fields.

```python
structured = structured_request(request, FilingSummary)
result = parse_structured(await provider.complete(structured), FilingSummary)

assert result.fiscal_year == 2024  # not a string comparison on prose
assert 0.0 <= result.confidence <= 1.0
```

`parse_structured` rejects a refusal and rejects truncated output *before*
parsing — truncated JSON can parse as valid-but-wrong, which is exactly how a
half-built financial record reaches a report.

### 2. Grounding checks

Verify that every citation the model emits resolves to a source that was
actually supplied:

```python
report = check_grounding(cited_ids, allowed_ids)
assert report.is_grounded
assert report.unknown_ids == []
```

A fabricated citation is the most dangerous hallucination in a research product
because it is indistinguishable from a real one at a glance.

### 3. Provenance validation

Grounding rules are types, so violations are construction-time errors rather
than review findings:

```python
with pytest.raises(ValidationError):
    Provenance(kind=AssertionKind.FACT)  # no citation

with pytest.raises(ValidationError):
    Provenance(
        kind=AssertionKind.DERIVED, computation="dcf_v1", model="claude-opus-5"
    )  # model as calculator
```

### 4. Determinism assertions

Financial calculations must return identical results across runs.
`assert_deterministic` is the guard that stops a model call being slipped into a
computation path later.

## Fakes are behaviourally faithful

The fakes in `fie_testing` are built to reproduce behaviour, not to be
convenient:

- `FakeAnthropicClient` reproduces the SDK's response shape — refusals with
  empty content, cache-token fields, thinking blocks preceding text — so
  `ClaudeProvider`'s real translation logic runs instead of being bypassed. It
  records every request payload, so tests assert on what was actually sent:
  that `temperature` was dropped, that the system prompt carried a cache
  breakpoint, that thinking was adaptive.

- `FakeRedis` implements key/value semantics, expiry, Lua-based lock release,
  and stream consumer groups with pending-entry lists and delivery counts. That
  is what makes dead-lettering and stale-message reclaim deterministically
  testable — those paths need controllable delivery counts and idle time. The
  same logic then runs against real Redis in the integration suite.

A fake that returns canned values would let all of this pass while the real code
was broken.

## Determinism in tests

Time, randomness, and sleep are injected rather than mocked globally:

```python
policy = RetryPolicy(initial_backoff_seconds=1.0, multiplier=2.0, jitter=0.0)
await retry_async(operation, policy=policy, sleep=recorder)

assert recorder.delays == [1.0, 2.0]  # the exact schedule, no real waiting
```

`FakeRedis` and `CircuitBreaker` take virtual clocks for the same reason: a test
that sleeps is a test nobody runs.

## Coverage

The gate is 90%; the suite currently sits at 95.73%. Coverage is a floor, not a
goal — the packages that matter most (provenance, resilience, the provider
translation layer) are at or near 100% because their failure modes are the ones
that produce wrong financial output rather than an obvious crash.
