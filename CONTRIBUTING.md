# Contributing

## Branching

```
main        production-ready
└── develop integration
    ├── feature/*
    ├── bugfix/*
    └── experiment/*
```

No direct commits to `main` or `develop`. Every change arrives by pull request
and must pass every CI stage.

```bash
git switch develop && git pull
git switch -c feature/marketmind-entity-extraction
# ... work ...
git push -u origin feature/marketmind-entity-extraction
```

## Before opening a pull request

Run the same gate CI runs:

```bash
ruff check .
ruff format --check .
mypy
pytest packages --cov
```

Integration tests need the dev stack:

```bash
docker compose -f docker-compose.dev.yml up -d postgres redis neo4j ollama
```

## Definition of done

A change is not complete until all of these hold:

- [ ] Implementation exists
- [ ] Unit tests exist
- [ ] Integration tests exist where the change touches infrastructure
- [ ] API tests exist where the change touches an HTTP surface
- [ ] AI evaluation tests exist where the change touches model behaviour
- [ ] Documentation updated
- [ ] Structured logging added for meaningful operations
- [ ] Errors are typed (`FIEError` subclasses), not bare exceptions
- [ ] Security requirements satisfied — no secrets in code, no new unvalidated input
- [ ] Coverage stays at or above 90%
- [ ] CI passes, including the Docker build

## Phase discipline

The platform is built sequentially. Do not start work belonging to a later phase
while the current phase's gates are failing, and do not add domain logic to a
shared package to make a product feature easier — that boundary is what keeps
the six products independently deployable.

Each phase is recorded in `docs/phases/`.

## Architectural rules

These are not style preferences; a PR that breaks one will be rejected.

**1. Claude never computes a financial result.** Valuation, forecasting,
accounting, risk scoring, and cost calculation live in deterministic Python
services. Claude interprets and explains their output.

**2. Only `fie_ai.providers.*` imports a vendor SDK.** Application and agent
code speaks `fie_ai.contracts` so providers stay swappable.

**3. Shared packages contain no domain logic.** `packages/` is infrastructure.
Money types, ratios, and risk models belong in `apps/`.

**4. Values carry their provenance.** Anything derived from a source or a model
is wrapped in `Attributed[T]` with a `Provenance`. Facts are cited, estimates
state their assumptions, generated content names its model.

**5. Every outbound call is bounded.** Timeout, retry, and circuit breaker —
use `ResiliencePolicy` rather than calling a dependency directly.

**6. Event handlers are idempotent.** Delivery is at-least-once.

## Writing tests

Assert on behaviour, not implementation. Never assert on exact model text — use
structured output, schema validation, and grounding checks instead. See
[docs/testing.md](testing.md).

Inject clocks, sleeps, and randomness rather than sleeping for real.

## Commit messages

Explain why, not what — the diff already says what.

```
Reject development-marker secrets in production

The length check alone passed the platform's own default JWT secret,
which is 38 characters. Added substring detection for known development
markers so a leaked default fails at startup.
```
