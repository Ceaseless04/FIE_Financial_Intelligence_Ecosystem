# Financial Intelligence Ecosystem

A modular AI financial intelligence platform: six interconnected products on one
shared infrastructure layer.

| Product | Purpose | Phase |
|---|---|---|
| **MarketMind** | Financial knowledge graph — entities, relationships, GraphRAG | 2 |
| **Atlas** | Autonomous financial research — filings, valuation, research reports | 3 |
| **CFO.ai** | AI financial planning & analysis — budgets, forecasting, scenarios | 4 |
| **Sentinel** | Enterprise risk intelligence — detection, scoring, propagation | 5 |
| **Venture Intelligence** | Startup & VC due diligence | 6 |
| **FinOps Intelligence** | AI infrastructure cost intelligence | 7 |

## Current status

**Phase 1 — Shared Platform: complete.** Phases 2–9 are not started.

The platform is built strictly sequentially. Each phase leaves the repository in
a working state, and no phase begins while the previous phase's gates fail. See
[docs/phases/phase-1.md](docs/phases/phase-1.md) for what Phase 1 delivered and
what the Phase 2 entry criteria are.

```
649 passing (614 unit + 35 integration) · 95.73% coverage (gate: 90%)
ruff clean · mypy --strict clean
```

Integration tests were run against live PostgreSQL, Redis, Neo4j, and Ollama —
not mocks. The 5 remaining skips are the live-Claude provider tests, which need
`FIE_CLAUDE_API_KEY`.

## The architectural rule

> Claude is the primary reasoning and orchestration model. It is **never** the
> source of truth for a financial calculation.

Valuation, forecasting, accounting, risk scoring, and cost computation all live
in deterministic Python services. Claude interprets, orchestrates, and explains
those results. This is not a convention documented in prose — it is enforced in
code by [`fie_schemas.provenance`](packages/schemas/src/fie_schemas/provenance.py):

```python
# A fact cannot be constructed without a citation.
Provenance(kind=AssertionKind.FACT)  # ValidationError

# A derived value cannot claim to come from a model.
Provenance(
    kind=AssertionKind.DERIVED, computation="dcf_v1", model="claude-opus-5"
)  # ValidationError

# An estimate cannot hide its assumptions.
Provenance(kind=AssertionKind.ESTIMATE)  # ValidationError
```

## Repository layout

```
apps/                    Six product services (Phases 2-7, not yet started)
packages/
  common/                Config, error hierarchy, retry/circuit-breaker/timeout
  observability/         Structured logging, OpenTelemetry tracing, metrics
  schemas/               Shared contracts: envelopes, provenance, health
  ai/                    AIProvider abstraction: Claude, Ollama, vLLM
  auth/                  JWT authentication, RBAC authorization
  database/              PostgreSQL, Redis, Neo4j infrastructure
  events/                Domain events, Redis Streams bus, idempotency, DLQ
  testing/               Shared fakes, factories, assertions, infra probes
infrastructure/          Dockerfiles, OTel collector configs, Terraform (Phase 9)
docs/                    Architecture and per-phase records
scripts/                 Database bootstrap, backups
```

Shared packages provide infrastructure only. **No financial domain logic lives
in `packages/`** — that boundary is what keeps the six products independently
testable and deployable.

## Getting started

Requires Python 3.11+ and Docker.

```bash
# 1. Configure
cp .env.example .env

# 2. Start infrastructure
docker compose -f docker-compose.dev.yml up -d postgres redis neo4j ollama
docker compose -f docker-compose.dev.yml exec ollama ollama pull llama3.1:8b

# 3. Install
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install --no-deps -e packages/common -e packages/observability \
  -e packages/schemas -e packages/ai -e packages/auth \
  -e packages/database -e packages/events -e packages/testing

# 4. Verify
pytest packages -q
```

The development stack runs with no API key and no network egress — Ollama is the
default provider locally. Set `FIE_CLAUDE_API_KEY` and
`FIE_AI_DEFAULT_PROVIDER=claude` to use Claude.

## Using the AI abstraction

Application and agent code never imports a vendor SDK:

```python
from fie_ai import CompletionRequest, Message, Role, build_router

router = build_router()  # reads FIE_AI_* configuration

response = await router.complete(
    CompletionRequest(
        messages=[Message(role=Role.USER, content="Summarize this filing.")],
        system="You are a filing analyst.",
        max_tokens=4096,
    )
)
print(response.text, response.usage.total_tokens)
```

Swapping Claude for Ollama is a configuration change, not a code change. The
router falls back across providers on failure *and* on refusal — a refusal is
deterministic on the same model, so it skips retry and switches provider.

## Testing

```bash
pytest packages -m unit           # fast, no external dependencies
pytest packages -m integration    # requires the dev stack running
pytest packages --cov            # full suite + 90% coverage gate
```

Integration tests skip with an actionable reason when infrastructure is not
running. CI sets `FIE_REQUIRE_INFRA=1`, which turns a missing service into a
failure so a broken container can never look like a green build.

AI behaviour is tested through structured-output validation and grounding
checks, never by matching exact model text — see
[`fie_ai.structured`](packages/ai/src/fie_ai/structured.py).

## Quality gate

Every pull request must pass, in order:

```
lint → format → typecheck → unit → integration → security → coverage → docker
```

Run it locally before pushing:

```bash
ruff check . && ruff format --check . && mypy && pytest packages --cov
```

## Branching

```
main        production-ready
└── develop integration
    ├── feature/*
    ├── bugfix/*
    └── experiment/*
```

No direct commits to `main` or `develop`. Pull requests are mandatory and must
pass every CI stage.

## Documentation

- [docs/architecture.md](docs/architecture.md) — system design and boundaries
- [docs/phases/phase-1.md](docs/phases/phase-1.md) — Phase 1 record and Phase 2 entry criteria
- [docs/testing.md](docs/testing.md) — testing strategy, especially for AI
- [docs/security.md](docs/security.md) — secrets, auth, production guardrails
