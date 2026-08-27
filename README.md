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

**Phase 1 — Shared Platform: complete.**
**Phase 2 — MarketMind (knowledge graph): complete.**
**Phase 3 — Atlas (financial research): complete.**
**Phase 4 — CFO.ai (planning & analysis): complete.**
Phases 5–9 are not started.

The platform is built strictly sequentially. Each phase leaves the repository in
a working state, and no phase begins while the previous phase's gates fail. See
[docs/phases/phase-4.md](docs/phases/phase-4.md) for what CFO.ai delivered, the
bugs the suite caught, and the Phase 5 entry criteria.

```
1333 passing (1092 unit + 125 integration + 116 API) · 92.24% coverage (gate: 90%)
ruff clean · mypy --strict clean · three migration histories verified up/down/up
```

Integration tests were run against live PostgreSQL (pgvector), Redis, Neo4j, and
Ollama — not mocks. The 9 skips are live-provider tests needing either
`FIE_CLAUDE_API_KEY` or an Ollama model that is not pulled by default; both skip
with an actionable message.

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
apps/
  marketmind/            Knowledge graph: entities, relationships, GraphRAG
  atlas/                 Financial research: filings, analysis, valuation, reports
  cfo-ai/                Planning & analysis: budgets, variance, forecasts, scenarios
                         (Sentinel, Venture, FinOps: Phases 5-7)
packages/
  common/                Config, error hierarchy, retry/circuit-breaker/timeout
  observability/         Structured logging, OpenTelemetry tracing, metrics
  schemas/               Shared contracts: envelopes, provenance, health
  ai/                    AIProvider abstraction: Claude, Ollama, vLLM; embeddings
  auth/                  JWT authentication, RBAC authorization
  database/              PostgreSQL, Redis, Neo4j infrastructure
  events/                Domain events, Redis Streams bus, idempotency, DLQ
  finance/               Currency-safe decimal Money and fiscal periods —
                         value primitives, no financial semantics
  testing/               Shared fakes, factories, assertions, infra probes
infrastructure/          Dockerfiles, OTel collector configs, Terraform (Phase 9)
docs/                    Architecture and per-phase records
scripts/                 Database bootstrap, backups
```

Each app owns its own `pyproject.toml`, migration history (with a namespaced
Alembic version table), Dockerfile, and test suites, so it can be built, tested,
and deployed without the others.

Shared packages provide infrastructure and value primitives. **No financial
domain logic lives in `packages/`** — that boundary is what keeps the six
products independently testable and deployable.

The line runs between *what a number is* and *what a number means*.
`packages/finance` holds `Money`, which refuses a float and refuses to add two
currencies, and `FiscalPeriod`, which knows a quarter is not a year — the same
category as `fie_schemas.provenance`, a mechanism rather than a judgement.
Valuation, accounting identities, budgeting, variance rules, and risk scoring
live in the application that owns them. The alternative was a private copy of
`Money` in each product: four by Phase 7, and a rounding fix in one reaching
none of the others.

## Getting started

Requires Python 3.11+ and Docker.

```bash
# 1. Configure
cp .env.example .env

# 2. Start infrastructure
docker compose -f docker-compose.dev.yml up -d postgres redis neo4j ollama
docker compose -f docker-compose.dev.yml exec ollama ollama pull llama3.1:8b
# MarketMind embeds with this model; the width must match the pgvector column.
docker compose -f docker-compose.dev.yml exec ollama ollama pull nomic-embed-text

# 3. Install
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install --no-deps -e packages/common -e packages/observability \
  -e packages/schemas -e packages/ai -e packages/auth \
  -e packages/database -e packages/events -e packages/finance \
  -e packages/testing \
  -e apps/marketmind -e apps/atlas -e apps/cfo-ai

# 4. Migrate — each app owns its own history and version table
(cd apps/marketmind && alembic upgrade head)
(cd apps/atlas && alembic upgrade head)
(cd apps/cfo-ai && alembic upgrade head)

# 5. Verify
pytest -q
```

The development stack runs with no API key and no network egress — Ollama is the
default provider locally. Set `FIE_CLAUDE_API_KEY` and
`FIE_AI_DEFAULT_PROVIDER=claude` to use Claude.

Run the APIs locally against those containers:

```bash
python -m marketmind             # host and port come from MARKETMIND_API_*
# http://localhost:8001/docs     (disabled when FIE_ENVIRONMENT=production)

python -m atlas                  # host and port come from ATLAS_API_*
# http://localhost:8002/docs

python -m cfo_ai                 # host and port come from CFO_API_*
# http://localhost:8003/docs
```

`requirements.txt` is the runtime dependency set that container images install;
`requirements-dev.txt` includes it and adds the test and lint tooling. Keeping
them separate is what stops pytest, ruff, and mypy from shipping to production.

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
pytest -m unit                    # fast, no external dependencies
pytest -m integration             # requires the dev stack running
pytest -m api                     # HTTP contract tests
pytest --cov                      # full suite + 90% coverage gate
```

Integration tests skip with an actionable reason when infrastructure is not
running. `FIE_REQUIRE_INFRA` turns that skip into a failure so a broken
container can never look like a green build — set it to `1` for every service,
or to a list (`postgres,redis,neo4j`, which is what CI uses) when the
environment deliberately does not provide all of them.

AI behaviour is tested through structured-output validation and grounding
checks, never by matching exact model text — see
[`fie_ai.structured`](packages/ai/src/fie_ai/structured.py).

## Quality gate

Every pull request must pass, in order:

```
lint → format → typecheck → unit → integration → api → migrations
  → security → coverage → docker
```

Run it locally before pushing:

```bash
ruff check . && ruff format --check . && mypy && pytest --cov
(cd apps/marketmind && alembic upgrade head && alembic check)
(cd apps/atlas && alembic upgrade head && alembic check)
(cd apps/cfo-ai && alembic upgrade head && alembic check)
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
- [docs/phases/phase-1.md](docs/phases/phase-1.md) — shared platform record
- [docs/phases/phase-4.md](docs/phases/phase-4.md) — CFO.ai record: direction
  grounding, and the checks that looked like they were working
- [docs/phases/phase-3.md](docs/phases/phase-3.md) — Atlas record: numeric
  grounding, Decimal discipline end to end, and the bugs the suite caught
- [docs/phases/phase-2.md](docs/phases/phase-2.md) — MarketMind record, the bugs
  the integration suite caught, and Phase 3 entry criteria
- [docs/testing.md](docs/testing.md) — testing strategy, especially for AI
- [docs/security.md](docs/security.md) — secrets, auth, production guardrails
