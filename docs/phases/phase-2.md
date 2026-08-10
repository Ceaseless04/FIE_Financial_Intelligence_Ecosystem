# Phase 2 — MarketMind (Financial Knowledge Graph)

**Status: complete.** Phase 3 (Atlas) may begin.

## Objective

Build the knowledge substrate the other five products read from: entities,
relationships, the documents behind them, and cited retrieval over both.

## Scope boundary

The specification is explicit that MarketMind manages entities, relationships,
knowledge, retrieval, and graph reasoning — and that it must **not** make
investment recommendations, optimise portfolios, budget, or decide startup
investments.

That boundary is enforced in code rather than documented and hoped for:

- `Entity` rejects attributes named `rating`, `recommendation`, `price_target`,
  `fair_value`, `investment_score`, `portfolio_weight`, `valuation`, or
  `buy_sell_hold` at construction time. The failure mode this prevents is
  mundane and likely: someone stores a score on a node "just for now" and three
  products start reading it as fact.
- The extraction prompt forbids evaluative output, and the synthesis prompt
  forbids investment advice, ratings, and price targets.
- No module in `apps/marketmind` computes a financial quantity. There is no
  arithmetic on money anywhere in the app.

## Delivered

| Area | Module | What it does |
|---|---|---|
| Domain | `domain/entities.py` | 7 entity types, 6 authoritative identifier types with normalisation, merge semantics, the scope guard |
| Domain | `domain/relationships.py` | 14 relationship types, endpoint-type schema, symmetry, temporal validity windows |
| Domain | `domain/documents.py` | Documents with content hashing; chunks whose offsets must resolve back to the source |
| Ingestion | `ingestion/chunking.py` | Deterministic, structure-aware chunking with stable ids and preserved character offsets |
| Ingestion | `ingestion/metadata.py` | Pure-regex recovery of tickers, CIKs, LEIs, form types, and fiscal periods |
| Ingestion | `ingestion/extraction.py` | Model-based entity/relationship extraction behind structured output, citation checking, and schema validation |
| Ingestion | `ingestion/pipeline.py` | Document → chunk → embed → extract → resolve → graph, with events |
| Resolution | `resolution/normalization.py` | Legal-suffix stripping, person-name canonicalisation, blocking keys |
| Resolution | `resolution/resolver.py` | The deterministic identity ladder and its auditable decisions |
| Graph | `graph/queries.py`, `schema.py`, `repository.py` | Parameterised Cypher behind a label allowlist; idempotent upserts; bounded traversal; shortest path |
| Storage | `storage/models.py`, `repository.py` | pgvector chunk store, cosine search filtered by embedding model, entity-mention join |
| Retrieval | `retrieval/graphrag.py` | Vector seed → graph expansion → cited synthesis → **citation verification** |
| API | `api/` | Ingestion, entity lookup and search, relationships, visualization, shortest path, stats, retrieval, health |
| Platform | `fie_ai.embeddings` | `EmbeddingProvider` abstraction plus an Ollama implementation |
| Platform | `fie_testing.fake_embeddings` | Deterministic hash embeddings with real vector geometry |

Also delivered: the `marketmind` Alembic history with its own namespaced version
table, `Dockerfile.marketmind`, dev and production compose services, and the
`MARKETMIND_*` configuration block.

## Gate results

Verified on 2026-08-10 against live Docker infrastructure (PostgreSQL 16 with
pgvector, Redis 7, Neo4j 5, Ollama):

```
945 passed, 9 skipped
Coverage:          94.47%   (gate: 90%)
ruff check:        All checks passed
ruff format:       131 files already formatted
mypy --strict:     no issues in 84 source files
alembic:           upgrade -> downgrade -> upgrade -> check, all clean
```

MarketMind contributes 300 of those tests: 206 unit, 45 integration against real
Neo4j and pgvector, and 49 API contract tests.

The 9 skips are the live-provider tests: 5 need `FIE_CLAUDE_API_KEY`, 4 need an
Ollama model that is not pulled by default. Both skip with an actionable message
rather than failing.

### Phase gate

> "Graph ingestion and retrieval tests must pass before merging."

`tests/integration/test_ingestion_and_retrieval.py` is that gate. It runs the
whole path against both real databases: a filing is chunked, embedded, stored,
extracted, resolved, written to Neo4j, and then answered from — with the
citation verification asserted end to end.

## Decisions worth carrying forward

**Identity is deterministic and never delegated to a model.** The resolver
decides by identifier match, identifier *conflict*, canonical name, alias, then
guarded fuzzy similarity. Three reasons: re-ingestion must be reproducible, a
merge must be explainable when someone asks why two companies became one, and a
wrong merge silently corrupts every traversal that touches the node. Every
decision carries its evidence.

**A conflicting identifier outranks an identical name.** Two companies both
called "Acme Corp" with different CIKs are two companies. Where some identifiers
agree and others conflict, the resolver refuses to decide rather than guessing.

**Citations are verified, not decorated.** GraphRAG checks every cited source id
against the context the model was actually given, strips the ones that do not
resolve, and reports `grounded: false`. A caller never receives a reference it
cannot follow. Answering from an empty context raises instead of improvising.

**Chunk offsets are an invariant, not metadata.** Ingestion verifies that every
chunk's offsets still resolve to its stored text and fails the unit of work if
not — because if that ever drifts, every citation derived from the document
points somewhere other than it claims.

**Chunk ids are derived, not random.** `sha256(document_id:index)`. A random id
means re-ingestion keeps the old chunk row (the store upserts on document and
index) while the mention rows written in the same pass point at an id that does
not exist.

**Retrieval filters on the embedding model.** Vectors from two models are not
comparable, and a model rollout leaves both in the table at once. Without the
filter, the result is confident nonsense rather than an error.

**The extraction contract is enforced after the model, not just requested of
it.** Fabricated chunk citations, edges between nonsensical endpoint types, and
relationships referencing unextracted entities are all rejected — and the
rejections are returned in the ingestion report rather than dropped, so a run
that lost half its edges does not look like a clean success.

## Issues found and fixed during the phase

Each was caught by running something rather than reading it:

1. **Symmetric edges were written twice.** `dedupe_key` canonicalises the
   endpoints, but a Cypher `MERGE` pattern is directed:
   `(a)-[:COMPETES_WITH]->(b)` does not match an existing
   `(b)-[:COMPETES_WITH]->(a)` however the properties compare. Symmetric edges
   now agree on a stored direction as well as a key. Only the real Neo4j
   integration test could have found this.
2. **A relationship with a missing endpoint reported success.** The query's
   leading `MATCH` clauses return no rows rather than raising, and the
   repository fell back to returning the input id — so a dangling edge was
   counted in the ingestion totals and published as a `relationship.created`
   event. It now raises `NotFoundError`.
3. **Chunk ids were random**, breaking re-ingestion as described above.
4. **The migration's `DROP SCHEMA ... CASCADE` deleted Alembic's own version
   table**, leaving the database unmigratable mid-downgrade. The downgrade is
   now deliberately asymmetric: tables go, schema stays.
5. **Autogenerate would have dropped the HNSW index.** SQLAlchemy cannot express
   pgvector operator classes, so the index is migration-only — and invisible to
   the metadata comparison. `alembic check` caught it; `include_object` now
   excludes it, and `alembic check` runs in CI.
6. **The correlation id was missing from exactly the responses that need it.**
   The ambient request context unwinds when a handler raises, so by the time
   Starlette's outermost error middleware ran the 500 handler, the id was gone.
   It is now also stashed on the request.
7. **`psycopg` was never installed.** `PostgresSettings.sync_dsn` selects it for
   Alembic, so every migration failed at import. Phase 1 had the same latent
   gap; no migration had been run until now.
8. **The infrastructure gate was all-or-nothing.** `FIE_REQUIRE_INFRA=1` would
   have failed CI for the absent model server on every pull request. It now
   accepts a service list, so a container that failed to start still fails the
   run while a service CI does not provide still skips.
9. **CI used `postgres:16-alpine`**, which cannot create the `vector`
   extension. Switched to `pgvector/pgvector:pg16`, matching the compose files.

## Events published

Consumers subscribe to `fie.events.marketmind`.

| Event | Payload highlights |
|---|---|
| `marketmind.document.ingested` | Document id, content hash, chunk and entity counts |
| `marketmind.document.duplicate` | The redelivered id and the existing one |
| `marketmind.entity.created` | Entity id, type, canonical name, identifier keys |
| `marketmind.entity.merged` | **Surviving and retired ids**, match reason, confidence |
| `marketmind.relationship.created` | Endpoints, type, supporting documents |
| `marketmind.graph.updated` | Per-run totals including rejections |

`entity.merged` exists specifically for downstream products. Atlas, Sentinel,
and Venture cache entity ids; when resolution decides two nodes were always the
same company, one id stops existing. A consumer holding the retired id must be
told what replaced it, or it silently reads an empty neighbourhood and reports
"no relationships found" for a company with fifty.

## Known limitations

Stated rather than hidden — each is a deliberate deferral, not an oversight:

- **Event publication is not transactional.** A crash between the graph write
  and the publish loses the notification. Consumers are idempotent and
  re-ingestion replays, so this is at-least-once with a small loss window.
  Closing it needs a transactional outbox, which belongs with the rest of the
  cross-service delivery guarantees in Phase 8.
- **Ingestion is synchronous.** The caller learns what was extracted and what
  was rejected. Moving it onto a queue before the failure modes are visible
  would hide them behind a job id; that is a Phase 8 change.
- **Postgres and Neo4j cannot share a transaction.** Postgres is written first
  inside a caller-managed session scope, so a graph failure propagates and the
  rollback undoes the document. Redelivery replays safely because every graph
  write is idempotent.
- **Extraction is capped at 60 chunks per document** by default. A full 10-K
  would otherwise become hundreds of model calls. The truncation is logged.
- **Traversal depth is capped at 3** and node counts are bounded. A wider
  expansion on a dense financial graph returns a result no reader can interpret
  and no browser can lay out.

## Phase 3 entry criteria

Satisfied:

- [x] Graph ingestion tests pass against real Neo4j
- [x] Retrieval tests pass against real pgvector
- [x] API contract tests cover authentication, authorization, and validation
- [x] Coverage ≥ 90% (94.47%)
- [x] Lint, format, and strict type checking clean
- [x] Migrations apply, reverse, and re-apply against a real database
- [x] No investment recommendations, portfolio optimisation, budgeting, or
      startup investment logic anywhere in the app

## What Phase 3 will need from Phase 2

Atlas builds on: the entity and relationship graph for company context, the
GraphRAG retrieval service for cited evidence, `marketmind.*` events to keep its
own view current (particularly `entity.merged`), and `SourceReference` chains
that resolve to exact character spans so a research report's citations are
checkable.

What Atlas must add, and MarketMind deliberately does not have: valuation,
financial statement analysis, and anything that computes a number from a
filing. Those are deterministic Python services in `apps/atlas`, with Claude
interpreting and explaining their output rather than producing it.
