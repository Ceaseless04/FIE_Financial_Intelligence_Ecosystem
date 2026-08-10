# Architecture

## The central constraint

Claude is the primary reasoning and orchestration model. It is never the source
of truth for a financial calculation.

This single rule shapes most of the design. Valuation, forecasting, accounting,
risk scoring, and cost computation are deterministic Python services. Claude
reads their output, interprets it, and explains it. The split matters because a
language model producing a discounted cash flow figure is unverifiable and
irreproducible — two properties that make a financial product unusable
regardless of how often the number happens to be right.

The rule is enforced, not merely documented. `fie_schemas.provenance` refuses to
construct a `DERIVED` value that names a model, and refuses to construct a
`FACT` without a citation. Code that tries to pass model output off as a
calculation fails at construction time.

## Layering

```
        ┌──────────────────────────────────────────────────┐
apps/   │  Atlas   CFO.ai   MarketMind   Sentinel   ...     │  domain logic
        └───────────────────────┬──────────────────────────┘
                                │ depends on
        ┌───────────────────────▼──────────────────────────┐
packages│  ai   auth   database   events   observability    │  infrastructure
        │  schemas   common   testing                      │  (no domain logic)
        └──────────────────────────────────────────────────┘
```

The dependency arrow points one way. A shared package must never import from an
app, and must never contain financial semantics — no money types, no ratios, no
risk models. That boundary is what keeps six products independently testable and
deployable out of one repository. Phase 1 has no domain logic at all, by
constraint.

The boundary runs the other way too. MarketMind stores knowledge, not judgement:
its `Entity` model rejects attributes like `rating` or `price_target` outright,
because valuation belongs to Atlas and investment judgement to Venture. An
enforced boundary survives contact with a deadline; a documented one does not.

### Package dependencies

```
common ──────────────┐
   ▲                 │
   ├── schemas ──────┤
   ├── observability ┤
   │       ▲         │
   │       └── ai ───┤
   ├── auth ─────────┤
   ├── database ─────┤
   │       ▲         │
   │       └── events┤
   └────────── testing (depends on all; test-only)
```

No cycles. `fie_events` depends on `fie_database` because Redis Streams is the
transport; swapping transports would touch one package.

## AI provider abstraction

```
AIProvider (abstract)
├── ClaudeProvider   default in production
├── OllamaProvider   local development, evaluation, cost control
└── VLLMProvider     self-hosted batch inference
```

`fie_ai.providers.claude` is the only module in the repository that imports the
`anthropic` SDK. Everything above speaks `fie_ai.contracts`. Switching providers
is configuration, not code.

The base class is a template method: concrete providers implement three small
hooks (`_complete`, `_stream`, `_ping`) and the base adds timeout, retry,
circuit breaking, tracing, and token/latency metrics exactly once. No provider
can forget them and no caller has to remember them.

### What the provider layer absorbs

The neutral contract is deliberately more permissive than any single vendor, and
the provider reconciles the difference:

| Neutral contract | Claude reality |
|---|---|
| `temperature` is settable | Current model families reject it with a 400 — dropped, not forwarded |
| `thinking: ADAPTIVE` | Maps to adaptive thinking; `budget_tokens` no longer exists |
| `thinking: DISABLED` + `effort` | Illegal above `high` effort — rejected locally with a clear message |
| Response text | `stop_reason` is checked *before* content is indexed, because a refusal is HTTP 200 with empty content |
| `max_tokens > 16000` | Automatically switched to the streaming path to avoid an HTTP timeout |

### Failure versus refusal

`AIRouter` distinguishes two things that look similar and need opposite
handling:

- **Failure** — the provider is down, timing out, or rate-limited. Retry inside
  the provider; once its circuit trips, move to a fallback.
- **Refusal** — the model declined on safety grounds. Retrying the identical
  request against the same model reproduces the identical refusal, so refusal
  skips retry entirely and goes straight to a different provider.

Conflating them produces either wasted retry budget or an unnecessary downgrade.

## Grounding and provenance

Four of the six products carry constraints about separating fact from analysis.
Rather than restate that as prose in six codebases, it is a type:

| Kind | Required evidence |
|---|---|
| `FACT` | At least one `SourceReference`; may not name a model |
| `DERIVED` | The name of the deterministic computation; may not name a model |
| `ESTIMATE` | The assumptions it rests on |
| `GENERATED` | The model that produced it |

`Attributed[T]` binds a value to its provenance so attribution cannot be lost
when the value crosses a service boundary or lands in a report.

`fie_ai.structured.check_grounding` closes the other half: it verifies that every
source a model cites actually resolves to a source that was supplied to it. A
fabricated citation is the most dangerous hallucination in a research product
precisely because it looks exactly like a real one.

## Resilience

Every outbound call is wrapped in a `ResiliencePolicy`: timeout per attempt,
bounded exponential backoff with jitter, and a circuit breaker per dependency.

Two design details worth stating:

- The breaker wraps each *individual attempt*, not the whole retry loop, so a
  half-open trial call is one attempt rather than a full retry budget.
- Caller errors (a 400, a 401, a schema violation) do not trip the breaker. A
  malformed request says nothing about dependency health, and letting it open
  the circuit would take out a healthy dependency.

Errors carry a `retryable` flag; the retry helper keys on it rather than on
exception-message matching.

## Events

Redis Streams with consumer groups gives at-least-once delivery. Three failure
modes are handled explicitly:

| Failure | Handling |
|---|---|
| Handler raises | Message left unacknowledged; redelivery is the retry |
| Consumer crashes | Pending messages idle past a threshold are claimed by another consumer |
| Poison message | After the delivery budget, moved to a dead-letter stream and acknowledged |

At-least-once means duplicates happen. `IdempotentHandler` claims
`(consumer_group, event_id)` in Redis before the handler runs and releases the
claim if it fails — so a transient failure is still retried while a successful
one is never repeated. Double-processing a "filing ingested" event is not an
acceptable outcome in a financial system.

Streams are per-product (`fie.events.atlas`), not per-event-type, so a consumer
interested in several of a product's events reads one stream and filters.

## Knowledge graph and retrieval (MarketMind)

MarketMind is the shared intelligence layer: it records what is true and how
things connect, and every other product reads from it. Its full record is in
[phases/phase-2.md](phases/phase-2.md); the architectural points are these.

**Two stores, one model of the world.** Neo4j holds structure — entities, typed
relationships, validity windows. Postgres holds the documents that structure was
derived from, plus the pgvector embeddings used to find them. Splitting them is
not incidental: a graph database storing multi-megabyte filing bodies degrades
every traversal that touches those nodes. An `entity_mentions` table joins the
two, which is what turns a graph node into a quotable citation.

**GraphRAG, in six steps.** Vector search seeds; the entities mentioned in those
chunks become graph anchors; a bounded traversal collects surrounding structure;
chunks and paths become one context block with explicit source ids; the model
answers from that context only; and **the citations are verified against the
supplied ids before the answer is returned.**

That last step is what makes the system usable in a financial product. Without
it, the output is confident, well-formatted, and unverifiable. A fabricated
source id is stripped and the answer is flagged `grounded: false` rather than
being returned as though it were sourced.

**Identity is deterministic.** Entity resolution never uses a model. It decides
on a ladder — shared authoritative identifier, then *conflicting* identifier
(decisive against a match, even when names are identical), then canonical name,
alias, and finally guarded fuzzy similarity. The decision and its evidence are
returned so a surprising merge can be explained without re-running the pipeline.
A wrong merge silently corrupts every traversal that touches the node, which is
why the thresholds are deliberately conservative.

**The model is used where it is actually the right tool** — recognising that
"Tim Cook, who leads Apple" states an `EXECUTIVE_OF` relationship — and nowhere
else. Identifier parsing, chunking, resolution, and deduplication are all
deterministic regex and rules.

## Observability

Every log line, span, and published event carries the same correlation id, so
one research request can be followed across Atlas → MarketMind → Sentinel →
CFO.ai in a single query.

Redaction runs as a processor on every record rather than at call sites, because
a call site can be forgotten. The same applies to span attributes, which reach
the same backends as logs and leak just as easily. The production OTel collector
strips credential-shaped attributes again as defence in depth against
third-party instrumentation.

Token accounting is recorded here but **not priced** here. Cost calculation is
FinOps domain logic (Phase 7) and belongs in a deterministic service. Cache-read
and cache-write tokens are tracked as distinct kinds because they bill at
different rates; collapsing them would make any downstream attribution wrong.

## Configuration and secrets

All configuration resolves from environment variables. Production guardrails run
at startup: `require_production_secret` rejects missing secrets, known
placeholders, values containing development markers (`insecure`,
`local-development`, `change-me`, …), and anything under 32 characters.

The marker check exists because a length check alone is insufficient — the
platform's own default JWT secret is 38 characters, long enough to pass a naive
length test while being a publicly known string. Every credential in
`docker-compose.dev.yml` is rejected by these guardrails when
`FIE_ENVIRONMENT=production`, so the development stack can stay zero-setup
without its credentials being able to reach a real deployment.

## Phase sequencing

Phases 1 and 2 are complete. Phases 3–9 follow in order, each gated on the
previous phase's tests passing. The order is not arbitrary: MarketMind came
second because the knowledge graph is the shared intelligence layer that Atlas,
Sentinel, and Venture all read from, and building those first would have meant
building it three times.

Each app owns its own `pyproject.toml`, Alembic history with a namespaced
version table, Dockerfile, and test suites. Sharing one migration history across
products would make them contend for the head revision and let one product's
autogenerate run propose dropping another's tables.
