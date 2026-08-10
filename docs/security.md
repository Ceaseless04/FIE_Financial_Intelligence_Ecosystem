# Security

## Secrets

All configuration resolves from environment variables or a secrets manager.
Nothing is hardcoded, and `.env` is gitignored — `.env.example` is the committed
template.

### Production guardrails

`fie_common.config.require_production_secret` runs from settings validators at
startup, so a misconfigured deployment fails immediately rather than at the
first authenticated request. In production it rejects:

| Condition | Rationale |
|---|---|
| Missing or blank | Nothing to authenticate with |
| A known placeholder (`change-me`, `postgres`, `your-api-key`, …) | Copied from a template |
| A value containing a development marker (`insecure`, `local-development`, `do-not-use`, …) | A development default that leaked |
| Shorter than 32 characters | Brute-forceable |

The marker check is not redundant with the length check. The platform's own
default JWT secret — `insecure-development-secret-do-not-use` — is 38
characters, so a length test alone would pass it. Every credential in
`docker-compose.dev.yml` trips one of these rules under
`FIE_ENVIRONMENT=production`, which is what lets the development stack stay
zero-setup without its credentials being able to reach a real deployment.

Outside production the check is a no-op, deliberately.

## Redaction

Credentials are redacted by a processor that runs on **every** log record, not
at call sites — a call site can be forgotten, a processor cannot. The same
applies to OpenTelemetry span attributes, which reach the same backends as logs.

Redaction is recursive across nested dicts and lists, and case-insensitive.
Covered keys include `password`, `api_key`, `anthropic_api_key`,
`authorization`, `token`, `refresh_token`, `client_secret`, `database_url`, and
`connection_string`.

The production OTel collector strips credential-shaped attributes again before
telemetry leaves the cluster — defence in depth against third-party
instrumentation that does not know about the application's redaction rules.

## Authentication

JWT with HS256 by default. Verification checks signature, expiry, issuer, and
audience, and requires the `exp`, `iat`, `sub`, `iss`, and `aud` claims to be
present.

**Access and refresh tokens are separated by a `typ` claim that is checked on
every decode.** Without that check, a long-lived refresh token would be accepted
as an access token, which defeats short access-token lifetimes entirely. This is
a common and serious flaw, so it is tested explicitly.

The `none` algorithm is rejected outright in production configuration, and a
forged unsigned token is covered by a test.

Clock-skew leeway is configurable (default 10s) so services on slightly
different clocks do not reject each other's valid tokens.

## Password handling

bcrypt with a per-password random salt, cost factor 12 by default.

bcrypt silently truncates input past 72 bytes, which means two different long
passwords sharing a 72-byte prefix would verify against each other. Rather than
inherit that, over-length input is rejected explicitly. The length check is
measured in UTF-8 **bytes**, not characters — multi-byte input reaches the byte
limit well before the character count suggests.

Verification returns `False` rather than raising on a malformed stored hash, so
a corrupted value is indistinguishable from a wrong password by error type.

## Authorization

RBAC with `product:resource:action` permissions and segment-aware wildcards
(`atlas:*` covers `atlas:research:read`; `atlas:*:read` does not cover
`atlas:research:write`).

**Tenant isolation is checked separately from permissions.** Cross-tenant reads
are the failure mode that matters most in a multi-tenant financial platform, and
folding tenancy into the permission string would make it easy to grant
accidentally. An admin still cannot cross tenants implicitly.

## Injection

- **Cypher** — parameters are always bound, never interpolated. Covered by an
  integration test that stores a hostile string and reads it back intact.
- **SQL** — SQLAlchemy parameter binding throughout.
- **Neo4j read path** — mutating clauses are rejected on `execute_read`. Read
  queries route to follower replicas in a cluster, where a mutation fails at
  runtime but succeeds against a single-node development instance; the guard
  catches it before it reaches production.

## Resource exhaustion

| Control | Where |
|---|---|
| Statement timeout | Postgres connections (default 30s) — one unbounded analytical query must not starve the transactional path |
| Connection pool bounds | `pool_size` + `max_overflow` |
| Per-attempt request timeout | `ResiliencePolicy` |
| Circuit breakers | Per dependency; stop retry storms against a dead service |
| TTL on every cache entry | `RedisClient.set_json` — unbounded entries turn Redis into an unmanaged database |
| TTL on every distributed lock | `RedisClient.lock` — a crashed holder cannot deadlock the system |
| Stream length cap | Event publishing (approximate trim) |
| Dead-letter after N attempts | One poison event cannot block a partition forever |

Redis in production runs with `maxmemory-policy noeviction`: eviction would
silently drop events from a stream, which is worse than refusing writes.

## Container and network posture

Production images run as a non-root user from a multi-stage build that leaves
the compiler toolchain behind. The internal Docker network has no egress; only
the gateway bridges outward, and databases are exposed on the internal network
rather than published to the host.

## CI security stages

| Stage | Tool |
|---|---|
| Dependency vulnerabilities | `pip-audit --strict` |
| Static security analysis | `bandit` (plus ruff's `S` ruleset inline) |
| Secret detection | `gitleaks` |
| Container image CVEs | `trivy`, failing on HIGH/CRITICAL |

Production promotion fails if a critical vulnerability exists, a secret is
detected, or authentication/authorization tests fail.

## Not yet implemented

Phase 9 (Production Hardening) still owns: OAuth flows beyond JWT, API rate
limiting middleware, audit logging, secrets-manager integration (currently
environment variables), key rotation, and disaster-recovery procedures.
