---
artifact: build-plan
version: 1
scope: iam-optional-core
branch: iam/optional-core
partition: "serial -- one builder; every chunk edits iam's pyproject or the dependency-alignment test, so a split would only merge-conflict"
last_validated: 2026-10-24
---

# Build plan: iam installs without core, NATS or the agent layer (#943)

## Goals

**Response taken:** proceed, citing it. The plan comment on #943 (the issue's agreed plan) and the
owner's rulings of 2026-10-24, relayed in the build brief, say what this is for, what success is,
and what is out of scope.

**Product:** a consumer outside the platform (testament's coordinator) takes `3tears-iam` for
GitHub sign-in, CI OIDC, session tokens and client-IP resolution without installing core,
NATS, Postgres drivers or the agent layer.

**Architecture that must hold:**
- iam keeps building on core rather than beside it: `dpop`, `CollectionAttemptLimiter` and
  TOTP seed sealing still use core's own code, now behind one `[core]` extra (`3tears[security]`).
  No forked seal: sealed seeds are stored data.
- `ImpersonationCategory` moves to a new dependency-free contracts leaf, `3tears-access-contracts`
  (`threetears.access.contracts`). agent-acl re-exports the same object, so identity and every
  existing import path hold.
- A missing extra fails by name (`3tears-iam[core]`), and only for core's own absence: a fault
  inside an installed core is never relabelled as a missing extra.

**Owner rulings (2026-10-24, binding):** the leaf, its name style and its contents (only what iam
and agent-acl share); TOTP defers the core import to `seal_seed`/`unseal_seed`, with
`DecryptionError` resolved by a module `__getattr__`; one `[core]` extra; `3tears-nats[client]`
leaves the hard deps and `[nats]` stays as the KV switch; agent-acl leaves iam's deps; dpop's
`ReplayGuard` import is TYPE_CHECKING-only; the guard shape (catch `ModuleNotFoundError` only for
`threetears.core*`, raise `ImportError` naming `3tears-iam[core]`), saml's guard fixed the same
way; the enforcement set below; no version bump.

**Constraints:** lockstep bounds `>=0.77.0,<0.78.0`; version stays 0.77.1; no agent attribution;
namespace dirs `threetears/` and `threetears/access/` carry no `__init__.py`.

**Tradeoffs accepted:** a star import of `threetears.iam.totp` on the base install raises, because
`DecryptionError` stays in `__all__` (the mcp precedent keeps lazy names in `__all__` too).
API-growth enforcement goes red for the new leaf's names until the maintainer's MINOR bump.

**Inferred:** the leaf lives at `packages/access-contracts/` beside `packages/media-contracts/`;
`threetears.access` collides with nothing in the workspace.

**Decisions made mid-build:** (appended while building)

**Level:** High.

## Shared-vocabulary inventory (iam x agent-acl)

Shared, so moved: `ImpersonationCategory` (iam's `stepup` imports it; agent-acl declares it).
Not shared, so left: agent-acl's `INTERNAL_AUDIENCE`, the `act_reason` value `"impersonation"`,
`ActorType`/`MemberType`/`ScopeType`/`LimitingSide` (agent-acl only; iam carries `act_reason` and
`aud` as free strings); iam's `TokenType`, `ConnectionScope`, `ConnectionFieldKind` (iam only).

## Chunks

### 01 -- the contracts leaf  [ ]
`packages/access-contracts`: pyproject (hatchling, 0.77.1, no deps), `py.typed`, README, LICENSE,
`ImpersonationCategory`; agent-acl declares the leaf and re-exports the same object. Wiring: root
README table row and diagram, docs/adoption page and index row, root pyproject (exclude-newer,
mypy_path, files), `scripts/typecheck.sh`, `contract_packages` in test_dependency_alignment and
test_imports_declared, agent-acl's recorded floor, uv.lock.
Tests: leaf surface test (members and values pinned); agent-acl identity test
(`threetears.agent.acl.ImpersonationCategory is threetears.access.contracts.ImpersonationCategory`).

### 02 -- iam's optional core  [ ]
pyproject: hard deps lose `3tears[security]`, `3tears-agent-acl`, `3tears-nats[client]`, gain
`3tears-access-contracts`; extras `core = ["3tears[security]..."]`, `nats = ["3tears-nats[client]..."]`.
`_core_extra.py` guard helper; dpop and stores.attempt_limiter guarded; ReplayGuard and
CollectionRegistry TYPE_CHECKING-only; totp's core import deferred, `DecryptionError` via
`__getattr__`; saml's guard narrowed to its own modules; stepup imports the leaf.
Docs: stale pyproject comment and README claims fixed; CHANGELOG names the break and the fix.
Tests: `test_imports_without_core.py` (meta-path finder refusing core, nats, the agent layer,
sqlalchemy, asyncpg, nats-py, nkeys: base modules import; token mint/verify, TOTP and a
MemoryAttemptLimiter round trip run; dpop, attempt_limiter and seal_seed raise naming
`3tears-iam[core]`; a fault inside core is not relabelled); `test_import_cost.py` (allowlist
probe per base module); `test_package_boundaries.py` (AST: no module-level core/agent/nats import
outside the named `[core]` modules); iam moved into `_DEPENDENCY_FLOORS` with its rationale.

### 03 -- testament's three asks: device flow, memory stores, PKCE  [ ]
Owner-approved scope addition (coordinator, 2026-10-24), built after 01-02 are committed; the
last #943 comment states what was promised.
- **A. Device flow (RFC 8628)** in `iam.github`: a device-code request (client_id, scopes) returning
  a typed value (device_code, user_code, verification_uri, expires_in, interval); a poll that
  honours the interval, adds 5 s on `slow_down`, stops at `expires_in`, and maps
  `authorization_pending`, `slow_down`, `expired_token`, `access_denied`, `device_flow_disabled`
  to typed outcomes or errors, never `None`. The token goes through the existing `fetch_profile`.
  On the `GithubOAuth2Client` protocol and the httpx client alike; parity fakes kept in step.
- **B. `stores.memory`**: expired entries swept on write (amortised, ordered by expiry); optional
  `max_entries` cap refusing new entries with a typed error; redemption semantics and the
  injectable clock unchanged; docstring states single process, one event loop, state lost on
  restart, not thread-safe.
- **C. PKCE**: `authorize_url(code_challenge=)` (S256 via `pkce`, sending
  `code_challenge_method=S256`), `exchange_code(code_verifier=)`; both optional because GitHub
  Enterprise Server 3.21 lacks them, documented as recommended.
Tests: httpx MockTransport like the existing github tests (device request, each poll outcome,
slow_down widening, expiry stop); memory stores issue, expire, issue again and see it swept;
the cap refusing; PKCE URL params and token request body.

## Verification

Ceiling: `TESTAMENT=off ./scripts/test.sh iam -q`, `... agent/acl -q`, `... access-contracts -q`,
lint, typecheck of iam, agent/acl and the leaf, and the touched enforcement files by `-k`.
Product check: wheels built and installed bare into a throwaway venv; `nats`, `nkeys`,
`sqlalchemy`, `asyncpg`, `threetears.core` absent; oidc, github, tokens and totp import and run;
`[core]` brings dpop back. The main session runs the pool gate and integration.

## Status

- [ ] 01 the contracts leaf
- [ ] 02 iam's optional core
- [ ] 03 device flow, memory stores, PKCE
