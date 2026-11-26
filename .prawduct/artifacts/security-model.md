---
artifact: security-model
version: 1
depends_on: []
last_validated: null
---

# Security Model

## Summary

3tears is a library family, not a service. It ships security primitives (RBAC evaluator, token and proof verifiers, secret references, an audit envelope, an SSRF guard) and the rules for composing them. The host app (a hub, an agent pod, a tool pod) owns deployment, key custody, network policy and which identities exist.

`handles_sensitive_data` is recorded as `credentials` (datasource passwords, API keys, NATS credentials) in `.prawduct/project-state.yaml`. The library holds credentials by reference, resolves them at use time and masks them in errors. It stores no end-user PII of its own; audit rows and agent checkpoints can carry it, and both have an erasure path.

Three facts shape the model:

- Authorization is evaluated live against grants. Tokens carry identity only.
- Every verifier pins its signature algorithm to EdDSA (or a literal list) and refuses the rest.
- Controls that depend on the host (network policy, key storage, who may mint tokens) are listed in "What the library does not do".

## Trust boundaries

| Boundary | Library side | Host side |
|---|---|---|
| NATS bus | Subject allow-lists per principal, auth-callout codecs and a generic responder loop (`packages/nats/src/threetears/nats/subject_permissions.py`, `auth_callout_responder.py`) | Runs the callout responder, supplies the `PrincipalResolver` and `GrantPolicy`, operates the NATS server and its system account |
| Identity | Verifies hub-signed tokens against a JWKS (`packages/core/src/threetears/core/security/identity_token.py`) | Holds the signing key in HSM or KMS; the library offers only a test keypair generator |
| L3 (PostgreSQL) and warehouses | Drivers, secret references, a refused-credential guard (`packages/datasources/src/threetears/datasources/drivers/connect_guard.py`) | Creates DB roles, network paths and the secret mounts |
| Outbound web | SSRF guard and robots gate in `packages/scrape/src/threetears/scrape/_private_hosts.py` and `robots.py` | Chooses the egress exit and runs the exit daemon |
| Browser sidecar | Documents its HTTP contract (`docs/sidecar-http-contract.md`) | Fronts port 8088 and decides who may be issued a session token |
| Release | Lockstep bounds and trusted publishing (`docs/releasing.md`) | Pins the whole family to one exact version in each consumer |

### What the library does not do

- It does not authenticate end users by itself. `packages/iam` supplies the primitives (passwords, OIDC, SAML, TOTP, WebAuthn, DPoP); a host service wires them (`packages/iam/src/threetears/iam/__init__.py`).
- It owns no IAM database schema. State sits behind Protocols in `packages/iam/src/threetears/iam/stores/`.
- It does not run the Tor, VPN or proxy daemon behind an egress driver (`packages/core/src/threetears/core/egress.py`).
- It does not store the hub signing key. Production custody is the operator's (`identity_token.py`).
- It does not authenticate callers of the browser sidecar. The sidecar "authenticates nobody"; the platform decides who holds a token (`docs/sidecar-http-contract.md`, section 4).
- It does not rate-limit sidecar routes (`docs/sidecar-http-contract.md`, section 5).
- It does not write per-tool role assignments. The deploying app creates them when a tool registers (`packages/agent/acl/src/threetears/agent/acl/builtin_roles.py`).
- It does not decide that a NATS deployment enforces auth. Subject permissions apply only once the host moves the bus off anonymous connections (`subject_permissions.py`).

## Authentication

Authentication is layered by hop. Each layer verifies a signed artifact and does not trust the message that carried it.

| Hop | Artifact | Source |
|---|---|---|
| Connection to NATS | Auth-callout request, decoded unverified because the system account is the trust boundary; the response is signed with the auth account key | `packages/nats/src/threetears/nats/auth_callout.py` |
| Caller to platform | Hub-issued EdDSA identity token; the algorithm is never read from the header | `packages/core/src/threetears/core/security/identity_token.py` |
| Agent to proxy | Proof of possession bound to the token (`ath`), the call (`bh`) and a single-use `jti` | `packages/core/src/threetears/core/security/pop.py` |
| Proxy to pod | Proxy-signed assertion binding identity, call body hash, nonce and target pod | `packages/core/src/threetears/core/security/proxy_assertion.py` |
| Tool pod acting as caller | Call-scoped delegated token (agent as `sub`, pod as `act`, pod key as `cnf`) plus a per-request proof | `delegated_token.py`, `door_request.py` in the same directory |
| Service to service | Signed envelope with issuer, door, operation, body hash and `jti` replay guard | `packages/core/src/threetears/core/security/service_envelope.py` |
| Hub to agent turn | Signed turn assertion; the router and agent take user, customer and conversation ids from it, never the body | `packages/core/src/threetears/core/security/turn_assertion.py` |

Session credentials in `packages/iam`:

- Session tokens assert identity only. They never carry a role, group or scope, so a revoked grant applies at once (`packages/iam/src/threetears/iam/tokens.py`).
- Passwords use argon2id with NFKC normalization; bcrypt verifies migrated hashes and upgrades on login (`passwords.py`).
- API keys are 256-bit random secrets hashed with SHA-256 for direct lookup (`apikeys.py`).
- Refresh tokens rotate. Presenting a rotated token revokes the session family, with a short replay-grace window that requires proof of the bound key (`rotation.py`).
- DPoP proofs are checked at token-minting endpoints only (`dpop.py`). Step-up re-authentication checks how long ago the human last proved identity (`stepup.py`).
- OAuth `state` is a signed, short-lived JWT with a single-use nonce consumed atomically (`oauth_state.py`).
- Breach-password screening is local by default; the remote range API is opt-in and fails open (`breach.py`).
- Client IP comes from `X-Forwarded-For` only when the direct peer is in a configured trusted-proxy CIDR; the default trusts nothing (`clientip.py`).
- NATS connection credentials expire after a day by default. Access is removed by kicking the connection and refusing the reconnect; a lost kick or partition leaves the credential live until expiry (`docs/nats-credential-renewal.md`).
- Door proofs roll out through dated windows. A window is capped at `MAX_PROOF_WINDOW`, and config can tighten a door but never loosen it past its date (`packages/core/src/threetears/core/security/door_proof.py`).

MCP servers resolve an identity per request: env-var identity on stdio, bearer token over Streamable HTTP, and default-deny dispatch (`packages/mcp/src/threetears/mcp/auth.py`, `http_server.py`).

## Authorization

The evaluator in `packages/agent/acl/src/threetears/agent/acl/evaluator.py` answers "can actor do action on namespace".

Model:

- The only grant principal is a group. Users and agents join groups; role assignments bind a group to a role at a scope (`types.py`).
- Scopes: one namespace, a namespace-name subtree, every namespace of one type in one customer, or all (`types.py`).
- A role is a map of resource type to actions; `*` is the wildcard bucket (`types.py`).
- When both a user and an agent act, the effective actions are the intersection of the two sides (`evaluator.py`).
- A namespace owner short-circuits to all actions. An unowned namespace (`owner_namespace` of `None`) matches nothing, by an explicit guard (`evaluator.py`).

Walls, each enforced independently in the evaluator:

| Wall | Rule | Source |
|---|---|---|
| Cross-customer group | A contributing group must be platform-scoped or share the namespace's customer, checked again at side resolution so a buggy loader cannot leak | `evaluator.py` |
| Role ownership | A customer-authored role contributes nothing against another customer's namespace, even through a platform-scoped group | `evaluator.py` |
| Delegation ceiling | An admin may write a grant only for actions they already hold at every namespace the grant would reach; an empty set of namespaces yields no authority | `packages/agent/acl/src/threetears/agent/acl/delegation.py` |
| Wildcard bucket | Refused for customer-authored roles | `delegation.py` |
| Impersonation | Tokens with `act_reason` of `impersonation` hit a deny-list overlay; a non-internal `aud` is refused, and a missing `aud` fails closed | `packages/agent/acl/src/threetears/agent/acl/authorize.py` |
| Workspace | A call without `customer_id` is unroutable; a cross-customer call is a categorical deny before the evaluator runs | `packages/agent/workspace/src/threetears/agent/workspace/authorize.py` |

Every resource-typed helper wraps one primitive, `authorize`, so a fix lands once (`authorize.py`). Row visibility for list endpoints is one shared SQL generator that encodes the same rules (`query_visibility.py`). Subtree grants admit no customer's rows there, deliberately. Both sets of rules apply to API-shaped surfaces: object-level checks happen per namespace, and list filtering happens in SQL rather than in the client.

Caches: a write evicts the writer's own cache at once, and other pods follow through the row broadcast and table generation (`write_eviction.py`, `generation_follow.py`). MCP tool grants reload on an epoch bump with a catch-up tick (`packages/mcp/src/threetears/mcp/rbac.py`).

## Secrets handling

- A secret is referenced as `scheme://locator` and resolved at use time. Shipped backends: `env://` (dev) and `k8s://` (projected secret files, default `/var/run/secrets/3tears`). `vault://`, `aws-secretsmanager://` and `gcp-sm://` are registered but unimplemented (`packages/core/src/threetears/core/security/secret_refs.py`).
- A secret that must be stored is sealed with AES-256-GCM under an HKDF-derived key from an operator master key. The master key is itself a secret reference and never sits beside the ciphertext. Opened values return as `SecretStr` (`packages/core/src/threetears/core/security/encryption.py`).
- Driver connect errors carry the server's SQLSTATE and message, mask the resolved password, and drop the backend exception chain (`packages/datasources/src/threetears/datasources/drivers/errors.py`).
- A refused warehouse credential pauses fleet-wide through L2 and L3 state, so replicas stop feeding the account lockout counter (`connect_guard.py`).
- A process with no warehouse credential queries through the hub on a forwarded identity. The token is a provider read on every call, never a captured string (`packages/datasources/src/threetears/datasources/query_client.py`).
- Tests that read a credential from the environment must carry the `secrets` marker, so the credential-less pool skips them visibly (`tests/enforcement/test_secrets_marker.py`).
- The sidecar's `complete` route returns a solved session's raw cookie jar, unsealed; the sidecar holds no key (`docs/sidecar-http-contract.md`, section 4).

## Data privacy

Classification: credentials by reference; user identifiers and free text in audit `details` and in agent checkpoints.

The erasure rule is anonymize, never delete (`packages/agent/audit/src/threetears/agent/audit/anonymize.py`):

- Every audit row stays. No id changes.
- `details` keep keys; values under a key not on the explicit safe list become the marker. An unclassified key is masked.
- `ip_address` goes through `anonymize_ip`.
- `PERSONAL_DETAIL_KEYS` records classified personal keys; `tests/enforcement/test_audit_details_keys_are_classified.py` holds producers to the classification.

Agent-to-hub erasure contract (`packages/agent/audit/src/threetears/agent/audit/erasure.py`):

- The hub derives the calling agent from the verified token, never the body, and answers `AGENT_MISMATCH` if the body differs.
- The match is `agent column = verified agent AND actor_user_id = ANY(ids)`. It is never widened.
- A tool pod may erase an owner's rows only with a WRITE grant on that owner's data in the pod's agent-data declaration.
- A retry is safe: matched rows already anonymized change nothing.

Agent checkpoints: only fields that identify a person are rewritten (a human message `name`, and keys in `IDENTIFYING_METADATA_KEYS`). Unknown keys are kept, the inverse of the audit rule, because the graph reloads and resumes from a checkpoint (`packages/langgraph/src/threetears/langgraph/anonymize.py`).

Audit delivery: `publish_audit` is durable JetStream, at-least-once, idempotent on envelope `id`. The producer swallows publish failure and logs at WARN, so a tool call never depends on audit availability. An audit event can therefore be lost when the producer cannot publish (`packages/agent/audit/src/threetears/agent/audit/publish.py`). The envelope rejects unknown fields and naive timestamps, and carries `acting_as_principal_id` for impersonation (`envelope.py`). Hub-less deployments persist through `persist.py`.

Retention periods and regulatory scope (GDPR and similar) are not set by the library; see Open questions.

## Abuse prevention

### Egress and outbound fetching

- `ScrapeTool` refuses non-public hosts by default (`block_private_hosts=True`). The check is an httpx request hook, so it runs on every redirect hop and every link a driver follows, not only the first URL (`packages/scrape/src/threetears/scrape/_private_hosts.py`, `tool.py`).
- `robots.txt` handling is on by default for both halves. `Crawl-delay` slows requests; `Disallow` escalates to a human instead of fetching (`packages/scrape/src/threetears/scrape/robots.py`).
- An egress driver describes an exit for both httpx and a browser. `health()` fetches an address-reporting endpoint through the exit, so a proxy that forwards directly is detected (`packages/core/src/threetears/core/egress.py`).
- Outbound upstream HTTP goes through one traced, retried, circuit-broken client (`packages/core/src/threetears/core/http_client.py`).
- The sidecar binds `127.0.0.1` by default (`BIND_HOST`). Compose sets `0.0.0.0` explicitly and annotates what that admits. Sensitive routes keep the session-token check (`docs/sidecar-http-contract.md`, section 4).
- The sidecar is AGPL and isolated in its own container; it imports nothing from `threetears` (`packages/scrape/src/threetears/scrape/operator.py`).

### Filesystem sandbox

- `PathSandbox` jails named roots with symlink-escape rejection and validates keys syntactically: empty, length, control characters, absolute paths and parent references (`packages/core/src/threetears/core/security/sandbox.py`).
- The workspace sandbox keeps only the jail and the syntax check. Authority for file access is the RBAC evaluator, with `read_file_matching:<glob>` and `write_file_matching:<glob>` actions (`packages/agent/workspace/src/threetears/agent/workspace/sandbox.py`).
- Writes pass declared validators between the sandbox and the transactional write (`validators.py`). A per-file lease serialises writers (`lease.py`).

### Bus isolation

- No principal gets bare `>` or the global `_INBOX.>`. Identity-bound subjects (per-pod, per-principal inbox) are the isolation unit (`packages/nats/src/threetears/nats/subject_permissions.py`).
- Hub requests that touch shared buckets compose the subject from the verified principal. A key is one literal subject with no wildcard or empty token, so a request cannot reach another principal's keys (`packages/nats/src/threetears/nats/collection_key_requests.py`).
- A missing KV grant looks like a broker timeout. Diagnostics convert the server's `permissions violation` into the missing grant (`packages/nats/src/threetears/nats/diagnostics.py`).

### LLM and agent surfaces

- Tool calls run under the verified identity from the proxy assertion, not the envelope's claimed identity; a per-call `CallContext` scope carries conversation, user, customer and agent ids (`packages/agent/tools/src/threetears/agent/tools/call_scope.py`).
- Authorization is checked per resource inside each tool (workspace tools call `authorize_workspace_access` first). A model's output cannot widen a grant, because grants are rows evaluated live.
- The delegation ceiling stops an agent-reachable admin path from minting more authority than the author holds (`delegation.py`).
- The sandbox, RBAC glob actions and SSRF guard bound what a tool the model drives can read and fetch.
- Scrape escalates to a human on a bot wall or `Disallow` rather than evading (`robots.py`, `operator.py`).

The code contains no prompt-injection filter and no content-trust labelling of tool output or retrieved memory. See Open questions.

## Supply chain

- Every `3tears*` package releases at one version; every intra-family dependency is bounded `>=X.Y.0,<X.(Y+1).0`. Unbounded siblings allowed a mixed family that built clean and failed at runtime (`CLAUDE.md`, `docs/releasing.md`). The bounds are enforced by `tests/enforcement/test_intra_family_version_bounds.py`; API growth requires a minor bump (`tests/enforcement/test_api_growth_requires_a_minor_bump.py`).
- PyPI and npm publish through trusted publishing (OIDC, `id-token: write`) in `.github/workflows/release.yml`. PyPI is gated by the manual-approval `pypi` environment. npm publishes with provenance and is gated by the `NPM_PUBLISH` repository variable (`docs/releasing.md`).
- A release is a pushed tag from `main`. A manual dispatch republishes an already-tagged version and creates no release (`docs/releasing.md`).
- A release that adds a package needs its pending trusted publisher registered first; nothing checks this, and a miss leaves a family that cannot install (`docs/releasing.md`).
- A failing live Claude CLI batch blocks publish (`docs/releasing.md`, step 4).
- JWT verification sites are listed in `tests/enforcement/test_jwt_alg_pinning.py`. Redshift private-attribute access is confined to one module (`packages/datasources/src/threetears/datasources/drivers/_redshift_connector_internals.py`).
- `uv.lock` is committed. Release tooling runs `uv run --frozen` (`scripts/bump-version.sh`).
- Commits carry no agent attribution, checked in CI by `scripts/check-attribution.sh` and `tests/enforcement/test_no_agent_attribution.py`.
- Consumers pin the whole family to one exact version (`CLAUDE.md`).

## Direction

Ratified 2026-11-26 from the test-backed candidates below, which the owner approved in bulk once
duplicates were removed ("Yes to all recommendations"). The candidates with no gate await the
owner's ruling one at a time and are not ratified.

- **A JWS is verified with a pinned algorithm: reject on the declared header algorithm before a
  key is selected, then pass a literal single-element list to `decode`.**
  Why: reading the algorithm off the token admits `alg: none` and the asymmetric-confusion
  downgrade. Gate: `tests/enforcement/test_jwt_alg_pinning.py`.
- **Every audit `details` key a package publishes is classified as safe or personal.**
  Why: erasure masks unclassified keys on its own; the gate keeps the classification from falling
  behind the code, so no key is masked or kept without a decision. Gate:
  `tests/enforcement/test_audit_details_keys_are_classified.py`.

## Candidate norms (unratified)

- Verify JWS with a pinned algorithm; never read it from the header. `tests/enforcement/test_jwt_alg_pinning.py`
- A session token carries identity, never roles or permissions. `packages/iam/src/threetears/iam/tokens.py` (design rule; no gate found)
- An admin may write a grant only for actions they hold at every scope it reaches. `packages/agent/acl/src/threetears/agent/acl/delegation.py`
- An unowned namespace matches no caller. `packages/agent/acl/src/threetears/agent/acl/evaluator.py`
- Erasure anonymizes audit rows and never deletes them; unclassified keys mask. `packages/agent/audit/src/threetears/agent/audit/anonymize.py`, `tests/enforcement/test_audit_details_keys_are_classified.py`
- A tool pod's identity comes from a verified token, never the request body. `packages/agent/audit/src/threetears/agent/audit/erasure.py`
- The SSRF check runs on every request hop. `packages/scrape/src/threetears/scrape/_private_hosts.py`
- Secrets are referenced, never held by value in config. `packages/core/src/threetears/core/security/secret_refs.py`
- A test that reads a credential carries the `secrets` marker. `tests/enforcement/test_secrets_marker.py`
- A broad exception handler re-raises or calls `contain()`. `tests/enforcement/test_no_silent_swallow.py`
- Every intra-family dependency carries the lockstep bound. `tests/enforcement/test_intra_family_version_bounds.py`
- No agent attribution on any surface. `tests/enforcement/test_no_agent_attribution.py`

## Open questions

1. Prompt injection: no code control was found for untrusted tool output, scraped pages or retrieved memory entering a model context. Is this intentionally the host's job?
2. The sidecar access posture in `docs/sidecar-http-contract.md` section 4 is headed "RECOMMENDATION, needs a ruling", though its points 1 to 3 read as decided. Which is current?
3. Does the tool server itself verify the proxy assertion and identity token on every dispatch? `packages/agent/tools/src/threetears/agent/tools/server.py` was not traced; the assertion module states the pod verifies it.
4. Where are the deny-list rules for impersonation defined? `authorize.py` references an `ImpersonationCategory` from `threetears.interfaces.access`, which was not read.
5. Retention and regulatory scope (GDPR, others) are not stated anywhere read. Is there a retention rule for audit rows and checkpoints?
6. Is there dependency vulnerability scanning (pip-audit, Dependabot) or SHA-pinned GitHub Actions? None was found in `.github/workflows/ci.yml`, and no `.github/dependabot.yml` exists.
7. Rate limiting: `packages/iam/src/threetears/iam/stores/attempt_limiter.py` exists but was not read. Which login and API paths are covered?
8. Is NATS auth enforced in every production deployment, or is anonymous mode still in use anywhere? `subject_permissions.py` describes the move as pending.
9. `project-state.yaml` records `risk_profile.overall: null`. What proportionality level should this model target?
