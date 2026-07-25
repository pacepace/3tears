# scrape-task-08: Human-in-the-loop render sessions (VNC) + fetch-side health learning

**Status:** DESIGN, not yet approved to build -- presenting for review before any code, per this
project's own "Sequence strictly" precedent (scrape-task-01 through 04 each got explicit
go-ahead before implementation started).

**Driver:** some targets sit behind a bot wall (Cloudflare interstitial, a captcha, a
"verify you are human" gate) that no unattended fetch will ever pass. A human can pass it in
seconds. Today there is no path for that human, and -- separately, and worse -- a target that
starts getting blocked accumulates no memory of it at all, so the system neither learns nor
recovers.

**Scope note:** the two problems are presented together because they share one mechanism. A
human path is useless without knowing *when* to summon a human, and knowing when to summon a
human is the same question as "did this target change, break, or get blocked" -- which the
package currently cannot answer.

---

## The requirement

1. **A human must be able to take over a real browser session**, clear a challenge, and hand
   control back so the scrape completes -- over VNC, with a JavaScript client in the operator's
   browser, against the display the sidecar container already runs.
2. **A scrape that needs a human must say so** rather than failing indistinguishably from every
   other error, so an external system can queue it.
3. **A human's work must be reusable.** One solve should cover subsequent polls of that target,
   not be re-paid on every fetch.
4. **Failures must feed the existing self-healing loop**, so being blocked becomes a signal the
   system learns from and minimises, not an error that repeats forever.
5. **Opening a session must be authorization-gated**, per queue, using the platform's existing
   RBAC -- 3tears does not own the policy, only consumes it.

---

## Verified, not assumed

Read directly from the code this session, not recalled:

**The container already has everything VNC needs except VNC.** `sidecar/entrypoint.sh` starts
Xvfb on `:99` at `1920x1080x24`; `sidecar/main.py`'s `_lifespan` launches nodriver with
`headless=False` against it. Chromium is genuinely headful on a real X display today. Missing:
`x11vnc`, `websockify`, noVNC static assets.

**One browser, one display, one shared profile.** `main.py` holds a single global `_browser`
with a pinned `user_data_dir`, opening a throwaway tab per request (`new_tab=True`) and closing
it after. Every concurrent render paints onto the same X root window. Consequence: a naive
"point VNC at `:99`" exposes every concurrently-rendering target's tab to whoever is connected,
with full mouse and keyboard. `_create_isolated_tab` already exists (used by `/v1/download`) and
creates a real isolated `BrowserContext` -- the primitive needed to fix this is already in the
file.

**There is no fetch-side learning anywhere.** `run_eval_loop` takes `html` -- it begins *after* a
successful render. The render happens in the consumer, which resolves a driver and calls
`render()` itself. A driver exception never reaches the eval loop. `ScrapeRecipe` carries
`won_at`, `last_validated_at`, and `consecutive_validation_failures`, and that counter is
incremented in exactly two places (`_reuse_recipe`, `_reuse_row_recipe`) -- both on *selector*
misses. Nothing anywhere records that a fetch failed.

**Being blocked currently destroys a working recipe.** This is live behaviour today, not a
hypothetical. A bot wall returns HTML -- the wrong HTML. `_reuse_recipe` runs the stored
selectors against the interstitial, they don't match, `validation_status` is set to `"failed"`,
and `consecutive_validation_failures` increments exactly as if the site had been redesigned.
Three polls later the target crosses `DEFAULT_FAILURE_THRESHOLD` and spends a full
candidate-generation-plus-judge round -- several LLM calls -- learning how to extract data from a
Cloudflare challenge page. The winning "recipe" is then whatever selectors best fit an
interstitial. Getting blocked costs money and corrupts the target's memory.

**Nothing can distinguish the three failure modes.** "The site changed", "our recipe is wrong",
and "we got blocked" share one counter and one response. There is no content fingerprint stored
anywhere in the package, so no comparison is even possible.

---

## What already exists -- reuse inventory

The single biggest risk on this task is writing bespoke versions of primitives this monorepo
already ships. Surveyed before designing; each row is a thing we are **not** writing.

| Need | Existing primitive | How it's used here |
|---|---|---|
| Failure-state machine (healthy / tripped / probing) | `threetears.models.circuit_breaker.CircuitBreaker` -- CLOSED/OPEN/HALF_OPEN, consecutive-failure threshold, recovery timeout, single-probe admission | Vocabulary and semantics adopted wholesale. See "Circuit state" below for why the *instance* isn't reused directly |
| Depending on a breaker without dragging in LangChain | `core/http_client.py`'s `CircuitBreakerLike` structural protocol -- `core` deliberately injects rather than importing `threetears.models` | Same seam, same reason: `scrape` takes an injected breaker through the existing protocol |
| Cross-pod attempt counting | `core.coordination.windowed_counter.WindowedCounter`, `distributed_counter.DistributedCounter` | Blocked-attempt counting across a multi-pod fleet, instead of a per-process integer |
| Retry pacing / not hammering a wall | `core.coordination.token_bucket.TokenBucket` | Paces re-attempts against a target known to be challenging |
| "Only one pod opens a session for this target" | `core.coordination.lease.KVLease` -- TTL-bounded distributed mutex | Session exclusivity and fencing |
| "Has this already been done" | `core.coordination.idempotency.IdempotencyKeyStore` | A completed HITL solve is claimed once, not replayed |
| Deferred re-attempt ("re-check in 6h") | `3tears-scheduled-jobs` -- payload-agnostic, cross-pod-locked tick engine, `relative_delay` / `one_shot_at` schedule types with missed-fire policy | Scheduling a blocked target's next probe. No bespoke retry loop |
| Inbound callback from an external system | `3tears-agent-wake` -- webhook subscriptions, `hmac_util`, `webhook_adapter`, `dispatch` | The external queue telling us "this one's cleared" |
| Storing credentials at rest | `core.security.encryption` -- AES-256-GCM under an operator master key via HKDF, `seal()` / `open_secret()`, master key resolved through `core.security.secret_refs` | Solved-session cookies are credentials. Sealed, never plaintext |
| Per-resource authorization | `threetears.agent.acl.authorize_on_entity` + `AclCache`, following the `memory/authorize.py`, `identity/authorize.py`, `intention/authorize.py` shape (action constants + namespace + package-specific `AccessDenied`) | Gating HITL session open, per queue |
| Audit trail | `3tears-agent-audit` -- one `AuditEvent` envelope + `publish_audit`, consumed platform-side into `platform_audit.audit_events` | A human driving an authenticated browser is an audit event |
| Event publication | `3tears-nats` `Subjects` builders + `subject_permissions` | Announcing "this target needs a human" |
| "Paused for a human" vocabulary | `threetears.langgraph.streaming` -- `detect_interrupt`, `StreamInterruptEvent`, `tool_status='interrupted'` ("not a failure, the graph is pausing for a human decision") | The platform already has a word for this state. Mirror it rather than coining a parallel one |
| Traced/retried/circuit-broken HTTP | `core.http_client.TracedHttpClient` | Sidecar-facing calls, replacing raw `httpx` use where practical |
| Isolated browser context | `sidecar/main.py`'s existing `_create_isolated_tab` | Per-target isolation inside one HITL session |
| Page-text normalisation for comparison | `extraction.strip_boilerplate` | Input to the content fingerprint |

**Genuinely new, because nothing covers it:** challenge detection from a rendered page; the
sidecar's VNC/session endpoints; the HITL session state machine; the fetch-health columns.
Everything else composes what's listed above.

---

## Design

### 1. Blocked becomes an outcome -- classified on failure, not pattern-matched

**A marker list is the wrong mechanism and is rejected as the authority.** "Match Cloudflare's
current interstitial markup" is a hand-written parser for one vendor's page as it looks this
week. Vendors reword and restyle these pages, and a fixture set captured today is a snapshot,
not a specification. Building detection on a marker list reintroduces exactly the brittleness
this package was built to eliminate, in the one place it would fail silently -- a rotted marker
means a blocked target is misread as "the site changed" and burns its recipe, which is the bug
being fixed.

**Instead, detection reuses the machinery that already handles "our understanding of this page
stopped working": ask, once, at the moment of failure.**

The trigger is deterministic and free -- the page failed to yield data. Two branches that already
exist in `eval_loop.py` are the hooks:

- `_reuse_recipe` / `_reuse_row_recipe` -- the stored strategy validated against a fresh page and
  didn't match
- `_regenerate_recipe` / `_regenerate_row_recipe` -- the "no structurally-valid candidates"
  branch, already logged and handled today

Both currently assume one cause (the page changed) and respond one way. Instead, they ask a
classification question first:

> Here is the page, and the field schema we expected to find on it. Is this the content we
> wanted, a changed version of it, a bot/human-verification wall, or something else?

Returning a `PageVerdict` (`kind`: `content` / `changed` / `blocked` / `empty` / `other`, plus
`evidence` and `confidence`), through `llm_retry.bounded_retry_structured_call` with a forced
response shape -- the same reliability posture and the same helper `_judge_candidates` and every
extraction call already use. New module `challenge.py` holds the prompt, the verdict model, and
the routing; nothing about it is vendor-specific, so a wall it has never seen classifies on
meaning rather than markup.

**Cost: this is not an added call.** It replaces a blind regeneration we were already paying for.
On the reuse-failure path it runs *instead of* proceeding toward candidate generation, and
regeneration only follows if the verdict says the page actually changed. A blocked target
currently costs a full candidate-generation-plus-judge round; under this design it costs one
classification and no regeneration at all -- strictly cheaper than today.

**The fingerprint stops it repeating.** The verdict is stored with the fingerprint of the page it
judged (§2). Same fingerprint next poll, same verdict, no call. A target walled for a week costs
one classification, not seven.

**Markers, if used at all, are a fast path and never the authority.** A cheap obvious-case check
may skip the LLM call, but it is only ever allowed to *shortcut* to the same verdict the
classifier would reach, is never consulted to rule a wall *out*, and is expected to rot
harmlessly -- when it stops matching, the classifier simply runs, as it always would have.

Routing, once classified:

- `blocked` → persist `ScrapeExtraction` with `validation_status="blocked"` and the verdict's
  evidence in `field_confidences`; **the recipe is untouched** -- no counter increment, no
  regeneration; record fetch-side health (§2) and surface for HITL
- `changed` → regenerate now (§2)
- `content` / `empty` / `other` → today's behaviour: count the failure, no regeneration below
  threshold

That fixes the recipe-destruction bug without a single vendor string in the codebase.

Both eval-loop entry points gain an optional `page_status: int | None = None` (defaulted, so
every existing caller is unaffected) since a status code is real evidence for the classifier;
`challenge.py` stays independently callable for a consumer that wants to ask earlier.

`validation_status` gains `"blocked"` alongside `validated` / `needs_review` / `failed`. Note for
review: the README already records that no consumer currently filters on `validation_status` at
all, so this new value is inert until one does -- a pre-existing gap, out of scope here, but a
fourth value widens it slightly.

### 2. Fetch-side health, and a fingerprint to tell failures apart

A new `ScrapeTargetHealth` entity carries:

| Column | Purpose |
|---|---|
| `content_fingerprint` | sha256 of `strip_boilerplate(html)`, normalised -- captured whenever a recipe validates |
| `consecutive_fetch_failures` | fetch-stage failures (blocked, transport, timeout) -- deliberately separate from the extraction counter |
| `circuit_state` | `closed` / `open` / `half_open`, the `CircuitBreaker` vocabulary |
| `blocked_until` | when the next probe is permitted |
| `last_blocked_at`, `last_block_kind` | evidence for the operator and for tuning detection |
| `session_state_sealed`, `session_state_expires_at` | §4 |

**Decided: its own entity, not the recipe row.** `ScrapeTargetHealth` / `ScrapeTargetHealthCollection`,
keyed by `target_id`, table `scrape_target_health`, alongside the existing three. The columns above
live there, and `ScrapeRecipe` is left exactly as it is.

The rejected alternative was folding these onto `ScrapeRecipe`, which is already one row per target
and already read on every poll, so it would have saved a lookup. It was rejected because the entity
means "the winning extraction strategy" and half its columns would then be fetch health, and because
a target blocked before it ever won a recipe would have to create a strategy-less recipe row. That
in turn would have forced a new correctness requirement on `run_eval_loop`'s reuse branch, which today
tests only `consecutive_validation_failures < failure_threshold` and would happily try to reuse an
empty `{}` strategy. A separate entity makes that whole failure mode impossible rather than guarded:
health for a target that has never had a recipe is simply a health row with no recipe row, which is
an honest description of the state.

Cost, stated plainly: one extra lookup per poll, and a fourth table with its own migration and
collection. The lookup is L1-cached like every other three-tier read, so the cost is a cache hit in
the common case.

**The fingerprint is what makes the three failure modes separable:**

| Fetch | Extraction | Fingerprint vs. last-validated | Meaning | Response |
|---|---|---|---|---|
| ok | valid | -- | healthy | reuse, reset counters, re-stamp fingerprint |
| challenge detected | not attempted | -- | blocked | HITL path; recipe untouched |
| ok | invalid | **changed** | the site changed | regenerate **immediately** -- waiting three polls is pure latency when we have positive evidence |
| ok | invalid | unchanged | our recipe is wrong, or the data genuinely isn't there | count it, but don't spend LLM calls re-deriving against an identical page |
| transport error | not attempted | -- | transient | fetch-failure counter, backoff, no recipe change |

Rows 3 and 4 are a real improvement independent of HITL: faster healing when the page actually
changed, and no repeated candidate generation against a page that demonstrably has not.

### 3. Circuit state, and why the existing breaker isn't instantiated directly

`models.circuit_breaker.CircuitBreaker` is the right state machine and its three states,
threshold, recovery timeout and single-probe admission are adopted exactly. It is not used as
the durable store because it is in-memory and `threading.Lock`-based -- process-local by
construction. A blocked target must stay blocked across pods and restarts, so the durable state
lives in the recipe row, with `WindowedCounter` / `DistributedCounter` for the cross-pod counts
and `TokenBucket` for probe pacing.

Where a per-process fast-fail is still wanted, `scrape` accepts an injected breaker through
`core.http_client`'s existing `CircuitBreakerLike` protocol -- the same seam `core` already uses
to avoid importing `threetears.models` and its LangChain weight. No new protocol.

Re-probing a blocked target is scheduled through `3tears-scheduled-jobs` (`relative_delay`),
not a bespoke sleep-and-retry.

### 4. Reusing the human's work

On a successful human solve, the sidecar exports that browser context's cookies and storage
state. The MIT package seals it with `core.security.encryption.seal()` and stores it on the
recipe row with an expiry. Subsequent unattended renders pass it back so the session resumes
already-cleared.

These are session credentials. Sealed at rest under an operator-supplied master key resolved via
`secret_refs`, never written plaintext, never logged, and excluded from any debug dump.

Driver contract gains `session_state: str | None = None` on `render()`, following the
established "accept the full signature, use what you need" convention already set by
`link_selector` / `results_path` / `seen_urls` -- every other backend accepts and ignores it. The
sidecar applies it to the isolated context before navigating.

**Tension with the earlier per-target-profile idea, resolved:** one Chromium process has exactly
one `user_data_dir`, so per-target *profiles* cannot coexist inside one shared browser. Per-target
*isolated browser contexts* plus exported cookie/storage state achieves the same isolation and the
same reuse, and uses machinery `main.py` already has. What it loses is whole-profile fidelity
(service workers, some fingerprint continuity); if a challenge system proves to key on that, the
fallback is a dedicated browser for that one target.

### 5. The HITL session (sidecar)

Container additions: `x11vnc`, `websockify`, noVNC static assets. Xvfb, Chromium and the headful
launch are already there.

`x11vnc` and `websockify` start **on demand** when a session opens and stop on teardown -- no idle
VNC surface. This matches the operational model: the display comes up when a person arrives, not
before.

New sidecar endpoints (the sidecar remains a dumb browser-as-a-service -- no 3tears imports, the
AGPL boundary is unchanged):

| Endpoint | Purpose |
|---|---|
| `POST /v1/hitl/session` | Create a session, start VNC, return `{session_id, vnc_path, token, expires_at}` |
| `GET /v1/hitl/session/{id}` | Session state and open tabs |
| `POST /v1/hitl/session/{id}/tab` | Bring one target into the session: isolated context, navigate, replay `nav_steps` |
| `POST /v1/hitl/session/{id}/tab/{tab}/complete` | Human says cleared: verify, export sealed state, close the tab, free the slot |
| `DELETE /v1/hitl/session/{id}` | Teardown, stop VNC, drop contexts |
| `GET /vnc/{token}` | The noVNC client itself |

**Bounded working set.** A session has a fixed slot count. A target occupies a slot from
`/tab` until `/complete`; backgrounding a slow one still holds its slot. Items are pulled in as
slots free. No unbounded tab growth.

**Nothing is held while waiting.** A target that needs a human is not parked in a live browser.
It is reported back and forgotten; the session re-drives it from `url` + `nav_steps` when an
operator actually arrives. Waiting therefore costs zero container resources, and re-driving is
deterministic because `nav_steps` replay is already how this package reaches gated pages.

**Security.** The session token is unguessable, short-lived, scoped to one session and bound to
its TTL. `x11vnc` binds loopback only; `websockify` is the sole path in. A session has a hard TTL
with a reaper. The sidecar never authenticates a human -- it honours a token that the MIT side
minted only after authorizing the request.

**Concurrency, stated honestly.** One Xvfb display means one operator session at a time; a second
request queues or is refused. Multiple concurrent operators need a display pool
(`:100`, `:101`, …), each with its own Chromium and `x11vnc`. The display number is a parameter
from the start so the pool is a later configuration change rather than a rewrite, but v1 is
single-display.

### 6. Authorization, audit, announcement

Authorization lives in the MIT package (`hitl/authorize.py`), never the sidecar, which cannot
import 3tears. It mirrors `memory/authorize.py` exactly: action constants
(`scrape.hitl.session.open`, `scrape.hitl.session.view`), a namespace per queue, a package-specific
`HitlAccessDenied`, evaluated through `authorize_on_entity` with `AclCache`. 3tears ships the
evaluator; the deployment supplies roles and assignments.

Every session open, tab open, complete, and teardown publishes an `AuditEvent` via
`publish_audit` -- a human driving a browser holding a target's authenticated session is exactly
what the unified audit trail is for.

"This target needs a human" is published over NATS using the existing `Subjects` builders. That
is the whole integration surface with the queue.

---

## Explicitly out of scope

- **The queue itself.** External. We return a HITL outcome and publish an event; something else
  decides what to do with it.
- **Operator UI beyond the noVNC client.** The session's tab strip is the working view for v1.
- **Who may work which queue.** Policy is deployment-supplied RBAC data, not code here.
- **Automated challenge solving.** Not attempted, not wanted.
- **Multi-display concurrency.** Parameterised, not implemented.
- **Consumers reading `validation_status`.** Pre-existing gap, noted, unchanged.

---

## Files to create / modify

**Create**
- `packages/scrape/src/threetears/scrape/challenge.py` -- `detect_challenge`, `ChallengeSignal`
- `packages/scrape/src/threetears/scrape/hitl/authorize.py` -- RBAC gate
- `packages/scrape/src/threetears/scrape/hitl/session.py` -- session client + state machine
- `packages/scrape/sidecar/hitl.py` -- session endpoints, VNC lifecycle
- `packages/scrape/sidecar/static/` -- noVNC assets
- tests alongside each

**Modify**
- `eval_loop.py` -- challenge short-circuit, fingerprint routing, fetch-health updates
- `collections.py` -- new `ScrapeTargetHealth` entity + `ScrapeTargetHealthCollection`; `ScrapeRecipe` untouched
- `migrations.py` -- `v010` creates `scrape_target_health` (fetch health, fingerprint, sealed session state)
- `driver.py` + all 8 drivers -- `session_state` parameter (accept-and-ignore except the browser backends)
- `sidecar/Dockerfile`, `entrypoint.sh` -- `x11vnc`, `websockify`, noVNC
- `tests/test_migrations_drift.py` -- already introspection-based as of the current fix branch, so it picks up the new columns automatically
- `packages/scrape/README.md`

---

## Anti-patterns to avoid

- Writing a new circuit breaker, retry loop, distributed counter, lease, scheduler, audit
  envelope, encryption helper, or RBAC evaluator. All exist. See the reuse inventory.
- Letting a blocked fetch touch `consecutive_validation_failures`. That is the bug being fixed.
- Inferring a challenge from a healthy-looking page. Detection reacts to failure evidence.
- Storing cookies unsealed, or logging them.
- Exposing the raw X display over VNC without per-session isolation.
- A HITL branch in the eval loop that duplicates the extraction path. One short-circuit, then the
  existing flow, unchanged.
- Target-specific logic anywhere -- the same test `request_shape_finder.py` states: would this help
  a different, unrelated target of the same class?

---

## Acceptance criteria

1. A rendered Cloudflare-style interstitial produces `validation_status="blocked"` and leaves
   `consecutive_validation_failures` and `extraction_strategy` **byte-identical** to their prior
   values. Regression test asserts the recipe is untouched.
2. A page whose fingerprint changed and whose recipe fails regenerates on the **first** failure,
   not the third.
3. A page whose fingerprint is unchanged and whose recipe fails does **not** trigger candidate
   generation before the threshold.
4. A target with health but no recipe (blocked before it ever extracted successfully) is a health
   row with no recipe row. No strategy-less `ScrapeRecipe` is ever written, and `run_eval_loop`'s
   reuse branch is unchanged.
5. An operator with a granted role can open a session for a permitted queue; one without is
   denied with `HitlAccessDenied`, distinguishable from "nothing queued".
6. A completed solve yields sealed session state that a subsequent unattended render consumes to
   fetch the target successfully with no human involved.
7. Sealed state is unreadable without the master key; a tampered token is rejected.
8. Session teardown stops `x11vnc`/`websockify` and drops contexts; the TTL reaper collects an
   abandoned session.
9. `./scripts/check-all.sh` green; the introspection-based drift guard covers every new column.

---

## Open questions for review

**Resolved during review:**

- **Recipe row vs. a separate health entity** -- SETTLED: its own entity, `ScrapeTargetHealth`. See §2.
- **Challenge detection by fixtures** -- SETTLED, and it changed the mechanism. A marker set
  calibrated against real blocked pages is a snapshot: vendors reword these pages, so today's
  fixtures specify nothing about tomorrow. Detection is now a classification asked at the moment of
  failure and cached by fingerprint, with markers demoted to an optional fast path that is allowed
  to rot. See §1.

**Still open:**

1. **Session state TTL** -- how long is a solved session assumed good before a human is needed
   again? Probably per-target; a sensible default is unknown until observed in the wild.
2. **Whether `page_status` on the eval loop is enough**, or whether the loop should take the whole
   `RenderedPage`. The latter is cleaner and a wider breaking change.
