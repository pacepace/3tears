# scrape-task-08: Human-in-the-loop render sessions (VNC) + fetch-side health learning

**Status:** BUILT. Sections 1 through 8 are implemented and tested.

Sections 1 and 2 shipped first (`health.py` with `ScrapeTargetHealth` and the content
fingerprint, migration `v010`, `challenge.py`'s `PageVerdict` and `classify_failed_page`, the
eval-loop failure routing and the `ScrapeTool` opt-in). Section 3 is the durable fetch circuit
in `circuit.py`. Sections 4 through 6 are the sidecar's on-demand VNC, the HITL session with
isolated per-tab contexts, and the sealed session-state reuse in `session_state.py`. Sections
7 and 8 are `threetears.core.egress` and `robots.py`, both added after this document was first
written and both recorded here rather than only in a build plan that clones never receive.

Section 6's "RBAC, audit, announcement" was rewritten during the build and is the one part
NOT implemented, deliberately. Authorization needs an identity 3tears does not have, and the
operator conversation and audit trail belong to the platform that uses this library rather
than to the library. What replaced it is two seams a platform drives -- `list_walled()` to ask
what is stuck, and `record_human_cleared()` to say a person has fixed it -- with the loop
documented end to end in the package README.

A status line is the one place a reader goes to learn the state of this work, so it is worth
more care than the rest of the document: stale, it does not merely fail to inform, it asserts
something false. This family records the go-ahead here for the same reason (scrape-task-01
"APPROVED TO START", -02 and -03 naming the shipped predecessor).

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

Read directly from the code this session, not recalled. **This section is a snapshot of the
state BEFORE any of this was built** -- it names functions the strategy collapse has since
removed (`_reuse_recipe`, `_reuse_row_recipe`) and describes the recipe-destruction bug as live,
which section 2's classification routing has since fixed.
It is kept as the evidence the design rested on, not as a description of the code today:

**The container already has everything VNC needs except VNC.** `sidecar/entrypoint.sh` starts
Xvfb on `:99` at `1920x1080x24`; `sidecar/main.py`'s `_lifespan` launches nodriver with
`headless=False` against it. Chromium is genuinely headful on a real X display today. Missing:
`x11vnc`; the noVNC client ships in the MIT wheel.

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
| Depending on a breaker without dragging in LangChain | `core/http_client.py`'s `CircuitBreakerLike` structural protocol -- `core` deliberately injects rather than importing `threetears.models` | Same seam, reused: `scrape` takes an injected breaker, through `ProbeObservableBreaker` -- `CircuitBreakerLike` plus a readable `state`, which releasing a probe requires. See §3; the LangChain-weight argument is `core`'s reason for the original seam, not `scrape`'s, since `3tears-models` is already a hard dependency here |
| Cross-pod attempt counting | `core.coordination.windowed_counter.WindowedCounter`, `distributed_counter.DistributedCounter` | Blocked-attempt counting across a multi-pod fleet, instead of a per-process integer |
| Retry pacing / not hammering a wall | `core.coordination.token_bucket.TokenBucket` | Paces re-attempts against a target known to be challenging |
| "Only one pod opens a session for this target" | `core.coordination.lease.KVLease` -- TTL-bounded distributed mutex | Session exclusivity and fencing |
| "Has this already been done" | `core.coordination.idempotency.IdempotencyKeyStore` | A completed HITL solve is claimed once, not replayed |
| Deferred re-attempt ("re-check in 6h") | `3tears-scheduled-jobs` -- payload-agnostic, cross-pod-locked tick engine, `relative_delay` / `one_shot_at` schedule types with missed-fire policy | Scheduling a blocked target's next probe. No bespoke retry loop |
| Inbound callback from an external system | `3tears-agent-wake` -- webhook subscriptions, `hmac_util`, `webhook_adapter`, `dispatch` | The external queue telling us "this one's cleared" |
| Storing credentials at rest | `core.security.encryption` -- AES-256-GCM under an operator master key via HKDF, `seal()` / `open_secret()`, master key resolved through `core.security.secret_refs` | Solved-session cookies are credentials. Sealed, never plaintext |
| Per-resource authorization | `threetears.agent.acl.authorize_on_entity` + `AclCache`, following the `memory/authorize.py`, `identity/authorize.py`, `intention/authorize.py` shape | **Not used -- see §6.** Gating HITL access is the consuming platform's job; this package ships the health-row fact and the hub approval seams instead |
| Audit trail | `3tears-agent-audit` -- one `AuditEvent` envelope + `publish_audit`, consumed platform-side into `platform_audit.audit_events` | A human driving an authenticated browser is an audit event |
| Event publication | `3tears-nats` `Subjects` builders + `subject_permissions` | Announcing "this target needs a human" |
| "Paused for a human" vocabulary | `threetears.langgraph.streaming` -- `detect_interrupt`, `StreamInterruptEvent`, `tool_status='interrupted'` ("not a failure, the graph is pausing for a human decision") | The platform already has a word for this state. Mirror it rather than coining a parallel one |
| Traced/retried/circuit-broken HTTP | `core.http_client.TracedHttpClient` | Sidecar-facing calls, replacing raw `httpx` use where practical |
| Isolated browser context | `sidecar/main.py`'s existing `_create_isolated_tab` | Per-target isolation inside one HITL session |
| Page-text normalisation for comparison | `extraction.html_to_text` | Input to the content fingerprint. As designed this said `strip_boilerplate`; the shipped fingerprint uses `html_to_text` plus whitespace collapse, which is the readable-text extraction the comparison actually wants -- `strip_boilerplate` truncates for prompt budget, which would make the digest depend on where the truncation fell |

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

- `_run_reuse_cycle` -- the stored strategy validated against a fresh page and didn't match.
  One cycle now serves all four strategy shapes; it was four separate `_reuse_*` functions
  when this was written, and the collapse into `_StrategyShape` is what made the hook exist
  once instead of four times
- `_persist_no_survivors` -- the "no structurally-valid candidates" branch, already logged and
  handled today, likewise shared by all four regeneration shapes

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

**Cost, stated as a ledger rather than a slogan.** An earlier draft claimed "this is not an
added call". That does not survive contact with the actual paths, so the honest version:

| Situation | Today | Under this design |
|---|---|---|
| Recipe validates | 0 | 0 |
| Reuse fails, page identical to the last validated one | 0 | 0, the fingerprint comparison is free and settles it |
| Reuse fails, page differs, first time | 0 | **1** (the classification) |
| Reuse fails, page differs, later polls | 0 | 0, cached verdict |
| Blocked target whose wall page is byte-stable | a full candidate-generation-plus-judge round, and a destroyed recipe | 1 classification, and the recipe intact |
| Blocked target whose wall page carries a per-request token | the same destroyed recipe | 1 classification **per poll**, and the recipe intact |
| Genuinely changed target | generation+judge on the third failing poll | 1 classification + generation+judge on the *first* |

So one classification is genuinely added at the moment a target's page first stops matching, and
it buys back the entire regeneration round on every blocked target plus two polls of latency on
every changed one. That is the trade, and it is worth making. Claiming it is free is not.

**The cache does not bound a blocked target's cost, and this section previously claimed it
did.** The fingerprint digests the page's readable text, and a real Cloudflare interstitial
renders a per-request Ray ID into exactly that text. So the fingerprint changes on every poll,
the cache never hits, and a walled target costs one classification per poll rather than one
ever. The claim was written before that was checked.

Three responses were considered. Normalising ids out of the fingerprint reintroduces
vendor-shaped pattern matching in the one place this design rejected it, and would silently
suppress genuine content changes that happen to look like ids. Fingerprinting structure rather
than text trades one brittleness for another. The response actually taken is **none, here**:
what bounds the cost of a walled target is not fetching it on every poll, which is
`blocked_until` and the circuit backoff in section 3. That is what makes the classification
rate a SEPARATE claim from the fetch rate rather than a restatement of it: the verdict cache
cannot bound classification here, so only the suppressed fetch can, and section 3's gate is
therefore load-bearing for both.

The same limit applies to a target walled before it ever won a recipe: it reaches
classification only after paying a full `generate_candidates` round, because there is no
stored strategy to fail fast. The same gate is the answer there too, and for the same
reason -- it sits in front of the fetch, so nothing downstream of the fetch runs at all.

**Three checks, cheapest first, and only the last one costs anything.** Classification is never
the first question asked on a failure:

1. **Is this page identical to the one the recipe last validated against?** (`content_fingerprint`.)
   If so, the page provably has not changed and provably is not a new wall, because a wall would
   have different content. Count the failure exactly as today and spend nothing. This is the free
   path, and it is the common one.
2. **Have we already classified this exact page?** (`classified_fingerprint`.) If so, reuse that
   verdict. This hits for a wall that renders the same bytes every time, and misses for one
   that stamps a per-request id into its text -- see the ledger above for why that is left to
   backoff rather than papered over here.
3. **Otherwise, ask.** One classification call, cached against the fingerprint of the page it judged.

A cache hit is also the answer to "we already regenerated against this page and it did not
stick": a repeat `changed` verdict for the same fingerprint routes to counting the failure, not
to regenerating again. Without that, an unlearnable page would burn a generation round every
single poll, which is strictly worse than the three-poll cadence it replaced.

**The fingerprint stops it repeating, where the page repeats.** The verdict is stored with the
fingerprint of the page it judged (§2). Same fingerprint next poll, same verdict, no call. A
wall that renders identically therefore costs one classification for as long as it stands; one
that stamps a per-request id into its visible text does not, and is bounded by backoff instead.

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

Section 3's fetch circuit later adds a fifth value to the tool's JSON payload, `"backoff"`, without
adding it to the `ValidationStatus` Literal. The Literal is the domain of what gets *stored* on
`ScrapeExtraction`, and a suppressed poll stores nothing; `"backoff"` is a statement about our own
behaviour rather than about a page, so it has no row to live on.

### 2. Fetch-side health, and a fingerprint to tell failures apart

A new `ScrapeTargetHealth` entity carries:

| Column | Purpose |
|---|---|
| `content_fingerprint` | sha256 of `html_to_text(html)` with whitespace collapsed -- captured whenever a recipe validates |
| `consecutive_fetch_failures` | fetch-stage failures (blocked, transport, timeout) -- deliberately separate from the extraction counter |
| `circuit_state` | `closed` / `open` / `half_open`, the `CircuitBreaker` vocabulary |
| `blocked_until` | when the next probe is permitted |
| `last_blocked_at` | evidence for the operator and for tuning detection |
| `last_block_kind` | DECLARED BUT NEVER WRITTEN. Kept because the column costs nothing and the distinction it would carry is real, but nothing populates it today, so a reader must not treat it as evidence. `health.py` records the same |
| `classified_fingerprint`, `classified_verdict`, `classified_evidence` | the verdict cache: which page was last classified, what it was judged to be, and why |
| `session_state_sealed`, `session_state_expires_at` | §4 |
| `last_egress` | which exit the last observation was configured to leave by, reported by the fetcher rather than assumed by the caller (§7 -- configured, not observed). Without it "this target is walled" cannot be told apart from "this target is walled FROM THIS EXIT", and one blocked exit poisons a target permanently |
| `robots_blocked_at`, `robots_blocked_reason` | a `Disallow` that needs a person (§8). Deliberately NOT the circuit's columns: a policy decision is not a fetch failure, and counting it as one would back off a site that works perfectly |

The three `classified_*` columns are what makes "same fingerprint next poll, same verdict, no
call" implementable. They cannot be folded into `content_fingerprint`, which answers a
different question: `content_fingerprint` is the page as it looked when extraction last
*succeeded*, and is the comparison value for "has the site changed". A classification is
always asked about a page that just *failed*, so storing it in the same column would destroy
the only reference the comparison has.

A cached verdict also means "we have already acted on this exact page", which is what stops a
`changed` verdict regenerating on every subsequent poll after a regeneration that did not
stick. See §1's routing.

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
lives on the `ScrapeTargetHealth` row (`circuit_state`, `consecutive_fetch_failures`,
`blocked_until`), with `WindowedCounter` for the cross-pod counts and `TokenBucket` for probe
pacing. Section 2 settled that; this paragraph said "the recipe row" until the contradiction
was caught in review, and `ScrapeRecipe` is explicitly left untouched by all of this.

**How the rules stay in one place, which "adopted exactly" alone does not settle.** Storing
the state elsewhere is the easy half; the trap is then re-deriving the transitions next to
the store, because a second copy of a state machine is a second copy that can disagree. So
`CircuitBreaker` gained a `restore()` classmethod: the durable row is hydrated into a real
breaker, the transition is driven by calling `check()` / `record_success()` /
`record_failure()`, and the resulting state is written back. `TargetCircuit` therefore
contains storage, backoff arithmetic, and the judgement of which outcome counts as a fetch
failure -- and no transition logic at all. `restore()` deliberately does NOT restore an
in-flight probe: that belongs to whichever process issued it, and no other process can
observe it, which is exactly why cross-pod single-probe admission needs the `TokenBucket`
rather than the flag.

The gate sits at the FETCH boundary, not in the eval loop. This is not a placement
preference: the eval loop is handed a page that has already been fetched, so a gate there
could only suppress work downstream of the cost the circuit exists to avoid. A target inside
its window consequently reaches neither the candidate generator nor the page classifier.

`restore()` not carrying the in-flight flag has a second consequence, beyond needing the
`TokenBucket`: a restored HALF_OPEN breaker consults no timer at all, so a row left HALF_OPEN
admits a fresh probe on every poll. That row is reachable whenever a caller dies between the
fetch and the outcome report, and it would delete the decay while leaving every individual
transition correct. So the promotion writes `blocked_until` as the probe's own reservation
and honours it -- a pacer of last resort, built from the column already being written, for a
deployment that configured no `TokenBucket`.

Where a per-process fast-fail is still wanted, `scrape` accepts an injected breaker through
`core.http_client`'s existing `CircuitBreakerLike` protocol -- the same seam `core` already uses
to avoid importing `threetears.models` and its LangChain weight. It is
injected as a lookup KEYED BY TARGET rather than as one breaker, because a `TargetCircuit`
serves a whole set of targets: a shared breaker would let one walled target fast-fail every
other target on the same tool, and let a healthy target's success reset a count another
target had accumulated. `CircuitBreakerRegistry` is already per-key, and taking the key is
what keeps that property instead of dropping it at this seam.

Two protocols ARE new, which an earlier draft of this section denied. `ProbeObservableBreaker`
narrows `CircuitBreakerLike` with a readable `state`, because releasing a probe requires first
knowing one was admitted, and a breaker that cannot answer that gets wedged rather than
released -- the constraint belongs in the signature, not in a `getattr`. `ReprobeScheduler` is
the two-method seam `reprobe.py` satisfies -- book a probe, and cancel one when the circuit
closes, since a close is the one outcome that books nothing and so cannot supersede an
outstanding booking by replacing it, so the polling caller never takes on
`3tears-scheduled-jobs`. Neither adds a dependency: the LangChain-weight argument above is
about `core`'s reason for the original seam, and `circuit.py` already imports
`threetears.models.circuit_breaker` at module top, since `3tears-models` is a hard dependency
of `3tears-scrape` regardless.

The lookup's lifetime is the caller's, and the obvious choice has a sharp edge worth naming.
`CircuitBreakerRegistry` holds a plain dict with no eviction. Keyed by provider name -- what
it was built for -- that is bounded by a handful of entries. Keyed by scrape target it is
not, because `_derive_target_id` mints a fresh `adhoc_<sha256>` per distinct
`(url, field_schema)`, so a long-running tool handed the bare registry accumulates one
breaker per URL it has ever scraped. Deliberately not solved by evicting from inside
`TargetCircuit`: choosing a cache policy for the caller's process would discard circuit state
a walled target is relying on, on a schedule the caller never asked for. A long-lived
deployment injects a bounded lookup; a short-lived one has nothing to do.

The two circuits run on very different clocks -- seconds against minutes to hours -- so the
durable one routinely suppresses a fetch the in-process one has already admitted a probe for.
That probe then never resolves, and `CircuitBreakerLike` has no way to say "never mind", so
the suppressed path reports the failure it effectively had. Only where a probe was genuinely
admitted, though: telling a CLOSED breaker about a fetch that was never attempted trips it on
failures it never saw, after which the wrong circuit answers `check` and the caller is told to
retry in seconds when the truth is hours -- turning the suppression into a fixed-cadence poll,
which is the opposite of the whole section.

Re-probing a blocked target is scheduled through `3tears-scheduled-jobs` (`relative_delay`),
not a bespoke sleep-and-retry. That arrives as the optional `3tears-scrape[reprobe]` extra
rather than a hard dependency, because scheduled-jobs brings NATS and APScheduler with it and
a POLLING caller needs none of it -- its next poll is already the re-probe, gated by
`blocked_until`. Only an event-driven caller has nothing to wake it. The booked job's id is
derived from the target so a re-booking replaces the outstanding probe; with a random id
every superseded booking would survive and eventually fire, turning the longest backoff into
the biggest burst.

A transport failure shares this circuit with a wall, since both mean the content did not
arrive and retrying immediately will not change that, but only a wall stamps
`last_blocked_at` -- otherwise the column that answers "when was this target last behind a
wall" quietly becomes "when did anything last go wrong" and sends an operator hunting for a
challenge page that was really a DNS failure.

### 4. Reusing the human's work

On a successful human solve, the sidecar exports that browser context's cookies and storage
state. The MIT package seals it with `core.security.encryption.seal()` and stores it on the
`ScrapeTargetHealth` row (`session_state_sealed` / `session_state_expires_at`, shipped in
`v010`) with an expiry -- not on the recipe row, which section 2 settled and which the
migration reflects. Subsequent unattended renders pass it back so the session resumes
already-cleared.

These are session credentials. Sealed at rest under an operator-supplied master key resolved via
`secret_refs`, never written plaintext, never logged, and excluded from any debug dump.

Driver contract gains `session_state: dict[str, Any] | None = None` on `render()`, following the
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

Container additions, in two groups. For the display itself: `x11vnc`. Xvfb, Chromium and the
headful launch were already there.

`websockify` and the noVNC static assets were here too, and are deliberately gone. Serving the
client and relaying RFB belong to the MIT container that shares this pod: it ships noVNC in its
own wheel, pinned to the page that loads it, and reaches `x11vnc` across the pod's shared network
namespace. Keeping a second, unauthenticated route to the same display would keep exactly what
the capability check in front of the relay exists to prevent.

For making that display OPERABLE, all added after live verification rather than designed in:
`openbox` as a window manager, because bare Xvfb maps windows with no decoration and no way to
switch between them; `tint2` as a taskbar, because without one a minimised window is
unrecoverable; `x11-xserver-utils` for `xrdb`, which is how `UI_SCALE` reaches openbox and
tint2 so an operator can size the text; and `x11-utils` for `xprop`, which `entrypoint.sh`
polls to confirm the window manager actually came up. The image also patches openbox's
`rc.xml` down to one virtual desktop and unbinds the mousewheel from `GoToDesktop`, because an
idle scroll on the desktop background silently moved the operator to an empty desktop and their
targets appeared to vanish.

`x11vnc` starts **on demand** when a session opens and stops on teardown -- no idle VNC surface.
This matches the operational model: the display comes up when a person arrives, not before.

New sidecar endpoints (the sidecar remains a dumb browser-as-a-service -- no 3tears imports, the
AGPL boundary is unchanged):

| Endpoint | Purpose |
|---|---|
| `POST /v1/hitl/session` | Create a session, start VNC, return `{session_id, token, expires_at}` |
| `GET /v1/hitl/session/{id}` | Session state and open tabs |
| `POST /v1/hitl/session/{id}/tab` | Bring one target into the session: isolated context, navigate, replay `nav_steps` |
| `POST /v1/hitl/session/{id}/tab/{tab}/complete` | Human says cleared: verify, export sealed state, close the tab, free the slot |
| `DELETE /v1/hitl/session/{id}` | Teardown, stop VNC, drop contexts |
| `POST` / `GET` / `DELETE /v1/hitl/vnc` | Bring the display up, report whether it is up, take it down. Predates the session API and stays for the case it does not cover |

No response names a place to point a browser, and that is the point: where an operator goes is
decided by the platform that mounts the operator router, under a prefix this container never
learns.

**The operator's own surface is not here.** It is a mountable `APIRouter` in `3tears-scrape`
(`threetears.scrape.operator`), served by the MIT container in this pod: the operator page, the
vendored noVNC client, and the WebSocket that relays RFB from `x11vnc` over loopback. So a
platform fronts one origin with one TLS endpoint and one authentication point, and it is the
origin it already has rather than a second one belonging to this container.

That stream carries the session token in a WebSocket `Sec-WebSocket-Protocol` entry. Forced
rather than preferred: a browser cannot set arbitrary headers on an upgrade, and the only other
thing it can do is a query parameter, which writes a live credential into access logs, browser
history and referrer headers. The page takes the token from the URL FRAGMENT, which never
reaches a server at all.

Checking that token is a CAPABILITY check and not authorization. Who was entitled to hold one is
decided by the platform, where identity lives.

**Bounded working set.** A session has a fixed slot count. A target occupies a slot from
`/tab` until `/complete`; backgrounding a slow one still holds its slot. Items are pulled in as
slots free. No unbounded tab growth.

**Nothing is held while waiting.** A target that needs a human is not parked in a live browser.
It is reported back and forgotten; the session re-drives it from `url` + `nav_steps` when an
operator actually arrives. Waiting therefore costs zero container resources, and re-driving is
deterministic because `nav_steps` replay is already how this package reaches gated pages.

**Security.** The session token is unguessable, short-lived, scoped to one session and bound to
its TTL. `x11vnc` binds loopback only, which on Kubernetes means it is reachable by the MIT
container sharing this pod's network namespace and by nothing else -- that binding IS the access
control on the display port, not a hardening extra. A session has a hard TTL with a reaper. The sidecar never authenticates a human -- it honours a token that the MIT side
minted only after authorizing the request.

**Concurrency, stated honestly.** One Xvfb display means one operator session at a time; a second
request queues or is refused. Multiple concurrent operators need a display pool
(`:100`, `:101`, …), each with its own Chromium and `x11vnc`. The display number is a parameter
from the start so the pool is a later configuration change rather than a rewrite, but v1 is
single-display.

### 6. Authorization, audit, announcement

**This section was rewritten mid-build, and the original plan was WRONG.** It is recorded here
rather than deleted, because the file inventory below marks two files as deliberately not
built and this is the reason.

The original: an RBAC gate at `hitl/authorize.py` mirroring `memory/authorize.py`, a session
state machine, and audit publishing, all shipped by this package. The error is one of layer.
3tears is a LIBRARY; the platforms built on it own identity, roles, the operator queue, and
the conversation that reaches a person. A `HitlAccessDenied` and an `AclCache` lookup here
would be a second, weaker copy of machinery the hub already runs -- and the one place it would
diverge is the place that matters, since the sidecar holds no identity and structurally cannot
evaluate a policy no matter which package the evaluator ships in.

What this package provides instead is the two seams a platform needs, and nothing else:

- **"This target needs a human" is a fact on the health row**, discoverable via
  `list_walled()`, which answers with both kinds -- a bot wall and a robots refusal. A platform
  polls it, or subscribes to the existing `Subjects` builders. That is the whole queue surface.
- **The approval itself is the hub's existing HITL contract**: `Subjects.hub_approval_record()`
  / `hub_approval_resolve()`, `TearsTool.requires_confirmation`, and the LangGraph interrupt.
  The hub already does ACL, audit and resume for every other tool that reaches for a person;
  a scrape reaching for one is not special enough to deserve its own path.

The sidecar's session token proves only that a caller holds something this container minted.
Deciding who was ENTITLED to it happens where identity lives, which is not here and was never
going to be.

### 7. Egress: which exit a request leaves by

**Requirement, raised 2026-07-26, mid-build.** TOR egress is fundamental to the scraper, with
Cloudflare WARP as an option, and both behind a driver seam so a third exit later is one class
rather than a change to the scraper.

**faidh's existing `ProxyStrategy` is prior art, and this seam does NOT migrate it here.**
This was decided rather than assumed, against the standing rule that two unrelated egress
abstractions must not end up in one codebase. They will not, because they are
not in one codebase: faidh is a consuming application, `threetears.core.egress` is library
code, and the dependency runs one way. What must not happen -- and would have, silently -- is
faidh keeping `ProxyStrategy` FOREVER alongside this seam, so that a third exit has to be added
twice.

The decision: faidh migrates onto `EgressDriver` and deletes `ProxyStrategy`/`DirectProxy`/
`TorProxy`, as a change in faidh's own repo on faidh's own schedule. It is not a precondition
for shipping this seam, and it is not optional either; it is tracked in the backlog so
"later" has somewhere to live rather than being a word in a design document. The shapes already
correspond -- `DirectProxy` is `DirectEgress`, `TorProxy` is `SocksEgress("tor")` -- which is
why this is a deletion rather than a rewrite.

**The goal, settled and not to be relitigated.** TOR serves BOTH non-attribution and block
evasion, and neither reliably. It is wanted for toolbox completeness -- "just another tool in
our tool box that we need to have so we're complete" -- with the limits understood up front:
TOR exits are public, enumerable, heavily challenged by bot walls, and frequently blocked
outright by exactly the attribution-averse sites someone would reach for it against. Routing a
target through TOR RAISES the challenge rate that sections 1 through 3 exist to lower, and the
human path in sections 4 through 6 is what pays for that. This is a reason to make the exit
selectable per target, not a reason to omit it.

**Built in `threetears.core.egress`, not in this package.** Every app on this framework
eventually wants a request to leave by something other than the container's default route, and
putting it where it was first needed is how ten apps end up with ten of them. The reuse was
already there: `core.http_client` calls itself "the one transport" for outbound HTTP and
already exposed an `httpx.AsyncBaseTransport` seam -- and httpx proxying IS a transport, so an
exit needed no new plumbing, only a driver that produces one. An explicit transport still wins
over a configured egress, because that seam is documented as the test seam.

`EgressDriver` has two halves on purpose. An exit is not an HTTP concept, so a driver answers
both "what transport should httpx bind" and "what does a browser need on its command line". A
driver that could only do the first would let a deployment proxy its API calls while its
scrapes went out direct, both reporting the same configured exit -- worse than no proxying,
because the deployment believes it has the property.

`DirectEgress` is a driver rather than a special case, so "direct" cannot quietly mean "the
seam was bypassed". An unknown driver name raises rather than falling back to direct, and an
exit with no address or no name is refused at construction: both are the same failure, an exit
that reports itself as `tor` in every log and leaves by the container's own IP.

**Nothing here starts a daemon.** A driver describes an exit; running `tor` or `warp-cli` is
deployment work, and a library owning process lifecycle for someone else's network would be
wrong about it in every deployment that already had one.

**Per-target, via browser contexts rather than the command-line flag.** An earlier version of
this section said per-target selection was not possible and one container was one exit. That
was true of `--proxy-server`, which Chromium applies process-wide, and wrong about contexts:
`Target.createBrowserContext` takes its own `proxyServer`, so two targets in one browser can
leave by two exits. The claim was corrected after probing the running image's own CDP
bindings rather than recalling the flag's behaviour.

So the container's `EGRESS_PROXY` is a DEFAULT, and `RenderRequest.egress_proxy` overrides it
for one render, which gets its own context and disposes it with the tab. `last_egress` on the
health row records which exit the render was CONFIGURED to leave by. The value is reported by
the fetcher rather than assumed by the caller: the sidecar returns it, the driver carries it
back on `RenderedPage.egress`, and the circuit stamps that rather than a constructor-time name
which a per-render override would have made wrong. What it buys is that a dropped proxy
argument surfaces as a mismatch, because an older sidecar that ignores the argument reports its
own exit instead of echoing the one it was asked for.

It is not evidence that traffic left that way. A per-context proxy Chromium accepted and then
ignored would still be recorded under the name it was asked for, and nothing inside the process
can tell the difference. Confirming it needs an observer outside the process, reading the
address the container presents to a third party; that verification is tracked separately and
deliberately, because a unit test asserting it would be asserting on its own fake.

`None` means no exit was configured, which is a different fact from choosing the default route.
That choice is `DirectEgress` and records as `direct`.

With more than one exit, the useful fact stops being "this target is walled" and becomes
"walled FROM THIS EXIT", without which a target blocked through one route looks permanently
walled and a working alternative is never tried.

One configuration hazard is worth naming because it is invisible: egress is wired separately on
the drivers and on `ScrapeTool` itself, and getting either half alone leaks the container's
address on the other. Drivers proxied with an unproxied gate means the page leaves by the
configured exit while the `robots.txt` read in front of it does not. The gate proxied with an
unproxied driver is the worse one, because what goes out direct is the page fetch itself, the
request the exit was configured for. Both halves work either way; the target simply learns the
real address from the request nobody was thinking about.

`ScrapeTool` warns on both shapes, reading the gate it actually holds rather than its own
constructor argument -- a caller can build a `RobotsGate` with its own egress and pass it in, so
the argument describes what the default gate WOULD have been. It says nothing when robots is
disabled, since there is no second request to be split from. A warning rather than a refusal,
since a deployment may want exactly that, but it should have to be a decision.

In the gate-proxied shape the warning names the unproxied drivers, which is how a backend that
cannot honour an exit at all gets reported. Most backends cannot: `CamoufoxDriver` launches a
browser with no proxy support, and `DocumentDriver`, `ListingDetailDriver` and
`MultiDocumentDriver`'s listing fetch each build a bare `httpx.AsyncClient`. Threading an exit
through them is tracked in the backlog; until then the bypass is loud rather than closed.

Note the asymmetry, because it bounds what this warning is worth. In the drivers-proxied shape
the message names the PROXIED drivers, so a deployment that proxied what it could and left the
rest is told about the `robots.txt` read and told nothing about the backends going direct. Two
configurations reach that shape: proxying drivers individually while leaving the gate alone, and
passing a gate explicitly, since `ScrapeTool(egress=X, robots=RobotsGate())` gives a gate with no
exit -- the default gate inherits `egress` only through the absent-argument sentinel, so
supplying any gate opts out of the inheritance. The gate-proxied shape is what a plain
`ScrapeTool(egress=...)` produces, and it is the one that reports the bypass.

### 8. robots.txt: wait when asked, escalate when refused

**Requirement, raised 2026-07-26, mid-build.** The option to check `robots.txt` and honour it:
respect rate limits, and flag a target for a human if it says no bots. Both configurable, both
enabled by default -- a scraper whose politeness is opt-in is impolite in every deployment
nobody configured, and those are the deployments nobody is watching.

**The two halves are different decisions.** `Crawl-delay` asks us to be slower and changes
scheduling only; the target still gets fetched. `Disallow` asks us not to fetch at all, and
neither obeying nor ignoring it is right: obeying makes a target permanently invisible with no
way to say "we have an agreement with this site", and ignoring is what gives crawlers their
reputation. So it escalates, through the same human path a bot wall already takes.

**A human working the page over VNC is not a bot.** The Robots Exclusion Protocol governs
automated agents, not people operating browsers, so a `Disallow` that stops the unattended
fetcher does not stop an operator who opens a session and works the target themselves. That is
what makes the escalation close rather than dead-end. Two things keep it a position rather
than a loophole: the exemption is for a session a person is actually IN, not "open a session
and let the robot drive through it", and `Crawl-delay` does NOT get the exemption, because
load on someone's server is caused equally by either.

**Composition with the fetch circuit.** Both gate the fetch and they are different kinds of
gate. `Crawl-delay` is a FLOOR on politeness that applies to a target working perfectly;
`blocked_until` is a CEILING on cost that applies to one that is not. A fetch satisfies both,
and neither may weaken the other -- in particular a circuit probe is not exempt from the crawl
delay, or the politeness contract breaks precisely when a target is already unhappy with us.

**Every unusable `robots.txt` means "allowed".** Missing, 500, empty, garbage, unreachable,
unparseable: they all mean the site has not told us anything, and treating any as a refusal
lets one bad response to a text file stop a scrape silently -- which looks exactly like a
target that quietly stopped producing data. The crawl clock starts on a FETCH rather than a
check, because the circuit can suppress a fetch after robots was consulted and a check that
led nowhere must not consume the site's patience.

Parsing is `urllib.robotparser` from the standard library. The grammar is looser than its
reputation and implementations disagree about wildcards and `Allow` precedence; the stdlib's
reading is defensible, already installed, and adding a package to read a text file fetched
once per origin is a poor trade. The optional cross-pod pacer is
`core.coordination.TokenBucket`, the same primitive and reasoning as the circuit's probe pacer:
without it the delay is honoured per process, which is a lie in a fleet, because five pods each
waiting ten seconds present a request every two.


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
- `packages/scrape/src/threetears/scrape/challenge.py` -- `PageVerdict`, `classify_failed_page`
- `packages/scrape/src/threetears/scrape/health.py` -- `ScrapeTargetHealth` + collection + writers
- `packages/scrape/src/threetears/scrape/circuit.py` -- `TargetCircuit`, `BackoffPolicy`, the fetch gate
- `packages/scrape/src/threetears/scrape/reprobe.py` -- scheduled-jobs adapter, `[reprobe]` extra only
- `packages/scrape/src/threetears/scrape/session_state.py` -- seal, open and store a human's solve
- `packages/scrape/src/threetears/scrape/robots.py` -- `RobotsGate`, `RobotsPolicy` (§8)
- `packages/core/src/threetears/core/egress.py` -- `EgressDriver` and friends (§7); in core, not
  in scrape, because an exit is not a scraping concept
- `packages/scrape/sidecar/hitl.py` -- session endpoints, VNC lifecycle
- tests alongside each

**Planned and NOT built, deliberately**
- ~~`packages/scrape/src/threetears/scrape/hitl/authorize.py` -- RBAC gate~~
- ~~`packages/scrape/src/threetears/scrape/hitl/session.py` -- session client + state machine~~

  Both were dropped once §6 was rewritten. 3tears is a library; the platforms that consume it
  own identity, the operator queue and the conversation that reaches a human. An RBAC gate and
  a session state machine HERE would be a second, weaker copy of what the hub already has --
  see §6 for the seams that replaced them (`Subjects.hub_approval_record`,
  `TearsTool.requires_confirmation`). Listed rather than deleted because the file inventory is
  the first place a reader checks for "was this forgotten or decided".
- ~~`packages/scrape/sidecar/static/` -- noVNC assets~~ Debian's `novnc` package already ships
  the client; vendoring a copy would be a fork to maintain for nothing.

**Modify**
- `eval_loop.py` -- challenge short-circuit, fingerprint routing, fetch-health updates
- `tool.py` -- the fetch gate and the outcome report; the fetch boundary is where the circuit lives
- `packages/models/.../circuit_breaker.py` -- `CircuitBreaker.restore()`, the durable-state seam
- `collections.py` -- untouched by the health work in the end. The entity and collection live
  in `health.py`, and the planned re-export was dropped rather than added: a second import path
  for one class is a second thing to keep in step, and consumers import from `health` directly
- `migrations.py` -- `v010` creates `scrape_target_health` (fetch health, fingerprint, sealed
  session state); `v011` adds `last_egress`; `v012` adds the robots-block columns. Three
  migrations rather than one because `v010` had already shipped to `develop` -- an applied
  migration is immutable, so §7 and §8 add columns rather than editing history
- `driver.py` + all 8 drivers -- `session_state` parameter (accept-and-ignore except the browser backends)
- `sidecar/Dockerfile`, `entrypoint.sh` -- `x11vnc` (noVNC ships in the MIT wheel, not here)
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
5. A repeatedly blocked target's **fetch rate decays**: the circuit opens at the threshold and
   each probe that finds the wall still standing doubles the wait, to a ceiling.
6. Its **classification rate decays too**, which criterion 5 does not imply. Proven over many
   polls against a page carrying a per-request id -- the shape that provably defeats the
   verdict cache, so only the suppressed fetch can bound it.
7. No new state machine exists: the transitions come from `CircuitBreaker` via `restore()`,
   and the reused primitive is named at each site.
8. ~~An operator with a granted role can open a session for a permitted queue; one without is
   denied with `HitlAccessDenied`, distinguishable from "nothing queued".~~ **Withdrawn with
   the §6 rewrite**, which struck the file this criterion tested. Authorization is the
   consuming platform's, evaluated where identity lives; the sidecar holds none and cannot
   satisfy this criterion in any package. What replaces it: the session token proves only that
   a caller holds something this container minted, and the queue is a fact on the health row
   that a platform reads and gates for itself.
9. A completed solve yields sealed session state that a subsequent unattended render consumes to
   fetch the target successfully with no human involved.
10. Sealed state is unreadable without the master key; a tampered token is rejected.
11. Session teardown stops `x11vnc` and drops contexts; the TTL reaper collects an
    abandoned session.
12. `./scripts/check-all.sh` green; the introspection-based drift guard covers every new column.

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
