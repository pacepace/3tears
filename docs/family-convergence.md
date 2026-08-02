# Family Convergence: One Platform, Five Sharp Apps

**Status:** Proposal — 2026-08-02
**Scope:** 3tears, discodon, metallm, scriob, samsung-frame-art-loader, hallucinote

---

## 1. What this document proposes

That 3tears becomes the **single shared home** for every cross-app capability in the
family — evals, LLM access, memory, observability, identity, MCP conventions, content
acquisition, and headless chat UI — with each capability **sourced from the app that
already paid for it**, never built in 3tears from scratch. Apps keep only what makes
them singular; everything else they import.

Concretely:

- Discodon's eval system (the family's only complete one) is extracted into a
  four-package `3tears-eval-*` group that every sibling consumes.
- Hallucinote's MCP surface conventions (the family's most mature) are harvested into
  `3tears-mcp` as code rather than copied as prose.
- metallm's memory, model, and observability layers — already extracted into 3tears —
  are adopted by the holdouts (discodon foremost).
- Tiled/zoomable image acquisition is built once as a slim 3tears package from
  samsung-frame-art-loader's existing design, with `3tears-scrape` adapting it as a
  content type.
- Chat UI converges on a headless TypeScript kit published from this monorepo —
  shared behavior, per-app pixels.

The explicit non-goal: no app rebases its product core onto another app's. Discodon's
persona engine, metallm's personality graph, scriob's story plane, samsung's mat
engine, and hallucinote's Live bridge stay app-local forever.

## 2. Problem statement

Six long-lived codebases are converging on the same needs — LLM orchestration, eval,
memory, auth, telemetry, MCP surfaces, chat UI — and are currently solving them at
six different levels of completeness, largely independently. A survey of all six
repos (2026-08-02) found:

**Duplication that already happened:**

- **Three scraping stacks** (metallm's SearXNG/trafilatura path, discodon's research
  scraping, `3tears-scrape` itself).
- **Two OTel bootstraps and two dashboard sets** (metallm on `3tears-observe`;
  discodon on its own pinned `opentelemetry-*` stack).
- **Three parallel auth implementations** (`3tears-iam`, which scriob consumes;
  metallm's local bcrypt/JWT `auth.py`; discodon's Authlib GitHub OAuth +
  hand-rolled signed-cookie sessions) — GitHub OAuth specifically now exists
  twice, and the owners/allowlist/break-glass bootstrap pattern twice.
- **Three hand-rolled chat frontends** (metallm ~25k LOC, scriob ~1,100-line
  `Chat.tsx`, discodon's web chat), with a fourth (samsung) on the way.
- **Five different answers to "how do we eval"**: discodon's ~43k-line system,
  scriob's continuity-judge harness, samsung's MCP-driver eval, hallucinote's
  hand-operated scenario briefs, and metallm's nothing — despite metallm being the
  most LLM-central app in the family.
- **MCP conventions copied as prose**: samsung's API contract cites hallucinote's
  design 13+ times by name, re-deriving in documentation what exists as tested code.

**Adoption inversion:** metallm — the repo 3tears was extracted *from* — is now its
laggard consumer (pinned ~6 releases behind; still shipping local auth, backup, and
enforcement tests that scriob has already replaced with `3tears-{iam,backup,enforcement}`).
The newest big app (scriob) is the deepest consumer (15 of 20 packages). Without a
deliberate convergence direction, each new app re-decides everything, and each old
app drifts.

**The cost trajectory:** every capability listed above is about to be needed by at
least two more apps. Left alone, the family builds each one two to four more times,
then maintains the copies forever.

**Why convergence is cheap *now*:** the hard prerequisites already exist. 3tears is a
PyPI-published, lockstep-versioned family with mechanically enforced low coupling
(import-cost tests, dependency-alignment scanners); scriob has proven that deep
adoption works; discodon's eval system was explicitly designed with extraction seams,
two of which are already enforced by import-boundary tests; and hallucinote's MCP
stack is packaged code, not folklore.

## 3. Principles for the solution

1. **One shared home.** 3tears is the only shared library. No second library, no
   per-app platform forks. A capability is either app-singular or it lives here.
2. **Extract, don't invent.** Every capability added to 3tears is sourced from the
   app that already built and debugged it. 3tears' job is generalization and
   packaging, not greenfield design. (This is how the library was born — metallm —
   and how every successful package since has landed: scrape from faidh, channels
   from scriob's needs, the distributed lock from metallm's scheduler.)
3. **Second-consumer rule.** Packaging is paid for when a second real consumer pulls
   on a capability — not before (speculative generality) and not long after (a third
   copy gets written). Discodon's eval docs state this rule; the survey shows the
   second consumers now exist for evals, MCP conventions, and chat UI.
4. **Apps stay the best at what they do.** Product cores are never shared, and no
   app accepts a capability regression to adopt a shared package. Where the shared
   version is missing something an app needs (e.g. per-persona memory grain), the
   app's requirement is generalized *upstream* rather than worked around locally.
5. **Share behavior, not pixels.** UI convergence stops at the headless layer.
   Wire protocols, state machines, and sanitization are shared; components, tokens,
   and look-and-feel are app-owned.
6. **Contracts are the unit of convergence.** Where engines must stay different
   (conversation orchestration), the family converges on the *observation and data
   shapes* — the eval contracts, the stream-event protocol, the acquisition
   contract — so different engines remain comparable, instrumentable, and swappable.
7. **Leaf-package discipline, enforced.** New packages follow the existing house
   rules: minimal declared deps, extras for optional weight, import-cost tests,
   dependency-alignment enforcement, lockstep family versioning. Weight-sensitive
   consumers (a Raspberry Pi) must be able to take a slim slice.

## 4. The solution

### 4.1 Evals — `3tears-eval-{contracts,run,gen,analysis}` (from discodon)

Discodon's eval system is the seed: ~43k source lines whose analysis core is nearly
dependency-free and whose internal boundaries are already enforced by AST
import-boundary tests. The four aspects communicate through stored documents, not
function calls, so the package split follows seams that already exist:

- **`3tears-eval-contracts`** — the document models (runs, results, test cases,
  campaigns, analyses), the self-describing measure registry, identity/fingerprinting,
  the tolerant-read schema discipline (`schema_version` + legacy coercion), and the
  storage Protocol. Pydantic-only, near-zero deps — the `3tears-media-contracts`
  shape. This is the family-wide lingua franca for LLM quality data, and the
  non-negotiable split: it's what UIs, CI tooling, and exporters bind to.
- **`3tears-eval-run`** — runner, simulator, toolworld, expression DSL, judge,
  cassette proxy, jobs, budget. LLM clients and the persona-under-test are injected
  (the Protocols — `PersonaLike`, `SimulatorLLM`, the factory seam — already exist);
  each app supplies a thin host adapter.
- **`3tears-eval-analysis`** — stats (deliberately numpy-free), covariates, measure
  walk, bundle pipeline, LLM analysis generation, and the flat-row report
  projections every presentation surface renders from.
- **`3tears-eval-gen`** — LLM-assisted variation expansion, rubric/boundary
  proposers, classifier case generation. The most independent aspect; it only
  writes documents.

Consumers demonstrably want different subsets (hallucinote: run+judge only;
scriob: analysis/trends; samsung: the runner; CI: run without gen), and lockstep
versioning makes multi-package consumption free within the family — so granularity
costs import lines, not compatibility decisions. Presentation (REST/MCP routes,
React) stays app-side as adapters over the analysis projections.

Donated content: metallm's sycophancy-judge prompt becomes shared judge material;
hallucinote's brief/rubric/verdict scenario schema informs the corpus format.

Two clarifications from an investigation of 3tears itself (2026-08-02): the
library's only in-house eval machinery is `scrape`'s recipe judge loop — a
*runtime self-healing* cycle (LLM judges pick and persist winning extraction
recipes), not an offline eval system; once `3tears-eval-run` lands, its
judge/verdict primitives should back that loop, making scrape an internal
consumer like the apps. And a naming collision to manage at extraction time:
`datasources.definition.measure` (BI semantic-model measures, in flight) is
unrelated to the eval measure registry — the packages must disambiguate.

### 4.2 LLM substrate — `3tears-models` (exists; metallm lineage)

Already the family standard: LangChain-native construction, usage tracking with
locked OTel span attributes, pricing, circuit breaking, friendly errors. The change
is adoption (discodon retires its hand-rolled OpenRouter/LangChain plumbing) and
enrichment (samsung's live-probed OpenRouter findings — usage-cost semantics,
`/key` endpoint lag — fold upstream rather than seeding another first-party client).
LangSmith is explicitly **not** the family pattern: nothing in 3tears uses it, and
the one config-gated consumer (metallm) treats it as optional; the standard is OTel
via the usage tracker.

### 4.3 Memory — `3tears-agent-memory` (exists; metallm lineage)

The "Tom likes pizza / Barbara loves the Beach Boys" layer: extraction from
conversation, hybrid vector search, MMR reranking, consolidation. Production-proven
(26 migrations). Every app that learns durable user facts uses it. The one
generalization required: discodon needs **per-persona subjective memory** — Bob's
memories of Tom and Alice's memories of Tom are different data, a character
feature, not an implementation detail. If the current keying (shaped to metallm's
one-shared-agent grain) can't express an (agent, subject) pair, that grain
generalization is discodon's upstream contribution.

### 4.4 Conversation engines — organs shared, skeletons app-local

Discodon's persona/turn engine and metallm's personality graph are product cores and
stay put. Their *generic organs* converge: memory extraction/retrieval nodes (via
agent-memory), the PII sanitize/unsanitize wrapper around tool execution (metallm
lineage — valuable to any app running tools over user text), and judge prompts (eval
content). The engines converge on the **eval-contracts turn/trace observation
shape**, so quality is measured identically across different orchestrators without
either being rewritten. Migrating onto the `3tears-langgraph` substrate remains an
option any app can take when its turn loop comes up for rewrite anyway — it is not
a convergence requirement.

### 4.5 MCP conventions — into `3tears-mcp` (from hallucinote)

Hallucinote's stack is the family's most mature agent surface (13 noun-shaped tools
× 124 actions) and is already cited as precedent by siblings — as prose. The
harvest, as code: the typed action registry with structural invariant enforcement
(declarative-op XOR handler, boot-time param-collision detection), teaching errors
that carry the fix (`valid_actions` + `required`/`optional` + `example` + `hint`;
reject unknown params rather than dropping them), flat-schema synthesis from action
unions, the long-poll job pattern with its coupled timeout ladder (including the
FastMCP inline-sync-dispatch finding), the tri-state capability matrix
(SUPPORTED / NOT_IMPLEMENTED / UNSUPPORTED_UPSTREAM — "a binary supported/not is a
lie"), guides-as-resources, and the two-process content-fingerprint version
handshake. Packaging constraint from the source: hallucinote's Live-side half is
contractually stdlib-only, so the shared layer must be consumable server-side-only
or vendorable-by-copy. Lineage note: hallucinote credits cordyceps for the original
pattern.

### 4.6 Tiled/zoomable images — a slim acquisition package (from samsung's design)

Tiled zoomable imagery (IIIF Image API, Deep Zoom/DZI, Zoomify, slippy tiles) is a
general web content type — maps, research figures, pathology, art — not a museum
niche. The capability is built **once, here**, from samsung-frame-art-loader's
already-designed (and deliberately unwritten) acquisition contract: acquisition
method × source class × fetch status with partial-tiles as a first-class outcome,
argv-not-shell invocation of `dezoomify-rs`, URL-scheme allowlisting, zero-byte and
free-space guards. Packaged as a **slim leaf** (contracts-shape: minimal deps), not
inside `3tears-scrape` — scrape is the family's heaviest slice and the first
consumer is a memory-capped Pi. `3tears-scrape` grows a thin driver adapting the
leaf, so future consumers arriving via scrape get tiled images without knowing the
art loader exists. The contract is binary-agnostic so a native IIIF/DZI fetcher can
replace the external binary without an API break. Samsung's legacy glue (including
its `shell=True` invocation) is deleted, not lifted.

### 4.7 Scraping — consolidate on `3tears-scrape` (exists; faidh lineage)

Three stacks become one. Lowest urgency of the workstreams — behind evals — but the
end state is that metallm's fetch/extract path and discodon's research scraping are
consumers, not implementations.

### 4.8 Identity — `3tears-iam` + `3tears-agent-acl` (exist; scriob is proof-of-life)

The only implementation of passwords, tokens, OAuth, and RBAC in the family. The
change is metallm retiring its local `auth.py` and backup for the shared packages,
and promoting scriob's credential-cascade/secret-sealing layer (which metallm has
already hand-copied — the third copy should be prevented, not written).

Discodon adopts iam too: it already runs GitHub OAuth (via Authlib), a DB-backed
username allowlist with hardcoded permanent owners, a break-glass admin password,
and hand-rolled HMAC-signed session cookies — the family's third parallel auth
implementation. The seam scriob proved maps 1:1: iam's github/oauth-state/
passwords/tokens replace discodon's Authlib client, break-glass hashing, and
cookie mint/verify (discodon's documented no-server-side-revocation tradeoff
carries over; both are self-contained signed credentials), while the allowlist,
builtin owners, and cookie transport stay app-local. Bonus: the "hardcoded
owners + DB allowlist + break-glass password" trio is now independently
duplicated in discodon and scriob — a candidate to promote into iam as an
optional small-deployment module under the second-consumer rule.

Hallucinote (stdio + loopback by design) never adopts iam.

### 4.9 Observability — `3tears-observe` (exists; metallm lineage)

One spine: consistent spans, cost/usage metrics, log correlation, one dashboard set.
metallm's thin re-export shim is the adoption playbook. Discodon drops its
independently pinned OTel stack; samsung and hallucinote gain telemetry they
currently lack (within samsung's recorded "no backends" stance — see open
questions — and hallucinote's server-side half only).

### 4.10 Chat UI — a headless TypeScript kit (protocol from 3tears; seeds from scriob and metallm)

Share behavior, never pixels. The expensive, invisible 80% of a chat UI is identical
across apps: the streaming state machine (token append, tool activity,
interrupt/retry, reconnect), scroll anchoring, optimistic sends, branching,
HITL approve/reject, and sanitized markdown/code rendering. The kit:

- Speaks the family stream protocol — the `StreamEvent`/`Frame` contract already in
  `3tears-langgraph`/`3tears-channels` (scriob's client is today a thin adapter over
  exactly this; metallm's raw `astream_events` filtering is the divergence to close).
- Lives in this monorepo and publishes to npm **in lockstep with the protocol it
  speaks** — protocol and client must version together.
- Is headless: framework-light hooks + unstyled primitives (seed: scriob's
  React-free `chat.ts` socket client; feature reference: metallm's frontend —
  branching, i18n, accessibility).
- Enforces its own purity the way discodon's `eval-kit` does: a mechanical
  import-boundary test, so the kit can never grow a dependency on any app's
  contexts or router.

Shared for security rather than economy: the markdown/HTML sanitization layer —
one place to fix an XSS class instead of four. Also shared as *tooling*: discodon's
DTCG design-token pipeline (tokens.json → CSS vars), with every app owning its own
palette. Explicitly not shared: styled components, layouts, tokens themselves.
Four products that feel nothing alike, running the same chat engine.

### 4.11 Config management — promote the contract, not the system (from discodon)

App config is three layers with three different verdicts:

- **Env/bootstrap config** (`DD_*`, `METALLM_*`, `SCRIOB_*` settings modules) stays
  app-local. 3tears' recorded philosophy — "the library instruments; the host
  configures" — is correct at this layer; a shared framework would add ceremony,
  not value.
- **Secrets and credentials** converge fully on `threetears.core.security`
  (reference-based `resolve_secret`, sealing) plus scriob's credential cascade.
  This is already in motion (metallm is mid-migration); discodon's AES-256-GCM
  store is the family's *third* independent AES-256-GCM implementation — the same
  story auth was before iam — and discodon becomes a consumer here.
- **The operator-editable runtime config store** is the promotable piece, sourced
  from discodon, which has the family's most complete instance: an encrypted
  DB-backed store with a four-layer precedence contract (code defaults → git seeds
  that fill empty slots only → store as master once set → env override wins) and
  operator edit surfaces (web UI + CLI). The family keeps rebuilding partial
  instances of exactly this — metallm's provider/model tables, scriob's DB-driven
  model catalogue with per-turn credential resolution — because "operator edits
  beat redeploys" is a need every serving app hits. What generalizes: the
  precedence resolver, the seed semantics, and a store Protocol. What stays
  app-local: naming conventions, the scoping model (discodon is flat
  one-DB-per-environment; scriob needs per-tenant), and the storage backend.

Priority: second tier, behind evals/memory/chat — every app's config works today,
so the cost being avoided is future divergence rather than present duplication
(except secrets, where consolidation is already underway).

### 4.12 Prompt management and versioning (identity from discodon; durable tier from scriob's pattern)

Prompts couple to both evals and administration, and the two couplings resolve
differently:

- **Eval coupling resolves through identity, not the store.** Discodon's eval
  system never imports its prompt registry — it records **content-addressed
  identity** (SHA-256 content hashes as true identity, version numbers as human
  convenience) and classifies every score-determining input as *apparatus* /
  *lever* / *label* so comparisons can badge exactly which condition moved. That
  discipline moves into `3tears-eval-contracts`; each app's host adapter answers
  one question — "what is the content hash of each prompt component this run
  used?" — regardless of whether its prompts live in a DB (discodon, metallm),
  static code (scriob), or files. Eval promotion therefore does **not** gate on
  prompt-registry promotion.
- **The registry is promoted as the third instance of the operator-editable
  store pattern** (§4.11's config store and the model catalogues are the first
  two): seed from code, store as master, operator edit surfaces, environment
  scoping — plus a prompt-specific layer (types, sections, rendering,
  content-hash dedup). The second consumer already exists: metallm stores
  system-owned judge prompts and *machine-rewritten* user prompts. Machine
  writers make **actor attribution** load-bearing (who changed this prompt:
  seed, operator, or system) — hallucinote's event-sourced actor model is the
  family precedent.
- **The seed-drift discomfort gets an architectural fix, not a workflow.**
  Seed-from-code-then-DB-master is two stores on different versioning substrates
  with a one-way door; discodon's export→review→promote flow is a hand-built
  return path. The principled design, already proven in-family by scriob's
  git-as-L3: the durable tier of the prompt store is git — operator and system
  edits become commits, so identity, diff, review, rollback, and history come
  free, and "promote to defaults" becomes a merge instead of a bespoke tool.
  Where git-backing is too heavy, the fallback is an append-only
  content-addressed store with lineage (parent hash, actor, source:
  seed|operator|system) and a mechanical diff-against-seed — drift always
  visible, never silent.

## 5. Implications per family member

### 3tears

- **Adopts (new packages/content):** `3tears-eval-{contracts,run,gen,analysis}`
  (from discodon); MCP conventions hardening into `3tears-mcp` (from hallucinote);
  slim tiled-image acquisition package (from samsung's design); headless chat kit +
  npm publishing lane (protocol its own, seeds from scriob/metallm); scriob's
  credential-cascade layer; samsung's OpenRouter findings into `models`; memory
  grain generalization (for discodon).
- **Drops:** nothing, but inherits obligations: the release path (three documented
  release incidents, most recently a PyPI publish with no tag) must be hardened
  before it carries five consumers' eval infrastructure.
- **Normalization work:** resolve the Python floor (core currently ≥3.14; samsung's
  integration findings audited all 16 blocking sites, showing core could floor at
  3.12 — which serves both discodon and the Pi); slim the `iam` dependency
  declaration toward its actual usage; define the supported-version window story so
  consumer skew is bounded; stand up the JS/npm side of the monorepo.

### discodon

- **Adopts:** `3tears-observe` (drops its pinned OTel stack; metallm's shim is the
  playbook), `3tears-models` (drops hand-rolled OpenRouter/LangChain plumbing),
  `3tears-agent-memory` (upgrades capped FIFO working notes to durable per-persona
  user facts), `3tears-iam` (drops its Authlib OAuth client, break-glass password
  hashing, and cookie mint/verify; keeps allowlist/owners/transport local, per the
  scriob seam), and — as **first consumer of its own extraction** — the
  `3tears-eval-*` packages, keeping its host adapters (persona factory, service
  facade, REST/MCP/React surfaces) local.
- **Contributes:** the entire eval system; the per-persona memory grain requirement;
  the eval-kit import-boundary pattern and DTCG token pipeline to the chat-UI
  workstream; the runtime-config precedence contract and seed semantics (while
  migrating its own secrets encryption onto `core.security`); the
  content-addressed prompt-identity discipline and versioned-input vocabulary
  (into eval-contracts) and the prompt registry as seed for the shared store.
- **Keeps:** the persona/entity/turn engine, prompt graph, and all Discord/Mastodon
  product surface. Explicitly does **not** rebase onto `3tears-langgraph` or adopt
  LangSmith.
- **Normalization work:** finish the in-flight eval schema work (the cell/control
  concept, case-set fingerprints, run-scoped judge configs) *before* extraction
  freezes the schema; execute the pre-extraction unlock list its own docs specify
  (storage Protocol, dedicated container namespace, universe→scope rename, moving
  three scoring helpers below the future run/analysis boundary, a deliberate home
  for toolworld given the host's inverted dependency on it); align on the resolved
  Python floor.

### metallm

- **Adopts:** `3tears-iam` + `3tears-backup` + `3tears-enforcement` (scriob proved
  all three), the shared stream protocol for its frontend, `3tears-eval-*` (it has
  zero eval infrastructure today — the family's biggest gap), and eventually
  `3tears-scrape`.
- **Contributes:** it already contributed most of the platform (models, memory,
  observe, tools lineage); newly: the sycophancy-judge prompt, the PII
  sanitize/unsanitize tool wrapper, and its frontend as the chat-kit feature
  reference.
- **Drops:** local `auth.py`, local backup, 26 bespoke enforcement test files, raw
  `astream_events` filtering, and (eventually) its SearXNG/trafilatura fetch path.
- **Normalization work:** close the ~6-release version lag first — every other
  adoption assumes a current pin; retire the symlink/pin drift alongside.

### scriob

- **Adopts:** `3tears-eval-{contracts,analysis}` for its continuity-judge corpus —
  turning its append-only baselines log into the family's trend/reporting machinery;
  the headless chat kit (replacing its hand-rolled `Chat.tsx` internals while
  keeping its look).
- **Contributes:** the stream-transport/client seed, the credential-cascade layer
  (promoted upstream), and its standing role as the reference consumer — its
  "library fit audit" doc is the model for adoption decisions.
- **Drops:** bespoke chat state machinery; the five-places-by-hand pin ritual once
  consumption standardizes.
- **Normalization work:** smallest of any app — it is already the north star's
  shape. Mostly: adopt eval contracts, swap chat internals.

### samsung-frame-art-loader

- **Adopts:** the tiled-image acquisition package (as first consumer, replacing its
  unwritten local plan), `3tears-eval-run` for its MCP-driver eval, the shared MCP
  conventions (which it currently follows as prose), and the headless chat kit if
  its curation UI grows chat.
- **Contributes:** the acquisition contract design; the OpenRouter live-probed
  findings; the 3tears Python-3.14-blocker audit; its DurableStore-protocol-shaped
  storage seam as the documented re-adoption path.
- **Drops:** the legacy 2024 acquisition glue (already scheduled for deletion,
  including the `shell=True` invocation) — deleted, not lifted.
- **Normalization work:** minimal and mostly aligned with work it planned anyway;
  its dependency-weight constraint (memory-capped Pi) is a *design input* the slim
  packaging must satisfy, not a blocker to work around.

### hallucinote

- **Adopts:** exactly one thing — a programmatic eval runner (`3tears-eval-run` +
  contracts) over its existing scenario briefs and canonical verdicts, replacing the
  human-operated subagent ritual and enabling CI gating. Optionally, server-side
  telemetry via `observe`.
- **Contributes:** the entire MCP conventions layer (registry, teaching errors,
  long-poll ladder, tri-state capability matrix, version handshake, dispatcher-level
  provenance) and its brief/rubric scenario schema.
- **Drops:** nothing — its architecture (no LLM client, no auth, no web stack) is
  correct for what it is.
- **Normalization work:** none forced. Its Live-side stdlib-only contract is a
  packaging constraint the shared MCP layer must respect (server-side-only or
  vendorable-by-copy), and that constraint improves the shared design.

## 6. Open questions

1. **Python floor.** Relax 3tears core to ≥3.12 (the audited 16 sites) or move
   discodon to 3.14? Relaxing serves two consumers (discodon, the Pi) and is
   recommended, but it constrains future use of 3.14-only features in core.
2. **Eval extraction timing.** The recommendation is "after discodon's in-flight
   schema-changing work lands" — but that work's completion date effectively gates
   the family's highest-value workstream. Is partial extraction (contracts first,
   run/analysis after) worth the two-step migration cost?
3. **`3tears-eval-gen` as a fourth package vs folded into run.** Recommended
   four-way split rests on lockstep making granularity cheap; if the package count
   itself becomes a maintenance concern, gen is the fold candidate — provided
   import paths are chosen day one so a later split isn't a breaking rename.
4. **dezoomify-rs licensing.** Verify the license class before it becomes a managed
   dependency of an MIT-published family. Shell-out-as-separate-process is fine
   either way; vendoring the binary may not be. The nodriver AGPL sidecar is the
   in-house precedent if isolation is needed.
5. **Memory grain.** Does `agent-memory`'s keying support an (agent, subject) grain
   today, or is schema work required for discodon's per-persona memories? Needs a
   design read before discodon's adoption is scheduled.
6. **npm publishing.** The chat kit needs a JS release lane in this monorepo
   (workspace layout, lockstep version stamping, publish workflow). Same-repo
   lockstep is the recommendation; the alternative (separate JS repo) reintroduces
   a version matrix against the wire protocol.
7. **Version-window policy.** Lockstep pinning is per-consumer, but what skew does
   the family *support* across consumers? metallm's 6-release lag went unnoticed by
   tooling; a stated window (and a check) would make lag a signal instead of a
   surprise.
8. **samsung's "no backends" observability stance vs `observe` adoption.** Its
   recorded position (structured logs, no exporters without a collector) is
   reasonable for an appliance; `3tears-observe`'s zero-dep core with no-op
   passthrough may satisfy both — needs an explicit reconciliation rather than a
   silent override of a recorded decision.
9. **Eval presentation sharing.** The React `eval-kit` stays discodon-local until a
   second app wants the *UI* (not just the projections). When that happens, does it
   join the chat kit's npm lane?
10. **Config-store promotion vs the "no config framework" philosophy.** Offering a
    runtime-config package amends a recorded 3tears position ("the library
    instruments; the host configures"). The reconciliation is probably "the
    philosophy governs *library* config; the new package is an *application*
    capability like iam" — but that's a ruling to make explicitly, not drift into.
11. **Git-backed vs append-only prompt store.** The git-as-L3 design (§4.12) is
    the principled fix for seed drift, but it makes the prompt store's write
    path depend on a git tier some deployments may not want. Decide whether the
    shared store offers both durable tiers behind one contract (the
    `DurableStore` seam suggests it can), and whether config (§4.11) and
    prompts share that store or merely its contract.
12. **Where convergence decisions live going forward.** This document proposes; each
    repo's governance must ratify the parts that bind it (floors, consumption
    patterns, what it drops). A lightweight cross-repo decision record — probably
    here in `3tears/docs/` — is needed so per-app sessions stop re-deriving the
    direction.
