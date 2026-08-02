# Family Convergence: One Platform, Five Sharp Apps

**Status:** Proposal — 2026-08-02
**Scope:** 3tears, discodon, metallm, scriob, samsung-frame-art-loader, hallucinote

---

## 1. What this document proposes

That 3tears becomes the **single shared home** for every cross-app capability in the
family — evals, prompt management, LLM access, memory, identity, observability,
config, MCP conventions, chat UI, and content acquisition — with each capability
**sourced from the app that already paid for it**, never built in 3tears from
scratch. Apps keep only what makes them singular; everything else they import.

Concretely:

- Discodon's eval system (the family's only complete one) is extracted into a
  four-package `3tears-eval-*` group that every sibling consumes.
- Prompt versioning converges on content-addressed identity, and prompt changes
  ship the way code ships: draft variant → eval against control → evidence-gated
  promotion.
- Hallucinote's MCP surface conventions (the family's most mature) are harvested
  into `3tears-mcp` as code rather than copied as prose.
- The packages that already exist — models, memory, observe, iam — are adopted by
  the holdouts, discodon foremost.
- Tiled/zoomable image acquisition is built once, as a slim package from
  samsung-frame-art-loader's existing design.
- Chat UI converges on a headless TypeScript kit published from this monorepo —
  shared behavior, per-app pixels.

The explicit non-goal: no app rebases its product core onto another app's.
Discodon's persona engine, metallm's personality graph, scriob's story plane,
samsung's mat engine, and hallucinote's Live bridge stay app-local forever.

## 2. Problem statement

Six long-lived codebases are converging on the same needs and currently solve them
at six different levels of completeness, largely independently. A survey of all six
repos (2026-08-02) found:

**Duplication that already happened:**

- **Three parallel auth implementations** (`3tears-iam`, which scriob consumes;
  metallm's local bcrypt/JWT; discodon's Authlib GitHub OAuth + hand-rolled
  signed-cookie sessions). GitHub OAuth exists twice; the owners/allowlist/
  break-glass bootstrap pattern twice.
- **Three scraping stacks** (metallm's, discodon's, `3tears-scrape` itself).
- **Three independent AES-256-GCM secrets implementations.**
- **Two OTel bootstraps and two dashboard sets.**
- **Three hand-rolled chat frontends**, with a fourth on the way.
- **Five different answers to "how do we eval"**: discodon's ~43k-line system,
  scriob's continuity harness, samsung's MCP-driver eval, hallucinote's
  hand-operated scenario briefs, and metallm's nothing — despite metallm being
  the most LLM-central app in the family.
- **MCP conventions copied as prose**: samsung's API contract cites hallucinote's
  design by name over a dozen times, re-deriving in documentation what exists as
  tested code.

**Adoption inversion:** metallm — the repo 3tears was extracted *from* — is now its
laggard consumer (~6 releases behind, still shipping local equivalents of three
shared packages), while the newest big app (scriob) is the deepest consumer (15 of
20 packages). Without a deliberate direction, each new app re-decides everything
and each old app drifts.

**Why convergence is cheap now:** the prerequisites exist. 3tears is
PyPI-published and lockstep-versioned with mechanically enforced low coupling;
scriob proves deep adoption works; discodon's eval system was designed with
extraction seams, two already enforced by tests; hallucinote's MCP stack is
packaged code, not folklore. Left alone, every capability above gets built two to
four more times, then maintained forever.

## 3. Principles for the solution

1. **One shared home.** 3tears is the only shared library. A capability is either
   app-singular or it lives here.
2. **Extract, don't invent.** Every capability added to 3tears is sourced from the
   app that already built and debugged it; 3tears generalizes and packages. (This
   is how the library was born, and how every successful package has landed.)
3. **Second-consumer rule.** Packaging is paid for when a second real consumer
   pulls on a capability — not before (speculative generality), not long after (a
   third copy gets written). The second consumers now exist for evals, MCP
   conventions, and chat UI.
4. **Apps stay the best at what they do.** Product cores are never shared, and no
   app accepts a capability regression to adopt a shared package. Where the shared
   version is missing something an app needs, the requirement is generalized
   upstream, not worked around locally.
5. **Share behavior, not pixels.** UI convergence stops at the headless layer;
   components, tokens, and look-and-feel are app-owned.
6. **Contracts are the unit of convergence.** Where engines must stay different,
   the family converges on the observation and data shapes — eval contracts, the
   stream-event protocol, the acquisition contract — so different engines remain
   comparable and instrumentable.
7. **Leaf-package discipline, enforced.** New packages follow the house rules:
   minimal deps, extras for optional weight, import-cost tests, lockstep family
   versioning. Weight-sensitive consumers (a Raspberry Pi) must be able to take a
   slim slice.

## 4. The solution

Ordered by theme: the quality system (evals, prompts), the AI substrate (models,
memory, conversation engines), platform services (identity, observability,
config), agent and user surfaces (MCP, chat UI), and content acquisition.

### 4.1 Evals — `3tears-eval-{contracts,run,gen,analysis}` (new; from discodon)

Discodon's eval system is the seed: its analysis core is nearly dependency-free,
its internal boundaries are already enforced by import-boundary tests, and its
four aspects communicate through stored documents rather than function calls — so
the package split follows seams that already exist:

- **`3tears-eval-contracts`** — document models (runs, results, test cases,
  campaigns, analyses), the measure registry, identity/fingerprinting, the
  tolerant-read schema discipline, and the storage Protocol. Pydantic-only. This
  is the family-wide lingua franca for LLM quality data and the non-negotiable
  split: it's what UIs, CI tooling, and exporters bind to.
- **`3tears-eval-run`** — runner, simulator, judge, cassettes, jobs, budget. LLM
  clients and the system-under-test are injected via existing Protocols; each app
  supplies a thin host adapter.
- **`3tears-eval-analysis`** — stats, covariates, bundle pipeline, LLM analysis
  generation, and the flat-row report projections every presentation surface
  renders from.
- **`3tears-eval-gen`** — LLM-assisted variation expansion and rubric/case
  proposers. The most independent aspect; it only writes documents.

Consumers demonstrably want different subsets (hallucinote: run+judge; scriob:
analysis/trends; samsung: the runner; CI: run without gen), and lockstep
versioning makes multi-package consumption free within the family. Presentation
(REST/MCP routes, React) stays app-side as adapters over the analysis
projections.

Donated content: metallm's sycophancy-judge prompt; hallucinote's
brief/rubric/verdict scenario schema. Two footnotes: 3tears' only in-house eval
machinery — scrape's runtime recipe-judge loop — becomes an internal consumer of
eval-run's judge primitives once they land; and the eval measure registry must
disambiguate its naming from the unrelated BI measures in
`datasources.definition`.

### 4.2 Prompt management — identity from discodon; durable tier from scriob's pattern

Prompts cross-cut evals and administration. Three planes, three answers:

- **Storage/authoring — shared.** The prompt registry is promoted as an instance
  of the operator-editable store pattern (§4.8), with a prompt-specific layer:
  types, sections, rendering, content-hash dedup. The second consumer already
  exists — metallm stores system-owned judge prompts and *machine-rewritten* user
  prompts — and machine writers make **actor attribution** load-bearing (who
  changed this prompt: seed, operator, or system; hallucinote's event-sourced
  actor model is the precedent). The durable tier is git, per scriob's
  git-as-L3 pattern: operator and system edits become commits, so identity,
  diff, review, rollback, and history come free, and "promote to defaults"
  becomes a merge. This is the architectural fix for the known discomfort of
  seed-from-code-then-DB-drifts. Where git-backing is too heavy, the fallback is
  an append-only content-addressed store with lineage and a mechanical
  diff-against-seed — drift always visible, never silent.
- **Assembly (dynamic composition) — app-local.** Discodon's prompt-graph DAG,
  metallm's personality node, and scriob's per-object prompts are product cores
  with no second consumer pulling for a shared engine. What is shared: every
  engine emits an **assembly-provenance record** — content hashes of each
  component that entered the prompt plus a composition/variant hash — cheap for
  static and dynamic engines alike, and what makes different engines comparable.
- **Measurement / A/B — shared via eval-contracts.** Content-addressed identity
  (SHA-256 as true identity, version numbers as human convenience) and the
  apparatus/lever/label input classification move into contracts. Dynamic
  prompts mean rendered text can never be the A/B unit; the lever is a *variant
  identity*, with apparatus proof (component hashes, frozen case sets,
  deterministic seeds) showing everything else held. Discodon already ships this
  shape as run-scoped prompt overlays recorded in run identity.

The planes close into one promoted workflow, the **variant lifecycle**: an edit
creates a draft variant → an eval campaign runs it as a cell against control →
the verdict attaches to the variant's content hash → promotion is gated on that
evidence → the promoted default lands in the git-durable tier. Prompts ship the
way code ships — through gates. Note the decoupling: eval extraction does *not*
wait on the shared store — the eval coupling is hashes, not imports, so any app
participates by answering "what is the content hash of each prompt component this
run used?"

### 4.3 LLM substrate — `3tears-models` (exists; metallm lineage)

Already the family standard: LangChain-native construction, usage tracking with
locked OTel span attributes, pricing, circuit breaking. The change is adoption
(discodon retires its hand-rolled OpenRouter/LangChain plumbing) and enrichment
(samsung's live-probed OpenRouter findings fold upstream rather than seeding
another first-party client). LangSmith is explicitly **not** the family pattern —
the standard is OTel via the usage tracker.

### 4.4 Memory — `3tears-agent-memory` (exists; metallm lineage)

The "Tom likes pizza" layer: fact extraction from conversation, hybrid vector
search, reranking, consolidation; production-proven. Every app that learns
durable user facts uses it. One generalization required: discodon needs
**per-persona subjective memory** — Bob's memories of Tom and Alice's memories of
Tom are different data, a character feature. If the current keying (shaped to
metallm's one-shared-agent grain) can't express an (agent, subject) pair, that
generalization is discodon's upstream contribution.

### 4.5 Conversation engines — organs shared, skeletons app-local

Discodon's persona/turn engine and metallm's personality graph stay put. Their
generic organs converge: memory extraction/retrieval (via agent-memory), the PII
sanitize/unsanitize wrapper around tool execution (metallm lineage), judge
prompts (eval content). The engines converge on the **eval-contracts turn/trace
observation shape**, so quality is measured identically across different
orchestrators without either being rewritten. Migrating onto `3tears-langgraph`
remains an option an app can take when its turn loop comes up for rewrite anyway
— it is not a convergence requirement.

### 4.6 Identity — `3tears-iam` + `3tears-agent-acl` (exist; scriob is proof-of-life)

The only implementation of passwords, tokens, OAuth, and RBAC in the family.
metallm retires its local auth and backup for the shared packages; scriob's
credential-cascade/secret-sealing layer (which metallm has already hand-copied)
is promoted before a third copy gets written. Discodon adopts iam along the exact
seam scriob proved: iam's OAuth/state/passwords/tokens replace its Authlib
client, break-glass hashing, and cookie mint/verify, while the allowlist,
builtin owners, and cookie transport stay app-local. The "hardcoded owners + DB
allowlist + break-glass password" trio — now independently duplicated in discodon
and scriob — is a candidate for an optional small-deployment iam module.
Hallucinote (stdio + loopback by design) never adopts iam.

### 4.7 Observability — `3tears-observe` (exists; metallm lineage)

One spine: consistent spans, cost/usage metrics, log correlation, one dashboard
set. metallm's thin re-export shim is the adoption playbook. Discodon drops its
independently pinned OTel stack; samsung and hallucinote gain telemetry they
currently lack (within samsung's recorded "no backends" stance — see open
questions — and hallucinote's server-side half only).

### 4.8 Config — promote the contract, not the system (from discodon)

Three layers, three verdicts:

- **Env/bootstrap config stays app-local.** 3tears' recorded philosophy — "the
  library instruments; the host configures" — is correct at this layer.
- **Secrets and credentials converge fully** on `threetears.core.security` plus
  scriob's credential cascade. Already in motion; discodon's secrets store is the
  family's third independent AES-256-GCM implementation — the same story auth
  was before iam — and becomes a consumer.
- **The operator-editable runtime store is the promotable piece**, sourced from
  discodon's — the family's most complete: encrypted DB-backed store, four-layer
  precedence (code defaults → seeds fill empty slots only → store as master →
  env override wins), operator edit surfaces. The family keeps rebuilding
  partial instances (metallm's provider tables, scriob's model catalogue)
  because "operator edits beat redeploys" is a need every serving app hits. What
  generalizes: the precedence resolver, seed semantics, and a store Protocol.
  What stays app-local: naming, scoping model, storage backend.

Priority: second tier, behind evals/memory/chat — every app's config works today,
so the cost avoided is future divergence, not present duplication.

### 4.9 MCP conventions — into `3tears-mcp` (from hallucinote)

Hallucinote's stack is the family's most mature agent surface and is already
cited as precedent by siblings — as prose. The harvest, as code: the typed action
registry with enforced structural invariants, teaching errors that carry their
own fix, flat-schema synthesis over action unions, the long-poll job pattern with
its coupled timeout ladder, the tri-state capability matrix (supported /
not-implemented / upstream-can't — "a binary supported/not is a lie"),
guides-as-resources, and the content-fingerprint version handshake for code
vendored into host processes. Packaging constraint from the source: hallucinote's
Live-side half is contractually stdlib-only, so the shared layer must be
consumable server-side-only or vendorable-by-copy. (Lineage: hallucinote credits
cordyceps for the original pattern.)

### 4.10 Chat UI — a headless TypeScript kit (protocol from 3tears; seeds from scriob and metallm)

Share behavior, never pixels. The expensive, invisible majority of a chat UI is
identical across apps — streaming state machine, scroll anchoring, optimistic
sends, branching, HITL approve/reject, sanitized markdown rendering. The kit:

- Speaks the family stream protocol (the `StreamEvent`/`Frame` contract already
  in `3tears-langgraph`/`3tears-channels`; scriob's client is already a thin
  adapter over it — metallm's raw event filtering is the divergence to close).
- Lives in this monorepo and publishes to npm **in lockstep with the protocol it
  speaks**.
- Is headless: hooks + unstyled primitives (seed: scriob's React-free socket
  client; feature reference: metallm's frontend), with its purity enforced by a
  mechanical import-boundary test, the way discodon's eval-kit does it.

Shared for security rather than economy: the markdown/HTML sanitization layer —
one place to fix an XSS class instead of four. Shared as tooling: discodon's
design-token pipeline, with every app owning its own palette. Explicitly not
shared: styled components, layouts, tokens. Four products that feel nothing
alike, running the same chat engine.

### 4.11 Tiled/zoomable images — a slim acquisition package (new; from samsung's design)

Tiled zoomable imagery (IIIF, Deep Zoom, Zoomify, slippy tiles) is a general web
content type — maps, research figures, pathology, art — not a museum niche. The
capability is built once, here, from samsung-frame-art-loader's already-designed
(and deliberately unwritten) acquisition contract: acquisition method × source
class × fetch status with partial-tiles as a first-class outcome, safe subprocess
invocation of `dezoomify-rs`, URL allowlisting, integrity guards. Packaged as a
**slim leaf**, not inside `3tears-scrape` — scrape is the family's heaviest slice
and the first consumer is a memory-capped Pi. Scrape grows a thin driver adapting
the leaf, so future consumers arriving via scrape get tiled images without
knowing the art loader exists. The contract is binary-agnostic, so a native
IIIF/DZI fetcher can replace the external binary without an API break. Samsung's
legacy glue is deleted, not lifted.

### 4.12 Scraping — consolidate on `3tears-scrape` (exists; faidh lineage)

Three stacks become one. Lowest urgency of the workstreams, but the end state is
that metallm's fetch/extract path and discodon's research scraping are consumers,
not implementations.

## 5. Implications per family member

### 3tears

- **Gains:** the eval package group (discodon), MCP conventions (hallucinote),
  tiled-image acquisition (samsung's design), the headless chat kit + an npm
  publishing lane, the shared runtime store contract for config and prompts
  (discodon), scriob's credential cascade, samsung's OpenRouter findings, the
  memory grain generalization.
- **Obligations:** harden the release path (it has had real incidents, including
  an untagged PyPI publish) before it carries five consumers' eval
  infrastructure.
- **Normalization:** resolve the Python floor (core is ≥3.14; samsung's audit
  shows relaxing to 3.12 is a bounded change serving both discodon and the Pi);
  slim iam's dependency declaration toward its actual usage; define a
  supported-version window so consumer skew is bounded; stand up the JS/npm side
  of the monorepo.

### discodon

- **Adopts:** `observe` (dropping its pinned OTel stack), `models` (dropping
  hand-rolled provider plumbing), `agent-memory` (upgrading capped working notes
  to durable per-persona user facts), `iam` (dropping its OAuth client,
  break-glass hashing, and cookie mint/verify; keeping allowlist and transport),
  and — as first consumer of its own extraction — the `eval-*` packages, keeping
  its host adapters and surfaces local.
- **Contributes:** the eval system; the prompt-identity discipline and the
  registry seeding the shared store; the runtime-config precedence contract; the
  per-persona memory grain requirement; the import-boundary pattern and token
  pipeline for the chat kit.
- **Keeps:** the persona/entity/turn engine, prompt graph, and all product
  surface. Does not rebase onto `3tears-langgraph`; does not adopt LangSmith.
- **Normalization:** finish the in-flight eval schema work, then execute the
  pre-extraction unlock list its own eval docs specify; align on the resolved
  Python floor.

### metallm

- **Adopts:** `iam`, `backup`, `enforcement` (scriob proved all three), the
  shared stream protocol for its frontend, the `eval-*` packages (it has zero
  eval infrastructure today — the family's biggest gap), and eventually
  `scrape`.
- **Contributes:** it already contributed most of the platform; newly: the
  sycophancy-judge prompt, the PII sanitization wrapper, machine-written prompts
  as the store's second consumer, and its frontend as the chat-kit feature
  reference.
- **Drops:** local auth, local backup, bespoke enforcement tests, raw stream
  filtering, and eventually its fetch path.
- **Normalization:** close the version lag first — every other adoption assumes
  a current pin.

### scriob

- **Adopts:** `eval-{contracts,analysis}` for its continuity-judge corpus
  (turning its append-only baselines log into the family's trend machinery); the
  headless chat kit (keeping its look).
- **Contributes:** the stream-transport/client seed, the credential cascade, the
  git-as-L3 pattern for the prompt/config durable tier, and its standing role as
  the reference consumer.
- **Drops:** bespoke chat state machinery; the hand-maintained pin ritual once
  consumption standardizes.
- **Normalization:** smallest of any app — it is already the north star's shape.

### samsung-frame-art-loader

- **Adopts:** the tiled-image acquisition package (as first consumer, replacing
  its unwritten local plan), `eval-run` for its MCP-driver eval, and the shared
  MCP conventions it currently follows as prose.
- **Contributes:** the acquisition contract design, the OpenRouter findings, the
  Python-floor audit.
- **Drops:** the legacy acquisition glue (already scheduled for deletion) —
  deleted, not lifted.
- **Normalization:** minimal and aligned with planned work; its Pi weight
  constraint is a design input the slim packaging must satisfy, not a blocker.

### hallucinote

- **Adopts:** exactly one thing — a programmatic eval runner over its existing
  scenario briefs and canonical verdicts, replacing the human-operated ritual and
  enabling CI gating. Optionally, server-side telemetry.
- **Contributes:** the entire MCP conventions layer and its brief/rubric
  scenario schema.
- **Drops:** nothing — its architecture (no LLM client, no auth, no web stack)
  is correct for what it is.
- **Normalization:** none forced. Its Live-side stdlib-only contract is a
  packaging constraint the shared MCP layer must respect, and that constraint
  improves the shared design.

## 6. Open questions

1. **Python floor.** Relax 3tears core to ≥3.12 (recommended; serves discodon
   and the Pi) or move discodon to 3.14? Relaxing constrains 3.14-only features
   in core.
2. **Eval extraction timing.** "After discodon's in-flight schema work lands"
   gates the family's highest-value workstream on that work's completion. Is
   partial extraction (contracts first) worth a two-step migration?
3. **Eval package count.** If four packages proves heavy, `gen` folds into
   `run` — provided import paths are chosen day one so a later split isn't a
   breaking rename.
4. **dezoomify-rs licensing.** Verify before it becomes a managed dependency of
   an MIT-published family; shell-out is fine, vendoring the binary may not be.
   The AGPL-isolated sidecar is the in-house precedent if needed.
5. **Memory grain.** Does `agent-memory`'s keying support (agent, subject)
   today, or is schema work required for discodon's per-persona memories?
6. **npm publishing.** The chat kit needs a JS release lane in this monorepo;
   same-repo lockstep is recommended — a separate JS repo reintroduces a version
   matrix against the wire protocol.
7. **Version-window policy.** What consumer skew does the family support?
   metallm's lag went unnoticed; a stated window plus a check makes lag a
   signal instead of a surprise.
8. **samsung's "no backends" observability stance.** Its recorded position is
   reasonable for an appliance; `observe`'s zero-dep no-op core may satisfy
   both — reconcile explicitly rather than silently overriding a recorded
   decision.
9. **Eval presentation sharing.** The React eval-kit stays discodon-local until
   a second app wants the UI, not just the projections. Does it then join the
   chat kit's npm lane?
10. **Config-store promotion vs the "no config framework" philosophy.** A
    runtime-config package amends a recorded 3tears position. Likely
    reconciliation: the philosophy governs *library* config; the store is an
    *application* capability like iam. Rule explicitly, don't drift.
11. **Does the prompt-graph assembly engine ever promote?** Promote the
    assembly-provenance contract now; revisit the engine only if metallm's
    personality layer — the likeliest second consumer — pulls for it.
12. **Git-backed vs append-only prompt store.** Git-as-L3 is the principled fix
    for seed drift but adds a git tier some deployments may not want. Decide
    whether the shared store offers both durable tiers behind one contract, and
    whether config (§4.8) and prompts share the store or merely its contract.
13. **Where convergence decisions live.** This document proposes; each repo's
    governance must ratify what binds it. A lightweight cross-repo decision
    record — probably here in `3tears/docs/` — stops per-app sessions from
    re-deriving the direction.
