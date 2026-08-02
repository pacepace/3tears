# Family Convergence: One Platform, Five Sharp Apps

**Status:** Proposal — 2026-08-02
**Scope:** 3tears, discodon, metallm, scriob, samsung-frame-art-loader, hallucinote

---

## 1. What this document proposes

That 3tears becomes the **single shared home** for every cross-app capability in the
family — the state substrate, evals, prompt management, LLM access, memory,
identity, observability, config, MCP conventions, chat UI, and content
acquisition — with each capability
**sourced from the app that already paid for it**, never built in 3tears from
scratch. Apps keep only what makes them singular; everything else they import.

Concretely:

- The family's five answers to "where does long-lived state live" converge on
  `threetears.core` collections under one **state doctrine** — Pydantic at the
  boundaries, collections for operational state, git-backed files for curated
  content (§4.1). Discodon's ad-hoc caches are the first migration; samsung
  converges on the store *protocol* rather than the framework.
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
- **Five answers to "where does long-lived shared state live"** (state survey,
  same date): metallm holds every persistent entity in core collections and
  keeps Pydantic out of its data layer entirely; scriob splits collections
  (control plane) from markdown-backed Pydantic (stories); samsung recorded a
  reasoned rejection and hand-built a structural match of core's store
  protocol; hallucinote lock-guards stdlib dicts, correctly; discodon holds a
  ~dozen ad-hoc caches — eleven state dicts/sets aliased across ~8 delegate
  classes with no lock, three separate threads reaching into event-loop dicts
  (one of them behind a lock), and an auth-token cache where a token revoked
  in one process stays valid in every other until restart.
- **Hand-rolled cache coherence, three times**: discodon built a ZMQ
  invalidation bus to keep one 60-second TTL dict coherent across processes;
  metallm mutates an unlocked path cache from request tasks and a NATS
  callback; scriob wholesale-clears a per-pod auth map from an epoch listener.
  Each is a private rebuild of the L2 invalidation the core library ships.
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
8. **State earns its store.** In-process state that outlives a request, is
   shared across tasks or threads, or is mutated from more than one place
   belongs in a collection, not a module-level dict. State that fails the test
   — request-scoped values, build-once lookup tables — stays a plain object.
   The discipline is for the state that bites, not a ceremony for all of it.

## 4. The solution

Ordered by theme: the state substrate beneath everything else, then the quality
system (evals, prompts), the AI substrate (models, memory, conversation
engines), platform services (identity, observability, config), agent and user
surfaces (MCP, chat UI), and content acquisition.

### 4.1 State substrate — `threetears.core` collections (exists; metallm lineage)

Everything below runs on long-lived process state — registries, session maps,
caches, per-user accumulators. The state survey (§2) found the family holds
that state five different ways, and that the differences are discipline, not
need. This section converges the discipline, not just the library.

#### The doctrine: three layers, already proven

The family's two deepest consumers arrived at the same split independently,
and it holds for everyone:

- **Boundaries are Pydantic.** Config, wire DTOs, LLM tool schemas —
  validated at construction, passed by value. metallm's data layer has zero
  Pydantic imports and its API layer is nothing but; samsung's HTTP models
  file is its only Pydantic file. Nothing here changes; every app already
  does this.
- **Operational state is collections.** Entities whose data lives in L1 with
  change tracking, optimistic locking, and a write-through path — control
  planes (scriob's tenants/users/tokens/usage), domain rows (metallm's 16
  entities), and the caches-with-invalidation every serving app keeps
  rebuilding by hand.
- **Curated content is files, with collections as the cache over it.** The
  content-repo pattern (§4.3): Pydantic models with lossless file round-trips
  own the authored domain (scriob's `MarkdownModel`), and a collection with a
  git L3 serves it fast.

The collision people expect — "we're a Pydantic shop; 3tears entities keep
their data in L1" — dissolves under the assignment. A Pydantic model is a
validated snapshot; a collection entity is a live proxy. Both are correct,
for different state, and discodon has already proved which one the entity
layer wants: its `LivePersonaStateView` and persona-tool getter closures are
a hand-rolled live proxy, built because snapshots went stale. What the
doctrine forbids is the third thing every app grew instead: the module-level
dict that is neither.

#### Why single-pod apps still qualify

Multi-pod is deliberately absent from principle 8's test. A single-process
asyncio app hits the same failure class: any read-modify-write spanning an
`await` can interleave, and "pure asyncio" apps never are — metallm has ~27
`asyncio.to_thread` call sites, and discodon runs three separate threads
that reach into its event-loop dicts (one of them behind a lock). The
collection path doesn't prevent interleaving; it converts the silent lost
update into a `ConcurrentModificationError` — a stack trace instead of a
heisenbug. And core is built for the small case: all three tiers are
optional on `CollectionRegistry.configure`, NATS is an extra, so an
L1-only process adopts the discipline with no infrastructure. When the app
later goes multi-pod, the code is already written.

The reason to converge this now rather than per-incident: **AI writes the
dicts.** Most family code is AI-authored, and the model's default for shared
state is a module-level dict — no lock, no invalidation, no persistence
contract. A narrow enforced pattern is what keeps generated code on rails
(the EAD argument, applied to state), and an enforcement scanner that flags
new module-level mutable dicts is cheap once the doctrine is stated (open
question 17).

#### What adoption looks like (metallm is the reference)

metallm's shape is the one to copy, not core's README: entities subclass
`BaseEntity` with **typed `@property` accessors** over `_get_raw`, so mypy
still checks every field access — the static-typing loss of dynamic
attribute access is real, and typed accessors are the family answer to it.
Two conventions ride along: names distinguish live proxies from snapshots
(an entity is not a DTO; don't let one impersonate the other), and Pydantic
models sit inside `serialize`/`deserialize` where rows need validation on
the way to storage.

One honest gap: **no family production consumer uses the sync subscript
bridge.** metallm and scriob both drive collections through the async API
plus typed accessors; the `users[user_id]` pull-through path is unproven at
family scale (open question 15).

#### Where the framework doesn't fit, converge on the contract

samsung evaluated core and recorded a rejection that stands: core's L1 is a
named in-memory SQLite *cache*, and an app whose SQLite file must *be* the
store — sync core, memory-capped Pi — needs a different shape. Its answer is
the family pattern for this case (principle 6): `SqliteDurableStore`
structurally matches core's `DurableStore` protocol without importing it, so
later adoption is an adapter, not a rewrite. The protocol should be
published and conformance-tested so the match can't silently drift (open
question 16). hallucinote stays out entirely — the Live-side stdlib-only
contract makes the question moot, and its lock discipline is already sound.

### 4.2 Evals — `3tears-eval-{contracts,run,gen,analysis}` (new; from discodon)

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
projections — with one boundary shift now in flight: discodon's chart-substrate
decision (2026-08-02, accepted) compiles a typed finding payload to a
**Vega-Lite spec** server-side in Python, rendered by `vega-embed` in the
browser and `vl-convert` headlessly for MCP and REST. The spec compiler is
portable Python over the projections and belongs in `eval-analysis` when it
extracts; pixels and theming stay app-side (Vega-Lite config is where an app's
palette lives). That is what lets an agent receive a chart over MCP without a
headless browser, and it shrinks open question 9 — sharing charts no longer
requires sharing React.

Donated content: metallm's sycophancy-judge prompt; hallucinote's
brief/rubric/verdict scenario schema. Two footnotes: 3tears' only in-house eval
machinery — scrape's runtime recipe-judge loop — becomes an internal consumer of
eval-run's judge primitives once they land; and the eval measure registry must
disambiguate its naming from the unrelated BI measures in
`datasources.definition`.

### 4.3 Prompt management — identity from discodon; durable tier from scriob's pattern

Prompts cross-cut evals and administration. Three planes, three answers:
**storage is shared, assembly stays app-local, measurement goes through
eval-contracts.**

#### Storage: two kinds of prompts, two homes

- **Product prompts** (judge defaults, generators, starter templates) are part
  of the app. They live in the app's repo and evolve with code review.
- **Instance content** (discodon's personas — backstories, cognitive styles,
  per-persona classifiers, tuned presets) are curated works living in an
  *instance* of the app. Their home is a **content repo**: plain diffable files
  in a git repo the app operates on, with the instance DB demoted from master
  to serving cache. Scriob (stories) and hallucinote (songs) already run this
  pattern.

The content repo changes the mechanics:

- Operator and machine edits become commits — scriob's `GitL3Backend` is the
  write-through precedent, and commit authorship gives actor attribution for
  free.
- Environment promotion (dev instance → prod instance) becomes merge or
  cherry-pick.
- In-app seeds become starter content, instantiated into a new instance's repo
  at first boot. Seed drift is not managed; it structurally disappears, because
  the app repo is no longer a live tier. The rare reverse flow — a tuned
  persona good enough to ship as starter content — is an explicit PR to the app
  repo.
- Secrets never enter the content repo. Credentials stay sealed in the DB
  store, referenced by name.
- Deployments that can't carry a git tier fall back to an append-only
  content-addressed store with lineage and a mechanical diff-against-seed —
  drift visible, never silent.

The registry surface itself (types, sections, rendering, content-hash dedup) is
promoted over the shared store contract (§4.9).

#### Assembly: app-local, emitting shared provenance

Discodon's prompt-graph DAG, metallm's personality node, and scriob's
per-object prompts are product cores; no second consumer is pulling for a
shared engine. What is shared: every engine emits an **assembly-provenance
record** — content hashes of each component that entered the prompt, plus a
composition/variant hash. Cheap for static and dynamic engines alike, and what
makes different engines comparable.

#### Measurement: A/B by variant identity

Content-addressed identity (SHA-256 as true identity, version numbers as human
convenience) and the apparatus/lever/label input classification move into
eval-contracts. Rendered text can never be the A/B unit — it differs every
turn. The lever is a *variant identity*; apparatus proof (component hashes,
frozen case sets, deterministic seeds) shows everything else held. Discodon
already ships this shape as run-scoped prompt overlays recorded in run
identity.

#### The variant lifecycle

The planes close into one promoted workflow: an edit creates a draft variant →
an eval campaign runs it as a cell against control → the verdict attaches to
the variant's content hash → promotion is gated on that evidence → the promoted
version merges in the content repo. Prompts ship the way code ships — through
gates. One decoupling note: eval extraction does *not* wait on the shared
store. The eval coupling is hashes, not imports.

#### The pattern generalizes

**App repo = the product; content repo = curated instance works; DB = cache
plus end-user data.** Four apps fit: scriob (stories), hallucinote (songs),
discodon (personas), samsung (theme briefs, work lists, review decisions) —
which satisfies the second-consumer rule for promoting `GitL3Backend` into a
shared package. The boundary test: *would a human review a diff of this?*
metallm's per-user prompts fail it — user data in a multi-user app stays in DB
rows.

Samsung adds the binary rule: **the content repo holds authored text plus
manifests that reference blobs by content hash; the bytes live in a blob tier**
(disk or object store, with backup — never git, and not git-LFS either).
Scriob already runs this split: git for text, S3 for blobs. A side benefit:
the instance's identity lives in its content repo rather than in whichever
path an env var points at, retiring the mistyped-path-bootstraps-an-empty-
instance failure class samsung has on file.

hallucinote is the pattern's next test, and the payoff is a capability, not
hygiene: its songs are per-branch SQLite binaries (`<slug>-<branch>.db`) —
the workflow is already git-branch-shaped, but a binary file can't merge, so
two users can't work the same song across machines. Demoting the DB to cache
behind a canonical lossless text projection — the same demotion scriob and
discodon make — turns collaboration into ordinary git: branch, edit, merge at
file granularity, captures following the blob rule. Merge semantics inside a
clip are genuinely hard (two people editing the same notes is a musical
conflict, not a textual one), but a binary conflict offers only "pick one";
text at least localizes the argument. No shared server enters hallucinote's
architecture — git is the sync transport, as everywhere else in the pattern.

### 4.4 LLM substrate — `3tears-models` (exists; metallm lineage)

Already the family standard: LangChain-native construction, usage tracking with
locked OTel span attributes, pricing, circuit breaking. The change is adoption
(discodon retires its hand-rolled OpenRouter/LangChain plumbing) and enrichment
(samsung's live-probed OpenRouter findings fold upstream rather than seeding
another first-party client). LangSmith is explicitly **not** the family pattern —
the standard is OTel via the usage tracker.

### 4.5 Memory — `3tears-agent-memory` (exists; metallm lineage)

The "Tom likes pizza" layer: fact extraction from conversation, hybrid vector
search, reranking, consolidation; production-proven. Every app that learns
durable user facts uses it. One generalization required: discodon needs
**per-persona subjective memory** — Bob's memories of Tom and Alice's memories of
Tom are different data, a character feature. If the current keying (shaped to
metallm's one-shared-agent grain) can't express an (agent, subject) pair, that
generalization is discodon's upstream contribution.

### 4.6 Conversation engines — organs shared, skeletons app-local

Discodon's persona/turn engine and metallm's personality graph stay put. Their
generic organs converge: memory extraction/retrieval (via agent-memory), the PII
sanitize/unsanitize wrapper around tool execution (metallm lineage), judge
prompts (eval content). The engines converge on the **eval-contracts turn/trace
observation shape**, so quality is measured identically across different
orchestrators without either being rewritten. Migrating onto `3tears-langgraph`
remains an option an app can take when its turn loop comes up for rewrite anyway
— it is not a convergence requirement.

### 4.7 Identity — `3tears-iam` + `3tears-agent-acl` (exist; scriob is proof-of-life)

The only implementation of passwords, tokens, OAuth, and RBAC in the family.
What changes:

- **metallm** retires its local auth and backup for the shared packages.
- **scriob's credential-cascade/secret-sealing layer** is promoted before a
  third copy gets written — metallm has already hand-copied it once.
- **discodon** adopts iam along the exact seam scriob proved: iam's
  OAuth/state/passwords/tokens replace its Authlib client, break-glass hashing,
  and cookie mint/verify; the allowlist, builtin owners, and cookie transport
  stay app-local.
- The "hardcoded owners + DB allowlist + break-glass password" trio —
  independently duplicated in discodon and scriob — is a candidate for an
  optional small-deployment iam module.

Hallucinote (stdio + loopback by design) never adopts iam.

### 4.8 Observability — `3tears-observe` (exists; metallm lineage)

One spine: consistent spans, cost/usage metrics, log correlation, one dashboard
set. metallm's thin re-export shim is the adoption playbook. Discodon drops its
independently pinned OTel stack; samsung and hallucinote gain telemetry they
currently lack (within samsung's recorded "no backends" stance — see open
questions — and hallucinote's server-side half only).

### 4.9 Config — promote the contract, not the system (from discodon)

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

Config and prompt content share **one store contract** (precedence resolver,
seed semantics, store Protocol, actor attribution, content-addressed versioning)
with a **pluggable durable tier**: operational config defaults to the DB store
(immediate-effect knobs, secrets-adjacent, nobody reviews a diff of a port
number), curated content defaults to the git tier (§4.3), and an instance
wanting git-backed config — the GitOps shape — gets it behind the same API.

Priority: second tier, behind evals/memory/chat — every app's config works today,
so the cost avoided is future divergence, not present duplication.

### 4.10 MCP conventions — into `3tears-mcp` (from hallucinote)

Hallucinote's stack is the family's most mature agent surface, and siblings
already cite it as precedent — as prose. The harvest, as code:

- the typed action registry with enforced structural invariants
- teaching errors that carry their own fix
- flat-schema synthesis over action unions
- the long-poll job pattern with its coupled timeout ladder
- the tri-state capability matrix — supported / not-implemented /
  upstream-can't, because "a binary supported/not is a lie"
- guides-as-resources
- the content-fingerprint version handshake for code vendored into host
  processes

Packaging constraint from the source: hallucinote's Live-side half is
contractually stdlib-only, so the shared layer must be consumable
server-side-only or vendorable-by-copy. (Lineage: hallucinote credits cordyceps
for the original pattern.)

#### Composition: capabilities contribute actions; the app owns the surface

Eval is the forcing case — it exposes MCP to create, run, and analyze evals,
and it is about to become a shared package. The rule that composes: **one MCP
surface per app, owned by the host's RBAC-gated `McpServer`; shared packages
never run their own servers — they ship typed action groups the host mounts.**
Discodon already runs this shape (one server aggregating eval, logs, entity,
and prompt tools), samsung serves HTTP and MCP from one process and port, and
hallucinote's action registry — populated by import side-effects of action
modules — is the registration mechanism itself, generalized. Behind the
server, `3tears-registry` routes calls across pods; registration and routing
stay orthogonal. There is no family-central controller: an agent connects to
an app, and the app decides exposure, naming, and RBAC. Eval's
create/run/analyze actions therefore ship as an action module defined against
eval-contracts and the storage Protocol; where that module lives is open
question 18.

### 4.11 Chat UI — a headless TypeScript kit (protocol from 3tears; seeds from scriob and metallm)

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

### 4.12 Tiled/zoomable images — a slim acquisition package (new; from samsung's design)

Tiled zoomable imagery (IIIF, Deep Zoom, Zoomify, slippy tiles) is a general
web content type — maps, research figures, pathology, art — not a museum niche.
The capability is built once, here, from samsung-frame-art-loader's
already-designed (and deliberately unwritten) acquisition contract: acquisition
method × source class × fetch status with partial-tiles as a first-class
outcome, safe subprocess invocation of `dezoomify-rs`, URL allowlisting,
integrity guards.

Packaging: a **slim leaf**, not inside `3tears-scrape` — scrape is the family's
heaviest slice and the first consumer is a memory-capped Pi. Scrape grows a
thin driver adapting the leaf, so future consumers arriving via scrape get
tiled images without knowing the art loader exists. The contract is
binary-agnostic — a native IIIF/DZI fetcher can replace the external binary
without an API break. Samsung's legacy glue is deleted, not lifted.

### 4.13 Scraping — consolidate on `3tears-scrape` (exists; faidh lineage)

Three stacks become one. Lowest urgency of the workstreams, but the end state is
that metallm's fetch/extract path and discodon's research scraping are consumers,
not implementations.

## 5. Implications per family member

### 3tears

- **Gains:** the eval package group (discodon), MCP conventions (hallucinote),
  tiled-image acquisition (samsung's design), the headless chat kit + an npm
  publishing lane, the shared runtime store contract for config and prompts
  (discodon), scriob's credential cascade, samsung's OpenRouter findings, the
  memory grain generalization, the Vega-Lite spec compiler (discodon's chart
  substrate), and samsung's L1-is-a-cache integration finding.
- **Obligations:** harden the release path (it has had real incidents, including
  an untagged PyPI publish) before it carries five consumers' eval
  infrastructure.
- **Normalization:** resolve the Python floor (core is ≥3.14; samsung's audit
  shows relaxing to 3.12 is a bounded change serving both discodon and the Pi);
  slim iam's dependency declaration toward its actual usage; define a
  supported-version window so consumer skew is bounded; stand up the JS/npm side
  of the monorepo; publish `DurableStore` as a conformance-tested contract
  (open question 16); grow `3tears-enforcement` a module-level-mutable-dict
  scanner once principle 8 is ratified (open question 17).

### discodon

- **Adopts:** `observe` (dropping its pinned OTel stack), `models` (dropping
  hand-rolled provider plumbing), `agent-memory` (upgrading capped working notes
  to durable per-persona user facts), `iam` (dropping its OAuth client,
  break-glass hashing, and cookie mint/verify; keeping allowlist and transport),
  and — as first consumer of its own extraction — the `eval-*` packages, keeping
  its host adapters and surfaces local. Under principle 8 (gated on the Python
  floor, open question 1): core collections for the caches whose coherence is
  hand-rolled today — the budget TTL cache first, retiring the ZMQ invalidation
  bus built solely to keep it coherent; the model registry, which today runs
  unlocked while its two sibling registries carry locks; the allowlist and
  MCP-token caches, which lack cross-process invalidation entirely — then the
  entity-manager context dicts as open question 14 resolves.
- **Contributes:** the eval system; the prompt-identity discipline and the
  registry seeding the shared store; the runtime-config precedence contract; the
  per-persona memory grain requirement; the import-boundary pattern and token
  pipeline for the chat kit.
- **Keeps:** the persona/entity/turn engine, prompt graph, and all product
  surface — and its Pydantic boundary layer everywhere it stands. Does not
  rebase onto `3tears-langgraph`; does not adopt LangSmith.
- **Normalization:** finish the in-flight eval schema work and the
  chart-substrate migration (Vega-Lite lands; `recharts` leaves the web bundle
  only when telemetry's chart migrates too — tracked app-side), then execute
  the pre-extraction unlock list its own eval docs specify; align on the
  resolved Python floor.

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
  a current pin. Then fold its self-rolled coherence into the platform it
  ships: the active-path cache — an unlocked dict mutated by request tasks and
  a NATS callback — becomes a collection with L2 invalidation.

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
  Python-floor audit, and the protocol-match adoption pattern — a local store
  tracking `DurableStore` structurally, adoptable later by adapter.
- **Drops:** the legacy acquisition glue (already scheduled for deletion) —
  deleted, not lifted.
- **Normalization:** minimal and aligned with planned work; its Pi weight
  constraint is a design input the slim packaging must satisfy, not a blocker.

### hallucinote

- **Adopts:** a programmatic eval runner over its existing scenario briefs and
  canonical verdicts, replacing the human-operated ritual and enabling CI
  gating. Optionally, server-side telemetry. And — the one new capability on
  offer — the content-repo demotion of its song DBs (§4.3): a lossless text
  projection in git with the SQLite file rebuilt as cache, which is what lets
  two users work the same song from different machines over plain git.
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
   chat kit's npm lane? The chart layer has left this question: compiled
   Vega-Lite specs ship from `eval-analysis` and any host renders them — only
   the surrounding React surfaces remain app-local.
10. **Config-store promotion vs the "no config framework" philosophy.** A
    runtime-config package amends a recorded 3tears position. Likely
    reconciliation: the philosophy governs *library* config; the store is an
    *application* capability like iam. Rule explicitly, don't drift.
11. **Does the prompt-graph assembly engine ever promote?** Promote the
    assembly-provenance contract now; revisit the engine only if metallm's
    personality layer — the likeliest second consumer — pulls for it.
12. **Content-repo mechanics.** The design is decided (one store contract,
    pluggable durable tier: git for curated content, DB for operational config
    — §4.3, §4.9); the mechanics aren't: repo-per-instance vs
    directory-per-persona, hosting expectations (a local bare repo must
    suffice — no GitHub dependency for a deployment), and how draft variants
    map onto branches vs uncommitted overlay commits.
13. **Where convergence decisions live.** This document proposes; each repo's
    governance must ratify what binds it. A lightweight cross-repo decision
    record — probably here in `3tears/docs/` — stops per-app sessions from
    re-deriving the direction.
14. **EntityManagerContext migration shape.** Discodon's eleven aliased state
    dicts are hot-path and single-writer-by-convention; do they become
    collections, or a plain owner object with the aliasing removed? Principle
    8 says "shared across threads → collection," but the EM loop's latency
    budget is real and unmeasured here — this needs a measurement, not a
    ruling.
15. **The sync subscript bridge.** No family production consumer uses it —
    metallm and scriob both drive the async API plus typed accessors.
    Discodon's sync call sites would be its first real exercise. Prove it
    there, or declare the async API the family pattern and the bridge a
    laptop convenience?
16. **`DurableStore` conformance.** samsung tracks the protocol structurally,
    on purpose, without the import. Publish the protocol with a copyable
    conformance test so the match breaks loudly instead of drifting?
17. **Enforcing principle 8.** `3tears-enforcement` is the natural home for a
    scanner that flags new module-level mutable dicts (with a waiver pragma
    for the legitimate ones). Scanner first, or doctrine ratification first?
18. **Where eval's MCP actions live.** §4.10's composition rule says shared
    packages ship typed action groups and hosts mount them. Does eval's module
    ship inside `eval-run`/`eval-analysis`, or as a separate `3tears-eval-mcp`
    so consumers that want the engine without an agent surface skip the
    server weight?
19. **The song-DB text projection.** hallucinote's collaboration payoff (§4.3)
    rests on a canonical lossless text format for ~27 tables of musical data,
    and on merge granularity chosen so two users usually conflict on different
    files. Format, per-file grain, and the rebuild-DB-from-text path are all
    undesigned — and clip-level merge semantics may deserve a "manual
    resolution only" rule day one.
