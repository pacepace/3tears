# Knowledge generalization: one governed-knowledge layer for every capability source — design + decisions

**Status:** DECIDED (2026-07-15) — design captured; build sharded below.
**Driver:** devops (kubectl / aws / flyctl) becomes the next **kind** of capability source. The
governed-knowledge layer must describe *how to use a datasource* and *how to use kubectl in our
environment* through one system, with the domain specifics living where they belong.

> **Revised 2026-07-15, same day, after an adversarial review of the first draft.** The first draft
> asserted that consuming *products* own the target registry and proposed a net-new `target_id`
> anchor. Both were wrong, and the review found the reason: **the registry generalization already
> shipped** (Fork-1 / `gu-task-08`). `platform.datasources` is already a *capability-source* registry
> with a `kind` discriminator, and it is **3tears-owned** (`threetears.datasources`). This effort is
> therefore much smaller than the first draft claimed: devops is a **new kind on an existing
> registry**, not a new registry. The retracted claims are recorded in "Corrections" below rather
> than quietly deleted, because the false version was committed and someone may have read it.

> **Design rules for this whole effort:**
> 1. **Reuse the generalization that exists.** Fork-1 already generalized the registry. Do not invent
>    a parallel vocabulary next to it; extend it. The first draft's `target_id` violated this rule.
> 2. **3tears carries shapes; the product carries grammar.** The test is *"does 3tears have to
>    understand this to do its job?"* The merge does not need to know what a `sql_fragment` **means**
>    to carry it through a shadow chain — so it carries an opaque string and never the word "sql".
>    The analyzer *does* need to know what `required_predicates` means — so the analyzer stays in the
>    product. (Note this is **not** the same as "3tears is domain-free" — see "What 3tears owns".)
> 3. **No back-compat shims.** These renames break the hub. Per `14-eng-ai-bot/CLAUDE.md`, rename and
>    update every call site in one commit.
> 4. **No speculative generality.** `EntryEnforcement`'s own docstring sets the rule: *"deliberately
>    NOT a general SQL policy language; grow it only when a concrete trap needs a new field."*
> 5. **Knowledge describes; it never executes.** See D12.

---

## The requirement (user direction 2026-07-15)

> "The goal here is to move the concepts and generalizations into 3tears and the specifics into the
> products that consume 3tears. So the same knowledge system will be used for devops as will be used
> for datasources. The goal is twofold: we should be able to output conceptual information about the
> data (the what) and we should be able to output runbooks about the data (the how). The knowledge
> system is consumed by both humans and AI."

Distilled:
- **One knowledge system, every kind of capability source.**
- **Two outputs** — the *what* and the *how* — which already map onto `ConceptSnapshot` /
  `EntrySnapshot`. No new entity is needed for them.
- **Two audiences** — humans and AI render the same resolved knowledge differently.

---

## What already exists (read this first — it is most of the answer)

**`threetears.datasources` owns the capability-source registry, and it is already kind-generalized.**

`CapabilitySourceEntity` (`packages/datasources/src/threetears/datasources/entities.py:141`):

> generalized from the former datasource-only entity (Fork-1, gu-task-08): the SAME
> `platform.datasources` registry now holds database datasources, external API imports, and MCP
> imports, discriminated by the `kind` field.

- `CapabilitySourceKind` (`entities.py:54`): `DATASOURCE` | `API_IMPORT` | `MCP_IMPORT`.
- `Column("kind", STRING_TYPE)` (`collections.py:228`); `TableSchema(name="datasources")`
  (`collections.py:210`).
- **Kind-conditional config** (`collections.py:241`): a `datasource` row keeps an encrypted JSON
  blob; `api_import` / `mcp_import` rows store a `secret_refs` `scheme://locator` in the same column.
- **Two axes, already separated**: `kind` (what sort of source) vs `DataSourceType` (the *driver*
  axis, meaningful only for `kind='datasource'`).
- `origin_datasource_id` (`collections.py:269`) + `resolve_origin_datasource_id()`
  (`collections.py:443`) — the platform-shared origin link the knowledge layer's KNW-77 widening uses.

**The knowledge anchor already points at this registry.** `v014_knowledge_datasource_anchor` FKs
`playbook_entries.datasource_id` / `concepts.datasource_id` → `platform.datasources(id)`. So knowledge
is *already* anchored to a generalized capability-source registry — the anchor is merely **named after
the registry's original kind**.

**Therefore the devops shape is:** a new `CapabilitySourceKind`, a way to reach that kind, and a
product-side tool + analyzer + gate. **Not** a new registry, and **not** a new anchor.

## What 3tears owns, and what it does not

The first draft got this wrong by over-simplifying to "3tears is domain-free". It is not, and
deliberately so:

**3tears owns:**
- the **capability-source registry** and domain model (`threetears.datasources` — *"the single source
  of truth for 'what is a datasource' across every 3tears consumer"*), including the `kind` axis;
- the **driver abstraction** behind a factory + extras (`datasources/drivers/`);
- the **governance substrate** (`threetears.knowledge`): the three-scope ladder (`scope.py:44`),
  shadow-chain resolution (`chains.py:140`, generic over `id_of`/`scope_of`/`origin_id_of`), the
  effective/layered split (`merge.py:179`, `merge.py:211`), invariant/situational tiering, the budget
  trim, the fail-closed injection seam, drafts + promotion.

**The consuming product owns:**
- the **agent tool** that executes (`14-eng-ai-bot/src/aibots/hub/datasources/tools.py`);
- the **structural analyzer** for that grammar (`hub/datasources/query_enforcement.py:337`,
  `analyze_query`, sqlglot);
- the **gate** wiring the analyzer in front of live execution (`tools.py:239` `enforce_constraints`);
- the **vocabulary** the renderer speaks.

3tears' contribution to that enforcement flow is exactly one thing: the `EntryEnforcement` shape
(`merge.py:90`), which the merge carries and never interprets. **That is the split working, and it is
the template for every new kind.**

---

## Decisions

### D1 — The test: shapes ride in the substrate, grammar lives with the product
`threetears.knowledge` carries a field only if the merge/scope machinery can do its job **without
knowing what the field means**. `EntryEnforcement` passes (inert passthrough). `sql_fragment` fails —
core does not need the word "sql" to carry an opaque string.

### D2 — The anchor is the RBAC carrier, not a grouping key
`v014_knowledge_datasource_anchor`: *"the datasource is the sole routing / RBAC anchor — the
`knowledge.<datasource>` namespace carries the datasource's `customer_id`"*, FK `ON DELETE RESTRICT`.
The anchor **carries customer ownership**; it feeds `three_scope_visibility_clause(...,
scope_namespace_type="knowledge")` (`collections.py:346`, `:569`).

**Consequence:** devops knowledge cannot be grouped by `tags` — a tag carries no ownership, so the
visibility clause would have nothing to evaluate. A cluster must be a **registered capability-source
row**, which is exactly what the existing registry provides.

### D3 — Align the anchor to the existing generalization; do not invent a third name
The registry's generalized entity is `CapabilitySourceEntity`. The knowledge anchor should say so:
`datasource_id` → **`capability_source_id`**.

> **Open — vocabulary.** The registry *entity* is `CapabilitySourceEntity` but the *table* is still
> `platform.datasources`, and the *package* is still `threetears.datasources`. Fork-1 generalized the
> entity and left the table and package named after the original kind. So there are already two
> names for one thing, and the anchor must pick one. Options: (a) `capability_source_id` — align to
> the entity, accept the table/package staying stale; (b) leave `datasource_id` — align to the table,
> accept the anchor staying stale; (c) rename table+package too — correct, much larger, and outside
> this effort. **Needs a decision before task-01.** The first draft's `target_id` is rejected: a
> third name for a thing that already has two.

**No sibling anchors, whatever the name.** This repo already ran that experiment: `v012_playbooks.py:114-115`
gave `playbook_entries` both `datasource_id` and `namespace_id` (both nullable); `v014` dropped
`namespace_id` and made `datasource_id` NOT NULL. Adding `cluster_id` / `aws_account_id` re-makes a
mistake this codebase already unmade — and is unnecessary, because `kind` already exists.

### D4 — `EntrySnapshot.namespace_id` is a corpse; remove it
`merge.py:174` declares it; the hub dropped the column in `v014`; **nothing in any of the three repos
sets it** (verified). It is the retraction that never propagated product → substrate.

### D5 — Binding vocabulary is domain-neutral
The *contracts* are already domain-neutral; only the *names* are SQL:

| today (core) | becomes | the contract, unchanged |
|---|---|---|
| `sql_fragment` (`concept_merge.py:145`) | `binding_fragment` | curated raw material for the model; **never executed by the platform** |
| `datasource_table_ref` (`concept_merge.py:144`) | `binding_ref` | the address the agent uses to name the thing |
| `datasource_table_id` (`concept_merge.py:143`) | `binding_id` | the finer binding hint *within* the anchor |
| `build_table_ref` (`concept_merge.py:66`) | `build_binding_ref` | see task-02 — arity is SQL-shaped |

`sql_fragment`'s own docstring — *"curated raw material for the model … never executed"* — is true of
a kubectl label selector verbatim.

### D6 — Render vocabulary is the product's; 3tears owns render structure
3tears owns *which sections in what order*; the product supplies *the words*. Today the words are
hardcoded SQL (`middleware.py:117-152` headers, `"Bound table:"` at `:989`).

### D7 — A runbook is an **output**, not an entity
> "a chain can produce a runbook, so can a single kubectl knowledge that doesn't chain." — user

A runbook is what the system *produces* when asked "how do I do X". It is a **render target**, never a
row.

```
stored:     concepts (the what) + entries (the how)
              ↓  retrieve → merge → resolve scope
resolved:   effective + layered views
              ↓  render, by audience
produced:   runbook (human)  |  injected governed block (AI)
```

### D8 — Composition is one entity with nullable `steps` — not a second entity
A leaf has no steps; a composite does. Both are entries. The deciding argument is **shadow resolution
across the shapes**: a customer can shadow a platform 3-step composite with a one-line leaf (*"just
run `make deploy`"*), or the reverse — coherent, desirable, free from the existing rule. Two entities
forbid that for no benefit. The "steps isn't inert like enforcement" objection fails: the budget
**already tiers on a column** (`always_inject`, `middleware.py:493`).

### D9 — Steps reference **lineage**, and resolve through the merge
A step points at an entry's chain root; the merge resolves it to *the caller's effective winner*. A
customer shadowing step B gets B' composed into every composite referencing B, no composite edit,
that customer only. This is the whole reason composition belongs in this system.

### D10 — Composites are all-or-nothing budget units
`_rank_and_trim_shared` (`middleware.py:582`) is greedy per-item. **Trimming step 3 of a 5-step
composite hands the agent a confidently wrong procedure.** Whole or nothing. Genuinely new machinery.

### D11 — Fail closed. Missing governed knowledge is a critical failure
> "just like an agent without access to knowledge on a datasource it has fails closed. This is
> critical failure and it needs to be or it will be doing something wrong, wrong data, wrong
> kubectl." — user

An agent missing governing knowledge does not do *less* — it confidently does the **wrong thing** to
real data or a real cluster. A broken step ref fails the turn closed, as an unrenderable invariant
already does (`GovernedKnowledgeRenderError`, `middleware.py:90`). This converts a
referential-integrity problem into an availability problem **on purpose**; integrity on `steps` is
therefore not optional.

### D12 — Knowledge describes; it never executes
Knowledge tells the agent what to do → the agent decides → the tool executes → enforcement gates the
execution. Composites sit entirely on the left. A step carrying `on_failure` / `retry` / `timeout` is
a workflow engine inside a knowledge base, and it bypasses both the agent's judgment *and* the
enforcement gate. **Most likely constraint to erode under delivery pressure — guard it in review.**

### D13 — "chain" is reserved for shadow lineage
`chain` already means a shadow lineage here (`chains.py:69` `ResolvedChain`, `chains.py:140`
`resolve_shadow_chains`, `merge.py:77` `MAX_SHADOW_CHAIN_DEPTH`) — the **semantic inverse** of
composition:

| | existing `chain` | composition |
|---|---|---|
| members | N units, **one wins** | N units, **all run** |
| logic | disjunctive — pick the nearest | conjunctive — do them in order |
| axis | vertical, across scopes | horizontal, across steps |

Plus LangChain (an LCEL composition) is imported throughout `middleware.py`. Use **`steps` / leaf /
composite**; "chain" must not appear in the composition concept.

---

## Corrections to the first draft (commit `80723df`)

Recorded, not deleted — the wrong version was committed and pushed.

| first draft claimed | reality | found by |
|---|---|---|
| the consuming product owns the target registry | **3tears owns it** — `threetears.datasources`, *"the single source of truth for 'what is a datasource' across every 3tears consumer"* | `docs/adoption/datasources.md` |
| generalize the anchor to a net-new `target_id` | the registry is **already** kind-generalized (Fork-1 / gu-task-08); the general term is **capability source** | `entities.py:54,141` |
| 3tears' SQL naming `datasources` / `datasource_tables` is "the deepest leak", blocking task-01/02 | **not a leak** — those are 3tears' own tables (`collections.py:210`). Blocker withdrawn; task-01/02 unblocked | `datasources/collections.py:210` |
| devops needs a new registration entity per target kind | devops needs **a new `CapabilitySourceKind`** on the existing registry | `entities.py:54` |

**Root cause:** the first draft reasoned from `threetears.knowledge` and the hub outward, and never
read `docs/adoption/` — the map written precisely to answer "what does 3tears already do". A design
that proposes a generalization must first check whether it shipped.

**What survives:** D1, D2, D4–D13 stand unchanged. Only the registry-ownership and anchor-naming
claims (D3) were wrong.

## Open

- **Anchor vocabulary** — `capability_source_id` vs `datasource_id` vs renaming the table/package.
  See D3. **Blocks task-01.**
- **A real (smaller) coupling finding.** `agent-knowledge` reaches the registry by **raw inline SQL**
  (`collections.py:372,594`: `SELECT origin_datasource_id FROM datasources`) rather than through
  `threetears.datasources` — which already exposes `resolve_origin_datasource_id()`
  (`datasources/collections.py:443`) doing exactly that. So the KNW-77 widening is duplicated, and
  the cross-package coupling is invisible to the `dependency.missing` enforcement check
  (`docs/separate-concerns-decisions.md`), which only sees *imports*. Not a blocker; worth a shard.
- **Does a command kind get a `Driver`?** `threetears.datasources.drivers` is a factory + extras over
  SQL backends, and `Driver.fetch()` is SQL-shaped. A `kubectl` kind does not fit it. Either a
  sibling execution abstraction in 3tears, or the product owns command execution outright. **Decide
  before any devops build**; it does not block task-01–04.
- **Is "capability source" right for a cluster?** A datasource sources data; an `api_import` sources
  tools; a cluster is a thing you *act on*. Arguably still a capability source — but it stretches the
  word, and the word is already load-bearing. **Ask before adding the kind.**
- **The command-side enforcement shape.** Deliberately not designed — design rule 4. When a concrete
  kubectl trap exists it lands as an `EntryEnforcement` sibling in core (inert) plus an analyzer in
  the product, mirroring the SQL flow.

## Shards

| shard | scope | status |
|---|---|---|
| `knowledge-generalization-task-01-target-anchor.md` | anchor rename; delete `namespace_id` | DRAFT — blocked on D3 vocabulary only |
| `knowledge-generalization-task-02-binding-vocabulary.md` | `sql_fragment` / `datasource_table_*` / `build_table_ref` | READY |
| `knowledge-generalization-task-03-render-layer.md` | product vocabulary; runbook + injected-block audiences | READY |
| `knowledge-generalization-task-04-composition.md` | `steps`, leaf/composite, all-or-nothing budget, cycles | DEFERRED |

Task-01 and task-02 are cross-repo renames and should land in one wave with the hub call-site updates
(design rule 3). Task-04 is designed but **not scheduled**; it is captured so task-01's anchor
decisions are made knowing it is coming — they do not foreclose it.
