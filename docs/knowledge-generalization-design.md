# Knowledge generalization: one governed-knowledge layer for datasources *and* devops — design + decisions

**Status:** DECIDED (2026-07-15) — design captured; build sharded below.
**Driver:** devops (kubectl / aws / flyctl) becomes the **second consumer** of the knowledge layer.
Datasources were the first. The enhancement is **generic** 3tears framework work: the same governed
knowledge system must describe *how to use a datasource* and *how to use kubectl in our environment*,
without 3tears learning what either one is.

> Captured because the conclusion previously lived only in a design chat and was re-derived from
> scratch twice. Standing failures it records against: **(1)** the knowledge layer was described as
> "domain-agnostic" when its core types carry SQL vocabulary (`sql_fragment`, `datasource_table_ref`,
> `build_table_ref`) and its injected prompt says *"before writing SQL"*; **(2)** a dead field
> (`EntrySnapshot.namespace_id`) survived in 3tears for a decision the hub retracted in `v014`,
> because the retraction was never propagated. This file is the fix.

> **Design rules for this whole effort:**
> 1. **3tears carries shapes; products carry semantics.** The test is *"does 3tears have to
>    understand this to do its job?"* The merge does not need to know what a `sql_fragment` **means**
>    to carry it through a shadow chain — so it carries an opaque string and never the word "sql".
>    The analyzer *does* need to know what `required_predicates` means — so the analyzer lives in the
>    product.
> 2. **Do the structurally-best thing — no back-compat shims.** These renames break the hub. Per
>    `14-eng-ai-bot/CLAUDE.md` ("NO BACKWARDS-COMPATIBILITY SHIMS"), rename in 3tears and update every
>    call site in one commit. No dual-name aliases, no deprecation re-exports, no "one release" flags.
> 3. **No speculative generality.** `EntryEnforcement`'s own docstring sets the rule: *"deliberately
>    NOT a general SQL policy language; grow it only when a concrete trap needs a new field."* The
>    command-side enforcement shape is **not designed here** — no concrete kubectl trap exists yet to
>    design against. See "Open" below.
> 4. **Knowledge describes; it never executes.** See D12.

---

## The requirement (decided — user direction 2026-07-15)

> "The goal here is to move the concepts and generalizations into 3tears and the specifics into the
> products that consume 3tears. So the same knowledge system will be used for devops as will be used
> for datasources. The goal is twofold: we should be able to output conceptual information about the
> data (the what) and we should be able to output runbooks about the data (the how). The knowledge
> system is consumed by both humans and AI."

Distilled:
- **One knowledge system, two domains.** Datasources and devops targets are two *kinds* of the same
  governed thing. 3tears cannot special-case either.
- **Two outputs.** The *what* (conceptual information) and the *how* (procedural). These already map
  onto the existing `ConceptSnapshot` / `EntrySnapshot` split — no new entity is needed for them.
- **Two audiences.** Humans and AI consume the same resolved knowledge through different renderers.

---

## What 3tears owns, and what it does not

The governance semantics are expensive, subtle, and **completely domain-independent**. They stay in
3tears and every domain reuses them unchanged:

- the three-scope ladder (`Scope`, `derive_scope` — `packages/core/src/threetears/knowledge/scope.py:44`)
- shadow-chain resolution, nearest-scope-wins, whole-unit replace
  (`chains.py:140` `resolve_shadow_chains`, generic over `id_of` / `scope_of` / `origin_id_of`)
- the effective / layered view split (`merge.py:179`, `merge.py:211`)
- invariant vs situational tiering, the shared budget trim, the fail-closed injection seam
- draft harvest + promotion

Everything with a **grammar** belongs to the consuming product:

- the **target registry** (which cluster / account / app / datasource exists, who owns it)
- the **execution tool** that actually runs the SQL / the command
- the **structural analyzer** for that grammar (`sqlglot` for SQL; a command parser for kubectl)
- the **gate** that runs the analyzer immediately before live execution
- the **vocabulary** the renderer speaks ("before writing SQL" vs "before running a command")

The SQL implementation of this pattern already exists and is the reference:
`14-eng-ai-bot/src/aibots/hub/datasources/query_enforcement.py:337` (`analyze_query`, the pure
analyzer) wired by `14-eng-ai-bot/src/aibots/hub/datasources/tools.py:239` (`enforce_constraints`)
in front of the live `Driver`. **3tears contributes exactly one thing to that flow:** the
`EntryEnforcement` shape (`merge.py:90`), which the merge carries and never interprets. That is the
split working correctly, and it is the template for every future domain.

---

## Decisions

### D1 — The test: shapes ride in core, semantics live with the domain owner
3tears carries a field only if the merge/scope machinery can do its job **without knowing what the
field means**. `EntryEnforcement` passes (inert passthrough). `sql_fragment` fails — core does not
need the word "sql" to carry an opaque string.

### D2 — The anchor is the RBAC carrier, not a grouping key
`v014_knowledge_datasource_anchor` states it: *"the datasource is the sole routing / RBAC anchor —
the `knowledge.<datasource>` namespace carries the datasource's `customer_id`"*, with an
`ON DELETE RESTRICT` FK. This is **why** the anchor is single and required, and it is why grouping
devops knowledge by `tags` is wrong: a tag carries no ownership.

**Consequence:** a devops target (cluster / AWS account / Fly app) must become a **registered,
customer-owned, RBAC-bearing entity** — the same species as a datasource — before knowledge can
anchor to it. That is product work, and it is the real cost of this feature.

### D3 — Generalize the anchor to `target_id`; do not add sibling anchors
`datasource_id` → `target_id`: single, required, **opaque to 3tears**. 3tears never learns what kind
of target it is; the product's registry owns that.

The sibling-anchor alternative is already **empirically rejected in this codebase**: `v012_playbooks`
gave `playbook_entries` both `datasource_id` and `namespace_id` (both nullable);
`v014_knowledge_datasource_anchor` dropped `namespace_id` and made `datasource_id` NOT NULL. Adding
`cluster_id` / `aws_account_id` / `fly_app_id` would re-make a mistake this repo already unmade.

### D4 — `EntrySnapshot.namespace_id` is a corpse; remove it
`merge.py:174` declares it; the hub dropped the column in `v014`; **nothing in any of the three repos
sets it**. It is the retraction that never propagated from product to substrate. Delete it.

### D5 — Binding vocabulary is domain-neutral
The *contracts* are already domain-neutral; only the *names* are SQL. Rename accordingly:

| today (core) | becomes | the contract, unchanged |
|---|---|---|
| `sql_fragment` (`concept_merge.py:145`) | `binding_fragment` | curated raw material for the model; **never executed by the platform** |
| `datasource_table_ref` (`concept_merge.py:144`) | `binding_ref` | the address the agent uses to name the thing (`schema.table`; `namespace/deployment`) |
| `datasource_table_id` (`concept_merge.py:143`) | `binding_id` | v014's *"finer binding HINT — which table within the datasource"* → which resource within the target |
| `build_table_ref` (`concept_merge.py:66`) | `build_binding_ref` | unchanged behaviour |

"never executed by the platform" is true of a kubectl label selector verbatim. The two-level shape
(`target_id` anchors + `binding_id` refines) generalizes without alteration.

### D6 — Render vocabulary is product-supplied; 3tears owns render structure
3tears owns *which sections, in what order* (glossary before procedures, invariant before
situational, shadow disclosure). The product supplies the **words**. Today those words are hardcoded
SQL: `middleware.py:117-144` headers (*"before writing SQL"*, *"touches the tables or columns in your
query"*), `middleware.py:152` `_BLOCK_PREAMBLE`, and `"Bound table:"` in `_render_concept`
(`middleware.py:963`). Those are vocabulary, not structure, and vocabulary is the product's.

### D7 — A runbook is an **output**, not an entity
A runbook is what the system *produces* when asked "how do I do X". It can be produced from a single
entry, or from a composite (D8). It is a **render target**, never a row, and it needs no table.

```
stored:     concepts (the what) + entries (the how)
              ↓  retrieve → merge → resolve scope
resolved:   effective + layered views
              ↓  render, by audience
produced:   runbook (human)  |  injected governed block (AI)
```

### D8 — Composition is one entity with nullable `steps` — not a second entity
An entry with no steps is a **leaf**; an entry with steps is a **composite**. Both are entries.

The deciding argument is **shadow resolution across the two shapes**: platform authors
`deploy = [A, B, C]`; a customer shadows it with a one-line leaf (*"just run `make deploy`"*), or the
reverse. Nearest scope wins, whole-unit replace — coherent, desirable, and **free from the rule that
already exists**. Two entities forbid that override for no benefit.

The objection that `steps` "is not inert like `enforcement`, so it cannot ride as an optional column"
does not hold: the budget **already tiers** on a column — `always_inject` exempts an entry from the
trim entirely (`_split_invariant_entries`, `middleware.py:493`). "Has steps → all-or-nothing" is a
third tier in an existing tiering, not a new axis.

### D9 — Steps reference **lineage**, and resolve through the merge
A step points at an entry's **chain root**, not at text. The merge resolves that reference to *the
caller's effective winner*. A customer who shadows step B with B' gets B' composed into every
composite that references B, with no composite edit, for that customer only.

This is the entire reason composition belongs in this system rather than beside it. Inline-text steps
would make a composite nothing but an entry with a numbered body.

### D10 — Composites are all-or-nothing budget units
The current trim is greedy per-item (`_rank_and_trim_shared`, `middleware.py:582`): sort, accumulate,
stop at the budget. **Trimming step 3 of a 5-step composite hands the agent a confidently wrong
procedure.** A composite is included whole or excluded whole. This is genuinely new machinery.

### D11 — Fail closed. Missing governed knowledge is a critical failure, not a degradation
> "just like an agent without access to knowledge on a datasource it has fails closed. This is
> critical failure and it needs to be or it will be doing something wrong, wrong data, wrong
> kubectl."

An agent missing the knowledge that governs a target does not do *less* — it confidently does the
**wrong thing** to real data or a real cluster. So:

- an invariant (`always_inject`) that cannot render raises `GovernedKnowledgeRenderError`
  (`middleware.py:90`) — already built, already correct;
- a **broken step reference fails the same way**, and a composite spanning targets the caller cannot
  fully see is **not rendered at all**. A partial composite is worse than no composite.

This converts a referential-integrity problem into an availability problem **on purpose**.
Referential integrity on `steps` is therefore not optional (FK or write-time check).

### D12 — Knowledge describes; it never executes
The line: knowledge tells the agent what to do → the agent decides → the tool executes → enforcement
gates the execution. Composites sit **entirely on the left**.

The moment a step carries `on_failure` / `retry` / `timeout`, this is a workflow engine inside a
knowledge base — owning a DSL, a scheduler, and error semantics forever — and an executable step
bypasses both the agent's judgment *and* the enforcement gate. If deterministic devops execution is
wanted later, that is a workflow engine that **consumes** knowledge, not a knowledge feature.

### D13 — "chain" the noun is reserved; composition uses `steps` / leaf / composite
`chain` already means a **shadow lineage** in this codebase (`chains.py:69` `ResolvedChain`,
`chains.py:140` `resolve_shadow_chains`, `merge.py:77` `MAX_SHADOW_CHAIN_DEPTH`, *"the nearest-scope
member of a chain"*) — and it is the **semantic inverse** of composition:

| | existing `chain` | composition |
|---|---|---|
| members | N units, **one wins** | N units, **all run** |
| logic | disjunctive — pick the nearest | conjunctive — do them in order |
| axis | vertical, across scopes | horizontal, across steps |

Third meaning in play: LangChain (an LCEL runnable composition) is imported throughout
`middleware.py`. The two axes get clearly distinct words — **shadow chain** (which version wins) vs
**steps** (what order to do them in) — and "chain" never appears in the composition concept.

---

## Open

- **The registry contract — the deepest leak, and the one real unknown.** 3tears' *agent-side* SQL
  hardcodes **hub-owned domain tables**, not just SQL-flavoured field names:
  `SELECT origin_datasource_id FROM datasources` (`packages/agent/knowledge/.../collections.py:372`
  and `:594`, the KNW-77 origin-link widening) and `LEFT JOIN datasource_tables dt` (`:584`, which
  carries the bound `schema.table` name through with the concept). The substrate structurally depends
  on the product's registry schema. Renaming fields does not fix this. Three ways out, undecided:
  1. **Neutral names as a contract** — `targets` / `target_bindings` / `origin_target_id`. 3tears
     still knows table names, but every product must supply that shape. Cheapest; makes 3tears'
     implicit dependency explicit and enforced.
  2. **Product-supplied resolution** — inject the widening + binding-ref join as a fragment or
     resolver. Cleanest separation; most machinery.
  3. **Denormalize** — carry `binding_ref` on the concept row, resolve the widening set product-side
     before the call. Removes the join entirely, but `binding_ref` goes stale on rename; today's join
     is what keeps it fresh.

  Decide before task-01/02 start — it changes both. See those shards' "Open / pending".
- **Where the devops product lives.** A new domain module inside `14-eng-ai-bot` (alongside
  `hub/datasources/`) or a separate hub. The domain-module pattern ports either way, but it decides
  who owns the target registry. **Blocks nothing in the shards below**; blocks the product work.
- **Whether `target_kind` rides on the knowledge row** or resolves from the product's target
  registry. A turn is scoped to one target today (`knowledge_datasource_id` on `configurable`), so
  kind is arguably a per-turn property, not a per-row one — but that filter is optional, so a turn
  *can* span targets. Decide when the renderer needs it (task-03), not before.
- **The command-side enforcement shape.** Deliberately not designed — see design rule 3. When a
  concrete kubectl trap exists, it lands as a sibling of `EntryEnforcement` in core (inert
  passthrough) plus an analyzer in the product, exactly mirroring the SQL flow.

## Shards

| shard | scope | status |
|---|---|---|
| `knowledge-generalization-task-01-target-anchor.md` | `datasource_id` → `target_id`; delete `namespace_id` | DRAFT — blocked on the registry contract |
| `knowledge-generalization-task-02-binding-vocabulary.md` | `sql_fragment` / `datasource_table_*` / `build_table_ref` renames | DRAFT — blocked on the registry contract |
| `knowledge-generalization-task-03-render-layer.md` | product-supplied vocabulary; runbook + injected-block audiences | READY |
| `knowledge-generalization-task-04-composition.md` | `steps`, leaf/composite, all-or-nothing budget, cycles | DEFERRED |

Task-01 and task-02 are both cross-repo renames and should land in one release wave with the hub
call-site updates (design rule 2). Both are **DRAFT, not READY**: the registry contract (see "Open")
decides whether they are a rename or a re-architecture, and it must be settled first. Task-03 is
independent of that question and can start now.

Task-04 is designed but **not scheduled** — it is the devops system's likely next need, captured now
so the anchor decisions in task-01 are made knowing it is coming. Task-01's single-required-anchor
decision does **not** foreclose it (see D10/D11).
