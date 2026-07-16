# knowledge-generalization-task-01: Rename the knowledge anchor to `capability_source_id`

**Status:** READY. (Was DRAFT pending the D3 vocabulary decision — decided 2026-07-15.)
**Wave:** land with **task-02** and **task-05** in one release. All three are cross-repo renames;
separate waves mean three breaking changes to the hub instead of one (design rule 3).
**Scope:** `3tears-core` (`threetears.knowledge`) + `3tears-agent-knowledge` (collections), with
call-site updates in `14-eng-ai-bot` (hub collections, routes, migration) and `14-eng-ai-bot-agents`
(authoring tools, generated client).
**Origin:** `knowledge-generalization-design.md` D2 / D3 / D4.

> **Rewritten 2026-07-15** after the design review. The first version proposed a net-new `target_id`
> and declared a "registry contract" blocker; both were wrong — the registry generalization already
> shipped (Fork-1 / `gu-task-08`) and the tables in question are 3tears' own. The file was previously
> named `...-task-01-target-anchor.md`; renamed because "target" is rejected vocabulary (D3), and a
> rejected name sitting in a filename is the same stale-name debt this effort exists to pay down.

---

## Objective

Rename the knowledge anchor to name **what it actually points at**, and delete the vestigial
`namespace_id`.

## Why this is a rename and not an architecture change

`playbook_entries.datasource_id` / `concepts.datasource_id` already FK to `platform.datasources`
(`v014_knowledge_datasource_anchor`), and that registry has been the generalized **capability-source**
registry since Fork-1 / `gu-task-08` — `CapabilitySourceEntity`
(`packages/datasources/.../entities.py:141`), discriminated by `CapabilitySourceKind`
(`entities.py:54`), `Column("kind")` at `datasources/collections.py:228`.

**The knowledge layer is already anchored to a generalized registry.** A future kubectl kind needs
**no anchor change at all** — only this rename, so the name stops lying.

## The name is already wrong — verified, not prospective

This is the argument for doing it now rather than deferring:

- `v014`'s FK is to `platform.datasources(id)` with **no kind constraint**;
- the hub's knowledge code applies **no kind filter** (its only `kind` hits are the draft-envelope
  discriminator — unrelated);
- `agent-knowledge` **never mentions `kind`** at all.

So an entry can already anchor to an `api_import` / `mcp_import` row **today**, under a column called
`datasource_id`. The lie is live. It is also the specific lie that led the first draft of this design
to conclude devops needed a brand-new anchor — it has already misled one reader, and it will mislead
the next.

## Why the anchor is not a grouping key (do not "just use tags")

`v014`: *"the datasource is the sole routing / RBAC anchor — the `knowledge.<datasource>` namespace
carries the datasource's `customer_id`"*, FK `ON DELETE RESTRICT`. The anchor **carries customer
ownership**; it feeds `three_scope_visibility_clause(..., scope_namespace_type="knowledge")`
(`agent/knowledge/collections.py:346`, `:569`). A tag carries no ownership — the visibility clause
would have nothing to evaluate.

## Design constraints

- **Single and required. No sibling anchors.** `v012_playbooks.py:114-115` gave `playbook_entries`
  both `datasource_id` and `namespace_id`, both nullable; `v014` dropped `namespace_id` and made
  `datasource_id` NOT NULL. Adding `cluster_id` / `aws_account_id` re-makes a mistake this codebase
  already unmade — and is unnecessary: `kind` already exists on the registry row.
- **Do not add `kind` to `threetears.knowledge`.** The kind lives on the registry row, one join away.
  Core must merge and scope without it. (The renderer may need it — task-03's call, not this shard's.)
- **Do not touch the merge algorithm.** `resolve_shadow_chains` (`chains.py:140`) never reads the
  anchor; anchor filtering happens in the serving SQL *before* the merge. Fields and queries only.
- **No back-compat shim** (design rule 3): rename and update every call site in one commit.
- **The table rename is task-05, not this shard.** This shard renames the *column*; task-05 renames
  `platform.datasources` → `platform.capability_sources`. They land in the same wave but shard
  separately because their blast radii differ.

## Work

**`3tears-core` — `packages/core/src/threetears/knowledge/`**
- `merge.py:173` — `EntrySnapshot.datasource_id` → `capability_source_id`.
- `merge.py:174` — **delete** `EntrySnapshot.namespace_id`. Verified corpse: the hub dropped the
  column in `v014`, and nothing in any of the three repos sets it. It survives only because the
  retraction never propagated product → substrate.
- `concept_merge.py` — `ConceptSnapshot.datasource_id` → `capability_source_id` (its
  `datasource_table_id` is a different field — task-02).

**`3tears-agent-knowledge` — `packages/agent/knowledge/.../`**
- `collections.py:256` / `:474` schema column; `list_visible_to_user` param + WHERE fragments
  (`:370-374`, `:590-596`); `_row_to_snapshot` (`:162`) / `_row_to_concept_snapshot` (`:196`).
- `integration.py:285` `retrieve_entries` / `:354` `retrieve_concepts` params.
- `middleware.py:312` — `configurable["knowledge_datasource_id"]` → `knowledge_capability_source_id`.

**`14-eng-ai-bot` (hub)** — `hub/knowledge/playbook_collections.py` (schema +
`list_enforcement_for_datasource` → `list_enforcement_for_capability_source`; the returned
`EntryEnforcement` shape is unchanged), `playbook_routes.py`, `hub/knowledge/tools.py`, plus a
column-rename migration keeping the FK and its `ON DELETE RESTRICT`. DDL only — follow
`docs/how-to-add-a-migration.md` and the YugabyteDB DDL/DML separation rule `v014`/`v020` both cite.

**`14-eng-ai-bot-agents`** — `devx/schema/tools/knowledge/create_entry.py` / `edit_entry.py`,
`devx/schema/prompt.py`, generated hub client.

**Stale-docstring sweep (in scope — same bug class as `namespace_id`):**
`14-eng-ai-bot/src/aibots/hub/knowledge/tools.py:16` still documents entries as anchoring on
*"`datasource_id` / `namespace_id`"* — the design `v014` retracted. Fix it while renaming.

## Acceptance

- `grep -ri datasource packages/core/src/threetears/knowledge` returns nothing.
- `EntrySnapshot.namespace_id` is gone (nothing referenced it — verified).
- Both agent-side collections filter on `capability_source_id`, visibility clause unchanged and still
  evaluated SQL-side (the trust boundary is the SQL, never a Python post-filter).
- The hub reads/writes the renamed column with FK + `ON DELETE RESTRICT` intact; no shim, no alias.
- KNW-77 origin-link widening still gathers `{D, P}` — behaviour byte-identical.
- `hub/knowledge/tools.py:16` no longer documents the retracted `namespace_id` anchor.
- `./scripts/check-all.sh` green.
