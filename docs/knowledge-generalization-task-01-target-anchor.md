# knowledge-generalization-task-01: Align the knowledge anchor with the capability-source registry

**Status:** DRAFT — blocked **only** on the D3 vocabulary decision (`capability_source_id` vs
`datasource_id` vs renaming the table/package). Not blocked on anything structural. Once the name is
picked, this is a mechanical rename plus one field deletion.
**Scope:** `3tears-core` (`threetears.knowledge`) + `3tears-agent-knowledge` (collections), with
matching call-site updates in `14-eng-ai-bot` (hub collections, routes, migration) and
`14-eng-ai-bot-agents` (authoring tools, generated client). Cross-repo, one release wave.
**Origin:** `knowledge-generalization-design.md` D2 / D3 / D4.

> **This shard was rewritten 2026-07-15** after the design review found the registry generalization
> already shipped. The first version proposed a net-new `target_id` and declared a "registry
> contract" blocker. Both were wrong — see the design doc's "Corrections". The work is smaller than
> that version implied.

---

## Objective

Stop naming the knowledge anchor after **one kind** of the thing it points at, and delete the
vestigial `namespace_id`.

## Why this is a rename and not an architecture change

`playbook_entries.datasource_id` / `concepts.datasource_id` already FK to `platform.datasources`
(`v014_knowledge_datasource_anchor`), and that registry is **already** the generalized
capability-source registry — `CapabilitySourceEntity` (`packages/datasources/.../entities.py:141`),
discriminated by `CapabilitySourceKind` (`entities.py:54`: `DATASOURCE` | `API_IMPORT` |
`MCP_IMPORT`), with `Column("kind")` at `datasources/collections.py:228`.

**So the knowledge layer is already anchored to a generalized registry.** The anchor is simply named
after the registry's original kind. A future `kubectl` kind needs **no anchor change at all** — only
this rename, so the name stops lying.

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
  Core must merge and scope without it. (The renderer may need it — that is task-03's call.)
- **Do not touch the merge algorithm.** `resolve_shadow_chains` (`chains.py:140`) never reads the
  anchor; anchor filtering happens in the serving SQL *before* the merge. Fields and queries only.
- **No back-compat shim** (design rule 3): rename and update every call site in one commit.

## Work

**`3tears-core` — `packages/core/src/threetears/knowledge/`**
- `merge.py:173` — `EntrySnapshot.datasource_id` → the D3 name.
- `merge.py:174` — **delete** `EntrySnapshot.namespace_id`. Verified corpse: the hub dropped the
  column in `v014`, and nothing in any of the three repos sets it. It survives only because the
  retraction never propagated product → substrate.
- `concept_merge.py` — `ConceptSnapshot.datasource_id` → the D3 name (its `datasource_table_id` is a
  different field — task-02).

**`3tears-agent-knowledge` — `packages/agent/knowledge/.../`**
- `collections.py:256` / `:474` schema column; `list_visible_to_user` param + WHERE fragments
  (`:370-374`, `:590-596`); `_row_to_snapshot` (`:162`) / `_row_to_concept_snapshot` (`:196`).
- `integration.py:285` `retrieve_entries` / `:354` `retrieve_concepts` params.
- `middleware.py:312` — `configurable["knowledge_datasource_id"]` → the D3 name.

**`14-eng-ai-bot` (hub)** — `hub/knowledge/playbook_collections.py` (schema +
`list_enforcement_for_datasource`; the returned `EntryEnforcement` shape is unchanged),
`playbook_routes.py`, `hub/knowledge/tools.py`, plus a column-rename migration keeping the FK and its
`ON DELETE RESTRICT`. DDL only — follow `docs/how-to-add-a-migration.md` and the YugabyteDB DDL/DML
separation rule `v014`/`v020` both cite.

**`14-eng-ai-bot-agents`** — `devx/schema/tools/knowledge/create_entry.py` / `edit_entry.py`,
`devx/schema/prompt.py`, generated hub client.

**Stale-docstring sweep (in scope — it is the same bug class as `namespace_id`):**
`14-eng-ai-bot/src/aibots/hub/knowledge/tools.py:16` still documents entries as anchoring on
*"`datasource_id` / `namespace_id`"* — the design `v014` retracted. Fix it while renaming.

## Open / pending (why this is DRAFT)

**The D3 vocabulary decision only.** The registry entity is `CapabilitySourceEntity`, the table is
`platform.datasources`, the package is `threetears.datasources` — Fork-1 generalized the entity and
left the other two named after the original kind. Two names already exist for one thing; this shard
must pick, not add a third:

- **(a) `capability_source_id`** — align to the entity. Anchor becomes honest; table/package stay stale.
- **(b) leave `datasource_id`** — align to the table. Zero churn; the anchor keeps lying, and keeps
  implying devops knowledge needs a different anchor when it does not.
- **(c) rename table + package too** — correct and complete; much larger, and beyond this effort.

## Acceptance

- The knowledge anchor carries the D3 name in core, agent, and hub; no third name introduced.
- `EntrySnapshot.namespace_id` is gone (nothing referenced it — verified).
- Both agent-side collections filter on the renamed anchor with the visibility clause unchanged and
  still evaluated SQL-side (the trust boundary is the SQL, never a Python post-filter).
- The hub reads/writes the renamed column with FK + `ON DELETE RESTRICT` intact; no shim.
- KNW-77 origin-link widening still gathers `{D, P}` — behaviour byte-identical.
- `hub/knowledge/tools.py:16` no longer documents the retracted `namespace_id` anchor.
- `./scripts/check-all.sh` green.
