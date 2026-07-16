# knowledge-generalization-task-01: Generalize the knowledge anchor to `target_id`

**Status:** DRAFT — blocked on the registry contract (see "Open / pending"). The *field rename* is
decided; whether 3tears keeps naming the product's registry tables is not. Do not start until that
settles — it decides whether this shard is a rename or a re-architecture.
**Scope:** `3tears-core` (`threetears.knowledge`) + `3tears-agent-knowledge` (collections), with
matching call-site updates in `14-eng-ai-bot` (hub collections, routes, migrations) and
`14-eng-ai-bot-agents` (authoring tools, generated client). Cross-repo, one release wave.
**Origin:** `knowledge-generalization-design.md` D2 / D3 / D4. devops (kubectl / aws / flyctl) is the
second consumer of the knowledge layer; the anchor must stop being named after the first consumer.

---

## Objective

Rename the knowledge anchor from `datasource_id` to `target_id` across the knowledge layer — single,
required, **opaque to 3tears** — and delete the vestigial `namespace_id`. 3tears must never learn
what *kind* of target it holds; the consuming product's registry owns that.

## Why the anchor is not a grouping key (read this before touching anything)

`14-eng-ai-bot/src/aibots/hub/migrations/v014_knowledge_datasource_anchor.py`:

> the datasource is the sole routing / RBAC anchor — the `knowledge.<datasource>` namespace carries
> the datasource's `customer_id` and every playbook entry / concept binds to a REQUIRED
> `datasource_id`.

The anchor **carries customer ownership**. It feeds `three_scope_visibility_clause(...,
scope_namespace_type="knowledge")` in `packages/agent/knowledge/.../collections.py:346` and `:569`,
and its FK is `ON DELETE RESTRICT` (a target cannot be dropped while knowledge anchors to it).

**Consequence for devops:** a cluster / AWS account / Fly app must become a *registered,
customer-owned, RBAC-bearing entity* before knowledge can anchor to it. Anchoring devops knowledge by
`tags` is wrong — a tag carries no ownership, so the visibility clause has nothing to evaluate.

## Design constraints

- **Single and required. Do not add sibling anchors.** This repo already ran that experiment:
  `v012_playbooks.py:114-115` gave `playbook_entries` both `datasource_id` and `namespace_id`, both
  nullable; `v014` dropped `namespace_id` and made `datasource_id` NOT NULL. Adding `cluster_id` /
  `aws_account_id` / `fly_app_id` re-makes a mistake this codebase already unmade.
- **`target_id` is opaque in core.** No `target_kind` in `threetears.knowledge` unless the renderer
  proves it needs one (that is task-03's call, and it is Open in the design doc). Core must be able
  to merge and scope without knowing the kind.
- **No back-compat shim.** Per `14-eng-ai-bot/CLAUDE.md`, rename and update every call site in one
  commit. No `datasource_id` alias, no dual-read, no deprecation re-export.
- **Do not touch the merge algorithm.** `resolve_shadow_chains` (`chains.py:140`) never reads the
  anchor; anchor filtering happens in the serving SQL *before* the merge. This shard is fields and
  queries, not resolution logic.

## Work

**`3tears-core` — `packages/core/src/threetears/knowledge/`**
- `merge.py:173` — `EntrySnapshot.datasource_id` → `target_id`.
- `merge.py:174` — **delete** `EntrySnapshot.namespace_id`. It is a corpse: the hub dropped the
  column in `v014`, nothing in any of the three repos sets it, and it survives only because the
  retraction never propagated product → substrate. Removing it is the point of D4, not a drive-by.
- `concept_merge.py` — `ConceptSnapshot.datasource_id` → `target_id` (note: the concept's
  `datasource_table_id` is a *different* field and belongs to task-02).
- Update the D1 scope docstring table and every `:ivar` that names a datasource.

**`3tears-agent-knowledge` — `packages/agent/knowledge/.../collections.py`**
- `PlaybookEntryCollection.schema` (`:256`) and `ConceptCollection.schema` (`:474`) — rename the
  `datasource_id` column declaration.
- `list_visible_to_user` on both collections — the `datasource_id` keyword param → `target_id`, and
  the WHERE fragments at `:370-374` / `:590-596`.
- `_row_to_snapshot` (`:162`) / `_row_to_concept_snapshot` (`:196`) — the snapshot field.
- `integration.py:285` `retrieve_entries` / `:354` `retrieve_concepts` — the `datasource_id` param.
- `middleware.py:312` — `configurable["knowledge_datasource_id"]` → `knowledge_target_id`.

**`14-eng-ai-bot` (hub)**
- `hub/knowledge/playbook_collections.py` — schema + `list_enforcement_for_datasource` (name and
  param; the *shape* it returns is unchanged, it is still `EntryEnforcement`).
- `hub/knowledge/playbook_routes.py`, `hub/knowledge/tools.py` — call sites + API field names.
- A new migration renaming the columns on `platform.playbook_entries` / `platform.concepts`, keeping
  the FK and its `ON DELETE RESTRICT`. DDL only; follow `docs/how-to-add-a-migration.md` and the
  YugabyteDB DDL/DML separation rule that `v014` and `v020` both call out.

**`14-eng-ai-bot-agents`** — `devx/schema/tools/knowledge/create_entry.py` / `edit_entry.py`,
`devx/schema/prompt.py`, and the generated hub client.

## Open / pending (why this is DRAFT)

**The registry contract.** 3tears' agent-side SQL names hub-owned tables directly:

```
collections.py:372,594   (SELECT origin_datasource_id FROM datasources WHERE id = $n)   -- KNW-77 widening
```

The KNW-77 origin-link gather (`collections.py:361-374`) widens `target_id IN (D, P)` where
`P = D.origin_datasource_id` — a correlated subquery against the **product's** `datasources` table,
issued from **3tears** code. Renaming `datasource_id` → `target_id` on the knowledge row does not
address it: the substrate still knows the product's registry schema.

Resolve per the design doc's "Open — the registry contract" before starting:
1. neutral names as an enforced contract (`targets`, `origin_target_id`);
2. product-supplied resolution (inject the widening fragment/resolver);
3. denormalize (product resolves the widening set before the call).

Option 1 is a rename; options 2 and 3 change this shard's shape and size. **Pick first.**

## Acceptance

- `threetears.knowledge` exports no symbol and declares no field containing "datasource"; a
  `grep -ri datasource packages/core/src/threetears/knowledge` returns nothing.
- `EntrySnapshot.namespace_id` is gone; no consumer referenced it (verified — nothing set it).
- Both agent-side collections filter on `target_id` with the visibility clause unchanged and still
  evaluated SQL-side (the trust boundary is the SQL, never a Python post-filter).
- The hub reads and writes the renamed columns with the FK + `ON DELETE RESTRICT` intact; hub tests
  green with no shim.
- KNW-77 origin-link widening still gathers `{D, P}` for a linked target (behaviour unchanged
  whatever option the registry contract picks).
- `./scripts/check-all.sh` green.
