# knowledge-generalization-task-02: Domain-neutral binding vocabulary in core

**Status:** DRAFT — blocked on the registry contract (see "Open / pending"), same blocker as task-01.
Land in the same release wave as task-01; both are cross-repo renames and splitting them across waves
means two breaking changes to the hub instead of one.
**Scope:** `3tears-core` (`threetears.knowledge.concept_merge`) + `3tears-agent-knowledge`
(`ConceptCollection`), with call-site updates in `14-eng-ai-bot` and `14-eng-ai-bot-agents`.
**Origin:** `knowledge-generalization-design.md` D5. The knowledge layer is described as
domain-agnostic; its core concept type carries SQL vocabulary.

---

## Objective

Rename the concept **binding** fields in core so they describe their *contract* instead of their
first consumer's grammar. The contracts do not change — only the names, and only because the names
are the last place the word "sql" appears in a substrate that is supposed to be domain-neutral.

## The renames

| today (core) | becomes | the contract, **unchanged** |
|---|---|---|
| `sql_fragment` (`concept_merge.py:145`) | `binding_fragment` | curated raw material for the model; **never executed by the platform** |
| `datasource_table_ref` (`concept_merge.py:144`) | `binding_ref` | the address the agent uses to name the thing |
| `datasource_table_id` (`concept_merge.py:143`) | `binding_id` | the finer binding hint *within* the anchor |
| `build_table_ref` (`concept_merge.py:66`) | `build_binding_ref` | behaviour unchanged |

Why these renames are safe: each contract is **already** domain-neutral, and the docstrings say so.
`sql_fragment` is documented as *"curated raw material for the model — a filter / expression fragment
… never executed"*. That sentence is true of a kubectl label selector verbatim. `v014` calls
`datasource_table_id` *"a FINER binding HINT (which table within the datasource)"* — which generalizes
to "which resource within the target" with no semantic change.

So the two-level shape survives intact: **`target_id` anchors** (task-01, carries RBAC) and
**`binding_id` refines** (this shard, carries precision).

## Design constraints

- **Rename only. Change no behaviour.** `binding_fragment` stays never-executed — the datasource tool
  boundary owns SQL safety, and the equivalent boundary will own command safety. If this shard
  changes what any field *does*, it has escaped its scope.
- **`binding_ref` stays the thing the agent can actually address.** `_render_concept`
  (`middleware.py:963`) deliberately emits the resolved `schema.table` name and **never** the raw
  UUID, because *"the agent addresses tables by schema.table and has no tool to resolve a table UUID,
  so a raw-UUID binding is un-actionable governance."* That reasoning is domain-independent and must
  survive the rename verbatim — including the unresolved-binding fallback line at `middleware.py:991`.
- **The label is vocabulary, not structure.** `"Bound table:"` in `_render_concept` is a *word*, and
  words belong to the product — but changing it belongs to **task-03**, not here. This shard renames
  fields; task-03 makes the renderer speak the product's vocabulary. Do not do task-03's job.
- **No back-compat shim** (design rule 2). One commit, all call sites.

## Work

**`3tears-core` — `packages/core/src/threetears/knowledge/concept_merge.py`**
- `:143-145` field renames on `ConceptSnapshot`; `:66` `build_table_ref` → `build_binding_ref`.
- `packages/core/src/threetears/knowledge/__init__.py` — the `build_table_ref` export.
- Update the `ConceptSnapshot` docstring (`:98`) — it currently explains the binding in SQL terms.

**`3tears-agent-knowledge`**
- `collections.py:196` `_row_to_concept_snapshot`, `:474` `ConceptCollection.schema`, `:525`
  `list_visible_to_user` (the `datasource_table_id` param), `:583-585` the projection aliases.
- `middleware.py:963` `_render_concept` — field references only; leave the wording to task-03.

**`14-eng-ai-bot` / `14-eng-ai-bot-agents`** — hub concept collection + routes, the concept authoring
tools, the generated client, and a column-rename migration mirroring task-01's.

## Open / pending (why this is DRAFT)

**The registry contract**, same blocker as task-01, and this shard is where it bites hardest.
`ConceptCollection.list_visible_to_user` resolves `binding_ref` with a join **from 3tears code into a
hub-owned table**:

```
collections.py:584   LEFT JOIN datasource_tables dt ON dt.id = co.datasource_table_id
collections.py:581   dt.schema_name AS bound_schema_name, dt.table_name AS bound_table_name
```

3tears knows the product's binding-registry table *and its column shape* (`schema_name` /
`table_name` — themselves SQL words). A kubectl binding has no `schema_name`; it has a namespace and
a kind. So `build_binding_ref(schema, table)` cannot simply be renamed — **its arity and inputs are
domain-specific**, which is the one place a pure rename genuinely fails.

This is the strongest argument for the design doc's option 3 (denormalize `binding_ref` onto the
concept row, drop the join): it deletes 3tears' dependency on the product's binding-table shape
entirely, and `build_binding_ref` moves product-side where it belongs. The cost is staleness — the
join is what keeps `binding_ref` fresh across a rename today, and the `"(unresolved — bound table no
longer exists)"` fallback at `middleware.py:991` exists *because* the join can miss. Denormalizing
trades that live check for a write-time one.

**Decide the registry contract before starting.** It determines whether `build_binding_ref` stays in
core at all.

## Acceptance

- `grep -ri "sql_fragment\|datasource_table\|build_table_ref" packages/core/src/threetears/knowledge`
  returns nothing.
- `ConceptSnapshot` carries `binding_id` / `binding_ref` / `binding_fragment`; behaviour byte-identical.
- `binding_fragment` is still never executed anywhere in any repo (grep the call sites to prove it).
- `_render_concept` still emits a resolved name and never a raw UUID, and still renders the
  unresolved-binding fallback.
- Hub + agents call sites updated in the same wave, no aliases; `./scripts/check-all.sh` green.
