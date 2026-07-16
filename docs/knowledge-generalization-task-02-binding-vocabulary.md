# knowledge-generalization-task-02: Domain-neutral binding vocabulary in core

**Status:** READY. (Was DRAFT on a "registry contract" blocker that the design review **withdrew** —
the tables in question are 3tears' own domain model, not a product's. See the design doc's
"Corrections".) Land in the same wave as task-01: both are cross-repo renames, and splitting them
across waves means two breaking changes to the hub instead of one.
**Scope:** `3tears-core` (`threetears.knowledge.concept_merge`) + `3tears-agent-knowledge`
(`ConceptCollection`), with call-site updates in `14-eng-ai-bot` and `14-eng-ai-bot-agents`.
**Origin:** `knowledge-generalization-design.md` D5.

---

## Objective

Rename the concept **binding** fields in core so they describe their *contract* rather than their
first consumer's grammar. Contracts do not change — only names, and only because the names are the
last place "sql" appears in the governance substrate.

## The renames

| today (core) | becomes | the contract, **unchanged** |
|---|---|---|
| `sql_fragment` (`concept_merge.py:145`) | `binding_fragment` | curated raw material for the model; **never executed by the platform** |
| `datasource_table_ref` (`concept_merge.py:144`) | `binding_ref` | the address the agent uses to name the thing |
| `datasource_table_id` (`concept_merge.py:143`) | `binding_id` | the finer binding hint *within* the anchor |
| `build_table_ref` (`concept_merge.py:66`) | see "The one rename that is not a rename" | — |

Each contract is **already** domain-neutral and the docstrings say so. `sql_fragment` is documented as
*"curated raw material for the model — a filter / expression fragment … never executed"*; true of a
kubectl label selector verbatim. `v014` calls `datasource_table_id` *"a FINER binding HINT (which
table within the datasource)"* → "which resource within the capability source", no semantic change.

The two-level shape survives: **the anchor carries RBAC** (task-01) and **`binding_id` refines**.

## The one rename that is not a rename — `build_table_ref`

```python
def build_table_ref(schema_name: str | None, table_name: str | None) -> str | None   # concept_merge.py:66
```

Its **arity and inputs are SQL-specific**. A kubectl binding has no `schema_name`/`table_name`; it has
a namespace and a kind. `ConceptCollection.list_visible_to_user` feeds it from a join:

```
collections.py:584   LEFT JOIN datasource_tables dt ON dt.id = co.datasource_table_id
collections.py:581   dt.schema_name AS bound_schema_name, dt.table_name AS bound_table_name
```

That join is **not** a leak — `datasource_tables` is 3tears' own domain model
(`DataSourceTableCollection`, `datasources/collections.py:478`), physically created by the host's
migrations (`14-eng-ai-bot/.../v001_initial_schema.sql`). 3tears models the table; the host
instantiates it.

But the *shape* it projects is SQL-shaped, so a signature rename alone will not carry a command kind.
Three options — **pick during this shard, it is small and local**:

1. **Kind-dispatched builders.** `build_binding_ref` becomes a small registry keyed by
   `CapabilitySourceKind`; the SQL builder is today's function verbatim. Symmetrical with the
   registry's existing kind-conditional `connection_config` (`datasources/collections.py:241`) — the
   house pattern.
2. **Move it out of core.** The *product* builds its own ref and the concept row stores the result.
   Cleanest; costs the live-resolution freshness the join provides today.
3. **Generalize the inputs** to an ordered tuple of parts. Fewest edits; loses the type meaning and
   invites a stringly-typed mess.

Option 1 matches what Fork-1 already did for `connection_config` and is the recommendation.

## Design constraints

- **Rename only. Change no behaviour.** `binding_fragment` stays never-executed — the datasource tool
  boundary owns SQL safety; the equivalent boundary will own command safety. If this shard changes
  what a field *does*, it has escaped scope. (`build_table_ref` above is the sole exception, and it is
  a shape change, not a behaviour change.)
- **`binding_ref` stays the thing the agent can actually address.** `_render_concept`
  (`middleware.py:963`) deliberately emits the resolved name and **never** the raw UUID: *"the agent
  addresses tables by schema.table and has no tool to resolve a table UUID, so a raw-UUID binding is
  un-actionable governance."* Domain-independent reasoning — must survive verbatim, including the
  unresolved-binding fallback (`middleware.py:991`).
- **The label is vocabulary; that is task-03.** `"Bound table:"` is a *word*, and words belong to the
  product — but changing it is task-03's job. This shard renames fields. Do not do task-03's work.
- **No back-compat shim** (design rule 3).

## Work

**`3tears-core`** — `concept_merge.py:143-145` field renames on `ConceptSnapshot`; `:66`
`build_table_ref` per the option chosen; `knowledge/__init__.py` exports; the `ConceptSnapshot`
docstring (`:98`), which explains the binding in SQL terms.

**`3tears-agent-knowledge`** — `collections.py:196` `_row_to_concept_snapshot`, `:474` schema, `:525`
`list_visible_to_user` (the `datasource_table_id` param), `:581-585` the join + projection aliases;
`middleware.py:963` field references only.

**`14-eng-ai-bot` / `14-eng-ai-bot-agents`** — hub concept collection + routes, concept authoring
tools, generated client, and a column-rename migration mirroring task-01's.

## Acceptance

- `grep -ri "sql_fragment\|datasource_table\|build_table_ref" packages/core/src/threetears/knowledge`
  returns nothing.
- `ConceptSnapshot` carries `binding_id` / `binding_ref` / `binding_fragment`; behaviour byte-identical
  for the SQL kind.
- `binding_fragment` is still never executed anywhere in any repo (grep the call sites to prove it).
- `_render_concept` still emits a resolved name, never a raw UUID, and still renders the
  unresolved-binding fallback.
- The chosen `build_binding_ref` shape can express a non-SQL binding without a further core change —
  demonstrated by a unit test constructing a hypothetical non-SQL ref (a test, not a shipped kind).
- Hub + agents call sites updated in the same wave, no aliases; `./scripts/check-all.sh` green.
