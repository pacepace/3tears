# knowledge-generalization-task-05: Rename the registry table to `platform.capability_sources`

**Status:** READY. Decided 2026-07-15 (user) — the full sweep: rename the anchor column **and** the
table it points at, rather than half the fix.
**Wave:** land with **task-01** and **task-02** in one release (design rule 3).
**Scope:** `3tears-datasources` (the registry's owner) + every consumer of the table name across
`3tears` and `14-eng-ai-bot`. **Blast radius is wider than the knowledge layer** — this is why it is a
separate shard from task-01, not because it lands separately.
**Origin:** `knowledge-generalization-design.md` D3.

> **Not a knowledge-layer shard.** It lives in this effort because the knowledge generalization is
> what exposed the stale name, and because renaming the anchor column without the table would leave
> `capability_source_id → platform.datasources` — a half-rename that reads worse than either endpoint.

---

## Objective

Rename `platform.datasources` → `platform.capability_sources`, so the table's name matches the entity
it has held since Fork-1 / `gu-task-08`.

## Why

`CapabilitySourceEntity` (`packages/datasources/.../entities.py:141`) is explicit:

> generalized from the former datasource-only entity (Fork-1, gu-task-08): the SAME
> `platform.datasources` registry now holds database datasources, external API imports, and MCP
> imports, discriminated by the `kind` field.

Fork-1 generalized the **entity** and left the **table** named after the original kind. The result is
two names for one thing, and it is not cosmetic — it actively misleads:

- the adoption doc that describes this package was still documenting the pre-Fork-1 world until
  2026-07-15 (fixed in `bc6a632`), and it caused the first draft of this very design to conclude that
  devops needed a whole new registry;
- the knowledge anchor inherited the stale name and now FKs to a multi-kind registry under a
  single-kind name (task-01).

`datasource` remains a perfectly good word — it is the name of a **kind**
(`CapabilitySourceKind.DATASOURCE`), not the name of the registry.

## Design constraints

- **Rename only. No semantic change.** No column changes, no kind changes, no FK changes beyond
  repointing them at the new table name. If this shard changes behaviour, it has escaped scope.
- **The `kind` axis and the `DataSourceType` driver axis both stay exactly as they are.** `kind` says
  *what sort of source a row is*; `DataSourceType` says *which driver reaches it*, and remains
  meaningful only for `kind='datasource'` rows. This shard does not touch either.
- **The package / import path stays `threetears.datasources`.** Deliberately out of scope (D3):
  renaming a published PyPI distribution plus every consumer's imports is a different cost class, and
  it blocks nothing. **Recorded as remaining debt** — after this shard, the entity, table, and column
  all agree, and only the package name trails.
- **No back-compat shim** (design rule 3) — no view aliasing the old name, no dual-read.
- **DDL only; DDL/DML separation.** A YugabyteDB `ALTER TABLE ... RENAME` auto-commits; follow
  `docs/how-to-add-a-migration.md` and the rule `v014`/`v020` both cite.

## Work

**Migration (`14-eng-ai-bot`)** — `ALTER TABLE datasources RENAME TO capability_sources`, plus every
FK, index, and constraint that names the old table (including the knowledge FKs from task-01 and the
`origin_datasource_id` self-reference). `v001_initial_schema.sql` is regenerated for fresh deploys;
the migration brings existing deployments into agreement — the pattern `v014` documents.

**`3tears-datasources`** — `TableSchema(name="datasources")` (`collections.py:210`) and the raw SQL in
`resolve_origin_datasource_id` (`collections.py:470`).

**`3tears-agent-knowledge`** — the inline registry SQL: `SELECT origin_datasource_id FROM datasources`
(`collections.py:372`, `:594`). See the design doc's Open note — this shard only renames it; whether
`agent-knowledge` should be reaching the registry by raw SQL at all is a separate finding.

**Sweep** — `grep -rn "FROM datasources\|JOIN datasources\|platform\.datasources"` across `3tears` and
`14-eng-ai-bot`. The column `origin_datasource_id` is **out of scope**: it names the *link*, and
renaming it to `origin_capability_source_id` is defensible but is not required for the table rename to
be coherent. Decide explicitly rather than by omission.

## Open / pending

- **`origin_datasource_id`** — rename to `origin_capability_source_id` in this wave, or leave it? It
  is the same class of staleness, but it is a *column on* the registry rather than the registry
  itself, and including it widens the diff. Recommend including it: it is the same wave, the same
  reviewers, and leaving it means a third rename later.
- **`datasource_tables`** — the finer binding registry (`DataSourceTableCollection`,
  `datasources/collections.py:478`) is also named after the `datasource` kind, and a kubectl binding
  will not be a "table". Task-02 owns the *concept-side* binding vocabulary; this table's own name is
  the same question one level down. **Do not resolve it here** — it depends on task-02's
  `build_binding_ref` decision.

## Acceptance

- `platform.capability_sources` exists; `platform.datasources` does not; no aliasing view.
- Every FK / index / constraint that referenced the old name is repointed, `ON DELETE RESTRICT`
  preserved on the knowledge anchors.
- `grep -rn "platform\.datasources\|FROM datasources\|JOIN datasources"` across both repos returns
  nothing.
- Entity, table, and knowledge anchor column all agree; only the package name trails (known debt).
- No behaviour change: the `kind` and `DataSourceType` axes are untouched; hub + 3tears suites green.
- `./scripts/check-all.sh` green.
