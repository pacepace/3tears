# migrations-task-01: Canonical Migration Runner in 3tears

**Status:** Ready for implementation
**Scope:** `3tears/packages/core/` (the runner), every 3tears package that ships schema (workspace, memory, tools, channels), `14-eng-ai-bot/` (hub adopts it). `(3tears)` label; cross-repo.

---

## Objective

There are two migration systems in the platform today. `14-eng-ai-bot/migrations/` uses Alembic; `3tears/packages/agent-workspace/src/threetears/agent/workspace/migrations.py` uses an in-package `MigrationRunner`. The split caused a real bug this session: workspace DDL existed in the package's `migrations.py` but not in the hub's alembic tree, so agent schemas provisioned by the hub were missing workspace tables. Resolution was to hand-mirror the DDL into an alembic migration, which now has to be kept in sync with the source in 3tears — a different version of the same bug.

Promote the 3tears `MigrationRunner` to the canonical schema-management tool. Every package (including hub-owned schema) declares its migrations in a standard, runner-compatible shape. The hub's broker invokes the runner, not alembic, to provision agent schemas. Alembic retires.

This is what Pace asked for explicitly: "we need this to be in 3tears and it needs to be done right so other systems including 14-eng-ai-bot can use it properly."

---

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| MIG-01 | `MigrationRunner` lives in `3tears/packages/core/src/threetears/core/data/migrations.py` (may need to move there from its current home) as the canonical tool. Tracks applied versions per-schema, idempotent, supports upgrade-head + rollback-one. | P0 |
| MIG-02 | Every 3tears package that ships schema contributes its migrations via a standard entry-point pattern: `<package>.migrations.register(runner)` adds version callables. Hub discovers + composes these entry points at startup. | P0 |
| MIG-03 | Hub-owned platform schema (customers, agents, namespaces, etc.) moves out of alembic and into a `14-eng-ai-bot/src/aibots/hub/migrations/` module using the same runner. Hub's platform schema and per-agent schemas share one runner surface. | P0 |
| MIG-04 | One-time migration translates existing alembic state (`alembic_version` table content + applied migrations) into the runner's version-tracking table. Post-migration, alembic tooling is gone from the repo. | P0 |
| MIG-05 | Runner supports two scopes: **platform schema** (one-time application on hub startup; owned by the hub) and **per-agent schema** (applied on agent provision; composed from every package's agent-scoped migrations). The runner knows which is which from the migration registration call. | P0 |
| MIG-06 | Every agent schema produced by the new runner has every package's migrations applied in order. Package ordering is declarative (each package declares `depends_on` on other packages); the runner topologically sorts. | P0 |
| MIG-07 | Dev + test ergonomics: a package's test suite can spin up a schema with just that package's migrations applied, not the full platform. Useful for isolated unit testing. | P1 |
| MIG-08 | Migration-authoring docs describe the one blessed way to add schema: a migration file in the package's `migrations/` directory following a template. One runner, one template, one path to write. | P0 |

---

## Design Context

The 3tears `MigrationRunner` today takes a `DataStore` (an asyncpg-shaped executor), tracks applied versions in a `_migrations` table per schema, and applies registered versions in order. It's the right shape; it needs expansion to cover:

- Multi-package composition: today each package owns a standalone runner; we need one runner that composes migrations from many packages into one ordered apply sequence.
- Platform-scope vs agent-scope: the hub's schema migrations apply once per platform; the package migrations apply once per agent schema. The runner must track both.
- Topological ordering: if memory migrations depend on context_items (which they do today, via `conversations.id` FK), ordering has to be correct every time regardless of which order `register` calls happen.

### What alembic bought us that the runner needs to also buy

- Online apply + rollback commands.
- A single version-tracking row (alembic's `alembic_version`) per schema.
- Auto-generated migration file boilerplate via `alembic revision --autogenerate` — less important; we generally hand-write migrations anyway.

Things alembic costs us:
- Two sources of truth for schema (the package's declared tables vs the alembic file).
- Command-line tooling mismatch between packages.
- The real bug from this session (workspace DDL out of sync).

### The runner's apply sequence for a new agent schema

1. Hub's broker receives first L3 query for a new agent → triggers `provision_agent_namespace`.
2. Provisioner creates the schema + sets search_path.
3. Provisioner calls `runner.apply_for_agent_schema(schema_name)`.
4. Runner loads the composed agent-scope migration list (topologically ordered across every registered package).
5. Runner walks the list, applies each with idempotent `CREATE TABLE IF NOT EXISTS` etc., records version in the schema's `_migrations` table.
6. On any failure: explicit rollback of the partial state, drop the schema, raise.

---

## Files to Create / Modify

### Create

- `3tears/packages/core/src/threetears/core/data/migrations/runner.py` — expanded `MigrationRunner` with multi-package composition + platform/agent scope. (May be a rename/promote of the existing file.)
- `3tears/packages/core/src/threetears/core/data/migrations/registry.py` — the entry-point discovery mechanism; packages register their migrations via `importlib.metadata` entry points or an explicit registration list.
- Per 3tears package: a `migrations/` subdirectory with individual migration modules following the blessed template.
- `14-eng-ai-bot/src/aibots/hub/migrations/` — platform-scope migrations, expressed in the runner's format (translation of existing alembic files).
- `14-eng-ai-bot/migrations/agent/translation_check.py` — one-time verification that the new runner produces a schema byte-equivalent to what alembic produced for existing agent schemas. Safety gate before alembic is retired.

### Modify

- `14-eng-ai-bot/src/aibots/hub/broker/migrations.py::run_migrations_for_schema` — call the runner instead of alembic.
- `14-eng-ai-bot/src/aibots/hub/broker/namespace_provisioner.py::provision_agent_namespace` — delegate to the runner for the migration step.
- Every 3tears package that ships schema — move migrations into the blessed shape.

### Retire

- `14-eng-ai-bot/migrations/` alembic tree — after translation + verification.
- `3tears/packages/agent-workspace/src/threetears/agent/workspace/migrations.py` standalone runner — replaced by the shared runner.

---

## Implementation Notes

1. Expand the runner first in its own branch: multi-package composition + topological order + scope separation. Unit tests cover ordering, missing-dependency errors, idempotent re-apply, rollback on failure.
2. Translate platform-schema alembic migrations to runner format. Checksum verification: apply runner to a fresh DB, apply alembic to another fresh DB, diff schemas — must match byte-for-byte.
3. Translate agent-schema alembic migrations (001, 002, 003) to runner format. Same checksum verification per agent schema.
4. Land the switchover in one coordinated PR: runner plugs into `provision_agent_namespace` and `hub/migrations`, alembic config + alembic files + alembic dependency are deleted in the same commit. Verification-in-place is the checksum gate already run in steps 2+3; no runtime feature flag, no dual-stack release window.
5. Delete the old 3tears package-local `migrations.py` files that the runner now supersedes in the same PR.
6. If the checksum gate is red, the switchover PR does not merge — fix the translation before flipping, do not ship the runner alongside alembic as a hedge.

---

## Anti-patterns

- **DO NOT** keep both systems running indefinitely. Dual systems are the problem this task exists to solve; stopping mid-migration preserves the bug class.
- **DO NOT** auto-generate migrations from pydantic models or table definitions. That's where alembic's autogenerate pays; it's also where it silently miscomputes destructive migrations. Hand-write migrations with intent.
- **DO NOT** let a package's migrations depend on the hub's concrete schema names (`platform.agents` etc.). Migrations should operate against the current schema via `search_path`, not hard-coded schema names. That way the same package migrations can run against a test harness or a real platform schema without edits.
- **DO NOT** skip the checksum verification step. Silent schema drift during migration translation would bite months later.
- **DO NOT** ship a `USE_NEW_MIGRATION_RUNNER` (or similarly named) runtime feature flag. Per `14-eng-ai-bot/CLAUDE.md` "NO BACKWARDS-COMPATIBILITY SHIMS": the switchover is gated by pre-merge checksum verification, then flipped atomically — no release runs with both systems. Runtime flags for dual-stack migration tooling reliably ossify and produce the exact "two sources of truth" bug this task exists to kill.
- **DO NOT** leave alembic artifacts in the repo "as a reference." The repo is the source of truth; keeping alembic files lets them rot and occasionally re-attract hand edits. Delete cleanly; git history preserves the translation reference if needed.

---

## Success Criteria

- [ ] One runner, one registration pattern, one template for writing migrations.
- [ ] Every existing alembic migration translated; checksum-verified against fresh schemas produced by each tool.
- [ ] Hub's `provision_agent_namespace` uses the runner.
- [ ] Alembic is gone from the repo (config, files, dependency).
- [ ] Test harness can spin up a package's schema in isolation.
- [ ] New migrations (from in-flight shards) author against the runner pattern.

---

## Verification

```bash
uv run --directory 3tears/packages/core pytest tests/unit/data/test_migrations_runner.py -v
uv run --directory 14-eng-ai-bot pytest tests/unit/hub/broker/test_migrations.py -v

# checksum verification: apply both systems to fresh DBs, diff
uv run --directory 14-eng-ai-bot python scripts/migration_translation_check.py
```
