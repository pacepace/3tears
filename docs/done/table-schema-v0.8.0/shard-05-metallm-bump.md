# v0.8.0-task-05: Bump metallm 3tears pin to v0.8.0, validate Alembic auto-gen parity

## Objective

Bump metallm's 3tears dependency from v0.7.5 to v0.8.0 and validate that the v0.8.0 enriched TableSchemas produce zero phantom Alembic auto-generate output against the live prod schema. This is the **release-gate check** for the whole v0.8.0 initiative — if auto-gen produces drop-FK / drop-index / drop-column noise, the release isn't ready.

Ship as metallm v0.15.0.

This shard depends on the previous shards landing AND on 3tears v0.8.0 being tagged + published.

---

## Locked design decisions (from README.md)

This shard is integration + verification, no new design.

---

## Files to Modify (in the metallm repo)

- `api/Dockerfile` — bump `ARG THREETEARS_VERSION=v0.7.5` to `v0.8.0`, update the rationale comment block above the line
- `.github/workflows/tests.yml` — bump `ref: v0.7.5` to `v0.8.0`, update the rationale comment
- `api/pyproject.toml` — bump `version = "0.14.7"` to `version = "0.15.0"`
- `api/uv.lock` — regenerate via `uv sync` after the pin bump; 3tears packages will resolve to v0.8.0 in the lockfile

## Files to NOT Modify (in the metallm repo)

- `api/src/data/models.py` — the four factory calls (`memories_table(metadata)` etc.) stay exactly as they are. v0.8.0 factories have the same public signature; the only thing that changes is their internals.
- Any alembic migration. v0.8.0 introduces no schema changes; the existing migrations 001-089 are still authoritative for prod DDL.

---

## Alembic parity validation — the gate

This is the most important check in the entire v0.8.0 release. After bumping the pin and syncing, run Alembic auto-generate against the live prod schema (or a fresh testcontainer with all migrations applied) and verify the output is acceptable:

```bash
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/metallm
./scripts/dev-up.sh                          # start docker postgres
# Wait for migrations to complete (alembic 001-089 run at API startup)
cd api
uv run alembic revision --autogenerate -m "v0.8.0 parity check" --rev-id 090_parity_check
# Inspect the generated file at api/alembic/versions/090_parity_check.py
```

### What counts as a phantom migration (BLOCKING)

The generated migration's `upgrade()` and `downgrade()` MUST NOT contain any of these ops against the 7 tables registered from 3tears (memories, media, media_content, memory_chunks, conversation_memory_refs, context_items, mcp_tool_grants):

- `op.add_column(...)` — TableSchema is missing a column that prod has
- `op.drop_column(...)` — TableSchema declares a column prod doesn't have
- `op.create_index(...)` or `op.drop_index(...)` — index drift
- `op.create_foreign_key(...)` or `op.drop_constraint(...)` (for FK or PK constraints) — constraint drift
- `op.alter_column(type_=...)` — column type changed
- `op.alter_column(nullable=...)` — nullability changed
- `op.execute(...)` — anything inline

### What's permitted (NOT phantom)

The auto-gen output MAY contain these without failing the gate:

- `op.alter_column(server_default=...)` on `tsvector` columns. Postgres reports back a server_default for trigger-maintained columns that doesn't match the declaration; SQLAlchemy can't easily detect this. Document the specific occurrence inline in the parity-check migration file (as a comment) and consider it allowed noise.
- `op.alter_column(server_default=sa.text('...'))` where the diff is just whitespace / formatting on otherwise-equivalent expressions.
- Operations against tables that are NOT in the 7-table parity set (workspaces, ACL tables, etc. are not in metallm sa_metadata, so they shouldn't appear at all; if they do, that's a metallm registration bug, not a parity gate issue).

### Resolution workflow

If auto-gen produces a BLOCKING op:

1. Read each op
2. Identify which TableSchema (or which `*Model` SQLAlchemy declaration in metallm) is wrong
3. Fix the schema in 3tears (if it's a 3tears-owned table) or the metallm model (if metallm-owned)
4. Re-run auto-gen
5. Repeat until only permitted noise remains

Common causes of phantom migrations:
- TableSchema missing a column that prod has (e.g. forgot `Column("alias", STRING_TYPE, nullable=True)` from v0.14.6)
- TableSchema missing an index that prod has (e.g. `ix_memories_user_alias` — confirm this is in the `memories` TableSchema after shard 03; if it isn't, shard 03 is incomplete)
- TableSchema missing a composite FK that prod has (e.g. `media_content.(agent_id, media_id) → media`)

**DO NOT commit the parity-check migration file.** It's a diagnostic. Delete it after verification. The release ships with the existing 089 as head.

---

## Standard validation

After parity-check passes:

```bash
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/metallm
./scripts/test-backend.sh tests/unit/ tests/enforcement/ -q
# 2861+ passing (post-v0.14.7 baseline)
```

If the unit suite drops below the v0.14.7 baseline, investigate before shipping.

---

## Browser smoke test (post-deploy)

After v0.15.0 deploys:

1. Use the metallm MCP `memories` tool to write a memory with `memory_add` — should succeed (regression test for the v0.14.6 incident).
2. Use `memory_add(alias="cave-altar")` — should store the alias.
3. Use `memory_search(alias="cave-altar")` — should return the memory by alias.
4. Use `conversation_store(message_ids=[<id>])` — should write a chunk through the L1 cache without error.
5. Tail `logs/api.log` (or query loki for prod) for any new "no such column" or "no column named" errors — there should be none.

Any failure here means the parity check missed something; rollback to v0.14.7 and investigate.

---

## Release sequence

Per the established flow used across v0.14.5-v0.14.7:

1. Branch `hotfix/v0.15.0-3tears-v0.8.0-pin` off `develop`
2. Bump Dockerfile + tests.yml + pyproject.toml + uv.lock
3. Verify locally: `./scripts/test-backend.sh tests/unit/ tests/enforcement/ -q`
4. Commit + push, open PR to `develop`
5. Wait for CI green (`gh pr checks --watch`)
6. Pace reviews + merges to develop
7. Open PR from develop to main
8. Pace reviews + merges to main
9. Tag v0.15.0 + push (`git tag -a v0.15.0 -m "..." && git push origin v0.15.0`)
10. Wait for `build-images.yml` green
11. Write substantive release notes; `gh release edit v0.15.0 --notes-file ...`
12. Pace deploys; subagent confirms `memory_add` works on prod

---

## Anti-patterns

- DO NOT commit the parity-check migration file. It's diagnostic-only.
- DO NOT skip the parity check. The whole point of v0.8.0 is "TableSchema is the single source of truth for Alembic auto-gen." Skipping the check means we don't actually know if that holds.
- DO NOT bypass branch protection on main. Same `develop -> main -> tag` flow as every prior release.
- DO NOT change any metallm code beyond the four pin / version bumps. v0.8.0 is non-breaking on the public factory API; nothing else needs updating.
- DO NOT release v0.15.0 if Saoirse-flow memory_add / conversation_store regress on the smoke test. Roll back to v0.14.7.

---

## Success Criteria

- [ ] `api/Dockerfile` THREETEARS_VERSION bumped to `v0.8.0` with updated rationale comment
- [ ] `.github/workflows/tests.yml` ref bumped to `v0.8.0` with updated rationale comment
- [ ] `api/pyproject.toml` version bumped to `0.15.0`
- [ ] `api/uv.lock` regenerated; all 3tears packages resolve to `0.8.0`
- [ ] `alembic revision --autogenerate` against the live-migration-equivalent schema produces upgrade/downgrade containing only PERMITTED noise (TSVECTOR `server_default` drift, or formatting-only diffs on equivalent expressions) — NO `add_column` / `drop_column` / `create_index` / `drop_index` / `create_foreign_key` / `drop_constraint` / `alter_column(type_=...)` / `alter_column(nullable=...)` ops against the 7 critical tables
- [ ] 2861+ unit + enforcement tests passing
- [ ] PR opened, CI green, merged to develop, merged to main
- [ ] v0.15.0 tagged + pushed
- [ ] Docker images at `ghcr.io/pacepace/metallm-*:v0.15.0`
- [ ] Substantive release notes published on the GitHub Release
- [ ] Post-deploy smoke test passes (memory_add, alias storage, conversation_store)

---

## Verification

```bash
# Pre-PR (local):
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/metallm
./scripts/test-backend.sh tests/unit/ tests/enforcement/ -q

# Parity check (requires docker postgres running with all migrations applied):
./scripts/dev-up.sh
cd api
uv run alembic revision --autogenerate -m "parity check" --rev-id parity_check_temp
# Verify: api/alembic/versions/parity_check_temp_*.py upgrade()/downgrade() are pass-only
rm api/alembic/versions/parity_check_temp_*.py

# Post-deploy (production):
# Via metallm MCP:
#   - memory_add(content="test", alias="v015-smoke-1") -> [memory:<id>]
#   - memory_search(alias="v015-smoke-1") -> finds it
#   - conversation_store(message_ids=[<recent_msg_id>]) -> [memory:<id>]
```

---

## Enforcement Test Suggestions

- [ ] Drift risk: someone changes a 3tears TableSchema after v0.8.0 without coordinating with metallm. The Alembic parity check is the canonical guard but only runs manually. Suggested test: a CI step on the metallm repo that runs `alembic revision --autogenerate` and fails if the output is non-empty. Catches the v0.14.6-class divergence automatically. STRONGLY RECOMMENDED — flag as a separate follow-up task once v0.8.0 ships.
