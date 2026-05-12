# collections-task-05: Eliminate DATETIME_TYPE; converge on aware-UTC everywhere

**Status:** PROPOSED. Multi-shard cross-repo migration. Blocks metallm v0.5.0 → v0.6.0 sign-off.
**Scope:** `3tears` (all packages declaring `DATETIME_TYPE`), `14-eng-ai-bot` (hub tables), and `metallm` (raw `BaseCollection` consumers). Touches 43 column declarations + 5–7 DDL migrations + base.py strip removal + 6 defensive `tzinfo` wraps in non-collections code.
**Origin:** metallm `MessagesCollection.date_created` (TIMESTAMPTZ) was bound through `BaseCollection.save_entity`'s unconditional `tzinfo` strip with no per-column re-wrap (because `MessagesCollection` extends raw `BaseCollection`, not `SchemaBackedCollection`). Cache stored naive, asyncpg returned aware on cache-miss, `history_rows.sort` raised `TypeError: can't compare offset-naive and offset-aware datetimes` on every multi-turn conversation.

---

## Objective

Remove the **hybrid "naive UTC at rest, aware everywhere else"** convention from the entire 3tears workspace. Replace `DATETIME_TYPE` (TIMESTAMP / naive) with `DATETIMETZ_TYPE` (TIMESTAMPTZ / aware-UTC) for every datetime column. Internal code already holds aware-UTC throughout (per CLAUDE.md datetime rule); after this task the database also holds aware-UTC, so no boundary coercion exists between them.

**Why this is the RIGHT choice (evidence from collections-task-05-audit):**

- The current convention is hybrid by design. `DATETIME_TYPE` exists only because TIMESTAMP columns predate `SchemaBackedCollection`. The CLAUDE.md datetime rule already says "always timezone-aware UTC" — TIMESTAMP columns violate that on disk.
- Six defensive `if value.tzinfo is None: value.replace(tzinfo=UTC)` wraps exist in non-collections code (`agent-audit/envelope.py`, `agent-memory/retrieval.py`, `agent-memory/collections.py`, `agent-workspace/pin.py`, `core/backends/nats_proxy.py`, `registry/health.py`). Each wrap is a plaster over a leak. Eliminating `DATETIME_TYPE` removes the leak source.
- Enforcement coverage is incomplete. `test_column_type_alignment.py` only checks Column ↔ migration alignment; it does not enforce aware-UTC on writes. A raw-asyncpg consumer (like metallm's `MessagesCollection`) can still bind naive to TIMESTAMPTZ and silently corrupt the wire value on non-UTC hosts. After this sweep we add an AST walker that bans `Column(..., DATETIME_TYPE, ...)` so the loophole closes.
- PostgreSQL TIMESTAMP → TIMESTAMPTZ DDL conversion is **lossless** and automatic. Existing data persists unchanged (the Postgres timestamps are already in UTC; only the type tag changes).

**Why this unblocks metallm:**

After the sweep, the strip in `base.py:773-776` is gone. metallm's `MessagesCollection` (raw `BaseCollection` subclass with TIMESTAMPTZ `date_created`) writes aware datetimes through to L1, L2, and L3 — all three layers consistent. The multi-turn `TypeError` resolves through the convergence rather than via a metallm-local migration to `SchemaBackedCollection`.

---

## Column Migration Census

From the audit report (collections-task-05-audit):

### 3tears packages — 23 `DATETIME_TYPE` columns

| Package | Collections | Columns |
|---------|---|:---:|
| `core` | (test fixture examples) | 2 |
| `agent-memory` | Memory, MemoryRefs, MemoryChunks, MemoryMedia | 7 |
| `agent-tools` | ContextItem | 3 |
| `agent-workspace` | Workspace, WorkspacePin (×2 instances), Workspace fixtures | 3 |
| `conversations` | Conversation, ConversationRef | 3 |
| `agent-acl` | Group, RoleAssignment (some still DATETIME_TYPE) | 2 |
| `agent-audit` | (already DATETIMETZ_TYPE — no migration) | 0 |
| **Subtotal** | | **20** (+ 3 fixture/example sites) |

### 14-eng-ai-bot — 20 `DATETIME_TYPE` columns

| Subsystem | Tables | Columns |
|-----------|---|:---:|
| `hub.agents` | agents, agent_tokens | 4 |
| `hub.customers` | customers, users, api_keys | 6 |
| `hub.datasources` | datasources, agent_data_versions | 2 |
| `hub.channels` | channels | 2 |
| (other hub tables) | (per audit) | 6 |
| **Subtotal** | | **20** |

### Total: 43 columns to flip + 5–7 DDL migrations + base.py simplification + 6 defensive wraps to remove.

---

## Shard Plan

Shards run in dependency order. Phases A and B can parallelize internally; phases C and D must serialize after A+B.

### Phase A — 3tears package sweeps (parallel)

Each shard: flip every `DATETIME_TYPE` Column declaration in its package to `DATETIMETZ_TYPE`, ship a corresponding alembic / SQL migration that runs `ALTER TABLE ... ALTER COLUMN ... TYPE TIMESTAMPTZ`, update any package-local tests that asserted `tzinfo is None`, run the package's own test suite + the workspace-level enforcement tests.

- **A1: 3tears agent-memory** (7 columns) — `packages/agent-memory/`
- **A2: 3tears agent-tools** (3 columns) — `packages/agent-tools/`
- **A3: 3tears agent-workspace** (3 columns) — `packages/agent-workspace/`
- **A4: 3tears conversations** (3 columns) — `packages/conversations/`
- **A5: 3tears agent-acl residual** (2 columns) — `packages/agent-acl/`
- **A6: 3tears core fixtures** (2 columns) — `packages/core/` test fixtures referencing `DATETIME_TYPE`

**Common output per A-shard:**
- New migration file in the package's migrations dir
- Updated `Column(..., DATETIMETZ_TYPE, ...)` declarations
- Test updates: any test asserting `args[N].tzinfo is None` for these columns flipped to `args[N].tzinfo is UTC`
- Green `pytest packages/<pkg>/tests/`

### Phase B — 14-eng-ai-bot hub sweep (parallel with A)

- **B1: 14-eng-ai-bot hub agents + customers + api_keys** (10 columns) — `src/aibots/hub/agents/`, `src/aibots/hub/customers/`
- **B2: 14-eng-ai-bot hub datasources + channels + remaining** (10 columns) — `src/aibots/hub/datasources/`, `src/aibots/hub/channels/`, others

**Common output per B-shard:**
- Hub migration files (cross-repo coordination per TASK_TEMPLATE.md "(3tears)" rule)
- `Column(..., DATETIMETZ_TYPE, ...)` flips
- `tests/enforcement/test_column_type_alignment.py` continues to pass
- Green `./scripts/test-backend.sh tests/`

### Phase C — base.py strip removal (after ALL of A + B)

- **C1: 3tears core simplification** — `packages/core/src/threetears/core/collections/base.py`
  - Delete `base.py:773-776` (the unconditional strip)
  - Delete `_coerce_datetime_for_write` in `schema_backed.py` (the DATETIME_TYPE write path)
  - Delete the `DATETIME_TYPE` branch in `_normalize_write_value` (`schema_backed.py:1061-1062`)
  - Simplify `_coerce_datetime_for_write_tz`: drop the "naive arrives because save_entity stripped" defensive re-wrap; the naive branch becomes an assertion or a `raise` because all callers now pass aware
  - Mark `DATETIME_TYPE` constant as `# DEPRECATED — kept temporarily for downstream import compatibility; no production code may use it. Removal targeted for collections-task-06.` or remove outright if no remaining imports
  - Update `test_schema_backed.py`: tests assuming `DATETIME_TYPE` exist (`test_aware_datetime_becomes_naive_utc`, etc.) — convert to assert TIMESTAMPTZ behavior or delete if redundant with existing TIMESTAMPTZ tests

### Phase D — defensive wrap removal + enforcement test (after C)

- **D1: drop the 6 defensive wraps** in non-collections code (per audit):
  - `agent-audit/envelope.py` — relax the `tzinfo is None: raise` if it's now unreachable
  - `agent-memory/retrieval.py`, `agent-memory/collections.py` — drop the wraps
  - `agent-workspace/pin.py` — drop the wrap
  - `core/backends/nats_proxy.py` — drop the wrap
  - `registry/health.py` — drop the wrap
- **D2: add AST enforcement** — new test in both repos:
  - `3tears/packages/core/tests/enforcement/test_no_datetime_type_columns.py`
  - `14-eng-ai-bot/tests/enforcement/test_no_datetime_type_columns.py`
  - Walks every `Column(...)` declaration in source. Fails if any uses `DATETIME_TYPE`. Allows `DATETIMETZ_TYPE`.
  - This is the timebomb-defuser. Once a future contributor reaches for `DATETIME_TYPE` by reflex, CI catches them at the declaration site.

### Phase E — metallm verification (after D)

- **E1: metallm restart + multi-turn smoke** — verify the original metallm bug (history_rows tz mismatch) is gone via MCP browser test. No metallm code change required.
- **E2: optional follow-on** — file collections-task-06 to migrate metallm's `MessagesCollection` and `ConversationsCollection` to `SchemaBackedCollection` for full canonical-pattern adoption (not blocking metallm v0.5.0 sign-off; quality-of-life refactor).

---

## DDL Migration Pattern

Every Phase A and B shard needs at least one alembic migration. Pattern (idempotent, YugabyteDB-safe):

```python
def upgrade() -> None:
    op.execute("""
        ALTER TABLE memories
            ALTER COLUMN date_created TYPE TIMESTAMPTZ
                USING date_created AT TIME ZONE 'UTC',
            ALTER COLUMN date_updated TYPE TIMESTAMPTZ
                USING date_updated AT TIME ZONE 'UTC'
    """)
```

The `AT TIME ZONE 'UTC'` clause is critical. PostgreSQL interprets a TIMESTAMP column's bare value as the **session timezone** when converting to TIMESTAMPTZ. We assert "this naive value semantically represented UTC" via `AT TIME ZONE 'UTC'` so the wire value is byte-stable across hosts. Without this clause, a TIMESTAMP cell holding `2026-01-02 03:04:05` becomes `2026-01-02 03:04:05+05:00` (or whatever the session tz is) — silently shifted.

The migration is idempotent: running it twice on a TIMESTAMPTZ column is a no-op (Postgres returns "type unchanged"). Wrap in `IF` checks if defensive idempotence per metallm CLAUDE.md migration policy is needed.

---

## Tests Affected

### Tests that will need updates (assert on naive values today)

From the audit:
- `3tears/packages/core/tests/unit/collections/test_schema_backed.py::test_aware_datetime_becomes_naive_utc` — currently asserts `args[8].tzinfo is None`; convert to TIMESTAMPTZ-form or delete
- Any per-package test asserting `tzinfo is None` on returned entities

### Tests that should keep passing (load-bearing)

- `test_column_type_alignment.py` (both repos) — verifies Column ↔ migration alignment. Stays green throughout the sweep as long as each shard ships migration + Column flip together.
- `test_datetime_stringification.py` (hub) — bans internal `.isoformat()`. Unchanged.
- `test_datetimetz_keeps_aware_utc_on_insert`, `test_datetimetz_naive_input_wrapped_with_utc`, `test_datetimetz_cas_fence_keeps_aware_utc`, `test_fetch_normalizes_datetimetz_to_aware_utc` — already cover the path that becomes the only path. Stay green.

### Tests added (Phase D)

- `test_no_datetime_type_columns.py` (both repos) — AST walker banning `Column(..., DATETIME_TYPE, ...)`. New net coverage.

---

## Anti-patterns

- **DO NOT** ship the base.py strip removal (Phase C) before all consumer columns are flipped. The strip is currently load-bearing for `DATETIME_TYPE` collections; removing it before they migrate would break their cache↔DB read consistency.
- **DO NOT** skip the `AT TIME ZONE 'UTC'` clause in DDL migrations. Without it the bare TIMESTAMP value gets re-interpreted in session tz on the conversion, silently shifting the wire value.
- **DO NOT** keep `DATETIME_TYPE` "for backwards compat". Either remove the constant or mark it deprecated with the enforcement test wired up. A "deprecated but importable" constant is a timebomb in five years.
- **DO NOT** change column declarations and DDL in separate commits/shards. The alignment enforcement test will fail in between, and code review loses the "these go together" coupling. One shard = one Column flip + one migration.
- **DO NOT** weaken `_coerce_datetime_for_write_tz`'s naive branch silently. After Phase C the code path is unreachable in production but must still raise with a clear message if hit, to surface drift.

---

## Dispatch / Subagent Notes

This task uses the strict-git, strict-bash dispatch template. Each Phase A and B shard runs in its own subagent (worktree isolation per SR-S-04). Phase C, D, E run in the main session because:

- Phase C touches 3tears core that A/B shards read from — race-prone in worktrees
- Phase D requires C done; running in main session avoids worktree merge conflicts on small file edits
- Phase E is verification, not implementation

The strict-bash subagent rules apply: no compound commands, no `2>&1` pipes, no `| tail`, no quoted SQL on CLI. See `~/.claude/.../memory/feedback_no_compound_commands.md`.

---

## Success Criteria

- [ ] Zero `DATETIME_TYPE` Column declarations in 3tears packages (verified by `test_no_datetime_type_columns.py`)
- [ ] Zero `DATETIME_TYPE` Column declarations in 14-eng-ai-bot (verified by mirror enforcement test)
- [ ] `BaseCollection.save_entity` no longer strips `tzinfo` (verified by reading base.py + `test_schema_backed.py`)
- [ ] Six defensive `tzinfo is None` wraps removed (verified by audit grep returning empty)
- [ ] All 3tears + 14-eng-ai-bot tests green
- [ ] metallm multi-turn conversation works end-to-end via MCP browser (the original failing path)
- [ ] No `TypeError: can't compare offset-naive and offset-aware datetimes` in `metallm/logs/api.log` after a 3-turn conversation
- [ ] Enforcement test added in both 3tears and hub repos so the convention can never regress

---

## Verification

After Phase E1:

```bash
# 3tears full suite
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears && uv run pytest packages/

# Hub full suite
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/14-eng-ai-bot && ./scripts/test-backend.sh

# metallm: drive a 3-turn conversation through the MCP browser, confirm no
# TypeErrors, no Background-task-failed, message persistence consistent.

# Audit greps that should ALL be empty
grep -rn "DATETIME_TYPE\b" /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears/packages/ /Users/pace/crypt/pub/dev-wsl/vscode/3tears/14-eng-ai-bot*/src/
grep -rn "if .*tzinfo is None.*replace.tzinfo=UTC" /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears/packages/ /Users/pace/crypt/pub/dev-wsl/vscode/3tears/14-eng-ai-bot*/src/
```

---

## Enforcement Test Suggestions

After Phase D2 lands:

- [x] `test_no_datetime_type_columns.py` — banned constant in `Column()` decls. **Already in plan.**
- [ ] Future: AST test that bans `.replace(tzinfo=None)` outside of border-context files. Today the strip in base.py is the only legitimate use; once it's gone, the only reason to strip is for an external API that demands naive — that should be allowed at borders only, mirroring the `test_datetime_stringification.py` pattern.
