# collections-task-03c: TIMESTAMPTZ Codec Bug + DATETIMETZ_TYPE

**Status:** SHIPPED. `DATETIMETZ_TYPE` is a first-class column tag in `SchemaBackedCollection`. Every TIMESTAMPTZ column on the hub side (rbac quartet + audit_events.timestamp) is re-tagged. CLAUDE.md datetime section carries the column-type-dependent write-boundary rule.
**Scope:** `3tears` repo (`packages/core` — write/read coercion + column tag), `14-eng-ai-bot` (hub) — re-tag 6 TIMESTAMPTZ columns, CLAUDE.md update.
**Origin:** `test_role_update_audit_row` failed with a CAS predicate mismatch on the role-update path during `collections-task-03`. The mismatch was invisible on the UTC CI host but reproducible on a developer machine in PDT (UTC-7).

---

## Objective

Fix a silent data-corruption bug class in `SchemaBackedCollection`: TIMESTAMPTZ columns written through the framework's `DATETIME_TYPE` coercion path were being shifted by the host's local timezone offset on non-UTC machines. The bug was invisible on UTC CI / prod hosts (the shift is a no-op when local == UTC) and reproducible on every PDT / EST / etc. developer machine.

The fix introduces `DATETIMETZ_TYPE` as a first-class column tag. Columns declared as `DATETIMETZ_TYPE` keep `tzinfo` aware on the write boundary — the coercion ensures aware-UTC for naive and aware inputs alike, so asyncpg's TIMESTAMPTZ codec sees a value it does not need to shift. `DATETIME_TYPE` (TIMESTAMP without TZ) keeps the existing strip-tzinfo behaviour. The two tags are mutually exclusive: a column is one or the other based on the L3 DDL.

The cautionary tale below is the most important artefact in this doc. A future contributor adding a TIMESTAMPTZ column will reach for `DATETIME_TYPE` by reflex (it is the "datetime" tag, after all). Read the bug class section before doing that.

---

## The Bug Class — read this before tagging a TIMESTAMPTZ column

### What asyncpg does

asyncpg's TIMESTAMPTZ codec — the function that converts a Python `datetime` into the wire format PostgreSQL expects for a `TIMESTAMP WITH TIME ZONE` column — runs roughly:

```python
def encode_timestamptz(value: datetime) -> bytes:
    return _encode_microseconds(value.astimezone(UTC))
```

The `astimezone(UTC)` call is the trap. For an aware datetime (one that carries `tzinfo`), `astimezone(UTC)` is a true no-op when the input is already UTC and a correct projection when the input is some other timezone. For a **naive** datetime (one whose `tzinfo` is `None`), `astimezone(UTC)` does NOT no-op. Python interprets the naive value as **the host's local timezone** — `time.localtime()`'s offset — and converts it to UTC.

On a UTC host, `local == UTC` and the conversion is a no-op. The bug is invisible.

On a PDT host (UTC-7 in summer, UTC-8 in winter), a naive datetime that semantically represents UTC `12:00:00` becomes UTC `19:00:00` on the wire. The row stores `19:00:00`. Every read returns `19:00:00`. Every CAS predicate that round-trips a stored value through the same coercion strips `tzinfo` again and shifts a second time — every CAS predicate fails to match the stored row, every UPDATE returns 0 rows, every CAS path raises `409 Conflict`.

### How it surfaced

`collections-task-03` converted `RoleCollection` to `SchemaBackedCollection` with `cas_column='date_updated'`. The role-update endpoint's flow:

1. `POST /admin/v1/customers/{cid}/roles/{rid}` writes the role. `date_updated` is `datetime.now(UTC)` — aware. The Collection's `_normalize_write_value` coercion (using `DATETIME_TYPE`, which strips tzinfo) emits a naive datetime to asyncpg. asyncpg's TIMESTAMPTZ codec calls `astimezone(UTC)` which interprets the naive value as local time on the dev machine. The wire value is shifted by 7 hours.
2. `GET /admin/v1/customers/{cid}/roles/{rid}` reads back. asyncpg's TIMESTAMPTZ decoder returns aware-UTC. The Collection deserializes; the entity's `date_updated` is the shifted value.
3. `PATCH /admin/v1/customers/{cid}/roles/{rid}` triggers the CAS UPDATE. The CAS fence is `WHERE date_updated = $N`; the bound value is the entity's `date_updated` from step 2. The Collection strips `tzinfo` again on the write boundary; asyncpg shifts a second time; the predicate value is now `original + 14h`, but the row contains `original + 7h`; predicate misses; UPDATE returns 0 rows; the endpoint surfaces `409 Conflict`.

The integration test `test_role_update_audit_row` exercised exactly this path. It passed on the UTC CI host (no shift, no mismatch). It failed locally on the principal's PDT machine. The surface symptom was a `409`. The root cause was a silent timezone shift two layers down.

### Why the existing `DATETIME_TYPE` was wrong for TIMESTAMPTZ

`DATETIME_TYPE` was introduced when every datetime column in the platform was `TIMESTAMP` (no tz). asyncpg's TIMESTAMP (without TZ) codec **rejects aware datetimes** with `DataError`. The `DATETIME_TYPE` write coercion strips `tzinfo` (after projecting to UTC via `astimezone(UTC).replace(tzinfo=None)`) so callers that hold aware-UTC throughout (per CLAUDE.md datetime rule) round-trip cleanly through asyncpg.

When TIMESTAMPTZ columns first appeared (the v016 RBAC tables and the v013 audit_events.timestamp, both pre-dating `SchemaBackedCollection`), they were tagged `DATETIME_TYPE` because that was the only datetime tag. The strip-tzinfo coercion was wrong for TIMESTAMPTZ — it produced a naive datetime that asyncpg's TIMESTAMPTZ codec then re-interpreted through local time — but the bug was invisible on UTC hosts and stayed invisible until a CAS path round-tripped through the column on a non-UTC dev machine.

The fix is to tag the column for what it actually is. TIMESTAMPTZ columns get `DATETIMETZ_TYPE`. The coercion for that tag ensures aware-UTC at the boundary, so asyncpg's `astimezone(UTC)` is a true no-op.

### What `DATETIMETZ_TYPE` does

Write coercion (`_coerce_datetime_for_write_tz`):

```python
def _coerce_datetime_for_write_tz(value: Any) -> datetime | None:
    if value is None:
        result = None
    elif not isinstance(value, datetime):
        result = value
    elif value.tzinfo is None:
        result = value.replace(tzinfo=UTC)   # naive -> aware-UTC
    else:
        result = value.astimezone(UTC)        # aware -> aware-UTC
    return result
```

Naive inputs are wrapped with UTC (the contract for `SchemaBackedCollection` is "core code holds aware UTC; if a value reached this point as naive it was a UTC-projected naive per CLAUDE.md datetime rule"). Aware inputs are normalized via `astimezone(UTC)` so non-UTC tzinfo values (rare but legal) flatten to UTC before the codec runs its own no-op `astimezone(utc)`.

Read coercion (`_coerce_datetime_for_read_tz`):

```python
def _coerce_datetime_for_read_tz(value: Any) -> datetime | None:
    if value is None:
        result = None
    elif not isinstance(value, datetime):
        result = value
    elif value.tzinfo is None:
        result = value.replace(tzinfo=UTC)   # naive -> aware-UTC (defensive)
    else:
        result = value.astimezone(UTC)        # aware -> aware-UTC (normalize)
    return result
```

asyncpg's TIMESTAMPTZ decoder already returns aware-UTC, so the typical path is a no-op. The defensive branch handles edge cases where a TIMESTAMPTZ value re-enters the read normalizer from a non-asyncpg source (L2 JSON deserialization, hand-rolled proxy pools that round-trip through string isoformat) and arrives naive. CAS predicates that round-trip through the Collection see a stable shape regardless of which pool answered.

Source: `packages/core/src/threetears/core/collections/schema_backed.py:111` (`DATETIMETZ_TYPE`), `:389` (`_coerce_datetime_for_write_tz`), `:429` (`_coerce_datetime_for_read_tz`), `:987` (write dispatch in `_normalize_write_value`), `:1013` (read dispatch in `_normalize_read_value`). Commit `2353cc6` (3tears).

---

## Design Decisions

### Two column tags, mutually exclusive

`DATETIME_TYPE` for `TIMESTAMP` (no tz). `DATETIMETZ_TYPE` for `TIMESTAMPTZ`. The L3 DDL decides which tag the column gets — the column type in the migration is the source of truth. There is no "auto-detect" path: the framework cannot know what type the database column is without a round-trip query, and a round-trip on every write would be a performance regression for a one-time configuration choice.

The contract is: declare the tag that matches the migration. If the migration says `TIMESTAMP`, the column is `DATETIME_TYPE`. If the migration says `TIMESTAMPTZ`, the column is `DATETIMETZ_TYPE`. Mismatch fails loud — `DATETIME_TYPE` on a TIMESTAMPTZ column reproduces the bug class above; `DATETIMETZ_TYPE` on a TIMESTAMP column raises `DataError` from asyncpg's TIMESTAMP codec on the first aware write.

### No back-compat shim

Per CLAUDE.md prime directive, no feature flag for "old vs new datetime coercion." Every TIMESTAMPTZ column on the hub side is re-tagged in commit `32d685f` (one-shot move). The `DATETIME_TYPE` write coercion is unchanged for genuine TIMESTAMP-without-tz columns; the new tag is purely additive for callers that need it.

3tears agent-tools / agent-memory / conversations are unaffected because all their declared tables use TIMESTAMP-without-tz. The new tag is purely additive for them — they would adopt `DATETIMETZ_TYPE` only if a future migration converts a column to TIMESTAMPTZ.

### Read coercion is defensive, not strictly necessary

asyncpg's TIMESTAMPTZ decoder already returns aware-UTC — the read path through asyncpg is a no-op. The defensive branch catches naive arrivals from non-asyncpg sources (the L2 KV path stores datetimes as ISO strings; the NATS proxy pool round-trips through JSON). Without the defensive branch, a CAS predicate that read through L2 and wrote through L3 could see a tzinfo-asymmetric value pair and mis-fence.

The defensive branch is a small invariant cost (one isinstance check + one branch) and a large robustness gain. Belt-and-suspenders.

### CLAUDE.md datetime section update

Lines 99-104 of the hub CLAUDE.md (commit `fa7352d`) carry the new column-type-dependent write-boundary rule:

> 2. YugabyteDB WRITE — column-type dependent:
>    - `TIMESTAMP` (no tz): convert aware → naive UTC. Use `DATETIME_TYPE` in `SchemaBackedCollection`.
>    - `TIMESTAMPTZ`: keep aware UTC. Use `DATETIMETZ_TYPE` in `SchemaBackedCollection`. **DO NOT strip tzinfo** — asyncpg's TIMESTAMPTZ codec calls `astimezone(UTC)` on every parameter; a naive datetime is interpreted as the client's LOCAL timezone, silently shifting the wall-clock value by the local-tz offset on non-UTC hosts (invisible on UTC CI/prod, breaks CAS predicates on PDT/EST/etc. dev machines). See `collections-task-03c` (commit `2353cc6` in 3tears, `32d685f` in hub).

The cross-reference back to this shard doc is intentional: a future contributor reading the CLAUDE.md note has a deeper artefact to consult. The CLAUDE.md note is the rule; this doc is the cautionary tale.

---

## What Landed

### 3tears core — commit `2353cc6`

- `DATETIMETZ_TYPE = "datetimetz"` constant. Source: `packages/core/src/threetears/core/collections/schema_backed.py:111`.
- `_coerce_datetime_for_write_tz` write boundary helper. Source: `packages/core/src/threetears/core/collections/schema_backed.py:389`.
- `_coerce_datetime_for_read_tz` read boundary helper. Source: `packages/core/src/threetears/core/collections/schema_backed.py:429`.
- Dispatch in `_normalize_write_value` (`:987`) and `_normalize_read_value` (`:1013`).
- L2 JSON serialization branch in `_decode_l2_value` (`:587`).
- Unit coverage:
  - DATETIMETZ_TYPE columns bind aware-UTC on insert.
  - Naive input bound to a DATETIMETZ_TYPE column is wrapped with UTC.
  - CAS fence on a DATETIMETZ_TYPE column is aware-UTC.
  - Read normalization wraps naive arrivals with UTC.

### Hub — commit `32d685f`

Six TIMESTAMPTZ columns re-tagged:

- `groups.date_created`, `groups.date_updated` — `src/aibots/hub/rbac/collections.py:160-161`.
- `group_members.date_added` — `src/aibots/hub/rbac/collections.py:411`.
- `roles.date_created`, `roles.date_updated` — `src/aibots/hub/rbac/collections.py:615-616`.
- `role_assignments.date_granted` — `src/aibots/hub/rbac/collections.py:802`.
- `audit_events.timestamp` — `src/aibots/hub/admin/audit.py:189`.

`GroupCollection` and `RoleCollection` both carry `cas_column='date_updated'`, so the CAS path immediately benefits. `GroupMember` / `RoleAssignment` / `AuditEvent` are append-only but their stored timestamps were silently TZ-shifted on dev hosts and now round-trip correctly.

### CLAUDE.md update — commit `fa7352d`

`/Users/pace/crypt/pub/dev-wsl/vscode/3tears/14-eng-ai-bot/CLAUDE.md` lines 99-104 — column-type-dependent write-boundary rule.

---

## Files Modified (load-bearing primitives)

- `packages/core/src/threetears/core/collections/schema_backed.py:111` — `DATETIMETZ_TYPE` constant.
- `packages/core/src/threetears/core/collections/schema_backed.py:389` — `_coerce_datetime_for_write_tz` write coercion.
- `packages/core/src/threetears/core/collections/schema_backed.py:429` — `_coerce_datetime_for_read_tz` read coercion.
- `packages/core/src/threetears/core/collections/schema_backed.py:987` — `_normalize_write_value` dispatch.
- `packages/core/src/threetears/core/collections/schema_backed.py:1013` — `_normalize_read_value` dispatch.
- `14-eng-ai-bot/src/aibots/hub/rbac/collections.py` — six TIMESTAMPTZ tags on the rbac quartet.
- `14-eng-ai-bot/src/aibots/hub/admin/audit.py:189` — `audit_events.timestamp` tag.
- `14-eng-ai-bot/CLAUDE.md` — datetime section update.

---

## Anti-patterns

These are **rejected on code review**, specific to this shard's lessons:

- **Reaching for `DATETIME_TYPE` on a TIMESTAMPTZ column because "datetime is the datetime tag."** It is not. `DATETIME_TYPE` strips tzinfo at the write boundary. asyncpg's TIMESTAMPTZ codec then re-interprets the resulting naive datetime through the host's local timezone. The bug is invisible on UTC hosts. Use `DATETIMETZ_TYPE` for every TIMESTAMPTZ column.
- **Trusting "passes on CI" as evidence the datetime path is correct.** CI runs on UTC. Local-timezone bugs are invisible there by construction. Run the datetime-touching integration tests on a non-UTC dev machine before declaring a CAS path stable. `test_role_update_audit_row` is the canonical reproducer; CAS-fenced TIMESTAMPTZ paths should have a similar local-host integration test.
- **Stripping `tzinfo` "because the column already stores UTC."** The column does not interpret the value — the codec does. asyncpg's TIMESTAMPTZ codec calls `astimezone(UTC)` on every parameter regardless of whether the column stores UTC. Stripping tzinfo turns an unambiguous "aware UTC" into an ambiguous "naive datetime that the codec has to interpret somehow," and the codec's interpretation is "local timezone."
- **Auto-detecting the column type at runtime.** The cost is a round-trip query on every write to discover the column type. The right place to declare the column type is the `TableSchema` declaration, matching the migration's column type. Mismatch fails loud — `DATETIMETZ_TYPE` on a TIMESTAMP column raises `DataError` on the first aware write.
- **Mixing `DATETIME_TYPE` and `DATETIMETZ_TYPE` on the same column "for compatibility."** A column is one or the other. The two coercions are mutually incompatible — one strips tzinfo, the other preserves it. There is no path where both make sense for the same column.
- **Adding a TIMESTAMPTZ column to a Collection without a CAS-path test.** CAS predicates are the surface where the bug manifests. A TIMESTAMPTZ column on an append-only Collection still needs read-back assertions, but the canonical surface is CAS. If your Collection has `cas_column` and the column is TIMESTAMPTZ, the test plan must include a CAS UPDATE round-trip on a non-UTC host.
- **Wrapping the bug fix in a "feature flag" so old code keeps the strip-tzinfo behaviour.** Per CLAUDE.md prime directive: no back-compat shims. The fix is one-shot. Re-tag every TIMESTAMPTZ column in the same shard.
- **Using `time.gmtime()` / `time.timezone` to "force UTC" elsewhere in the call stack.** Process-global timezone state does not affect asyncpg's codec — the codec calls `astimezone(UTC)` on the value passed to it, and a naive value will be interpreted through Python's local-tz logic regardless of process-level overrides. The fix has to be at the value: aware-UTC by the time the codec sees it.

---

## Enforcement Guards

| Guard | Location | What it catches |
|---|---|---|
| Unit tests on `_coerce_datetime_for_write_tz` | `packages/core/tests/unit/test_schema_backed.py` | naive input wrapped with UTC; aware non-UTC normalized to UTC; None passthrough |
| Unit tests on `_coerce_datetime_for_read_tz` | same | naive arrival wrapped with UTC; aware UTC passthrough; aware non-UTC normalized |
| `test_role_update_audit_row` (hub integration) | `tests/integration/test_role_update_audit_row.py` | end-to-end CAS round-trip on the role-update path; reproducer for the original bug |

There is **no AST walker** for this bug class. The bug is a runtime / wire-encoding mismatch, not a structural pattern. The guards are unit tests on the coercion helpers + the integration test that reproduces the original failure. A future TIMESTAMPTZ column comes with a CAS-path integration test in the same shard.

---

## Verification

```bash
# 3tears core — DATETIMETZ unit coverage
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears
uv run --directory packages/core pytest tests/unit/test_schema_backed.py -v -k "datetimetz"

# Hub — RBAC + audit unit tests round-trip TIMESTAMPTZ values
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/14-eng-ai-bot
uv run pytest tests/unit/hub/rbac/test_collections.py -v
uv run pytest tests/unit/hub/admin/test_audit.py -v

# Hub — integration test that originally reproduced the bug
uv run pytest tests/integration/test_role_update_audit_row.py -v

# Reproduction protocol on a non-UTC host:
TZ=America/Los_Angeles uv run pytest tests/integration/test_role_update_audit_row.py -v
TZ=America/New_York uv run pytest tests/integration/test_role_update_audit_row.py -v
```

The `TZ=...` reproduction protocol is the canonical way to surface this bug class. Run it whenever a CAS-fenced TIMESTAMPTZ column lands.

---

## Commit Chain

| Repo | SHA | Description |
|------|-----|-------------|
| 3tears | `2353cc6` | fix(core): TIMESTAMPTZ column type for SchemaBackedCollection (collections-task-03c) |
| hub | `32d685f` | fix(hub): tag TIMESTAMPTZ columns as DATETIMETZ_TYPE on rbac + audit collections (collections-task-03c) |
| hub | `fa7352d` | docs(CLAUDE.md): TIMESTAMP vs TIMESTAMPTZ write-boundary distinction (collections-task-03c learning) |

---

## Related Shard Docs

- `collections-task-02-partition-column-primitive.md` (this repo) — partition primitive that this shard is sibling to. The framework changes here live alongside the partition machinery in the same module.
- `14-eng-ai-bot/docs/collections-task-03-hub-schema-backed-partition.md` — hub adoption shard whose `RoleCollection` CAS path surfaced the original bug as `test_role_update_audit_row`. The `1 failing integration test` line in that shard's test counts is the bug captured here.
- `14-eng-ai-bot/docs/collections-task-03b-walker-strict-flip.md` — sibling. Independent of this shard; landed on the same branch and on the same release.
- `14-eng-ai-bot/CLAUDE.md` (datetime section, lines 99-104) — institutional learning surface. Every CLAUDE.md datetime rule update should reference this shard's commits.

---

## Cautionary Summary

The TIMESTAMPTZ codec bug is invisible on UTC hosts and silently corrupts every CAS-fenced TIMESTAMPTZ write on every non-UTC dev machine. The platform's institutional defense is:

1. **`DATETIMETZ_TYPE` for TIMESTAMPTZ; `DATETIME_TYPE` for TIMESTAMP-without-tz.** The two tags are mutually exclusive. Match the migration column type.
2. **CLAUDE.md datetime section** carries the column-type-dependent rule. Read it when introducing a new datetime column.
3. **`TZ=America/Los_Angeles` reproduction protocol** for CAS-path integration tests. Run before declaring a TIMESTAMPTZ CAS round-trip stable.

A future contributor adding a TIMESTAMPTZ column reads this doc, picks `DATETIMETZ_TYPE`, ships a CAS-path test that runs under `TZ=America/Los_Angeles`, and never reproduces the bug. That is the goal.
