# collections-task-06: Neutral L3 store seam + `L3Backend`/`DurableStore` protocols

**Status:** PROPOSED. Behavior-preserving refactor + one additive capability, then a structured-ops migration. Two coordinated commits; no dual-stack.
**Scope:** `3tears` core (`collections/base.py`, `schema_backed.py`, `flush.py`, `data/collection_factory.py`, `backends/`, `collections/registry.py`) + every `BaseCollection` subclass across `datasources`/`agent-*`/`registry` (~65 framework overrides) and the ~15 core test files. Coordinated consumers: `metallm` (27 overrides + the AST drift test), `14-eng-ai-bot` (52 overrides), `14-eng-ai-bot-agents` (6). The `feature/agent-wake-foundation` worktree merges develop forward. `14-eng-ai-pentest-kit` is name-agnostic (verify-only).
**Origin:** The durable (L3) tier is the framework's primary collection extension point, but it is **named for a product** — `fetch_from_postgres` / `save_to_postgres` / `delete_from_postgres` / `persist_to_postgres` on `BaseCollection` (`collections/base.py:182-252,912`), overridden ~215× across the workspace. The org runs YugabyteDB as much as Postgres, and a downstream consumer (Scriob) needs a **git working-tree** as the durable tier — which the current contract cannot express, because the de-facto backend interface (`self.l3_pool`, typed `Any`) takes **raw SQL strings**. This is a half-extraction whose cost is now a cross-repo migration; the fix does it fully.

---

## Objective

Extract the L3 (durable) tier into a **product-neutral, pluggable backend**, in two layers that were conflated under `self.l3_pool`:

- **Layer A — the per-collection persistence seam** (`*_from/to_postgres`): rename to neutral `*_store` names. These still emit durable-store operations; they just stop pretending the store is Postgres.
- **Layer B — the transport**: formalize the implicit `fetch`/`fetchrow`/`execute`/`execute_batch`/`acquire`/`transaction` surface into an explicit **`L3Backend` Protocol** (mirroring the existing `L1Backend` at `cache/base.py:16`), with a named **`SqlL3Backend`** default wrapping the asyncpg pool (today it is a bare, duck-typed pool). `NatsProxyL3Backend` already conforms.

Plus two things the current contract lacks:

- **An atomic multi-entity persist hook** in `flush_pending` (today it persists per-entity with no transaction wrapper, `flush.py:229-300`), reusing the existing `save_to_store(..., conn=conn)` thread and the backend's `transaction()`.
- **A higher-level `DurableStore` (structured-ops) Protocol** — `fetch_one`/`upsert`/`delete`/`scan` keyed by table + columns, **no SQL string** — because the raw-SQL `L3Backend` is irreducibly SQL and a git backend cannot conform to it. This is the part that delivers the capability; **without it the whole exercise is a cosmetic rename.**

---

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| L3B-01 | Rename the abstract `fetch_from_postgres`/`save_to_postgres`/`delete_from_postgres` + concrete `persist_to_postgres` to `fetch_from_store`/`save_to_store`/`delete_from_store`/`persist_to_store`; update all `base.py` internal call sites (`_pull_through` 567, `_async_propagate_write` 670, `save_entity` 887/893, `persist` 914, `reload_entity` 923, `delete` 946) and `flush.py:251`. Do NOT rename `serialize`/`deserialize`. | P0 |
| L3B-02 | Add `L3Backend(Protocol)` (`@runtime_checkable`, mirror `cache/base.py:L1Backend`) in `backends/protocol.py` capturing `fetch`/`fetchrow`/`execute`/`execute_batch`/`acquire`/`transaction` + serialization/namespacing/error contract (docstrings). Extract `SqlL3Backend` (`backends/sql.py`) wrapping the asyncpg pool. `NatsProxyL3Backend` asserted-conformant via a `runtime_checkable` test. Retype `registry.py` (`_l3_pool`/`get_l3_pool`/`configure`/`bind_table`) and `base.py` `self.l3_pool` from `Any` → `L3Backend | None`. | P0 |
| L3B-03 | Add `DurableStore(Protocol)` (structured ops: `fetch_one(table, pk)` / `upsert(table, row, *, on_conflict, cas)` / `delete(table, pk)` / `scan(table, filters)` — no SQL). `SqlL3Backend` implements both `L3Backend` and `DurableStore` (relocating `_build_*_sql` generation out of `SchemaBackedCollection`/`collection_factory` into the SQL backend). This is the seam `GitL3Backend` (Scriob, separate repo) plugs into. | P0 |
| L3B-04 | Wire the atomic multi-entity persist hook in `flush_pending`: resolve the backend, toposort (as today), and if it exposes a usable `transaction()`, persist the toposorted batch in ONE transaction (one DB tx for SQL; one commit for a git backend). **Graceful degrade**: on a transaction failure, replay through today's exact per-entity loop so each entity keeps its `_is_fk_violation` classification + re-enqueue. Strictly additive; no test weakened. | P0 |
| L3B-05 | Consolidate the asyncpg status-tag parser: move `schema_backed.py:_parse_rowcount` to `backends/protocol.py` as `parse_rowcount(status)`; replace the duplicated `int(result.split()[-1])` in product code (`datasources` ×4, `acl`, a workspace test) with it. Keep `execute() -> str` (returning a bare int reintroduces the documented `_format_execute_tag` incident); the int-rowcount lives at the `DurableStore` level (`upsert` returns int natively). | P0 |
| L3B-06 | All ~215 overrides + call sites renamed; `grep -rE 'fetch_from_postgres\|save_to_postgres\|delete_from_postgres\|persist_to_postgres' --include='*.py'` returns empty in every repo/branch. `SchemaBackedCollection` (~307 inheritors) generates the new names; verify via a representative `hasattr(C,"<new>") and not hasattr(C,"save_to_postgres")` assertion. | P0 |
| L3B-07 | metallm `api/tests/enforcement/test_schema_agreement.py` AST target (`node.name != "save_to_postgres"`, ~line 131) retargeted; the empty-extraction `pytest.fail` guards KEPT; drift-detection re-proven via an inverted fixture (a deliberately drifted collection must fail red). pentest-kit `test_collection_contracts.py` is name-agnostic — no edit, confirm green. | P0 |
| L3B-08 | `feature/agent-wake-foundation` updated by **merge-from-develop** (its established cadence), NOT a hand re-sync and NOT a PyPI-ification (out of scope). Lockstep version bump (`bump-version.sh minor`, 0.x breaking → minor) + CHANGELOG `Changed` entries + 37 docs updated. | P1 |

---

## Design Context

**The raw-SQL-vs-git tension is the crux (why this is not a rename).** The de-facto `L3Backend`
interface takes raw SQL (`fetch(query, *params)`). A `GitL3Backend` cannot run
`SELECT ... WHERE id = $1`; pretending it can (regex-matching SQL) is the half-measure to avoid.
The honest extraction recognizes two abstraction levels:

1. **Low-level raw transport** (`L3Backend`: `fetch`/`fetchrow`/`execute`/`execute_batch`) — the
   ad-hoc-SQL escape hatch the `l3_pool` ivar docstring advertises (hub keyset pagination, JOINs).
   Irreducibly SQL; a git backend legitimately **does not implement it**.
2. **High-level structured ops** (`DurableStore`: `fetch_one`/`upsert`/`delete`/`scan`) — by table +
   column dict + pk, no SQL string. SQL backends implement these by generating SQL; a git backend
   implements them as file read/write/delete + commit. **This is the level the standard CRUD
   lifecycle uses.** The status-tag leak (`int(result.split()[-1])`) disappears here — `upsert`
   returns an int rowcount natively.

**The transaction hook is wiring, not invention.** `execute_batch(transaction=True)`,
`acquire()`, and `transaction()` already exist on the backend; `save_to_store(..., conn=conn)`
already threads a transaction connection. `flush_pending` simply does not use them yet.

**Status-tag decision — keep `execute() -> str`.** `NatsProxyL3Backend._format_execute_tag` exists
*because* returning a bare int crashed `int(...).split()` callers in production. The neutral move is
one framework-owned `parse_rowcount`, not a breaking return-type change.

**Why two commits (both mandatory; the no-dual-stack rule is about not shipping two *live* paths):**
- **Commit 1 (behavior-preserving):** rename Layer A → `*_store`; `L3Backend` Protocol + `SqlL3Backend`;
  retype registry/base; flush transaction hook; `parse_rowcount`. Introduce the `DurableStore`
  Protocol + `SqlL3Backend`'s implementation of it, but do not migrate collections onto it. All green.
- **Commit 2 (the capability):** migrate `SchemaBackedCollection`/`collection_factory` CRUD onto the
  structured `DurableStore` ops; relocate SQL generation into `SqlL3Backend`; rewrite product overrides
  that hand-write SQL to structured calls (those that genuinely need raw SQL stay on `L3Backend`,
  explicitly SQL-only). **This is what makes `GitL3Backend` implementable.**

**No PyPI broken-window:** metallm and 14-eng-ai-bot resolve 3tears via `tool.uv.sources` editable
path pins, so consumers compile against the live tree — coordination is same-checkout lockstep, not
publish-then-consume. **agent-wake is a worktree branch of this repo**, not a vendored copy — merge
develop forward; do not re-sync by hand or convert to a dependency here.

---

## Files to Create / Modify / Retire

### Create
- `packages/core/src/threetears/core/backends/protocol.py` — `L3Backend` + `DurableStore` Protocols, `L3Connection`/`L3Transaction`, `parse_rowcount`.
- `packages/core/src/threetears/core/backends/sql.py` — `SqlL3Backend` (implements both protocols).

### Modify
- `packages/core/.../collections/base.py` — rename the 4 seam methods + 7 internal call sites + docstrings + `__all__`; retype `self.l3_pool`.
- `packages/core/.../collections/flush.py` — `:251` retarget; atomic-transaction batch wrapper with per-entity fallback.
- `packages/core/.../collections/schema_backed.py`, `data/collection_factory.py` — rename generated impls; (Commit 2) move `_build_*_sql` into `SqlL3Backend`; import `parse_rowcount`.
- `packages/core/.../collections/registry.py`, `backends/__init__.py` — retype to `L3Backend | None`; re-export.
- The ~65 framework overrides (`datasources` 12, `agent/wake` 9, `agent/skills` 6, `agent/memory` 1, `registry/heartbeat_collection` 3, workspace/acl/conversations/epoch) — rename; swap `int(result.split()[-1])` → `parse_rowcount`.
- ~15 core test files — rename call/override surface; never weaken assertions. Add: protocol conformance test; flush atomic-batch test (happy = one tx, failure = per-entity fallback preserved); a `DurableStore`-only fake (proves a non-SQL backend can satisfy the protocol).
- (consumers) metallm/14-eng-ai-bot/14-eng-ai-bot-agents overrides + metallm AST test (L3B-07).

### Retire
- The old `*_postgres` method names everywhere — deleted in the same commit (no alias, no shim).

---

## Implementation Notes (phased)

1. Branch is `feature/scriob-foundation` (this worktree). Capture before-grep counts. Gate 0: `./scripts/check-all.sh` green.
2. **core first:** `base.py` seam + Protocol + call sites; `flush.py` + transaction hook; `schema_backed.py`; `collection_factory.py`. mypy --strict is the zero-missed-sites proof (a missed override breaks abstract-instantiation; a missed call → `attr-defined`).
3. Subpackage overrides leaf-first; rerun `./scripts/typecheck.sh` per package.
4. Core tests. Gate 1: old-name grep empty in `packages/`; `./scripts/check-all.sh` == 0.
5. Consumers (same change set, path-pinned): 14-eng-ai-bot, metallm, bot-agents. Gate 2: per-repo grep empty + green.
6. metallm AST test (L3B-07): retarget + prove via inverted fixture. Gate 3.
7. Merge develop → `feature/agent-wake-foundation`; resolve base.py drift. Gate 4.
8. `bump-version.sh minor`; CHANGELOG; docs; consumer floors. Gate 6.

## Anti-patterns

- DO NOT keep an alias/property forwarding old→new name (dual-stack — banned). Delete old in the same commit.
- DO NOT add a deprecation shim or feature flag.
- **DO NOT stop at Commit 1** — the structured `DurableStore` layer is the point; a rename without it leaves a git backend impossible.
- DO NOT weaken metallm's empty-extraction `pytest.fail` guard to "fix" a loud failure (that reintroduces silent-pass).
- DO NOT hand re-sync agent-wake's `packages/core`, or PyPI-ify agent-wake here.
- DO NOT rename `serialize`/`deserialize`, or change `execute() -> str` to int.

## Acceptance Criteria

- [ ] Old-name grep empty across `*.py` in 3tears + metallm + 14-eng-ai-bot + 14-eng-ai-bot-agents + 14-eng-ai-pentest-kit + the agent-wake branch.
- [ ] `./scripts/check-all.sh` (lint + mypy --strict + tests) exit 0 in every repo.
- [ ] `L3Backend` + `DurableStore` Protocols present; `SqlL3Backend` implements both; `NatsProxyL3Backend` conformance test green; a `DurableStore`-only fake proves SQL-free conformance.
- [ ] flush transaction hook wired + tested (happy = one tx; failure = per-entity fallback unchanged).
- [ ] `test_schema_backed.py` green; inherited-name assertion green.
- [ ] metallm drift-detector re-proven via inverted fixture (recorded in PR); pentest-kit suite green unmodified.
- [ ] agent-wake merged forward; lockstep version bumped (`--verify` passes); CHANGELOG `Changed` entries; 37 docs updated.

## Verification

```
grep -rEn 'fetch_from_postgres|save_to_postgres|delete_from_postgres|persist_to_postgres' --include='*.py'   # empty, each repo
./scripts/check-all.sh                                                                                       # exit 0, each repo
uv run --directory <metallm/api> pytest tests/enforcement/test_schema_agreement.py -v
./scripts/bump-version.sh --verify 0.11.0
```

Grep-empty + mypy-strict-green are belt-and-suspenders: grep catches the string/AST references mypy can't see; mypy catches the structural override/call mismatches grep can't reason about.
