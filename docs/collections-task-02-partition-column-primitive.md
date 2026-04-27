# collections-task-02: Partition Column Primitive (3tears side)

**Status:** SHIPPED. Walker default flipped to strict. Memories surface deferred to `collections-task-04` and cleared by that shard.
**Scope:** `3tears` repo — `packages/core` (primitive + walker), `packages/agent-tools` (context_items), `packages/agent-workspace` (workspaces), `packages/conversations` (conversations).
**Companion shards:** `collections-task-03` (hub adoption), `collections-task-03b` (walker strict-flip on hub side), `collections-task-04` (cross-agent retrieval / memories surface).

---

## Objective

Eliminate the **cross-partition bleed** class of bug structurally. A partitioned table (`memories`, `conversations`, `context_items`, `workspaces`, `workspace_files`, ...) carries one column whose value identifies which agent / conversation / workspace the row belongs to. Every read and write predicate on that table must include the partition column; if it does not, the SQL silently scans across every partition and either leaks data (a stray `find_by_id(memory_id)` returns rows the caller has no right to see) or corrupts state (a stray `UPDATE workspaces SET ... WHERE id = $1` rewrites a workspace owned by a different agent).

This shard delivered the **framework half** of the doctrine: a declarative `partition=True` flag on `Column`, a `__init_subclass__` guard at class definition time, a `@spans_partitions` decorator for the deliberate cross-partition fan-out path, and an AST walker that scans every package source root for SQL literals on partitioned tables and verifies they include the partition column. Three Collections (`context_items`, `conversations`, `workspaces`) adopted the primitive in the same shard; the memories surface deferred via `_DEFERRED_TABLES` to `collections-task-04` because cross-agent memory retrieval needed an authorization layer the framework did not yet provide.

---

## Problem Statement

`SchemaBackedCollection.find_by_id(id)` was the canonical ergonomic helper: pass the row id, get the entity. The id was unique fleet-wide, so the lookup was technically correct. But the helper let a caller resolve a row without proving they belonged to the partition that owned it. Two independent runtime paths surfaced the risk:

1. **Agent-pod multiplexing.** One agent pod handles many concurrent conversations. The pod's `ToolContextManager` carries a `dict[UUID, ContextManager]` keyed by conversation id, but the underlying `context_items` table had `id` as its sole primary key. A query that looked up a context item by id alone — without filtering on `conversation_id` — silently saw rows from sibling conversations on the same pod.
2. **Workspace state.** `workspaces.id` was likewise globally unique, and `WorkspaceCollection.find_by_id(workspace_id)` was the standard read entry point. A caller that received a workspace_id from an untrusted source (a tool argument, a checkpoint, a hub-broker proxy) could load and mutate workspace state owned by another agent.

Three reinforcing layers solve the class of bug: a declarative schema flag, a class-definition-time guard, and an AST walker. None of them is sufficient alone — the schema flag declares intent, the `__init_subclass__` guard catches authoring violations, the walker catches escaped SQL literals — but together they make a cross-partition query structurally impossible to ship.

---

## Design Decisions

### `Column.partition=True` is declarative, not procedural

A column carrying `partition=True` is the partition key for the table. Adding the flag changes three things at once: the schema validator coerces `immutable=True` (changing the partition value would corrupt the partition map), the validator asserts the column is part of the primary key (so row uniqueness flows through the partition), and `TableSchema.partition_column` exposes the name to callers (the AST walker, the SQL generator, the `__init_subclass__` guard).

Source: `packages/core/src/threetears/core/collections/schema_backed.py:120` (`Column` dataclass), `:179` (`TableSchema`), `:218` (`TableSchema.__post_init__` validation), `:293` (`TableSchema.partition_column` property). Commit `2ac0bdd`.

### Exactly one partition column per table

`TableSchema.__post_init__` rejects schemas that flag two columns as `partition=True`. Multi-column partitioning is conceivable (some warehouse layouts use customer_id + date) but adds combinatorial complexity to the AST walker's literal-match heuristic without solving any concrete problem the platform faces today. The single-column rule is a deliberate constraint, not a missing feature; if a future shard genuinely needs multi-column partitioning, it can extend the validator and the walker together. Commit `2ac0bdd`.

### Partition column must be in the primary key

The schema validator rejects a `partition=True` column that is not part of `primary_key`. Composite PK `(partition_column, id)` enforces row uniqueness through the partition: two rows with the same `id` but different `partition_column` values are legal at the schema level (different partitions own them). This is the structural property that makes `find_by_id(id)` unsafe and `find_by_id(partition_value, id)` safe — the lookup has to bind the partition.

Where a sibling FK depends on `id` alone (the FK contract pre-dates the partition flip), the migration adds a `UNIQUE (id)` constraint alongside the composite PK. The hub-side `v054` migration is the canonical example for that pattern (collections-task-03 / `8cd0cd5`); the 3tears-side conversions in this shard had no inbound FKs so no UNIQUE shim was needed.

### `__init_subclass__` guard fails at import time

`SchemaBackedCollection.__init_subclass__` runs at class-definition time. When the subclass's schema declares a partition column, every public method on the subclass must satisfy one of three contracts:

1. accept the partition column as a parameter (positional or keyword-only), so the SQL generator can include the partition predicate naturally; OR
2. be decorated with `@spans_partitions`, declaring "this method deliberately fans out across partition values supplied as a tuple"; OR
3. appear in `cls._partition_exempt_methods` — a narrow allowlist for genuine framework overrides where the partition is implicit (e.g. `count_total` on a per-pod operational metric).

A subclass that violates the guard fails at import time with `PartitionEnforcementError`. There is no runtime fallback. The bug surfaces during CI before any cross-partition leak can ship. Source: `packages/core/src/threetears/core/collections/schema_backed.py:610` (`PartitionEnforcementError`), `:631` (`spans_partitions`), `:784` (`SchemaBackedCollection.__init_subclass__`). Commit `2ac0bdd`.

### `@spans_partitions` requires `tuple[UUID, ...]`, not `list[UUID]`

The cross-partition argument is named with a plural suffix (`agent_ids`, `workspace_ids`) and validated at call time to be a non-empty `tuple`. Two rationales:

1. **Type signal.** `list` reads as "any agents you want." `tuple` reads as "an immutable, deliberately resolved set." The decorator validates the tuple shape at call time; a stray `list` argument fails loud rather than silently fanning out across whatever the caller happened to pass.
2. **Empty-tuple refusal.** The decorator refuses `()` so a fan-out that resolved to zero authorized partitions surfaces as a deliberate `TypeError` rather than a silent zero-row query. The service layer is responsible for short-circuiting to `[]` before invoking the Collection.

This contract is documented in `partition-column-pattern.md` (the canonical "how to apply the pattern" guide). Commit `2ac0bdd`.

### AST walker is the third defense layer

The framework guard catches authoring drift in the Collection layer (someone writes a method that forgets the partition column). It does not catch SQL literals scattered across the codebase — call sites that build SQL outside a Collection method and route through the pool directly. The walker fills that gap.

`packages/core/tests/enforcement/test_partition_column_enforcement.py` walks every declared package source root, collects every string literal beginning with a SQL keyword (`SELECT`, `INSERT`, `UPDATE`, `DELETE`, `WITH`, `VALUES`), and for each literal touching a partitioned table verifies the literal body contains the partition column name in some predicate position. The walker is deliberately heuristic — it does not parse SQL — so a literal that says `WHERE row_scope = 'customer'` passes whether the column actually filters meaningfully. The AST walker is a guard against forgetfulness, not a SQL semantic checker; the partition predicate is structural defense, and a developer who writes `WHERE conversation_id = 'whatever'` to silence the walker will surface the problem differently in code review.

Heuristics:
- Migration directories (`migrations/`) and test directories (`tests/`) are exempt — DDL legitimately does not filter rows; tests build fake data.
- Literals must START with a SQL keyword to count, so docstrings that mention "FROM memories" in prose do not false-fire.
- f-string placeholders are treated as opaque, so dynamic-scope-clause builders pass.
- A narrow exemption list (`_EXEMPT_LITERAL_FRAGMENTS`) covers DDL idioms and the dynamic-scope-clause helper.

Source: `packages/core/tests/enforcement/test_partition_column_enforcement.py:57` (`_PARTITIONED_TABLES`), `:80` (`_DEFERRED_TABLES`), `:98` (`_EXEMPT_LITERAL_FRAGMENTS`). Commit `b1e9b3c`.

### Walker shipped in `report` mode, then flipped to `strict`

The walker landed first in `report` mode (env var `PARTITION_ENFORCEMENT_MODE=report`) so the follow-on Collection conversions could land incrementally without flipping CI red. Once the in-scope Collections (`context_items`, `conversations`, `workspaces`) satisfied the walker, the default flipped to `strict` (commit `06b20ec`). Memories-surface tables (`memories`, `media`, `media_content`, `memory_chunks`, `conversation_memory_refs`) stayed in `_DEFERRED_TABLES` with explicit rationales tying back to `collections-task-04` (the cross-agent retrieval shard). Strict mode raised on `strict_violations` only; deferred hits surfaced via `pytest.skip` so the punch list stayed visible.

Lesson the hub side learned the hard way one shard later (collections-task-03b): if your walker has a cleanup pass behind it, ship the cleanup IN the same shard as the strict flip, not split. The 3tears side got this right — the strict flip and the in-scope Collection cleanups landed together — but the hub side split them and paid for it.

### Memories surface deferred

`MemoriesCollection` has a cross-agent retrieval pattern: a user logged into agent A wants to see memories the same user has accumulated across agents B and C in the same customer they have been authorized to read from. That access pattern is genuinely cross-partition and needs an authorization layer (the unified RBAC evaluator + a service-layer composition pattern) the framework had not yet integrated. Forcing the conversion in this shard would have either banned the cross-partition pattern entirely (wrong) or admitted it without authorization (worse).

The deferral is documented at `_DEFERRED_TABLES` in the walker, with each entry tagging the receiving shard. `collections-task-04` cleared the deferred surface — the walker's `_DEFERRED_TABLES` is empty as of that shard.

---

## What Landed

### Framework (commit `2ac0bdd`)

- `Column(partition: bool = False)` field — `packages/core/src/threetears/core/collections/schema_backed.py:175`.
- `TableSchema.__post_init__` — partition-column validation: at most one column per table, must be in `primary_key`, coerces `immutable=True`. Source: `packages/core/src/threetears/core/collections/schema_backed.py:218`.
- `TableSchema.partition_column` property — exposes the partition column name (or `None`). Source: `packages/core/src/threetears/core/collections/schema_backed.py:293`.
- `PartitionEnforcementError` — raised at class-definition time. Source: `packages/core/src/threetears/core/collections/schema_backed.py:610`.
- `@spans_partitions` decorator — call-time guard requiring tuple-shape, non-empty fan-out argument. Source: `packages/core/src/threetears/core/collections/schema_backed.py:631`.
- `SchemaBackedCollection.__init_subclass__` — class-definition-time guard. Source: `packages/core/src/threetears/core/collections/schema_backed.py:784`.
- `_partition_exempt_methods: ClassVar[frozenset[str]]` — narrow allowlist for framework overrides. Source: `packages/core/src/threetears/core/collections/schema_backed.py:782`.

12 unit tests cover schema validation (5), subclass enforcement (4), and decorator call-time guards (3).

### AST walker (commit `b1e9b3c`)

- `tests/enforcement/test_partition_column_enforcement.py` — walks `_PACKAGE_SRC_ROOTS`, collects SQL literals, validates partition-column presence.
- `_PARTITIONED_TABLES` discovery list — table -> partition column mapping for the 9 partitioned tables in scope.
- `_DEFERRED_TABLES` allowlist — receiving-shard rationale per table.
- `_EXEMPT_LITERAL_FRAGMENTS` narrow allowlist — DDL idioms + dynamic-scope-clause + one-time data translation.
- `PARTITION_ENFORCEMENT_MODE` env var — `report` (initial) / `strict` (post-`06b20ec`).

### Collection adoptions

- **context_items composite PK** (commit `6c91f35`): `ContextItemEntity._id` becomes a `(conversation_id, context_id)` tuple; `ContextItemCollection.touch` / `upsert_variable` accept `conversation_id` explicitly; `ToolContextManager` threads `self.conversation_id` at every call site; `v001` migration emits `PRIMARY KEY (conversation_id, context_id)`; L1 SQLAlchemy mirrors composite. Walker drops from 33 to 32 violations.
- **workspaces agent_id partition** (commit `0bcd295`): `WorkspaceCollection.find_by_id` requires `agent_id` positionally; `materialize.bind` / `.recover` gain `agent_id` keyword; every `UPDATE workspaces` SQL gains `AND agent_id = $N`; every `_FakeWorkspaceEntity` test fixture picks up `agent_id`. Walker drops to 23.
- **conversations composite PK** (commit `b15dabb`): `Conversation._id` becomes `(agent_id, id)` tuple; `find_by_user` takes `agent_id` positionally; SQL gains `AND agent_id = $1`; `v001` migration emits composite PK. No FK targets `conversations(id)` so no `UNIQUE (id)` shim. Walker drops to 30.

Note: the commit-by-commit walker counts are non-monotonic because each conversion both removed violations on the converted table and exposed new ones on the as-yet-unconverted siblings (a cleanup commit on conversations does not remove the workspaces violations until the workspaces commit lands).

### Strict-mode flip (commit `06b20ec`)

- `PARTITION_ENFORCEMENT_MODE` default flips from `report` to `strict`.
- Walker output: 0 strict violations, 14 deferred (memories surface).
- 1063 core tests green.

---

## Anti-patterns

These are **rejected on code review**, specific to this shard's lessons:

- **Adding a `partition=True` column without the `__init_subclass__` guard.** The guard is `SchemaBackedCollection.__init_subclass__`. If a Collection's schema declares a partition column but the Collection does not inherit `SchemaBackedCollection`, the guard does not fire. Every partitioned Collection must inherit `SchemaBackedCollection` (not bare `BaseCollection`).
- **Adding `_partition_exempt_methods` entries with rationales like "internal helper" or "tests need this."** The exemption set is a last resort with **specific** rationales. The first two resolution paths are: (1) add the partition column to the signature; (2) decorate with `@spans_partitions`. A test that needs cross-partition access is testing the cross-partition retrieval surface and should call the `@spans_partitions` method.
- **Translating `tuple[UUID, ...]` to `list[UUID]` inside the Collection method "to make `ANY($1::uuid[])` work."** asyncpg accepts both; `list(agent_ids)` at the SQL boundary is the right place to coerce. The tuple-shape contract on the API is what matters.
- **Walking the partition-column AST walker into a new mode without a self-test.** When `collections-task-03` adds a hub-side walker, the hub walker carries its own `_PARTITIONED_TABLES` and inherits the same heuristics. The walker file's tests assert on the discovery list shape; new partition tables come with a discovery-list assertion update in the same commit.
- **Dropping the partition column from a SQL literal "because the caller knows what partition they're in."** Future callers will not. The partition predicate is structural defense, not redundancy. The walker's heuristic is deliberately literal-match — silencing it by spelling the partition column in a tautological predicate (`row_scope = row_scope`) is a code-review reject.
- **Collapsing `find_by_id(partition, id)` into a nullable-aware `find_by_id(partition=None, id)`.** The non-null-partition signature is the contract. A nullable partition silently re-introduces the bleed class — the cross-partition retrieval pattern is `@spans_partitions`, not "make the partition argument optional."

---

## Enforcement Guards

| Guard | Location | What it catches |
|---|---|---|
| `TableSchema.__post_init__` | `packages/core/src/threetears/core/collections/schema_backed.py:218` | declaring multiple `partition=True` columns; declaring a partition column outside the primary key |
| `SchemaBackedCollection.__init_subclass__` | `packages/core/src/threetears/core/collections/schema_backed.py:784` | a public method on a partitioned Collection that neither accepts the partition column nor opts into `@spans_partitions` |
| `@spans_partitions` call-time guard | `packages/core/src/threetears/core/collections/schema_backed.py:631` | passing a `list` instead of `tuple`; passing an empty tuple |
| AST walker | `packages/core/tests/enforcement/test_partition_column_enforcement.py` | SQL literals on partitioned tables that omit the partition column name |

All four run in under 15 s. CI runs the AST walker in strict mode by default; the framework guards fire on every test that imports the package.

---

## Verification

```bash
# 3tears core (framework + walker self-tests)
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears
uv run --directory packages/core pytest tests/ -q

# Walker in strict mode (default)
uv run --directory packages/core pytest tests/enforcement/test_partition_column_enforcement.py -v

# Walker in report mode (legacy / cleanup-window override)
PARTITION_ENFORCEMENT_MODE=report uv run --directory packages/core pytest tests/enforcement/test_partition_column_enforcement.py -v

# Per-package adoption tests
uv run --directory packages/agent-tools pytest tests/ -q
uv run --directory packages/agent-workspace pytest tests/ -q
uv run --directory packages/conversations pytest tests/ -q
```

---

## Commit Chain

| SHA | Description |
|-----|-------------|
| `2ac0bdd` | feat(core): partition-column primitive on SchemaBackedCollection |
| `b1e9b3c` | test(enforcement): partition-column AST walker |
| `6c91f35` | refactor(agent-tools): context_items composite PK on (conversation_id, context_id) |
| `0bcd295` | refactor(agent-workspace): agent_id partition predicates on workspaces SQL |
| `b15dabb` | refactor(conversations): conversations composite PK on (agent_id, id) |
| `06b20ec` | feat(enforcement): flip PARTITION_ENFORCEMENT_MODE default to strict |

---

## Related Shard Docs

- `partition-column-pattern.md` — canonical "how to apply the partition column pattern" guide. Reference this doc when introducing a new partitioned table or a new cross-partition retrieval surface.
- `collections-task-03-hub-schema-backed-partition.md` (hub repo) — hub adoption of the primitive on the polymorphic + per-customer + per-agent + per-group hub tables.
- `collections-task-03b-walker-strict-flip.md` (hub repo) — the hub-side walker strict-mode flip + Pattern 1 / Pattern 2 cleanup.
- `collections-task-03c-timestamptz-tz-shift.md` (this repo) — the asyncpg TIMESTAMPTZ codec bug class and `DATETIMETZ_TYPE` introduction.
- `collections-task-04-...` — cross-agent retrieval shard that cleared the memories surface from `_DEFERRED_TABLES`.
