# Build plan: SQLite as an L3, and one L3 contract every store keeps

Branch: `feat/sqlite-l3`, off `origin/work/0950-fixes` (the 0.95.0 line, which develop does not yet hold). One PR,
area core, into the 0.95.0 line or develop after it.
Critic mode: cumulative at the end.

## Goals

aidc 2.0 keeps its durable state (send queues, delivery ledgers, watches, dead letters) as 3tears collections
(aidc TRS-25, option A, ratified 2026-11-25). Locally it runs one process on one machine, with no Postgres and no
NATS. Owner ruling 2026-11-25: build it as SQLite being a proper 3tears L3 (a durable store), NO_L2, the ordinary
L1 cache in front; a first-class L3 beside Postgres so any collection that works on Postgres works on SQLite and a
deployment switches by configuration; a durable file (fixed path, WAL, synchronous=FULL, one-process lock, format
version) with a kill -9 harness; one shared contract suite run against the existing stores unchanged (Postgres and
the git store, which is already a proper L3) and SQLite; the three documented as supported L3s.

Success: `SqliteL3Backend` passes the contract's schema tier as Postgres does; the git store passes the first tier
unchanged; a `DurableStoreCollection` over each store passes with L2 and without; a write survives SIGKILL after
each kind of write; `l3_store(url)` picks the store; the docs say when to use each.

Superseded: the earlier reading of U11 (an L1-only store whose SQLite L1 file is the record, refusing eviction).
On 0.95.0 the L1 is already a per-process WAL file with an LRU by declared policy, and a schema-backed save with no
L3 fails today (`save_to_store` answers 0); the owner chose the L3 design instead.

## Decisions taken (inferred; flag at the boundary)

- [DECISION: reuse the Postgres store's statement generator (`schema_sql`) and rewrite `$N`/`::type` for SQLite |
  the same SQL is the strongest guarantee the two stores behave alike (conflict modes, fenced update, NULL-safe and
  ordered CAS); SQLite 3.39+ reads the rest]
- [DECISION: values typed by declared column types (sqlite3 converters under store-private names) | raw-SQL reads
  then answer Python types too, not only the structured ops]
- [DECISION: one writer lock in the process, a separate read connection, statements on worker threads | WAL gives
  readers the last commit without waiting; a write from the task holding a transaction without `conn` is refused
  instead of deadlocking]
- [DECISION: additive migration only (new nullable or defaulted columns); anything else refused naming the
  difference; `user_version` holds the format | a stored format is lock-in; a silent misread is worse than a refusal]
- [DECISION: indexes, foreign keys, unique constraints beyond the primary key are not created | the local deployments
  this serves are small; Postgres keeps them through its migrations]
- [DECISION: the contract has two tiers; the git store runs the first | the git store is a store of JSON files written
  whole, with a commit for a transaction; typed values, column-wise updates and raw SQL are not what it is]
- [DECISION: `SqliteL3Backend.truncate` is declared an epoch-discipline primitive like `SqlL3Backend.truncate` | its
  one caller announces the whole-table change]
- [DECISION: branch renamed `feat/sqlite-l3` (was `feat/l1-file-store`) | the design it carries changed]
- Fixed on the way (no pre-existing exception): `processes.name_process`'s mypy error on this base.

## Chunk 01 (the only one)

- `backends/sqlite_file.py` (`SqliteL3Backend`), `backends/l3_url.py` (`l3_store`), `SchemaRegisteringStore`,
  `schema_backed._register_schema` through it; exports.
- Tests: `tests/l3_contract.py` (three classes), run in `tests/unit/backends/test_sqlite_l3_contract.py`,
  `tests/backends_git/test_git_store_l3_contract.py`, `tests/integration/test_sql_l3_contract_live.py`;
  `tests/unit/collections/test_collections_over_sqlite_l3.py`; the SIGKILL harness
  (`tests/unit/backends/test_sqlite_l3_survives_kill.py`, `sqlite_l3_child.py`).
- Docs: `docs/adoption/core-sqlite.md`, the integration guide, core-git.md; CHANGELOG.

Done when: lint, `typecheck.sh core`, `test.sh core`, the enforcement suite, the contract on Postgres
(`test-integration.sh core`) and the unit suite pass; Critic run.
