---
artifact: build-plan
version: 1
scope: sqlite-l3
branch: feat/sqlite-l3
depends_on: []
partition: "serial -- one store, its contract and its docs; the contract and the store interlock"
last_validated: 2026-11-25
---

# Build plan: SQLite as an L3, and one L3 contract every store keeps

Off `origin/work/0950-fixes` (the 0.95.0 line, which develop does not yet hold). One PR, area core, into the 0.95.0
line or develop after it. Critic mode: cumulative at the end.

The consumer this was built for is an application outside this repo; this repo is public, so no consumer is named in
it (code, tests, plan or `_public_api_consumers.txt`, whose entries say "an application that uses the store"). The
owner holds the reference to it for a consumer audit.

## Goals

A consumer keeps its durable state (send queues, delivery ledgers, watches, dead letters) as 3tears collections.
Locally it runs one process on one machine, with no Postgres and no NATS. Owner ruling 2026-11-25: build it as SQLite being a proper 3tears L3 (a durable store), NO_L2, the ordinary
L1 cache in front; a first-class L3 beside Postgres so any collection that works on Postgres works on SQLite and a
deployment switches by configuration; a durable file (fixed path, WAL, synchronous=FULL, one-process lock, format
version) with a kill -9 harness; one shared contract suite run against the existing stores unchanged (Postgres and
the git store, which is already a proper L3) and SQLite; the three documented as supported L3s.

Success: `SqliteL3Backend` passes the contract's schema tier as Postgres does; the git store passes the first tier
unchanged; a `DurableStoreCollection` over each store passes with L2 and without; a write survives SIGKILL after
each kind of write; `l3_store(url)` picks the store; the docs say when to use each.

Superseded: an earlier reading (an L1-only store whose SQLite L1 file is the record, refusing eviction).
On 0.95.0 the L1 is already a per-process WAL file with an LRU by declared policy, and a schema-backed save with no
L3 fails today (`save_to_store` answers 0); the owner chose the L3 design instead.

## Decisions taken (inferred; flag at the boundary)

Owner ruling 2026-11-25: SQLite matches Postgres's rules and errors wherever it can (unique indexes and
constraints, `NOT NULL`, the asyncpg error types) and refuses clearly where it cannot (a `now()`-defaulted column
added to a stored table, a table change inside an open transaction).

- [DECISION: reuse the Postgres store's statement generator (`schema_sql`) and rewrite `$N`/`::type` for SQLite |
  the same SQL is the strongest guarantee the two stores behave alike (conflict modes, fenced update, NULL-safe and
  ordered CAS); SQLite 3.39+ reads the rest]
- [DECISION: values typed by declared column types (sqlite3 converters under store-private names) | raw-SQL reads
  then answer Python types too, not only the structured ops]
- [DECISION: one writer lock in the process, a separate read connection, statements on worker threads | WAL gives
  readers the last commit without waiting; a write from the task holding a transaction without `conn` is refused
  instead of deadlocking]. Boundary review: a statement on the writing connection runs to its end even when its task
  is cancelled, and the lock and transaction state are released after it (a lock scoped to the await let a cancelled
  write be torn); the writer-running helpers are private and a released connection refuses statements.
- [DECISION: a table that must be created or changed while a transaction is open on the store is refused at
  registration | the DDL runs on the event loop (registration is synchronous, from a collection's constructor), so it
  cannot wait for a transaction that needs the loop to commit; collections are built at startup; owner-confirmed
  2026-11-25]
- [DECISION: additive migration only (new nullable or defaulted columns); anything else refused naming the
  difference; `user_version` holds the format | a stored format is lock-in; a silent misread is worse than a refusal].
  Boundary review: a new column defaulted to `now()` is refused, naming it (SQLite cannot `ADD COLUMN` with an
  expression default, and rebuilding the table is a whole-table read this store does not make on its own); the
  docs had promised it was added. Owner-confirmed 2026-11-25.
- [DECISION: unique indexes (partial and `nulls_not_distinct` included) and unique constraints ARE created, and
  `NOT NULL` is emitted for columns not declared nullable; other indexes and foreign keys are not | uniqueness and
  required values decide which rows a table admits, so skipping them broke "any collection that works on Postgres
  works on SQLite" (a dedup key silently accepting a second row); other indexes are speed only on small local tables,
  and Postgres keeps them through its migrations. Revised at the boundary review from "no constraints beyond the
  primary key"; owner-confirmed 2026-11-25]
- [DECISION: a constraint refusal raises the `asyncpg` exception Postgres raises (`UniqueViolationError` ...), and the
  schema tier asserts that one type | the platform documents `asyncpg.UniqueViolationError` for a duplicate and the
  NATS proxy rebuilds it for the same reason: a consumer's `except` must hold on either store; owner-confirmed
  2026-11-25]
- [DECISION: which compare-and-swap statement a write uses is one function, `schema_sql.build_upsert_statement`,
  both SQL stores call | a copy would leave SQLite on an old rule after the next race fix]
- [DECISION: the contract has two tiers; the git store runs the first | the git store is a store of JSON files written
  whole, with a commit for a transaction; typed values, column-wise updates and raw SQL are not what it is]
- [DECISION: `SqliteL3Backend.truncate` is declared an epoch-discipline primitive like `SqlL3Backend.truncate` | its
  one caller announces the whole-table change]
- [DECISION: branch renamed `feat/sqlite-l3` (was `feat/l1-file-store`) | the design it carries changed]
- Fixed on the way (no pre-existing exception): `processes.name_process`'s mypy error on this base; at the boundary
  review, the two tests failing on this base (`RowFollower.settled` spinning on Python 3.14, and the evals reference
  page out of date), and the mcp grant cache copying every grant per change (its integration speed test).

## Chunk 01 (the only one)

- `packages/core/src/threetears/core/backends/sqlite_file.py` (`SqliteL3Backend`),
  `packages/core/src/threetears/core/backends/l3_url.py` (`l3_store`), `SchemaRegisteringStore`,
  `schema_backed._register_schema` through it; exports.
- Tests: `packages/core/tests/l3_contract.py` (three classes), run in
  `packages/core/tests/unit/backends/test_sqlite_l3_contract.py`,
  `packages/core/tests/backends_git/test_git_store_l3_contract.py`,
  `packages/core/tests/integration/test_sql_l3_contract_live.py`;
  `packages/core/tests/unit/collections/test_collections_over_sqlite_l3.py`; the SIGKILL harness
  (`packages/core/tests/unit/backends/test_sqlite_l3_survives_kill.py`,
  `packages/core/tests/unit/backends/sqlite_l3_child.py`).
- Docs: `docs/adoption/core-sqlite.md`, the integration guide, core-git.md; CHANGELOG.

Done when: lint, `typecheck.sh core`, `test.sh core`, the enforcement suite, the contract on Postgres
(`test-integration.sh core`) and the unit suite pass; Critic run.

## Follow-up chunk: an adopted hand edit gets a new version (branch `feat/git-l3-hand-edit-version`)

Owner ruling 2026-11-25 ("Finish the git L3 follow-ups with the SQLite L3"): hand edits re-versioned so a
compare-and-swap cannot overwrite them. Built on this plan's branch, merged after it.

- `GitRepo.commit_paths` / `commit_files` stamp every adopted row file of a versioned table with a new version
  (now, or just past HEAD's); a moved file keeps its version and bytes.
- [DECISION: the version is stamped by decoding and re-encoding the row through the layout's codec, and the
  re-encode is documented, not avoided | a version-only rewrite would need a codec-level hook every product codec
  implements; the content is kept, only formatting changes]
- [DECISION: a row file the codec cannot read refuses the whole call with a `GitStoreError` naming it | adopting it
  unversioned would reopen the overwrite this chunk closes; a refusal names what to fix]
- Tests: `packages/core/tests/backends_git/test_hand_edit_version.py`. Docs: `docs/adoption/core-git.md`, CHANGELOG.

## Follow-up: the git L3's batched flush (branch `feat/git-l3-transaction`)

Owner ruling 2026-11-25, "Finish the git L3 follow-ups with the SQLite L3": a write-buffered flush over the git
store is one commit. `GitDurableStore.transaction(message=...)` opens a change set as a database store opens a
transaction; a refused set falls back to the flush's per-entity path. A set a non-writer would forward over the
broker's `max_payload` is refused unsent (`ForwardTooLargeError`, a `TransactionTooLargeError`), and the flush writes
it per entity without logging a failed batch.

Done when: the forwarded path (one commit, and the too-large refusal) is tested live, a hand edit on one row's file
refuses the set while the other rows land, `test.sh` and `test-integration.sh -rs` pass, and the Critic clears it.
