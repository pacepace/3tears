"""Every move of a stamped row between tenants is its own row of the moves table, so a restore hears each tenant it left.

Until the moves table (Pace 2026-11-25), a moveable column kept only its row's LAST move (``<column>_moved_from`` and
``_moved_epoch``): a row moved A to B, then B to C, both inside one restore's window, came back for B and C, and A's
followers kept a row no longer theirs. Now the write that moves a row appends one row to ``threetears_tenant_moves``
(:mod:`threetears.core.ddl.tenant_moves`) in the same statement, and a restore and the epoch-range readers read every
move since X from it. Each test here fails on the code before it:

- A to B to C after X: the restore and the writers' backstop name the key for A and for B (before: B alone);
- a move before X (or the range) is not read, every one inside it is (before: the range's second move went unread);
- quiet-time cleanup removes the moves below its floor, and keeps the rest (before: nothing removed them).

The write's own statement (the moves row appended from the row it returned, in the same statement) is held by
``test_tenant_moves``; here a move is made as that statement leaves the table and its moves.

The data statements run in an in-process DuckDB (the keyed helpers' SQL is one spelling there and on PostgreSQL, a few
PostgreSQL functions spelled as DuckDB's below).
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, Final, cast

import duckdb

from threetears.core.change_note_fold import RestoreFolder
from threetears.core.collections.cleanup_progress import CLEANUP_PROGRESS_TABLE
from threetears.core.collections.epoch_reads import epoch_keys_page, epoch_range, range_rows
from threetears.core.collections.noted_change import NotedChange
from threetears.core.collections.quiet_cleanup import LoadGate, QuietCleanup
from threetears.core.collections.schema_backed import STRING_TYPE, Column, TableSchema
from threetears.core.collections.scope_epochs import WHOLE, EpochSnapshot, ScopeEpochs
from threetears.core.collections.table_restore import TableChanges, restored_tables
from threetears.core.ddl.row_epoch import RESTORE_BUCKETS, ROW_STAMPED
from threetears.core.ddl.run_markers import RUN_MARKERS_TABLE
from threetears.core.ddl.tenant_moves import moved_assignments
from threetears.core.eviction import Declared, Policy
from threetears.core.keyed import ThreadedDuckDB
from threetears.core.tenancy import tenant_column, value_tenant

#: the tables' schema: DuckDB's own, which every connection's bare names resolve to, as a pod's ``search_path`` does
SCHEMA: Final = "main"

#: the moves table (``tenant_moves.TENANT_MOVES_TABLE``), named here so this module loads on the code before it
TENANT_MOVES_TABLE: Final = "threetears_tenant_moves"
TABLE: Final = "moving_runs"
BUCKET: Final = 3

#: a stamped table whose tenant column (and office, a declared invalidation column) a write may change in place
MOVING: Final = TableSchema(
    name=TABLE,
    primary_key="run_id",
    columns=[Column("run_id", STRING_TYPE), Column("customer_id", STRING_TYPE), Column("office", STRING_TYPE)],
    tenancy=tenant_column("customer_id"),
    row_epoch=ROW_STAMPED,
    moveable=("customer_id", "office"),
    invalidation_columns=("office",),
    l1_eviction=Declared(Policy.EVICTABLE, "a test's stamped table whose rows move between tenants"),
)

#: PostgreSQL's spellings the statements use, as DuckDB spells them
_SPELLINGS: Final = (
    (re.compile(r"encode\(sha256\(convert_to\((.+?), 'UTF8'\)\), 'hex'\)"), r"sha256(\1)"),
    (re.compile(r"array_to_json\(ARRAY\["), "to_json(["),
    (re.compile(r"json_build_object\("), "json_object("),
)


def _duck(query: str) -> str:
    for pattern, spelled in _SPELLINGS:
        query = pattern.sub(spelled, query)
    return query


def _declaration(schema: TableSchema) -> str:
    """the restore declaration the table's DDL gives its restore index, as the catalog then holds it."""
    [comment] = [statement for statement in schema.create_statements(yugabyte=True) if statement.startswith("COMMENT")]
    found = re.search(r"IS '(.*)'$", comment)
    assert found is not None, comment
    return found.group(1).replace("''", "'")


class _Db:
    """the moving table, its moves and run markers in DuckDB; the catalog answered from the declaration. Every
    statement is recorded."""

    def __init__(self) -> None:
        self.conn = duckdb.connect()
        moved = ", ".join(f"{c}_moved_from TEXT, {c}_moved_epoch BIGINT" for c in ("customer_id", "office"))
        self.conn.execute(
            f"CREATE TABLE {TABLE} (run_id TEXT PRIMARY KEY, customer_id TEXT, office TEXT, {moved}, "
            "epoch BIGINT NOT NULL, epoch_bucket SMALLINT NOT NULL, gone BOOLEAN NOT NULL DEFAULT false)"
        )
        self.conn.execute(
            f"CREATE TABLE {TENANT_MOVES_TABLE} (table_name TEXT, scope TEXT, epoch BIGINT, keys TEXT, tenant TEXT, "
            "columns TEXT, PRIMARY KEY (table_name, scope, epoch, keys))"
        )
        self.conn.execute(f"CREATE TABLE {RUN_MARKERS_TABLE} (table_name TEXT, scope TEXT, prefix TEXT, epoch BIGINT)")
        self._db = ThreadedDuckDB(self.conn)
        self.sent: list[tuple[str, tuple[Any, ...]]] = []

    def write(self, run: str, customer: str, office: str, epoch: int) -> None:
        """a write of the row as the store makes it: a new row, or the stored one updated in place, its move recorded
        by the ``SET`` terms and appended to the moves table (as ``MoveRecord.append_sql`` appends it from the row the
        write returned: ``test_tenant_moves`` runs that statement and holds it to this row)."""
        if not self.conn.execute(f"SELECT 1 FROM {TABLE} WHERE run_id = ?", [run]).fetchall():
            self.conn.execute(
                f"INSERT INTO {TABLE} (run_id, customer_id, office, epoch, epoch_bucket) VALUES (?, ?, ?, ?, ?)",
                [run, customer, office, epoch, BUCKET],
            )
            return
        new = {"customer_id": f"'{customer}'", "office": f"'{office}'"}
        terms = ", ".join(moved_assignments(column, value, str(epoch)) for column, value in new.items())
        self.conn.execute(
            f"UPDATE {TABLE} SET {terms}, customer_id = ?, office = ?, epoch = ? WHERE run_id = ?",
            [customer, office, epoch, run],
        )
        [(left_customer, left_office, moved_at)] = self.conn.execute(
            f"SELECT CASE WHEN customer_id_moved_epoch = epoch THEN customer_id_moved_from ELSE customer_id END, "
            f"CASE WHEN office_moved_epoch = epoch THEN office_moved_from ELSE office END, "
            f"customer_id_moved_epoch = epoch OR office_moved_epoch = epoch FROM {TABLE} WHERE run_id = ?",
            [run],
        ).fetchall()
        if moved_at:
            self.conn.execute(
                f"INSERT INTO {TENANT_MOVES_TABLE} VALUES (?, ?, ?, ?, ?, ?)",
                [
                    TABLE,
                    str(BUCKET),
                    epoch,
                    json.dumps([run]),
                    value_tenant(left_customer),
                    json.dumps({"office": left_office, "customer_id": left_customer}),
                ],
            )

    def moves(self) -> list[tuple[int, str]]:
        return sorted(
            (int(epoch), str(tenant))
            for epoch, tenant in self.conn.execute(
                f"SELECT epoch, tenant FROM {TENANT_MOVES_TABLE} WHERE table_name = ?", [TABLE]
            ).fetchall()
        )

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        self.sent.append((query, args))
        if "pg_description" in query:
            return [
                {"table_schema": SCHEMA, "table_name": TABLE, "declared": _declaration(MOVING), "run_markers": True}
            ]
        if "pg_attribute" in query:
            return [{"column_name": column, "column_type": "text"} for column in args[1] if column in MOVING.pk_columns]
        return await self._db.fetch(_duck(query), *args)

    async def execute(self, query: str, *args: Any) -> str:
        self.sent.append((query, args))
        return await self._db.execute(_duck(query), *args)

    async def copy_from_query(self, query: str, *args: Any, output: Any, **_: Any) -> str:
        """the rows as PostgreSQL's ``COPY ... CSV`` writes them: the first three columns quoted, a NULL empty."""
        self.sent.append((query, args))
        for row in await self._db.fetch(_duck(query), *args):
            values = list(row.values())
            fields = ['"' + str(value).replace('"', '""') + '"' for value in values[:3]]
            last = values[3]
            fields.append("" if last is None else '"' + str(last).replace('"', '""') + '"')
            await output((",".join(fields) + "\n").encode())
        return "COPY"

    @asynccontextmanager
    async def _acquire(self) -> AsyncIterator[_Db]:
        yield self

    def acquire(self) -> Any:
        return self._acquire()


async def _restored(db: _Db, floor: int) -> list[tuple[str | None, tuple[str, ...], dict[str, Any] | None]]:
    """what a restore from X = ``floor`` appends: each record's tenant, key and invalidation columns."""
    tables = await restored_tables(db)
    folder = RestoreFolder()
    folder.start()
    try:
        changes: list[NotedChange] = [change async for change in TableChanges(db, tables, floor, folder, None)]
    finally:
        await folder.close()
        await folder.closed()
    found = [
        (change.tenant, tuple(row["ids"]), row.get("columns"))
        for change in changes
        for body in change.bodies
        for row in json.loads(body)["rows"]
    ]
    return sorted(found, key=repr)


def _bracket() -> tuple[datetime, datetime, int, int]:
    now = datetime.now(UTC)
    since, until = now - timedelta(minutes=1), now + timedelta(minutes=1)
    floor, ceiling = epoch_range((since, until))
    return since, until, floor, ceiling


A, B, C, Z = value_tenant("ca"), value_tenant("cb"), value_tenant("cc"), value_tenant("cz")


async def test_a_row_moved_twice_since_x_is_heard_by_both_tenants_it_left() -> None:
    """Flips if a row moved A to B, then B to C, both after X, comes back from a restore (or the writers' backstop)
    for B and C alone: the row kept only its last move, so A's followers, which lost the log, would keep the row."""
    db = _Db()
    _, _, floor, ceiling = _bracket()
    db.write("r1", "ca", "gov", floor - 50)
    db.write("r1", "cb", "gov", floor + 10)
    db.write("r1", "cc", "sen", floor + 20)
    assert db.moves() == [(floor + 10, A), (floor + 20, B)], "each move is its own row, naming the tenant it left"

    restored = await _restored(db, floor)

    assert restored == sorted(
        [
            (C, ("r1",), {"office": "sen", "customer_id": "cc"}),
            (B, ("r1",), {"office": "gov", "customer_id": "cb"}),
            (A, ("r1",), {"office": "gov", "customer_id": "ca"}),
        ],
        key=repr,
    ), restored

    [table] = await restored_tables(db)
    backstop = [row async for rows in range_rows(db, table, floor, ceiling, page=10) for row in rows]
    assert sorted((row.tenant, row.keys, sorted((row.columns or {}).items())) for row in backstop) == sorted(
        [
            (C, ("r1",), [("customer_id", "cc"), ("office", "sen")]),
            (B, ("r1",), [("customer_id", "cb"), ("office", "gov")]),
            (A, ("r1",), [("customer_id", "ca"), ("office", "gov")]),
        ]
    ), backstop


async def test_a_move_outside_the_window_is_not_read_and_every_move_inside_it_is() -> None:
    """Flips if a restore from X reads a move made before X, or misses one made after it when the row moved again
    later; or if a keys-unknown page of an epoch range pages the key for a tenant the row left outside the range, or
    not for the tenant it left inside it (before the moves table, a row moved again after the range was not read at
    all: its epoch was past the ceiling)."""
    db = _Db()
    since, _, floor, _ = _bracket()

    def at(ms: int) -> int:
        return floor + (ms << 15)

    db.write("r1", "cz", "gov", at(-50_000))
    db.write("r1", "ca", "gov", at(-5_000))  # left Z before X
    db.write("r1", "cb", "gov", at(3_000))  # left A inside the window
    db.write("r1", "cc", "gov", at(6_000))  # left B inside the window

    left = {tenant for tenant, _, _ in await _restored(db, floor)} - {C}
    assert left == {A, B}, left

    # a range that ends between the second and third moves: A's move inside it, B's after it, Z's before it
    narrow = (since, since + timedelta(seconds=4))
    for tenant, expected in ((Z, []), (A, [("r1",)]), (B, [])):
        page = await epoch_keys_page(
            db,
            table=TABLE,
            platform_schema=SCHEMA,
            tenant_token=tenant,
            noted_between=narrow,
            cursor=None,
            limit=10,
            max_bytes=10_000,
        )
        assert not isinstance(page, str), page
        assert list(page.keys) == expected, (tenant, page)


async def test_the_moves_read_is_one_range_of_the_moves_tables_key() -> None:
    """Flips if a restore reads a table's moves other than by the moves table's key (the table, the bucket, the epoch
    from X): a read of every move, or of the moved table's rows a second time."""
    db = _Db()
    _, _, floor, _ = _bracket()
    db.write("r1", "ca", "gov", floor - 50)
    db.write("r1", "cb", "gov", floor + 10)
    await _restored(db, floor)
    moves = [(query, args) for query, args in db.sent if TENANT_MOVES_TABLE in query]
    assert len(moves) == RESTORE_BUCKETS, moves
    for query, args in moves:
        assert 't."table_name" = $1' in query and 't."scope" = $2' in query and 't."epoch" >= $3' in query, query
        assert args[0] == TABLE and args[2] == floor, args
    assert sorted(args[1] for _, args in moves) == sorted(str(bucket) for bucket in range(RESTORE_BUCKETS))


class _Epochs:
    """the scope epochs QuietCleanup listens to: a quiet snapshot, nothing writing."""

    def on_change(self, listener: Any) -> Any:
        return lambda: None

    async def snapshot(self) -> EpochSnapshot:
        return EpochSnapshot(version=1, writing=None, epochs={WHOLE: 1}, previous={})


async def test_quiet_cleanup_removes_the_moves_below_its_floor_and_keeps_the_rest() -> None:
    """Flips if quiet-time cleanup leaves a move below its floor F (no restore reads it), removes one at or above F (a
    restore may), or touches another table's moves."""
    db = _Db()
    db.conn.execute(
        f"CREATE TABLE {CLEANUP_PROGRESS_TABLE} (table_name TEXT, scope TEXT, cleaned_to BIGINT, "
        "PRIMARY KEY (table_name, scope))"
    )
    db.write("r1", "ca", "gov", 10)
    db.write("r1", "cb", "gov", 50)
    db.write("r1", "cc", "gov", 150)
    db.conn.execute(f"INSERT INTO {TENANT_MOVES_TABLE} VALUES ('other_table', '3', 40, '[\"x\"]', NULL, '{{}}')")

    async def window() -> int | None:
        return 100

    done = await QuietCleanup(db, (MOVING,), LoadGate(cast(ScopeEpochs, _Epochs())), window, run_markers=None).clean()

    assert done.finished, done
    assert db.moves() == [(150, B)], "the move below F went, the one above it stayed"
    assert db.conn.execute(f"SELECT table_name FROM {TENANT_MOVES_TABLE} WHERE table_name <> '{TABLE}'").fetchall() == [
        ("other_table",)
    ]
