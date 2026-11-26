"""A restore from the tables is as exact as the change-note ring it replaced: every declared invalidation column, and
every reset.

0.95 (append-only L3) rebuilds a lost change log, the writers' backstop and a keys-unknown page from the tables' own
rows stamped since X (``table_restore``, ``epoch_reads``). The ring's notes carried a row's key and EVERY declared
invalidation column, and a ``TRUNCATE`` left a whole-table note naming the reset. These tests hold the rebuilt reads to
that:

- a table declaring two invalidation columns (its tenant column and another) comes back with both, from the restore
  and from the backstop's read, a row a write moved in place again with the values it left;
- a reset after X (``SqlL3Backend.truncate``) leaves its reset marker, and the restore, the backstop and a keys-unknown
  page hear the table cleared whole, as a reset.

The data statements run in an in-process DuckDB (the keyed helpers' SQL is one spelling there and on PostgreSQL, a few
PostgreSQL functions spelled as DuckDB's below); the catalog statements are answered from the tables' declarations, as
``TableSchema.create_statements`` writes them. The tables' schemas are made inside each test, so the module loads on a
release whose ``TableSchema`` has no ``invalidation_columns``.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import duckdb

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.change_note_fold import RestoreFolder
from threetears.core.collections.epoch_change import RESET_REASON
from threetears.core.collections.epoch_reads import epoch_keys_page, epoch_range, range_rows
from threetears.core.collections.lost_writes import _changes_of
from threetears.core.collections.noted_change import NotedChange
from threetears.core.collections.schema_backed import STRING_TYPE, Column, TableSchema
from threetears.core.collections.table_restore import TableChanges, restored_tables
from threetears.core.ddl.row_epoch import ROW_STAMPED, row_versions
from threetears.core.ddl.run_markers import RUN_MARKERS_TABLE
from threetears.core.ddl.tenant_moves import TENANT_MOVES_TABLE
from threetears.core.eviction import Declared, Policy
from threetears.core.keyed import ThreadedDuckDB
from threetears.core.tenancy import tenant_column, value_tenant

SCHEMA: Final = "platform"
STAMPED: Final = "exact_runs"
VERSIONED: Final = "exact_results"
EPOCHS: Final = "exact_epochs"
SCOPE: Final = "join_state_code"


def _declaration(schema: TableSchema) -> str:
    """the restore declaration the table's DDL gives its restore index, as the catalog then holds it."""
    [comment] = [statement for statement in schema.create_statements(yugabyte=True) if statement.startswith("COMMENT")]
    found = re.search(r"IS '(.*)'$", comment)
    assert found is not None, comment
    return found.group(1).replace("''", "'")


def _schemas(**declared: Any) -> dict[str, TableSchema]:
    """the two tables: a stamped one whose tenant column and office a write may move in place, and a versioned one;
    each a column tenancy by ``customer_id``, with ``declared`` (its invalidation columns, when the test names them)."""
    stamped = TableSchema(
        name=STAMPED,
        primary_key="run_id",
        columns=[Column("run_id", STRING_TYPE), Column("customer_id", STRING_TYPE), Column("office", STRING_TYPE)],
        tenancy=tenant_column("customer_id"),
        row_epoch=ROW_STAMPED,
        moveable=("customer_id", "office"),
        l1_eviction=Declared(Policy.EVICTABLE, "a test's stamped table"),
        **declared,
    )
    versioned = TableSchema(
        name=VERSIONED,
        primary_key=(SCOPE, "run_id"),
        columns=[
            Column(SCOPE, STRING_TYPE),
            Column("run_id", STRING_TYPE),
            Column("customer_id", STRING_TYPE),
            Column("office", STRING_TYPE),
        ],
        tenancy=tenant_column("customer_id"),
        row_epoch=row_versions(SCOPE, epochs=EPOCHS, scope_prefix="state:"),
        l1_eviction=Declared(Policy.EVICTABLE, "a test's versioned table"),
        **declared,
    )
    return {STAMPED: stamped, VERSIONED: versioned}


#: PostgreSQL's spellings the reads use, as DuckDB spells them
_SPELLINGS: Final = (
    (re.compile(r"encode\(sha256\(convert_to\((.+?), 'UTF8'\)\), 'hex'\)"), r"sha256(\1)"),
    (re.compile(r"array_to_json\(ARRAY\["), "to_json(["),
    (re.compile(r"json_build_object\("), "json_object("),
)


def _duck(query: str) -> str:
    for pattern, spelled in _SPELLINGS:
        query = pattern.sub(spelled, query)
    return query


class _Db:
    """the two tables, their scope epochs and run markers in DuckDB; the catalog answered from the declarations; the
    DDL lock's statements answered as granted. Every statement is recorded."""

    def __init__(self, declared: dict[str, TableSchema]) -> None:
        self._conn = duckdb.connect()
        self._conn.execute(f"CREATE SCHEMA {SCHEMA}")
        moved = ", ".join(
            f"{column}_moved_from TEXT, {column}_moved_epoch BIGINT" for column in ("customer_id", "office")
        )
        self._conn.execute(
            f"CREATE TABLE {SCHEMA}.{STAMPED} (run_id TEXT, customer_id TEXT, office TEXT, {moved}, "
            "epoch BIGINT NOT NULL, epoch_bucket SMALLINT NOT NULL, gone BOOLEAN NOT NULL DEFAULT false)"
        )
        self._conn.execute(
            f"CREATE TABLE {SCHEMA}.{VERSIONED} ({SCOPE} TEXT, run_id TEXT, customer_id TEXT, office TEXT, "
            "epoch BIGINT NOT NULL, gone BOOLEAN NOT NULL DEFAULT false)"
        )
        self._conn.execute(f"CREATE TABLE {SCHEMA}.{EPOCHS} (scope TEXT, last_epoch BIGINT)")
        self._conn.execute(
            f"CREATE TABLE {SCHEMA}.{RUN_MARKERS_TABLE} (table_name TEXT, scope TEXT, prefix TEXT, epoch BIGINT)"
        )
        self._conn.execute(
            f"CREATE TABLE {SCHEMA}.{TENANT_MOVES_TABLE} (table_name TEXT, scope TEXT, epoch BIGINT, keys TEXT, "
            "tenant TEXT, columns TEXT)"
        )
        self._db = ThreadedDuckDB(self._conn)
        self.declared = declared
        self.sent: list[tuple[str, tuple[Any, ...]]] = []

    def stamped(
        self, run: str, customer: str, office: str, epoch: int, *, left: tuple[str, str, int] | None = None
    ) -> None:
        """a stamped row; ``left`` the customer and office a move at its epoch left, appended to the moves table as the
        moving write appends it (``tenant_moves``)."""
        old_customer, old_office, at = left if left is not None else (None, None, None)
        self._conn.execute(
            f"INSERT INTO {SCHEMA}.{STAMPED} VALUES (?, ?, ?, ?, ?, ?, ?, ?, 3, false)",
            [run, customer, office, old_customer, at, old_office, at, epoch],
        )
        if left is not None:
            self._conn.execute(
                f"INSERT INTO {SCHEMA}.{TENANT_MOVES_TABLE} VALUES (?, '3', ?, ?, ?, ?)",
                [
                    STAMPED,
                    left[2],
                    json.dumps([run]),
                    value_tenant(left[0]),
                    json.dumps({"office": left[1], "customer_id": left[0]}),
                ],
            )

    def version(self, state: str, run: str, customer: str, office: str, epoch: int) -> None:
        """a version of the versioned table, its scope touched at its epoch."""
        self._conn.execute(
            f"INSERT INTO {SCHEMA}.{VERSIONED} VALUES (?, ?, ?, ?, ?, false)", [state, run, customer, office, epoch]
        )
        self._conn.execute(f"INSERT INTO {SCHEMA}.{EPOCHS} VALUES (?, ?)", [f"state:{state}", epoch])

    def rows(self, table: str) -> int:
        return int(self._conn.execute(f"SELECT count(*) FROM {SCHEMA}.{table}").fetchall()[0][0])

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        self.sent.append((query, args))
        if "pg_description" in query and "to_regclass($1)" in query:
            # the table a reset names (run_markers.record_table_reset)
            name = str(args[0]).rpartition(".")[2]
            return [
                {"table_schema": SCHEMA, "table_name": name, "declared": name in self.declared, "run_markers": True}
            ]
        if "pg_description" in query:
            return [
                {"table_schema": SCHEMA, "table_name": name, "declared": _declaration(schema), "run_markers": True}
                for name, schema in sorted(self.declared.items())
            ]
        if "pg_attribute" in query:
            schema = self.declared[str(args[0]).split(".")[-1].strip('"')]
            return [{"column_name": column, "column_type": "text"} for column in args[1] if column in schema.pk_columns]
        if "current_database" in query:
            return [{"database_name": "aibots"}]
        if "pg_try_advisory_lock" in query:
            return [{"acquired": True}]
        if "pg_advisory_unlock" in query:
            return [{"released": True}]
        return await self._db.fetch(_duck(query), *args)

    async def execute(self, query: str, *args: Any) -> str:
        self.sent.append((query, args))
        await self._db.execute(_duck(query), *args)
        return "OK"

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
    async def _acquire(self) -> AsyncIterator[_Conn]:
        yield _Conn(self)

    def acquire(self) -> Any:
        return self._acquire()


class _Conn:
    """one connection of the database: its statements go to it."""

    def __init__(self, db: _Db) -> None:
        self._db = db

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        return await self._db.fetch(query, *args)

    async def execute(self, query: str, *args: Any) -> str:
        return await self._db.execute(query, *args)


async def _restored(db: _Db, floor: int) -> list[NotedChange]:
    """what a restore from X = ``floor`` appends: every table's changes, folded in the restore's worker."""
    tables = await restored_tables(db)
    folder = RestoreFolder()
    folder.start()
    try:
        return [change async for change in TableChanges(db, tables, floor, folder, None)]
    finally:
        await folder.close()
        await folder.closed()


def _records(changes: list[NotedChange]) -> list[tuple[str, str | None, tuple[str, ...], dict[str, Any] | None]]:
    """each restored row: its table, the tenant its record goes under, its key and its invalidation columns."""
    found = []
    for change in changes:
        for body in change.bodies:
            for row in json.loads(body)["rows"]:
                found.append((change.table, change.tenant, tuple(row["ids"]), row.get("columns")))
    return sorted(found, key=repr)


def _bracket() -> tuple[datetime, datetime, int]:
    now = datetime.now(UTC)
    since, until = now - timedelta(minutes=1), now + timedelta(minutes=1)
    return since, until, epoch_range((since, until))[0]


async def test_a_table_declaring_two_invalidation_columns_comes_back_carrying_both() -> None:
    """Flips if a restore's records, or the backstop's rows, of a table declaring two invalidation columns (its tenant
    column and ``office``) carry only the tenant column, as the restore declaration stored only it before; or if a
    stamped row a write moved since X does not come back once more with the customer and office it left, under the
    tenant it left; or if the restore declaration does not name both."""
    declared = _schemas(invalidation_columns=("office",))
    assert json.loads(_declaration(declared[STAMPED]))["columns"] == ["office", "customer_id"]
    assert json.loads(_declaration(declared[VERSIONED]))["columns"] == ["office", "customer_id"]
    db = _Db(declared)
    since, until, floor = _bracket()
    db.stamped("r1", "c1", "gov", floor + 5)
    db.stamped("r2", "c1", "gov", floor + 7, left=("c0", "sen", floor + 7))
    db.stamped("r3", "c1", "gov", floor - 5)  # before X: no record names it
    db.version("VA", "r9", "c2", "ag", floor + 9)

    restored = _records(await _restored(db, floor))

    c0, c1, c2 = value_tenant("c0"), value_tenant("c1"), value_tenant("c2")
    assert restored == sorted(
        [
            (STAMPED, c1, ("r1",), {"office": "gov", "customer_id": "c1"}),
            (STAMPED, c1, ("r2",), {"office": "gov", "customer_id": "c1"}),
            (STAMPED, c0, ("r2",), {"office": "sen", "customer_id": "c0"}),
            (VERSIONED, c2, ("VA", "r9"), {"office": "ag", "customer_id": "c2"}),
        ],
        key=repr,
    ), restored

    _, ceiling = epoch_range((since, until))
    tables = {table.table: table for table in await restored_tables(db)}
    backstop = [row async for rows in range_rows(db, tables[STAMPED], floor, ceiling, page=10) for row in rows]
    assert sorted((row.tenant, row.keys, sorted((row.columns or {}).items())) for row in backstop) == sorted(
        [
            (c1, ("r1",), [("customer_id", "c1"), ("office", "gov")]),
            (c1, ("r2",), [("customer_id", "c1"), ("office", "gov")]),
            (c0, ("r2",), [("customer_id", "c0"), ("office", "sen")]),
        ]
    ), backstop
    page = await epoch_keys_page(
        db,
        table=STAMPED,
        platform_schema=SCHEMA,
        tenant_token=c0,
        noted_between=(since, until),
        cursor=None,
        limit=10,
        max_bytes=10_000,
    )
    assert not isinstance(page, str), page
    assert list(zip(page.keys, page.columns, strict=True)) == [(("r2",), {"customer_id": "c0", "office": "sen"})], (
        "the tenant a row left pages it back, with the values it left"
    )


async def test_a_reset_after_x_is_heard_whole_by_the_restore_and_the_backstop() -> None:
    """Flips if a ``TRUNCATE`` after X (an operator's reset, ``SqlL3Backend.truncate``) leaves nothing a restore or the
    writers' backstop reads ``epoch >= X`` from (no row survives it, and no change note says so since the ring
    retired): the restore must append the table whole, as a reset, and so must the backstop's read of a window holding
    it; a keys-unknown page of the window says whole. A table reset before X is not heard."""
    declared = _schemas()
    db = _Db(declared)
    since, until, floor = _bracket()
    db.stamped("r1", "c1", "gov", floor - 50)
    db.stamped("r2", "c1", "gov", floor + 1)
    db.version("VA", "r9", "c2", "ag", floor - 40)

    store = SqlL3Backend(db)
    await store.truncate(f"{SCHEMA}.{STAMPED}", reason=RESET_REASON)
    await store.truncate(f"{SCHEMA}.{VERSIONED}", reason=RESET_REASON)

    assert db.rows(STAMPED) == 0 and db.rows(VERSIONED) == 0, "the reset emptied the tables"
    restored = await _restored(db, floor)
    resets = sorted((change.table, change.tenant) for change in restored if change.whole_table)
    assert resets == [(VERSIONED, None), (STAMPED, None)], restored
    assert {change.reason for change in restored if change.whole_table} == {RESET_REASON}, restored
    assert not _records(restored), "an emptied table has no row left to name"

    tables = {table.table: table for table in await restored_tables(db)}
    for name in (STAMPED, VERSIONED):
        advances = await _changes_of(
            db,
            platform_schema=SCHEMA,
            table=name,
            relations=[tables[name]],
            since=since,
            until=until,
            tenants=(None,),
        )
        wholes = [advance.whole for advance in advances if advance.whole is not None]
        assert [(each.table, each.whole_table, each.reason) for each in wholes] == [(name, True, RESET_REASON)], (
            advances
        )
        page = await epoch_keys_page(
            db,
            table=name,
            platform_schema=SCHEMA,
            tenant_token=value_tenant("c1"),
            noted_between=(since, until),
            cursor=None,
            limit=10,
            max_bytes=10_000,
        )
        assert not isinstance(page, str) and page.whole_table, (name, page)

    later = await _restored(db, epoch_range((until + timedelta(minutes=1), until + timedelta(minutes=2)))[0])
    assert not [change for change in later if change.whole_table], "a reset before X is not restored"
