"""what a ``ROW_STAMPED`` table's tests hold it to, shared by every package that declares one.

Append-only L3 (``design-append-only-l3.md`` decision 3): a stamped table is its own change record, so the
change-note ring can be retired. Each package that stamps its tables proves the same four things of each, and
writes no double of its own for them:

- **the declaration** renders the stored bucket, the ``(epoch_bucket ASC, epoch)`` index split a tablet a bucket on
  YugabyteDB and its ``"kind":"stamped"`` comment (:func:`assert_declared_stamped`);
- **the migration** that stamps an existing table names it as its declaration renders it, and takes the six steps
  in order, its backfill a keyed ``UPDATE`` a page of keys (:func:`assert_migration_stamps`, over
  :class:`MigrationStore`);
- **a write** carries a fresh epoch risen past the stored one, and a delete is a gone mark (over
  :class:`RecordingPool`, :func:`stamps_in`);
- **a restore** finds the table from the catalog's comments and reads its 16 buckets, one keyed statement a bucket
  (:func:`assert_restore_reads_every_bucket`).

Every double here is function-local state of one test; nothing is shared across tests.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock

from threetears.core.collections.table_restore import StampedTable, TableChanges, restored_tables
from threetears.core.data.migrations.stamp_tables import StampingTable, bucket_statements, stamping_table_of
from threetears.core.ddl.tenant_moves import moved_columns
from threetears.core.ddl.row_epoch import (
    BUCKET_COLUMN,
    EPOCH_COLUMN,
    GONE_COLUMN,
    RESTORE_BUCKETS,
    bucket_epoch_index_name,
    next_epoch,
)

__all__ = [
    "MigrationStore",
    "RecordingPool",
    "assert_declared_stamped",
    "assert_migration_stamps",
    "assert_restore_reads_every_bucket",
    "stamps_in",
]

#: what YugabyteDB's ``version()`` answers, as the migration probes it
_YUGABYTE_VERSION = "PostgreSQL 15.12-YB-2025.1.0.0-b0 on x86_64"


def _bare(name: str) -> str:
    return name.rpartition(".")[2]


def assert_declared_stamped(schema: Any) -> None:
    """``schema`` is ``ROW_STAMPED``, and its L3 DDL carries the stored bucket, the ``(epoch_bucket ASC, epoch)`` index
    split a tablet a bucket on YugabyteDB, and the restore's ``"kind":"stamped"`` comment naming its key.

    :param schema: the table's ``TableSchema``
    :ptype schema: TableSchema
    :return: nothing
    :rtype: None
    """
    assert schema.row_epoch is not None and not schema.row_epoch.versions, f"{schema.name} is ROW_STAMPED"
    index = bucket_epoch_index_name(_bare(schema.name))
    assert index in {declared.name for declared in schema.indexes}, schema.name
    columns = {column.name: column for column in schema.columns}
    assert not columns[EPOCH_COLUMN].nullable and not columns[GONE_COLUMN].nullable, schema.name
    create, *rest = schema.create_statements(yugabyte=True)
    assert f"{BUCKET_COLUMN} SMALLINT GENERATED ALWAYS AS (" in create and "STORED NOT NULL" in create, create
    assert f"{EPOCH_COLUMN} BIGINT NOT NULL" in create, create
    [statement] = [each for each in rest if each.startswith("CREATE INDEX") and index in each]
    assert f"({BUCKET_COLUMN} ASC, {EPOCH_COLUMN})" in statement, statement
    assert statement.endswith(
        "SPLIT AT VALUES ((1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (15))"
    ), statement
    [comment] = [each for each in rest if each.startswith("COMMENT ON INDEX")]
    declared = json.loads(comment.split(" IS '", 1)[1][:-1].replace("''", "'"))
    assert declared["kind"] == "stamped" and declared["buckets"] == RESTORE_BUCKETS == 16, declared
    assert tuple(declared["keys"]) == tuple(schema.pk_columns), declared


class RecordingPool:
    """an asyncpg-shaped pool that keeps every statement, whitespace folded, and answers each write one row.

    :ivar sent: each statement and its parameters, in order
    :ivar answer: what a read answers, given its statement and parameters (no rows when None)
    """

    def __init__(self, answer: Callable[[str, tuple[Any, ...]], list[dict[str, Any]]] | None = None) -> None:
        self.sent: list[tuple[str, tuple[Any, ...]]] = []
        self.answer = answer

    def _read(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        folded = " ".join(sql.split())
        self.sent.append((folded, params))
        return [] if self.answer is None else self.answer(folded, params)

    async def execute(self, sql: str, *params: Any) -> str:
        self._read(sql, params)
        return "UPDATE 1" if sql.lstrip().upper().startswith("UPDATE") else "INSERT 0 1"

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        return self._read(sql, params)

    async def fetchrow(self, sql: str, *params: Any) -> dict[str, Any] | None:
        rows = self._read(sql, params)
        return rows[0] if rows else None

    async def fetchval(self, sql: str, *params: Any) -> Any:
        rows = self._read(sql, params)
        return next(iter(rows[0].values())) if rows else None


def stamps_in(params: Sequence[Any], before: int) -> list[int]:
    """the epochs among a statement's parameters drawn after ``before`` (a stamp is the one int past it).

    :param params: the statement's parameters
    :ptype params: Sequence[Any]
    :param before: an epoch drawn before the write
    :ptype before: int
    :return: them
    :rtype: list[int]
    """
    return [value for value in params if isinstance(value, int) and not isinstance(value, bool) and value > before]


class MigrationStore:
    """the migration runner's ``DataStore`` as a stamping migration uses it: ``version()`` answers YugabyteDB, a keyset
    page of a table's keys answers ``keys`` once (and then none), and every statement is kept in order.

    :ivar keys: each table's keys a backfill's first page reads, by table
    :ivar statements: each statement and its parameters, in order
    """

    def __init__(self, keys: dict[str, list[dict[str, Any]]]) -> None:
        self.keys = keys
        self.statements: list[tuple[str, tuple[Any, ...]]] = []

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        folded = " ".join(sql.split())
        self.statements.append((folded, params))
        if folded.startswith("SELECT version()"):
            # the migration's own probe names the column, the keyed helpers' dialect probe another
            return [{"version": _YUGABYTE_VERSION, "v": _YUGABYTE_VERSION}]
        for table, keys in self.keys.items():
            named = f'FROM "{table}"' in folded or f"FROM {table} " in f"{folded} "
            if named and keys and not params:
                # the first page; a page after its last key reads none
                return list(keys)
        return []

    async def execute(self, sql: str, *params: Any) -> str:
        self.statements.append((" ".join(sql.split()), params))
        return "UPDATE 1"


def assert_migration_stamps(store: MigrationStore, frozen: Sequence[StampingTable], schemas: Sequence[Any]) -> None:
    """a stamping migration, run over ``store``, named each table as its declaration renders it and took the six steps
    for each in order: the epoch added, the gone flag, the backfill one keyed ``UPDATE`` of the page's keys (bound as
    arrays, only rows whose epoch is NULL, never the whole table), ``NOT NULL``, the stored bucket, the index split a
    tablet
    a bucket, its declaration; and never a ``TRUNCATE``, a ``DROP`` or a ``DELETE``.

    :param store: the store the migration ran over
    :ptype store: MigrationStore
    :param frozen: the tables the migration names
    :ptype frozen: Sequence[StampingTable]
    :param schemas: their declarations, in the same order
    :ptype schemas: Sequence[TableSchema]
    :return: nothing
    :rtype: None
    """
    assert [table.name for table in frozen] == [_bare(schema.name) for schema in schemas]
    for table, schema in zip(frozen, schemas, strict=True):
        assert table == stamping_table_of(schema), (table, stamping_table_of(schema))
        rendered = schema.create_statements(yugabyte=True)
        [index] = [each for each in rendered if each.startswith("CREATE INDEX") and table.index in each]
        [comment] = [each for each in rendered if each.startswith("COMMENT ON INDEX")]
        _, created, declared = bucket_statements(table, yugabyte=True)
        assert created.split("\n")[-1].replace('"', "").replace(" IF NOT EXISTS", "") == index.replace('"', "")
        assert declared == comment
    sent = [sql for sql, _ in store.statements]
    assert not [sql for sql in sent if sql.upper().startswith(("TRUNCATE", "DROP", "DELETE"))], sent
    for table in frozen:
        quoted = f'"{table.name}"'
        # its own statements: the index and its comment are named by the table's name, quoted apart from it
        mine = [(sql, params) for sql, params in store.statements if quoted in sql or f'"{table.index}"' in sql]
        steps = [sql for sql, _ in mine if not sql.startswith("SELECT")]
        assert steps[:2] == [
            f"ALTER TABLE {quoted} ADD COLUMN IF NOT EXISTS {EPOCH_COLUMN} bigint",
            f"ALTER TABLE {quoted} ADD COLUMN IF NOT EXISTS {GONE_COLUMN} boolean NOT NULL DEFAULT false",
        ], steps
        update, params = next((sql, params) for sql, params in mine if sql.startswith("UPDATE"))
        assert steps[2] == update and f'"{EPOCH_COLUMN}" IS NULL' in update, update
        assert all(isinstance(param, (list, tuple)) for param in params[1:]), (update, params)
        assert "ANY(" in update or "unnest(" in update, update
        assert steps[3:] == [
            f"ALTER TABLE {quoted} ALTER COLUMN {EPOCH_COLUMN} SET NOT NULL",
            *(" ".join(each.split()) for each in bucket_statements(table, yugabyte=True)),
        ], steps


class _Catalog:
    """the keeper's pool: the catalog's restore declarations (each table's, as its declaration comments its index), and
    a ``COPY`` a statement that answers no rows."""

    def __init__(self, schemas: Sequence[Any]) -> None:
        self.schemas = schemas
        self.sent: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.sent.append((query, args))
        rows = []
        for schema in self.schemas:
            [comment] = [each for each in schema.create_statements(yugabyte=False) if each.startswith("COMMENT ON")]
            declared = comment.split(" IS '", 1)[1][:-1].replace("''", "'")
            rows.append(
                {
                    "table_schema": "platform",
                    "table_name": _bare(schema.name),
                    "declared": declared,
                    "run_markers": False,
                }
            )
        return rows

    async def copy_from_query(self, query: str, *args: Any, output: Any, **_: Any) -> str:
        self.sent.append((query, args))
        return "COPY 0"

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[_Catalog]:
        yield self

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        yield


async def assert_restore_reads_every_bucket(schemas: Sequence[Any]) -> None:
    """a restore of a lost change log finds each of ``schemas`` from the catalog's comments alone and reads each one's
    16 buckets since X, one keyed statement a bucket binding ``(bucket, X)``: one range of its ``(bucket, epoch)``
    index. A table whose rows a write may move (``TableSchema.moveable``) is read a second time a bucket, within the
    same range, for the rows moved since X (:mod:`~threetears.core.ddl.tenant_moves`).

    :param schemas: the stamped tables' declarations
    :ptype schemas: Sequence[TableSchema]
    :return: nothing
    :rtype: None
    """
    pool = _Catalog(schemas)
    tables = await restored_tables(pool)
    assert sorted(table.table for table in tables) == sorted(_bare(schema.name) for schema in schemas)
    assert all(isinstance(table, StampedTable) and table.buckets == RESTORE_BUCKETS for table in tables)
    floor = next_epoch()
    changes = TableChanges(pool, tables, floor, MagicMock(), None)
    assert [change async for change in changes] == []
    moving = [table for table in tables if isinstance(table, StampedTable) and table.moved_columns]
    assert changes.buckets == RESTORE_BUCKETS * (len(tables) + len(moving))
    for table in tables:
        reads = [(query, args) for query, args in pool.sent if f'."{table.table}" AS t' in query]
        moved = [
            epoch_column for column in getattr(table, "moved_columns", ()) for epoch_column in moved_columns(column)
        ]
        moves = [(query, args) for query, args in reads if any(f'"{name}"' in query for name in moved)]
        plain = [(query, args) for query, args in reads if (query, args) not in moves]
        assert sorted(args for _, args in plain) == [(bucket, floor) for bucket in range(RESTORE_BUCKETS)], table.table
        # a moves read a bucket for a moveable table, binding X again for each moved column's epoch; none for another
        expected = (
            [(bucket, floor, *(floor,) * (len(moved) // 2)) for bucket in range(RESTORE_BUCKETS)] if moved else []
        )
        assert sorted(args for _, args in moves) == expected, table.table
        for query, _ in reads:
            assert f't."{BUCKET_COLUMN}" = $1' in query and f't."{EPOCH_COLUMN}" >= $2' in query, query
