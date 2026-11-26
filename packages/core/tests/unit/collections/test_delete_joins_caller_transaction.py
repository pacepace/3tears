"""``delete(entity_id, conn=...)`` joins the caller's :class:`CallerTransaction`: its writes commit or roll back with
the caller's, and its announcement waits for the commit (``CallerTransaction.after_commit``).

A stamped parent's delete on a caller's transaction refuses a declared ``NO ACTION`` child before anything is written,
marks its cascaded children gone before itself on the caller's connection, and announces the row and the children it
reached once the COMMIT landed (or went unanswered), never after a rollback. Before this a delete always opened its own
transaction, so a caller whose later write failed could not undo a delete already committed (the hub's datasource
create that replaced a leftover namespace and then failed left that namespace's grants removed), and hand-built one.

The tables run on an in-process DuckDB with a ``NO ACTION`` foreign key from ``dj_child.parent_id`` to ``dj_parent``.
Every statement runs on the one DuckDB connection (not a cursor each), so the caller's ``BEGIN`` / ``COMMIT`` /
``ROLLBACK`` is a real transaction the delete's statements are inside.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock

import duckdb
import pytest
from asyncpg.exceptions import SerializationError

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections import CallerTransaction
from threetears.core.collections.base import NO_L2
from threetears.core.collections.delete_cascade import DeleteCascade, DeleteRestrict, RestrictedDeleteError
from threetears.core.collections.schema_backed import STRING_TYPE, Column, SchemaBackedCollection, TableSchema
from threetears.core.ddl.row_epoch import ROW_STAMPED
from threetears.core.entities.base import BaseEntity
from threetears.core.keyed import ThreadedDuckDB
from threetears.core.testing.l1_tables import TEST_L1_TABLE

_PARENT = TableSchema(
    name="dj_parent",
    primary_key="id",
    columns=[Column("id", STRING_TYPE), Column("label", STRING_TYPE)],
    row_epoch=ROW_STAMPED,
    l1_eviction=TEST_L1_TABLE,
)

_CHILD = TableSchema(
    name="dj_child",
    primary_key="id",
    columns=[Column("id", STRING_TYPE), Column("parent_id", STRING_TYPE)],
    row_epoch=ROW_STAMPED,
    l1_eviction=TEST_L1_TABLE,
)

_STAMPED = "epoch BIGINT NOT NULL, gone BOOLEAN NOT NULL DEFAULT false, epoch_bucket SMALLINT NOT NULL"


class _Row(BaseEntity):
    primary_key_field = "id"


class _Children(SchemaBackedCollection[_Row]):
    primary_key_column: str = "id"
    schema = _CHILD

    @property
    def table_name(self) -> str:
        return "dj_child"

    @property
    def entity_class(self) -> type[_Row]:
        return _Row


class _Parents(SchemaBackedCollection[_Row]):
    primary_key_column: str = "id"
    schema = _PARENT
    delete_cascades: ClassVar[tuple[DeleteCascade, ...]] = (DeleteCascade("dj_child", "parent_id", "CASCADE"),)

    @property
    def table_name(self) -> str:
        return "dj_parent"

    @property
    def entity_class(self) -> type[_Row]:
        return _Row


class _RestrictedParents(_Parents):
    delete_cascades: ClassVar[tuple[DeleteCascade, ...]] = ()
    delete_restricts: ClassVar[tuple[DeleteRestrict, ...]] = (DeleteRestrict("dj_child", "parent_id", "NO ACTION"),)


class _OneConnection:
    """the DuckDB connection itself as every statement's cursor, so the caller's transaction spans them all."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

    def cursor(self) -> _OneConnection:
        return self

    def close(self) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


class _Transaction:
    """a real DuckDB transaction on the one connection; ``commit_raises`` fails its COMMIT after rolling back."""

    def __init__(self, pool: _Pool) -> None:
        self._pool = pool

    async def __aenter__(self) -> _Transaction:
        await self._pool.execute("BEGIN TRANSACTION")
        return self

    async def __aexit__(self, exc_type: object, *exc: object) -> bool:
        if exc_type is not None or self._pool.commit_raises is not None:
            await self._pool.execute("ROLLBACK")
            if exc_type is None and self._pool.commit_raises is not None:
                raise self._pool.commit_raises
            return False
        await self._pool.execute("COMMIT")
        return False


class _Held:
    def __init__(self, pool: _Pool) -> None:
        self._pool = pool

    async def __aenter__(self) -> _Pool:
        return self._pool

    async def __aexit__(self, *exc: object) -> None:
        return None


class _Pool(ThreadedDuckDB):
    """the DuckDB as an asyncpg-shaped pool and connection naming no dialect; it keeps every statement."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        super().__init__(_OneConnection(conn))
        self.statements: list[str] = []
        self.commit_raises: BaseException | None = None

    async def answers_from_yugabyte(self) -> bool:
        return False

    async def fetch(self, query: str, /, *params: Any) -> list[dict[str, Any]]:
        self.statements.append(" ".join(query.split()))
        return await super().fetch(query, *params)

    async def fetch_arrow(self, query: str, /, *params: Any, types: Any, size: int | None = None) -> Any:
        self.statements.append(" ".join(query.split()))
        return await super().fetch_arrow(query, *params, types=types, size=size)

    async def execute(self, query: str, /, *params: Any) -> str:
        self.statements.append(" ".join(query.split()))
        return await super().execute(query, *params)

    def acquire(self) -> _Held:
        return _Held(self)

    def transaction(self, **_options: Any) -> _Transaction:
        return _Transaction(self)


class _Abandoned(Exception):
    """the caller's later write failing, after the delete."""


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    """a parent ``p1`` with two children; every row live, stamped 1."""
    conn = duckdb.connect()
    conn.execute(f"CREATE TABLE dj_parent (id VARCHAR PRIMARY KEY, label VARCHAR NOT NULL, {_STAMPED})")
    conn.execute(
        f"CREATE TABLE dj_child (id VARCHAR PRIMARY KEY, parent_id VARCHAR REFERENCES dj_parent (id), {_STAMPED})"
    )
    conn.execute("INSERT INTO dj_parent VALUES ('p1', 'one', 1, false, 2)")
    conn.execute("INSERT INTO dj_child VALUES ('c1', 'p1', 1, false, 5), ('c2', 'p1', 1, false, 11)")
    yield conn
    conn.close()


def _collection(cls: type[Any], pool: _Pool, siblings: dict[str, Any] | None = None) -> Any:
    registry = MagicMock()
    registry.get_l1_backend.return_value = None
    registry.get_l3_pool.return_value = pool
    registry.register.return_value = None
    registry.generation_source = None
    registry.publish_invalidation = AsyncMock(return_value=None)
    registry.get_collection.side_effect = (siblings or {}).get
    config = MagicMock()
    config.collection_flush = "ALWAYS"
    config.collection_flush_tables = ""
    collection = cls(registry=registry, config=config, nats_client=NO_L2)
    store = SqlL3Backend(pool)
    store.register_schema(collection.schema.name, collection.schema)
    collection.l3_pool = store
    # the announcement each collection makes, watched: the row's own (the parent) and the cascade's (the children)
    collection.invalidate_cache_many = AsyncMock()
    return collection


def _family(pool: _Pool, parents: type[Any] = _Parents) -> tuple[Any, Any]:
    children = _collection(_Children, pool)
    return _collection(parents, pool, {"dj_child": children}), children


def _epochs(conn: duckdb.DuckDBPyConnection, table: str) -> dict[str, tuple[int, bool]]:
    return {
        str(key): (int(epoch), bool(gone))
        for key, epoch, gone in conn.execute(f"SELECT id, epoch, gone FROM {table}").fetchall()
    }


def _marks(pool: _Pool) -> list[str]:
    """the transaction's edges and each write's table, in order."""
    return [
        sql.split('"', 2)[1] if sql.startswith("UPDATE") else sql
        for sql in pool.statements
        if sql.startswith(("BEGIN", "COMMIT", "ROLLBACK", "UPDATE"))
    ]


async def test_a_delete_in_a_transaction_rolled_back_leaves_every_row_live_and_announces_nothing(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Flips if delete(conn=) opens a transaction of its own (its marks commit before the caller's rollback), or
    announces the row or its children before the caller's transaction commits."""
    pool = _Pool(db)
    parents, children = _family(pool)
    with pytest.raises(_Abandoned):
        async with CallerTransaction(pool):
            await parents.delete("p1", conn=pool)
            raise _Abandoned
    # the marks were written, inside the caller's transaction, and rolled back with it
    assert _marks(pool) == ["BEGIN TRANSACTION", "dj_child", "dj_parent", "ROLLBACK"], pool.statements
    assert _epochs(db, "dj_parent") == {"p1": (1, False)}
    assert _epochs(db, "dj_child") == {"c1": (1, False), "c2": (1, False)}
    parents.invalidate_cache_many.assert_not_awaited()
    children.invalidate_cache_many.assert_not_awaited()


async def test_a_delete_in_a_transaction_committed_marks_children_then_parent_and_announces_once(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Flips if the parent is marked before its children (a child's epoch above its parent's), the marks leave the
    caller's transaction, or the row or its children are announced other than once, after the commit."""
    pool = _Pool(db)
    parents, children = _family(pool)
    async with CallerTransaction(pool):
        await parents.delete("p1", conn=pool)
        # nothing announced while the transaction is open
        parents.invalidate_cache_many.assert_not_awaited()
        children.invalidate_cache_many.assert_not_awaited()
    assert _marks(pool) == ["BEGIN TRANSACTION", "dj_child", "dj_parent", "COMMIT"], pool.statements
    parent, kids = _epochs(db, "dj_parent"), _epochs(db, "dj_child")
    assert parent["p1"][1] and kids["c1"][1] and kids["c2"][1]
    assert max(kids["c1"][0], kids["c2"][0]) <= parent["p1"][0], (parent, kids)
    parents.invalidate_cache_many.assert_awaited_once_with(["p1"], rows=[None])
    children.invalidate_cache_many.assert_awaited_once()
    assert sorted(children.invalidate_cache_many.await_args.args[0]) == ["c1", "c2"]


async def test_a_restricted_delete_is_refused_before_any_write(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if a delete on a caller's transaction marks a row gone that a live ``NO ACTION`` child still names, or
    writes anything before refusing."""
    pool = _Pool(db)
    parents, children = _family(pool, _RestrictedParents)
    with pytest.raises(RestrictedDeleteError):
        async with CallerTransaction(pool):
            await parents.delete("p1", conn=pool)
    assert _marks(pool) == ["BEGIN TRANSACTION", "ROLLBACK"], pool.statements
    assert _epochs(db, "dj_parent") == {"p1": (1, False)}
    parents.invalidate_cache_many.assert_not_awaited()
    children.invalidate_cache_many.assert_not_awaited()


async def test_a_delete_needs_a_caller_transaction_on_its_connection(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if delete(conn=) writes on a connection no CallerTransaction opened (nothing would settle it)."""
    pool = _Pool(db)
    parents, _ = _family(pool)
    with pytest.raises(ValueError, match="CallerTransaction"):
        await parents.delete("p1", conn=pool)
    assert _marks(pool) == [], pool.statements


async def test_a_commit_never_answered_is_announced_and_one_refused_is_not(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if a COMMIT whose reply never came (it may have landed) goes unannounced, or one the database refused with
    its own error is announced (the unknown-outcome rule, at the caller's commit)."""
    pool = _Pool(db)
    parents, children = _family(pool)
    pool.commit_raises = SerializationError("could not serialize access")
    with pytest.raises(SerializationError):
        async with CallerTransaction(pool):
            await parents.delete("p1", conn=pool)
    parents.invalidate_cache_many.assert_not_awaited()
    children.invalidate_cache_many.assert_not_awaited()
    pool.commit_raises = ConnectionResetError("the reply never came")
    with pytest.raises(ConnectionResetError):
        async with CallerTransaction(pool):
            await parents.delete("p1", conn=pool)
    parents.invalidate_cache_many.assert_awaited_once_with(["p1"], rows=[None])
    children.invalidate_cache_many.assert_awaited_once()


class _Savepointing:
    """a connection whose ``transaction()`` nests, ending each cleanly or by rolling back as asyncpg's would."""

    def transaction(self, **_options: Any) -> _Savepointing:
        return self

    async def __aenter__(self) -> _Savepointing:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


async def test_a_savepoint_rolled_back_drops_its_steps_and_one_released_hands_them_on() -> None:
    """Flips if a step owed by a savepoint rolled back runs at the outer commit, or one released is lost."""
    conn = _Savepointing()
    ran: list[str] = []

    async def step(name: str) -> None:
        ran.append(name)

    async with CallerTransaction(conn) as outer:
        with pytest.raises(_Abandoned):
            async with CallerTransaction(conn):
                CallerTransaction.join(conn, writer="test").after_commit(lambda: step("rolled back"))
                raise _Abandoned
        async with CallerTransaction(conn):
            CallerTransaction.join(conn, writer="test").after_commit(lambda: step("released"))
        outer.after_commit(lambda: step("outer"))
        assert ran == []
    assert ran == ["released", "outer"]
