"""a stamped parent's delete marks its cascaded children gone before itself, in one transaction, so every child's
epoch is at most its parent's; quiet-time cleanup then removes parent and children below the floor with no foreign-key
error, and leaves (counts, never raises on) a gone parent a row still references.

Append-only L3 (decisions 2 and 5). A stamped parent's delete is a gone mark, so 3tears takes each declared
``ON DELETE CASCADE`` itself (``delete_cascade.apply_delete_cascade``). Marked parent first, each child carried a newer
stamp than its parent: the floor could pass the parent while a child was still above it, and under a ``NO ACTION``
foreign key cleanup's ``DELETE`` of the parent failed on the child and ``clear_stamped_gone`` raised.

The tables run on an in-process DuckDB, a real ``NO ACTION`` foreign key from ``sp_child.parent_id`` to ``sp_parent``,
behind a pool that names no dialect (so the helpers take it for L3). DuckDB runs each statement on its own cursor, so
the pool marks where the delete's transaction opens and commits rather than holding one; the epoch order and the
statements' order are the database's own.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, ClassVar, cast
from unittest.mock import AsyncMock, MagicMock

import duckdb
import pytest

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.base import NO_L2
from threetears.core.collections.delete_cascade import DeleteCascade
from threetears.core.collections.quiet_cleanup import LoadGate, QuietCleanup
from threetears.core.collections.schema_backed import STRING_TYPE, Column, SchemaBackedCollection, TableSchema
from threetears.core.collections.scope_epochs import WHOLE, EpochSnapshot, ScopeEpochs
from threetears.core.collections.stamped_cleanup import clear_stamped_gone
from threetears.core.collections.cleanup_progress import CLEANUP_PROGRESS_TABLE
from threetears.core.ddl.row_epoch import ROW_STAMPED
from threetears.core.entities.base import BaseEntity
from threetears.core.keyed import ThreadedDuckDB
from threetears.core.testing.l1_tables import TEST_L1_TABLE

_PARENT = TableSchema(
    name="sp_parent",
    primary_key="id",
    columns=[Column("id", STRING_TYPE), Column("label", STRING_TYPE)],
    row_epoch=ROW_STAMPED,
    l1_eviction=TEST_L1_TABLE,
)

_CHILD = TableSchema(
    name="sp_child",
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
        return "sp_child"

    @property
    def entity_class(self) -> type[_Row]:
        return _Row


class _Parents(SchemaBackedCollection[_Row]):
    primary_key_column: str = "id"
    schema = _PARENT
    delete_cascades: ClassVar[tuple[DeleteCascade, ...]] = (DeleteCascade("sp_child", "parent_id", "CASCADE"),)

    @property
    def table_name(self) -> str:
        return "sp_parent"

    @property
    def entity_class(self) -> type[_Row]:
        return _Row


class _Held:
    """an ``async with`` answering the pool, marking where it opens and closes in the pool's statements."""

    def __init__(self, pool: _Pool, opened: str | None, closed: str | None) -> None:
        self._pool, self._opened, self._closed = pool, opened, closed

    async def __aenter__(self) -> _Pool:
        if self._opened is not None:
            self._pool.statements.append(self._opened)
        return self._pool

    async def __aexit__(self, exc_type: object, *exc: object) -> None:
        if self._closed is not None:
            self._pool.statements.append(self._closed if exc_type is None else "ROLLBACK")


class _Pool(ThreadedDuckDB):
    """the DuckDB the tables live in, as an asyncpg-shaped pool naming no dialect; it keeps every statement."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        super().__init__(conn)
        self.statements: list[str] = []

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
        return _Held(self, None, None)

    def transaction(self, **_options: Any) -> _Held:
        return _Held(self, "BEGIN", "COMMIT")


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    """a parent ``p1`` with two children, a parent ``p2`` with none; every row live, stamped 1."""
    conn = duckdb.connect()
    conn.execute(f"CREATE TABLE sp_parent (id VARCHAR PRIMARY KEY, label VARCHAR NOT NULL, {_STAMPED})")
    conn.execute(
        f"CREATE TABLE sp_child (id VARCHAR PRIMARY KEY, parent_id VARCHAR REFERENCES sp_parent (id), {_STAMPED})"
    )
    conn.execute(
        f"CREATE TABLE {CLEANUP_PROGRESS_TABLE} (table_name VARCHAR, scope VARCHAR, cleaned_to BIGINT, "
        "PRIMARY KEY (table_name, scope))"
    )
    conn.execute("INSERT INTO sp_parent VALUES ('p1', 'one', 1, false, 2), ('p2', 'two', 1, false, 2)")
    conn.execute("INSERT INTO sp_child VALUES ('c1', 'p1', 1, false, 5), ('c2', 'p1', 1, false, 11)")
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
    return collection


def _parents(pool: _Pool) -> Any:
    children = _collection(_Children, pool)
    children.invalidate_cache_many = AsyncMock()
    return _collection(_Parents, pool, {"sp_child": children})


def _epochs(conn: duckdb.DuckDBPyConnection, table: str) -> dict[str, tuple[int, bool]]:
    return {
        str(key): (int(epoch), bool(gone))
        for key, epoch, gone in conn.execute(f"SELECT id, epoch, gone FROM {table}").fetchall()
    }


class _Quiet:
    """scope epochs with nothing loading: the gate cleanup asks."""

    snap = EpochSnapshot(version=1, writing=None, epochs={WHOLE: 1}, previous={})

    def on_change(self, listener: Callable[[], None]) -> Callable[[], None]:
        return lambda: None

    async def snapshot(self) -> EpochSnapshot:
        return self.snap


def _cleanup(pool: _Pool, floor: int) -> QuietCleanup:
    async def window() -> int | None:
        return floor

    # the parent's table first: a bucket at a time, cleanup reaches the parent before its children are removed
    return QuietCleanup(pool, (_PARENT, _CHILD), LoadGate(cast(ScopeEpochs, _Quiet())), window, run_markers=None)


async def test_a_parents_delete_marks_its_children_gone_first_in_one_transaction(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if the parent is marked before its children (a child's epoch above its parent's), or the marks are not one
    transaction."""
    pool = _Pool(db)
    await _parents(pool).delete("p1")
    parent, children = _epochs(db, "sp_parent"), _epochs(db, "sp_child")
    assert parent["p1"][1] and children["c1"][1] and children["c2"][1]
    assert max(children["c1"][0], children["c2"][0]) <= parent["p1"][0], (parent, children)
    marks = [sql for sql in pool.statements if sql.startswith(("BEGIN", "COMMIT", "UPDATE"))]
    assert [sql.split('"', 2)[1] if sql.startswith("UPDATE") else sql for sql in marks] == [
        "BEGIN",
        "sp_child",
        "sp_parent",
        "COMMIT",
    ], marks


async def test_cleanup_removes_a_parent_and_its_cascaded_children_with_no_foreign_key_error(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Flips if the floor can pass a gone parent while its gone child is still above it (cleanup's DELETE of the
    parent then fails on the child's foreign key and raises), or if cleanup raises on, or records past, a parent its
    child still names."""
    pool = _Pool(db)
    await _parents(pool).delete("p1")
    floor = _epochs(db, "sp_parent")["p1"][0] + 1
    first = await _cleanup(pool, floor).clean()
    # the parent's table came first: its row still named by a gone child below the floor is left, never raised on, and
    # the pass is not recorded finished so the next one comes back to it
    assert not first.finished and first.removed == 2
    assert set(_epochs(db, "sp_child")) == set()
    second = await _cleanup(pool, floor).clean()
    assert second.finished and second.removed == 1
    assert set(_epochs(db, "sp_parent")) == {"p2"}


async def test_a_gone_parent_a_live_child_still_names_is_left_and_counted(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if cleanup raises on a gone parent a live row still references, removes nothing else of its page, or
    calls the sweep finished."""
    db.execute("UPDATE sp_parent SET gone = true, epoch = 50")
    db.execute("UPDATE sp_child SET parent_id = 'p1'")
    swept = await clear_stamped_gone(_Pool(db), _PARENT, since=0, floor=100)
    assert (swept.removed, swept.referenced, swept.finished) == (1, 1, False)
    assert set(_epochs(db, "sp_parent")) == {"p1"}
