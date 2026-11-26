"""Integration: ``delete(entity_id, conn=...)`` on real Postgres joins the caller's transaction.

A stamped parent's delete on a caller's ``CallerTransaction`` marks its cascaded children gone, then itself, on the
caller's connection: a rollback leaves every row live and announces nothing; a commit leaves them gone, every child's
epoch at most its parent's, and announces the row and its children once. A ``NO ACTION`` foreign key from the child
to the parent is real, so a parent marked before a child it still names is the database's own state to read.

Uses the session-scoped ``db_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any, ClassVar
from unittest.mock import AsyncMock

import asyncpg
import pytest
from sqlalchemy import MetaData

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.collections.base import NO_L2
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.delete_cascade import DeleteCascade
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import STRING_TYPE, Column, SchemaBackedCollection, TableSchema
from threetears.core.config import DefaultCoreConfig
from threetears.core.ddl.row_epoch import ROW_STAMPED
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.l1_tables import TEST_L1_TABLE

pytestmark = pytest.mark.integration

_STAMPED = "epoch BIGINT NOT NULL, gone BOOLEAN NOT NULL DEFAULT false, epoch_bucket SMALLINT NOT NULL"

_DDL = f"""
CREATE TABLE djl_parent (id TEXT PRIMARY KEY, label TEXT NOT NULL, {_STAMPED});
CREATE TABLE djl_child (id TEXT PRIMARY KEY, parent_id TEXT REFERENCES djl_parent (id), {_STAMPED});
CREATE INDEX djl_child_parent ON djl_child (parent_id);
INSERT INTO djl_parent VALUES ('p1', 'one', 1, false, 2);
INSERT INTO djl_child VALUES ('c1', 'p1', 1, false, 5), ('c2', 'p1', 1, false, 11);
"""


class _Row(BaseEntity):
    primary_key_field = "id"


class _Children(SchemaBackedCollection[_Row]):
    primary_key_column: str = "id"
    schema = TableSchema(
        name="djl_child",
        primary_key="id",
        columns=[Column("id", STRING_TYPE), Column("parent_id", STRING_TYPE)],
        row_epoch=ROW_STAMPED,
        l1_eviction=TEST_L1_TABLE,
    )

    @property
    def table_name(self) -> str:
        return "djl_child"

    @property
    def entity_class(self) -> type[_Row]:
        return _Row


class _Parents(SchemaBackedCollection[_Row]):
    primary_key_column: str = "id"
    schema = TableSchema(
        name="djl_parent",
        primary_key="id",
        columns=[Column("id", STRING_TYPE), Column("label", STRING_TYPE)],
        row_epoch=ROW_STAMPED,
        l1_eviction=TEST_L1_TABLE,
    )
    delete_cascades: ClassVar[tuple[DeleteCascade, ...]] = (DeleteCascade("djl_child", "parent_id", "CASCADE"),)

    @property
    def table_name(self) -> str:
        return "djl_parent"

    @property
    def entity_class(self) -> type[_Row]:
        return _Row


class _Abandoned(Exception):
    """the caller's later write failing, after the delete."""


@pytest.fixture
async def pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    """a pool on a fresh schema holding a stamped parent ``p1`` and its two stamped children.

    :param db_container: the Postgres container's connection URL
    :ptype db_container: str
    :return: the pool
    :rtype: AsyncIterator[asyncpg.Pool]
    """
    own = f"djl_{uuid.uuid4().hex[:8]}"
    setup = await asyncpg.connect(db_container)
    await setup.execute(f"CREATE SCHEMA {own}")
    await setup.execute(f"SET search_path TO {own}")
    await setup.execute(_DDL)
    created = await asyncpg.create_pool(
        db_container, min_size=1, max_size=4, server_settings={"search_path": own}, init=init_connection
    )
    assert created is not None
    try:
        yield created
    finally:
        await created.close()
        await setup.execute(f"DROP SCHEMA {own} CASCADE")
        await setup.close()


def _family(pool: asyncpg.Pool) -> tuple[Any, Any]:
    l1 = SQLiteBackend(db_name=f"djl_{uuid.uuid4().hex[:8]}")
    metadata = MetaData()
    _Parents.schema.to_sqlalchemy_table(metadata)
    _Children.schema.to_sqlalchemy_table(metadata)
    l1.initialize(metadata)
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l3_pool=pool)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    children: Any = _Children(registry=registry, config=config, nats_client=NO_L2)
    parents: Any = _Parents(registry=registry, config=config, nats_client=NO_L2)
    # each announcement watched, and still made
    children.invalidate_cache_many = AsyncMock(wraps=children.invalidate_cache_many)
    parents.invalidate_cache_many = AsyncMock(wraps=parents.invalidate_cache_many)
    return parents, children


async def _epochs(pool: asyncpg.Pool, table: str) -> dict[str, tuple[int, bool]]:
    rows = await pool.fetch(f"SELECT id, epoch, gone FROM {table}")  # noqa: S608 -- a test's literal table
    return {row["id"]: (int(row["epoch"]), bool(row["gone"])) for row in rows}


async def test_a_delete_rolled_back_with_its_caller_leaves_every_row_live_and_announces_nothing(
    pool: asyncpg.Pool,
) -> None:
    parents, children = _family(pool)
    with pytest.raises(_Abandoned):
        async with pool.acquire() as conn, CallerTransaction(conn):
            await parents.delete("p1", conn=conn)
            # inside the transaction the marks are written: the caller's connection sees them
            assert (await conn.fetchval("SELECT gone FROM djl_parent WHERE id = 'p1'")) is True
            raise _Abandoned
    assert await _epochs(pool, "djl_parent") == {"p1": (1, False)}
    assert await _epochs(pool, "djl_child") == {"c1": (1, False), "c2": (1, False)}
    parents.invalidate_cache_many.assert_not_awaited()
    children.invalidate_cache_many.assert_not_awaited()


async def test_a_delete_committed_with_its_caller_marks_children_then_parent_and_announces_once(
    pool: asyncpg.Pool,
) -> None:
    parents, children = _family(pool)
    async with pool.acquire() as conn, CallerTransaction(conn):
        await parents.delete("p1", conn=conn)
        # not yet committed: another connection still reads every row live, and nothing is announced
        assert await _epochs(pool, "djl_parent") == {"p1": (1, False)}
        parents.invalidate_cache_many.assert_not_awaited()
    parent, kids = await _epochs(pool, "djl_parent"), await _epochs(pool, "djl_child")
    assert parent["p1"][1] and kids["c1"][1] and kids["c2"][1]
    assert max(kids["c1"][0], kids["c2"][0]) <= parent["p1"][0], (parent, kids)
    parents.invalidate_cache_many.assert_awaited_once_with(["p1"], rows=[None])
    children.invalidate_cache_many.assert_awaited_once()
    assert sorted(children.invalidate_cache_many.await_args.args[0]) == ["c1", "c2"]


async def test_a_delete_finds_a_parent_and_child_written_earlier_in_its_transaction(pool: asyncpg.Pool) -> None:
    parents, children = _family(pool)
    async with pool.acquire() as conn, CallerTransaction(conn):
        await conn.execute("INSERT INTO djl_parent VALUES ('p9', 'nine', 1, false, 2)")
        await conn.execute("INSERT INTO djl_child VALUES ('c9', 'p9', 1, false, 3)")
        await parents.delete("p9", conn=conn)
    parent, kids = await _epochs(pool, "djl_parent"), await _epochs(pool, "djl_child")
    assert parent["p9"][1] and kids["c9"][1], (parent, kids)
    assert kids["c9"][0] <= parent["p9"][0]
    parents.invalidate_cache_many.assert_awaited_once_with(["p9"], rows=[None])
    children.invalidate_cache_many.assert_awaited_once()
    assert list(children.invalidate_cache_many.await_args.args[0]) == ["c9"]
