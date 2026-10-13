"""a delete announces the rows its foreign-key actions reach, which the database rewrites where no collection sees.

Nothing in L1 ages, so a row a cascade deleted, or a ``SET NULL`` rewrote, would be served from a
cache until written again unless the delete that caused it names it.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, ClassVar

from uuid import uuid4

import pytest
from sqlalchemy import Column, MetaData, String, Table

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import WRITE_GENERATION, DeleteCascade
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.delete_cascade import (
    CascadedRows,
    announce_delete_cascade,
    read_delete_cascade,
)
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity


class _Entity(BaseEntity):
    """a row."""


class _Pool:
    """an L3 holding each table's rows; answers the cascade's ``SELECT pk FROM t WHERE col = ANY($1)``."""

    def __init__(self, tables: dict[str, list[dict[str, Any]]]) -> None:
        self.tables = tables
        self.reads: list[str] = []

    async def fetch(self, sql: str, values: list[Any]) -> list[dict[str, Any]]:
        self.reads.append(sql)
        match = re.match(r"SELECT (.+) FROM (\w+) WHERE (\w+) = ANY\(\$1\)", sql)
        assert match is not None, sql
        columns = [c.strip() for c in match.group(1).split(",")]
        return [{c: row[c] for c in columns} for row in self.tables[match.group(2)] if row[match.group(3)] in values]


class _Collection(BaseCollection[_Entity]):
    """one table of the pool, switched on."""

    write_generation = WRITE_GENERATION
    table: ClassVar[str] = ""

    @property
    def table_name(self) -> str:
        return self.table

    @property
    def entity_class(self) -> type[_Entity]:
        return _Entity

    async def fetch_from_store(self, entity_id: object) -> dict | None:
        return next((r for r in self.l3_pool.tables[self.table] if r["id"] == entity_id), None)  # type: ignore[union-attr]

    async def save_to_store(self, data: dict, original_timestamp: datetime | None = None) -> int:
        return 1

    async def delete_from_store(self, entity_id: object) -> None:
        pool = self.l3_pool
        rows = pool.tables[self.table]  # type: ignore[union-attr]
        rows[:] = [r for r in rows if r["id"] != entity_id]
        # the database's actions, which no collection sees
        for cascade in type(self).delete_cascades:
            children = pool.tables[cascade.child_table]  # type: ignore[union-attr]
            if cascade.action == "CASCADE":
                children[:] = [r for r in children if r[cascade.child_column] != entity_id]
            else:
                for r in children:
                    if r[cascade.child_column] == entity_id:
                        r[cascade.child_column] = None

    def serialize(self, data: dict) -> bytes:
        return json.dumps(data, default=str).encode()

    def deserialize(self, data: bytes) -> dict:
        return json.loads(data)


class _Shelves(_Collection):
    table = "shelves"
    delete_cascades = (DeleteCascade("books", "shelf_id", "CASCADE"),)


class _Books(_Collection):
    table = "books"
    delete_cascades = (DeleteCascade("books", "sequel_of", "SET NULL"),)


class _Source:
    """counts advances per table."""

    def __init__(self) -> None:
        self.advanced: list[str] = []

    async def current(self, table_name: str) -> str:
        return f"inc:{self.advanced.count(table_name)}"

    async def advance(self, table_name: str) -> str:
        self.advanced.append(table_name)
        return f"inc:{self.advanced.count(table_name)}"


def _library() -> tuple[CollectionRegistry, _Pool, _Source]:
    pool = _Pool(
        {
            "shelves": [{"id": "s1"}, {"id": "s2"}],
            "books": [
                {"id": "b1", "shelf_id": "s1", "sequel_of": None},
                {"id": "b2", "shelf_id": "s1", "sequel_of": "b1"},
                {"id": "b3", "shelf_id": "s2", "sequel_of": "b2"},
            ],
        }
    )
    metadata = MetaData()
    Table("shelves", metadata, Column("id", String(64), primary_key=True))
    Table(
        "books",
        metadata,
        Column("id", String(64), primary_key=True),
        Column("shelf_id", String(64)),
        Column("sequel_of", String(64)),
    )
    l1 = SQLiteBackend(db_name=f"cascade-{uuid4().hex}")
    l1.initialize(metadata)
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l3_pool=pool, kv_key_scope="test")  # type: ignore[arg-type]
    source = _Source()
    registry.set_generation_source(source)  # type: ignore[arg-type]
    return registry, pool, source


async def test_a_delete_announces_what_its_cascade_removes_and_what_that_unlinks() -> None:
    registry, _pool, source = _library()
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    shelves = _Shelves(registry, config)
    books = _Books(registry, config)
    for row in ({"id": "b1"}, {"id": "b2"}, {"id": "b3"}):
        books.write_to_cache_sync({**row, "shelf_id": "x", "sequel_of": None})

    await shelves.delete("s1")

    # b1 and b2 went with the shelf; b3 lost its link to b2: all three were cached, none is now
    assert [books.get_row_sync(key) for key in ("b1", "b2", "b3")] == [None, None, None]
    assert source.advanced.count("books") == 2  # the cascade, then the unlinking it caused
    assert source.advanced.count("shelves") == 1


async def test_the_rows_are_read_before_the_delete_reaches_them() -> None:
    registry, pool, _source = _library()
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    shelves = _Shelves(registry, config)
    _Books(registry, config)

    cascaded = await read_delete_cascade(shelves, ["s1"])

    # the books the cascade removes, then the books naming any of them as their original -- b2
    # among them, since the database runs both actions; announcing it twice costs nothing
    assert [(rows.table_name, rows.keys) for rows in cascaded] == [("books", ("b1", "b2")), ("books", ("b2", "b3"))]
    assert cascaded[0].rows == ({"id": "b1"}, {"id": "b2"})
    assert len(pool.reads) == 2


async def test_a_table_with_no_collection_here_is_advanced_with_no_rows() -> None:
    """its followers find an advance they did not hear and drop the table: the reach is unknown."""
    registry, _pool, source = _library()
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    shelves = _Shelves(registry, config)  # no books collection registered

    cascaded = await read_delete_cascade(shelves, ["s1"])
    assert cascaded == [CascadedRows("books", (), unheard=True)]
    await announce_delete_cascade(shelves, cascaded)

    assert source.advanced == ["books"]


async def test_nothing_is_read_for_a_collection_that_declares_nothing() -> None:
    registry, pool, _source = _library()
    books = _Books(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""))
    type(books).delete_cascades = ()
    try:
        assert await read_delete_cascade(books, ["b1"]) == []
    finally:
        type(books).delete_cascades = (DeleteCascade("books", "sequel_of", "SET NULL"),)
    assert pool.reads == []


@pytest.mark.parametrize("action", ["CASCADE", "SET NULL"])
def test_a_cascade_names_its_action(action: str) -> None:
    assert DeleteCascade("books", "shelf_id", action).action == action  # type: ignore[arg-type]
