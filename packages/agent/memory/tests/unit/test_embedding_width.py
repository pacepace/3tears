"""agent-memory's embedding width is a deployment setting, 1024 unless configured.

What needs no database: the width a registration, a collection and a table factory take, the
ledger names the migrations keep whatever the width, the DDL rendered at the width, PostgreSQL's
ceiling refused before any DDL, and a write of another width refused before any statement. The
catalogue side (verify, resize, a real search at 1536) is the integration suite's.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy import MetaData

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import (
    MediaContentCollection,
    MemoriesCollection,
    MemoryChunkCollection,
    media_content_table,
    memories_table,
    memory_chunks_table,
)
from threetears.agent.memory.migrations import register
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

_POSTGRES_VERSION = "PostgreSQL 16.4 (Debian 16.4-1.pgdg120+2) on x86_64-pc-linux-gnu"
_YUGABYTE_VERSION = "PostgreSQL 15.12-YB-2025.2.0.0-b0 on x86_64-pc-linux-gnu"

#: the migrations that create an embedding column, and the table each creates it on
_CREATING_VERSIONS = {1: "memories", 6: "media_content", 7: "memory_chunks"}


# parity-with: threetears.core.data.embedding_width.EmbeddingWidthStore
class _FakeStore:
    """records every statement; answers ``SELECT version()`` as the backend it is told to be."""

    def __init__(self, version: str = _POSTGRES_VERSION) -> None:
        """start with no statements.

        :param version: what ``SELECT version()`` answers
        :ptype version: str
        """
        self.executed: list[str] = []
        self.queried: list[str] = []
        self._version = version

    async def execute(self, sql: str, *params: Any) -> str:
        """record ``sql``.

        :param sql: statement text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: a status tag
        :rtype: str
        """
        self.executed.append(sql)
        return "OK"

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """record ``sql`` and answer the version query.

        :param sql: query text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: the version row
        :rtype: list[dict[str, Any]]
        """
        self.queried.append(sql)
        return [{"version": self._version}]


class _RecordingPool:
    """an asyncpg-pool shape recording every call."""

    def __init__(self) -> None:
        """start with no calls."""
        self.calls: list[str] = []

    async def execute(self, sql: str, *args: Any) -> str:
        """record the call."""
        self.calls.append(sql)
        return "INSERT 0 1"

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        """record the call; nothing is held."""
        self.calls.append(sql)
        return None

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        """record the call; nothing is held."""
        self.calls.append(sql)
        return []


def _registry(pool: Any = None) -> CollectionRegistry:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    return registry


def _config() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


def _memory_row(embedding: list[float]) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "memory_id": uuid.uuid4(),
        "agent_id": uuid.uuid4(),
        "customer_id": uuid.uuid4(),
        "user_id": uuid.uuid4(),
        "conversation_id": uuid.uuid4(),
        "type_memory": "fact",
        "content": "prefers tea",
        "embedding": embedding,
        "date_created": now,
        "date_updated": now,
    }


class TestTheRegistrationTakesTheWidth:
    @pytest.mark.parametrize("width", [0, -1, 16_001])
    def test_a_width_outside_one_to_sixteen_thousand_is_refused_at_registration(self, width: int) -> None:
        with pytest.raises(ValueError, match="between 1 and 16,000"):
            register(MigrationRunner(), embedding_dim=width)

    def test_the_ledger_names_are_the_same_at_any_width(self) -> None:
        default = register(MigrationRunner())
        wide = register(MigrationRunner(), embedding_dim=1536)
        assert wide.descriptions == default.descriptions
        assert default.descriptions[1] == "create_memories_table"
        assert default.descriptions[6] == "create_media_tables"
        assert default.descriptions[7] == "create_memory_chunks"

    @pytest.mark.parametrize(("width", "expected"), [(None, 1024), (1536, 1536), (3072, 3072)])
    async def test_each_creating_migration_renders_the_width(self, width: int | None, expected: int) -> None:
        pkg = register(MigrationRunner()) if width is None else register(MigrationRunner(), embedding_dim=width)
        store = _FakeStore(_YUGABYTE_VERSION)
        for version, table in _CREATING_VERSIONS.items():
            store.executed.clear()
            await pkg.versions[version](store)  # type: ignore[arg-type]
            created = next(sql for sql in store.executed if f"CREATE TABLE IF NOT EXISTS {table}" in sql)
            assert f"public.vector({expected})" in created, table
            assert "vector(1024)" not in created or expected == 1024, table

    async def test_the_default_width_asks_the_database_nothing(self) -> None:
        store = _FakeStore()
        pkg = register(MigrationRunner())
        for version in _CREATING_VERSIONS:
            await pkg.versions[version](store)  # type: ignore[arg-type]
        assert store.queried == []

    async def test_postgres_refuses_a_width_above_two_thousand_before_any_ddl(self) -> None:
        store = _FakeStore(_POSTGRES_VERSION)
        pkg = register(MigrationRunner(), embedding_dim=2001)
        with pytest.raises(ValueError, match="at most 2,000") as raised:
            await pkg.versions[1](store)  # type: ignore[arg-type]
        assert "16,000" in str(raised.value)
        assert store.executed == []


class TestTheCollectionsTakeTheWidth:
    def test_the_default_is_1024(self) -> None:
        memories = MemoriesCollection(_registry(), _config(), MagicMock(spec=MemoryAuthorizerDependencies))
        assert memories.schema.column("embedding").vector_dim == 1024

    def test_each_embedding_collection_takes_a_configured_width(self) -> None:
        memories = MemoriesCollection(
            _registry(), _config(), MagicMock(spec=MemoryAuthorizerDependencies), embedding_dim=1536
        )
        media_content = MediaContentCollection(_registry(), _config(), embedding_dim=1536)
        chunks = MemoryChunkCollection(_registry(), _config(), embedding_dim=1536)
        for collection in (memories, media_content, chunks):
            assert collection.schema.column("embedding").vector_dim == 1536, collection.table_name

    def test_a_configured_width_leaves_the_declaration_alone(self) -> None:
        MemoriesCollection(_registry(), _config(), MagicMock(spec=MemoryAuthorizerDependencies), embedding_dim=1536)
        MediaContentCollection(_registry(), _config(), embedding_dim=1536)
        MemoryChunkCollection(_registry(), _config(), embedding_dim=1536)
        for cls in (MemoriesCollection, MediaContentCollection, MemoryChunkCollection):
            assert cls.schema.column("embedding").vector_dim == 1024, cls.__name__

    def test_a_width_out_of_range_is_refused(self) -> None:
        with pytest.raises(ValueError, match="between 1 and 16,000"):
            MemoryChunkCollection(_registry(), _config(), embedding_dim=0)

    @pytest.mark.parametrize("factory", [memories_table, media_content_table, memory_chunks_table])
    def test_each_factory_takes_the_width_and_defaults_to_1024(self, factory: Any) -> None:
        assert factory(MetaData()).c.embedding.type.dim == 1024
        assert factory(MetaData(), embedding_dim=1536).c.embedding.type.dim == 1536


class TestAWriteOfAnotherWidthIsRefused:
    async def test_a_1024_embedding_into_a_1536_collection_writes_nothing(self) -> None:
        pool = _RecordingPool()
        memories = MemoriesCollection(
            _registry(pool), _config(), MagicMock(spec=MemoryAuthorizerDependencies), embedding_dim=1536
        )
        with pytest.raises(ValueError, match=r"expected 1536, got 1024"):
            await memories.save_entity(memories.create(_memory_row([0.1] * 1024)))
        assert pool.calls == []

    async def test_a_1536_embedding_into_a_1536_collection_is_written(self) -> None:
        pool = _RecordingPool()
        memories = MemoriesCollection(
            _registry(pool), _config(), MagicMock(spec=MemoryAuthorizerDependencies), embedding_dim=1536
        )
        await memories.save_entity(memories.create(_memory_row([0.1] * 1536)))
        assert any("INSERT INTO memories" in sql for sql in pool.calls)
