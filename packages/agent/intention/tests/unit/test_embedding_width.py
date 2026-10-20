"""agent-intention's embedding width is the deployment's, 1024 unless configured.

One width serves memory and intention, so a shared embedding provider feeds both. What needs no
database is pinned here; the catalogue side (verify, resize, a real dedup search at 1536) is the
integration suite's.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import MetaData

from threetears.agent.intention.collections import IntentionsCollection, intentions_table
from threetears.agent.intention.migrations import register
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

_POSTGRES_VERSION = "PostgreSQL 16.4 (Debian 16.4-1.pgdg120+2) on x86_64-pc-linux-gnu"
_YUGABYTE_VERSION = "PostgreSQL 15.12-YB-2025.2.0.0-b0 on x86_64-pc-linux-gnu"


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


def _collection(pool: Any = None, **kwargs: Any) -> IntentionsCollection:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    return IntentionsCollection(
        registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""), **kwargs
    )


def _row(embedding: list[float]) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "intention_id": uuid.uuid4(),
        "agent_id": uuid.uuid4(),
        "user_id": uuid.uuid4(),
        "status": "open",
        "content": "learn the cello",
        "embedding": embedding,
        "date_created": now,
        "date_updated": now,
    }


class TestTheRegistrationTakesTheWidth:
    @pytest.mark.parametrize("width", [0, -1, 16_001])
    def test_a_width_outside_one_to_sixteen_thousand_is_refused_at_registration(self, width: int) -> None:
        with pytest.raises(ValueError, match="between 1 and 16,000"):
            register(MigrationRunner(), embedding_dim=width)

    def test_the_ledger_name_is_the_same_at_any_width(self) -> None:
        assert register(MigrationRunner(), embedding_dim=1536).descriptions == {1: "create_intentions_table"}
        assert register(MigrationRunner()).descriptions == {1: "create_intentions_table"}

    @pytest.mark.parametrize(("width", "expected"), [(None, 1024), (1536, 1536), (3072, 3072)])
    async def test_the_table_is_created_at_the_width(self, width: int | None, expected: int) -> None:
        pkg = register(MigrationRunner()) if width is None else register(MigrationRunner(), embedding_dim=width)
        store = _FakeStore(_YUGABYTE_VERSION)
        await pkg.versions[1](store)  # type: ignore[arg-type]
        created = next(sql for sql in store.executed if "CREATE TABLE IF NOT EXISTS intentions" in sql)
        assert f"public.vector({expected})" in created

    async def test_the_default_width_asks_the_database_nothing(self) -> None:
        store = _FakeStore()
        await register(MigrationRunner()).versions[1](store)  # type: ignore[arg-type]
        assert store.queried == []

    async def test_postgres_refuses_a_width_above_two_thousand_before_any_ddl(self) -> None:
        store = _FakeStore(_POSTGRES_VERSION)
        with pytest.raises(ValueError, match="at most 2,000") as raised:
            await register(MigrationRunner(), embedding_dim=2001).versions[1](store)  # type: ignore[arg-type]
        assert "16,000" in str(raised.value)
        assert store.executed == []


class TestTheCollectionTakesTheWidth:
    def test_the_default_is_1024(self) -> None:
        assert _collection().schema.column("embedding").vector_dim == 1024

    def test_a_configured_width_is_the_instances_and_not_the_declarations(self) -> None:
        assert _collection(embedding_dim=1536).schema.column("embedding").vector_dim == 1536
        assert IntentionsCollection.schema.column("embedding").vector_dim == 1024

    def test_a_width_out_of_range_is_refused(self) -> None:
        with pytest.raises(ValueError, match="between 1 and 16,000"):
            _collection(embedding_dim=16_001)

    def test_the_factory_takes_the_width_and_defaults_to_1024(self) -> None:
        assert intentions_table(MetaData()).c.embedding.type.dim == 1024
        assert intentions_table(MetaData(), embedding_dim=1536).c.embedding.type.dim == 1536

    async def test_a_write_of_another_width_is_refused_before_any_statement(self) -> None:
        pool = _RecordingPool()
        intentions = _collection(pool, embedding_dim=1536)
        with pytest.raises(ValueError, match=r"expected 1536, got 1024"):
            await intentions.save_entity(intentions.create(_row([0.1] * 1024)))
        assert pool.calls == []

    async def test_a_write_of_the_width_is_written(self) -> None:
        pool = _RecordingPool()
        intentions = _collection(pool, embedding_dim=1536)
        await intentions.save_entity(intentions.create(_row([0.1] * 1536)))
        assert any("INSERT INTO intentions" in sql for sql in pool.calls)
