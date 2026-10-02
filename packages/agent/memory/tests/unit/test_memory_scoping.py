"""Tests for memory scoping -- agent_id and customer_id on MemoryEntity and on every hybrid search."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest
from uuid import uuid7

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import MediaContentCollection, MemoriesCollection
from threetears.agent.memory.entities import MemoryEntity
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing import entity_collection_stub


@pytest.fixture()
def mock_collection():
    """Collection stub declaring the ``memories`` composite pk shape.

    ``memories`` is partitioned on ``agent_id`` with composite pk
    ``(agent_id, memory_id)``, so the stub must declare that: the
    entity derives its addressing ``_id`` from the collection, and a
    stub omitting the shape makes every cache call address the bare
    ``memory_id``.
    """
    return entity_collection_stub(("agent_id", "memory_id"))


def _sample_data() -> dict:
    """Build sample memory data dict with scoping fields."""
    return {
        "memory_id": uuid7(),
        "agent_id": uuid7(),
        "customer_id": uuid7(),
        "user_id": uuid7(),
        "conversation_id": uuid7(),
        "message_id_source": uuid7(),
        "type_memory": "preference",
        "content": "User prefers dark mode",
        "embedding": [0.1, 0.2, 0.3],
        "media_id": None,
        "is_deleted": False,
        "date_deleted": None,
        "date_updated": None,
    }


class TestMemoryEntityAgentId:
    """Verify agent_id property getter and setter on MemoryEntity."""

    def test_agent_id_getter_returns_uuid(self) -> None:
        data = _sample_data()
        entity = MemoryEntity(data)

        result = entity.agent_id

        assert result == data["agent_id"]
        assert isinstance(result, UUID)

    def test_agent_id_getter_coerces_string(self) -> None:
        data = _sample_data()
        agent_uuid = data["agent_id"]
        data["agent_id"] = str(agent_uuid)
        entity = MemoryEntity(data)

        result = entity.agent_id

        assert result == agent_uuid
        assert isinstance(result, UUID)

    def test_agent_id_setter_updates_value(self) -> None:
        data = _sample_data()
        entity = MemoryEntity(data)
        new_agent = uuid7()

        entity.agent_id = new_agent

        assert entity.agent_id == new_agent

    def test_agent_id_setter_with_collection(self, mock_collection: tuple) -> None:
        coll, cache = mock_collection
        data = _sample_data()
        # a new entity lives in L1, so its setters write through, addressed by the composite key.
        entity = MemoryEntity(data, is_new=True, collection=coll)
        new_agent = uuid7()

        entity.agent_id = new_agent

        assert entity.agent_id == new_agent
        # collections-task-04: entity._id is now the composite tuple
        # ``(agent_id, memory_id)`` so set_field_sync receives that
        # tuple rather than the bare memory_id.
        coll.set_field_sync.assert_called_with(
            (data["agent_id"], data["memory_id"]),
            "agent_id",
            new_agent,
        )

    def test_agent_id_in_to_dict(self) -> None:
        data = _sample_data()
        entity = MemoryEntity(data)

        result = entity.to_dict()

        assert "agent_id" in result
        assert result["agent_id"] == data["agent_id"]


class TestMemoryEntityCustomerId:
    """Verify customer_id property getter and setter on MemoryEntity."""

    def test_customer_id_getter_returns_uuid(self) -> None:
        data = _sample_data()
        entity = MemoryEntity(data)

        result = entity.customer_id

        assert result == data["customer_id"]
        assert isinstance(result, UUID)

    def test_customer_id_getter_coerces_string(self) -> None:
        data = _sample_data()
        customer_uuid = data["customer_id"]
        data["customer_id"] = str(customer_uuid)
        entity = MemoryEntity(data)

        result = entity.customer_id

        assert result == customer_uuid
        assert isinstance(result, UUID)

    def test_customer_id_setter_updates_value(self) -> None:
        data = _sample_data()
        entity = MemoryEntity(data)
        new_customer = uuid7()

        entity.customer_id = new_customer

        assert entity.customer_id == new_customer

    def test_customer_id_setter_with_collection(self, mock_collection: tuple) -> None:
        coll, cache = mock_collection
        data = _sample_data()
        # a new entity lives in L1, so its setters write through, addressed by the composite key.
        entity = MemoryEntity(data, is_new=True, collection=coll)
        new_customer = uuid7()

        entity.customer_id = new_customer

        assert entity.customer_id == new_customer
        # collections-task-04: entity._id is now the composite tuple
        # ``(agent_id, memory_id)``.
        coll.set_field_sync.assert_called_with(
            (data["agent_id"], data["memory_id"]),
            "customer_id",
            new_customer,
        )

    def test_customer_id_in_to_dict(self) -> None:
        data = _sample_data()
        entity = MemoryEntity(data)

        result = entity.to_dict()

        assert "customer_id" in result
        assert result["customer_id"] == data["customer_id"]


class TestMemoryEntityScopingCoexistence:
    """Verify scoping fields coexist properly with existing user_id."""

    def test_all_scope_ids_readable(self) -> None:
        data = _sample_data()
        entity = MemoryEntity(data)

        assert entity.agent_id == data["agent_id"]
        assert entity.customer_id == data["customer_id"]
        assert entity.user_id == data["user_id"]

    def test_all_scope_ids_writable(self) -> None:
        data = _sample_data()
        entity = MemoryEntity(data)
        new_agent = uuid7()
        new_customer = uuid7()
        new_user = uuid7()

        entity.agent_id = new_agent
        entity.customer_id = new_customer
        entity.user_id = new_user

        assert entity.agent_id == new_agent
        assert entity.customer_id == new_customer
        assert entity.user_id == new_user

    def test_changes_track_scope_fields(self, mock_collection: tuple) -> None:
        coll, _ = mock_collection
        data = _sample_data()
        entity = MemoryEntity(data, is_new=False, collection=coll)
        new_agent = uuid7()
        new_customer = uuid7()

        entity.agent_id = new_agent
        entity.customer_id = new_customer

        changes = entity.get_changes()
        assert changes["agent_id"] == new_agent
        assert changes["customer_id"] == new_customer


class _RecordingPool:
    """L3 pool stand-in that records every statement and its parameters, returning no rows."""

    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        self.statements.append((" ".join(sql.split()), params))
        return []


def _registry(pool: _RecordingPool) -> tuple[CollectionRegistry, DefaultCoreConfig]:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    return registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


_SEARCH_KWARGS: dict[str, Any] = {
    "embedding": [1.0, 0.0],
    "user_text": "hello world",
    "top_k": 5,
    "candidate_limit": 7,
    "similarity_threshold": 0.0,
    "recency_half_life_hours": 24.0,
    "signal_weights": {"semantic": 1.0, "keyword": 0.0, "recency": 0.0},
}


class TestBuildScopeClause:
    """every hybrid search scopes by the full (agent, customer, user) triple.

    collections-task-04 made ``agent_id`` (partition column) and
    ``customer_id`` (sub-scope) mandatory; no search admits an optional /
    agent-only / customer-only call shape.
    """

    async def test_required_triple(self, permissive_memory_authorizer: MemoryAuthorizerDependencies) -> None:
        """both legs emit the predicates in (agent, customer, user) order from $2, then the limit."""
        pool = _RecordingPool()
        registry, config = _registry(pool)
        user_id, agent_id, customer_id = uuid7(), uuid7(), uuid7()

        await MemoriesCollection(
            registry=registry, config=config, authorizer=permissive_memory_authorizer
        ).hybrid_search(user_id=user_id, agent_id=agent_id, customer_id=customer_id, **_SEARCH_KWARGS)

        assert len(pool.statements) == 2, "a vector leg and a keyword leg"
        for sql, params in pool.statements:
            assert "WHERE agent_id = $2 AND customer_id = $3 AND user_id = $4" in sql
            assert list(params[1:4]) == [agent_id, customer_id, user_id]
            assert "LIMIT $5" in sql
            assert params[4] == 7

    async def test_with_table_prefix(self) -> None:
        """a joined search qualifies every scope predicate with its table alias."""
        pool = _RecordingPool()
        registry, config = _registry(pool)
        user_id, agent_id, customer_id = uuid7(), uuid7(), uuid7()

        await MediaContentCollection(registry=registry, config=config).hybrid_search(
            user_id=user_id, agent_id=agent_id, customer_id=customer_id, **_SEARCH_KWARGS
        )

        assert pool.statements
        for sql, params in pool.statements:
            assert "mc.agent_id = $2 AND mc.customer_id = $3 AND mc.user_id = $4" in sql
            assert list(params[1:4]) == [agent_id, customer_id, user_id]

    async def test_agent_id_required(self, permissive_memory_authorizer: MemoryAuthorizerDependencies) -> None:
        """omitting ``agent_id`` raises (no longer optional)."""
        registry, config = _registry(_RecordingPool())
        memories = MemoriesCollection(registry=registry, config=config, authorizer=permissive_memory_authorizer)

        with pytest.raises(TypeError):
            await memories.hybrid_search(user_id=uuid7(), customer_id=uuid7(), **_SEARCH_KWARGS)  # type: ignore[call-arg]

    async def test_customer_id_required(self, permissive_memory_authorizer: MemoryAuthorizerDependencies) -> None:
        """omitting ``customer_id`` raises (no longer optional)."""
        registry, config = _registry(_RecordingPool())
        memories = MemoriesCollection(registry=registry, config=config, authorizer=permissive_memory_authorizer)

        with pytest.raises(TypeError):
            await memories.hybrid_search(user_id=uuid7(), agent_id=uuid7(), **_SEARCH_KWARGS)  # type: ignore[call-arg]
