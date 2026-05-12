"""Tests for memory scoping -- agent_id and customer_id on MemoryEntity and helpers."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import UUID

import pytest
from uuid import uuid7

from threetears.agent.memory.collections import _build_user_scope_clause
from threetears.agent.memory.entities import MemoryEntity
from threetears.core.cache import MISSING


@pytest.fixture()
def mock_collection():
    """Mock collection with in-memory L1 cache simulation."""
    cache: dict[str, dict[str, object]] = {}
    coll = MagicMock()

    def write_to_cache(data: dict[str, object]) -> bool:
        pk = data.get("memory_id", "")
        cache[str(pk)] = dict(data)
        return True

    def get_field(entity_id: object, field: str) -> object:
        row = cache.get(str(entity_id))
        if row is None:
            return MISSING
        return row.get(field, MISSING)

    def set_field(entity_id: object, field: str, value: object) -> None:
        row = cache.get(str(entity_id))
        if row is not None:
            row[field] = value

    def get_row(entity_id: object) -> dict[str, object] | None:
        return cache.get(str(entity_id))

    coll.write_to_cache_sync = MagicMock(side_effect=write_to_cache)
    coll.get_field_sync = MagicMock(side_effect=get_field)
    coll.set_field_sync = MagicMock(side_effect=set_field)
    coll.get_row_sync = MagicMock(side_effect=get_row)

    return coll, cache


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
        entity = MemoryEntity(data, is_new=False, collection=coll)
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
        entity = MemoryEntity(data, is_new=False, collection=coll)
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


class TestBuildScopeClause:
    """Verify _build_user_scope_clause generates correct SQL fragments.

    collections-task-04 made ``agent_id`` (partition column) and
    ``customer_id`` (sub-scope) mandatory; the helper no longer admits
    optional / agent-only / customer-only call shapes.
    """

    def test_required_triple(self) -> None:
        """all three IDs always emit predicates in (agent, customer, user) order."""
        user_id = uuid7()
        agent_id = uuid7()
        customer_id = uuid7()

        conditions, params, last_idx = _build_user_scope_clause(
            user_id,
            agent_id=agent_id,
            customer_id=customer_id,
        )

        assert "agent_id = $2" in conditions
        assert "customer_id = $3" in conditions
        assert "user_id = $4" in conditions
        assert params == [agent_id, customer_id, user_id]
        assert last_idx == 4

    def test_with_table_prefix(self) -> None:
        """table prefix applies to every scope predicate uniformly."""
        user_id = uuid7()
        agent_id = uuid7()
        customer_id = uuid7()

        conditions, params, last_idx = _build_user_scope_clause(
            user_id,
            agent_id=agent_id,
            customer_id=customer_id,
            table_prefix="mc",
        )

        assert "mc.agent_id = $2" in conditions
        assert "mc.customer_id = $3" in conditions
        assert "mc.user_id = $4" in conditions
        assert params == [agent_id, customer_id, user_id]
        assert last_idx == 4

    def test_custom_start_param(self) -> None:
        """``start_param`` shifts every parameter index in declared order."""
        user_id = uuid7()
        agent_id = uuid7()
        customer_id = uuid7()

        conditions, params, last_idx = _build_user_scope_clause(
            user_id,
            agent_id=agent_id,
            customer_id=customer_id,
            start_param=5,
        )

        assert "agent_id = $5" in conditions
        assert "customer_id = $6" in conditions
        assert "user_id = $7" in conditions
        assert last_idx == 7

    def test_agent_id_required(self) -> None:
        """omitting ``agent_id`` raises (no longer optional)."""
        user_id = uuid7()
        customer_id = uuid7()

        with pytest.raises(TypeError):
            _build_user_scope_clause(
                user_id,
                customer_id=customer_id,  # type: ignore[call-arg]
            )

    def test_customer_id_required(self) -> None:
        """omitting ``customer_id`` raises (no longer optional)."""
        user_id = uuid7()
        agent_id = uuid7()

        with pytest.raises(TypeError):
            _build_user_scope_clause(
                user_id,
                agent_id=agent_id,  # type: ignore[call-arg]
            )
