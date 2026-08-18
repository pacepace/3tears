"""Tests for BaseEntity cache-proxy class."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.core.cache import MISSING
from threetears.core.entities.base import BaseEntity
from threetears.core.testing import entity_collection_stub


@pytest.fixture
def mock_collection():
    """Mock collection with in-memory L1 cache simulation."""
    cache: dict[str, dict[str, object]] = {}

    coll = MagicMock()

    def write_to_cache(data: dict[str, object]) -> bool:
        pk = data.get("id", "")
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
    coll.save_entity = AsyncMock()
    coll.reload_entity = AsyncMock()

    return coll, cache


class TestBaseEntity:
    """Tests for BaseEntity."""

    def test_create_without_collection(self) -> None:
        """Entity stores data in _changes, can read via attribute access."""
        entity = BaseEntity({"id": "abc", "name": "Alice", "age": 30})

        assert entity.id == "abc"
        assert entity.name == "Alice"
        assert entity.age == 30
        assert entity.is_new is True
        assert entity.is_dirty is True

    def test_create_with_collection(self, mock_collection: tuple[MagicMock, dict[str, dict[str, object]]]) -> None:
        """Writes to L1 on construction, _changes is empty."""
        coll, cache = mock_collection
        data = {"id": "e1", "name": "Bob", "score": 42}

        entity = BaseEntity(data, is_new=True, collection=coll)

        coll.write_to_cache_sync.assert_called_once_with(data)
        assert cache["e1"]["name"] == "Bob"
        assert object.__getattribute__(entity, "_changes") == {}
        assert entity.name == "Bob"

    def test_getattr_reads_from_changes_first(
        self, mock_collection: tuple[MagicMock, dict[str, dict[str, object]]]
    ) -> None:
        """Modified field reads show new value from _changes, not L1."""
        coll, _cache = mock_collection
        entity = BaseEntity({"id": "e2", "name": "Carol"}, is_new=False, collection=coll)

        entity.name = "Caroline"
        assert entity.name == "Caroline"

    def test_getattr_reads_from_l1_cache(self, mock_collection: tuple[MagicMock, dict[str, dict[str, object]]]) -> None:
        """Unmodified field reads from L1 cache."""
        coll, _cache = mock_collection
        entity = BaseEntity(
            {"id": "e3", "name": "Dan", "role": "admin"},
            is_new=False,
            collection=coll,
        )

        assert entity.role == "admin"
        coll.get_field_sync.assert_called_with("e3", "role")

    def test_getattr_raises_attributeerror(
        self, mock_collection: tuple[MagicMock, dict[str, dict[str, object]]]
    ) -> None:
        """Accessing nonexistent field raises AttributeError."""
        coll, _cache = mock_collection
        entity = BaseEntity({"id": "e4", "name": "Eve"}, is_new=False, collection=coll)

        with pytest.raises(AttributeError, match="no attribute 'missing_field'"):
            _ = entity.missing_field

    def test_setattr_records_change_and_updates_l1(
        self, mock_collection: tuple[MagicMock, dict[str, dict[str, object]]]
    ) -> None:
        """Setting a field updates both _changes and L1."""
        coll, cache = mock_collection
        entity = BaseEntity({"id": "e5", "name": "Frank"}, is_new=False, collection=coll)

        entity.name = "Franklin"

        coll.set_field_sync.assert_called_with("e5", "name", "Franklin")
        changes = object.__getattribute__(entity, "_changes")
        assert changes["name"] == "Franklin"
        assert cache["e5"]["name"] == "Franklin"

    def test_get_changes_returns_modified_fields(
        self, mock_collection: tuple[MagicMock, dict[str, dict[str, object]]]
    ) -> None:
        """Only changed fields returned for non-new entity."""
        coll, _cache = mock_collection
        entity = BaseEntity(
            {"id": "e6", "name": "Grace", "age": 25},
            is_new=False,
            collection=coll,
        )

        entity.age = 26
        changes = entity.get_changes()

        assert changes == {"age": 26}
        assert "name" not in changes

    def test_get_changes_returns_all_for_new(
        self, mock_collection: tuple[MagicMock, dict[str, dict[str, object]]]
    ) -> None:
        """New entity returns everything via to_dict()."""
        coll, _cache = mock_collection
        data = {"id": "e7", "name": "Hank", "level": 5}
        entity = BaseEntity(data, is_new=True, collection=coll)

        changes = entity.get_changes()

        assert changes == {"id": "e7", "name": "Hank", "level": 5}

    def test_is_dirty_after_modification(self, mock_collection: tuple[MagicMock, dict[str, dict[str, object]]]) -> None:
        """dirty=True after set, dirty=False after mark_clean."""
        coll, _cache = mock_collection
        entity = BaseEntity({"id": "e8", "name": "Iris"}, is_new=False, collection=coll)

        assert entity.is_dirty is False
        entity.name = "Ivy"
        assert entity.is_dirty is True
        entity.mark_clean()
        assert entity.is_dirty is False

    def test_mark_clean(self, mock_collection: tuple[MagicMock, dict[str, dict[str, object]]]) -> None:
        """Clears changes and dirty flag."""
        coll, _cache = mock_collection
        entity = BaseEntity({"id": "e9", "name": "Jack"}, is_new=True, collection=coll)

        entity.name = "Jackson"
        assert entity.is_dirty is True
        assert entity.is_new is True

        entity.mark_clean()

        assert entity.is_dirty is False
        assert entity.is_new is False
        assert object.__getattribute__(entity, "_changes") == {}

    def test_to_dict_from_l1(self, mock_collection: tuple[MagicMock, dict[str, dict[str, object]]]) -> None:
        """Returns full state from L1."""
        coll, _cache = mock_collection
        data = {"id": "e10", "name": "Kate", "active": True}
        entity = BaseEntity(data, is_new=False, collection=coll)

        result = entity.to_dict()

        assert result == {"id": "e10", "name": "Kate", "active": True}
        coll.get_row_sync.assert_called_with("e10")

    def test_to_dict_without_collection(self) -> None:
        """Returns _changes when no collection."""
        entity = BaseEntity({"id": "e11", "name": "Leo"})

        result = entity.to_dict()

        assert result == {"id": "e11", "name": "Leo"}

    @pytest.mark.asyncio
    async def test_save_delegates(self, mock_collection: tuple[MagicMock, dict[str, dict[str, object]]]) -> None:
        """Calls collection.save_entity."""
        coll, _cache = mock_collection
        entity = BaseEntity({"id": "e12", "name": "Mia"}, is_new=True, collection=coll)

        await entity.save()

        coll.save_entity.assert_awaited_once_with(entity)

    @pytest.mark.asyncio
    async def test_save_without_collection_raises(self) -> None:
        """RuntimeError when saving without collection."""
        entity = BaseEntity({"id": "e13", "name": "Ned"})

        with pytest.raises(RuntimeError, match="Cannot save entity without collection"):
            await entity.save()

    @pytest.mark.asyncio
    async def test_reload_delegates(self, mock_collection: tuple[MagicMock, dict[str, dict[str, object]]]) -> None:
        """Calls collection.reload_entity."""
        coll, _cache = mock_collection
        entity = BaseEntity({"id": "e14", "name": "Olive"}, is_new=False, collection=coll)

        await entity.reload()

        coll.reload_entity.assert_awaited_once_with(entity)

    @pytest.mark.asyncio
    async def test_reload_without_collection_raises(self) -> None:
        """RuntimeError when reloading without collection."""
        entity = BaseEntity({"id": "e15", "name": "Pat"})

        with pytest.raises(RuntimeError, match="Cannot reload entity without collection"):
            await entity.reload()

    def test_id_property(self) -> None:
        """Returns primary key value."""
        entity = BaseEntity({"id": "pk-123", "name": "Quinn"})
        assert entity.id == "pk-123"

    def test_customprimary_key_field(self) -> None:
        """Subclass with different PK field works."""

        class UserEntity(BaseEntity):
            primary_key_field: str = "user_id"

        entity = UserEntity({"user_id": "u-42", "name": "Rose"})

        assert entity.id == "u-42"

    def test_set_data_replaces_l1(self, mock_collection: tuple[MagicMock, dict[str, dict[str, object]]]) -> None:
        """set_data writes to L1 and clears changes."""
        coll, cache = mock_collection
        entity = BaseEntity({"id": "e16", "name": "Sam"}, is_new=True, collection=coll)

        entity.name = "Samuel"
        assert entity.is_dirty is True

        entity.set_data({"id": "e16", "name": "Samwise", "level": 10})

        assert cache["e16"]["name"] == "Samwise"
        assert entity.is_dirty is False
        assert entity.is_new is False
        assert object.__getattribute__(entity, "_changes") == {}

    def test_repr(self) -> None:
        """Includes class name, id, dirty state."""
        entity = BaseEntity({"id": "r1", "name": "Tina"})
        assert repr(entity) == "<BaseEntity id=r1 dirty=True>"

        entity.mark_clean()
        assert repr(entity) == "<BaseEntity id=r1 dirty=False>"


class TestAddressingIdDerivation:
    """``_id`` is derived from the collection's declared pk columns.

    the property the tenancy work depends on: a composite-pk entity
    addresses its row by the full tuple at every tier with no
    per-entity ``__init__`` override, and a single-pk entity is
    untouched.
    """

    def test_single_pk_id_and_addressing_id_coincide(self) -> None:
        """single-pk entity keeps the scalar shape on both accessors."""
        coll, _cache = entity_collection_stub(("id",))
        entity = BaseEntity({"id": "e1", "name": "Ada"}, is_new=True, collection=coll)

        assert entity.id == "e1"
        assert entity.addressing_id == "e1"

    def test_composite_pk_addressing_id_is_declared_order_tuple(self) -> None:
        """composite-pk entity addresses by the tuple, in declared order."""
        coll, _cache = entity_collection_stub(("customer_id", "id"))
        entity = BaseEntity(
            {"customer_id": "cust-A", "id": "row-1", "name": "Ada"},
            is_new=True,
            collection=coll,
        )

        assert entity.addressing_id == ("cust-A", "row-1")

    def test_composite_pk_id_stays_the_bare_row_id(self) -> None:
        """``id`` names ``primary_key_field``, not the addressing tuple."""
        coll, _cache = entity_collection_stub(("customer_id", "id"))
        entity = BaseEntity(
            {"customer_id": "cust-A", "id": "row-1"},
            is_new=True,
            collection=coll,
        )

        assert entity.id == "row-1"

    def test_composite_pk_honours_declared_column_order(self) -> None:
        """the tuple follows ``primary_key_columns``, not dict order."""
        coll, _cache = entity_collection_stub(("agent_id", "conversation_id"))
        entity = BaseEntity(
            {"conversation_id": "conv-9", "agent_id": "agent-1"},
            is_new=True,
            collection=coll,
        )

        assert entity.addressing_id == ("agent-1", "conv-9")

    def test_two_tenants_sharing_a_row_id_address_distinctly(self) -> None:
        """same ``id`` under two customers yields two addressing keys.

        the cross-tenant collision this derivation exists to prevent:
        without it both rows address as ``"row-1"`` and the second read
        answers from the first's cache entry.
        """
        coll, _cache = entity_collection_stub(("customer_id", "id"))
        first = BaseEntity({"customer_id": "cust-A", "id": "row-1"}, collection=coll)
        second = BaseEntity({"customer_id": "cust-B", "id": "row-1"}, collection=coll)

        assert first.id == second.id
        assert first.addressing_id != second.addressing_id

    def test_no_collection_keeps_scalar_shape(self) -> None:
        """a transient entity cannot know its table's key shape."""
        entity = BaseEntity({"customer_id": "cust-A", "id": "row-1"})

        assert entity.addressing_id == "row-1"

    def test_missing_pk_column_falls_back_to_scalar(self) -> None:
        """a payload short one pk column does not address a ``None``.

        the miss surfaces later as ``normalize_pk``'s arity error, which
        names the table and the expected columns; a tuple carrying
        ``None`` would instead read and write a real, wrong row.
        """
        coll, _cache = entity_collection_stub(("customer_id", "id"))
        entity = BaseEntity({"id": "row-1"}, is_new=True, collection=coll)

        assert entity.addressing_id == "row-1"

    def test_pk_columns_absent_from_collection_keeps_scalar_shape(self) -> None:
        """a stand-in collection with no declared pk shape is tolerated."""
        coll = MagicMock()
        coll.write_to_cache_sync = MagicMock(return_value=False)
        entity = BaseEntity({"id": "row-1"}, is_new=True, collection=coll)

        assert entity.addressing_id == "row-1"
