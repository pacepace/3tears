"""Tests for BaseEntity cache-proxy class."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.core.cache import MISSING
from threetears.core.entities.base import BaseEntity


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

    coll._write_to_cache_sync = MagicMock(side_effect=write_to_cache)
    coll._get_field_sync = MagicMock(side_effect=get_field)
    coll._set_field_sync = MagicMock(side_effect=set_field)
    coll._get_row_sync = MagicMock(side_effect=get_row)
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

        coll._write_to_cache_sync.assert_called_once_with(data)
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
        coll._get_field_sync.assert_called_with("e3", "role")

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

        coll._set_field_sync.assert_called_with("e5", "name", "Franklin")
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
        coll._get_row_sync.assert_called_with("e10")

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

    def test_custom_primary_key_field(self) -> None:
        """Subclass with different PK field works."""

        class UserEntity(BaseEntity):
            _primary_key_field: str = "user_id"

        entity = UserEntity({"user_id": "u-42", "name": "Rose"})

        assert entity.id == "u-42"

    def test_set_data_replaces_l1(self, mock_collection: tuple[MagicMock, dict[str, dict[str, object]]]) -> None:
        """_set_data writes to L1 and clears changes."""
        coll, cache = mock_collection
        entity = BaseEntity({"id": "e16", "name": "Sam"}, is_new=True, collection=coll)

        entity.name = "Samuel"
        assert entity.is_dirty is True

        entity._set_data({"id": "e16", "name": "Samwise", "level": 10})

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
