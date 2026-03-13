"""Tests for MemoryEntity cache-proxy class."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid7

import pytest

from threetears.core.cache import MISSING
from threetears.agent.memory.entities import MemoryEntity


@pytest.fixture
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

    coll._write_to_cache_sync = MagicMock(side_effect=write_to_cache)
    coll._get_field_sync = MagicMock(side_effect=get_field)
    coll._set_field_sync = MagicMock(side_effect=set_field)
    coll._get_row_sync = MagicMock(side_effect=get_row)
    coll.save_entity = AsyncMock()
    coll.reload_entity = AsyncMock()

    return coll, cache


def _sample_data() -> dict:
    return {
        "memory_id": uuid7(),
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


class TestMemoryEntityWithoutCollection:
    def test_create_stores_data_in_changes(self) -> None:
        data = _sample_data()
        entity = MemoryEntity(data)

        assert entity.id == data["memory_id"]
        assert entity.memory_id == data["memory_id"]
        assert entity.is_new is True
        assert entity.is_dirty is True

    def test_read_all_properties(self) -> None:
        data = _sample_data()
        entity = MemoryEntity(data)

        assert entity.user_id == data["user_id"]
        assert entity.conversation_id == data["conversation_id"]
        assert entity.message_id_source == data["message_id_source"]
        assert entity.type_memory == "preference"
        assert entity.content == "User prefers dark mode"
        assert entity.embedding == [0.1, 0.2, 0.3]
        assert entity.media_id is None
        assert entity.is_deleted is False
        assert entity.date_deleted is None
        assert entity.date_updated is None

    def test_to_dict_returns_all_fields(self) -> None:
        data = _sample_data()
        entity = MemoryEntity(data)

        result = entity.to_dict()

        assert result == data

    def test_primary_key_field(self) -> None:
        assert MemoryEntity._primary_key_field == "memory_id"


class TestMemoryEntityWithCollection:
    def test_create_writes_to_l1(self, mock_collection: tuple) -> None:
        coll, cache = mock_collection
        data = _sample_data()
        mid = str(data["memory_id"])

        entity = MemoryEntity(data, is_new=True, collection=coll)

        coll._write_to_cache_sync.assert_called_once_with(data)
        assert mid in cache
        assert entity.content == "User prefers dark mode"

    def test_setattr_tracks_changes(self, mock_collection: tuple) -> None:
        coll, cache = mock_collection
        data = _sample_data()

        entity = MemoryEntity(data, is_new=False, collection=coll)

        entity.content = "Updated content"
        assert entity.content == "Updated content"
        coll._set_field_sync.assert_called_with(data["memory_id"], "content", "Updated content")

    def test_property_setters(self, mock_collection: tuple) -> None:
        coll, _ = mock_collection
        data = _sample_data()
        entity = MemoryEntity(data, is_new=False, collection=coll)

        new_user = uuid7()
        new_conv = uuid7()
        new_msg = uuid7()
        new_media = uuid7()
        now = datetime.now(UTC)

        entity.user_id = new_user
        entity.conversation_id = new_conv
        entity.message_id_source = new_msg
        entity.type_memory = "fact"
        entity.content = "new content"
        entity.embedding = [1.0, 2.0]
        entity.media_id = new_media
        entity.is_deleted = True
        entity.date_deleted = now
        entity.date_updated = now

        assert entity.user_id == new_user
        assert entity.conversation_id == new_conv
        assert entity.message_id_source == new_msg
        assert entity.type_memory == "fact"
        assert entity.content == "new content"
        assert entity.embedding == [1.0, 2.0]
        assert entity.media_id == new_media
        assert entity.is_deleted is True
        assert entity.date_deleted == now
        assert entity.date_updated == now

    def test_get_changes_for_existing(self, mock_collection: tuple) -> None:
        coll, _ = mock_collection
        data = _sample_data()
        entity = MemoryEntity(data, is_new=False, collection=coll)

        entity.content = "changed"
        changes = entity.get_changes()

        assert changes == {"content": "changed"}
        assert "user_id" not in changes

    def test_mark_clean_resets_state(self, mock_collection: tuple) -> None:
        coll, _ = mock_collection
        data = _sample_data()
        entity = MemoryEntity(data, is_new=True, collection=coll)

        entity.content = "modified"
        entity.mark_clean()

        assert entity.is_dirty is False
        assert entity.is_new is False
        assert entity.get_changes() == {}

    async def test_save_delegates_to_collection(self, mock_collection: tuple) -> None:
        coll, _ = mock_collection
        data = _sample_data()
        entity = MemoryEntity(data, is_new=True, collection=coll)

        await entity.save()

        coll.save_entity.assert_awaited_once_with(entity)

    async def test_save_without_collection_raises(self) -> None:
        entity = MemoryEntity(_sample_data())

        with pytest.raises(RuntimeError, match="Cannot save entity without collection"):
            await entity.save()

    def test_repr(self, mock_collection: tuple) -> None:
        coll, _ = mock_collection
        data = _sample_data()
        entity = MemoryEntity(data, is_new=True, collection=coll)

        r = repr(entity)
        assert "MemoryEntity" in r
        assert str(data["memory_id"]) in r
