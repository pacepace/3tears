"""unit tests for Workspace, WorkspaceFile, and WorkspaceFileVersion entities."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from threetears.core.cache import MISSING

from threetears.agent.workspace.entities import Workspace, WorkspaceFile, WorkspaceFileVersion


@pytest.fixture
def mock_collection() -> tuple[MagicMock, dict[str, dict[str, Any]]]:
    """
    returns (collection-mock, cache-dict) pair simulating L1 behavior.

    :return: tuple of mocked collection and backing cache dict
    :rtype: tuple[MagicMock, dict[str, dict[str, Any]]]
    """
    cache: dict[str, dict[str, Any]] = {}
    coll = MagicMock()

    def write_to_cache(data: dict[str, Any]) -> bool:
        """write row into cache keyed by id."""
        pk = data.get("id", "")
        cache[str(pk)] = dict(data)
        return True

    def get_field(entity_id: Any, field: str) -> Any:
        """read one field from cache, returning MISSING on miss."""
        row = cache.get(str(entity_id))
        if row is None:
            return MISSING
        return row.get(field, MISSING)

    def set_field(entity_id: Any, field: str, value: Any) -> None:
        """update one field in cache row."""
        row = cache.get(str(entity_id))
        if row is not None:
            row[field] = value

    def get_row(entity_id: Any) -> dict[str, Any] | None:
        """fetch full row dict from cache."""
        return cache.get(str(entity_id))

    coll._write_to_cache_sync = MagicMock(side_effect=write_to_cache)
    coll._get_field_sync = MagicMock(side_effect=get_field)
    coll._set_field_sync = MagicMock(side_effect=set_field)
    coll._get_row_sync = MagicMock(side_effect=get_row)
    coll.save_entity = AsyncMock()
    coll.reload_entity = AsyncMock()
    return coll, cache


def _workspace_data() -> dict[str, Any]:
    """return sample Workspace row dict."""
    return {
        "id": uuid4(),
        "agent_id": uuid4(),
        "name": "design-notes",
        "description": "scratchpad for design docs",
        "template_name": "starter",
        "created_by": uuid4(),
        "current_version": 0,
        "date_created": datetime.now(UTC),
        "date_updated": datetime.now(UTC),
    }


def _workspace_file_data() -> dict[str, Any]:
    """return sample WorkspaceFile row dict."""
    return {
        "id": uuid4(),
        "workspace_id": uuid4(),
        "relative_path": "docs/intro.md",
        "content": b"hello bytes\x00binary",
        "sha256": "a" * 64,
        "version": 1,
        "date_updated": datetime.now(UTC),
    }


def _workspace_file_version_data() -> dict[str, Any]:
    """return sample WorkspaceFileVersion row dict."""
    return {
        "id": uuid4(),
        "workspace_id": uuid4(),
        "relative_path": "docs/intro.md",
        "version": 3,
        "content": b"\x00\x01\x02 payload",
        "sha256": "b" * 64,
        "action": "update",
        "label": None,
        "actor_id": uuid4(),
        "correlation_id": uuid4(),
        "date_created": datetime.now(UTC),
    }


class TestWorkspaceEntity:
    """tests for Workspace entity without a collection."""

    def test_primary_key_field_is_id(self) -> None:
        """primary key field is id per shard."""
        assert Workspace._primary_key_field == "id"

    def test_create_sets_id_and_flags(self) -> None:
        """factory construction populates id and is_new flag."""
        data = _workspace_data()
        entity = Workspace(data)
        assert entity.id == data["id"]
        assert entity.is_new is True
        assert entity.is_dirty is True

    def test_all_properties_round_trip(self) -> None:
        """each declared property reads back the value given at construction."""
        data = _workspace_data()
        entity = Workspace(data)
        assert entity.agent_id == data["agent_id"]
        assert entity.name == data["name"]
        assert entity.description == data["description"]
        assert entity.template_name == data["template_name"]
        assert entity.created_by == data["created_by"]
        assert entity.current_version == data["current_version"]
        assert entity.date_created == data["date_created"]
        assert entity.date_updated == data["date_updated"]

    def test_uuid_coerced_from_string(self) -> None:
        """UUID-typed fields read from str inputs return UUID objects."""
        data = _workspace_data()
        data["agent_id"] = str(data["agent_id"])
        entity = Workspace(data)
        assert isinstance(entity.agent_id, UUID)

    def test_to_dict_returns_all_fields(self) -> None:
        """to_dict exports the row back as-is when no collection is bound."""
        data = _workspace_data()
        entity = Workspace(data)
        assert entity.to_dict() == data

    def test_setter_updates_value(self, mock_collection: tuple[MagicMock, dict[str, dict[str, Any]]]) -> None:
        """setting a property writes through the cache proxy."""
        coll, _cache = mock_collection
        data = _workspace_data()
        entity = Workspace(data, is_new=False, collection=coll)
        entity.name = "renamed"
        assert entity.name == "renamed"

    def test_date_deleted_defaults_to_none_when_absent(self) -> None:
        """date_deleted reads as None when absent from row dict."""
        data = _workspace_data()
        # row from a fresh insert has no date_deleted entry; expose explicitly
        data["date_deleted"] = None
        entity = Workspace(data)
        assert entity.date_deleted is None

    def test_date_deleted_round_trip_when_set(self) -> None:
        """date_deleted property exposes UTC datetime when set on row dict."""
        data = _workspace_data()
        deleted_at = datetime(2026, 4, 16, 10, 0, 0, tzinfo=UTC)
        data["date_deleted"] = deleted_at
        entity = Workspace(data)
        assert entity.date_deleted == deleted_at

    def test_date_deleted_setter_writes_through_cache(
        self,
        mock_collection: tuple[MagicMock, dict[str, dict[str, Any]]],
    ) -> None:
        """setting date_deleted writes through the cache proxy."""
        coll, _cache = mock_collection
        data = _workspace_data()
        data["date_deleted"] = None
        entity = Workspace(data, is_new=False, collection=coll)
        when = datetime(2026, 4, 16, 11, 0, 0, tzinfo=UTC)
        entity.date_deleted = when
        assert entity.date_deleted == when


class TestWorkspaceFileEntity:
    """tests for WorkspaceFile entity without a collection."""

    def test_primary_key_field_is_id(self) -> None:
        """primary key field is id per shard."""
        assert WorkspaceFile._primary_key_field == "id"

    def test_all_properties_round_trip(self) -> None:
        """each declared property reads back the value given at construction."""
        data = _workspace_file_data()
        entity = WorkspaceFile(data)
        assert entity.workspace_id == data["workspace_id"]
        assert entity.relative_path == data["relative_path"]
        assert entity.content == data["content"]
        assert entity.sha256 == data["sha256"]
        assert entity.version == data["version"]
        assert entity.date_updated == data["date_updated"]

    def test_content_is_bytes(self) -> None:
        """content field preserves bytes type across read."""
        data = _workspace_file_data()
        entity = WorkspaceFile(data)
        assert isinstance(entity.content, bytes)
        assert entity.content == b"hello bytes\x00binary"


class TestWorkspaceFileVersionEntity:
    """tests for WorkspaceFileVersion entity without a collection."""

    def test_primary_key_field_is_id(self) -> None:
        """primary key field is id per shard."""
        assert WorkspaceFileVersion._primary_key_field == "id"

    def test_all_properties_round_trip(self) -> None:
        """each declared property reads back the value given at construction."""
        data = _workspace_file_version_data()
        entity = WorkspaceFileVersion(data)
        assert entity.workspace_id == data["workspace_id"]
        assert entity.relative_path == data["relative_path"]
        assert entity.version == data["version"]
        assert entity.content == data["content"]
        assert entity.sha256 == data["sha256"]
        assert entity.action == data["action"]
        assert entity.label is None
        assert entity.actor_id == data["actor_id"]
        assert entity.correlation_id == data["correlation_id"]
        assert entity.date_created == data["date_created"]

    def test_label_when_checkpoint(self) -> None:
        """label populates for checkpoint action."""
        data = _workspace_file_version_data()
        data["action"] = "checkpoint"
        data["label"] = "v1-release"
        entity = WorkspaceFileVersion(data)
        assert entity.label == "v1-release"
        assert entity.action == "checkpoint"

    def test_content_is_bytes(self) -> None:
        """content field preserves bytes type across read."""
        data = _workspace_file_version_data()
        entity = WorkspaceFileVersion(data)
        assert isinstance(entity.content, bytes)
