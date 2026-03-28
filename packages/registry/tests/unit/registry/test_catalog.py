"""tests for ToolCatalog and CatalogEntry."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.registry.catalog import CatalogEntry, ToolCatalog


# -- helpers --


def _make_entry(
    tool_name: str = "threetears.calculator",
    tool_version: str = "1.0.0",
    pod_id: str = "pod-001",
    status: str = "available",
) -> CatalogEntry:
    """create catalog entry for testing.

    :param tool_name: namespaced tool name
    :ptype tool_name: str
    :param tool_version: semver version string
    :ptype tool_version: str
    :param pod_id: identifier of serving pod
    :ptype pod_id: str
    :param status: availability status
    :ptype status: str
    :return: test catalog entry
    :rtype: CatalogEntry
    """
    result = CatalogEntry(
        tool_name=tool_name,
        tool_version=tool_version,
        full_name=f"{tool_name}@{tool_version}",
        pod_id=pod_id,
        description=f"test tool {tool_name}",
        input_schema={"type": "object", "properties": {}},
        status=status,
    )
    return result


def _make_mock_kv(entries: list[CatalogEntry] | None = None) -> AsyncMock:
    """create mock NATS KV store.

    :param entries: optional entries to pre-populate KV
    :ptype entries: list[CatalogEntry] | None
    :return: mock KV store
    :rtype: AsyncMock
    """
    kv = AsyncMock()
    if entries:
        kv.keys = AsyncMock(return_value=[e.full_name for e in entries])
        kv_entries_map: dict[str, MagicMock] = {}
        for entry in entries:
            mock_entry = MagicMock()
            mock_entry.value = json.dumps(entry.to_dict()).encode("utf-8")
            kv_entries_map[entry.full_name] = mock_entry

        async def mock_get(key: str) -> MagicMock:
            """return mock KV entry for key."""
            return kv_entries_map[key]

        kv.get = mock_get
    else:
        kv.keys = AsyncMock(return_value=[])
    kv.put = AsyncMock()
    kv.delete = AsyncMock()
    return kv


# -- CatalogEntry tests --


class TestCatalogEntry:
    """tests for CatalogEntry dataclass."""

    def test_entry_creation(self) -> None:
        """CatalogEntry stores all fields correctly."""
        entry = _make_entry()
        assert entry.tool_name == "threetears.calculator"
        assert entry.tool_version == "1.0.0"
        assert entry.full_name == "threetears.calculator@1.0.0"
        assert entry.pod_id == "pod-001"
        assert entry.status == "available"

    def test_entry_default_status(self) -> None:
        """CatalogEntry defaults to available status."""
        entry = CatalogEntry(
            tool_name="test.tool",
            tool_version="1.0",
            full_name="test.tool@1.0",
            pod_id="pod-x",
            description="test",
            input_schema={},
        )
        assert entry.status == "available"

    def test_entry_to_dict_roundtrip(self) -> None:
        """CatalogEntry serializes and deserializes correctly."""
        original = _make_entry()
        data = original.to_dict()
        restored = CatalogEntry.from_dict(data)
        assert restored.tool_name == original.tool_name
        assert restored.tool_version == original.tool_version
        assert restored.full_name == original.full_name
        assert restored.pod_id == original.pod_id
        assert restored.description == original.description
        assert restored.input_schema == original.input_schema
        assert restored.status == original.status

    def test_entry_to_dict_includes_all_fields(self) -> None:
        """to_dict includes all expected fields."""
        entry = _make_entry()
        data = entry.to_dict()
        expected_keys = {
            "tool_name", "tool_version", "full_name", "pod_id",
            "description", "input_schema", "output_schema", "status",
            "date_registered", "date_last_heartbeat",
        }
        assert set(data.keys()) == expected_keys


# -- ToolCatalog CRUD tests --


class TestToolCatalogCRUD:
    """tests for ToolCatalog register, deregister, get operations."""

    @pytest.mark.asyncio
    async def test_register_adds_entry(self) -> None:
        """register stores entry in catalog."""
        catalog = ToolCatalog()
        entry = _make_entry()
        await catalog.register(entry)
        assert catalog.get("threetears.calculator@1.0.0") is entry

    @pytest.mark.asyncio
    async def test_register_overwrites_existing(self) -> None:
        """register replaces entry with same full_name."""
        catalog = ToolCatalog()
        entry_a = _make_entry(pod_id="pod-001")
        entry_b = _make_entry(pod_id="pod-001")
        await catalog.register(entry_a)
        await catalog.register(entry_b)
        assert catalog.get("threetears.calculator@1.0.0") is entry_b

    @pytest.mark.asyncio
    async def test_deregister_removes_entry(self) -> None:
        """deregister removes entry from catalog."""
        catalog = ToolCatalog()
        entry = _make_entry()
        await catalog.register(entry)
        await catalog.deregister("threetears.calculator@1.0.0")
        assert catalog.get("threetears.calculator@1.0.0") is None

    @pytest.mark.asyncio
    async def test_deregister_nonexistent_is_noop(self) -> None:
        """deregister of nonexistent entry does not raise."""
        catalog = ToolCatalog()
        await catalog.deregister("nonexistent@1.0")

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing(self) -> None:
        """get returns None for unregistered tool."""
        catalog = ToolCatalog()
        assert catalog.get("missing@1.0") is None

    @pytest.mark.asyncio
    async def test_deregister_pod_removes_all_pod_tools(self) -> None:
        """deregister_pod removes all tools from specified pod."""
        catalog = ToolCatalog()
        entry_a = _make_entry(
            tool_name="tool.alpha",
            tool_version="1.0",
            pod_id="pod-A",
        )
        entry_b = _make_entry(
            tool_name="tool.beta",
            tool_version="1.0",
            pod_id="pod-A",
        )
        entry_c = _make_entry(
            tool_name="tool.gamma",
            tool_version="1.0",
            pod_id="pod-B",
        )
        await catalog.register(entry_a)
        await catalog.register(entry_b)
        await catalog.register(entry_c)

        removed = await catalog.deregister_pod("pod-A")

        assert len(removed) == 2
        assert catalog.get("tool.alpha@1.0") is None
        assert catalog.get("tool.beta@1.0") is None
        assert catalog.get("tool.gamma@1.0") is entry_c

    @pytest.mark.asyncio
    async def test_deregister_pod_returns_removed_names(self) -> None:
        """deregister_pod returns list of removed full_name values."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-X")
        await catalog.register(entry)
        removed = await catalog.deregister_pod("pod-X")
        assert "threetears.calculator@1.0.0" in removed

    @pytest.mark.asyncio
    async def test_deregister_pod_empty_for_unknown_pod(self) -> None:
        """deregister_pod returns empty list for unknown pod_id."""
        catalog = ToolCatalog()
        removed = await catalog.deregister_pod("nonexistent-pod")
        assert removed == []


# -- search and list tests --


class TestToolCatalogSearch:
    """tests for ToolCatalog search and list_available."""

    @pytest.mark.asyncio
    async def test_search_by_name(self) -> None:
        """search filters entries by tool name substring."""
        catalog = ToolCatalog()
        await catalog.register(_make_entry(tool_name="threetears.calculator"))
        await catalog.register(_make_entry(tool_name="threetears.dictionary"))
        results = catalog.search(name="calculator")
        assert len(results) == 1
        assert results[0].tool_name == "threetears.calculator"

    @pytest.mark.asyncio
    async def test_search_by_version(self) -> None:
        """search filters entries by exact version match."""
        catalog = ToolCatalog()
        await catalog.register(_make_entry(tool_version="1.0.0"))
        await catalog.register(_make_entry(
            tool_name="threetears.other",
            tool_version="2.0.0",
        ))
        results = catalog.search(version="2.0.0")
        assert len(results) == 1
        assert results[0].tool_version == "2.0.0"

    @pytest.mark.asyncio
    async def test_search_by_name_and_version(self) -> None:
        """search filters by both name and version."""
        catalog = ToolCatalog()
        await catalog.register(_make_entry(
            tool_name="threetears.calculator",
            tool_version="1.0.0",
        ))
        await catalog.register(_make_entry(
            tool_name="threetears.calculator",
            tool_version="2.0.0",
        ))
        results = catalog.search(name="calculator", version="2.0.0")
        assert len(results) == 1
        assert results[0].tool_version == "2.0.0"

    @pytest.mark.asyncio
    async def test_search_no_filters_returns_all(self) -> None:
        """search with no filters returns all entries."""
        catalog = ToolCatalog()
        await catalog.register(_make_entry(tool_name="tool.alpha"))
        await catalog.register(_make_entry(tool_name="tool.beta"))
        results = catalog.search()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_list_available_filters_by_status(self) -> None:
        """list_available returns only entries with available status."""
        catalog = ToolCatalog()
        entry_a = _make_entry(tool_name="tool.alpha", status="available")
        entry_b = _make_entry(tool_name="tool.beta", status="unavailable")
        await catalog.register(entry_a)
        await catalog.register(entry_b)
        available = catalog.list_available()
        assert len(available) == 1
        assert available[0].tool_name == "tool.alpha"

    @pytest.mark.asyncio
    async def test_list_available_empty_when_none_available(self) -> None:
        """list_available returns empty list when no tools available."""
        catalog = ToolCatalog()
        entry = _make_entry(status="unavailable")
        await catalog.register(entry)
        assert catalog.list_available() == []


# -- availability marking tests --


class TestToolCatalogAvailability:
    """tests for mark_available and mark_unavailable."""

    @pytest.mark.asyncio
    async def test_mark_available(self) -> None:
        """mark_available sets status to available."""
        catalog = ToolCatalog()
        entry = _make_entry(status="unavailable")
        await catalog.register(entry)
        result = catalog.mark_available("threetears.calculator@1.0.0")
        assert result is True
        assert entry.status == "available"

    @pytest.mark.asyncio
    async def test_mark_unavailable(self) -> None:
        """mark_unavailable sets status to unavailable."""
        catalog = ToolCatalog()
        entry = _make_entry(status="available")
        await catalog.register(entry)
        result = catalog.mark_unavailable("threetears.calculator@1.0.0")
        assert result is True
        assert entry.status == "unavailable"

    def test_mark_available_returns_false_for_missing(self) -> None:
        """mark_available returns False for nonexistent entry."""
        catalog = ToolCatalog()
        assert catalog.mark_available("missing@1.0") is False

    def test_mark_unavailable_returns_false_for_missing(self) -> None:
        """mark_unavailable returns False for nonexistent entry."""
        catalog = ToolCatalog()
        assert catalog.mark_unavailable("missing@1.0") is False


# -- KV persistence tests --


class TestToolCatalogKVPersistence:
    """tests for ToolCatalog KV persistence operations."""

    @pytest.mark.asyncio
    async def test_load_from_kv_populates_catalog(self) -> None:
        """load_from_kv loads entries from KV store."""
        entry = _make_entry(status="available")
        kv = _make_mock_kv(entries=[entry])
        catalog = ToolCatalog()
        await catalog.load_from_kv(kv)
        loaded = catalog.get("threetears.calculator@1.0.0")
        assert loaded is not None
        assert loaded.tool_name == "threetears.calculator"

    @pytest.mark.asyncio
    async def test_load_from_kv_marks_all_unavailable(self) -> None:
        """load_from_kv marks all loaded entries as unavailable."""
        entry = _make_entry(status="available")
        kv = _make_mock_kv(entries=[entry])
        catalog = ToolCatalog()
        await catalog.load_from_kv(kv)
        loaded = catalog.get("threetears.calculator@1.0.0")
        assert loaded is not None
        assert loaded.status == "unavailable"

    @pytest.mark.asyncio
    async def test_load_from_kv_with_multiple_entries(self) -> None:
        """load_from_kv loads all entries from KV store."""
        entries = [
            _make_entry(tool_name="tool.alpha", tool_version="1.0"),
            _make_entry(tool_name="tool.beta", tool_version="2.0"),
        ]
        kv = _make_mock_kv(entries=entries)
        catalog = ToolCatalog()
        await catalog.load_from_kv(kv)
        assert catalog.get("tool.alpha@1.0") is not None
        assert catalog.get("tool.beta@2.0") is not None

    @pytest.mark.asyncio
    async def test_register_writes_to_kv(self) -> None:
        """register writes entry to KV when KV is configured."""
        kv = _make_mock_kv()
        catalog = ToolCatalog()
        await catalog.load_from_kv(kv)
        entry = _make_entry()
        await catalog.register(entry)
        kv.put.assert_called_once()
        call_args = kv.put.call_args
        assert call_args[0][0] == "threetears.calculator@1.0.0"

    @pytest.mark.asyncio
    async def test_deregister_deletes_from_kv(self) -> None:
        """deregister deletes entry from KV when KV is configured."""
        entry = _make_entry()
        kv = _make_mock_kv(entries=[entry])
        catalog = ToolCatalog()
        await catalog.load_from_kv(kv)
        await catalog.deregister("threetears.calculator@1.0.0")
        kv.delete.assert_called_once_with("threetears.calculator@1.0.0")

    @pytest.mark.asyncio
    async def test_register_without_kv_does_not_fail(self) -> None:
        """register works without KV configured (in-memory only)."""
        catalog = ToolCatalog()
        entry = _make_entry()
        await catalog.register(entry)
        assert catalog.get("threetears.calculator@1.0.0") is entry
