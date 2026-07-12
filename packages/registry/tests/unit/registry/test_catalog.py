"""tests for ToolCatalog, CatalogEntry, and ToolEndpoint."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint


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
    endpoint = ToolEndpoint(
        pod_id=pod_id,
        status=status,
    )
    result = CatalogEntry(
        tool_name=tool_name,
        tool_version=tool_version,
        full_name=f"{tool_name}@{tool_version}",
        description=f"test tool {tool_name}",
        input_schema={"type": "object", "properties": {}},
        endpoints=[endpoint],
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


# -- ToolEndpoint tests --


class TestToolEndpoint:
    """tests for ToolEndpoint dataclass."""

    def test_endpoint_creation(self) -> None:
        """ToolEndpoint stores all fields correctly."""
        endpoint = ToolEndpoint(pod_id="pod-001", status="available")
        assert endpoint.pod_id == "pod-001"
        assert endpoint.status == "available"
        assert endpoint.in_flight == 0
        assert isinstance(endpoint.date_last_heartbeat, datetime)

    def test_endpoint_defaults(self) -> None:
        """ToolEndpoint defaults to pending status and zero in-flight.

        default is 'pending' so new endpoints require an explicit probe
        confirmation before they become routable -- preventing the
        footgun where a bare ``ToolEndpoint(pod_id=X)`` would otherwise
        be picked up as routable immediately.
        """
        endpoint = ToolEndpoint(pod_id="pod-x")
        assert endpoint.status == "pending"
        assert endpoint.in_flight == 0

    def test_endpoint_to_dict(self) -> None:
        """to_dict includes pod_id, status, date_last_heartbeat."""
        endpoint = ToolEndpoint(pod_id="pod-001", status="available")
        data = endpoint.to_dict()
        assert data["pod_id"] == "pod-001"
        assert data["status"] == "available"
        assert "date_last_heartbeat" in data

    def test_endpoint_to_dict_excludes_in_flight(self) -> None:
        """to_dict does not include in_flight (ephemeral runtime state)."""
        endpoint = ToolEndpoint(pod_id="pod-001", in_flight=5)
        data = endpoint.to_dict()
        assert "in_flight" not in data

    def test_endpoint_from_dict_roundtrip(self) -> None:
        """ToolEndpoint serializes and deserializes correctly."""
        original = ToolEndpoint(pod_id="pod-001", status="available")
        data = original.to_dict()
        restored = ToolEndpoint.from_dict(data)
        assert restored.pod_id == original.pod_id
        assert restored.status == original.status
        assert restored.date_last_heartbeat == original.date_last_heartbeat

    def test_endpoint_from_dict_resets_in_flight(self) -> None:
        """from_dict always resets in_flight to zero."""
        endpoint = ToolEndpoint(pod_id="pod-001", in_flight=7)
        data = endpoint.to_dict()
        restored = ToolEndpoint.from_dict(data)
        assert restored.in_flight == 0

    def test_endpoint_from_dict_defaults_status_to_unavailable(self) -> None:
        """from_dict defaults status to unavailable when not present."""
        data = {
            "pod_id": "pod-001",
            "date_last_heartbeat": datetime.now(UTC).isoformat(),
        }
        restored = ToolEndpoint.from_dict(data)
        assert restored.status == "unavailable"


# -- CatalogEntry tests --


class TestCatalogEntry:
    """tests for CatalogEntry dataclass."""

    def test_entry_creation(self) -> None:
        """CatalogEntry stores all fields correctly."""
        entry = _make_entry()
        assert entry.tool_name == "threetears.calculator"
        assert entry.tool_version == "1.0.0"
        assert entry.full_name == "threetears.calculator@1.0.0"
        assert entry.endpoints[0].pod_id == "pod-001"
        assert entry.status == "available"

    def test_entry_status_available_with_available_endpoint(self) -> None:
        """CatalogEntry status is available when at least one endpoint is available."""
        entry = _make_entry(status="available")
        assert entry.status == "available"

    def test_entry_status_unavailable_with_no_endpoints(self) -> None:
        """CatalogEntry status is unavailable when no endpoints exist."""
        entry = CatalogEntry(
            tool_name="test.tool",
            tool_version="1.0",
            full_name="test.tool@1.0",
            description="test",
            input_schema={},
            endpoints=[],
        )
        assert entry.status == "unavailable"

    def test_entry_status_unavailable_when_all_endpoints_unavailable(self) -> None:
        """CatalogEntry status is unavailable when all endpoints are unavailable."""
        entry = CatalogEntry(
            tool_name="test.tool",
            tool_version="1.0",
            full_name="test.tool@1.0",
            description="test",
            input_schema={},
            endpoints=[
                ToolEndpoint(pod_id="pod-a", status="unavailable"),
                ToolEndpoint(pod_id="pod-b", status="unavailable"),
            ],
        )
        assert entry.status == "unavailable"

    def test_entry_status_available_with_mixed_endpoints(self) -> None:
        """CatalogEntry status is available when at least one endpoint is available among many."""
        entry = CatalogEntry(
            tool_name="test.tool",
            tool_version="1.0",
            full_name="test.tool@1.0",
            description="test",
            input_schema={},
            endpoints=[
                ToolEndpoint(pod_id="pod-a", status="unavailable"),
                ToolEndpoint(pod_id="pod-b", status="available"),
            ],
        )
        assert entry.status == "available"

    def test_entry_to_dict_includes_all_fields(self) -> None:
        """to_dict includes all expected fields."""
        entry = _make_entry()
        data = entry.to_dict()
        expected_keys = {
            "tool_name",
            "tool_version",
            "full_name",
            "description",
            "input_schema",
            "output_schema",
            "timeout_seconds",
            "requires_confirmation",
            "endpoints",
            "date_registered",
        }
        assert set(data.keys()) == expected_keys

    def test_entry_to_dict_roundtrip(self) -> None:
        """CatalogEntry serializes and deserializes correctly."""
        original = _make_entry()
        data = original.to_dict()
        restored = CatalogEntry.from_dict(data)
        assert restored.tool_name == original.tool_name
        assert restored.tool_version == original.tool_version
        assert restored.full_name == original.full_name
        assert restored.description == original.description
        assert restored.input_schema == original.input_schema
        assert restored.status == original.status
        assert len(restored.endpoints) == len(original.endpoints)
        assert restored.endpoints[0].pod_id == original.endpoints[0].pod_id
        assert restored.endpoints[0].status == original.endpoints[0].status

    def test_entry_to_dict_roundtrip_multiple_endpoints(self) -> None:
        """CatalogEntry roundtrips correctly with multiple endpoints."""
        entry = CatalogEntry(
            tool_name="test.tool",
            tool_version="1.0",
            full_name="test.tool@1.0",
            description="test",
            input_schema={},
            endpoints=[
                ToolEndpoint(pod_id="pod-a", status="available"),
                ToolEndpoint(pod_id="pod-b", status="unavailable"),
            ],
        )
        data = entry.to_dict()
        restored = CatalogEntry.from_dict(data)
        assert len(restored.endpoints) == 2
        assert restored.endpoints[0].pod_id == "pod-a"
        assert restored.endpoints[0].status == "available"
        assert restored.endpoints[1].pod_id == "pod-b"
        assert restored.endpoints[1].status == "unavailable"

    def test_get_endpoint_found(self) -> None:
        """get_endpoint returns endpoint matching pod_id."""
        entry = _make_entry(pod_id="pod-001")
        endpoint = entry.get_endpoint("pod-001")
        assert endpoint is not None
        assert endpoint.pod_id == "pod-001"

    def test_get_endpoint_not_found(self) -> None:
        """get_endpoint returns None for unknown pod_id."""
        entry = _make_entry(pod_id="pod-001")
        endpoint = entry.get_endpoint("pod-999")
        assert endpoint is None

    def test_add_endpoint_new(self) -> None:
        """add_endpoint appends new endpoint for unknown pod_id."""
        entry = _make_entry(pod_id="pod-001")
        new_endpoint = ToolEndpoint(pod_id="pod-002", status="available")
        entry.add_endpoint(new_endpoint)
        assert len(entry.endpoints) == 2
        assert entry.get_endpoint("pod-002") is new_endpoint

    def test_add_endpoint_replaces_existing(self) -> None:
        """add_endpoint replaces endpoint for existing pod_id."""
        entry = _make_entry(pod_id="pod-001", status="available")
        replacement = ToolEndpoint(pod_id="pod-001", status="unavailable")
        entry.add_endpoint(replacement)
        assert len(entry.endpoints) == 1
        assert entry.endpoints[0].status == "unavailable"

    def test_remove_endpoint_found(self) -> None:
        """remove_endpoint removes matching endpoint and returns True."""
        entry = CatalogEntry(
            tool_name="test.tool",
            tool_version="1.0",
            full_name="test.tool@1.0",
            description="test",
            input_schema={},
            endpoints=[
                ToolEndpoint(pod_id="pod-a"),
                ToolEndpoint(pod_id="pod-b"),
            ],
        )
        removed = entry.remove_endpoint("pod-a")
        assert removed is True
        assert len(entry.endpoints) == 1
        assert entry.endpoints[0].pod_id == "pod-b"

    def test_remove_endpoint_not_found(self) -> None:
        """remove_endpoint returns False for unknown pod_id."""
        entry = _make_entry(pod_id="pod-001")
        removed = entry.remove_endpoint("pod-999")
        assert removed is False
        assert len(entry.endpoints) == 1


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
    async def test_register_overwrites_existing_same_pod(self) -> None:
        """register merges endpoint when same pod registers same tool again."""
        catalog = ToolCatalog()
        entry_a = _make_entry(pod_id="pod-001", status="available")
        entry_b = _make_entry(pod_id="pod-001", status="unavailable")
        await catalog.register(entry_a)
        await catalog.register(entry_b)
        merged = catalog.get("threetears.calculator@1.0.0")
        assert merged is entry_a
        assert len(merged.endpoints) == 1
        assert merged.endpoints[0].status == "unavailable"

    @pytest.mark.asyncio
    async def test_register_merges_endpoints_from_different_pods(self) -> None:
        """register adds new endpoint when different pod registers same tool."""
        catalog = ToolCatalog()
        entry_a = _make_entry(pod_id="pod-001")
        entry_b = _make_entry(pod_id="pod-002")
        await catalog.register(entry_a)
        await catalog.register(entry_b)
        merged = catalog.get("threetears.calculator@1.0.0")
        assert merged is entry_a
        assert len(merged.endpoints) == 2
        assert merged.get_endpoint("pod-001") is not None
        assert merged.get_endpoint("pod-002") is not None

    @pytest.mark.asyncio
    async def test_register_merges_updates_description_and_schema(self) -> None:
        """register updates description and schema on merge."""
        catalog = ToolCatalog()
        entry_a = _make_entry(pod_id="pod-001")
        await catalog.register(entry_a)
        entry_b = CatalogEntry(
            tool_name="threetears.calculator",
            tool_version="1.0.0",
            full_name="threetears.calculator@1.0.0",
            description="updated description",
            input_schema={"type": "object", "properties": {"x": {"type": "number"}}},
            output_schema={"type": "number"},
            endpoints=[ToolEndpoint(pod_id="pod-002")],
        )
        await catalog.register(entry_b)
        merged = catalog.get("threetears.calculator@1.0.0")
        assert merged.description == "updated description"
        assert merged.input_schema == {"type": "object", "properties": {"x": {"type": "number"}}}
        assert merged.output_schema == {"type": "number"}

    @pytest.mark.asyncio
    async def test_register_updates_timeout_seconds_on_merge(self) -> None:
        """register updates timeout_seconds when tool re-registers with new value.

        regression test: catalog.register() previously preserved the old
        timeout_seconds on merge, so a tool that changed its declared
        timeout would be stuck at the original value forever.
        """
        catalog = ToolCatalog()
        entry_a = CatalogEntry(
            tool_name="test.slow_wait",
            tool_version="1.0",
            full_name="test.slow_wait@1.0",
            description="slow tool",
            input_schema={"type": "object", "properties": {}},
            timeout_seconds=120.0,
            endpoints=[ToolEndpoint(pod_id="pod-001")],
        )
        await catalog.register(entry_a)
        assert catalog.get("test.slow_wait@1.0").timeout_seconds == 120.0

        entry_b = CatalogEntry(
            tool_name="test.slow_wait",
            tool_version="1.0",
            full_name="test.slow_wait@1.0",
            description="slow tool",
            input_schema={"type": "object", "properties": {}},
            timeout_seconds=180.0,
            endpoints=[ToolEndpoint(pod_id="pod-001")],
        )
        await catalog.register(entry_b)
        assert catalog.get("test.slow_wait@1.0").timeout_seconds == 180.0, (
            "timeout_seconds must update on re-registration, not stay at old value"
        )

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
    async def test_deregister_pod_removes_all_pod_endpoints(self) -> None:
        """deregister_pod removes endpoints for specified pod; entries with no endpoints left are removed."""
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
    async def test_deregister_pod_keeps_entry_with_remaining_endpoints(self) -> None:
        """deregister_pod keeps entry when other endpoints remain."""
        catalog = ToolCatalog()
        entry = CatalogEntry(
            tool_name="tool.shared",
            tool_version="1.0",
            full_name="tool.shared@1.0",
            description="shared tool",
            input_schema={},
            endpoints=[
                ToolEndpoint(pod_id="pod-A", status="available"),
                ToolEndpoint(pod_id="pod-B", status="available"),
            ],
        )
        await catalog.register(entry)

        removed = await catalog.deregister_pod("pod-A")

        assert "tool.shared@1.0" in removed
        remaining = catalog.get("tool.shared@1.0")
        assert remaining is not None
        assert len(remaining.endpoints) == 1
        assert remaining.endpoints[0].pod_id == "pod-B"

    @pytest.mark.asyncio
    async def test_deregister_pod_returns_affected_names(self) -> None:
        """deregister_pod returns list of affected full_name values."""
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
        await catalog.register(
            _make_entry(
                tool_name="threetears.other",
                tool_version="2.0.0",
            )
        )
        results = catalog.search(version="2.0.0")
        assert len(results) == 1
        assert results[0].tool_version == "2.0.0"

    @pytest.mark.asyncio
    async def test_search_by_name_and_version(self) -> None:
        """search filters by both name and version."""
        catalog = ToolCatalog()
        await catalog.register(
            _make_entry(
                tool_name="threetears.calculator",
                tool_version="1.0.0",
            )
        )
        await catalog.register(
            _make_entry(
                tool_name="threetears.calculator",
                tool_version="2.0.0",
            )
        )
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
        """list_available returns only entries with at least one available endpoint."""
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
    """tests for mark_available, mark_unavailable, and mark_pod_endpoints_available."""

    @pytest.mark.asyncio
    async def test_mark_available(self) -> None:
        """mark_available sets endpoint status to available."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-001", status="unavailable")
        await catalog.register(entry)
        result = catalog.mark_available("threetears.calculator@1.0.0", "pod-001")
        assert result is True
        assert entry.endpoints[0].status == "available"

    @pytest.mark.asyncio
    async def test_mark_unavailable(self) -> None:
        """mark_unavailable sets endpoint status to unavailable."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-001", status="available")
        await catalog.register(entry)
        result = catalog.mark_unavailable("threetears.calculator@1.0.0", "pod-001")
        assert result is True
        assert entry.endpoints[0].status == "unavailable"

    def test_mark_available_returns_false_for_missing_entry(self) -> None:
        """mark_available returns False for nonexistent entry."""
        catalog = ToolCatalog()
        assert catalog.mark_available("missing@1.0", "pod-001") is False

    def test_mark_unavailable_returns_false_for_missing_entry(self) -> None:
        """mark_unavailable returns False for nonexistent entry."""
        catalog = ToolCatalog()
        assert catalog.mark_unavailable("missing@1.0", "pod-001") is False

    @pytest.mark.asyncio
    async def test_mark_available_returns_false_for_missing_endpoint(self) -> None:
        """mark_available returns False for existing entry but missing pod_id."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-001")
        await catalog.register(entry)
        assert catalog.mark_available("threetears.calculator@1.0.0", "pod-999") is False

    @pytest.mark.asyncio
    async def test_mark_unavailable_returns_false_for_missing_endpoint(self) -> None:
        """mark_unavailable returns False for existing entry but missing pod_id."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-001")
        await catalog.register(entry)
        assert catalog.mark_unavailable("threetears.calculator@1.0.0", "pod-999") is False

    @pytest.mark.asyncio
    async def test_mark_pod_endpoints_available(self) -> None:
        """mark_pod_endpoints_available marks all endpoints for pod as available."""
        catalog = ToolCatalog()
        entry_a = _make_entry(
            tool_name="tool.alpha",
            tool_version="1.0",
            pod_id="pod-A",
            status="unavailable",
        )
        entry_b = _make_entry(
            tool_name="tool.beta",
            tool_version="1.0",
            pod_id="pod-A",
            status="unavailable",
        )
        entry_c = _make_entry(
            tool_name="tool.gamma",
            tool_version="1.0",
            pod_id="pod-B",
            status="unavailable",
        )
        await catalog.register(entry_a)
        await catalog.register(entry_b)
        await catalog.register(entry_c)

        marked = catalog.mark_pod_endpoints_available("pod-A")

        assert len(marked) == 2
        assert "tool.alpha@1.0" in marked
        assert "tool.beta@1.0" in marked
        assert entry_a.endpoints[0].status == "available"
        assert entry_b.endpoints[0].status == "available"
        assert entry_c.endpoints[0].status == "unavailable"

    @pytest.mark.asyncio
    async def test_mark_pod_endpoints_available_returns_empty_for_unknown_pod(self) -> None:
        """mark_pod_endpoints_available returns empty list for unknown pod_id."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-001", status="unavailable")
        await catalog.register(entry)
        marked = catalog.mark_pod_endpoints_available("pod-999")
        assert marked == []

    @pytest.mark.asyncio
    async def test_mark_pod_endpoints_available_only_affects_target_pod(self) -> None:
        """mark_pod_endpoints_available does not affect endpoints of other pods."""
        catalog = ToolCatalog()
        entry = CatalogEntry(
            tool_name="tool.shared",
            tool_version="1.0",
            full_name="tool.shared@1.0",
            description="shared tool",
            input_schema={},
            endpoints=[
                ToolEndpoint(pod_id="pod-A", status="unavailable"),
                ToolEndpoint(pod_id="pod-B", status="unavailable"),
            ],
        )
        await catalog.register(entry)

        marked = catalog.mark_pod_endpoints_available("pod-A")

        assert marked == ["tool.shared@1.0"]
        assert entry.get_endpoint("pod-A").status == "available"
        assert entry.get_endpoint("pod-B").status == "unavailable"


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
    async def test_load_from_kv_marks_all_endpoints_unavailable(self) -> None:
        """load_from_kv marks all loaded endpoints as unavailable."""
        entry = _make_entry(status="available")
        kv = _make_mock_kv(entries=[entry])
        catalog = ToolCatalog()
        await catalog.load_from_kv(kv)
        loaded = catalog.get("threetears.calculator@1.0.0")
        assert loaded is not None
        assert loaded.status == "unavailable"
        assert loaded.endpoints[0].status == "unavailable"

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
        key = call_args[0][0]
        payload = json.loads(call_args[0][1].decode("utf-8"))
        assert "threetears" in key
        assert "endpoints" in payload
        assert isinstance(payload["endpoints"], list)

    @pytest.mark.asyncio
    async def test_register_merge_writes_to_kv(self) -> None:
        """register writes merged entry to KV when merging endpoints."""
        kv = _make_mock_kv()
        catalog = ToolCatalog()
        await catalog.load_from_kv(kv)
        entry_a = _make_entry(pod_id="pod-001")
        await catalog.register(entry_a)
        kv.put.reset_mock()
        entry_b = _make_entry(pod_id="pod-002")
        await catalog.register(entry_b)
        kv.put.assert_called_once()
        call_args = kv.put.call_args
        payload = json.loads(call_args[0][1].decode("utf-8"))
        assert len(payload["endpoints"]) == 2

    @pytest.mark.asyncio
    async def test_deregister_deletes_from_kv(self) -> None:
        """deregister deletes entry from KV when KV is configured."""
        entry = _make_entry()
        kv = _make_mock_kv(entries=[entry])
        catalog = ToolCatalog()
        await catalog.load_from_kv(kv)
        await catalog.deregister("threetears.calculator@1.0.0")
        kv.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_deregister_pod_updates_kv_for_remaining_endpoints(self) -> None:
        """deregister_pod persists updated entry to KV when endpoints remain."""
        entry = CatalogEntry(
            tool_name="tool.shared",
            tool_version="1.0",
            full_name="tool.shared@1.0",
            description="shared tool",
            input_schema={},
            endpoints=[
                ToolEndpoint(pod_id="pod-A", status="available"),
                ToolEndpoint(pod_id="pod-B", status="available"),
            ],
        )
        kv = _make_mock_kv(entries=[entry])
        catalog = ToolCatalog()
        await catalog.load_from_kv(kv)
        kv.put.reset_mock()
        kv.delete.reset_mock()

        await catalog.deregister_pod("pod-A")

        kv.put.assert_called_once()
        call_args = kv.put.call_args
        payload = json.loads(call_args[0][1].decode("utf-8"))
        assert len(payload["endpoints"]) == 1
        assert payload["endpoints"][0]["pod_id"] == "pod-B"
        kv.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_deregister_pod_deletes_from_kv_when_no_endpoints(self) -> None:
        """deregister_pod deletes entry from KV when no endpoints remain."""
        entry = _make_entry(pod_id="pod-A")
        kv = _make_mock_kv(entries=[entry])
        catalog = ToolCatalog()
        await catalog.load_from_kv(kv)
        kv.delete.reset_mock()

        await catalog.deregister_pod("pod-A")

        kv.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_without_kv_does_not_fail(self) -> None:
        """register works without KV configured (in-memory only)."""
        catalog = ToolCatalog()
        entry = _make_entry()
        await catalog.register(entry)
        assert catalog.get("threetears.calculator@1.0.0") is entry
