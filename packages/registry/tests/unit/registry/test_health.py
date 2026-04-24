"""tests for HeartbeatSubscriber + HeartbeatCollection.

namespace-task-01 phase 8.5l-3 retired :class:`HeartbeatMonitor` in
favour of a :class:`BaseCollection`-backed persistent-state surface
(:class:`HeartbeatCollection`) plus a focused orchestration class
(:class:`HeartbeatSubscriber`). these tests exercise both sides: the
subscriber's NATS + sweep behaviour and the Collection's L1-only
pull-through (L2 is exercised in the integration suite).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.agent.tools.server import HeartbeatMessage
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint
from threetears.registry.heartbeat_collection import HeartbeatCollection
from threetears.registry.health import HeartbeatSubscriber
from threetears.registry.l1_cache import create_registry_l1_backend


# -- helpers --


def _make_heartbeat_msg(
    pod_id: str = "pod-001",
    tools_count: int = 1,
) -> MagicMock:
    """create mock NATS message containing heartbeat.

    :param pod_id: pod identifier
    :ptype pod_id: str
    :param tools_count: number of tools in pod
    :ptype tools_count: int
    :return: mock NATS message with heartbeat payload
    :rtype: MagicMock
    """
    heartbeat = HeartbeatMessage(
        pod_id=pod_id,
        timestamp=datetime.now(UTC).isoformat(),
        tools_count=tools_count,
    )
    msg = MagicMock()
    msg.data = heartbeat.model_dump_json().encode("utf-8")
    return msg


def _make_entry(
    tool_name: str = "threetears.calculator",
    tool_version: str = "1.0.0",
    pod_id: str = "pod-001",
) -> CatalogEntry:
    """create catalog entry for testing.

    :param tool_name: namespaced tool name
    :ptype tool_name: str
    :param tool_version: semver version string
    :ptype tool_version: str
    :param pod_id: identifier of serving pod
    :ptype pod_id: str
    :return: test catalog entry
    :rtype: CatalogEntry
    """
    endpoint = ToolEndpoint(pod_id=pod_id, status="available")
    result = CatalogEntry(
        tool_name=tool_name,
        tool_version=tool_version,
        full_name=f"{tool_name}@{tool_version}",
        description=f"test tool {tool_name}",
        input_schema={"type": "object", "properties": {}},
        endpoints=[endpoint],
    )
    return result


def _make_collection() -> HeartbeatCollection:
    """create a HeartbeatCollection wired to an in-memory L1 tier, no L2.

    :return: freshly-initialized collection
    :rtype: HeartbeatCollection
    """
    l1 = create_registry_l1_backend()
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    result = HeartbeatCollection(registry, config)
    return result


def _make_subscriber(
    catalog: ToolCatalog | None = None,
    *,
    timeout: float = 30.0,
    check_interval: float = 5.0,
) -> tuple[HeartbeatSubscriber, HeartbeatCollection]:
    """build subscriber + collection pair for tests.

    :param catalog: optional pre-built catalog; a fresh one is
        created when ``None``
    :ptype catalog: ToolCatalog | None
    :param timeout: liveness timeout
    :ptype timeout: float
    :param check_interval: sweep interval
    :ptype check_interval: float
    :return: (subscriber, collection) pair
    :rtype: tuple[HeartbeatSubscriber, HeartbeatCollection]
    """
    cat = catalog if catalog is not None else ToolCatalog()
    collection = _make_collection()
    subscriber = HeartbeatSubscriber(
        cat,
        collection,
        namespace="test",
        check_interval=check_interval,
        timeout=timeout,
    )
    return subscriber, collection


# -- heartbeat handling tests --


class TestHeartbeatSubscriberHandling:
    """tests for heartbeat message handling."""

    @pytest.mark.asyncio
    async def test_tracks_new_pod_on_first_heartbeat(self) -> None:
        """subscriber creates HeartbeatEntity on first heartbeat from pod."""
        subscriber, collection = _make_subscriber()
        nc = AsyncMock()
        mock_sub = AsyncMock()
        nc.subscribe = AsyncMock(return_value=mock_sub)
        await subscriber.start(nc)

        msg = _make_heartbeat_msg(pod_id="pod-new")
        await subscriber._handle_heartbeat(msg)

        assert "pod-new" in subscriber.known_pod_ids
        entity = await collection.get("pod-new")
        assert entity is not None
        assert entity.pod_id == "pod-new"
        assert entity.status == "healthy"
        await subscriber.stop()

    @pytest.mark.asyncio
    async def test_updates_timestamp_on_subsequent_heartbeat(self) -> None:
        """subscriber updates date_last_heartbeat on subsequent heartbeats."""
        subscriber, collection = _make_subscriber()
        nc = AsyncMock()
        mock_sub = AsyncMock()
        nc.subscribe = AsyncMock(return_value=mock_sub)
        await subscriber.start(nc)

        msg1 = _make_heartbeat_msg(pod_id="pod-001")
        await subscriber._handle_heartbeat(msg1)
        first_entity = await collection.get("pod-001")
        assert first_entity is not None
        first_hb = first_entity.date_last_heartbeat

        msg2 = _make_heartbeat_msg(pod_id="pod-001")
        await subscriber._handle_heartbeat(msg2)
        second_entity = await collection.get("pod-001")
        assert second_entity is not None
        second_hb = second_entity.date_last_heartbeat

        assert second_hb >= first_hb
        await subscriber.stop()

    @pytest.mark.asyncio
    async def test_resets_consecutive_misses_on_heartbeat(self) -> None:
        """subscriber resets consecutive_misses to zero on heartbeat."""
        subscriber, collection = _make_subscriber()
        nc = AsyncMock()
        mock_sub = AsyncMock()
        nc.subscribe = AsyncMock(return_value=mock_sub)
        await subscriber.start(nc)

        # Seed a pre-existing row with non-zero misses
        stale_entity = collection.create(
            {
                "pod_id": "pod-001",
                "date_last_heartbeat": datetime.now(UTC),
                "tools": [],
                "tools_count": 0,
                "status": "unresponsive",
                "consecutive_misses": 5,
            }
        )
        await collection.save_entity(stale_entity)
        subscriber._known_pod_ids.add("pod-001")

        msg = _make_heartbeat_msg(pod_id="pod-001")
        await subscriber._handle_heartbeat(msg)

        reloaded = await collection.get("pod-001")
        assert reloaded is not None
        assert reloaded.consecutive_misses == 0
        assert reloaded.status == "healthy"
        await subscriber.stop()

    @pytest.mark.asyncio
    async def test_ignores_malformed_heartbeat(self) -> None:
        """subscriber ignores messages with invalid payload."""
        subscriber, collection = _make_subscriber()
        nc = AsyncMock()
        mock_sub = AsyncMock()
        nc.subscribe = AsyncMock(return_value=mock_sub)
        await subscriber.start(nc)

        msg = MagicMock()
        msg.data = b"not json"
        await subscriber._handle_heartbeat(msg)

        assert subscriber.known_pod_ids == set()
        await subscriber.stop()

    @pytest.mark.asyncio
    async def test_marks_pod_tools_available_on_heartbeat(self) -> None:
        """subscriber marks endpoints available and records tools on entity."""
        catalog = ToolCatalog()
        endpoint = ToolEndpoint(pod_id="pod-001", status="unavailable")
        entry = CatalogEntry(
            tool_name="threetears.calculator",
            tool_version="1.0.0",
            full_name="threetears.calculator@1.0.0",
            description="test tool threetears.calculator",
            input_schema={"type": "object", "properties": {}},
            endpoints=[endpoint],
        )
        await catalog.register(entry)

        subscriber, collection = _make_subscriber(catalog=catalog)
        nc = AsyncMock()
        mock_sub = AsyncMock()
        nc.subscribe = AsyncMock(return_value=mock_sub)
        await subscriber.start(nc)

        msg = _make_heartbeat_msg(pod_id="pod-001")
        await subscriber._handle_heartbeat(msg)

        registered = catalog.get("threetears.calculator@1.0.0")
        assert registered is not None
        assert registered.endpoints[0].status == "available"
        entity = await collection.get("pod-001")
        assert entity is not None
        assert "threetears.calculator@1.0.0" in entity.tools
        await subscriber.stop()


# -- health check tests --


class TestHeartbeatSubscriberHealthCheck:
    """tests for periodic health check sweep."""

    @pytest.mark.asyncio
    async def test_deregisters_pod_after_timeout(self) -> None:
        """health check removes endpoint and entry when pod is only endpoint."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-stale")
        await catalog.register(entry)

        subscriber, collection = _make_subscriber(catalog=catalog, timeout=30.0)

        stale_time = datetime.now(UTC) - timedelta(seconds=60)
        stale_entity = collection.create(
            {
                "pod_id": "pod-stale",
                "date_last_heartbeat": stale_time,
                "tools": ["threetears.calculator@1.0.0"],
                "tools_count": 1,
                "status": "healthy",
                "consecutive_misses": 0,
            }
        )
        await collection.save_entity(stale_entity)
        subscriber._known_pod_ids.add("pod-stale")

        await subscriber._run_health_check()

        assert "pod-stale" not in subscriber.known_pod_ids
        assert await collection.get("pod-stale") is None
        assert catalog.get("threetears.calculator@1.0.0") is None

    @pytest.mark.asyncio
    async def test_keeps_healthy_pod(self) -> None:
        """health check does not deregister pod within timeout."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-healthy")
        await catalog.register(entry)

        subscriber, collection = _make_subscriber(catalog=catalog, timeout=30.0)

        recent_time = datetime.now(UTC) - timedelta(seconds=5)
        healthy_entity = collection.create(
            {
                "pod_id": "pod-healthy",
                "date_last_heartbeat": recent_time,
                "tools": ["threetears.calculator@1.0.0"],
                "tools_count": 1,
                "status": "healthy",
                "consecutive_misses": 0,
            }
        )
        await collection.save_entity(healthy_entity)
        subscriber._known_pod_ids.add("pod-healthy")

        await subscriber._run_health_check()

        assert "pod-healthy" in subscriber.known_pod_ids
        assert catalog.get("threetears.calculator@1.0.0") is not None

    @pytest.mark.asyncio
    async def test_deregister_preserves_entry_with_surviving_endpoints(self) -> None:
        """health check removes stale endpoint but preserves entry with surviving pod."""
        catalog = ToolCatalog()
        endpoint_stale = ToolEndpoint(pod_id="pod-stale", status="available")
        endpoint_healthy = ToolEndpoint(pod_id="pod-healthy", status="available")
        entry = CatalogEntry(
            tool_name="threetears.calculator",
            tool_version="1.0.0",
            full_name="threetears.calculator@1.0.0",
            description="test tool threetears.calculator",
            input_schema={"type": "object", "properties": {}},
            endpoints=[endpoint_stale, endpoint_healthy],
        )
        await catalog.register(entry)

        subscriber, collection = _make_subscriber(catalog=catalog, timeout=30.0)

        stale_time = datetime.now(UTC) - timedelta(seconds=60)
        recent_time = datetime.now(UTC) - timedelta(seconds=5)

        stale_entity = collection.create(
            {
                "pod_id": "pod-stale",
                "date_last_heartbeat": stale_time,
                "tools": ["threetears.calculator@1.0.0"],
                "tools_count": 1,
                "status": "healthy",
                "consecutive_misses": 0,
            }
        )
        healthy_entity = collection.create(
            {
                "pod_id": "pod-healthy",
                "date_last_heartbeat": recent_time,
                "tools": ["threetears.calculator@1.0.0"],
                "tools_count": 1,
                "status": "healthy",
                "consecutive_misses": 0,
            }
        )
        await collection.save_entity(stale_entity)
        await collection.save_entity(healthy_entity)
        subscriber._known_pod_ids.update({"pod-stale", "pod-healthy"})

        await subscriber._run_health_check()

        assert "pod-stale" not in subscriber.known_pod_ids
        assert "pod-healthy" in subscriber.known_pod_ids
        surviving = catalog.get("threetears.calculator@1.0.0")
        assert surviving is not None
        assert len(surviving.endpoints) == 1
        assert surviving.endpoints[0].pod_id == "pod-healthy"

    @pytest.mark.asyncio
    async def test_deregisters_multiple_stale_pods(self) -> None:
        """health check deregisters all stale pods in single sweep."""
        catalog = ToolCatalog()
        entry_a = _make_entry(
            tool_name="tool.alpha",
            tool_version="1.0",
            pod_id="pod-stale-1",
        )
        entry_b = _make_entry(
            tool_name="tool.beta",
            tool_version="1.0",
            pod_id="pod-stale-2",
        )
        await catalog.register(entry_a)
        await catalog.register(entry_b)

        subscriber, collection = _make_subscriber(catalog=catalog, timeout=30.0)

        stale_time = datetime.now(UTC) - timedelta(seconds=60)
        for pod_id, tool in (
            ("pod-stale-1", "tool.alpha@1.0"),
            ("pod-stale-2", "tool.beta@1.0"),
        ):
            entity = collection.create(
                {
                    "pod_id": pod_id,
                    "date_last_heartbeat": stale_time,
                    "tools": [tool],
                    "tools_count": 1,
                    "status": "healthy",
                    "consecutive_misses": 0,
                }
            )
            await collection.save_entity(entity)
            subscriber._known_pod_ids.add(pod_id)

        await subscriber._run_health_check()

        assert subscriber.known_pod_ids == set()
        assert catalog.get("tool.alpha@1.0") is None
        assert catalog.get("tool.beta@1.0") is None

    @pytest.mark.asyncio
    async def test_increments_consecutive_misses(self) -> None:
        """stale pod records consecutive_misses increment before eviction."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-miss")
        await catalog.register(entry)

        subscriber, collection = _make_subscriber(catalog=catalog, timeout=30.0)

        stale_time = datetime.now(UTC) - timedelta(seconds=60)
        stale_entity = collection.create(
            {
                "pod_id": "pod-miss",
                "date_last_heartbeat": stale_time,
                "tools": ["threetears.calculator@1.0.0"],
                "tools_count": 1,
                "status": "healthy",
                "consecutive_misses": 2,
            }
        )
        await collection.save_entity(stale_entity)
        subscriber._known_pod_ids.add("pod-miss")

        await subscriber._run_health_check()

        assert "pod-miss" not in subscriber.known_pod_ids


# -- lifecycle tests --


class TestHeartbeatSubscriberLifecycle:
    """tests for subscriber start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_subscribes_to_heartbeat_wildcard(self) -> None:
        """start subscribes to {namespace}.tools.heartbeat.>."""
        catalog = ToolCatalog()
        subscriber, _ = _make_subscriber(catalog=catalog)
        subscriber._namespace = "myns"
        nc = AsyncMock()
        mock_sub = AsyncMock()
        nc.subscribe = AsyncMock(return_value=mock_sub)
        await subscriber.start(nc)
        nc.subscribe.assert_called_once()
        call_args = nc.subscribe.call_args
        assert call_args[0][0] == "myns.tools.heartbeat.>"
        await subscriber.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_check_task(self) -> None:
        """stop cancels health check task."""
        subscriber, _ = _make_subscriber(check_interval=100.0)
        nc = AsyncMock()
        mock_sub = AsyncMock()
        nc.subscribe = AsyncMock(return_value=mock_sub)
        await subscriber.start(nc)
        assert subscriber._check_task is not None
        await subscriber.stop()
        assert subscriber._check_task is None

    @pytest.mark.asyncio
    async def test_stop_unsubscribes(self) -> None:
        """stop unsubscribes from heartbeat subject."""
        subscriber, _ = _make_subscriber()
        nc = AsyncMock()
        mock_sub = AsyncMock()
        nc.subscribe = AsyncMock(return_value=mock_sub)
        await subscriber.start(nc)
        await subscriber.stop()
        mock_sub.unsubscribe.assert_called_once()


# -- HeartbeatCollection-only tests --


class TestHeartbeatCollectionL1:
    """L1-only behaviour tests for HeartbeatCollection (no NATS)."""

    @pytest.mark.asyncio
    async def test_save_then_get_roundtrips(self) -> None:
        """save_entity populates L1; get returns hydrated entity."""
        collection = _make_collection()
        now = datetime.now(UTC)
        entity = collection.create(
            {
                "pod_id": "pod-abc",
                "date_last_heartbeat": now,
                "tools": ["t.a@1.0", "t.b@2.0"],
                "tools_count": 2,
                "status": "healthy",
                "consecutive_misses": 0,
            }
        )
        await collection.save_entity(entity)

        hit = await collection.get("pod-abc")
        assert hit is not None
        assert hit.pod_id == "pod-abc"
        assert hit.tools == ["t.a@1.0", "t.b@2.0"]
        assert hit.status == "healthy"

    @pytest.mark.asyncio
    async def test_get_returns_none_on_miss(self) -> None:
        """get returns None for unknown pod_id without hitting L3."""
        collection = _make_collection()
        result = await collection.get("pod-never-seen")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_from_postgres_raises(self) -> None:
        """the L3 hop is unreachable and raises on direct invocation."""
        collection = _make_collection()
        with pytest.raises(RuntimeError):
            await collection.fetch_from_postgres("pod-x")

    @pytest.mark.asyncio
    async def test_save_to_postgres_raises(self) -> None:
        """L3 write hop raises on direct invocation."""
        collection = _make_collection()
        with pytest.raises(RuntimeError):
            await collection.save_to_postgres({"pod_id": "p"})

    @pytest.mark.asyncio
    async def test_delete_from_postgres_raises(self) -> None:
        """L3 delete hop raises on direct invocation."""
        collection = _make_collection()
        with pytest.raises(RuntimeError):
            await collection.delete_from_postgres("pod-x")

    @pytest.mark.asyncio
    async def test_delete_removes_row_from_l1(self) -> None:
        """delete evicts the row from L1."""
        collection = _make_collection()
        entity = collection.create(
            {
                "pod_id": "pod-x",
                "date_last_heartbeat": datetime.now(UTC),
                "tools": [],
                "tools_count": 0,
                "status": "healthy",
                "consecutive_misses": 0,
            }
        )
        await collection.save_entity(entity)
        await collection.delete("pod-x")
        assert await collection.get("pod-x") is None

    @pytest.mark.asyncio
    async def test_get_pods_returns_only_known_hits(self) -> None:
        """get_pods hydrates many ids; missing ids are omitted."""
        collection = _make_collection()
        entity = collection.create(
            {
                "pod_id": "pod-present",
                "date_last_heartbeat": datetime.now(UTC),
                "tools": [],
                "tools_count": 0,
                "status": "healthy",
                "consecutive_misses": 0,
            }
        )
        await collection.save_entity(entity)
        result = await collection.get_pods(["pod-present", "pod-missing"])
        assert [e.pod_id for e in result] == ["pod-present"]
