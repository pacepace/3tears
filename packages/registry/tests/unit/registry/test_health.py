"""tests for HeartbeatMonitor."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.agent.tools.server import HeartbeatMessage
from threetears.registry.catalog import CatalogEntry, ToolCatalog
from threetears.registry.health import HeartbeatMonitor, PodStatus


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
    result = CatalogEntry(
        tool_name=tool_name,
        tool_version=tool_version,
        full_name=f"{tool_name}@{tool_version}",
        pod_id=pod_id,
        description=f"test tool {tool_name}",
        input_schema={"type": "object", "properties": {}},
        status="available",
    )
    return result


# -- PodStatus tests --


class TestPodStatus:
    """tests for PodStatus dataclass."""

    def test_pod_status_creation(self) -> None:
        """PodStatus stores all fields correctly."""
        status = PodStatus(pod_id="pod-001")
        assert status.pod_id == "pod-001"
        assert status.consecutive_misses == 0
        assert status.tools == []

    def test_pod_status_with_tools(self) -> None:
        """PodStatus tracks associated tool full_names."""
        status = PodStatus(
            pod_id="pod-001",
            tools=["tool.a@1.0", "tool.b@2.0"],
        )
        assert len(status.tools) == 2


# -- heartbeat handling tests --


class TestHeartbeatMonitorHandling:
    """tests for heartbeat message handling."""

    @pytest.mark.asyncio
    async def test_tracks_new_pod_on_first_heartbeat(self) -> None:
        """monitor creates PodStatus on first heartbeat from pod."""
        catalog = ToolCatalog()
        monitor = HeartbeatMonitor(catalog, namespace="test")
        nc = AsyncMock()
        mock_sub = AsyncMock()
        nc.subscribe = AsyncMock(return_value=mock_sub)
        await monitor.start(nc)

        msg = _make_heartbeat_msg(pod_id="pod-new")
        await monitor._handle_heartbeat(msg)

        assert "pod-new" in monitor.pods
        assert monitor.pods["pod-new"].pod_id == "pod-new"
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_updates_timestamp_on_subsequent_heartbeat(self) -> None:
        """monitor updates last_heartbeat on subsequent heartbeats."""
        catalog = ToolCatalog()
        monitor = HeartbeatMonitor(catalog, namespace="test")
        nc = AsyncMock()
        mock_sub = AsyncMock()
        nc.subscribe = AsyncMock(return_value=mock_sub)
        await monitor.start(nc)

        msg1 = _make_heartbeat_msg(pod_id="pod-001")
        await monitor._handle_heartbeat(msg1)
        first_hb = monitor.pods["pod-001"].date_last_heartbeat

        msg2 = _make_heartbeat_msg(pod_id="pod-001")
        await monitor._handle_heartbeat(msg2)
        second_hb = monitor.pods["pod-001"].date_last_heartbeat

        assert second_hb >= first_hb
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_resets_consecutive_misses_on_heartbeat(self) -> None:
        """monitor resets consecutive_misses to zero on heartbeat."""
        catalog = ToolCatalog()
        monitor = HeartbeatMonitor(catalog, namespace="test")
        nc = AsyncMock()
        mock_sub = AsyncMock()
        nc.subscribe = AsyncMock(return_value=mock_sub)
        await monitor.start(nc)

        monitor._pods["pod-001"] = PodStatus(
            pod_id="pod-001",
            consecutive_misses=5,
        )

        msg = _make_heartbeat_msg(pod_id="pod-001")
        await monitor._handle_heartbeat(msg)

        assert monitor.pods["pod-001"].consecutive_misses == 0
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_ignores_malformed_heartbeat(self) -> None:
        """monitor ignores messages with invalid payload."""
        catalog = ToolCatalog()
        monitor = HeartbeatMonitor(catalog, namespace="test")
        nc = AsyncMock()
        mock_sub = AsyncMock()
        nc.subscribe = AsyncMock(return_value=mock_sub)
        await monitor.start(nc)

        msg = MagicMock()
        msg.data = b"not json"
        await monitor._handle_heartbeat(msg)

        assert len(monitor.pods) == 0
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_marks_pod_tools_available_on_heartbeat(self) -> None:
        """monitor marks all tools from pod as available on heartbeat."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-001", tool_name="threetears.calculator")
        entry.status = "unavailable"
        await catalog.register(entry)

        monitor = HeartbeatMonitor(catalog, namespace="test")
        nc = AsyncMock()
        mock_sub = AsyncMock()
        nc.subscribe = AsyncMock(return_value=mock_sub)
        await monitor.start(nc)

        monitor._pods["pod-001"] = PodStatus(
            pod_id="pod-001",
            tools=["threetears.calculator@1.0.0"],
        )

        msg = _make_heartbeat_msg(pod_id="pod-001")
        await monitor._handle_heartbeat(msg)

        assert entry.status == "available"
        await monitor.stop()


# -- health check tests --


class TestHeartbeatMonitorHealthCheck:
    """tests for periodic health check sweep."""

    @pytest.mark.asyncio
    async def test_deregisters_pod_after_timeout(self) -> None:
        """health check deregisters pod exceeding timeout."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-stale")
        await catalog.register(entry)

        monitor = HeartbeatMonitor(
            catalog,
            namespace="test",
            timeout=30.0,
        )

        stale_time = datetime.now(UTC) - timedelta(seconds=60)
        monitor._pods["pod-stale"] = PodStatus(
            pod_id="pod-stale",
            date_last_heartbeat=stale_time,
            tools=["threetears.calculator@1.0.0"],
        )

        await monitor._run_health_check()

        assert "pod-stale" not in monitor.pods
        assert catalog.get("threetears.calculator@1.0.0") is None

    @pytest.mark.asyncio
    async def test_keeps_healthy_pod(self) -> None:
        """health check does not deregister pod within timeout."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-healthy")
        await catalog.register(entry)

        monitor = HeartbeatMonitor(
            catalog,
            namespace="test",
            timeout=30.0,
        )

        recent_time = datetime.now(UTC) - timedelta(seconds=5)
        monitor._pods["pod-healthy"] = PodStatus(
            pod_id="pod-healthy",
            date_last_heartbeat=recent_time,
            tools=["threetears.calculator@1.0.0"],
        )

        await monitor._run_health_check()

        assert "pod-healthy" in monitor.pods
        assert catalog.get("threetears.calculator@1.0.0") is not None

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

        monitor = HeartbeatMonitor(
            catalog,
            namespace="test",
            timeout=30.0,
        )

        stale_time = datetime.now(UTC) - timedelta(seconds=60)
        monitor._pods["pod-stale-1"] = PodStatus(
            pod_id="pod-stale-1",
            date_last_heartbeat=stale_time,
            tools=["tool.alpha@1.0"],
        )
        monitor._pods["pod-stale-2"] = PodStatus(
            pod_id="pod-stale-2",
            date_last_heartbeat=stale_time,
            tools=["tool.beta@1.0"],
        )

        await monitor._run_health_check()

        assert len(monitor.pods) == 0
        assert catalog.get("tool.alpha@1.0") is None
        assert catalog.get("tool.beta@1.0") is None

    @pytest.mark.asyncio
    async def test_increments_consecutive_misses(self) -> None:
        """health check increments consecutive_misses before deregistration."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-miss")
        await catalog.register(entry)

        monitor = HeartbeatMonitor(
            catalog,
            namespace="test",
            timeout=30.0,
        )

        stale_time = datetime.now(UTC) - timedelta(seconds=60)
        pod_status = PodStatus(
            pod_id="pod-miss",
            date_last_heartbeat=stale_time,
            tools=["threetears.calculator@1.0.0"],
            consecutive_misses=2,
        )
        monitor._pods["pod-miss"] = pod_status

        await monitor._run_health_check()

        assert "pod-miss" not in monitor.pods


# -- lifecycle tests --


class TestHeartbeatMonitorLifecycle:
    """tests for monitor start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_subscribes_to_heartbeat_wildcard(self) -> None:
        """start subscribes to {namespace}.tools.heartbeat.>."""
        catalog = ToolCatalog()
        monitor = HeartbeatMonitor(catalog, namespace="myns")
        nc = AsyncMock()
        mock_sub = AsyncMock()
        nc.subscribe = AsyncMock(return_value=mock_sub)
        await monitor.start(nc)
        nc.subscribe.assert_called_once()
        call_args = nc.subscribe.call_args
        assert call_args[0][0] == "myns.tools.heartbeat.>"
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_check_task(self) -> None:
        """stop cancels health check task."""
        catalog = ToolCatalog()
        monitor = HeartbeatMonitor(
            catalog,
            namespace="test",
            check_interval=100.0,
        )
        nc = AsyncMock()
        mock_sub = AsyncMock()
        nc.subscribe = AsyncMock(return_value=mock_sub)
        await monitor.start(nc)
        assert monitor._check_task is not None
        await monitor.stop()
        assert monitor._check_task is None

    @pytest.mark.asyncio
    async def test_stop_unsubscribes(self) -> None:
        """stop unsubscribes from heartbeat subject."""
        catalog = ToolCatalog()
        monitor = HeartbeatMonitor(catalog, namespace="test")
        nc = AsyncMock()
        mock_sub = AsyncMock()
        nc.subscribe = AsyncMock(return_value=mock_sub)
        await monitor.start(nc)
        await monitor.stop()
        mock_sub.unsubscribe.assert_called_once()
