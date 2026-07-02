"""tests for multi-pod tool routing across registry components."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from threetears.agent.tools.context_envelope import CallContext

from threetears.agent.tools.server import HeartbeatMessage, RegistrationManifest, ToolManifestEntry
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.nats import IncomingMessage, RequestError, set_default_namespace
from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint
from threetears.registry.discovery import DiscoverRequest, DiscoverToolEntry, DiscoveryHandler
from threetears.registry.health import HeartbeatSubscriber
from threetears.registry.heartbeat_collection import HeartbeatCollection
from threetears.registry.l1_cache import create_registry_l1_backend
from threetears.registry.proxy import ProxyCallResponse
from threetears.registry.registration import ProbeResponse, RegistrationHandler
from threetears.registry.routing import LeastConnectionsStrategy

from ._dispatch_auth import make_authed_request, make_proxy


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    """default namespace so :class:`Subjects` builders are deterministic."""
    set_default_namespace("test")


def _build_heartbeat_subscriber(
    catalog: ToolCatalog,
    timeout: float = 30.0,
) -> tuple[HeartbeatSubscriber, HeartbeatCollection]:
    """construct a HeartbeatSubscriber + HeartbeatCollection pair for a test.

    :param catalog: tool catalog to be driven by the subscriber
    :ptype catalog: ToolCatalog
    :param timeout: liveness timeout for the sweep
    :ptype timeout: float
    :return: subscriber + collection pair wired against a fresh L1
    :rtype: tuple[HeartbeatSubscriber, HeartbeatCollection]
    """
    l1 = create_registry_l1_backend()
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    collection = HeartbeatCollection(registry, config)
    subscriber = HeartbeatSubscriber(
        catalog,
        collection,
        namespace="test",
        timeout=timeout,
    )
    return subscriber, collection


# -- helpers --


def _make_manifest(
    pod_id: str = "pod-001",
    tools: list[dict[str, Any]] | None = None,
) -> RegistrationManifest:
    """create registration manifest for testing.

    :param pod_id: pod identifier
    :ptype pod_id: str
    :param tools: optional list of tool dicts
    :ptype tools: list[dict[str, Any]] | None
    :return: test registration manifest
    :rtype: RegistrationManifest
    """
    if tools is None:
        tools = [
            {
                "name": "threetears.calculator",
                "version": "1.0.0",
                "description": "calculator tool",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]
    tool_entries = [ToolManifestEntry(**t) for t in tools]
    result = RegistrationManifest(pod_id=pod_id, tools=tool_entries)
    return result


def _make_nats_msg(
    data: bytes,
    reply: str | None = "reply.subject",
) -> IncomingMessage:
    """build a wrapper :class:`IncomingMessage` envelope.

    :param data: raw message payload bytes
    :ptype data: bytes
    :param reply: optional reply subject; ``None`` for fire-and-forget
    :ptype reply: str | None
    :return: wrapper-shaped envelope
    :rtype: IncomingMessage
    """
    return IncomingMessage(data=data, reply_subject=reply, subject="3tears.tools.register")


_DEFAULT_CORRELATION_ID = UUID("01948a00-6666-7000-8000-0000abcdef01")


# the enforce-only auth scaffolding (token+pop requests + a JWKS-wired proxy) is shared across the
# registry dispatch-test modules; see ``_dispatch_auth``. these aliases keep the test bodies reading
# the same while every dispatch is now authenticated (v0.13.9 verifies identity + pop, fail-closed).
_make_call_request = make_authed_request
_make_proxy = make_proxy


def _make_tool_response(
    success: bool = True,
    content: str = "result: 4",
    correlation_id: UUID | None = None,
) -> bytes:
    """build the bytes :meth:`NatsClient.request_raw` returns for a tool reply.

    :param success: whether tool execution succeeded
    :ptype success: bool
    :param content: result content string
    :ptype content: str
    :param correlation_id: request correlation identifier stamped on
        the echoed :class:`CallContext`; defaults to a stable UUID
    :ptype correlation_id: UUID | None
    :return: serialized response bytes
    :rtype: bytes
    """
    effective_correlation_id = correlation_id if correlation_id is not None else _DEFAULT_CORRELATION_ID
    response = ProxyCallResponse(
        success=success,
        content=content,
        context=CallContext(correlation_id=effective_correlation_id),
    )
    return response.model_dump_json().encode("utf-8")


def _make_endpoint(
    pod_id: str = "pod-001",
    status: str = "available",
    in_flight: int = 0,
) -> ToolEndpoint:
    """create tool endpoint for testing.

    :param pod_id: identifier of pod serving this tool
    :ptype pod_id: str
    :param status: availability status
    :ptype status: str
    :param in_flight: number of currently in-flight calls
    :ptype in_flight: int
    :return: test tool endpoint
    :rtype: ToolEndpoint
    """
    result = ToolEndpoint(
        pod_id=pod_id,
        status=status,
        in_flight=in_flight,
        date_last_heartbeat=datetime.now(UTC),
    )
    return result


def _make_catalog_entry(
    tool_name: str = "threetears.calculator",
    tool_version: str = "1.0.0",
    endpoints: list[ToolEndpoint] | None = None,
) -> CatalogEntry:
    """create catalog entry with endpoints for testing.

    :param tool_name: namespaced tool name
    :ptype tool_name: str
    :param tool_version: semver version string
    :ptype tool_version: str
    :param endpoints: list of tool endpoints
    :ptype endpoints: list[ToolEndpoint] | None
    :return: test catalog entry
    :rtype: CatalogEntry
    """
    if endpoints is None:
        endpoints = []
    result = CatalogEntry(
        tool_name=tool_name,
        tool_version=tool_version,
        full_name=f"{tool_name}@{tool_version}",
        description=f"test tool {tool_name}",
        input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        endpoints=endpoints,
    )
    return result


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


# -- multi-pod registration tests --


class TestMultiPodRegistration:
    """tests for multiple pods registering same tool."""

    @pytest.mark.asyncio
    async def test_two_pods_register_same_tool(self) -> None:
        """two pods register same tool; both succeed and entry has two endpoints."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test")
        nc = AsyncMock()
        await handler.start(nc)

        manifest_a = _make_manifest(pod_id="pod-A")
        msg_a = _make_nats_msg(data=manifest_a.model_dump_json().encode("utf-8"))
        await handler.handle_registration(msg_a)

        response_a = json.loads(nc.publish_reply.call_args.kwargs["message"].model_dump_json())
        assert response_a["success"] is True

        nc.reset_mock()

        manifest_b = _make_manifest(pod_id="pod-B")
        msg_b = _make_nats_msg(data=manifest_b.model_dump_json().encode("utf-8"))
        await handler.handle_registration(msg_b)

        response_b = json.loads(nc.publish_reply.call_args.kwargs["message"].model_dump_json())
        assert response_b["success"] is True

        entry = catalog.get("threetears.calculator@1.0.0")
        assert entry is not None
        assert len(entry.endpoints) == 2

        pod_ids = {ep.pod_id for ep in entry.endpoints}
        assert pod_ids == {"pod-A", "pod-B"}

    @pytest.mark.asyncio
    async def test_three_pods_register_same_tool(self) -> None:
        """three pods register same tool; entry has three endpoints."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test")
        nc = AsyncMock()
        await handler.start(nc)

        for pod_id in ("pod-A", "pod-B", "pod-C"):
            manifest = _make_manifest(pod_id=pod_id)
            msg = _make_nats_msg(data=manifest.model_dump_json().encode("utf-8"))
            await handler.handle_registration(msg)

        entry = catalog.get("threetears.calculator@1.0.0")
        assert entry is not None
        assert len(entry.endpoints) == 3

        pod_ids = {ep.pod_id for ep in entry.endpoints}
        assert pod_ids == {"pod-A", "pod-B", "pod-C"}

    @pytest.mark.asyncio
    async def test_reregistration_updates_endpoint(self) -> None:
        """pod re-registering same tool updates endpoint, does not duplicate."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test")
        nc = AsyncMock()
        await handler.start(nc)

        manifest = _make_manifest(pod_id="pod-A")
        msg = _make_nats_msg(data=manifest.model_dump_json().encode("utf-8"))
        await handler.handle_registration(msg)

        entry = catalog.get("threetears.calculator@1.0.0")
        assert entry is not None
        assert len(entry.endpoints) == 1
        first_heartbeat = entry.endpoints[0].date_last_heartbeat

        nc.reset_mock()

        manifest_again = _make_manifest(pod_id="pod-A")
        msg_again = _make_nats_msg(data=manifest_again.model_dump_json().encode("utf-8"))
        await handler.handle_registration(msg_again)

        response = json.loads(nc.publish_reply.call_args.kwargs["message"].model_dump_json())
        assert response["success"] is True

        entry_after = catalog.get("threetears.calculator@1.0.0")
        assert entry_after is not None
        assert len(entry_after.endpoints) == 1
        assert entry_after.endpoints[0].pod_id == "pod-A"
        assert entry_after.endpoints[0].date_last_heartbeat >= first_heartbeat


# -- multi-pod routing tests --


class TestMultiPodRouting:
    """tests for routing across multiple pod endpoints."""

    def test_routes_to_different_pods(self) -> None:
        """routing strategy can select from multiple endpoints with zero in_flight."""
        ep_a = _make_endpoint(pod_id="pod-A", in_flight=0)
        ep_b = _make_endpoint(pod_id="pod-B", in_flight=0)
        strategy = LeastConnectionsStrategy()

        selected = strategy.select([ep_a, ep_b])
        assert selected is not None
        assert selected.pod_id in {"pod-A", "pod-B"}

    def test_least_connections_prefers_idle_pod(self) -> None:
        """routing strategy selects idle pod over busy pod."""
        ep_a = _make_endpoint(pod_id="pod-A", in_flight=3)
        ep_b = _make_endpoint(pod_id="pod-B", in_flight=0)
        strategy = LeastConnectionsStrategy()

        selected = strategy.select([ep_a, ep_b])
        assert selected is not None
        assert selected.pod_id == "pod-B"
        assert selected.in_flight == 0

    @pytest.mark.asyncio
    async def test_in_flight_decremented_on_timeout(self) -> None:
        """in_flight counter returns to zero after call timeout."""
        catalog = ToolCatalog()
        entry = _make_catalog_entry(
            endpoints=[_make_endpoint(pod_id="pod-A", in_flight=0)],
        )
        await catalog.register(entry)

        proxy = _make_proxy(catalog, namespace="test", timeout=1.0)
        nc = AsyncMock()
        nc.request_raw = AsyncMock(side_effect=TimeoutError("timed out"))
        await proxy.start(nc)

        request = _make_call_request()
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy.handle_call(msg)

        registered_entry = catalog.get("threetears.calculator@1.0.0")
        assert registered_entry is not None
        assert registered_entry.endpoints[0].in_flight == 0

    @pytest.mark.asyncio
    async def test_in_flight_decremented_on_success(self) -> None:
        """in_flight counter returns to zero after successful call."""
        catalog = ToolCatalog()
        entry = _make_catalog_entry(
            endpoints=[_make_endpoint(pod_id="pod-A", in_flight=0)],
        )
        await catalog.register(entry)

        proxy = _make_proxy(catalog, namespace="test", timeout=5.0)
        nc = AsyncMock()
        nc.request_raw = AsyncMock(return_value=_make_tool_response())
        await proxy.start(nc)

        request = _make_call_request()
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy.handle_call(msg)

        registered_entry = catalog.get("threetears.calculator@1.0.0")
        assert registered_entry is not None
        assert registered_entry.endpoints[0].in_flight == 0


# -- multi-pod failover tests --


class TestMultiPodFailover:
    """tests for pod failure handling with surviving pods."""

    @pytest.mark.asyncio
    async def test_dead_pod_endpoint_removed_survivor_continues(self) -> None:
        """dead pod endpoint removed; surviving pod continues serving tool."""
        catalog = ToolCatalog()
        entry = _make_catalog_entry(
            endpoints=[
                _make_endpoint(pod_id="pod-A"),
                _make_endpoint(pod_id="pod-B"),
            ],
        )
        await catalog.register(entry)

        subscriber, collection = _build_heartbeat_subscriber(catalog, timeout=30.0)

        stale_time = datetime.now(UTC) - timedelta(seconds=60)
        recent_time = datetime.now(UTC) - timedelta(seconds=5)

        stale_entity = collection.create(
            {
                "pod_id": "pod-A",
                "date_last_heartbeat": stale_time,
                "tools": ["threetears.calculator@1.0.0"],
                "tools_count": 1,
                "status": "healthy",
                "consecutive_misses": 0,
            }
        )
        healthy_entity = collection.create(
            {
                "pod_id": "pod-B",
                "date_last_heartbeat": recent_time,
                "tools": ["threetears.calculator@1.0.0"],
                "tools_count": 1,
                "status": "healthy",
                "consecutive_misses": 0,
            }
        )
        await collection.save_entity(stale_entity)
        await collection.save_entity(healthy_entity)
        subscriber.track_pods({"pod-A", "pod-B"})

        await subscriber.run_health_check()

        assert "pod-A" not in subscriber.known_pod_ids
        assert "pod-B" in subscriber.known_pod_ids

        surviving_entry = catalog.get("threetears.calculator@1.0.0")
        assert surviving_entry is not None
        assert len(surviving_entry.endpoints) == 1
        assert surviving_entry.endpoints[0].pod_id == "pod-B"

        strategy = LeastConnectionsStrategy()
        selected = strategy.select(surviving_entry.endpoints)
        assert selected is not None
        assert selected.pod_id == "pod-B"

    @pytest.mark.asyncio
    async def test_all_pods_dead_removes_entry(self) -> None:
        """all pods timing out removes entire catalog entry."""
        catalog = ToolCatalog()
        entry = _make_catalog_entry(
            endpoints=[
                _make_endpoint(pod_id="pod-A"),
                _make_endpoint(pod_id="pod-B"),
            ],
        )
        await catalog.register(entry)

        subscriber, collection = _build_heartbeat_subscriber(catalog, timeout=30.0)

        stale_time = datetime.now(UTC) - timedelta(seconds=60)
        for pod_id in ("pod-A", "pod-B"):
            entity = collection.create(
                {
                    "pod_id": pod_id,
                    "date_last_heartbeat": stale_time,
                    "tools": ["threetears.calculator@1.0.0"],
                    "tools_count": 1,
                    "status": "healthy",
                    "consecutive_misses": 0,
                }
            )
            await collection.save_entity(entity)
            subscriber.track_pod(pod_id)

        await subscriber.run_health_check()

        assert subscriber.known_pod_ids == set()
        assert catalog.get("threetears.calculator@1.0.0") is None


# -- multi-pod discovery tests --


class TestMultiPodDiscovery:
    """tests for discovery of tools served by multiple pods."""

    @pytest.mark.asyncio
    async def test_discovery_returns_tool_once_with_multiple_pods(self) -> None:
        """tool with three endpoints appears once in discovery with endpoint_count=3."""
        catalog = ToolCatalog()
        entry = _make_catalog_entry(
            endpoints=[
                _make_endpoint(pod_id="pod-A"),
                _make_endpoint(pod_id="pod-B"),
                _make_endpoint(pod_id="pod-C"),
            ],
        )
        await catalog.register(entry)

        handler = DiscoveryHandler(catalog, namespace="test")
        nc = AsyncMock()
        await handler.start(nc)

        request = DiscoverRequest(
            agent_id="agent-001",
            tool_manifest=[
                DiscoverToolEntry(name="threetears.calculator", version="1.0.0"),
            ],
        )
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await handler.handle_discover(msg)

        nc.publish_reply.assert_called_once()
        response_data = json.loads(nc.publish_reply.call_args.kwargs["message"].model_dump_json())
        assert len(response_data["tools"]) == 1
        tool_result = response_data["tools"][0]
        assert tool_result["name"] == "threetears.calculator"
        assert tool_result["status"] == "available"
        assert tool_result["endpoint_count"] == 3

    @pytest.mark.asyncio
    async def test_discovery_available_if_any_endpoint_available(self) -> None:
        """tool shows as available when at least one endpoint is available."""
        catalog = ToolCatalog()
        entry = _make_catalog_entry(
            endpoints=[
                _make_endpoint(pod_id="pod-A", status="unavailable"),
                _make_endpoint(pod_id="pod-B", status="available"),
            ],
        )
        await catalog.register(entry)

        handler = DiscoveryHandler(catalog, namespace="test")
        nc = AsyncMock()
        await handler.start(nc)

        request = DiscoverRequest(
            agent_id="agent-001",
            tool_manifest=[
                DiscoverToolEntry(name="threetears.calculator", version="1.0.0"),
            ],
        )
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await handler.handle_discover(msg)

        nc.publish_reply.assert_called_once()
        response_data = json.loads(nc.publish_reply.call_args.kwargs["message"].model_dump_json())
        assert len(response_data["tools"]) == 1
        tool_result = response_data["tools"][0]
        assert tool_result["status"] == "available"
        assert tool_result["endpoint_count"] == 2
