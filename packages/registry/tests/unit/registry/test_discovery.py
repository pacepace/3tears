"""tests for DiscoveryHandler."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.nats import IncomingMessage, set_default_namespace
from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint
from threetears.registry.discovery import (
    DiscoverRequest,
    DiscoverToolEntry,
    DiscoveryHandler,
)


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    """default namespace so :class:`Subjects` builders are deterministic."""
    set_default_namespace("test")


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
    endpoint = ToolEndpoint(pod_id=pod_id, status=status)
    result = CatalogEntry(
        tool_name=tool_name,
        tool_version=tool_version,
        full_name=f"{tool_name}@{tool_version}",
        description=f"test tool {tool_name}",
        input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        output_schema={"type": "object", "properties": {"result": {"type": "number"}}},
        endpoints=[endpoint],
    )
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
    return IncomingMessage(data=data, reply_subject=reply, subject="aibots.tools.discover")


def _make_discover_request(
    agent_id: str = "agent-001",
    tools: list[dict[str, str]] | None = None,
) -> DiscoverRequest:
    """create discovery request for testing.

    :param agent_id: agent identifier
    :ptype agent_id: str
    :param tools: optional list of tool dicts with name and version
    :ptype tools: list[dict[str, str]] | None
    :return: test discovery request
    :rtype: DiscoverRequest
    """
    if tools is None:
        tools = [{"name": "threetears.calculator", "version": "1.0.0"}]
    tool_entries = [DiscoverToolEntry(**t) for t in tools]
    result = DiscoverRequest(agent_id=agent_id, tool_manifest=tool_entries)
    return result


# -- available tool resolution tests --


class TestDiscoveryAvailable:
    """tests for discovery of available tools."""

    @pytest.mark.asyncio
    async def test_returns_schema_for_available_tool(self) -> None:
        """discovery returns full schema for available tool in catalog."""
        catalog = ToolCatalog()
        entry = _make_entry()
        await catalog.register(entry)

        handler = DiscoveryHandler(catalog, namespace="test")
        nc = AsyncMock()
        await handler.start(nc)

        request = _make_discover_request()
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await handler.handle_discover(msg)

        nc.publish_reply.assert_called_once()
        response_data = json.loads(nc.publish_reply.call_args.kwargs["message"].model_dump_json())
        assert response_data["agent_id"] == "agent-001"
        assert len(response_data["tools"]) == 1
        tool_result = response_data["tools"][0]
        assert tool_result["name"] == "threetears.calculator"
        assert tool_result["version"] == "1.0.0"
        assert tool_result["status"] == "available"
        assert tool_result["description"] == "test tool threetears.calculator"
        assert tool_result["input_schema"] == {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        }
        assert tool_result["output_schema"] == {
            "type": "object",
            "properties": {"result": {"type": "number"}},
        }
        assert tool_result["endpoint_count"] == 1


# -- unavailable tool resolution tests --


class TestDiscoveryUnavailable:
    """tests for discovery of unavailable or missing tools."""

    @pytest.mark.asyncio
    async def test_returns_unavailable_for_missing_tool(self) -> None:
        """discovery returns unavailable status for tool not in catalog."""
        catalog = ToolCatalog()
        handler = DiscoveryHandler(catalog, namespace="test")
        nc = AsyncMock()
        await handler.start(nc)

        request = _make_discover_request(
            tools=[{"name": "threetears.nonexistent", "version": "1.0.0"}],
        )
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await handler.handle_discover(msg)

        nc.publish_reply.assert_called_once()
        response_data = json.loads(nc.publish_reply.call_args.kwargs["message"].model_dump_json())
        assert len(response_data["tools"]) == 1
        tool_result = response_data["tools"][0]
        assert tool_result["name"] == "threetears.nonexistent"
        assert tool_result["version"] == "1.0.0"
        assert tool_result["status"] == "unavailable"
        assert tool_result["endpoint_count"] == 0

    @pytest.mark.asyncio
    async def test_returns_unavailable_for_tool_with_unavailable_status(self) -> None:
        """discovery returns unavailable for tool registered but not available."""
        catalog = ToolCatalog()
        entry = _make_entry(status="unavailable")
        await catalog.register(entry)

        handler = DiscoveryHandler(catalog, namespace="test")
        nc = AsyncMock()
        await handler.start(nc)

        request = _make_discover_request()
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await handler.handle_discover(msg)

        nc.publish_reply.assert_called_once()
        response_data = json.loads(nc.publish_reply.call_args.kwargs["message"].model_dump_json())
        assert len(response_data["tools"]) == 1
        assert response_data["tools"][0]["status"] == "unavailable"
        assert response_data["tools"][0]["endpoint_count"] == 0


# -- mixed manifest resolution tests --


class TestDiscoveryMixed:
    """tests for discovery with partially available manifest."""

    @pytest.mark.asyncio
    async def test_returns_mixed_results(self) -> None:
        """discovery returns mixed available/unavailable for partial manifest."""
        catalog = ToolCatalog()
        await catalog.register(
            _make_entry(
                tool_name="threetears.calculator",
                tool_version="1.0.0",
            )
        )

        handler = DiscoveryHandler(catalog, namespace="test")
        nc = AsyncMock()
        await handler.start(nc)

        request = _make_discover_request(
            tools=[
                {"name": "threetears.calculator", "version": "1.0.0"},
                {"name": "threetears.dictionary", "version": "2.0.0"},
            ],
        )
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await handler.handle_discover(msg)

        nc.publish_reply.assert_called_once()
        response_data = json.loads(nc.publish_reply.call_args.kwargs["message"].model_dump_json())
        assert len(response_data["tools"]) == 2

        by_name = {t["name"]: t for t in response_data["tools"]}
        assert by_name["threetears.calculator"]["status"] == "available"
        assert by_name["threetears.calculator"]["endpoint_count"] == 1
        assert by_name["threetears.dictionary"]["status"] == "unavailable"
        assert by_name["threetears.dictionary"]["endpoint_count"] == 0


# -- empty manifest tests --


class TestDiscoveryEmpty:
    """tests for discovery with empty manifest."""

    @pytest.mark.asyncio
    async def test_empty_manifest_returns_empty_list(self) -> None:
        """discovery returns empty tools list for empty manifest."""
        catalog = ToolCatalog()
        handler = DiscoveryHandler(catalog, namespace="test")
        nc = AsyncMock()
        await handler.start(nc)

        request = _make_discover_request(tools=[])
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await handler.handle_discover(msg)

        nc.publish_reply.assert_called_once()
        response_data = json.loads(nc.publish_reply.call_args.kwargs["message"].model_dump_json())
        assert response_data["tools"] == []


# -- multi-endpoint tests --


class TestDiscoveryMultiEndpoint:
    """tests for discovery with tools served by multiple endpoints."""

    @pytest.mark.asyncio
    async def test_returns_correct_endpoint_count_for_multi_pod_tool(self) -> None:
        """discovery returns endpoint_count matching number of registered endpoints."""
        catalog = ToolCatalog()
        entry = CatalogEntry(
            tool_name="threetears.calculator",
            tool_version="1.0.0",
            full_name="threetears.calculator@1.0.0",
            description="test tool threetears.calculator",
            input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
            output_schema={"type": "object", "properties": {"result": {"type": "number"}}},
            endpoints=[
                ToolEndpoint(pod_id="pod-001", status="available"),
                ToolEndpoint(pod_id="pod-002", status="available"),
                ToolEndpoint(pod_id="pod-003", status="available"),
            ],
        )
        await catalog.register(entry)

        handler = DiscoveryHandler(catalog, namespace="test")
        nc = AsyncMock()
        await handler.start(nc)

        request = _make_discover_request()
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await handler.handle_discover(msg)

        nc.publish_reply.assert_called_once()
        response_data = json.loads(nc.publish_reply.call_args.kwargs["message"].model_dump_json())
        assert len(response_data["tools"]) == 1
        tool_result = response_data["tools"][0]
        assert tool_result["status"] == "available"
        assert tool_result["endpoint_count"] == 3

    @pytest.mark.asyncio
    async def test_discover_all_returns_tools_once_with_multiple_endpoints(self) -> None:
        """discover-all returns each tool once with correct endpoint_count."""
        catalog = ToolCatalog()
        entry = CatalogEntry(
            tool_name="threetears.calculator",
            tool_version="1.0.0",
            full_name="threetears.calculator@1.0.0",
            description="test tool threetears.calculator",
            input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
            output_schema={"type": "object", "properties": {"result": {"type": "number"}}},
            endpoints=[
                ToolEndpoint(pod_id="pod-001", status="available"),
                ToolEndpoint(pod_id="pod-002", status="available"),
            ],
        )
        await catalog.register(entry)

        handler = DiscoveryHandler(catalog, namespace="test")
        nc = AsyncMock()
        await handler.start(nc)

        request = _make_discover_request(tools=[])
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await handler.handle_discover(msg)

        nc.publish_reply.assert_called_once()
        response_data = json.loads(nc.publish_reply.call_args.kwargs["message"].model_dump_json())
        assert len(response_data["tools"]) == 1
        tool_result = response_data["tools"][0]
        assert tool_result["name"] == "threetears.calculator"
        assert tool_result["version"] == "1.0.0"
        assert tool_result["status"] == "available"
        assert tool_result["endpoint_count"] == 2


# -- lifecycle tests --


class TestDiscoveryLifecycle:
    """tests for discovery handler start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_subscribes_with_queue_group(self) -> None:
        """start subscribes to {namespace}.tools.discover with queue group."""
        set_default_namespace("myns")
        catalog = ToolCatalog()
        handler = DiscoveryHandler(catalog, namespace="myns")
        nc = AsyncMock()
        await handler.start(nc)
        nc.subscribe.assert_called_once()
        call_args = nc.subscribe.call_args
        # wrapper subscribe is kw-only with typed Subject
        assert call_args.kwargs["subject"].path == "myns.tools.discover"
        assert call_args.kwargs["queue"] == "registry"

    @pytest.mark.asyncio
    async def test_stop_unsubscribes(self) -> None:
        """stop unsubscribes from discovery subject through the wrapper."""
        catalog = ToolCatalog()
        handler = DiscoveryHandler(catalog, namespace="test")
        nc = AsyncMock()
        mock_sub = MagicMock()
        nc.subscribe = AsyncMock(return_value=mock_sub)
        await handler.start(nc)
        await handler.stop()
        nc.unsubscribe.assert_called_once_with(mock_sub)
