"""tests for CallProxy."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint
from threetears.registry.proxy import CallProxy, ProxyCallRequest, ProxyCallResponse


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
        input_schema={"type": "object", "properties": {}},
        endpoints=[endpoint],
    )
    return result


def _make_nats_msg(
    data: bytes,
    reply: str | None = "reply.subject",
) -> MagicMock:
    """create mock NATS message.

    :param data: raw message payload bytes
    :ptype data: bytes
    :param reply: optional reply subject
    :ptype reply: str | None
    :return: mock NATS message
    :rtype: MagicMock
    """
    msg = MagicMock()
    msg.data = data
    msg.reply = reply
    return msg


def _make_call_request(
    agent_id: str = "agent-001",
    tool_name: str = "threetears.calculator",
    tool_version: str = "1.0.0",
    arguments: dict[str, Any] | None = None,
    correlation_id: str = "corr-abc-123",
) -> ProxyCallRequest:
    """create proxy call request for testing.

    :param agent_id: agent identifier
    :ptype agent_id: str
    :param tool_name: namespaced tool name
    :ptype tool_name: str
    :param tool_version: semver version string
    :ptype tool_version: str
    :param arguments: tool input parameters
    :ptype arguments: dict[str, Any] | None
    :param correlation_id: request correlation identifier
    :ptype correlation_id: str
    :return: test proxy call request
    :rtype: ProxyCallRequest
    """
    if arguments is None:
        arguments = {"expression": "2+2"}
    result = ProxyCallRequest(
        agent_id=agent_id,
        tool_name=tool_name,
        tool_version=tool_version,
        arguments=arguments,
        correlation_id=correlation_id,
    )
    return result


def _make_tool_response(
    success: bool = True,
    content: str = "result: 4",
    metadata: dict[str, Any] | None = None,
    error: str | None = None,
    correlation_id: str = "corr-abc-123",
) -> MagicMock:
    """create mock NATS reply from tool pod.

    :param success: whether tool execution succeeded
    :ptype success: bool
    :param content: result content string
    :ptype content: str
    :param metadata: optional additional metadata
    :ptype metadata: dict[str, Any] | None
    :param error: error message if execution failed
    :ptype error: str | None
    :param correlation_id: request correlation identifier
    :ptype correlation_id: str
    :return: mock NATS reply message
    :rtype: MagicMock
    """
    response = ProxyCallResponse(
        success=success,
        content=content,
        metadata=metadata,
        error=error,
        correlation_id=correlation_id,
    )
    reply = MagicMock()
    reply.data = response.model_dump_json().encode("utf-8")
    return reply


# -- successful proxy tests --


class TestCallProxySuccess:
    """tests for successful call proxying."""

    @pytest.mark.asyncio
    async def test_proxies_call_to_correct_pod(self) -> None:
        """proxy forwards call to internal subject for correct pod."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-alpha")
        await catalog.register(entry)

        proxy = CallProxy(catalog, namespace="test", timeout=5.0)
        nc = AsyncMock()
        nc.request = AsyncMock(return_value=_make_tool_response())
        await proxy.start(nc)

        request = _make_call_request()
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy._handle_call(msg)
        await asyncio.sleep(0)

        nc.request.assert_called_once()
        call_args = nc.request.call_args
        assert call_args[0][0] == "test.tools.internal.pod-alpha"
        assert call_args[1]["timeout"] == 5.0

    @pytest.mark.asyncio
    async def test_does_not_modify_arguments(self) -> None:
        """proxy forwards arguments to tool pod without modification."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-001")
        await catalog.register(entry)

        proxy = CallProxy(catalog, namespace="test")
        nc = AsyncMock()
        nc.request = AsyncMock(return_value=_make_tool_response())
        await proxy.start(nc)

        original_args = {"expression": "2+2", "precision": 4, "debug": True}
        request = _make_call_request(arguments=original_args)
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy._handle_call(msg)
        await asyncio.sleep(0)

        nc.request.assert_called_once()
        forwarded_payload = json.loads(nc.request.call_args[0][1])
        assert forwarded_payload["arguments"] == original_args

    @pytest.mark.asyncio
    async def test_does_not_modify_results(self) -> None:
        """proxy returns tool pod response without modification."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-001")
        await catalog.register(entry)

        proxy = CallProxy(catalog, namespace="test")
        tool_response = _make_tool_response(
            success=True,
            content="calculation complete: 42",
            metadata={"elapsed_ms": 150, "cached": False},
        )
        nc = AsyncMock()
        nc.request = AsyncMock(return_value=tool_response)
        await proxy.start(nc)

        request = _make_call_request()
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy._handle_call(msg)
        await asyncio.sleep(0)

        publish_calls = nc.publish.call_args_list
        assert len(publish_calls) == 1
        response_data = json.loads(publish_calls[0][0][1])
        assert response_data["success"] is True
        assert response_data["content"] == "calculation complete: 42"
        assert response_data["metadata"] == {"elapsed_ms": 150, "cached": False}

    @pytest.mark.asyncio
    async def test_correlation_id_preserved(self) -> None:
        """proxy preserves correlation_id through entire proxy chain."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-001")
        await catalog.register(entry)

        proxy = CallProxy(catalog, namespace="test")
        nc = AsyncMock()
        nc.request = AsyncMock(
            return_value=_make_tool_response(correlation_id="corr-xyz-789"),
        )
        await proxy.start(nc)

        request = _make_call_request(correlation_id="corr-xyz-789")
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy._handle_call(msg)
        await asyncio.sleep(0)

        forwarded_payload = json.loads(nc.request.call_args[0][1])
        assert forwarded_payload["correlation_id"] == "corr-xyz-789"

        response_data = json.loads(nc.publish.call_args[0][1])
        assert response_data["correlation_id"] == "corr-xyz-789"


# -- unavailable tool tests --


class TestCallProxyUnavailable:
    """tests for call proxy with unavailable tools."""

    @pytest.mark.asyncio
    async def test_returns_tool_unavailable_for_missing_tool(self) -> None:
        """proxy returns TOOL_UNAVAILABLE for tool not in catalog."""
        catalog = ToolCatalog()
        proxy = CallProxy(catalog, namespace="test")
        nc = AsyncMock()
        await proxy.start(nc)

        request = _make_call_request(
            tool_name="threetears.nonexistent",
            tool_version="1.0.0",
        )
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy._handle_call(msg)
        await asyncio.sleep(0)

        nc.publish.assert_called_once()
        response_data = json.loads(nc.publish.call_args[0][1])
        assert response_data["success"] is False
        assert response_data["error_code"] == "TOOL_UNAVAILABLE"
        assert "threetears.nonexistent@1.0.0" in response_data["error"]

    @pytest.mark.asyncio
    async def test_returns_tool_unavailable_for_unavailable_status(self) -> None:
        """proxy returns TOOL_UNAVAILABLE when routing strategy finds no available endpoint."""
        catalog = ToolCatalog()
        entry = _make_entry(status="unavailable")
        await catalog.register(entry)

        proxy = CallProxy(catalog, namespace="test")
        nc = AsyncMock()
        await proxy.start(nc)

        request = _make_call_request()
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy._handle_call(msg)
        await asyncio.sleep(0)

        nc.publish.assert_called_once()
        response_data = json.loads(nc.publish.call_args[0][1])
        assert response_data["success"] is False
        assert response_data["error_code"] == "TOOL_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_unavailable_does_not_forward_to_pod(self) -> None:
        """proxy does not forward request when tool is unavailable."""
        catalog = ToolCatalog()
        proxy = CallProxy(catalog, namespace="test")
        nc = AsyncMock()
        await proxy.start(nc)

        request = _make_call_request(
            tool_name="threetears.missing",
            tool_version="1.0.0",
        )
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy._handle_call(msg)
        await asyncio.sleep(0)

        nc.request.assert_not_called()


# -- timeout tests --


class TestCallProxyTimeout:
    """tests for call proxy timeout handling."""

    @pytest.mark.asyncio
    async def test_returns_tool_timeout_on_nats_timeout(self) -> None:
        """proxy returns TOOL_TIMEOUT when NATS request times out and resets in_flight."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-slow")
        await catalog.register(entry)

        proxy = CallProxy(catalog, namespace="test", timeout=2.0)
        nc = AsyncMock()
        nc.request = AsyncMock(side_effect=TimeoutError("request timed out"))
        await proxy.start(nc)

        request = _make_call_request()
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy._handle_call(msg)
        await asyncio.sleep(0)

        nc.publish.assert_called_once()
        response_data = json.loads(nc.publish.call_args[0][1])
        assert response_data["success"] is False
        assert response_data["error_code"] == "TOOL_TIMEOUT"
        assert "2.0" in response_data["error"]

        endpoint = entry.endpoints[0]
        assert endpoint.in_flight == 0

    @pytest.mark.asyncio
    async def test_timeout_preserves_correlation_id(self) -> None:
        """proxy preserves correlation_id in timeout error response."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-slow")
        await catalog.register(entry)

        proxy = CallProxy(catalog, namespace="test")
        nc = AsyncMock()
        nc.request = AsyncMock(side_effect=TimeoutError("timeout"))
        await proxy.start(nc)

        request = _make_call_request(correlation_id="corr-timeout-001")
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy._handle_call(msg)
        await asyncio.sleep(0)

        response_data = json.loads(nc.publish.call_args[0][1])
        assert response_data["correlation_id"] == "corr-timeout-001"

    @pytest.mark.asyncio
    async def test_default_timeout_is_120_not_30(self) -> None:
        """proxy default timeout must be 120s (platform default), not 30s."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-001")
        await catalog.register(entry)

        proxy = CallProxy(catalog, namespace="test")
        assert proxy._timeout == 120.0, (
            f"CallProxy default is {proxy._timeout}s but must be 120s. "
            f"Hardcoded 30s killed slow tools for an entire day."
        )

    @pytest.mark.asyncio
    async def test_per_tool_timeout_from_catalog(self) -> None:
        """proxy uses per-tool timeout_seconds from catalog entry when declared."""
        catalog = ToolCatalog()
        entry = CatalogEntry(
            tool_name="test.slow_wait",
            tool_version="1.0",
            full_name="test.slow_wait@1.0",
            description="a slow tool",
            input_schema={"type": "object", "properties": {}},
            timeout_seconds=300.0,
            endpoints=[ToolEndpoint(pod_id="pod-slow", status="available")],
        )
        await catalog.register(entry)

        proxy = CallProxy(catalog, namespace="test", timeout=120.0)
        nc = AsyncMock()
        nc.request = AsyncMock(return_value=_make_tool_response())
        await proxy.start(nc)

        request = _make_call_request(
            tool_name="test.slow_wait",
            tool_version="1.0",
        )
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy._handle_call(msg)
        await asyncio.sleep(0)

        nc.request.assert_called_once()
        call_kwargs = nc.request.call_args
        assert call_kwargs[1]["timeout"] == 300.0, (
            "proxy should use per-tool timeout (300s) from catalog, "
            "not proxy default (120s)"
        )

    @pytest.mark.asyncio
    async def test_falls_back_to_proxy_default_when_no_tool_timeout(self) -> None:
        """proxy uses its own default when tool does not declare timeout_seconds."""
        catalog = ToolCatalog()
        entry = CatalogEntry(
            tool_name="test.fast_tool",
            tool_version="1.0",
            full_name="test.fast_tool@1.0",
            description="a fast tool",
            input_schema={"type": "object", "properties": {}},
            timeout_seconds=None,
            endpoints=[ToolEndpoint(pod_id="pod-fast", status="available")],
        )
        await catalog.register(entry)

        proxy = CallProxy(catalog, namespace="test", timeout=120.0)
        nc = AsyncMock()
        nc.request = AsyncMock(return_value=_make_tool_response())
        await proxy.start(nc)

        request = _make_call_request(
            tool_name="test.fast_tool",
            tool_version="1.0",
        )
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy._handle_call(msg)
        await asyncio.sleep(0)

        nc.request.assert_called_once()
        call_kwargs = nc.request.call_args
        assert call_kwargs[1]["timeout"] == 120.0

    @pytest.mark.asyncio
    async def test_slow_tool_survives_with_declared_timeout(self) -> None:
        """slow_tool declaring timeout_seconds=120 must not be killed at 30s.

        this is the exact scenario that broke us: slow_tool sleeps 100s,
        proxy had a 30s default, tool got killed. with the fix, the tool's
        declared timeout (120s) is used, so the 100s sleep completes.
        """
        catalog = ToolCatalog()
        entry = CatalogEntry(
            tool_name="test.slow_wait",
            tool_version="1.0",
            full_name="test.slow_wait@1.0",
            description="waits then returns",
            input_schema={"type": "object", "properties": {}},
            timeout_seconds=120.0,
            endpoints=[ToolEndpoint(pod_id="pod-slow", status="available")],
        )
        await catalog.register(entry)

        proxy = CallProxy(catalog, namespace="test")
        nc = AsyncMock()
        nc.request = AsyncMock(return_value=_make_tool_response(
            content="waited 100 seconds successfully",
        ))
        await proxy.start(nc)

        request = _make_call_request(
            tool_name="test.slow_wait",
            tool_version="1.0",
            arguments={"seconds": 100},
        )
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy._handle_call(msg)
        await asyncio.sleep(0)

        nc.request.assert_called_once()
        call_kwargs = nc.request.call_args
        assert call_kwargs[1]["timeout"] == 120.0, (
            "slow_tool with timeout_seconds=120 must get 120s, not 30s"
        )

        response_data = json.loads(nc.publish.call_args[0][1])
        assert response_data["success"] is True
        assert "waited 100 seconds" in response_data["content"]


# -- in-flight tracking tests --


class TestCallProxyInFlightTracking:
    """tests for endpoint in-flight call count tracking."""

    @pytest.mark.asyncio
    async def test_in_flight_tracking(self) -> None:
        """proxy increments in_flight during call and decrements after."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-tracked")
        await catalog.register(entry)

        endpoint = entry.endpoints[0]
        assert endpoint.in_flight == 0

        captured_in_flight: list[int] = []

        async def capture_request(*args: Any, **kwargs: Any) -> MagicMock:
            """capture in_flight value during NATS request execution.

            :param args: positional arguments forwarded from NATS client
            :ptype args: Any
            :param kwargs: keyword arguments forwarded from NATS client
            :ptype kwargs: Any
            :return: mock NATS reply message
            :rtype: MagicMock
            """
            captured_in_flight.append(endpoint.in_flight)
            return _make_tool_response()

        proxy = CallProxy(catalog, namespace="test")
        nc = AsyncMock()
        nc.request = AsyncMock(side_effect=capture_request)
        await proxy.start(nc)

        request = _make_call_request()
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy._handle_call(msg)
        await asyncio.sleep(0)

        assert captured_in_flight == [1]
        assert endpoint.in_flight == 0

    @pytest.mark.asyncio
    async def test_in_flight_decremented_on_error(self) -> None:
        """proxy decrements in_flight even when NATS request raises."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-error")
        await catalog.register(entry)

        endpoint = entry.endpoints[0]
        assert endpoint.in_flight == 0

        proxy = CallProxy(catalog, namespace="test")
        nc = AsyncMock()
        nc.request = AsyncMock(side_effect=TimeoutError("timeout"))
        await proxy.start(nc)

        request = _make_call_request()
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy._handle_call(msg)
        await asyncio.sleep(0)

        assert endpoint.in_flight == 0


# -- routing strategy tests --


class TestCallProxyRouting:
    """tests for routing strategy integration."""

    @pytest.mark.asyncio
    async def test_routes_to_least_connections_endpoint(self) -> None:
        """proxy routes call to endpoint with fewest in-flight connections."""
        endpoint_busy = ToolEndpoint(pod_id="pod-busy", status="available", in_flight=5)
        endpoint_idle = ToolEndpoint(pod_id="pod-idle", status="available", in_flight=0)

        catalog = ToolCatalog()
        entry = CatalogEntry(
            tool_name="threetears.calculator",
            tool_version="1.0.0",
            full_name="threetears.calculator@1.0.0",
            description="test tool threetears.calculator",
            input_schema={"type": "object", "properties": {}},
            endpoints=[endpoint_busy, endpoint_idle],
        )
        await catalog.register(entry)

        proxy = CallProxy(catalog, namespace="test", timeout=5.0)
        nc = AsyncMock()
        nc.request = AsyncMock(return_value=_make_tool_response())
        await proxy.start(nc)

        request = _make_call_request()
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy._handle_call(msg)
        await asyncio.sleep(0)

        nc.request.assert_called_once()
        call_args = nc.request.call_args
        assert call_args[0][0] == "test.tools.internal.pod-idle"

    @pytest.mark.asyncio
    async def test_custom_routing_strategy(self) -> None:
        """proxy uses injected routing strategy for endpoint selection."""
        endpoint_first = ToolEndpoint(pod_id="pod-first", status="available")
        endpoint_second = ToolEndpoint(pod_id="pod-second", status="available")

        catalog = ToolCatalog()
        entry = CatalogEntry(
            tool_name="threetears.calculator",
            tool_version="1.0.0",
            full_name="threetears.calculator@1.0.0",
            description="test tool threetears.calculator",
            input_schema={"type": "object", "properties": {}},
            endpoints=[endpoint_first, endpoint_second],
        )
        await catalog.register(entry)

        strategy = MagicMock()
        strategy.select = MagicMock(return_value=endpoint_second)

        proxy = CallProxy(
            catalog, namespace="test", timeout=5.0, routing_strategy=strategy,
        )
        nc = AsyncMock()
        nc.request = AsyncMock(return_value=_make_tool_response())
        await proxy.start(nc)

        request = _make_call_request()
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy._handle_call(msg)
        await asyncio.sleep(0)

        strategy.select.assert_called_once_with(entry.endpoints)
        nc.request.assert_called_once()
        call_args = nc.request.call_args
        assert call_args[0][0] == "test.tools.internal.pod-second"


# -- lifecycle tests --


class TestCallProxyLifecycle:
    """tests for call proxy start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_subscribes_with_queue_group(self) -> None:
        """start subscribes to {namespace}.tools.call with queue group."""
        catalog = ToolCatalog()
        proxy = CallProxy(catalog, namespace="myns")
        nc = AsyncMock()
        await proxy.start(nc)
        nc.subscribe.assert_called_once()
        call_args = nc.subscribe.call_args
        assert call_args[0][0] == "myns.tools.call"
        assert call_args[1]["queue"] == "registry"

    @pytest.mark.asyncio
    async def test_stop_unsubscribes(self) -> None:
        """stop unsubscribes from call subject."""
        catalog = ToolCatalog()
        proxy = CallProxy(catalog, namespace="test")
        nc = AsyncMock()
        mock_sub = AsyncMock()
        nc.subscribe = AsyncMock(return_value=mock_sub)
        await proxy.start(nc)
        await proxy.stop()
        mock_sub.unsubscribe.assert_called_once()
