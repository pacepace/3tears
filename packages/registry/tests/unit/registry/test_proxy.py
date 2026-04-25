"""tests for CallProxy."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from threetears.agent.tools.context_envelope import CallContext

from threetears.nats import IncomingMessage, RequestError, set_default_namespace
from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint
from threetears.registry.auth import AllowAllAuthorizer
from threetears.registry.proxy import CallProxy, ProxyCallRequest, ProxyCallResponse


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
        input_schema={"type": "object", "properties": {}},
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
    return IncomingMessage(data=data, reply_subject=reply)


# static UUID used as the default correlation id in test fixtures.
# correlation_id lives on CallContext.correlation_id (UUID) since
# context-task-01 removed the top-level flat string field from
# ProxyCallRequest / CallRequest.
_DEFAULT_CORRELATION_ID = UUID("01948a00-0000-7000-8000-0000000abc12")


_DEFAULT_AGENT_ID = UUID("01948a00-aaaa-7000-8000-000000a9e777")


def _make_call_request(
    agent_id: UUID | None = None,
    tool_name: str = "threetears.calculator",
    tool_version: str = "1.0.0",
    arguments: dict[str, Any] | None = None,
    correlation_id: UUID | None = None,
) -> ProxyCallRequest:
    """create proxy call request for testing.

    :param agent_id: agent identifier stamped on the carried
        :class:`CallContext`; defaults to a stable UUID
    :ptype agent_id: UUID | None
    :param tool_name: namespaced tool name
    :ptype tool_name: str
    :param tool_version: semver version string
    :ptype tool_version: str
    :param arguments: tool input parameters
    :ptype arguments: dict[str, Any] | None
    :param correlation_id: request correlation identifier stamped on
        the carried :class:`CallContext`; defaults to a stable UUID
    :ptype correlation_id: UUID | None
    :return: test proxy call request
    :rtype: ProxyCallRequest
    """
    if arguments is None:
        arguments = {"expression": "2+2"}
    effective_correlation_id = (
        correlation_id if correlation_id is not None else _DEFAULT_CORRELATION_ID
    )
    effective_agent_id = (
        agent_id if agent_id is not None else _DEFAULT_AGENT_ID
    )
    result = ProxyCallRequest(
        tool_name=tool_name,
        tool_version=tool_version,
        arguments=arguments,
        context=CallContext(
            correlation_id=effective_correlation_id,
            agent_id=effective_agent_id,
        ),
    )
    return result


def _make_tool_response(
    success: bool = True,
    content: str = "result: 4",
    metadata: dict[str, Any] | None = None,
    error: str | None = None,
    correlation_id: UUID | None = None,
) -> bytes:
    """build the bytes :meth:`NatsClient.request_raw` would return for a tool reply.

    the canonical wrapper's ``request_raw`` returns ``bytes``
    (formerly the proxy unpacked ``msg.data`` from a nats-py reply
    message), so test fixtures hand the same shape to AsyncMock-based
    test doubles.

    :param success: whether tool execution succeeded
    :ptype success: bool
    :param content: result content string
    :ptype content: str
    :param metadata: optional additional metadata
    :ptype metadata: dict[str, Any] | None
    :param error: error message if execution failed
    :ptype error: str | None
    :param correlation_id: request correlation identifier stamped on
        the echoed :class:`CallContext`; defaults to a stable UUID
    :ptype correlation_id: UUID | None
    :return: serialized response bytes
    :rtype: bytes
    """
    effective_correlation_id = (
        correlation_id if correlation_id is not None else _DEFAULT_CORRELATION_ID
    )
    response = ProxyCallResponse(
        success=success,
        content=content,
        metadata=metadata,
        error=error,
        context=CallContext(correlation_id=effective_correlation_id),
    )
    return response.model_dump_json().encode("utf-8")


# -- successful proxy tests --


class TestCallProxySuccess:
    """tests for successful call proxying."""

    @pytest.mark.asyncio
    async def test_proxies_call_to_correct_pod(self) -> None:
        """proxy forwards call to internal subject for correct pod."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-alpha")
        await catalog.register(entry)

        proxy = CallProxy(catalog, AllowAllAuthorizer(), namespace="test", timeout=5.0)
        nc = AsyncMock()
        nc.request_raw = AsyncMock(return_value=_make_tool_response())
        await proxy.start(nc)

        request = _make_call_request()
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy.handle_call(msg)
        await asyncio.sleep(0)

        nc.request_raw.assert_called_once()
        call_args = nc.request_raw.call_args
        # wrapper request_raw is kw-only with typed Subject + timedelta
        assert call_args.kwargs["subject"].path == "test.tools.internal.pod-alpha"
        assert call_args.kwargs["timeout"].total_seconds() == 5.0

    @pytest.mark.asyncio
    async def test_does_not_modify_arguments(self) -> None:
        """proxy forwards arguments to tool pod without modification."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-001")
        await catalog.register(entry)

        proxy = CallProxy(catalog, AllowAllAuthorizer(), namespace="test")
        nc = AsyncMock()
        nc.request_raw = AsyncMock(return_value=_make_tool_response())
        await proxy.start(nc)

        original_args = {"expression": "2+2", "precision": 4, "debug": True}
        request = _make_call_request(arguments=original_args)
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy.handle_call(msg)
        await asyncio.sleep(0)

        nc.request_raw.assert_called_once()
        forwarded_payload = json.loads(nc.request_raw.call_args.kwargs["payload"])
        assert forwarded_payload["arguments"] == original_args

    @pytest.mark.asyncio
    async def test_does_not_modify_results(self) -> None:
        """proxy returns tool pod response without modification."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-001")
        await catalog.register(entry)

        proxy = CallProxy(catalog, AllowAllAuthorizer(), namespace="test")
        tool_response = _make_tool_response(
            success=True,
            content="calculation complete: 42",
            metadata={"elapsed_ms": 150, "cached": False},
        )
        nc = AsyncMock()
        nc.request_raw = AsyncMock(return_value=tool_response)
        await proxy.start(nc)

        request = _make_call_request()
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy.handle_call(msg)
        await asyncio.sleep(0)

        publish_calls = nc.publish_reply.call_args_list
        assert len(publish_calls) == 1
        response_data = json.loads(publish_calls[0].kwargs["message"].model_dump_json())
        assert response_data["success"] is True
        assert response_data["content"] == "calculation complete: 42"
        assert response_data["metadata"] == {"elapsed_ms": 150, "cached": False}

    @pytest.mark.asyncio
    async def test_correlation_id_preserved(self) -> None:
        """proxy preserves correlation_id through entire proxy chain.

        correlation_id rides on ``context.correlation_id`` (UUID); the
        proxy forwards the context verbatim onto the inner
        :class:`CallRequest` and stringifies it onto the
        :class:`ProxyCallResponse` echoed to the caller.
        """
        correlation_id = UUID("01948a00-1111-7000-8000-00000000cafe")

        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-001")
        await catalog.register(entry)

        proxy = CallProxy(catalog, AllowAllAuthorizer(), namespace="test")
        nc = AsyncMock()
        nc.request_raw = AsyncMock(
            return_value=_make_tool_response(correlation_id=correlation_id),
        )
        await proxy.start(nc)

        request = _make_call_request(correlation_id=correlation_id)
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy.handle_call(msg)
        await asyncio.sleep(0)

        forwarded_payload = json.loads(nc.request_raw.call_args.kwargs["payload"])
        # CallRequest no longer has a top-level correlation_id; it
        # rides on context.correlation_id instead
        assert "correlation_id" not in forwarded_payload
        assert forwarded_payload["context"]["correlation_id"] == str(correlation_id)

        response_data = json.loads(
            nc.publish_reply.call_args.kwargs["message"].model_dump_json()
        )
        # ProxyCallResponse also moved correlation_id onto context
        assert "correlation_id" not in response_data
        assert response_data["context"]["correlation_id"] == str(correlation_id)


# -- unavailable tool tests --


class TestCallProxyUnavailable:
    """tests for call proxy with unavailable tools."""

    @pytest.mark.asyncio
    async def test_returns_tool_unavailable_for_missing_tool(self) -> None:
        """proxy returns TOOL_UNAVAILABLE for tool not in catalog."""
        catalog = ToolCatalog()
        proxy = CallProxy(catalog, AllowAllAuthorizer(), namespace="test")
        nc = AsyncMock()
        await proxy.start(nc)

        request = _make_call_request(
            tool_name="threetears.nonexistent",
            tool_version="1.0.0",
        )
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy.handle_call(msg)
        await asyncio.sleep(0)

        nc.publish_reply.assert_called_once()
        response_data = json.loads(
            nc.publish_reply.call_args.kwargs["message"].model_dump_json()
        )
        assert response_data["success"] is False
        assert response_data["error_code"] == "TOOL_UNAVAILABLE"
        assert "threetears.nonexistent@1.0.0" in response_data["error"]

    @pytest.mark.asyncio
    async def test_returns_tool_unavailable_for_unavailable_status(self) -> None:
        """proxy returns TOOL_UNAVAILABLE when routing strategy finds no available endpoint."""
        catalog = ToolCatalog()
        entry = _make_entry(status="unavailable")
        await catalog.register(entry)

        proxy = CallProxy(catalog, AllowAllAuthorizer(), namespace="test")
        nc = AsyncMock()
        await proxy.start(nc)

        request = _make_call_request()
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy.handle_call(msg)
        await asyncio.sleep(0)

        nc.publish_reply.assert_called_once()
        response_data = json.loads(
            nc.publish_reply.call_args.kwargs["message"].model_dump_json()
        )
        assert response_data["success"] is False
        assert response_data["error_code"] == "TOOL_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_unavailable_does_not_forward_to_pod(self) -> None:
        """proxy does not forward request when tool is unavailable."""
        catalog = ToolCatalog()
        proxy = CallProxy(catalog, AllowAllAuthorizer(), namespace="test")
        nc = AsyncMock()
        await proxy.start(nc)

        request = _make_call_request(
            tool_name="threetears.missing",
            tool_version="1.0.0",
        )
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy.handle_call(msg)
        await asyncio.sleep(0)

        nc.request_raw.assert_not_called()


# -- timeout tests --


class TestCallProxyTimeout:
    """tests for call proxy timeout handling."""

    @pytest.mark.asyncio
    async def test_returns_tool_timeout_on_nats_timeout(self) -> None:
        """proxy returns TOOL_TIMEOUT when NATS request times out and resets in_flight."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-slow")
        await catalog.register(entry)

        proxy = CallProxy(catalog, AllowAllAuthorizer(), namespace="test", timeout=2.0)
        nc = AsyncMock()
        nc.request_raw = AsyncMock(side_effect=TimeoutError("request timed out"))
        await proxy.start(nc)

        request = _make_call_request()
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy.handle_call(msg)
        await asyncio.sleep(0)

        nc.publish_reply.assert_called_once()
        response_data = json.loads(
            nc.publish_reply.call_args.kwargs["message"].model_dump_json()
        )
        assert response_data["success"] is False
        assert response_data["error_code"] == "TOOL_TIMEOUT"
        assert "2.0" in response_data["error"]

        endpoint = entry.endpoints[0]
        assert endpoint.in_flight == 0

    @pytest.mark.asyncio
    async def test_timeout_preserves_correlation_id(self) -> None:
        """proxy preserves correlation_id in timeout error response."""
        correlation_id = UUID("01948a00-2222-7000-8000-000000012345")

        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-slow")
        await catalog.register(entry)

        proxy = CallProxy(catalog, AllowAllAuthorizer(), namespace="test")
        nc = AsyncMock()
        nc.request_raw = AsyncMock(side_effect=TimeoutError("timeout"))
        await proxy.start(nc)

        request = _make_call_request(correlation_id=correlation_id)
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy.handle_call(msg)
        await asyncio.sleep(0)

        response_data = json.loads(
            nc.publish_reply.call_args.kwargs["message"].model_dump_json()
        )
        assert response_data["context"]["correlation_id"] == str(correlation_id)

    @pytest.mark.asyncio
    async def test_default_timeout_is_120_not_30(self) -> None:
        """proxy default timeout must be 120s (platform default), not 30s."""
        catalog = ToolCatalog()
        entry = _make_entry(pod_id="pod-001")
        await catalog.register(entry)

        proxy = CallProxy(catalog, AllowAllAuthorizer(), namespace="test")
        assert proxy.timeout == 120.0, (
            f"CallProxy default is {proxy.timeout}s but must be 120s. "
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

        proxy = CallProxy(catalog, AllowAllAuthorizer(), namespace="test", timeout=120.0)
        nc = AsyncMock()
        nc.request_raw = AsyncMock(return_value=_make_tool_response())
        await proxy.start(nc)

        request = _make_call_request(
            tool_name="test.slow_wait",
            tool_version="1.0",
        )
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy.handle_call(msg)
        await asyncio.sleep(0)

        nc.request_raw.assert_called_once()
        call_kwargs = nc.request_raw.call_args
        # timeout is now a timedelta (kw-only on the wrapper)
        assert call_kwargs.kwargs["timeout"].total_seconds() == 300.0, (
            "proxy should use per-tool timeout (300s) from catalog, not proxy default (120s)"
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

        proxy = CallProxy(catalog, AllowAllAuthorizer(), namespace="test", timeout=120.0)
        nc = AsyncMock()
        nc.request_raw = AsyncMock(return_value=_make_tool_response())
        await proxy.start(nc)

        request = _make_call_request(
            tool_name="test.fast_tool",
            tool_version="1.0",
        )
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy.handle_call(msg)
        await asyncio.sleep(0)

        nc.request_raw.assert_called_once()
        call_kwargs = nc.request_raw.call_args
        assert call_kwargs.kwargs["timeout"].total_seconds() == 120.0

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

        proxy = CallProxy(catalog, AllowAllAuthorizer(), namespace="test")
        nc = AsyncMock()
        nc.request_raw = AsyncMock(
            return_value=_make_tool_response(
                content="waited 100 seconds successfully",
            )
        )
        await proxy.start(nc)

        request = _make_call_request(
            tool_name="test.slow_wait",
            tool_version="1.0",
            arguments={"seconds": 100},
        )
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy.handle_call(msg)
        await asyncio.sleep(0)

        nc.request_raw.assert_called_once()
        call_kwargs = nc.request_raw.call_args
        assert call_kwargs.kwargs["timeout"].total_seconds() == 120.0, "slow_tool with timeout_seconds=120 must get 120s, not 30s"

        response_data = json.loads(
            nc.publish_reply.call_args.kwargs["message"].model_dump_json()
        )
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

        async def capture_request(*args: Any, **kwargs: Any) -> bytes:
            """capture in_flight value during NATS request execution.

            :param args: positional arguments forwarded from NATS wrapper
            :ptype args: Any
            :param kwargs: keyword arguments forwarded from NATS wrapper
            :ptype kwargs: Any
            :return: serialized response bytes (matches request_raw shape)
            :rtype: bytes
            """
            captured_in_flight.append(endpoint.in_flight)
            return _make_tool_response()

        proxy = CallProxy(catalog, AllowAllAuthorizer(), namespace="test")
        nc = AsyncMock()
        nc.request_raw = AsyncMock(side_effect=capture_request)
        await proxy.start(nc)

        request = _make_call_request()
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy.handle_call(msg)
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

        proxy = CallProxy(catalog, AllowAllAuthorizer(), namespace="test")
        nc = AsyncMock()
        nc.request_raw = AsyncMock(side_effect=TimeoutError("timeout"))
        await proxy.start(nc)

        request = _make_call_request()
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy.handle_call(msg)
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

        proxy = CallProxy(catalog, AllowAllAuthorizer(), namespace="test", timeout=5.0)
        nc = AsyncMock()
        nc.request_raw = AsyncMock(return_value=_make_tool_response())
        await proxy.start(nc)

        request = _make_call_request()
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy.handle_call(msg)
        await asyncio.sleep(0)

        nc.request_raw.assert_called_once()
        call_args = nc.request_raw.call_args
        assert call_args.kwargs["subject"].path == "test.tools.internal.pod-idle"

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

        proxy = CallProxy(catalog, AllowAllAuthorizer(),
            namespace="test",
            timeout=5.0,
            routing_strategy=strategy,
        )
        nc = AsyncMock()
        nc.request_raw = AsyncMock(return_value=_make_tool_response())
        await proxy.start(nc)

        request = _make_call_request()
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy.handle_call(msg)
        await asyncio.sleep(0)

        strategy.select.assert_called_once_with(entry.endpoints)
        nc.request_raw.assert_called_once()
        call_args = nc.request_raw.call_args
        # wrapper request_raw is kw-only with typed Subject
        assert call_args.kwargs["subject"].path == "test.tools.internal.pod-second"


# -- lifecycle tests --


class TestCallProxyLifecycle:
    """tests for call proxy start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_subscribes_with_queue_group(self) -> None:
        """start subscribes to {namespace}.tools.call with queue group."""
        set_default_namespace("myns")
        catalog = ToolCatalog()
        proxy = CallProxy(catalog, AllowAllAuthorizer(), namespace="myns")
        nc = AsyncMock()
        await proxy.start(nc)
        nc.subscribe.assert_called_once()
        call_args = nc.subscribe.call_args
        # wrapper subscribe is kw-only with typed Subject
        assert call_args.kwargs["subject"].path == "myns.tools.call"
        assert call_args.kwargs["queue"] == "registry"

    @pytest.mark.asyncio
    async def test_stop_unsubscribes(self) -> None:
        """stop unsubscribes from call subject through the wrapper."""
        catalog = ToolCatalog()
        proxy = CallProxy(catalog, AllowAllAuthorizer(), namespace="test")
        nc = AsyncMock()
        mock_sub = MagicMock()
        nc.subscribe = AsyncMock(return_value=mock_sub)
        await proxy.start(nc)
        await proxy.stop()
        nc.unsubscribe.assert_called_once_with(mock_sub)
