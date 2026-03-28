"""tests for CallProxy."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.registry.catalog import CatalogEntry, ToolCatalog
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

        nc.publish.assert_called_once()
        response_data = json.loads(nc.publish.call_args[0][1])
        assert response_data["success"] is False
        assert response_data["error_code"] == "TOOL_UNAVAILABLE"
        assert "threetears.nonexistent@1.0.0" in response_data["error"]

    @pytest.mark.asyncio
    async def test_returns_tool_unavailable_for_unavailable_status(self) -> None:
        """proxy returns TOOL_UNAVAILABLE for tool with unavailable status."""
        catalog = ToolCatalog()
        entry = _make_entry(status="unavailable")
        await catalog.register(entry)

        proxy = CallProxy(catalog, namespace="test")
        nc = AsyncMock()
        await proxy.start(nc)

        request = _make_call_request()
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy._handle_call(msg)

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

        nc.request.assert_not_called()


# -- timeout tests --


class TestCallProxyTimeout:
    """tests for call proxy timeout handling."""

    @pytest.mark.asyncio
    async def test_returns_tool_timeout_on_nats_timeout(self) -> None:
        """proxy returns TOOL_TIMEOUT when NATS request times out."""
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

        nc.publish.assert_called_once()
        response_data = json.loads(nc.publish.call_args[0][1])
        assert response_data["success"] is False
        assert response_data["error_code"] == "TOOL_TIMEOUT"
        assert "2.0" in response_data["error"]

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

        response_data = json.loads(nc.publish.call_args[0][1])
        assert response_data["correlation_id"] == "corr-timeout-001"


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
