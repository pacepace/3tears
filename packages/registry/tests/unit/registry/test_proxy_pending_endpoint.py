"""tests for CallProxy refusing to forward to pending endpoints.

a freshly registered tool pod whose reachability probe has not
yet confirmed round-trip is represented in the catalog by a
'pending' endpoint. CallProxy must refuse to forward calls to
such endpoints and return a structured TOOL_NOT_READY error
rather than letting the NATS request-reply sit until the 120s
call timeout fires.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from threetears.nats import IncomingMessage, set_default_namespace
from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint

from ._dispatch_auth import make_authed_request, make_proxy


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    """default namespace so :class:`Subjects` builders are deterministic."""
    set_default_namespace("test")


# -- helpers --


# the enforce-only auth scaffolding (token+pop requests + a JWKS-wired proxy) is shared across the
# registry dispatch-test modules; see ``_dispatch_auth``. these aliases keep the test bodies reading
# the same while every dispatch is now authenticated (v0.13.9 verifies identity + pop, fail-closed).
_make_call_request = make_authed_request
_make_proxy = make_proxy


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
    return IncomingMessage(data=data, reply_subject=reply, subject="aibots.tools.call")


def _make_entry_with_pending_endpoint(
    tool_name: str = "threetears.calculator",
    tool_version: str = "1.0.0",
    pod_id: str = "pod-pending",
) -> CatalogEntry:
    """create catalog entry whose only endpoint is pending.

    :param tool_name: namespaced tool name
    :ptype tool_name: str
    :param tool_version: version string
    :ptype tool_version: str
    :param pod_id: pod identifier
    :ptype pod_id: str
    :return: catalog entry with single pending endpoint
    :rtype: CatalogEntry
    """
    endpoint = ToolEndpoint(pod_id=pod_id, status="pending")
    result = CatalogEntry(
        tool_name=tool_name,
        tool_version=tool_version,
        full_name=f"{tool_name}@{tool_version}",
        description="pending probe",
        input_schema={"type": "object", "properties": {}},
        endpoints=[endpoint],
    )
    return result


# -- pending endpoint refusal tests --


class TestCallProxyRefusesPendingEndpoints:
    """tests for TOOL_NOT_READY response when endpoints are pending."""

    @pytest.mark.asyncio
    async def test_returns_tool_not_ready_for_pending_only_tool(self) -> None:
        """proxy returns TOOL_NOT_READY error when all endpoints are pending."""
        catalog = ToolCatalog()
        entry = _make_entry_with_pending_endpoint()
        await catalog.register(entry)

        proxy = _make_proxy(catalog, namespace="test")
        nc = AsyncMock()
        await proxy.start(nc)

        request = _make_call_request()
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy.handle_call(msg)
        await asyncio.sleep(0)

        nc.publish_reply.assert_called_once()
        response_data = json.loads(nc.publish_reply.call_args.kwargs["message"].model_dump_json())
        assert response_data["success"] is False
        assert response_data["error_code"] == "TOOL_NOT_READY"

    @pytest.mark.asyncio
    async def test_does_not_forward_to_pending_pod(self) -> None:
        """proxy must NOT issue NATS request to pending pod's internal subject."""
        catalog = ToolCatalog()
        entry = _make_entry_with_pending_endpoint(pod_id="pod-not-ready")
        await catalog.register(entry)

        proxy = _make_proxy(catalog, namespace="test")
        nc = AsyncMock()
        nc.request_raw = AsyncMock()
        await proxy.start(nc)

        request = _make_call_request()
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy.handle_call(msg)
        await asyncio.sleep(0)

        nc.request_raw.assert_not_called()

    @pytest.mark.asyncio
    async def test_tool_not_ready_preserves_correlation_id(self) -> None:
        """TOOL_NOT_READY response carries the original correlation_id."""
        correlation_id = UUID("01948a00-4444-7000-8000-0000000010ad")

        catalog = ToolCatalog()
        entry = _make_entry_with_pending_endpoint()
        await catalog.register(entry)

        proxy = _make_proxy(catalog, namespace="test")
        nc = AsyncMock()
        await proxy.start(nc)

        request = _make_call_request(correlation_id=correlation_id)
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy.handle_call(msg)
        await asyncio.sleep(0)

        response_data = json.loads(nc.publish_reply.call_args.kwargs["message"].model_dump_json())
        # ProxyCallResponse echoes correlation_id on context, not top-level
        assert response_data["context"]["correlation_id"] == str(correlation_id)

    @pytest.mark.asyncio
    async def test_mixed_pending_and_available_routes_to_available(self) -> None:
        """proxy routes to available endpoint when other endpoints are pending."""
        catalog = ToolCatalog()
        entry = CatalogEntry(
            tool_name="threetears.calculator",
            tool_version="1.0.0",
            full_name="threetears.calculator@1.0.0",
            description="mixed",
            input_schema={"type": "object", "properties": {}},
            endpoints=[
                ToolEndpoint(pod_id="pod-pending", status="pending"),
                ToolEndpoint(pod_id="pod-ready", status="available"),
            ],
        )
        await catalog.register(entry)

        correlation_id = UUID("01948a00-5555-7000-8000-00000000fa11")
        # responder echoes correlation_id inside the CallContext
        # envelope since context-task-01 removed the top-level flat
        # field from ProxyCallResponse. the wrapper's request_raw
        # returns raw bytes, so the fake just hands those bytes back.
        reply_bytes = (
            b'{"success": true, "content": "ok", "context": '
            b'{"correlation_id": "' + str(correlation_id).encode("ascii") + b'"}}'
        )

        proxy = _make_proxy(catalog, namespace="test")
        nc = AsyncMock()
        nc.request_raw = AsyncMock(return_value=reply_bytes)
        await proxy.start(nc)

        request = _make_call_request(correlation_id=correlation_id)
        msg = _make_nats_msg(data=request.model_dump_json().encode("utf-8"))
        await proxy.handle_call(msg)
        await asyncio.sleep(0)

        nc.request_raw.assert_called_once()
        call_args = nc.request_raw.call_args
        # wrapper request_raw is kw-only with typed Subject
        assert call_args.kwargs["subject"].path == "test.tools.internal.pod-ready"

    @pytest.mark.asyncio
    async def test_tool_not_ready_distinct_from_tool_unavailable(self) -> None:
        """TOOL_NOT_READY used for pending endpoints; TOOL_UNAVAILABLE for unavailable."""
        catalog = ToolCatalog()
        entry_unavail = CatalogEntry(
            tool_name="tool.down",
            tool_version="1.0",
            full_name="tool.down@1.0",
            description="down",
            input_schema={},
            endpoints=[ToolEndpoint(pod_id="pod-X", status="unavailable")],
        )
        entry_pending = _make_entry_with_pending_endpoint(
            tool_name="tool.warming",
            tool_version="1.0",
            pod_id="pod-Y",
        )
        await catalog.register(entry_unavail)
        await catalog.register(entry_pending)

        proxy = _make_proxy(catalog, namespace="test")
        nc = AsyncMock()
        await proxy.start(nc)

        # pending endpoint: TOOL_NOT_READY
        req_pending = _make_call_request(
            tool_name="tool.warming",
            tool_version="1.0",
        )
        msg_pending = _make_nats_msg(
            data=req_pending.model_dump_json().encode("utf-8"),
        )
        await proxy.handle_call(msg_pending)
        await asyncio.sleep(0)
        resp_pending = json.loads(nc.publish_reply.call_args_list[-1].kwargs["message"].model_dump_json())
        assert resp_pending["error_code"] == "TOOL_NOT_READY"

        # unavailable endpoint: TOOL_UNAVAILABLE
        req_unavail = _make_call_request(
            tool_name="tool.down",
            tool_version="1.0",
        )
        msg_unavail = _make_nats_msg(
            data=req_unavail.model_dump_json().encode("utf-8"),
        )
        await proxy.handle_call(msg_unavail)
        await asyncio.sleep(0)
        resp_unavail = json.loads(nc.publish_reply.call_args_list[-1].kwargs["message"].model_dump_json())
        assert resp_unavail["error_code"] == "TOOL_UNAVAILABLE"
