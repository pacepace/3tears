"""tests for RegistrationHandler."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.agent.tools.server import RegistrationManifest, ToolManifestEntry
from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint
from threetears.registry.registration import (
    ProbeResponse,
    RegistrationHandler,
    RegistrationResponse,
)


# -- helpers --


def _make_probe_reply(pod_id: str, ready: bool = True) -> MagicMock:
    """build a NATS reply mock carrying a valid ProbeResponse payload.

    :param pod_id: pod identifier echoed into the ack
    :ptype pod_id: str
    :param ready: readiness flag asserted by the pod
    :ptype ready: bool
    :return: mock NATS message with JSON-encoded ProbeResponse in ``.data``
    :rtype: MagicMock
    """
    reply = MagicMock()
    reply.data = ProbeResponse(pod_id=pod_id, ready=ready).model_dump_json().encode("utf-8")
    return reply


def _make_registry_nc() -> AsyncMock:
    """build an AsyncMock NATS client that replies to every probe subject.

    probe subjects follow ``<ns>.tools.probe.<pod_id>``; the mock's
    ``request`` method parses pod_id out of the subject and echoes it
    back in a valid :class:`ProbeResponse`. tests never have to wire
    probe replies per pod_id.

    :return: configured AsyncMock NATS client
    :rtype: AsyncMock
    """
    async def _reply(subject: str, *_args: Any, **_kwargs: Any) -> MagicMock:
        pod_id = subject.rsplit(".", 1)[-1]
        return _make_probe_reply(pod_id)

    nc = AsyncMock()
    nc.request = AsyncMock(side_effect=_reply)
    return nc


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


def _make_entry(
    tool_name: str = "threetears.calculator",
    tool_version: str = "1.0.0",
    pod_id: str = "pod-001",
    status: str = "available",
) -> CatalogEntry:
    """create catalog entry with single endpoint for testing.

    :param tool_name: namespaced tool name
    :ptype tool_name: str
    :param tool_version: semver version string
    :ptype tool_version: str
    :param pod_id: pod identifier for endpoint
    :ptype pod_id: str
    :param status: endpoint availability status
    :ptype status: str
    :return: test catalog entry with one endpoint
    :rtype: CatalogEntry
    """
    endpoint = ToolEndpoint(
        pod_id=pod_id,
        status=status,
        in_flight=0,
    )
    result = CatalogEntry(
        tool_name=tool_name,
        tool_version=tool_version,
        full_name=f"{tool_name}@{tool_version}",
        description=f"{tool_name} tool",
        input_schema={"type": "object"},
        endpoints=[endpoint],
    )
    return result


# -- manifest validation tests --


class TestRegistrationHandlerValidation:
    """tests for manifest validation in registration handler."""

    @pytest.mark.asyncio
    async def test_rejects_malformed_json(self) -> None:
        """handler rejects message with invalid JSON payload."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test")
        nc = _make_registry_nc()
        await handler.start(nc)

        msg = _make_nats_msg(data=b"not json")
        await handler._handle_registration(msg)

        nc.publish.assert_called_once()
        response_data = json.loads(nc.publish.call_args[0][1])
        assert response_data["success"] is False
        assert "malformed" in response_data["error"]

    @pytest.mark.asyncio
    async def test_rejects_empty_pod_id(self) -> None:
        """handler rejects manifest with empty pod_id."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test")
        nc = _make_registry_nc()
        await handler.start(nc)

        manifest = _make_manifest(pod_id="")
        msg = _make_nats_msg(data=manifest.model_dump_json().encode("utf-8"))
        await handler._handle_registration(msg)

        nc.publish.assert_called_once()
        response_data = json.loads(nc.publish.call_args[0][1])
        assert response_data["success"] is False
        assert "pod_id" in response_data["error"]

    @pytest.mark.asyncio
    async def test_rejects_empty_tools_list(self) -> None:
        """handler rejects manifest with empty tools list."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test")
        nc = _make_registry_nc()
        await handler.start(nc)

        manifest = RegistrationManifest(pod_id="pod-001", tools=[])
        msg = _make_nats_msg(
            data=manifest.model_dump_json().encode("utf-8"),
        )
        await handler._handle_registration(msg)

        nc.publish.assert_called_once()
        response_data = json.loads(nc.publish.call_args[0][1])
        assert response_data["success"] is False
        assert "tools" in response_data["error"]


# -- multi-pod registration tests --


class TestRegistrationHandlerMultiPod:
    """tests for additive multi-pod registration."""

    @pytest.mark.asyncio
    async def test_allows_registration_from_different_pod(self) -> None:
        """handler allows same tool@version from different pod."""
        catalog = ToolCatalog()
        existing = _make_entry(pod_id="pod-OTHER")
        await catalog.register(existing)

        handler = RegistrationHandler(catalog, namespace="test")
        nc = _make_registry_nc()
        await handler.start(nc)

        manifest = _make_manifest(pod_id="pod-NEW")
        msg = _make_nats_msg(data=manifest.model_dump_json().encode("utf-8"))
        await handler._handle_registration(msg)

        nc.publish.assert_called_once()
        response_data = json.loads(nc.publish.call_args[0][1])
        assert response_data["success"] is True
        assert "threetears.calculator@1.0.0" in response_data["registered_tools"]

    @pytest.mark.asyncio
    async def test_allows_reregistration_from_same_pod(self) -> None:
        """handler allows re-registration of tool from same pod."""
        catalog = ToolCatalog()
        existing = _make_entry(pod_id="pod-001")
        await catalog.register(existing)

        handler = RegistrationHandler(catalog, namespace="test")
        nc = _make_registry_nc()
        await handler.start(nc)

        manifest = _make_manifest(pod_id="pod-001")
        msg = _make_nats_msg(data=manifest.model_dump_json().encode("utf-8"))
        await handler._handle_registration(msg)

        nc.publish.assert_called_once()
        response_data = json.loads(nc.publish.call_args[0][1])
        assert response_data["success"] is True
        assert "threetears.calculator@1.0.0" in response_data["registered_tools"]

    @pytest.mark.asyncio
    async def test_second_pod_adds_endpoint(self) -> None:
        """registering from second pod adds endpoint to existing entry."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test")
        nc = _make_registry_nc()
        await handler.start(nc)

        manifest_a = _make_manifest(pod_id="pod-A")
        msg_a = _make_nats_msg(data=manifest_a.model_dump_json().encode("utf-8"))
        await handler._handle_registration(msg_a)

        manifest_b = _make_manifest(pod_id="pod-B")
        msg_b = _make_nats_msg(data=manifest_b.model_dump_json().encode("utf-8"))
        await handler._handle_registration(msg_b)

        entry = catalog.get("threetears.calculator@1.0.0")
        assert entry is not None
        assert len(entry.endpoints) == 2
        pod_ids = {ep.pod_id for ep in entry.endpoints}
        assert pod_ids == {"pod-A", "pod-B"}


# -- successful registration tests --


class TestRegistrationHandlerSuccess:
    """tests for successful tool registration."""

    @pytest.mark.asyncio
    async def test_registers_single_tool(self) -> None:
        """handler registers single tool from manifest."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test")
        nc = _make_registry_nc()
        await handler.start(nc)

        manifest = _make_manifest()
        msg = _make_nats_msg(data=manifest.model_dump_json().encode("utf-8"))
        await handler._handle_registration(msg)

        nc.publish.assert_called_once()
        response_data = json.loads(nc.publish.call_args[0][1])
        assert response_data["success"] is True
        assert response_data["pod_id"] == "pod-001"
        assert "threetears.calculator@1.0.0" in response_data["registered_tools"]

        entry = catalog.get("threetears.calculator@1.0.0")
        assert entry is not None
        assert len(entry.endpoints) == 1
        assert entry.endpoints[0].pod_id == "pod-001"
        assert entry.status == "available"

    @pytest.mark.asyncio
    async def test_registers_multiple_tools(self) -> None:
        """handler registers all tools from manifest atomically."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test")
        nc = _make_registry_nc()
        await handler.start(nc)

        manifest = _make_manifest(
            pod_id="pod-multi",
            tools=[
                {
                    "name": "threetears.calculator",
                    "version": "1.0.0",
                    "description": "calculator",
                    "input_schema": {"type": "object"},
                },
                {
                    "name": "threetears.dictionary",
                    "version": "1.0.0",
                    "description": "dictionary",
                    "input_schema": {"type": "object"},
                },
            ],
        )
        msg = _make_nats_msg(data=manifest.model_dump_json().encode("utf-8"))
        await handler._handle_registration(msg)

        response_data = json.loads(nc.publish.call_args[0][1])
        assert response_data["success"] is True
        assert len(response_data["registered_tools"]) == 2
        assert catalog.get("threetears.calculator@1.0.0") is not None
        assert catalog.get("threetears.dictionary@1.0.0") is not None

    @pytest.mark.asyncio
    async def test_no_reply_when_no_reply_subject(self) -> None:
        """handler does not publish response when no reply subject."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test")
        nc = _make_registry_nc()
        await handler.start(nc)

        manifest = _make_manifest()
        msg = _make_nats_msg(
            data=manifest.model_dump_json().encode("utf-8"),
            reply=None,
        )
        await handler._handle_registration(msg)

        nc.publish.assert_not_called()
        assert catalog.get("threetears.calculator@1.0.0") is not None


# -- lifecycle tests --


class TestRegistrationHandlerLifecycle:
    """tests for handler start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_subscribes_to_register_subject(self) -> None:
        """start subscribes to {namespace}.tools.register."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="myns")
        nc = _make_registry_nc()
        await handler.start(nc)
        nc.subscribe.assert_called_once()
        call_args = nc.subscribe.call_args
        assert call_args[0][0] == "myns.tools.register"

    @pytest.mark.asyncio
    async def test_stop_unsubscribes(self) -> None:
        """stop unsubscribes from registration subject."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test")
        nc = _make_registry_nc()
        mock_sub = AsyncMock()
        nc.subscribe = AsyncMock(return_value=mock_sub)
        await handler.start(nc)
        await handler.stop()
        mock_sub.unsubscribe.assert_called_once()


# -- wire format tests --


class TestRegistrationResponse:
    """tests for RegistrationResponse model."""

    def test_success_response_serialization(self) -> None:
        """RegistrationResponse serializes success correctly."""
        resp = RegistrationResponse(
            success=True,
            pod_id="pod-001",
            registered_tools=["tool.a@1.0", "tool.b@2.0"],
        )
        data = json.loads(resp.model_dump_json())
        assert data["success"] is True
        assert data["pod_id"] == "pod-001"
        assert len(data["registered_tools"]) == 2
        assert data["error"] is None

    def test_error_response_serialization(self) -> None:
        """RegistrationResponse serializes error correctly."""
        resp = RegistrationResponse(
            success=False,
            pod_id="pod-fail",
            error="conflict detected",
        )
        data = json.loads(resp.model_dump_json())
        assert data["success"] is False
        assert data["error"] == "conflict detected"
        assert data["registered_tools"] == []
