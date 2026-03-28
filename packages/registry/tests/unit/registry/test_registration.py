"""tests for RegistrationHandler."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.agent.tools.server import RegistrationManifest, ToolManifestEntry
from threetears.registry.catalog import CatalogEntry, ToolCatalog
from threetears.registry.registration import RegistrationHandler, RegistrationResponse


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


# -- manifest validation tests --


class TestRegistrationHandlerValidation:
    """tests for manifest validation in registration handler."""

    @pytest.mark.asyncio
    async def test_rejects_malformed_json(self) -> None:
        """handler rejects message with invalid JSON payload."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test")
        nc = AsyncMock()
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
        nc = AsyncMock()
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
        nc = AsyncMock()
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


# -- conflict detection tests --


class TestRegistrationHandlerConflicts:
    """tests for tool conflict detection."""

    @pytest.mark.asyncio
    async def test_rejects_conflict_from_different_pod(self) -> None:
        """handler rejects tool already registered by different pod."""
        catalog = ToolCatalog()
        existing = CatalogEntry(
            tool_name="threetears.calculator",
            tool_version="1.0.0",
            full_name="threetears.calculator@1.0.0",
            pod_id="pod-OTHER",
            description="existing tool",
            input_schema={"type": "object"},
            status="available",
        )
        await catalog.register(existing)

        handler = RegistrationHandler(catalog, namespace="test")
        nc = AsyncMock()
        await handler.start(nc)

        manifest = _make_manifest(pod_id="pod-NEW")
        msg = _make_nats_msg(data=manifest.model_dump_json().encode("utf-8"))
        await handler._handle_registration(msg)

        nc.publish.assert_called_once()
        response_data = json.loads(nc.publish.call_args[0][1])
        assert response_data["success"] is False
        assert "conflict" in response_data["error"]
        assert "pod-OTHER" in response_data["error"]

    @pytest.mark.asyncio
    async def test_allows_reregistration_from_same_pod(self) -> None:
        """handler allows re-registration of tool from same pod."""
        catalog = ToolCatalog()
        existing = CatalogEntry(
            tool_name="threetears.calculator",
            tool_version="1.0.0",
            full_name="threetears.calculator@1.0.0",
            pod_id="pod-001",
            description="existing tool",
            input_schema={"type": "object"},
            status="available",
        )
        await catalog.register(existing)

        handler = RegistrationHandler(catalog, namespace="test")
        nc = AsyncMock()
        await handler.start(nc)

        manifest = _make_manifest(pod_id="pod-001")
        msg = _make_nats_msg(data=manifest.model_dump_json().encode("utf-8"))
        await handler._handle_registration(msg)

        nc.publish.assert_called_once()
        response_data = json.loads(nc.publish.call_args[0][1])
        assert response_data["success"] is True
        assert "threetears.calculator@1.0.0" in response_data["registered_tools"]


# -- successful registration tests --


class TestRegistrationHandlerSuccess:
    """tests for successful tool registration."""

    @pytest.mark.asyncio
    async def test_registers_single_tool(self) -> None:
        """handler registers single tool from manifest."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test")
        nc = AsyncMock()
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
        assert entry.pod_id == "pod-001"
        assert entry.status == "available"

    @pytest.mark.asyncio
    async def test_registers_multiple_tools(self) -> None:
        """handler registers all tools from manifest atomically."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test")
        nc = AsyncMock()
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
        nc = AsyncMock()
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
        nc = AsyncMock()
        await handler.start(nc)
        nc.subscribe.assert_called_once()
        call_args = nc.subscribe.call_args
        assert call_args[0][0] == "myns.tools.register"

    @pytest.mark.asyncio
    async def test_stop_unsubscribes(self) -> None:
        """stop unsubscribes from registration subject."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test")
        nc = AsyncMock()
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
