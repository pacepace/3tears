"""unit tests for the ``requires_confirmation`` flag on TearsTool.

HITL exploit-approval gate (pentest chunk 4): verify the
``requires_confirmation`` class attribute carries the documented default
(``False``), a subclass override reaches the instance,
``publish_registration`` stamps the flag onto every manifest entry, and a
manifest JSON round-trip preserves the authored value. Mirrors
``test_tool_face_flags.py``: the flag rides the registration manifest so the
agent-side tool wrapper can gate the tool at call time behind the
``HumanInTheLoopMiddleware`` interrupt.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid7

import pytest

from threetears.agent.tools.base_tool import (
    MCPToolDefinition,
    TearsTool,
    ToolResult,
)
from threetears.agent.tools.server import RegistrationManifest, ToolServer


class _BaseStubTool(TearsTool):
    """minimal TearsTool concrete subclass for these tests."""

    def __init__(self, name: str = "test.confirm_stub", version: str = "1.0") -> None:
        """initialize stub.

        :param name: namespaced tool name
        :ptype name: str
        :param version: semver string
        :ptype version: str
        """
        self._name = name
        self._version = version

    async def execute(self, **kwargs: Any) -> ToolResult:
        """no-op execute body.

        :param kwargs: ignored
        :ptype kwargs: Any
        :return: trivial result
        :rtype: ToolResult
        """
        return ToolResult(success=True, content="")

    def mcp_schema(self) -> MCPToolDefinition:
        """return the canonical schema for this stub.

        :return: schema with empty-object input
        :rtype: MCPToolDefinition
        """
        return MCPToolDefinition(
            name=self._name,
            version=self._version,
            description="stub tool for requires_confirmation tests",
            input_schema={"type": "object", "properties": {}},
        )

    def mcp_name(self) -> str:
        """return the stub's mcp name.

        :return: name
        :rtype: str
        """
        return self._name

    def mcp_version(self) -> str:
        """return the stub's version.

        :return: version
        :rtype: str
        """
        return self._version


class _IntrusiveTool(_BaseStubTool):
    """``requires_confirmation=True`` -- an exploit/intrusive tool."""

    requires_confirmation = True


class TestTearsToolRequiresConfirmationDefault:
    """documented ``requires_confirmation`` class-attribute default."""

    def test_base_class_default_is_false(self) -> None:
        """``TearsTool.requires_confirmation`` defaults to ``False``."""
        assert TearsTool.requires_confirmation is False

    def test_subclass_without_override_inherits_default(self) -> None:
        """subclass that sets no attribute inherits ``False``."""
        assert _BaseStubTool.requires_confirmation is False
        assert _BaseStubTool().requires_confirmation is False

    def test_subclass_override_reaches_instance(self) -> None:
        """``requires_confirmation=True`` reaches class and instance."""
        assert _IntrusiveTool.requires_confirmation is True
        assert _IntrusiveTool().requires_confirmation is True


class TestRegistrationManifestStampsRequiresConfirmation:
    """every ``ToolManifestEntry`` carries the tool's ``requires_confirmation``."""

    @pytest.mark.asyncio
    async def test_publish_registration_stamps_default_false(self) -> None:
        """default tool emits ``requires_confirmation=False``."""
        server = ToolServer(
            agent_id=uuid7(),
            customer_id=uuid7(),
            namespace_collection=None,
            nats_url="nats://test:4222",
        )
        server.register(_BaseStubTool())
        mock_nc = AsyncMock()
        server._nc = mock_nc  # noqa: SLF001
        await server.publish_registration()
        manifest = mock_nc.publish.await_args.kwargs["message"]
        assert isinstance(manifest, RegistrationManifest)
        assert len(manifest.tools) == 1
        assert manifest.tools[0].requires_confirmation is False

    @pytest.mark.asyncio
    async def test_publish_registration_stamps_true_for_intrusive_tool(self) -> None:
        """intrusive tool emits ``requires_confirmation=True``."""
        server = ToolServer(
            agent_id=uuid7(),
            customer_id=uuid7(),
            namespace_collection=None,
            nats_url="nats://test:4222",
        )
        server.register(_IntrusiveTool(name="test.intrusive"))
        mock_nc = AsyncMock()
        server._nc = mock_nc  # noqa: SLF001
        await server.publish_registration()
        manifest = mock_nc.publish.await_args.kwargs["message"]
        assert manifest.tools[0].requires_confirmation is True


class TestManifestRequiresConfirmationRoundTrip:
    """manifest JSON round-trip carries the ``requires_confirmation`` flag."""

    @pytest.mark.asyncio
    async def test_round_trip_preserves_requires_confirmation(self) -> None:
        """``RegistrationManifest.model_validate_json`` preserves the flag."""
        server = ToolServer(
            agent_id=uuid7(),
            customer_id=uuid7(),
            namespace_collection=None,
            nats_url="nats://test:4222",
        )
        server.register(_IntrusiveTool(name="test.intrusive_roundtrip"))
        mock_nc = AsyncMock()
        server._nc = mock_nc  # noqa: SLF001
        await server.publish_registration()
        manifest = mock_nc.publish.await_args.kwargs["message"]
        restored = RegistrationManifest.model_validate_json(manifest.model_dump_json())
        assert restored.tools[0].requires_confirmation is True
