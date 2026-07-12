"""unit tests for tool face flags on TearsTool.

gu-task-02 (GU-02-01 / GU-02-02 / GU-02-03 / GU-02-04): verify the
three face class attributes carry the documented defaults, subclass
overrides reach the instance, ``publish_registration`` stamps the
face flags onto every manifest entry, and a manifest JSON round-trip
preserves the authored values. Mirrors ``test_tool_eligibility.py``
exactly, one link shorter (no never-visible warning for faces yet).
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

    def __init__(self, name: str = "test.face_stub", version: str = "1.0") -> None:
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
            description="stub tool for face-flag tests",
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


class _ApiMcpTool(_BaseStubTool):
    """``face_api=True, face_mcp=True`` -- an external-reach tool."""

    face_api = True
    face_mcp = True


class TestTearsToolFaceDefaults:
    """GU-02-01 / GU-02-03: documented face class-attribute defaults."""

    def test_base_class_default_face_platform_tool_true(self) -> None:
        """``TearsTool.face_platform_tool`` defaults to ``True`` (GU-02-01)."""
        assert TearsTool.face_platform_tool is True

    def test_base_class_default_face_api_false(self) -> None:
        """``TearsTool.face_api`` defaults to ``False`` (GU-02-01)."""
        assert TearsTool.face_api is False

    def test_base_class_default_face_mcp_false(self) -> None:
        """``TearsTool.face_mcp`` defaults to ``False`` (GU-02-01)."""
        assert TearsTool.face_mcp is False

    def test_subclass_without_override_inherits_defaults(self) -> None:
        """subclass that sets no face attribute inherits defaults (GU-02-03)."""
        assert _BaseStubTool.face_platform_tool is True
        assert _BaseStubTool.face_api is False
        assert _BaseStubTool.face_mcp is False
        assert _BaseStubTool().face_platform_tool is True
        assert _BaseStubTool().face_api is False
        assert _BaseStubTool().face_mcp is False

    def test_subclass_api_mcp_override_reaches_instance(self) -> None:
        """``face_api=True, face_mcp=True`` reaches class and instance."""
        assert _ApiMcpTool.face_api is True
        assert _ApiMcpTool.face_mcp is True
        assert _ApiMcpTool.face_platform_tool is True
        assert _ApiMcpTool().face_api is True
        assert _ApiMcpTool().face_mcp is True
        assert _ApiMcpTool().face_platform_tool is True


class TestRegistrationManifestStampsFaceFlags:
    """GU-02-02: every ``ToolManifestEntry`` carries the tool's face flags."""

    @pytest.mark.asyncio
    async def test_publish_registration_stamps_default_face_flags(self) -> None:
        """default tool emits ``face_platform_tool=True, face_api=False, face_mcp=False``."""
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
        entry = manifest.tools[0]
        assert entry.face_platform_tool is True
        assert entry.face_api is False
        assert entry.face_mcp is False

    @pytest.mark.asyncio
    async def test_publish_registration_stamps_api_mcp_face_flags(self) -> None:
        """external-reach tool emits ``face_api=True, face_mcp=True``."""
        server = ToolServer(
            agent_id=uuid7(),
            customer_id=uuid7(),
            namespace_collection=None,
            nats_url="nats://test:4222",
        )
        server.register(_ApiMcpTool(name="test.api_mcp"))
        mock_nc = AsyncMock()
        server._nc = mock_nc  # noqa: SLF001
        await server.publish_registration()
        manifest = mock_nc.publish.await_args.kwargs["message"]
        entry = manifest.tools[0]
        assert entry.face_platform_tool is True
        assert entry.face_api is True
        assert entry.face_mcp is True


class TestManifestFaceFlagRoundTrip:
    """GU-02-04: manifest JSON round-trip carries the three face flags."""

    @pytest.mark.asyncio
    async def test_round_trip_preserves_face_flags(self) -> None:
        """``RegistrationManifest.model_validate_json`` preserves face flags."""
        server = ToolServer(
            agent_id=uuid7(),
            customer_id=uuid7(),
            namespace_collection=None,
            nats_url="nats://test:4222",
        )
        server.register(_ApiMcpTool(name="test.api_mcp_roundtrip"))
        mock_nc = AsyncMock()
        server._nc = mock_nc  # noqa: SLF001
        await server.publish_registration()
        manifest = mock_nc.publish.await_args.kwargs["message"]
        restored = RegistrationManifest.model_validate_json(manifest.model_dump_json())
        entry = restored.tools[0]
        assert entry.face_platform_tool is True
        assert entry.face_api is True
        assert entry.face_mcp is True
