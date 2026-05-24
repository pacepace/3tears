"""unit tests for tool eligibility flags on TearsTool.

agent-tools-eligibility shard 01 (TE-01 / TE-02 / TE-03 / TE-07 /
TE-08): verify the two class attributes carry the documented
defaults, subclass overrides reach the manifest envelope on
publish, and registering a tool with both flags False emits the
operator-facing structured warning.
"""

from __future__ import annotations

import logging
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

    def __init__(self, name: str = "test.elig_stub", version: str = "1.0") -> None:
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
            description="stub tool for eligibility tests",
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


class _SkillOnlyTool(_BaseStubTool):
    """``tool_eligible=False, skill_eligible=True`` -- the wake pre-check pattern."""

    tool_eligible = False
    skill_eligible = True


class _UnifiedSurfaceTool(_BaseStubTool):
    """``tool_eligible=True, skill_eligible=True`` -- tool-shaped skill."""

    skill_eligible = True


class _NeverVisibleTool(_BaseStubTool):
    """``tool_eligible=False, skill_eligible=False`` -- triggers warning."""

    tool_eligible = False
    skill_eligible = False


class TestTearsToolDefaults:
    """TE-01 / TE-02 / TE-08: documented class-attribute defaults."""

    def test_base_class_default_tool_eligible_true(self) -> None:
        """``TearsTool.tool_eligible`` defaults to ``True`` (TE-01)."""
        assert TearsTool.tool_eligible is True

    def test_base_class_default_skill_eligible_false(self) -> None:
        """``TearsTool.skill_eligible`` defaults to ``False`` (TE-02)."""
        assert TearsTool.skill_eligible is False

    def test_subclass_without_override_inherits_defaults(self) -> None:
        """subclass that does not set either attribute inherits defaults."""
        assert _BaseStubTool.tool_eligible is True
        assert _BaseStubTool.skill_eligible is False
        assert _BaseStubTool().tool_eligible is True
        assert _BaseStubTool().skill_eligible is False

    def test_subclass_skill_only_override(self) -> None:
        """``tool_eligible=False, skill_eligible=True`` reaches the instance."""
        assert _SkillOnlyTool.tool_eligible is False
        assert _SkillOnlyTool.skill_eligible is True

    def test_subclass_unified_surface_override(self) -> None:
        """``skill_eligible=True`` does not silently flip ``tool_eligible``."""
        assert _UnifiedSurfaceTool.tool_eligible is True
        assert _UnifiedSurfaceTool.skill_eligible is True


class TestRegistrationManifestStampsFlags:
    """TE-03: every ``ToolManifestEntry`` carries the tool's flags."""

    @pytest.mark.asyncio
    async def test_publish_registration_stamps_default_flags(self) -> None:
        """default tool emits ``tool_eligible=True, skill_eligible=False``."""
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
        # publish() received one positional arg: the manifest.
        call_kwargs = mock_nc.publish.await_args.kwargs
        manifest = call_kwargs["message"]
        assert isinstance(manifest, RegistrationManifest)
        assert len(manifest.tools) == 1
        entry = manifest.tools[0]
        assert entry.tool_eligible is True
        assert entry.skill_eligible is False

    @pytest.mark.asyncio
    async def test_publish_registration_stamps_skill_only_flags(self) -> None:
        """skill-only tool emits ``tool_eligible=False, skill_eligible=True``."""
        server = ToolServer(
            agent_id=uuid7(),
            customer_id=uuid7(),
            namespace_collection=None,
            nats_url="nats://test:4222",
        )
        server.register(_SkillOnlyTool(name="test.skill_only"))
        mock_nc = AsyncMock()
        server._nc = mock_nc  # noqa: SLF001
        await server.publish_registration()
        manifest = mock_nc.publish.await_args.kwargs["message"]
        entry = manifest.tools[0]
        assert entry.tool_eligible is False
        assert entry.skill_eligible is True

    @pytest.mark.asyncio
    async def test_publish_registration_stamps_unified_surface_flags(self) -> None:
        """tool-shaped skill emits both flags True."""
        server = ToolServer(
            agent_id=uuid7(),
            customer_id=uuid7(),
            namespace_collection=None,
            nats_url="nats://test:4222",
        )
        server.register(_UnifiedSurfaceTool(name="test.unified"))
        mock_nc = AsyncMock()
        server._nc = mock_nc  # noqa: SLF001
        await server.publish_registration()
        manifest = mock_nc.publish.await_args.kwargs["message"]
        entry = manifest.tools[0]
        assert entry.tool_eligible is True
        assert entry.skill_eligible is True


class TestRegistrationWarningWhenInvisible:
    """TE-07: registering a never-visible tool emits a structured WARNING."""

    @pytest.mark.asyncio
    async def test_warning_emitted_when_both_flags_false(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """``tool_eligible=False, skill_eligible=False`` -> WARNING with mcp_name."""
        server = ToolServer(
            agent_id=uuid7(),
            customer_id=uuid7(),
            namespace_collection=None,
            nats_url="nats://test:4222",
        )
        server.register(_NeverVisibleTool(name="test.never_visible"))
        mock_nc = AsyncMock()
        server._nc = mock_nc  # noqa: SLF001
        with caplog.at_level(logging.WARNING, logger="threetears.agent.tools.server"):
            await server.publish_registration()
        warning_records = [
            r for r in caplog.records if r.levelno == logging.WARNING and "tool_eligible=False" in r.getMessage()
        ]
        assert warning_records, "expected at least one WARNING about both flags False"
        record = warning_records[0]
        # structured extra carries the tool's mcp_name so an operator
        # can find the offending registration.
        extra = getattr(record, "extra_data", None) or getattr(record, "__dict__", {}).get("extra_data")
        assert extra is not None
        assert extra.get("mcp_name") == "test.never_visible"
        assert extra.get("tool_class") == "_NeverVisibleTool"

    @pytest.mark.asyncio
    async def test_no_warning_emitted_for_tool_eligible_default(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """defaults must NOT trigger the never-visible warning."""
        server = ToolServer(
            agent_id=uuid7(),
            customer_id=uuid7(),
            namespace_collection=None,
            nats_url="nats://test:4222",
        )
        server.register(_BaseStubTool())
        mock_nc = AsyncMock()
        server._nc = mock_nc  # noqa: SLF001
        with caplog.at_level(logging.WARNING, logger="threetears.agent.tools.server"):
            await server.publish_registration()
        offending = [r for r in caplog.records if "tool_eligible=False" in r.getMessage()]
        assert not offending, "default-flag tool should not trigger the never-visible warning"

    @pytest.mark.asyncio
    async def test_no_warning_for_skill_only_pattern(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """``tool_eligible=False, skill_eligible=True`` is a valid pattern."""
        server = ToolServer(
            agent_id=uuid7(),
            customer_id=uuid7(),
            namespace_collection=None,
            nats_url="nats://test:4222",
        )
        server.register(_SkillOnlyTool(name="test.skill_only_no_warn"))
        mock_nc = AsyncMock()
        server._nc = mock_nc  # noqa: SLF001
        with caplog.at_level(logging.WARNING, logger="threetears.agent.tools.server"):
            await server.publish_registration()
        offending = [r for r in caplog.records if "tool_eligible=False" in r.getMessage()]
        assert not offending, "skill-only tool should not trigger the never-visible warning"
