"""unit tests for :class:`threetears.mcp.server.McpServer` -- dispatch + RBAC + error mapping.

constructs a minimal `McpServer` against an in-memory `ToolRegistry`,
spies the SDK server's handler to extract the framework's dispatch
result, and exercises the four error paths (unknown tool, identity
failure, authorizer raise, denied) plus the happy path.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import mcp.types as mcp_types
import pytest
from threetears.mcp.auth import Identity
from threetears.mcp.server import McpServer
from threetears.mcp.tool import McpTool, ToolRegistry


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _registry_with(tool: McpTool) -> ToolRegistry:
    """build an isolated registry with ``tool`` registered."""
    registry = ToolRegistry()
    registry.register(tool)
    return registry


def _identity_provider(*, identity: Identity) -> Any:
    """fake IdentityProvider returning a fixed identity."""
    provider = MagicMock()
    provider.identify = AsyncMock(return_value=identity)
    return provider


def _authorizer(*, allows: bool = True) -> Any:
    """fake Authorizer returning the requested decision."""
    authz = MagicMock()
    authz.allows = AsyncMock(return_value=allows)
    authz.start = AsyncMock()
    return authz


def _make_tool(
    *,
    name: str = "probe",
    permission: str = "t.probe.read",
    handler: Any | None = None,
) -> McpTool:
    """build a sample McpTool descriptor."""
    if handler is None:
        async def _h(**kwargs: Any) -> str:  # noqa: ARG001
            return "default-result"
        handler = _h
    return McpTool(
        name=name,
        description="probe tool",
        input_schema={"type": "object", "properties": {}},
        required_permission=permission,
        handler=handler,
    )


def _payload(content_list: list[mcp_types.TextContent]) -> dict[str, Any]:
    """parse the JSON envelope a framework error returns."""
    return json.loads(content_list[0].text)


# ---------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------


class TestDispatchHappyPath:
    @pytest.mark.asyncio
    async def test_allowed_dispatch_returns_handler_text(self) -> None:
        """authorizer-allowed call returns the handler's string verbatim."""
        async def handler(**kwargs: Any) -> str:
            return f"called with {kwargs}"

        tool = _make_tool(handler=handler)
        identity = Identity(principal_type="user", principal_id=uuid4())
        server = McpServer(
            name="test",
            identity_provider=_identity_provider(identity=identity),
            authorizer=_authorizer(allows=True),
            registry=_registry_with(tool),
        )
        result = await server._dispatch("probe", {"foo": "bar"})  # noqa: SLF001
        assert len(result) == 1
        assert "called with {'foo': 'bar'}" in result[0].text

    @pytest.mark.asyncio
    async def test_dict_handler_result_is_json_serialised(self) -> None:
        """dict result is rendered as pretty JSON in the text content."""
        async def handler(**_kwargs: Any) -> dict[str, Any]:
            return {"status": "ok", "count": 3}

        tool = _make_tool(handler=handler)
        identity = Identity(principal_type="user", principal_id=uuid4())
        server = McpServer(
            name="test",
            identity_provider=_identity_provider(identity=identity),
            authorizer=_authorizer(allows=True),
            registry=_registry_with(tool),
        )
        result = await server._dispatch("probe", {})  # noqa: SLF001
        assert json.loads(result[0].text) == {"status": "ok", "count": 3}


# ---------------------------------------------------------------------
# error paths -- structured envelopes, no raw exceptions
# ---------------------------------------------------------------------


class TestDispatchErrorPaths:
    @pytest.mark.asyncio
    async def test_unknown_tool_returns_structured_error(self) -> None:
        """call to a non-registered tool returns isError envelope."""
        identity = Identity(principal_type="user", principal_id=uuid4())
        server = McpServer(
            name="test",
            identity_provider=_identity_provider(identity=identity),
            authorizer=_authorizer(allows=True),
            registry=ToolRegistry(),
        )
        result = await server._dispatch("nonesuch", {})  # noqa: SLF001
        envelope = _payload(result)
        assert envelope["isError"] is True
        assert envelope["error"]["code"] == "UNKNOWN_TOOL"
        assert envelope["tool_name"] == "nonesuch"

    @pytest.mark.asyncio
    async def test_identity_resolution_failure_returns_structured_error(self) -> None:
        """IdentityProvider raise -> IDENTITY_UNAVAILABLE envelope."""
        provider = MagicMock()
        provider.identify = AsyncMock(side_effect=RuntimeError("env var missing"))
        server = McpServer(
            name="test",
            identity_provider=provider,
            authorizer=_authorizer(allows=True),
            registry=_registry_with(_make_tool()),
        )
        result = await server._dispatch("probe", {})  # noqa: SLF001
        envelope = _payload(result)
        assert envelope["isError"] is True
        assert envelope["error"]["code"] == "IDENTITY_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_authorizer_raise_denies_with_authz_error(self) -> None:
        """Authorizer raise -> AUTHZ_ERROR envelope (default-deny safety)."""
        authz = MagicMock()
        authz.allows = AsyncMock(side_effect=RuntimeError("broken"))
        identity = Identity(principal_type="user", principal_id=uuid4())
        server = McpServer(
            name="test",
            identity_provider=_identity_provider(identity=identity),
            authorizer=authz,
            registry=_registry_with(_make_tool()),
        )
        result = await server._dispatch("probe", {})  # noqa: SLF001
        envelope = _payload(result)
        assert envelope["isError"] is True
        assert envelope["error"]["code"] == "AUTHZ_ERROR"

    @pytest.mark.asyncio
    async def test_denied_dispatch_returns_permission_denied(self) -> None:
        """authorizer returns False -> PERMISSION_DENIED envelope."""
        identity = Identity(principal_type="user", principal_id=uuid4())
        server = McpServer(
            name="test",
            identity_provider=_identity_provider(identity=identity),
            authorizer=_authorizer(allows=False),
            registry=_registry_with(_make_tool()),
        )
        result = await server._dispatch("probe", {})  # noqa: SLF001
        envelope = _payload(result)
        assert envelope["isError"] is True
        assert envelope["error"]["code"] == "PERMISSION_DENIED"
        assert "t.probe.read" in envelope["error"]["message"]

    @pytest.mark.asyncio
    async def test_handler_exception_returns_handler_error(self) -> None:
        """handler raise -> HANDLER_ERROR envelope (no traceback leaked)."""
        async def handler(**_kwargs: Any) -> str:
            raise ValueError("kaboom")

        tool = _make_tool(handler=handler)
        identity = Identity(principal_type="user", principal_id=uuid4())
        server = McpServer(
            name="test",
            identity_provider=_identity_provider(identity=identity),
            authorizer=_authorizer(allows=True),
            registry=_registry_with(tool),
        )
        result = await server._dispatch("probe", {})  # noqa: SLF001
        envelope = _payload(result)
        assert envelope["isError"] is True
        assert envelope["error"]["code"] == "HANDLER_ERROR"
        assert "ValueError" in envelope["error"]["message"]
        assert "kaboom" in envelope["error"]["message"]


# ---------------------------------------------------------------------
# server lifecycle
# ---------------------------------------------------------------------


class TestServerLifecycle:
    @pytest.mark.asyncio
    async def test_start_delegates_to_authorizer_start(self) -> None:
        """McpServer.start() forwards to Authorizer.start()."""
        authz = _authorizer()
        server = McpServer(
            name="test",
            identity_provider=_identity_provider(
                identity=Identity(principal_type="user", principal_id=uuid4()),
            ),
            authorizer=authz,
            registry=ToolRegistry(),
        )
        await server.start()
        authz.start.assert_awaited_once()

    def test_empty_name_rejected(self) -> None:
        """server name must be non-empty (matches Subjects discipline)."""
        with pytest.raises(ValueError, match="non-empty"):
            McpServer(
                name="",
                identity_provider=_identity_provider(
                    identity=Identity(principal_type="user", principal_id=uuid4()),
                ),
                authorizer=_authorizer(),
            )

    def test_name_property_exposes_constructor_value(self) -> None:
        """McpServer.name returns the configured server name."""
        server = McpServer(
            name="metallm-test",
            identity_provider=_identity_provider(
                identity=Identity(principal_type="user", principal_id=uuid4()),
            ),
            authorizer=_authorizer(),
        )
        assert server.name == "metallm-test"
