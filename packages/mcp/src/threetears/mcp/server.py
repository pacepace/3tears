"""``McpServer`` -- thin wrapper over the official ``mcp.server.Server``.

owns three things on top of the SDK:

1. tool registration: pulls :class:`McpTool` instances from a
   :class:`ToolRegistry` and registers them with the underlying SDK
   server so the client sees them at ``tools/list`` time.
2. RBAC gating: every dispatch flows through
   :meth:`Authorizer.allows(identity, tool.required_permission)`
   before the handler runs. denied dispatches return a structured
   MCP error envelope (per the MCP spec) -- not a Python exception
   in the response body.
3. error mapping: handler exceptions are caught at the framework
   border, logged with context, and returned to the client as
   structured ``isError=True`` content. nothing leaks past the
   border as a raw traceback.

stdio is the v1 transport. v2 HTTP slots in by adding a
``serve_http`` method alongside :meth:`serve_stdio`; the rest of
the framework (RBAC, identity, error mapping) is unchanged.

logging discipline: stdio servers MUST NOT write to stdout / stderr.
the framework configures nothing on its own -- callers wire logging
to a file or to NATS via ``threetears.observe.configure_logging``
BEFORE calling :meth:`serve_stdio`. an enforcement test guards
against any module under ``threetears.mcp`` writing to
``sys.stdout`` / ``sys.stderr`` directly.
"""

from __future__ import annotations

import json
from typing import Any

import mcp.types as mcp_types
from mcp.server import Server as _SdkServer
from mcp.server.stdio import stdio_server as _stdio_server
from threetears.observe import get_logger

from threetears.mcp.auth import Authorizer, IdentityProvider
from threetears.mcp.tool import ToolRegistry, get_default_registry

__all__ = [
    "McpServer",
]

log = get_logger(__name__)


class McpServer:
    """RBAC-gated MCP server wrapping the official SDK server.

    composition not inheritance: the SDK's :class:`mcp.server.Server`
    handles the protocol-level concerns (tools/list, tools/call,
    initialize handshake, notifications). this class owns the
    framework-level concerns (RBAC, identity, error mapping) and
    delegates protocol to the SDK.

    :param name: server name advertised to the MCP client at
        initialization
    :ptype name: str
    :param identity_provider: resolves the calling identity for
        every dispatch (v1 returns one identity per server lifetime)
    :ptype identity_provider: IdentityProvider
    :param authorizer: evaluates ``(identity, required_permission)``
        before each tool dispatch; framework default is
        :class:`LocalGrantAuthorizer`
    :ptype authorizer: Authorizer
    :param registry: tool registry to expose; defaults to the
        module-level singleton populated by :func:`register_tool`
    :ptype registry: ToolRegistry | None
    """

    def __init__(
        self,
        *,
        name: str,
        identity_provider: IdentityProvider,
        authorizer: Authorizer,
        registry: ToolRegistry | None = None,
    ) -> None:
        """capture deps; no I/O until :meth:`serve_stdio`.

        :param name: server name string
        :ptype name: str
        :param identity_provider: identity resolver
        :ptype identity_provider: IdentityProvider
        :param authorizer: permission evaluator
        :ptype authorizer: Authorizer
        :param registry: tool registry; defaults to module-level
        :ptype registry: ToolRegistry | None
        :return: nothing
        :rtype: None
        :raises ValueError: when ``name`` is empty
        """
        if not name:
            raise ValueError("server name must be non-empty")
        self._name = name
        self._identity_provider = identity_provider
        self._authorizer = authorizer
        self._registry = registry if registry is not None else get_default_registry()
        self._sdk_server: _SdkServer = _SdkServer(name)
        self._wire_handlers()

    @property
    def name(self) -> str:
        """advertised server name.

        :return: server name
        :rtype: str
        """
        return self._name

    def _wire_handlers(self) -> None:
        """register the ``list_tools`` + ``call_tool`` SDK handlers.

        idempotent in the sense that constructing two :class:`McpServer`
        instances each get their own SDK server (no cross-pollination).

        :return: nothing
        :rtype: None
        """

        @self._sdk_server.list_tools()
        async def _list_tools() -> list[mcp_types.Tool]:
            """return the registered tool inventory in registration order."""
            tools = []
            for tool in self._registry.all():
                tools.append(
                    mcp_types.Tool(
                        name=tool.name,
                        description=tool.description,
                        inputSchema=tool.input_schema,
                    ),
                )
            return tools

        @self._sdk_server.call_tool()
        async def _call_tool(name: str, arguments: dict[str, Any]) -> list[mcp_types.TextContent]:
            """dispatch one tool call; RBAC gate + error mapping."""
            return await self._dispatch(name, arguments)

    async def _dispatch(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> list[mcp_types.TextContent]:
        """resolve identity, gate via authorizer, run handler, map errors.

        the gate fires BEFORE the handler so denied callers cannot
        observe handler-side state (timing, log output, etc.). all
        framework errors -- unknown tool, denied permission, handler
        exception -- return as ``isError=True`` text content per the
        MCP spec, not as raised exceptions to the SDK transport.

        :param tool_name: requested tool name from the MCP client
        :ptype tool_name: str
        :param arguments: validated input arguments
        :ptype arguments: dict[str, Any]
        :return: list with a single :class:`TextContent`; for errors
            the content text is a JSON envelope with ``isError=True``
            and an ``error`` field carrying the structured detail
        :rtype: list[mcp_types.TextContent]
        """
        tool = self._registry.get(tool_name)
        if tool is None:
            log.warning(
                "MCP dispatch: unknown tool",
                extra={"extra_data": {"tool_name": tool_name}},
            )
            return self._error_content(
                tool_name=tool_name,
                code="UNKNOWN_TOOL",
                message=f"no tool named {tool_name!r} registered",
            )

        try:
            identity = await self._identity_provider.identify()
        except Exception:
            log.warning(
                "MCP dispatch: identity resolution failed",
                exc_info=True,
                extra={"extra_data": {"tool_name": tool_name}},
            )
            return self._error_content(
                tool_name=tool_name,
                code="IDENTITY_UNAVAILABLE",
                message="caller identity could not be resolved",
            )

        try:
            allowed = await self._authorizer.allows(identity, tool.required_permission)
        except Exception:
            log.warning(
                "MCP dispatch: authorizer raised; denying by default",
                exc_info=True,
                extra={"extra_data": {
                    "tool_name": tool_name,
                    "principal_id": str(identity.principal_id),
                    "required_permission": tool.required_permission,
                }},
            )
            return self._error_content(
                tool_name=tool_name,
                code="AUTHZ_ERROR",
                message="authorizer evaluation failed",
            )

        if not allowed:
            log.info(
                "MCP dispatch: denied",
                extra={"extra_data": {
                    "tool_name": tool_name,
                    "principal_id": str(identity.principal_id),
                    "required_permission": tool.required_permission,
                }},
            )
            return self._error_content(
                tool_name=tool_name,
                code="PERMISSION_DENIED",
                message=(
                    f"identity lacks required permission "
                    f"{tool.required_permission!r}"
                ),
            )

        try:
            result = await tool.handler(**arguments)
        except Exception as exc:
            log.warning(
                "MCP dispatch: handler exception",
                exc_info=True,
                extra={"extra_data": {
                    "tool_name": tool_name,
                    "principal_id": str(identity.principal_id),
                }},
            )
            return self._error_content(
                tool_name=tool_name,
                code="HANDLER_ERROR",
                message=f"{type(exc).__name__}: {exc}",
            )

        if isinstance(result, str):
            text = result
        elif isinstance(result, mcp_types.TextContent):
            return [result]
        else:
            text = json.dumps(result, default=str, indent=2)
        return [mcp_types.TextContent(type="text", text=text)]

    @staticmethod
    def _error_content(
        *,
        tool_name: str,
        code: str,
        message: str,
    ) -> list[mcp_types.TextContent]:
        """build the structured error envelope for a denied / failed dispatch.

        the MCP spec returns errors via ``isError=True`` content
        (``CallToolResult.isError``); the SDK reads the content
        envelope back. for v1 we encode the error detail as JSON in
        the text content -- LLM clients can parse; human clients can
        read.

        :param tool_name: tool name the dispatch targeted
        :ptype tool_name: str
        :param code: machine-readable error code
        :ptype code: str
        :param message: human-readable error message
        :ptype message: str
        :return: single-element content list carrying the JSON envelope
        :rtype: list[mcp_types.TextContent]
        """
        envelope = {
            "isError": True,
            "tool_name": tool_name,
            "error": {"code": code, "message": message},
        }
        return [mcp_types.TextContent(type="text", text=json.dumps(envelope, indent=2))]

    async def start(self) -> None:
        """initialize background state (authorizer cache prime + subscribe).

        called once before :meth:`serve_stdio`. delegates to
        :meth:`Authorizer.start` so cold-start grant priming happens
        before the first dispatch can land.

        :return: nothing
        :rtype: None
        """
        await self._authorizer.start()

    async def serve_stdio(self) -> None:
        """run the SDK's stdio transport loop until EOF.

        callers MUST have called :meth:`start` first AND configured
        logging away from stdout / stderr. this method blocks until
        the client disconnects.

        :return: nothing
        :rtype: None
        """
        async with _stdio_server() as (read_stream, write_stream):
            await self._sdk_server.run(
                read_stream,
                write_stream,
                self._sdk_server.create_initialization_options(),
            )
