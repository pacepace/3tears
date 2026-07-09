"""``McpServer`` -- thin wrapper over the official ``mcp.server.Server``.

owns three things on top of the SDK:

1. tool registration: pulls :class:`McpTool` instances from a
   :class:`ToolRegistry` and registers them with the underlying SDK
   server so the client sees them at ``tools/list`` time.
2. RBAC gating: every dispatch flows through
   :meth:`Authorizer.allows(identity, tool.required_permission)`
   before the handler runs. denied dispatches return a
   :class:`mcp.types.CallToolResult` with ``isError=True`` so the
   MCP client sees the failure at the protocol level (not as
   ``isError=False`` text content carrying inner JSON).
3. error mapping: handler exceptions are caught at the framework
   border, logged with context, and returned as ``CallToolResult
   (isError=True, content=[...])``. nothing leaks past the border
   as a raw traceback.

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
from typing import TYPE_CHECKING, Any

import mcp.types as mcp_types
from mcp.server import Server as _SdkServer
from mcp.server.stdio import stdio_server as _stdio_server
from threetears.observe import get_logger

from threetears.mcp.auth import Authorizer, IdentityProvider
from threetears.mcp.tool import ToolRegistry

if TYPE_CHECKING:
    from starlette.applications import Starlette

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
    :param registry: tool registry to expose. Required: per-product
        servers construct their own via a ``build_registry()``
        factory so registry lifetime is tied to one server process
        and tests get isolation by construction. The earlier-shipped
        module-level default was unused by every production caller
        and removed 2026-05-06.
    :ptype registry: ToolRegistry
    """

    def __init__(
        self,
        *,
        name: str,
        identity_provider: IdentityProvider,
        authorizer: Authorizer,
        registry: ToolRegistry,
    ) -> None:
        """capture deps; no I/O until :meth:`serve_stdio`.

        :param name: server name string
        :ptype name: str
        :param identity_provider: identity resolver
        :ptype identity_provider: IdentityProvider
        :param authorizer: permission evaluator
        :ptype authorizer: Authorizer
        :param registry: tool registry; required (no module-level default)
        :ptype registry: ToolRegistry
        :return: nothing
        :rtype: None
        :raises ValueError: when ``name`` is empty
        """
        if not name:
            raise ValueError("server name must be non-empty")
        self._name = name
        self._identity_provider = identity_provider
        self._authorizer = authorizer
        self._registry = registry
        self._sdk_server: _SdkServer = _SdkServer(name)
        self._started = False
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
        async def _call_tool(
            name: str,
            arguments: dict[str, Any],
        ) -> mcp_types.CallToolResult | list[mcp_types.TextContent]:
            """dispatch one tool call; RBAC gate + error mapping."""
            return await self._dispatch(name, arguments)

    async def _dispatch(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> mcp_types.CallToolResult | list[mcp_types.TextContent]:
        """resolve identity, gate via authorizer, run handler, map errors.

        the gate fires BEFORE the handler so denied callers cannot
        observe handler-side state (timing, log output, etc.). every
        framework error path -- unknown tool, denied permission,
        handler exception -- returns a :class:`CallToolResult` with
        ``isError=True`` so the MCP client sees the failure at the
        protocol level. happy-path returns a ``list[TextContent]``
        which the SDK wraps in ``CallToolResult(isError=False, ...)``.

        the structured error detail (machine-readable ``code`` field)
        is preserved in the content text as JSON so callers that want
        to pattern-match on ``code`` can; the protocol-level
        ``isError`` flag is what gates "did it succeed" decisions.

        :param tool_name: requested tool name from the MCP client
        :ptype tool_name: str
        :param arguments: validated input arguments
        :ptype arguments: dict[str, Any]
        :return: :class:`CallToolResult` (errors) or
            ``list[TextContent]`` (success; SDK wraps with
            ``isError=False``)
        :rtype: mcp_types.CallToolResult | list[mcp_types.TextContent]
        """
        tool = self._registry.get(tool_name)
        if tool is None:
            log.warning(
                "MCP dispatch: unknown tool",
                extra={"extra_data": {"tool_name": tool_name}},
            )
            return self._error_result(
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
            return self._error_result(
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
                extra={
                    "extra_data": {
                        "tool_name": tool_name,
                        "principal_id": str(identity.principal_id),
                        "required_permission": tool.required_permission,
                    }
                },
            )
            return self._error_result(
                tool_name=tool_name,
                code="AUTHZ_ERROR",
                message="authorizer evaluation failed",
            )

        if not allowed:
            log.info(
                "MCP dispatch: denied",
                extra={
                    "extra_data": {
                        "tool_name": tool_name,
                        "principal_id": str(identity.principal_id),
                        "required_permission": tool.required_permission,
                    }
                },
            )
            return self._error_result(
                tool_name=tool_name,
                code="PERMISSION_DENIED",
                message=(f"identity lacks required permission {tool.required_permission!r}"),
            )

        try:
            result = await tool.handler(**arguments)
        except Exception as exc:
            log.warning(
                "MCP dispatch: handler exception",
                exc_info=True,
                extra={
                    "extra_data": {
                        "tool_name": tool_name,
                        "principal_id": str(identity.principal_id),
                    }
                },
            )
            return self._error_result(
                tool_name=tool_name,
                code="HANDLER_ERROR",
                message=f"{type(exc).__name__}: {exc}",
            )

        # happy-path normalization: handler may return a str, a
        # single TextContent, a list of TextContent (already wire-
        # ready), or something JSON-serialisable. the SDK wraps
        # the returned list in CallToolResult(isError=False, ...).
        if isinstance(result, list) and all(isinstance(item, mcp_types.TextContent) for item in result):
            return result
        if isinstance(result, mcp_types.TextContent):
            return [result]
        if isinstance(result, str):
            text = result
        else:
            text = json.dumps(result, default=str, indent=2)
        return [mcp_types.TextContent(type="text", text=text)]

    @staticmethod
    def _error_result(
        *,
        tool_name: str,
        code: str,
        message: str,
    ) -> mcp_types.CallToolResult:
        """build the protocol-level error result for a denied / failed dispatch.

        returns a :class:`CallToolResult` with ``isError=True`` so
        the MCP client sees the failure via the protocol envelope,
        not via a text-content JSON body. the structured detail
        (``code`` + ``tool_name``) is preserved in the inner content
        as JSON so callers that want machine-readable error codes
        can parse; clients that only check ``isError`` see the
        failure regardless.

        :param tool_name: tool name the dispatch targeted
        :ptype tool_name: str
        :param code: machine-readable error code
        :ptype code: str
        :param message: human-readable error message
        :ptype message: str
        :return: :class:`CallToolResult` with ``isError=True``
        :rtype: mcp_types.CallToolResult
        """
        envelope = {
            "tool_name": tool_name,
            "error": {"code": code, "message": message},
        }
        return mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text=json.dumps(envelope, indent=2))],
            isError=True,
        )

    async def start(self) -> None:
        """initialize background state (authorizer cache prime + subscribe).

        called once before :meth:`serve_stdio`. delegates to
        :meth:`Authorizer.start` so cold-start grant priming happens
        before the first dispatch can land.

        :return: nothing
        :rtype: None
        """
        await self._authorizer.start()
        self._started = True

    async def stop(self) -> None:
        """tear down background state (authorizer catch-up tick, etc.).

        called once on shutdown (typically from a lifespan teardown
        or a SIGTERM handler). delegates to :meth:`Authorizer.stop`.

        :return: nothing
        :rtype: None
        """
        await self._authorizer.stop()

    async def serve_stdio(self) -> None:
        """run the SDK's stdio transport loop until EOF.

        callers MUST have called :meth:`start` first AND configured
        logging away from stdout / stderr. this method raises a
        clear error rather than silently default-denying every
        dispatch when start() was forgotten -- the
        :class:`LocalGrantAuthorizer` cache would be empty and
        every non-admin call would deny without explanation.

        :return: nothing
        :rtype: None
        :raises RuntimeError: when :meth:`start` has not been called
        """
        if not getattr(self, "_started", False):
            raise RuntimeError(
                "McpServer.serve_stdio called before start; the authorizer "
                "cache has not been primed. Call McpServer.start() first.",
            )
        async with _stdio_server() as (read_stream, write_stream):
            await self._sdk_server.run(
                read_stream,
                write_stream,
                self._sdk_server.create_initialization_options(),
            )

    def build_http_app(
        self,
        *,
        path: str = "/mcp",
        stateless: bool = True,
        json_response: bool = True,
    ) -> Starlette:
        """build the v2 Streamable-HTTP ASGI app serving this server.

        thin delegation into :mod:`threetears.mcp.http_server` (the
        transport bulk lives there so this v1 module stays small). the
        returned app runs the SAME wired ``self._sdk_server`` over the
        SDK's Streamable-HTTP transport -- the RBAC gate, identity
        resolution, and error mapping in :meth:`_dispatch` are reused
        unchanged. start-gated like :meth:`serve_stdio`: an unprimed
        authorizer cache would silently default-deny every non-admin
        call.

        :param path: URL path the MCP endpoint is served at
        :ptype path: str
        :param stateless: run the transport statelessly (required for
            correct per-request bearer identity)
        :ptype stateless: bool
        :param json_response: return plain JSON responses instead of
            SSE streams
        :ptype json_response: bool
        :return: Starlette ASGI app serving the Streamable-HTTP transport
        :rtype: starlette.applications.Starlette
        :raises RuntimeError: when :meth:`start` has not been called
        """
        if not getattr(self, "_started", False):
            raise RuntimeError(
                "McpServer.build_http_app called before start; the authorizer "
                "cache has not been primed. Call McpServer.start() first.",
            )
        from threetears.mcp.http_server import build_mcp_http_app

        return build_mcp_http_app(
            self._sdk_server,
            path=path,
            stateless=stateless,
            json_response=json_response,
        )

    async def serve_http(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
        path: str = "/mcp",
        stateless: bool = True,
        json_response: bool = True,
    ) -> None:
        """serve this server over the v2 Streamable-HTTP transport (uvicorn).

        parallels :meth:`serve_stdio`: same ``_started`` gate, then runs
        ``self._sdk_server`` over the SDK's Streamable-HTTP transport
        instead of stdio. delegates to :mod:`threetears.mcp.http_server`.
        consumers embedding the transport in a larger ASGI app mount
        :meth:`build_http_app`'s result directly instead of calling this.

        :param host: interface to bind
        :ptype host: str
        :param port: TCP port to bind (``0`` selects an ephemeral port)
        :ptype port: int
        :param path: URL path the MCP endpoint is served at
        :ptype path: str
        :param stateless: run the transport statelessly (required for
            correct per-request bearer identity)
        :ptype stateless: bool
        :param json_response: return plain JSON responses instead of
            SSE streams
        :ptype json_response: bool
        :return: nothing
        :rtype: None
        :raises RuntimeError: when :meth:`start` has not been called
        """
        from threetears.mcp.http_server import serve_mcp_http

        app = self.build_http_app(
            path=path,
            stateless=stateless,
            json_response=json_response,
        )
        await serve_mcp_http(app, host=host, port=port)
