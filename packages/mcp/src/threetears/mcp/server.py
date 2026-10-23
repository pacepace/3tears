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
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import jsonschema
import mcp.types as mcp_types
from mcp.server import Server as _SdkServer
from mcp.server.stdio import stdio_server as _stdio_server
from threetears.observe import contain, get_logger

from threetears.mcp.auth import Authorizer, GrantCacheUnavailableError, IdentityProvider
from threetears.mcp.tool import McpTool, ToolProvider, ToolRegistry

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from threetears.mcp.http_server import McpHttpTransport

__all__ = [
    "McpServer",
    "tool_error",
]

log = get_logger(__name__)

#: mcp 1.x registers handlers with ``@server.list_tools()`` / ``@server.call_tool()``; 2.x removed
#: both and takes ``on_list_tools`` / ``on_call_tool`` in the constructor. 3tears runs on either
#: (``mcp>=1.27,<3``), so consumers move to 2.x on their own schedule; 1.x is dropped when the owner says.
_SDK_V1 = hasattr(_SdkServer, "call_tool")

#: what a client is told when the tool provider fails; the provider's own error is logged, not sent
_TOOLS_UNAVAILABLE_MESSAGE = "the server's tool set could not be read"

#: how much of a refused result's structured content the refusal's log line keeps: enough for an
#: operator to see what the call returned, without one oversized result flooding the log
_MAX_LOGGED_STRUCTURED_CHARS = 4000


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
    :param registry: tool registry to expose. per-product servers
        construct their own via a ``build_registry()`` factory so
        registry lifetime is tied to one server process and tests get
        isolation by construction (there is no module-level default).
        exactly one of ``registry`` and ``tool_provider`` is given
    :ptype registry: ToolRegistry | None
    :param tool_provider: called on every ``tools/list`` and
        ``tools/call`` for the tools as they stand, for a tool set that
        follows a live registry; in place of ``registry``
    :ptype tool_provider: ToolProvider | None
    """

    def __init__(
        self,
        *,
        name: str,
        identity_provider: IdentityProvider,
        authorizer: Authorizer,
        registry: ToolRegistry | None = None,
        tool_provider: ToolProvider | None = None,
    ) -> None:
        """capture deps; no I/O until :meth:`serve_stdio`.

        :param name: server name string
        :ptype name: str
        :param identity_provider: identity resolver
        :ptype identity_provider: IdentityProvider
        :param authorizer: permission evaluator
        :ptype authorizer: Authorizer
        :param registry: a fixed tool registry
        :ptype registry: ToolRegistry | None
        :param tool_provider: the live tool set, in place of ``registry``
        :ptype tool_provider: ToolProvider | None
        :return: nothing
        :rtype: None
        :raises ValueError: when ``name`` is empty, or not exactly one of
            ``registry`` and ``tool_provider`` is given
        """
        if not name:
            raise ValueError("server name must be non-empty")
        # a fixed registry is the provider that always answers the same tools
        if registry is not None and tool_provider is None:
            self._tool_provider: ToolProvider = registry.all
        elif tool_provider is not None and registry is None:
            self._tool_provider = tool_provider
        else:
            raise ValueError("McpServer takes exactly one of registry= and tool_provider=")
        self._name = name
        self._identity_provider = identity_provider
        self._authorizer = authorizer
        self._started = False
        # one ERROR per grant-cache outage, not one traceback per refused call: the authorizer
        # already logged the failed reload with its traceback, and an outage refuses every call.
        self._grant_cache_outage_logged = False
        self._sdk_server: _SdkServer[Any] = self._build_sdk_server(name)

    @property
    def name(self) -> str:
        """advertised server name.

        :return: server name
        :rtype: str
        """
        return self._name

    def _build_sdk_server(self, name: str) -> _SdkServer[Any]:
        """build the SDK server with this server's ``tools/list`` + ``tools/call`` handlers.

        each :class:`McpServer` gets its own SDK server (no cross-pollination). mcp 1.x and
        2.x register handlers differently (see :data:`_SDK_V1`); both answer from the same
        :meth:`_tools` and :meth:`_dispatch`, so a caller sees one behaviour on either.

        :param name: advertised server name
        :ptype name: str
        :return: the wired SDK server
        :rtype: mcp.server.Server
        """
        if _SDK_V1:
            # typed as Any: the 1.x decorators do not exist in the 2.x stubs mypy checks against
            server: Any = _SdkServer(name)

            @server.list_tools()  # type: ignore[untyped-decorator]
            async def _list_tools() -> list[mcp_types.Tool]:
                return self._tools()

            # the call handler goes straight into request_handlers, which is all the 1.x call_tool
            # decorator does once its own work is done. _dispatch checks the arguments and answers
            # every refusal itself. two more things the decorator did are left out on purpose: its
            # cache of an earlier tools/list goes stale once a live provider changes a schema, and is
            # refreshed on a miss by calling the provider outside _dispatch, where a failure reached
            # the client as the exception's own text; and it turned an exception escaping the handler
            # into an isError result. what escapes _dispatch is a bug, which 2.x answers with a
            # JSON-RPC error, and 1.x now answers alike
            async def _call_tool(request: Any) -> Any:
                params = request.params
                # Any: 1.x's ServerResult is the model wrapping a result, 2.x's stubs name a union
                v1_types: Any = mcp_types
                return v1_types.ServerResult(await self._dispatch(params.name, params.arguments or {}))

            server.request_handlers[mcp_types.CallToolRequest] = _call_tool

            built: _SdkServer[Any] = server
            return built

        async def _on_list_tools(_ctx: Any, _params: Any) -> mcp_types.ListToolsResult:
            return mcp_types.ListToolsResult(tools=self._tools())

        async def _on_call_tool(_ctx: Any, params: Any) -> mcp_types.CallToolResult:
            # 2.x passes arguments=None for a call that sent none; 1.x passed {}. what escapes
            # _dispatch is a bug (it answers every refusal itself), so it is not caught here: on
            # either major the SDK answers the request with a JSON-RPC error
            return await self._dispatch(params.name, params.arguments or {})

        return _SdkServer(name, on_list_tools=_on_list_tools, on_call_tool=_on_call_tool)

    def _current_tools(self) -> dict[str, McpTool]:
        """the tools as the provider answers them now, by name, in its order.

        :return: tool by name
        :rtype: dict[str, McpTool]
        :raises ValueError: when the provider answers two tools with one name, which a client
            could not tell apart
        """
        tools: dict[str, McpTool] = {}
        for tool in self._tool_provider():
            if tool.name in tools:
                raise ValueError(f"the tool provider answered two tools named {tool.name!r}")
            tools[tool.name] = tool
        return tools

    def _tools(self) -> list[mcp_types.Tool]:
        """the tool inventory, in the provider's order.

        :return: one MCP tool per tool the provider answers
        :rtype: list[mcp_types.Tool]
        :raises RuntimeError: when the provider fails, after logging it, with the provider's
            exception as the cause; the SDK answers the ``tools/list`` with a JSON-RPC error
        """
        try:
            tools = self._current_tools()
        except Exception as exc:
            log.exception("MCP tools/list: the tool provider failed")
            # the SDK puts the exception's text in the JSON-RPC error it answers, on either major:
            # a fixed message keeps the provider's internals off the wire, as tools/call's
            # TOOLS_UNAVAILABLE does
            raise RuntimeError(_TOOLS_UNAVAILABLE_MESSAGE) from exc
        return [_listed_tool(tool) for tool in tools.values()]

    async def _dispatch(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> mcp_types.CallToolResult:
        """resolve identity, gate via authorizer, run handler, map errors.

        the gate fires BEFORE the handler so denied callers cannot
        observe handler-side state (timing, log output, etc.). every
        framework error path -- unknown tool, denied permission,
        handler exception -- returns a :class:`CallToolResult` with
        ``isError=True`` so the MCP client sees the failure at the
        protocol level. the happy path returns one too, with
        ``isError=False``: 2.x no longer wraps a bare content list.

        the structured error detail (machine-readable ``code`` field)
        is preserved in the content text as JSON so callers that want
        to pattern-match on ``code`` can; the protocol-level
        ``isError`` flag is what gates "did it succeed" decisions.

        :param tool_name: requested tool name from the MCP client
        :ptype tool_name: str
        :param arguments: validated input arguments
        :ptype arguments: dict[str, Any]
        :return: the call's result, failed or not
        :rtype: mcp_types.CallToolResult
        """
        try:
            tool = self._current_tools().get(tool_name)
        except Exception as exc:
            contain(
                exc,
                log=log,
                what="MCP dispatch: the tool provider failed",
                extra={"tool_name": tool_name},
                level="error",
            )
            return self._error_result(
                tool_name=tool_name,
                code="TOOLS_UNAVAILABLE",
                message=_TOOLS_UNAVAILABLE_MESSAGE,
            )
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

        # 1.x's call_tool decorator checked the arguments against the tool's inputSchema before
        # the handler ran, and 2.x's low-level server does not: checked here, against the tool as
        # the provider answers it now, in the decorator's words, so both majors refuse alike
        try:
            jsonschema.validate(instance=arguments, schema=tool.input_schema)
        except jsonschema.ValidationError as exc:
            return _plain_error_result(f"Input validation error: {exc.message}")

        try:
            identity = await self._identity_provider.identify()
        except RuntimeError as exc:
            # the one type the IdentityProvider protocol raises for "no identity"; anything else is a bug
            contain(
                exc,
                log=log,
                what="MCP dispatch: identity resolution failed",
                extra={"tool_name": tool_name},
            )
            return self._error_result(
                tool_name=tool_name,
                code="IDENTITY_UNAVAILABLE",
                message="caller identity could not be resolved",
            )

        try:
            allowed = await self._authorizer.allows(identity, tool.required_permission)
        except GrantCacheUnavailableError:
            if not self._grant_cache_outage_logged:
                self._grant_cache_outage_logged = True
                log.error(
                    "MCP dispatch: the grant cache is not current; refusing non-admin calls until it reloads",
                    extra={"extra_data": {"tool_name": tool_name, "required_permission": tool.required_permission}},
                )
            return self._error_result(
                tool_name=tool_name,
                code="AUTHZ_ERROR",
                message="authorization is temporarily unavailable (the grant cache is reloading); retry",
            )
        except Exception as exc:
            contain(
                exc,
                log=log,
                what="MCP dispatch: authorizer raised; denying by default",
                extra={
                    "tool_name": tool_name,
                    "principal_id": str(identity.principal_id),
                    "required_permission": tool.required_permission,
                },
            )
            return self._error_result(
                tool_name=tool_name,
                code="AUTHZ_ERROR",
                message="authorizer evaluation failed",
            )

        if self._grant_cache_outage_logged and not identity.is_admin:
            # an admin's call never reads the cache, so only a non-admin answer ends the outage
            self._grant_cache_outage_logged = False
            log.info("MCP dispatch: the grant cache answers again; calls are being authorized")
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
            contain(
                exc,
                log=log,
                what="MCP dispatch: handler exception",
                extra={
                    "tool_name": tool_name,
                    "principal_id": str(identity.principal_id),
                },
            )
            return self._error_result(
                tool_name=tool_name,
                code="HANDLER_ERROR",
                message=f"{type(exc).__name__}: {exc}",
            )

        call_result = _as_call_tool_result(result)
        if tool.output_schema is not None:
            # the result as the client receives it: the SDK sends it in JSON mode, where a UUID or a
            # datetime is a string
            wire = call_result.model_dump(mode="json", by_alias=True)
            refusal = _output_schema_refusal(wire, tool.output_schema)
            if refusal is not None:
                # the handler has run, so for a tool with side effects this log is the only record
                # of what it returned
                log.error(
                    "MCP dispatch: the tool's result does not match its outputSchema",
                    extra={
                        "extra_data": {
                            "tool_name": tool_name,
                            "reason": refusal,
                            "structured_content": _loggable_structured(wire.get("structuredContent")),
                        }
                    },
                )
                return self._error_result(tool_name=tool_name, code="OUTPUT_VALIDATION_ERROR", message=refusal)
        return call_result

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
        return _call_tool_result(
            content=[mcp_types.TextContent(type="text", text=json.dumps(envelope, indent=2))],
            is_error=True,
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

    def build_http_transport(
        self,
        *,
        path: str = "/mcp",
        stateless: bool = True,
        json_response: bool = True,
    ) -> McpHttpTransport:
        """build the Streamable-HTTP transport for a host app to embed.

        the embedding counterpart of :meth:`build_http_app`: the host
        adds the transport's ``routes`` (or ``endpoint``) to its own app
        and enters its ``lifespan()`` in its own lifespan, which a
        mounted :meth:`build_http_app` would never run (see
        :class:`~threetears.mcp.http_server.McpHttpTransport`).
        start-gated like :meth:`build_http_app`.

        :param path: URL path of the endpoint in the transport's routes
        :ptype path: str
        :param stateless: run the transport statelessly (required for
            correct per-request bearer identity)
        :ptype stateless: bool
        :param json_response: return plain JSON responses instead of
            SSE streams
        :ptype json_response: bool
        :return: the transport
        :rtype: threetears.mcp.http_server.McpHttpTransport
        :raises RuntimeError: when :meth:`start` has not been called
        """
        if not getattr(self, "_started", False):
            raise RuntimeError(
                "McpServer.build_http_transport called before start; the authorizer "
                "cache has not been primed. Call McpServer.start() first.",
            )
        from threetears.mcp.http_server import McpHttpTransport

        return McpHttpTransport(
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


def _call_tool_result(
    *,
    content: list[mcp_types.TextContent],
    is_error: bool,
    structured: dict[str, Any] | None = None,
) -> mcp_types.CallToolResult:
    """build a :class:`CallToolResult` on either SDK major.

    built from the wire names: 1.x's fields are ``isError`` / ``structuredContent`` and 2.x's
    are ``is_error`` / ``structured_content`` with the wire names as aliases, so a constructor
    keyword would be wrong on one of them while a validated dict is right on both.

    :param content: the content blocks
    :ptype content: list[mcp_types.TextContent]
    :param is_error: whether the call failed
    :ptype is_error: bool
    :param structured: the structured result, when there is one
    :ptype structured: dict[str, Any] | None
    :return: the result
    :rtype: mcp_types.CallToolResult
    """
    fields: dict[str, Any] = {"content": content, "isError": is_error}
    if structured is not None:
        fields["structuredContent"] = structured
    return mcp_types.CallToolResult.model_validate(fields)


def _plain_error_result(message: str) -> mcp_types.CallToolResult:
    """the bare-text error result mcp 1.x's ``call_tool`` decorator answered with.

    :param message: the error text
    :ptype message: str
    :return: an ``isError`` result carrying ``message``
    :rtype: mcp_types.CallToolResult
    """
    return _call_tool_result(content=[mcp_types.TextContent(type="text", text=message)], is_error=True)


def tool_error(error: Mapping[str, Any]) -> mcp_types.CallToolResult:
    """a tool-execution error a handler returns: ``isError`` set, ``error`` in both faces.

    the spec's tool-execution error -- the tool ran and the call failed, as opposed to a protocol
    error -- with the payload as ``structuredContent`` and as its serialized JSON in a text
    block, the same two faces a successful structured result has. :class:`McpServer` passes it
    through as the call's result; it is not logged, nor mapped to a framework error code, and a
    tool's ``output_schema`` does not apply to it::

        return tool_error({"code": "NOT_FOUND", "message": "no runner 'r1'", "retryable": False})

    :param error: the structured error, a JSON object
    :ptype error: Mapping[str, Any]
    :return: the call's result
    :rtype: mcp_types.CallToolResult
    """
    structured = dict(error)
    return _call_tool_result(
        content=[mcp_types.TextContent(type="text", text=json.dumps(structured, default=str, indent=2))],
        structured=structured,
        is_error=True,
    )


def _listed_tool(tool: McpTool) -> mcp_types.Tool:
    """the ``tools/list`` entry for ``tool``.

    :param tool: the tool
    :ptype tool: McpTool
    :return: the SDK's tool, with only the optional fields the tool sets
    :rtype: mcp_types.Tool
    """
    # model_validate with the wire names: 1.x names the fields inputSchema / outputSchema, 2.x
    # input_schema / output_schema with the wire names as aliases, and the alias is accepted by both
    listed: dict[str, Any] = {"name": tool.name, "description": tool.description, "inputSchema": tool.input_schema}
    if tool.output_schema is not None:
        listed["outputSchema"] = tool.output_schema
    if tool.title is not None:
        listed["title"] = tool.title
    if tool.annotations is not None:
        listed["annotations"] = tool.annotations.to_wire()
    return mcp_types.Tool.model_validate(listed)


def _as_call_tool_result(result: Any) -> mcp_types.CallToolResult:
    """a handler's return value as the call's result.

    :param result: what the handler returned
    :ptype result: Any
    :return: the result the client receives
    :rtype: mcp_types.CallToolResult
    """
    # a handler may return a fully-formed CallToolResult (e.g. an isError=True envelope carrying a
    # downstream {code,message}, as tool_error builds) and it passes through verbatim
    if isinstance(result, mcp_types.CallToolResult):
        return result

    # a plain content result is wrapped here rather than by the SDK: 1.x wrapped a returned list in
    # CallToolResult(isError=False), 2.x requires the handler to return the result
    if isinstance(result, list) and all(isinstance(item, mcp_types.TextContent) for item in result):
        return _call_tool_result(content=result, is_error=False)
    if isinstance(result, mcp_types.TextContent):
        return _call_tool_result(content=[result], is_error=False)
    if isinstance(result, str):
        return _call_tool_result(content=[mcp_types.TextContent(type="text", text=result)], is_error=False)

    # a structured result rides BOTH faces: the prose a model reads, and the spec's
    # ``structuredContent`` a program reads. returning the full CallToolResult rather than a
    # content list is what lets the structured field be set; the SDK's list wrapper has nowhere
    # to put it (search-spec.md §4.8).
    text = json.dumps(result, default=str, indent=2)
    return _call_tool_result(
        content=[mcp_types.TextContent(type="text", text=text)],
        structured=result if isinstance(result, dict) else {"result": result},
        is_error=False,
    )


def _output_schema_refusal(wire: dict[str, Any], output_schema: dict[str, Any]) -> str | None:
    """why a successful result breaks the tool's ``outputSchema``, or ``None`` when it does not.

    the spec's server MUST: a tool that declares an output schema returns structured results that
    conform to it. an error result is exempt. 2.x's low-level server checks nothing; 1.x's
    decorator checked only results it built itself, never a returned CallToolResult.

    :param wire: the call's result as the JSON the client receives
    :ptype wire: dict[str, Any]
    :param output_schema: the tool's declared output schema
    :ptype output_schema: dict[str, Any]
    :return: the reason, or ``None``
    :rtype: str | None
    """
    if wire.get("isError"):
        return None
    structured = wire.get("structuredContent")
    if structured is None:
        return "the tool declares an outputSchema but returned no structured content"
    try:
        jsonschema.validate(instance=structured, schema=output_schema)
    except jsonschema.ValidationError as exc:
        return f"the tool's structured result does not match its outputSchema: {exc.message}"
    return None


def _loggable_structured(structured: Any) -> str:
    """a refused result's structured content as JSON text, cut to a length a log line can carry.

    :param structured: the structured content, as the client would have received it
    :ptype structured: Any
    :return: its JSON, length-capped
    :rtype: str
    """
    text = json.dumps(structured)
    if len(text) > _MAX_LOGGED_STRUCTURED_CHARS:
        return f"{text[:_MAX_LOGGED_STRUCTURED_CHARS]}...[truncated]"
    return text
