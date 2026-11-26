"""the MCP face of a TearsTool: the :class:`~threetears.mcp.tool.McpTool` 3tears-mcp's ``McpServer`` serves it as.

one tool definition serves both faces: the platform mesh calls ``TearsTool.run`` through 3tears-agent-tools'
``ToolServer``, and an MCP client calls the same ``run`` through the tool :func:`mcp_tool` builds::

    registry.register(mcp_tool(SessionSend(service), required_permission="sessions.send", name="session_send"))

the tool's schema is its listing, its :class:`~threetears.tool_contract.tool.ToolResult` is mapped to the call's
result with its code on the protocol result, and an ``execute`` that declares a parameter annotated
:class:`~threetears.mcp.context.CallContext` is passed the call's context -- ``run`` passes a keyword outside the
input schema through unchanged, so ``run`` itself does not change. needs the ``mcp`` extra
(``3tears-tool-contract[mcp]``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

try:
    import mcp.types as mcp_types
    from threetears.mcp import CallContext, McpTool, McpToolAnnotations, tool_error
    from threetears.mcp.tool import context_parameter_of
    from threetears.mcp.errors import BUILTIN_ERROR_CODES, ErrorCode
except ImportError as _exc:
    raise ImportError(
        "threetears.tool_contract.mcp_face serves a TearsTool over 3tears-mcp, which is the mcp extra: "
        "install 3tears-tool-contract[mcp]",
    ) from _exc

from threetears.tool_contract.tool import TOOL_RESULT_ERROR_CODES, TearsTool, ToolResult

__all__ = ["mcp_tool"]

#: a ToolResult's codes as 3tears-mcp's built-in codes, which hold each with the same meaning; built at import, so a
#: code the contract adds and 3tears-mcp lacks fails here rather than on a call
_RESULT_CODES: Mapping[str, ErrorCode] = {code: BUILTIN_ERROR_CODES[code] for code in TOOL_RESULT_ERROR_CODES}


def mcp_tool(
    tool: TearsTool,
    *,
    required_permission: str,
    name: str | None = None,
    title: str | None = None,
    annotations: McpToolAnnotations | None = None,
) -> McpTool:
    """the :class:`~threetears.mcp.tool.McpTool` that serves ``tool`` over MCP.

    :param tool: the tool
    :ptype tool: TearsTool
    :param required_permission: the permission a caller needs (3tears-mcp's per-tool gate)
    :ptype required_permission: str
    :param name: the name the MCP face lists it under; ``tool.mcp_name()`` when not given (a canonical name has a
        dot, which an MCP client already prefixing the server's name does not need and some LLM tool APIs reject)
    :ptype name: str | None
    :param title: a human-readable display name
    :ptype title: str | None
    :param annotations: behaviour hints for the client
    :ptype annotations: McpToolAnnotations | None
    :return: the MCP tool
    :rtype: McpTool
    :raises ValueError: when the tool's ``face_mcp`` is ``False``, its ``execute`` declares the context in a way
        ``McpTool`` refuses, or its input schema has a property named ``tears_call_context`` (the keyword the face passes the context under)
    """
    if not tool.face_mcp:
        raise ValueError(
            f"{type(tool).__name__} ({tool.mcp_name()}) does not declare the MCP face: set face_mcp = True on the "
            "tool to serve it over MCP",
        )
    schema = tool.mcp_schema()
    tool_name = name if name is not None else tool.mcp_name()
    # the same rule a plain handler's context parameter has, read from execute's signature
    context_parameter = context_parameter_of(tool_name, tool.execute, schema.input_schema)
    structured = schema.output_schema is not None

    async def handler(*, tears_call_context: CallContext, **arguments: Any) -> mcp_types.CallToolResult:
        if context_parameter is not None:
            arguments[context_parameter] = tears_call_context
        result = await tool.run(**arguments)
        if not isinstance(result, ToolResult):
            raise TypeError(f"{type(tool).__name__}.execute returned {type(result).__name__}, not a ToolResult")
        return _call_result(result, structured=structured)

    return McpTool(
        name=tool_name,
        description=schema.description,
        input_schema=schema.input_schema,
        required_permission=required_permission,
        handler=handler,
        output_schema=schema.output_schema,
        title=title,
        annotations=annotations,
    )


def _call_result(result: ToolResult, *, structured: bool) -> mcp_types.CallToolResult:
    """a ToolResult as the call's result.

    :param result: what the tool returned
    :ptype result: ToolResult
    :param structured: whether the tool declares an output schema, so its content, a JSON object, is its structured
        content too, on a success and a failure alike
    :ptype structured: bool
    :return: the call's result
    :rtype: mcp.types.CallToolResult
    """
    if result.success:
        text = mcp_types.TextContent(type="text", text=result.content)
        parsed = _json_object(result.content) if structured else None
        if parsed is None:
            # a tool with an output schema whose content is no JSON object is refused by the server's output check
            return mcp_types.CallToolResult(content=[text], is_error=False)
        return mcp_types.CallToolResult(content=[text], is_error=False, structured_content=parsed)
    # a structured tool's failure content, a JSON object, travels as structured content too, as its successes do; the
    # output schema binds successes only (the server's output check and the SDK's clients skip an isError result), so
    # the failure's shape need not fit it
    text = result.content or result.error or ""
    answered = tool_error(text, code=_failure_code(result), message=result.error or None)
    failure = _json_object(text) if structured else None
    # the text kept as the tool wrote it, as a success's is; the parsed object beside it
    return answered if failure is None else answered.model_copy(update={"structured_content": failure})


def _failure_code(result: ToolResult) -> ErrorCode | str | None:
    """the code a failed result names: its ``error_code``, else its ``metadata["error_code"]``, else none.

    :param result: the failed result
    :ptype result: ToolResult
    :return: the code (``None`` is the server's ``TOOL_ERROR``)
    :rtype: ErrorCode | str | None
    """
    if result.error_code is not None:
        return _RESULT_CODES[result.error_code]
    named = (result.metadata or {}).get("error_code")
    return named if isinstance(named, str) else None


def _json_object(text: str) -> dict[str, Any] | None:
    """``text`` parsed as a JSON object, or ``None`` when it is not one.

    :param text: the content
    :ptype text: str
    :return: the object
    :rtype: dict[str, Any] | None
    """
    try:
        parsed = json.loads(text)
    except ValueError:
        # NOSILENT: content that is no JSON is answered as text alone; the server's output check refuses it
        return None
    return parsed if isinstance(parsed, dict) else None
