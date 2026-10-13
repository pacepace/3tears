"""Recall a previously-offloaded tool result by its context id.

when the ``tool_node`` offload seam stores a large tool result
out-of-band (see :mod:`threetears.langgraph.offload`), the model sees
a summary plus a recall handle of the form ``[ctx:<id>]`` instead of
the full dump. this tool is the other half of that mechanism: given the
``<id>``, it returns the full stored content so the model can pull it
back on demand in the same turn.

the tool is conversation-aware: it reads the per-conversation
:class:`~threetears.agent.tools.context.ToolContextManager` from the
active :class:`~threetears.agent.tools.call_scope.ToolCallScope` (the
exact pattern documented for every conversation-scoped tool -- the
``ToolServer`` installs the scope around each dispatch). a missing
scope, a missing manager, or an unknown id all degrade to a clear
non-success :class:`ToolResult` rather than raising.
"""

from __future__ import annotations

from datetime import tzinfo
from typing import Any

from langchain_core.tools import StructuredTool
from threetears.core.utils.local_time import resolve_zone

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.call_scope import current_scope, no_call_scope_message
from threetears.agent.tools.text_window import WindowedInput
from threetears.agent.tools.context import recalled_text

__all__ = [
    "ContextRecallTool",
    "create_context_recall_tool",
]


def create_context_recall_tool(config: dict[str, Any], description: str) -> StructuredTool:
    """Factory: create the context_recall tool.

    delegates to :func:`threetears.agent.tools.langchain_adapter.to_langchain_tool`
    so the in-process StructuredTool path and the NATS-dispatched
    ToolServer path share one execution body
    (:meth:`ContextRecallTool.execute`).

    the tool reads the conversation's context manager from the call scope,
    and on this path the scope gets one only from a ``context_factory``.
    without it every recall answered "no context manager in scope", even
    with the turn's ``call_context`` in the graph config.

    Expected ``config`` keys:

    - ``context_factory`` -- ``async (conversation_id, user_id) ->``
      :class:`~threetears.agent.tools.context.ToolContextManager`, the
      :data:`~threetears.agent.tools.call_scope.ContextFactory` a
      ToolServer takes; resolves the manager the graph writes offloaded
      results through. without it the tool can recall nothing.
    - ``tz`` -- the zone of the person the result is read to (a ``tzinfo`` or
      an IANA name): a result a later call refreshed says when, in it. without
      it the time is said in UTC.

    :param config: per-agent config dict
    :ptype config: dict[str, Any]
    :param description: tool description surfaced to the LLM
    :ptype description: str
    :return: a LangChain ``StructuredTool`` wrapping the recall tool
    :rtype: StructuredTool
    :raises TypeError: if ``context_factory`` is given and is not callable
    :raises ValueError: if ``tz`` is a name that is not a known zone
    """
    from threetears.agent.tools.langchain_adapter import to_langchain_tool

    context_factory = config.get("context_factory")
    if context_factory is not None and not callable(context_factory):
        raise TypeError(
            "create_context_recall_tool: config['context_factory'] must be an async "
            f"(conversation_id, user_id) -> ToolContextManager callable, not {type(context_factory).__name__}"
        )
    return to_langchain_tool(
        ContextRecallTool(tz=config.get("tz")),
        description=description,
        context_factory=context_factory,
    )


class ContextRecallTool(TearsTool):
    """TearsTool returning the full stored content for a context id.

    pairs with the offload seam: the model passes the ``<id>`` from a
    ``[ctx:<id>]`` handle and gets the full out-of-band content back.
    """

    def __init__(self, *, tz: tzinfo | str | None = None) -> None:
        """Bind the zone a refreshed result's time is said in.

        :param tz: a ``tzinfo`` or an IANA name; ``None`` says the time in UTC
        :ptype tz: tzinfo | str | None
        :raises ValueError: when ``tz`` is a name that is not a known zone
        """
        self._zone = resolve_zone(tz)

    _INPUT_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "context_id": {
                "type": "string",
                "description": "The id in a [ctx:<id>] mark, with or without the ctx: part.",
            },
            "offset": {
                "type": "integer",
                "minimum": 0,
                "default": 0,
                "description": WindowedInput.model_fields["offset"].description,
            },
        },
        "required": ["context_id"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """return the full stored content for ``context_id``.

        resolution order, all degrading to a clear non-success result:

        1. empty ``context_id`` -> "requires a context_id".
        2. no call scope / no context manager in scope -> "unavailable"; with
           no scope at all the error names what the caller must supply
           (:func:`~threetears.agent.tools.call_scope.no_call_scope_message`).
        3. unknown id -> "not found".
        4. found -> success with the stored content from ``offset``, at most
           :data:`~threetears.agent.tools.context.RECALL_MAX_CHARS` of it. A
           part opens with its window line, which names the next offset, and a
           result a later call with the same input replaced opens with one line
           saying so.

        :param kwargs: must include ``context_id`` (string handle); may include ``offset``.
        :ptype kwargs: Any
        :return: the stored content on success, else a clear failure
            result the LLM can reason about.
        :rtype: ToolResult
        """
        context_id = kwargs.get("context_id") or ""
        scope = current_scope()
        manager = scope.context_manager if scope is not None else None
        if not context_id:
            result = ToolResult(
                success=False,
                content="Give context_recall the context_id from a [ctx:<id>] mark.",
                error="missing context_id",
            )
        elif manager is None:
            # no scope at all is a caller that installed none -- named, so it can supply one; a
            # scope with no manager is a call that genuinely belongs to no conversation.
            result = ToolResult(
                success=False,
                content="Saved results are unavailable here: this call is not part of a conversation.",
                error=no_call_scope_message("context_recall") if scope is None else "no context manager in scope",
            )
        else:
            item = await manager.get_context_item(context_id)
            if item is None:
                result = ToolResult(
                    success=False,
                    content=f"Saved result '{context_id}' not found in this conversation.",
                    error="not found",
                )
            else:
                offset = max(int(kwargs.get("offset") or 0), 0)
                result = ToolResult(success=True, content=recalled_text(item, offset=offset, tz=self._zone))
        return result

    def mcp_schema(self) -> MCPToolDefinition:
        """return MCP-compatible tool definition for context_recall.

        :return: tool definition with name, version, description, input schema
        :rtype: MCPToolDefinition
        """
        result = MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description=(
                "Recall a previously-offloaded tool result by its context id (the '<id>' from a "
                "'[ctx:<id>]' handle). A long one comes in parts: each part names the offset of the next."
            ),
            input_schema=self._INPUT_SCHEMA,
        )
        return result

    def mcp_name(self) -> str:
        """return namespaced tool name.

        :return: namespaced tool name
        :rtype: str
        """
        return "threetears.context_recall"

    def mcp_version(self) -> str:
        """return tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"
