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

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.call_scope import current_scope

__all__ = [
    "ContextRecallInput",
    "ContextRecallTool",
    "create_context_recall_tool",
]


class ContextRecallInput(BaseModel):
    """Input for the context_recall tool."""

    context_id: str = Field(
        description=(
            "The context id to recall, as shown in a tool result's "
            "'[ctx:<id>]' handle. Either '<id>' or 'ctx:<id>' is accepted."
        ),
    )


def create_context_recall_tool(config: dict[str, Any], description: str) -> StructuredTool:
    """Factory: create the context_recall tool.

    delegates to :func:`threetears.agent.tools.langchain_adapter.to_langchain_tool`
    so the in-process StructuredTool path and the NATS-dispatched
    ToolServer path share one execution body
    (:meth:`ContextRecallTool.execute`). ``config`` is unused (the tool
    resolves its context manager per-call from the call scope), kept in
    the signature for :func:`register_builtins` factory-shape parity.

    :param config: per-agent config dict (unused; resolution is per-call)
    :ptype config: dict[str, Any]
    :param description: tool description surfaced to the LLM
    :ptype description: str
    :return: a LangChain ``StructuredTool`` wrapping the recall tool
    :rtype: StructuredTool
    """
    from threetears.agent.tools.langchain_adapter import to_langchain_tool

    return to_langchain_tool(
        ContextRecallTool(),
        description=description,
        args_schema=ContextRecallInput,
    )


class ContextRecallTool(TearsTool):
    """TearsTool returning the full stored content for a context id.

    pairs with the offload seam: the model passes the ``<id>`` from a
    ``[ctx:<id>]`` handle and gets the full out-of-band content back.
    """

    _INPUT_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "context_id": {
                "type": "string",
                "description": (
                    "The context id to recall, as shown in a tool result's "
                    "'[ctx:<id>]' handle. Either '<id>' or 'ctx:<id>' is accepted."
                ),
            },
        },
        "required": ["context_id"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """return the full stored content for ``context_id``.

        resolution order, all degrading to a clear non-success result:

        1. empty ``context_id`` -> "requires a context_id".
        2. no call scope / no context manager in scope -> "unavailable".
        3. unknown id -> "not found".
        4. found -> success with the full stored content.

        :param kwargs: must include ``context_id`` (string handle).
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
                content="context_recall requires a non-empty context_id.",
                error="missing context_id",
            )
        elif manager is None:
            result = ToolResult(
                success=False,
                content="context recall unavailable: no conversation context in this call scope.",
                error="no context manager in scope",
            )
        else:
            item = await manager.get_context_item(context_id)
            if item is None:
                result = ToolResult(
                    success=False,
                    content=f"context item not found for id: {context_id}",
                    error="not found",
                )
            else:
                result = ToolResult(success=True, content=str(item.get("content", "")))
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
                "Recall the full content of a previously-offloaded tool result "
                "by its context id (the '<id>' from a '[ctx:<id>]' handle)."
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
