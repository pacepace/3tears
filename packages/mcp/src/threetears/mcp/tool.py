"""tool primitives -- :class:`McpTool` dataclass + ``register_tool`` decorator.

every MCP tool exposed via :class:`~threetears.mcp.server.McpServer` is
described by an :class:`McpTool` instance. tools are typically declared
via the :func:`register_tool` decorator which wraps a handler and
records the tool into a per-process registry the server consumes at
:meth:`McpServer.serve` time.

each tool MUST declare a ``required_permission`` string (e.g.
``"metallm.conversations.read"``). a tool with ``required_permission``
set runs only when :meth:`Authorizer.allows(identity, permission)`
returns True; the framework defaults to default-deny. an AST
enforcement test forbids ``required_permission=None`` in production
``register_tool`` call sites.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "McpTool",
    "ToolHandler",
    "ToolRegistry",
    "register_tool",
]


ToolHandler = Callable[..., Awaitable[Any]]
"""signature for a tool's handler.

invoked with the validated input kwargs; returns whatever the tool
produces. the framework formats the return value into the MCP
response envelope (typically ``TextContent`` for v1).
"""


@dataclass(frozen=True, slots=True)
class McpTool:
    """one MCP tool's full descriptor.

    :ivar name: MCP tool name as the client sees it (must be unique
        per server)
    :ivar description: human-readable description; the LLM uses this
        to decide whether to invoke the tool
    :ivar input_schema: JSON Schema (draft 2020-12 compatible) for
        the tool's input arguments; the MCP SDK validates against
        this before dispatch
    :ivar required_permission: permission string the caller must
        hold; default-deny when no grant matches. v1 framework
        rejects ``None`` -- explicit ``"public"`` would still go
        through the authorizer (which is free to implement an
        always-allow policy for that exact string)
    :ivar handler: async callable invoked with validated input
        kwargs; return value is wrapped into MCP response envelope
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    required_permission: str
    handler: ToolHandler


class ToolRegistry:
    """per-process tool registry consumed by :class:`McpServer`.

    populated either via direct :meth:`register` calls or via the
    :func:`register_tool` decorator (which calls into a module-level
    default registry). servers that want isolation (e.g. tests
    constructing multiple servers in one process) can pass their own
    :class:`ToolRegistry` instance to :class:`McpServer`.
    """

    def __init__(self) -> None:
        """initialize an empty registry."""
        self._tools: dict[str, McpTool] = {}

    def register(self, tool: McpTool) -> None:
        """add a tool to the registry; raises on duplicate name.

        :param tool: tool descriptor
        :ptype tool: McpTool
        :return: nothing
        :rtype: None
        :raises ValueError: when a tool with the same name is
            already registered
        """
        if tool.name in self._tools:
            raise ValueError(
                f"tool name conflict: {tool.name!r} already registered",
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> McpTool | None:
        """look up a tool by name.

        :param name: MCP tool name
        :ptype name: str
        :return: tool descriptor or ``None`` if not registered
        :rtype: McpTool | None
        """
        return self._tools.get(name)

    def all(self) -> list[McpTool]:
        """return every registered tool in registration order.

        :return: list of tool descriptors
        :rtype: list[McpTool]
        """
        return list(self._tools.values())

    def names(self) -> list[str]:
        """return the names of every registered tool.

        :return: list of tool names
        :rtype: list[str]
        """
        return list(self._tools.keys())


_DEFAULT_REGISTRY = ToolRegistry()


def register_tool(
    *,
    name: str,
    description: str,
    input_schema: dict[str, Any],
    required_permission: str,
    registry: ToolRegistry | None = None,
) -> Callable[[ToolHandler], ToolHandler]:
    """decorator that registers ``handler`` as an MCP tool.

    canonical usage::

        @register_tool(
            name="list_conversations",
            description="list recent conversations",
            input_schema={"type": "object", "properties": {...}},
            required_permission="metallm.conversations.read",
        )
        async def handle_list_conversations(**kwargs) -> str:
            ...

    when ``registry`` is ``None`` the call uses the module-level
    default registry. tests that need isolation pass their own.

    :param name: MCP tool name (must be unique within the registry)
    :ptype name: str
    :param description: human-readable description
    :ptype description: str
    :param input_schema: JSON Schema for input arguments
    :ptype input_schema: dict[str, Any]
    :param required_permission: permission string the caller must
        hold; default-deny when no grant matches
    :ptype required_permission: str
    :param registry: target registry; defaults to module-level
    :ptype registry: ToolRegistry | None
    :return: decorator returning the handler unchanged
    :rtype: Callable[[ToolHandler], ToolHandler]
    """

    target = registry if registry is not None else _DEFAULT_REGISTRY

    def _decorator(handler: ToolHandler) -> ToolHandler:
        """register ``handler`` as :class:`McpTool` and return it unchanged."""
        target.register(
            McpTool(
                name=name,
                description=description,
                input_schema=input_schema,
                required_permission=required_permission,
                handler=handler,
            ),
        )
        return handler

    return _decorator


def get_default_registry() -> ToolRegistry:
    """return the module-level default tool registry.

    server entry-point modules call this to hand the registry to
    their :class:`McpServer`. tests typically construct their own
    registry instead of touching the default.

    :return: default registry instance
    :rtype: ToolRegistry
    """
    return _DEFAULT_REGISTRY


def reset_default_registry_for_testing() -> None:
    """clear the module-level default registry.

    test-only escape hatch. production code never resets the default
    registry; the singleton lives for the process's lifetime.

    :return: nothing
    :rtype: None
    """
    _DEFAULT_REGISTRY._tools.clear()  # noqa: SLF001
