"""Class-based tool factory registry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from langchain_core.tools import BaseTool

from threetears.core.frozen_state import frozen, with_entry, without_entry
from threetears.observe import contain, get_logger

__all__ = [
    "ToolFactory",
    "ToolRegistry",
]

ToolFactory = Callable[[dict[str, Any], str], BaseTool]

log = get_logger(__name__)


class ToolRegistry:
    """Registry mapping tool type names to factory functions.

    Multiple registries can coexist — this is a class, not a module-level dict.
    """

    def __init__(self) -> None:
        # a read-only snapshot, replaced whole on each register or unregister
        self._factories: Mapping[str, ToolFactory] = frozen()

    def register(self, tool_type: str, factory: ToolFactory) -> None:
        """Register a factory function for a tool type."""
        self._factories = with_entry(self._factories, tool_type, factory)

    def unregister(self, tool_type: str) -> None:
        """Remove a registered factory."""
        self._factories = without_entry(self._factories, tool_type)

    def list_types(self) -> list[str]:
        """Return registered tool type names."""
        return list(self._factories.keys())

    def has(self, tool_type: str) -> bool:
        """Check whether a tool type is registered."""
        return tool_type in self._factories

    def create(self, tool_type: str, config: dict[str, Any], description: str) -> BaseTool:
        """Create a single tool from a registered factory.

        Raises ``KeyError`` if *tool_type* is not registered.
        """
        factory = self._factories.get(tool_type)
        if factory is None:
            raise KeyError(f"Unknown tool type: {tool_type}")
        return factory(config, description)

    async def create_tools(self, tool_configs: list[dict[str, Any]]) -> list[BaseTool]:
        """Create tools from a list of config dicts.

        Each dict must have ``tool_type`` and ``description`` keys, plus an
        optional ``config`` dict.  Unknown tool types are logged and skipped
        (never crash).
        """
        tools: list[BaseTool] = []
        for cfg in tool_configs:
            tool_type = cfg["tool_type"]
            factory = self._factories.get(tool_type)
            if factory is None:
                log.warning(
                    "Unknown tool type, skipping",
                    extra={"extra_data": {"tool_type": tool_type}},
                )
                continue
            try:
                t = factory(cfg.get("config", {}), cfg["description"])
                tools.append(t)
            except Exception as exc:
                contain(exc, log=log, what="Failed to create tool", extra={"tool_type": tool_type, "error": str(exc)})
        return tools
