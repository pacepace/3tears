"""Class-based tool factory registry."""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.tools import BaseTool

ToolFactory = Callable[[dict[str, Any], str], BaseTool]


class ToolRegistry:
    """Registry mapping tool type names to factory functions.

    Multiple registries can coexist — this is a class, not a module-level dict.
    """

    def __init__(self) -> None:
        self._factories: dict[str, ToolFactory] = {}

    def register(self, tool_type: str, factory: ToolFactory) -> None:
        """Register a factory function for a tool type."""
        self._factories[tool_type] = factory

    def unregister(self, tool_type: str) -> None:
        """Remove a registered factory."""
        self._factories.pop(tool_type, None)

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
        from threetears.core.logging import get_logger

        _logger = get_logger(__name__)
        tools: list[BaseTool] = []
        for cfg in tool_configs:
            tool_type = cfg["tool_type"]
            factory = self._factories.get(tool_type)
            if factory is None:
                _logger.warning(
                    "Unknown tool type, skipping",
                    extra={"extra_data": {"tool_type": tool_type}},
                )
                continue
            try:
                t = factory(cfg.get("config", {}), cfg["description"])
                tools.append(t)
            except Exception as exc:
                _logger.warning(
                    "Failed to create tool",
                    extra={"extra_data": {"tool_type": tool_type, "error": str(exc)}},
                )
        return tools
