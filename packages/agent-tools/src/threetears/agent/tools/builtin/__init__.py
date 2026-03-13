"""Built-in tool registrations.

Each tool is in its own module.  Dependencies are optional — if a dep is
missing, that tool is skipped with a warning log.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from threetears.core.logging import get_logger

if TYPE_CHECKING:
    from threetears.agent.tools.registry import ToolRegistry

log = get_logger(__name__)


def register_builtins(registry: ToolRegistry) -> None:
    """Register all available built-in tools on *registry*.

    Missing optional dependencies cause individual tools to be skipped,
    not a crash.
    """
    # Each entry: (tool_type, module_path, factory_name)
    _BUILTINS: list[tuple[str, str, str]] = [
        ("calculator", "threetears.agent.tools.builtin.calculator", "create_calculator_tool"),
        ("unit_converter", "threetears.agent.tools.builtin.unit_converter", "create_unit_converter_tool"),
        ("timezone_converter", "threetears.agent.tools.builtin.timezone_converter", "create_timezone_converter_tool"),
        ("dictionary", "threetears.agent.tools.builtin.dictionary", "create_dictionary_tool"),
        ("current_date", "threetears.agent.tools.builtin.current_date", "create_current_date_tool"),
        ("web_search", "threetears.agent.tools.builtin.web_search", "create_web_search_tool"),
        ("web_fetch", "threetears.agent.tools.builtin.web_fetch", "create_web_fetch_tool"),
    ]

    import importlib

    for tool_type, module_path, factory_name in _BUILTINS:
        try:
            mod = importlib.import_module(module_path)
            factory = getattr(mod, factory_name)
            registry.register(tool_type, factory)
        except ImportError as exc:
            log.warning(
                "Skipping built-in tool (missing dependency)",
                extra={"extra_data": {"tool_type": tool_type, "error": str(exc)}},
            )
        except Exception as exc:
            log.warning(
                "Skipping built-in tool (registration error)",
                extra={"extra_data": {"tool_type": tool_type, "error": str(exc)}},
            )
