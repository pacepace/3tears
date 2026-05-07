"""Built-in tool registrations.

Each tool is in its own module.  Dependencies are optional — if a dep is
missing, that tool is skipped with a warning log.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from threetears.observe import get_logger

__all__ = [
    "STANDARD_BUILTIN_FACTORIES",
    "make_standard_builtins",
    "register_builtins",
]

if TYPE_CHECKING:
    from threetears.agent.tools.base_tool import TearsTool
    from threetears.agent.tools.registry import ToolRegistry

log = get_logger(__name__)


def _build_calculator() -> TearsTool:
    """zero-config calculator TearsTool factory."""
    from threetears.agent.tools.builtin.calculator import CalculatorTool

    return CalculatorTool()


def _build_current_date() -> TearsTool:
    """zero-config current_date TearsTool factory.

    timezone is resolved per-call from the :class:`ToolCallScope` /
    :class:`CallContext` (browser-supplied / channel-adapter-resolved
    user timezone) so no construction-time tz is needed in dev.
    """
    from threetears.agent.tools.builtin.current_date import CurrentDateTool

    return CurrentDateTool()


def _build_dictionary() -> TearsTool:
    """zero-config dictionary TearsTool factory (defaults to ``en``)."""
    from threetears.agent.tools.builtin.dictionary import DictionaryTool

    return DictionaryTool()


def _build_timezone_converter() -> TearsTool:
    """zero-config timezone_converter TearsTool factory."""
    from threetears.agent.tools.builtin.timezone_converter import TimezoneConverterTool

    return TimezoneConverterTool()


def _build_unit_converter() -> TearsTool:
    """zero-config unit_converter TearsTool factory."""
    from threetears.agent.tools.builtin.unit_converter import UnitConverterTool

    return UnitConverterTool()


# canonical map from ``mcp_name`` to a zero-arg factory returning a
# ready-to-register :class:`TearsTool` instance. mirrors
# :data:`threetears.agent.tools.aliases.STANDARD_TOOLS`: every entry
# in that frozenset MUST appear here, and every key here MUST appear
# in that frozenset. callers that need a slice (e.g. registering only
# the names matching an access pattern) iterate this map and filter.
STANDARD_BUILTIN_FACTORIES: dict[str, callable] = {  # type: ignore[type-arg]
    "threetears.calculator": _build_calculator,
    "threetears.current_date": _build_current_date,
    "threetears.dictionary": _build_dictionary,
    "threetears.timezone_converter": _build_timezone_converter,
    "threetears.unit_converter": _build_unit_converter,
}


def make_standard_builtins(
    mcp_names: Iterable[str] | None = None,
) -> list[TearsTool]:
    """build a list of zero-config standard builtin TearsTool instances.

    used by the SDK's in-process dev strategy to materialise the
    ``standard`` group on agent boot so the readiness gate finds them
    in the catalog. each builtin has a zero-arg constructor (or
    accepts construction with all defaults) so this helper needs no
    config dict.

    :param mcp_names: iterable of canonical ``threetears.<name>`` mcp
        names to materialise; ``None`` returns every standard builtin.
        unknown names are silently skipped (legitimate when an agent
        access pattern is a glob covering names that match other
        groups -- callers filter beforehand)
    :ptype mcp_names: Iterable[str] | None
    :return: ordered list of TearsTool instances ready for
        :meth:`ToolServer.register_tool`
    :rtype: list[TearsTool]
    """
    selected = (
        list(STANDARD_BUILTIN_FACTORIES.keys())
        if mcp_names is None
        else [n for n in mcp_names if n in STANDARD_BUILTIN_FACTORIES]
    )
    return [STANDARD_BUILTIN_FACTORIES[name]() for name in selected]


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
        ("analyze_media", "threetears.agent.tools.builtin.analyze_media", "create_analyze_media_tool"),
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
