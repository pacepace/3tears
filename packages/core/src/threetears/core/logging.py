"""Structured logging for 3tears -- re-exports from threetears.observe.logging.

All logging functionality lives in ``threetears.observe.logging``.  This
module re-exports the public API so that existing ``from threetears.core.logging
import get_logger`` imports continue to work without changes.
"""

from threetears.observe.logging import (
    ContextFormatter,
    ThreeTearsLogger,
    _color_enabled,
    _shorten_path,
    add_filter,
    clear_context,
    configure_logging,
    configure_third_party_logging,
    get_context,
    get_logger,
    path_strip_prefixes,
    set_context,
)

__all__ = [
    "ContextFormatter",
    "ThreeTearsLogger",
    "_color_enabled",
    "_shorten_path",
    "add_filter",
    "clear_context",
    "configure_logging",
    "configure_third_party_logging",
    "get_context",
    "get_logger",
    "path_strip_prefixes",
    "set_context",
]
