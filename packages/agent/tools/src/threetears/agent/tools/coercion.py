"""input-value coercion helpers for TearsTool subclasses, under their agent-tools name.

defined in 3tears-tool-contract (:mod:`threetears.tool_contract.coercion`) beside :class:`TearsTool`, whose ``run``
applies them; this module re-exports the same functions.
"""

from __future__ import annotations

from threetears.tool_contract.coercion import coerce_value, normalize_kwargs

__all__ = [
    "coerce_value",
    "normalize_kwargs",
]
