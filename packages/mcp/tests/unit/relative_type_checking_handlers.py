"""handlers whose module imports its annotation types relatively, and only for type checking.

a relative import binds a name like any other, so an annotation naming it is accounted for and draws no
warning. it never binds :mod:`threetears.mcp`'s ``CallContext``, though a package of the importer's
own may carry a ``threetears.mcp`` module that exports one: that one is the handler's own type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conftest import LoopbackMcp as Peer
    from .threetears.mcp import CallContext  # type: ignore[import-not-found]


async def relative_peer(*, peer: Peer | None = None) -> str:
    """a type imported relatively for type checking; it names nothing unaccounted for."""
    return "ok"


async def relative_lookalike_context(*, ctx: CallContext | None = None) -> str:
    """a ``CallContext`` from the importer's own ``threetears.mcp`` module, not the MCP call's."""
    return "ok"
