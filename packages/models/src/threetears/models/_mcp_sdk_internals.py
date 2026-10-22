"""the one owner of the MCP SDK (``mcp``) private member ``threetears.models`` touches.

mcp 2.x keeps a server's request handlers in a private table, ``Server._request_handlers``, behind
a public ``get_request_handler`` / ``add_request_handler``. There is no public way to give a copy
of a server a table of its own, and the CLI pool needs exactly that: a shallow copy shares the
table, so rebinding ``tools/call`` on a borrower's copy would rebind it on the original that every
borrower shares (:func:`threetears.models.claude_cli_pool._context_bound_server`).

mcp 1.x keeps the table public (``Server.request_handlers``) and never reaches this module. When
3tears drops mcp 1.x (the owner decides when), this stays until the SDK grows a public way to copy the table.
"""

from __future__ import annotations

from typing import Any

__all__ = ["give_own_request_handlers"]


def give_own_request_handlers(copy: Any, original: Any) -> None:
    """give ``copy`` a handler table of its own, holding what ``original``'s holds now.

    :param copy: a shallow copy of ``original`` (``copy.copy``), which shares its table until this
    :ptype copy: Any
    :param original: the mcp 2.x server it was copied from
    :ptype original: Any
    :return: nothing
    :rtype: None
    """
    table = original._request_handlers
    copy._request_handlers = dict(table)
