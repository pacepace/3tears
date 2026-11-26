"""the one owner of the MCP SDK (``mcp``) private member ``threetears.mcp`` touches.

mcp 2.x (verified 2.3.0) builds a per-request ``ServerSession`` over the connection's ``Connection``, which holds
what a server needs to push to a client outside any request of its own: the standalone channel
(``send_tool_list_changed`` and the like), whether there is one (``has_standalone_channel``), and the teardown
stack that runs when the connection closes (``exit_stack``, which the SDK documents for handlers to push cleanup
onto). A lowlevel handler is given only the session, and the session keeps the connection private
(``ServerSession._connection``); the SDK's own context rework (its TODO L54) is to hand handlers the connection.
Without it a server could not drop a closed session from the set it pushes to, and that set would grow for the
server's life.

this stays until the SDK hands a lowlevel handler its connection.
"""

from __future__ import annotations

from mcp.server.connection import Connection
from mcp.server.session import ServerSession

__all__ = ["connection_of"]


def connection_of(session: ServerSession) -> Connection:
    """the connection a request's session proxies.

    :param session: the request's session (``ServerRequestContext.session``)
    :ptype session: mcp.server.session.ServerSession
    :return: its connection
    :rtype: mcp.server.connection.Connection
    """
    return session._connection
