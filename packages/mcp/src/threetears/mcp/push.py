"""the notifications a server starts: list changes and resource updates, pushed to every client that can take them.

two eras, one server. on the handshake-era protocols (up to 2025-11-25) a connection that outlives its requests -- a
stateful Streamable-HTTP session, or stdio -- has a standalone stream, and :class:`Pushes` keeps each such
connection, with the resource URIs it subscribed to, until it closes. on 2026-07-28 there are no sessions: a client
asks for these on a ``subscriptions/listen`` request, whose response stream the SDK's listen handler serves from a
subscription bus; :class:`Pushes` publishes on that bus too.

what is pushed names only what changed (a tool list, a resource's URI); reading it again is a request of its own,
whose caller is resolved and authorized as every request's is.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from mcp.server.connection import Connection
from mcp.server.session import ServerSession
from mcp.server.subscriptions import (
    InMemorySubscriptionBus,
    ResourcesListChanged,
    ResourceUpdated,
    ServerEvent,
    ToolsListChanged,
)
from threetears.observe import contain, get_logger

from threetears.mcp._mcp_sdk_internals import connection_of

__all__ = ["Pushes"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class _Peer:
    """a connection the server can push to, and the resource URIs it subscribed to.

    :ivar connection: the SDK's connection
    :ivar uris: the URIs it subscribed to with ``resources/subscribe``
    """

    connection: Connection
    uris: frozenset[str] = frozenset()


class Pushes:
    """the connections and listen streams a server pushes its change notifications to."""

    def __init__(self) -> None:
        """start with no connection, and a bus no listen stream has subscribed to yet."""
        # a read-only snapshot replaced whole on a change, keyed by the connection's identity; an entry goes when its
        # connection closes (the connection's teardown stack drops it)
        self._peers: Mapping[int, _Peer] = MappingProxyType({})
        self._bus = InMemorySubscriptionBus()

    @property
    def bus(self) -> InMemorySubscriptionBus:
        """the bus the 2026-07-28 listen streams read from.

        :return: the bus
        :rtype: mcp.server.subscriptions.InMemorySubscriptionBus
        """
        return self._bus

    def track(self, session: ServerSession) -> None:
        """keep the connection ``session`` arrived on, until it closes, when it has a standalone stream.

        :param session: a request's session
        :ptype session: mcp.server.session.ServerSession
        :return: nothing
        :rtype: None
        """
        connection = connection_of(session)
        key = id(connection)
        if key in self._peers or not connection.has_standalone_channel:
            return
        self._peers = MappingProxyType({**self._peers, key: _Peer(connection)})
        connection.exit_stack.callback(self._forget, key)

    def _forget(self, key: int) -> None:
        """drop a closed connection.

        :param key: the connection's key
        :ptype key: int
        :return: nothing
        :rtype: None
        """
        self._peers = MappingProxyType({held: peer for held, peer in self._peers.items() if held != key})

    def subscribe(self, session: ServerSession, uri: str) -> bool:
        """record that the connection ``session`` arrived on wants updates of ``uri``.

        :param session: the ``resources/subscribe`` request's session
        :ptype session: mcp.server.session.ServerSession
        :param uri: the resource's URI
        :ptype uri: str
        :return: ``False`` when the connection is not one the server can push to
        :rtype: bool
        """
        key = id(connection_of(session))
        peer = self._peers.get(key)
        if peer is None:
            return False
        self._peers = MappingProxyType({**self._peers, key: _Peer(peer.connection, peer.uris | {uri})})
        return True

    def unsubscribe(self, session: ServerSession, uri: str) -> None:
        """record that the connection ``session`` arrived on no longer wants updates of ``uri``.

        :param session: the ``resources/unsubscribe`` request's session
        :ptype session: mcp.server.session.ServerSession
        :param uri: the resource's URI
        :ptype uri: str
        :return: nothing
        :rtype: None
        """
        key = id(connection_of(session))
        peer = self._peers.get(key)
        if peer is not None:
            self._peers = MappingProxyType({**self._peers, key: _Peer(peer.connection, peer.uris - {uri})})

    async def tools_changed(self) -> None:
        """send ``notifications/tools/list_changed`` to every connection and listen stream.

        :return: nothing
        :rtype: None
        """
        await self._push(lambda peer: peer.connection.send_tool_list_changed(), ToolsListChanged())

    async def resources_changed(self) -> None:
        """send ``notifications/resources/list_changed`` to every connection and listen stream.

        :return: nothing
        :rtype: None
        """
        await self._push(lambda peer: peer.connection.send_resource_list_changed(), ResourcesListChanged())

    async def resource_updated(self, uri: str) -> None:
        """send ``notifications/resources/updated`` for ``uri`` to every connection and stream subscribed to it.

        :param uri: the resource's URI
        :ptype uri: str
        :return: nothing
        :rtype: None
        """
        await self._push(
            lambda peer: peer.connection.send_resource_updated(uri),
            ResourceUpdated(uri=uri),
            only=lambda peer: uri in peer.uris,
        )

    async def _push(
        self,
        send: Callable[[_Peer], Awaitable[None]],
        event: ServerEvent,
        *,
        only: Callable[[_Peer], bool] = lambda _peer: True,
    ) -> None:
        """send one notification to each connection ``only`` admits, then publish ``event`` on the bus.

        one connection's failure is logged and the rest are still sent to: a notification is advice to the client,
        and one broken stream must not keep it from the others.

        :param send: sends it to one connection
        :ptype send: Callable[[_Peer], Awaitable[None]]
        :param event: the same notification for the listen streams
        :ptype event: mcp.server.subscriptions.ServerEvent
        :param only: which connections it goes to
        :ptype only: Callable[[_Peer], bool]
        :return: nothing
        :rtype: None
        """
        for peer in list(self._peers.values()):
            if not only(peer):
                continue
            try:
                await send(peer)
            except Exception as exc:
                contain(exc, log=log, what="MCP push: a notification could not be sent to a connection")
        await self._bus.publish(event)
