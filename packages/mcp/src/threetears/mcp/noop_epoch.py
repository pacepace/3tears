"""no-op :class:`EpochClient` / :class:`EpochListener` for stdio MCP servers.

stdio MCP servers are single-process, single-user (env-var admin
identity) by definition. cross-pod coherence is irrelevant -- there
are no sibling pods. but :class:`~threetears.mcp.auth.LocalGrantAuthorizer`
requires real :class:`~threetears.epoch.EpochClient` /
:class:`~threetears.epoch.EpochListener` references for its lifecycle
(prime via ``current``, subscribe to bumps).

these no-op helpers satisfy the framework's type contract without
opening NATS connections. every method returns immediately:

- :meth:`NoopEpochClient.current` returns ``0`` -- the script never
  observes a non-zero epoch.
- :meth:`NoopEpochClient.bump` returns ``0`` and does nothing -- the
  script is read-only on grants (mutations go through the REST
  surface that has its own real :class:`EpochClient`).
- :meth:`NoopEpochListener.subscribe` no-ops -- nothing to dispatch
  to.
- :meth:`NoopEpochListener.catch_up` no-ops -- there's no cache to
  refresh in admin-only mode.

the admin auto-grant short-circuits before any cache lookup, so
none of the no-op behaviour matters for the dominant CLI flow. if
a future stdio consumer needs non-admin grant semantics, swap in
real :class:`EpochClient` + :class:`EpochListener` against the
hub / metallm NATS bus -- mirror the production lifespan wiring.

these helpers implement the same Protocol shape that the
:class:`LocalGrantAuthorizer` constructor expects but are typed as
their own classes (no ``# type: ignore`` at the call site
required) -- the framework's protocols are runtime-checkable so
isinstance / structural-subtyping accepts them.
"""

from __future__ import annotations

from typing import Any

from threetears.nats.subjects import Subject

__all__ = [
    "NoopEpochClient",
    "NoopEpochListener",
]


class NoopEpochClient:
    """no-op :class:`~threetears.epoch.EpochClient` for stdio MCP servers.

    matches :class:`EpochClient` shape (``current``, ``bump``)
    without opening any Postgres or NATS connection. every method
    returns ``0`` immediately. used by per-product MCP server
    entry points (``metallm-mcp-server.py``, ``aibot-mcp-server.py``)
    that don't need cross-pod coherence.

    consumers wire this in place of a real :class:`EpochClient` and
    pass it to :class:`~threetears.mcp.auth.LocalGrantAuthorizer`'s
    ``epoch_client`` constructor argument. the authorizer's
    ``allows`` short-circuits via ``Identity.is_admin=True`` before
    consulting the cache, so the underlying epoch state never gets
    read.
    """

    async def current(self, subject: Subject) -> int:
        """return ``0`` -- stdio mode never observes a non-zero epoch.

        :param subject: target subject (ignored; no row is ever read)
        :ptype subject: Subject
        :return: ``0`` always
        :rtype: int
        """
        return 0

    async def bump(
        self,
        subject: Subject,
        payload: dict[str, Any] | None = None,
    ) -> int:
        """no-op -- stdio MCP scripts are read-only on grants.

        grants flow through the REST admin surface which has its
        own real :class:`EpochClient`; the stdio script never bumps.

        :param subject: target subject (ignored)
        :ptype subject: Subject
        :param payload: opaque hint (ignored)
        :ptype payload: dict[str, Any] | None
        :return: ``0`` always
        :rtype: int
        """
        return 0


class NoopEpochListener:
    """no-op :class:`~threetears.epoch.EpochListener` for stdio MCP servers.

    matches :class:`EpochListener` shape (``subscribe``, ``catch_up``,
    ``echo``, ``last_seen``) without opening a NATS subscription.
    every method returns ``0`` or no-ops immediately.

    the admin auto-grant short-circuits before any cache lookup, so
    the listener never has anything to dispatch.
    """

    async def subscribe(
        self,
        subject: Subject,
        on_bump: Any,
    ) -> None:
        """no-op -- the admin auto-grant short-circuits before cache lookup.

        :param subject: target subject (ignored)
        :ptype subject: Subject
        :param on_bump: bump callback (ignored)
        :ptype on_bump: Any
        :return: nothing
        :rtype: None
        """
        return None

    async def catch_up(
        self,
        subject: Subject,
        on_bump: Any,
    ) -> int:
        """no-op -- there is no cache to refresh in admin-only mode.

        :param subject: target subject (ignored)
        :ptype subject: Subject
        :param on_bump: bump callback (ignored)
        :ptype on_bump: Any
        :return: ``0`` (matches :meth:`EpochListener.catch_up` return type)
        :rtype: int
        """
        return 0

    async def echo(
        self,
        subject: Subject,
        echoed_epoch: int,
        on_bump: Any,
    ) -> None:
        """no-op -- per-message echo isn't relevant in admin-only stdio mode.

        :param subject: target subject (ignored)
        :ptype subject: Subject
        :param echoed_epoch: echoed epoch value (ignored)
        :ptype echoed_epoch: int
        :param on_bump: bump callback (ignored)
        :ptype on_bump: Any
        :return: nothing
        :rtype: None
        """
        return None

    def last_seen(self, subject: Subject) -> int:
        """return ``0`` -- the listener never observes any bumps.

        :param subject: target subject (ignored)
        :ptype subject: Subject
        :return: ``0`` always
        :rtype: int
        """
        return 0
