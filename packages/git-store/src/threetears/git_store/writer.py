"""one writer per repository: the process holding the repository's lease commits; the rest forward.

:class:`WriterRole` holds the lease ``"{key}.{branch}"`` with 3tears'
:meth:`~threetears.core.coordination.KVLease.hold`, which renews it in the background and reports
its loss. On winning it the repository's takeover runs once (catch up with origin, recover an
interrupted change, push what origin lacks), then the role serves the change sets other processes
forward to it (:func:`threetears.nats.forward.serve_owner`) for exactly as long as it holds the
lease. A product with operations of its own registers a handler on the same lease with
:meth:`WriterRole.serve`.

The lease is coordination, not the fence. Two processes may briefly both believe they hold it; the
forward subject's queue group still hands each request to one of them, and the durable fences are
git's: on one disk the index lock and the compare-and-swap check, across machines the push that git
refuses unless it is a fast-forward.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import TYPE_CHECKING, Final

from threetears.core.coordination import HeldLease, KVLease, LeaseLossReason, LeaseUnavailable
from threetears.nats.forward import ForwardHandler, forward, serve_owner
from threetears.observe import get_logger, spawn_background

if TYPE_CHECKING:
    from threetears.git_store.envelope import ForwardEnvelope
    from threetears.nats import NatsClient

__all__ = ["FORWARD_FAMILY", "WriterRole"]

log = get_logger(__name__)

#: the forward family every git-store request rides, so a NATS grant can name git-store forwards
#: (``Subjects.forward_scoped_wildcard("git-store")``) without granting every forward subject.
FORWARD_FAMILY: Final[str] = "git-store"

#: the name of the handler that runs change sets; product handlers take other names.
COMMIT_HANDLER: Final[str] = "commit"


class WriterRole:
    """this process's claim on being a repository's writer.

    :ivar lease_key: the lease's key, ``"{key}.{branch}"``
    """

    def __init__(
        self,
        nats: NatsClient,
        *,
        lease_key: str,
        bucket: str,
        ttl: timedelta,
        renew_every: timedelta,
        takeover: Callable[[], Awaitable[None]],
        forward_timeout: timedelta,
        log_extra: dict[str, str] | None = None,
        envelope: ForwardEnvelope,
    ) -> None:
        """describe the lease; nothing is claimed until :meth:`claim`.

        :param nats: the connected client the lease and the forwards use
        :ptype nats: NatsClient
        :param lease_key: the lease's key
        :ptype lease_key: str
        :param bucket: the lease bucket's name (the client adds its namespace)
        :ptype bucket: str
        :param ttl: how long the lease outlives a missed renewal
        :ptype ttl: timedelta
        :param renew_every: how often it is renewed
        :ptype renew_every: timedelta
        :param takeover: run once each time the lease is won, before any change is served
        :ptype takeover: Callable[[], Awaitable[None]]
        :param forward_timeout: how long a forwarded request waits for the writer's answer
        :ptype forward_timeout: timedelta
        :param log_extra: context for every log line
        :ptype log_extra: dict[str, str] | None
        :param envelope: seals what this process forwards and admits only sealed requests when it serves
        :ptype envelope: ForwardEnvelope
        :return: None
        :rtype: None
        """
        self.lease_key = lease_key
        self._envelope = envelope
        self._nats = nats
        self._lease = KVLease(nats, bucket_name=bucket)
        self._ttl = ttl
        self._renew_every = renew_every
        self._takeover = takeover
        self._forward_timeout = forward_timeout
        self._log_extra = {"lease_key": lease_key, **(log_extra or {})}
        self._handlers: dict[str, ForwardHandler] = {}
        self._held: HeldLease | None = None
        self._serving: AsyncExitStack | None = None
        self._claiming = asyncio.Lock()

    @property
    def is_writer(self) -> bool:
        """whether this process holds the lease now.

        :return: ``True`` while held and not lost
        :rtype: bool
        """
        return self._held is not None and self._held.held

    @property
    def pod_id(self) -> str:
        """the holder name this process claims the lease under.

        :return: the holder id
        :rtype: str
        """
        return self._lease.pod_id

    async def holder(self) -> str | None:
        """who holds the lease now, without contending for it.

        :return: the holder's id, or ``None`` when nobody holds it
        :rtype: str | None
        """
        return await self._lease.holder(self.lease_key)

    def subject_key(self, name: str) -> str:
        """the forward key a handler is served under.

        :param name: the handler's name
        :ptype name: str
        :return: the key
        :rtype: str
        """
        return f"{self.lease_key}#{name}"

    async def serve(self, name: str, handler: ForwardHandler) -> None:
        """serve ``handler`` under ``name`` whenever this process is the writer.

        :param name: the handler's name; :data:`COMMIT_HANDLER` is the change-set handler's
        :ptype name: str
        :param handler: ``bytes -> bytes``; what it raises reaches the sender as
            :class:`~threetears.nats.forward.ForwardedHandlerError`
        :ptype handler: ForwardHandler
        :return: None
        :rtype: None
        :raises ValueError: when ``name`` is already served
        """
        if name in self._handlers:
            raise ValueError(f"{name!r} is already served on {self.lease_key}")
        self._handlers[name] = handler
        if self._serving is not None:
            await self._serve_one(self._serving, name, handler)

    async def _serve_one(self, stack: AsyncExitStack, name: str, handler: ForwardHandler) -> None:
        """subscribe one handler for the length of ``stack``.

        :param stack: the serving stack of the current holding
        :ptype stack: AsyncExitStack
        :param name: the handler's name
        :ptype name: str
        :param handler: the handler
        :ptype handler: ForwardHandler
        :return: None
        :rtype: None
        """
        envelope = self._envelope

        async def admitted(data: bytes) -> bytes:
            return await handler(await envelope.open(name, data))

        # sender-posture: signed-envelope -- the deployment's Ed25519 seal, pinned key, single-use id (envelope.py)
        await stack.enter_async_context(
            serve_owner(self._nats, self.subject_key(name), admitted, family=FORWARD_FAMILY)
        )

    async def forward(self, name: str, payload: bytes) -> bytes:
        """send a request to the writer's ``name`` handler and return its reply.

        :param name: the handler's name
        :ptype name: str
        :param payload: the request
        :ptype payload: bytes
        :return: the reply
        :rtype: bytes
        :raises NoOwnerError: when no process serves it now
        :raises ForwardedHandlerError: when the handler raised
        """
        return await forward(
            self._nats,
            self.subject_key(name),
            self._envelope.seal(name, payload),
            timeout=self._forward_timeout,
            family=FORWARD_FAMILY,
        )

    async def claim(self) -> bool:
        """try once to become the writer; on winning, run takeover and start serving.

        :return: ``True`` when this process is the writer on return
        :rtype: bool
        :raises Exception: whatever takeover raised; the lease is released first
        """
        async with self._claiming:
            if self.is_writer:
                return True
            try:
                held = await self._lease.hold(
                    self.lease_key,
                    ttl=self._ttl,
                    renew_every=self._renew_every,
                    name="the git-store writer lease",
                    log_extra=self._log_extra,
                    on_lost=self._on_lost,
                )
            except LeaseUnavailable:
                return False
            # this process counts as the writer only once takeover has made the working tree current
            # and forwarded changes are being served; until then a change here still forwards or waits
            try:
                await self._envelope.bind()
                await self._takeover()
                stack = AsyncExitStack()
                try:
                    for name, handler in self._handlers.items():
                        await self._serve_one(stack, name, handler)
                except BaseException:
                    await stack.aclose()
                    raise
                self._serving = stack
            except BaseException:
                await held.release()
                raise
            self._held = held
            log.info(
                "git_store.writer.acquired: this process is the repository's writer",
                extra={"extra_data": self._log_extra},
            )
            return held.held

    def _on_lost(self, reason: LeaseLossReason) -> None:
        """the lease was lost: stop serving at once. Runs on the renewal task; must not block.

        :param reason: why it was lost
        :ptype reason: LeaseLossReason
        :return: None
        :rtype: None
        """
        log.warning(
            "git_store.writer.lost: the writer lease was lost; nothing more is committed here",
            extra={"extra_data": {**self._log_extra, "reason": reason.value}},
        )
        spawn_background(self._stop_serving(), name=f"git-store-unserve:{self.lease_key}", logger=log)

    async def _stop_serving(self) -> None:
        """drop every forward subscription of the holding that ended.

        :return: None
        :rtype: None
        """
        stack, self._serving = self._serving, None
        if stack is not None:
            await stack.aclose()

    async def release(self) -> None:
        """stop serving and give the lease up, for the next writer to take.

        :return: None
        :rtype: None
        """
        async with self._claiming:
            await self._stop_serving()
            held, self._held = self._held, None
            if held is not None and held.held:
                await held.release()
                log.info(
                    "git_store.writer.released: the writer lease was given up", extra={"extra_data": self._log_extra}
                )
