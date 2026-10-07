"""CoalescedRun -- an operation run by one replica at a time, where a request during a run runs it once more.

**The shape.** A refresh, a rebuild, a sync: an operation that makes some state match a source,
asked for by signals that arrive whenever they like -- a schedule, an upstream "ready" call, an
operator. Two runs at once would race each other's writes, so one replica runs it while it holds a
:class:`~threetears.core.coordination.lease.KVLease`. A signal that lands while it runs must not be
lost, because the source may have moved after the run read it; and a hundred signals during one run
need one more run, not a hundred.

**How.** :meth:`CoalescedRun.request` records that a run is wanted, as one key in the lease's own
bucket (beside the lease key, under the same owner scope, so a pod granted its own keys of the
platform's ``leases`` bucket needs nothing more). :meth:`CoalescedRun.drain` holds the lease and,
while a request is recorded, takes it (a compare-and-swap delete, so a request written meanwhile is
never taken with it) and runs once. Requests made during a run are one recorded request, so they
are one more run. A replica that cannot hold the lease leaves its request for the holder and
returns.

**The window at the end.** A request can land after the holder found none and before it lets the
lease go; the requester could not hold the lease, so it left the request. The holder therefore looks
again after it lets go, and holds the lease again when it finds one. Every request is run by some
replica that drains after it, as long as one does: a holder that dies leaves its request recorded
for the next :meth:`~CoalescedRun.drain` anywhere (the next signal, or a schedule).

**A lost lease ends the run.** The lease is renewed while the operation runs
(:meth:`~threetears.core.coordination.lease.KVLease.hold`). If it is lost, another replica may
already be running, so the run is cancelled, the request is recorded again (nothing it was asked to
do is known to be done) and :class:`~threetears.core.coordination.lease.LeaseLost` is raised.

**A run that fails is raised, and not retried.** Its request was taken; retrying a run that fails
every time would hold the lease forever. The next request runs it again.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import timedelta
from typing import Any, Final
from uuid import uuid7

from threetears.observe import get_logger

from threetears.core.coordination.lease import HeldLease, KVLease, LeaseLost, LeaseUnavailable

__all__ = ["CoalescedRun"]

log = get_logger(__name__)

#: what the request key adds to the operation's key
_REQUESTED_SUFFIX: Final = ".requested"


class CoalescedRun:
    """an operation run under a lease on one replica at a time; a request during a run runs it once more.

    :param lease: the lease factory; replicas of one deployment share its bucket and owner scope
    :ptype lease: KVLease
    :param key: the operation's lease key (its request key is this plus ``.requested``)
    :ptype key: str
    :param run: the operation
    :ptype run: Callable[[], Awaitable[Any]]
    :param ttl: how long the lease outlives a missed renewal
    :ptype ttl: timedelta
    :param renew_every: how often the lease is renewed while the operation runs; shorter than ``ttl``
    :ptype renew_every: timedelta
    :raises ValueError: when ``renew_every`` is not shorter than ``ttl``
    """

    def __init__(
        self,
        lease: KVLease,
        key: str,
        run: Callable[[], Awaitable[Any]],
        *,
        ttl: timedelta,
        renew_every: timedelta,
    ) -> None:
        if renew_every >= ttl:
            raise ValueError(f"renew_every {renew_every} must be shorter than ttl {ttl}, or the lease lapses mid-run")
        self._lease = lease
        self._key = key
        self._run = run
        self._ttl = ttl
        self._renew_every = renew_every
        self._requested_key = lease.stored_key(f"{key}{_REQUESTED_SUFFIX}")

    async def request(self) -> None:
        """record that a run is wanted; whichever replica drains next runs it.

        :return: nothing
        :rtype: None
        """
        bucket = await self._lease.bucket()
        # a fresh value every time, so a request made while one is being taken has its own revision
        await bucket.put(key=self._requested_key, value=str(uuid7()).encode())

    async def requested(self) -> bool:
        """whether a run is wanted and not yet taken.

        :return: True while a request is recorded
        :rtype: bool
        """
        bucket = await self._lease.bucket()
        return await bucket.get_entry(key=self._requested_key) is not None

    async def drain(self) -> int:
        """run once per recorded request while this replica can hold the lease.

        :return: how many runs this call made; 0 when nothing was wanted or another replica holds
            the lease (which then runs what was asked)
        :rtype: int
        :raises LeaseLost: when the lease was lost during a run; the run was cancelled and the
            request recorded again
        :raises Exception: what the operation raised; its request is taken and not retried
        """
        runs = 0
        wanted = await self.requested()
        while wanted:
            try:
                held = await self._lease.hold(
                    self._key, ttl=self._ttl, renew_every=self._renew_every, log_extra={"operation": self._key}
                )
            except LeaseUnavailable:
                log.info(
                    "another replica holds the lease; it runs what was asked", extra={"extra_data": {"key": self._key}}
                )
                break
            async with held:
                while await self._take():
                    await self._run_holding(held)
                    runs += 1
            # a request may have landed after the last take and before the release
            wanted = await self.requested()
        return runs

    async def _take(self) -> bool:
        """take the recorded request, if there is one.

        :return: True when a request was taken
        :rtype: bool
        """
        bucket = await self._lease.bucket()
        taken = False
        entry = await bucket.get_entry(key=self._requested_key)
        while entry is not None and not taken:
            # only the revision read: a request written since is a new one, left for the next take
            taken = await bucket.delete(key=self._requested_key, revision=entry[1])
            if not taken:
                entry = await bucket.get_entry(key=self._requested_key)
        return taken

    async def _run_holding(self, held: HeldLease) -> None:
        """run the operation once, cancelling it if the lease is lost meanwhile.

        :param held: the lease, renewing
        :ptype held: HeldLease
        :return: nothing
        :rtype: None
        :raises LeaseLost: when the lease was lost before the run ended
        """

        async def once() -> Any:
            return await self._run()

        work: asyncio.Task[Any] = asyncio.create_task(once(), name=f"coalesced-run:{self._key}")
        lost = asyncio.create_task(held.until_lost(), name=f"coalesced-run-lease:{self._key}")
        try:
            await asyncio.wait({work, lost}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            lost.cancel()
            # NOSILENT: the watch on the lease, cancelled on the line above once the run has ended
            with suppress(asyncio.CancelledError):
                await lost
        if not work.done():
            work.cancel()
            # NOSILENT: the run, cancelled on the line above because the lease under it was lost
            with suppress(asyncio.CancelledError):
                await work
            await self.request()
            log.warning(
                "lease lost during a run; cancelled it and asked again", extra={"extra_data": {"key": self._key}}
            )
            raise LeaseLost(
                f"the lease on {self._key!r} was lost during a run; the run was cancelled and asked for again"
            )
        work.result()
