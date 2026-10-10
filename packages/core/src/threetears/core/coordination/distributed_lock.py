"""A cross-pod job lock over NATS JetStream KV, held through :class:`~threetears.core.coordination.KVLease`.

``async with nats_distributed_lock(nats, "backup"):`` runs its body on one pod at a time. It is the
platform's one lease primitive with a context manager's shape: acquisition, background renewal by
compare-and-swap, loss detection, the maximum hold and the fenced release are all
:meth:`KVLease.hold <threetears.core.coordination.KVLease.hold>`'s. What stays here is only what a
``with`` block adds -- opening the lock's bucket, turning a held key into :class:`LockHeld`, and
interrupting the body when the hold is lost.

design notes
------------

- **The lease does the holding.** One :class:`~threetears.core.coordination.KVLease` per call, so
  the holder identity in the entry names this one hold and the release can never delete another's.
  Renewal runs every ``heartbeat`` (the lease's ``renew_every``), always as a compare-and-swap on the
  revision just read and only while the entry is this holder's; a failed renewal is retried while
  the entry cannot have expired; past ``max_hold`` renewal stops so the TTL hands a wedged holder's
  lock on. The release gives a renewal in flight a short while (one renewal interval, at most one
  KV operation's ceiling) -- never the TTL, so a silent broker cannot stretch a shutdown -- then
  deletes the entry only if it is still this hold's.
- **The holder is told when it loses the lock.** The context manager yields a :class:`LockHold`,
  whose :attr:`~LockHold.lost_reason` is a :class:`LockLossReason`. By default the body is also
  cancelled and the ``async with`` raises :class:`LockLost`: a body that keeps writing after its
  lock is gone is the damage a lock exists to prevent. ``cancel_on_loss=False`` is for a body whose
  correctness does not rest on the lock (it only saves duplicate work); it can still read the hold.
- **Bucket-level TTL, declared exactly as before.** The lock opens its bucket itself with
  :meth:`NatsClient.kv_bucket(name, ttl=ttl) <threetears.nats.NatsClient.kv_bucket>` and hands it to
  the lease bound. A fresh bucket therefore gets what it always got: ``max_age = ttl``, history 1,
  ``allow_msg_ttl`` on (the stream shape :mod:`threetears.nats.kv` declares every bucket with). An
  existing bucket is used as it stands: its ``max_age`` is never rewritten, and the declaring open
  reconciles only ``allow_direct`` / ``allow_msg_ttl``, as it always did. Entries carry NO per-key
  TTL (the lease's ``expire_entries`` is off): the bucket's ``max_age`` already removes an entry
  ``ttl`` after its last write, which is the orphan bound a dead holder needs, and a per-key TTL would
  need ``allow_msg_ttl`` on a bucket created before it was declared. Because the bucket's TTL is
  bucket-level and fixed at creation (and :meth:`NatsClient.kv_bucket` caches buckets by name), a
  caller passing a ``ttl`` that differs from the bucket's raises :class:`ValueError`; callers needing
  distinct TTLs use distinct ``bucket_name`` values.
- **The TTL is whole seconds.** It is the lease's TTL, which a lease entry expresses in whole
  seconds of at least one; a fractional TTL raises :class:`ValueError` (every caller uses the
  60-second default).
- **Graceful single-pod fallback.** ``client=None`` yields a hold that is never lost, without
  acquiring anything -- dev environments that do not run NATS.
- **Infrastructure callers only; it declares its bucket.** The lock opens its bucket with
  create-if-missing, which needs ``STREAM.CREATE`` -- a verb no agent or tool pod holds, and no pod
  grant names ``{ns}-scheduler-locks``. Every caller today is an infrastructure identity (hub sweeps
  and reconcilers, the scheduled-jobs tick, a derived collection's build lock). A pod needing
  cross-pod exclusion binds a hub-declared bucket through ``KVLease(create_if_missing=False)``.
- **It lives beside the lease, and needs no NATS client to import.** The lock is a shape over
  :class:`KVLease`, so it is core's, not the NATS package's (where it lived before 0.66.0, reaching
  up into core for the lease). Importing it loads neither ``nats-py`` nor ``nkeys``: it reaches the
  broker only through the :class:`~threetears.nats.kv.KvCapable` its caller hands it.
- **Bucket name namespacing.** The default bucket ``"scheduler-locks"`` rides through
  :meth:`NatsClient.kv_bucket` and picks up the client's ``nats_subject_namespace`` prefix (the
  resulting bucket is ``{namespace}-scheduler-locks``).

rolling upgrade contract
------------------------

Releases before 0.66.0 wrote a lock entry's value as the holder's raw token -- 32 lowercase hex
characters, nothing else. From 0.66.0 the value is the :class:`KVLease` JSON envelope
(``{"holder": ..., "expires_at": ..., "acquired_at": ...}``). During a deploy an old replica and a new
one contend for the same key in the same bucket, and neither may ever take the other's held lock
for a free one:

- **A new replica treats an old entry as another holder's.** The lease's create-if-absent fails on
  the existing key; the lease then reads it, cannot decode it as an envelope -- including an
  all-digit token, which parses as a JSON number -- and treats it as held by somebody else: the
  caller gets :class:`LockHeld`, and a renewal or release that meets one reports the lock
  :attr:`~LockLossReason.TAKEN` and never writes or deletes it. The old entry leaves the way it
  always did: its holder's release, or the bucket's ``max_age``.
- **An old replica treats a new entry as held.** Its acquisition is the same create-if-absent, which
  fails on any existing key whatever its value, so it raises ``LockHeld``; its renewal and release
  compare the stored value with their own token, which an envelope never equals, so they report the
  lock taken and leave the entry alone. Old code never reclaims an entry, so it cannot overwrite one.
- **The bucket is shared unchanged.** Both versions open it with the same ``kv_bucket(name, ttl)``
  call, so neither redeclares, recreates or narrows it, and either order of upgrade (or a rollback)
  works. Nothing is migrated: the old format simply stops being written.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import TYPE_CHECKING, Final

from threetears.nats.errors import LockLossReason

from threetears.core.coordination.lease import HeldLease, KVLease, LeaseUnavailable

if TYPE_CHECKING:
    # annotation only: the kv module reaches nats-py, which importing core must never need
    from threetears.nats.kv import KvCapable

__all__ = ["LockHeld", "LockHold", "LockLossReason", "LockLost", "nats_distributed_lock"]


class LockHeld(Exception):
    """Raised by :func:`nats_distributed_lock` when another holder owns the key.

    Distinct from :class:`threetears.nats.KvError`, which signals a
    transport / bucket-creation failure -- callers can treat ``LockHeld``
    as the expected "another pod is already running this job" branch
    and re-raise / surface ``KvError`` separately.
    """


class LockLost(Exception):
    """Raised when a lock was lost while its body ran.

    From the ``async with`` of :func:`nats_distributed_lock` when the body was interrupted
    because the lock was lost (the default), and from :meth:`LockHold.raise_if_lost`.
    Deliberately NOT a :class:`~threetears.nats.errors.KvError` and NOT a :class:`LockHeld`:
    callers treat ``KvError`` as "no lock infrastructure, run the body anyway" and
    ``LockHeld`` as "someone else is running it, skip" -- and this body has already run, in
    part.

    :ivar key: the lock key
    :ivar reason: why the lock was lost
    """

    def __init__(self, key: str, reason: LockLossReason) -> None:
        """record the key and the reason.

        :param key: the lock key
        :ptype key: str
        :param reason: why the lock was lost
        :ptype reason: LockLossReason
        """
        super().__init__(f"lock lost ({reason.value}): {key}")
        self.key = key
        self.reason = reason


class LockHold:
    """What the body of :func:`nats_distributed_lock` holds: the key, and whether it still does.

    ``async with nats_distributed_lock(nats, key) as hold:`` -- or ignore it; by default a
    lost lock interrupts the body and the ``async with`` raises :class:`LockLost`, so a body
    that never looks at the hold is still stopped. A body run with ``cancel_on_loss=False``
    checks it at its own safe points: ``hold.raise_if_lost()``, or ``await hold.lost.wait()``
    in a task of its own. A view of the lease that holds the key; it keeps no state of its own.
    """

    __slots__ = ("_held", "_key", "_lost")

    def __init__(self, key: str, held: HeldLease | None) -> None:
        """bind the hold to its key and the lease holding it.

        :param key: the lock key
        :ptype key: str
        :param held: the lease holding the key, or ``None`` for the single-pod no-op, which is never lost
        :ptype held: HeldLease | None
        """
        self._key = key
        self._held = held
        self._lost = held.lost if held is not None else asyncio.Event()

    @property
    def key(self) -> str:
        """the lock key.

        :return: the key
        :rtype: str
        """
        return self._key

    @property
    def lost(self) -> asyncio.Event:
        """set when the lock is lost; never cleared.

        :return: the loss event
        :rtype: asyncio.Event
        """
        return self._lost

    @property
    def lost_reason(self) -> LockLossReason | None:
        """why the lock was lost, or ``None`` while it is held.

        :return: the loss reason
        :rtype: LockLossReason | None
        """
        return self._held.lost_reason if self._held is not None else None

    def raise_if_lost(self) -> None:
        """raise :class:`LockLost` if the lock has been lost.

        :return: nothing
        :rtype: None
        :raises LockLost: when the lock has been lost
        """
        reason = self.lost_reason
        if reason is not None:
            raise LockLost(self._key, reason)


_DEFAULT_TTL: Final[timedelta] = timedelta(seconds=60)
_DEFAULT_HEARTBEAT: Final[timedelta] = timedelta(seconds=20)
_DEFAULT_BUCKET: Final[str] = "scheduler-locks"

#: How long one holder may keep renewing before renewal stops.
#:
#: A holder that DIES stops renewing and the TTL hands the lock on within ``ttl``. A holder that
#: WEDGES does not: its renewal is healthy and goes on while the body makes no progress, so the lock
#: is held for as long as the process lives. That is how one stuck pod blocked every other pod's tick
#: in the 2026-08 incident -- the lock behaved exactly as designed and the fleet starved anyway.
#:
#: The default is sized far above any caller in this workspace (scheduler ticks, a derived-collection
#: rebuild, a backup) so a healthy long job is never interrupted; a caller whose body legitimately
#: runs longer, or that wants a wedge noticed sooner, passes its own ``max_hold``.
_DEFAULT_MAX_HOLD: Final[timedelta] = timedelta(hours=6)


@asynccontextmanager
async def nats_distributed_lock(
    client: KvCapable | None,
    key: str,
    *,
    bucket_name: str = _DEFAULT_BUCKET,
    ttl: timedelta = _DEFAULT_TTL,
    heartbeat: timedelta = _DEFAULT_HEARTBEAT,
    cancel_on_loss: bool = True,
    max_hold: timedelta = _DEFAULT_MAX_HOLD,
) -> AsyncIterator[LockHold]:
    """Acquire a TTL-based distributed lock backed by NATS JetStream KV.

    Usage::

        try:
            async with nats_distributed_lock(nats, "my_job"):
                ... # body runs while we hold the lock
        except LockHeld:
            return  # another holder owns this key; skip this turn
        except LockLost:
            return  # we lost it mid-body; the body was interrupted

    The key is held by a :meth:`KVLease.hold <threetears.core.coordination.KVLease.hold>`, renewed
    every ``heartbeat`` by compare-and-swap so the lock stays alive for arbitrarily long bodies and a
    holder that has lost it never writes it back. On normal exit or exception the renewal stops and
    the key is deleted if it is still this holder's. If the process dies, renewal stops, the bucket's
    TTL removes the key within ``ttl``, and another claimer wins on its next attempt.

    The yielded :class:`LockHold` reports a loss. With ``cancel_on_loss`` (the default) a loss also
    cancels the body at its next ``await`` and the ``async with`` raises :class:`LockLost` instead of
    the cancellation; a cancellation from anywhere else still propagates as
    :class:`asyncio.CancelledError`. The loss cannot be reported before the body next yields to the
    event loop -- a body stalled in synchronous code learns of it when it resumes.

    When ``client`` is ``None`` the context manager yields a hold that is never lost, without
    acquiring anything -- safe for single-pod dev environments without NATS available.

    :param client: connected NATS client, or ``None`` to no-op
    :ptype client: KvCapable | None
    :param key: lock key (per-resource identifier, e.g. ``"backup"`` for a backup job or
        ``"agent_wake_tick"`` for the wake tick engine)
    :ptype key: str
    :param bucket_name: KV bucket suffix; the connected client's namespace is prefixed
        automatically. Defaults to ``"scheduler-locks"`` so existing prod state continues to bind to
        the same bucket.
    :ptype bucket_name: str
    :param ttl: the bucket's entry TTL, and the lease's; bounds the orphan-lock window after a holder
        dies between renewals. Whole seconds, at least one. **Bucket-level: the first caller to
        materialise a given ``bucket_name`` pins the TTL for every key in that bucket**; a later
        caller passing a different ``ttl`` against the same ``bucket_name`` raises
        :class:`ValueError` -- use a distinct ``bucket_name`` to vary TTL.
    :ptype ttl: timedelta
    :param heartbeat: renewal interval; must be strictly less than ``ttl`` or the lock would expire
        under a live holder
    :ptype heartbeat: timedelta
    :param cancel_on_loss: cancel the body when the lock is lost and raise :class:`LockLost` from the
        ``async with``. ``False`` only reports the loss on the yielded hold -- for a body whose
        correctness does not rest on the lock and which must not be interrupted mid-way
    :ptype cancel_on_loss: bool
    :param max_hold: the longest this holder renews the lock; past it renewal stops, the hold reports
        :attr:`LockLossReason.MAX_HOLD` and the TTL hands the lock on, so a wedged body cannot keep
        every other claimer out for as long as its process lives. Six hours by default
    :ptype max_hold: timedelta
    :return: async iterator yielding the :class:`LockHold` while the lock is held
    :rtype: AsyncIterator[LockHold]
    :raises ValueError: when ``heartbeat >= ttl`` (invalid invariant), when ``ttl`` is not whole
        seconds, when ``max_hold`` is negative, or when ``ttl`` does not match the bucket's
        already-cached TTL
    :raises LockHeld: when the key is already owned by another holder
    :raises LockLost: when the lock was lost while the body ran and ``cancel_on_loss`` is set
    :raises KvError: on transport / bucket failures (distinct from ``LockHeld``)
    """
    if client is None:
        yield LockHold(key, None)
        return
    if heartbeat >= ttl:
        msg = f"heartbeat {heartbeat} must be less than ttl {ttl}"
        raise ValueError(msg)
    if max_hold < timedelta(0):
        msg = f"max_hold {max_hold} must not be negative"
        raise ValueError(msg)

    # DECLARES its bucket (``create_if_missing`` left at its default), deliberately: see "Infrastructure
    # callers only" above. Opened here rather than by the lease so the declaration -- and with it the
    # bucket-level TTL -- is exactly the one every earlier release made (the upgrade contract).
    bucket = await client.kv_bucket(name=bucket_name, ttl=ttl)
    # JetStream KV bucket TTL is bucket-level and fixed at creation, and the client caches buckets by
    # name (first caller wins): a second caller passing a different ``ttl`` would silently inherit the
    # first caller's, and the apparent contract of this call ("expires after ``ttl``") would be wrong.
    if bucket.ttl is not None and bucket.ttl != ttl:
        msg = (
            f"nats_distributed_lock: bucket {bucket_name!r} was created "
            f"with ttl={bucket.ttl}; this call passed ttl={ttl}. KV "
            f"bucket TTL is bucket-level + fixed at creation -- use a "
            f"distinct bucket_name to vary TTL."
        )
        raise ValueError(msg)

    body_task = asyncio.current_task()
    # the loss reaches the body through ``_interrupt``; these two say whether it may still be
    # interrupted, and whether the cancellation it is unwinding is the loss's own.
    exiting = False
    interrupted = False

    def _interrupt(reason: LockLossReason) -> None:
        """cancel the body for a lost lock, unless it has finished or opted out.

        :param reason: why the lock was lost
        :ptype reason: LockLossReason
        :return: nothing
        :rtype: None
        """
        nonlocal interrupted
        if cancel_on_loss and not exiting and body_task is not None:
            interrupted = True
            body_task.cancel(f"nats_distributed_lock: lock lost ({reason.value}): {key}")

    # one lease per hold: its generated holder id names THIS hold, so the release's identity fence
    # can never mistake another hold's entry -- even this process's next one -- for its own.
    lease = KVLease(None, bucket=bucket)
    try:
        held = await lease.hold(
            key,
            ttl=ttl,
            renew_every=heartbeat,
            max_hold=max_hold,
            on_lost=_interrupt,
            name=f"the lock {key!r}",
            log_extra={"bucket": bucket_name, "ttl_seconds": ttl.total_seconds()},
            renew_failure_level=logging.WARNING,
        )
    except LeaseUnavailable as exc:
        raise LockHeld(f"lock already held: {key}") from exc

    cancelling_at_entry = body_task.cancelling() if body_task is not None else 0
    try:
        yield LockHold(key, held)
    except BaseException as exc:
        # The loss cancelled the body: take back exactly that cancellation, and turn it into
        # LockLost -- unless something else ALSO cancelled the task, whose cancellation must still
        # propagate. The same bookkeeping asyncio.timeout does.
        if interrupted and body_task is not None:
            interrupted = False
            remaining = body_task.uncancel()
            reason = held.lost_reason
            if isinstance(exc, asyncio.CancelledError) and reason is not None and remaining <= cancelling_at_entry:
                raise LockLost(key, reason) from exc
        raise
    else:
        # The body swallowed the cancellation (or finished before it arrived): withdraw it so it
        # cannot fire at the caller's next await, and still report the loss.
        reason = held.lost_reason
        if interrupted and body_task is not None and reason is not None:
            interrupted = False
            body_task.uncancel()
            raise LockLost(key, reason)
    finally:
        exiting = True
        # stops renewal and deletes the entry if it is still this hold's; never raises, so a cleanup
        # failure cannot replace the body's own outcome (the TTL frees an entry nobody deleted)
        await held.release()
