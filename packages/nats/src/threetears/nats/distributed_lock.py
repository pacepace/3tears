"""TTL-based distributed NATS lock primitive.

Extracted from a production ``scheduler_lock`` implementation
(which has run the production backup job for months) so any 3tears app
needing single-active-holder semantics across pods can pick it up
without re-implementing the heartbeat lifecycle. The agent-wake tick
engine (``threetears.agent.wake.tick``) is the second canonical
consumer; the existing ``scheduler_lock`` becomes a one-line
re-export of this primitive.

design notes
------------

- **NATS JetStream KV-backed.** Acquisition is an atomic
  ``bucket.create(key, value)`` (put-if-absent). The KV bucket's
  per-entry TTL bounds the worst-case orphan-lock window when the
  holder dies between heartbeats.
- **Bucket-level TTL: first caller wins.** :meth:`NatsClient.kv_bucket`
  caches buckets by name; the bucket's per-entry TTL is fixed at
  creation time and applies to every key the bucket stores. Per-key
  TTL (``msg_ttl`` on :meth:`NatsKvBucket.create`) requires
  ``AllowMsgTTL=True`` on the underlying JetStream stream and
  nats-server >= 2.11 -- not enabled by default on the
  ``scheduler-locks`` bucket, so the bucket-level TTL is the only
  knob this primitive actually controls. To make the
  first-caller-wins constraint loud rather than silent, the lock
  asserts that a subsequent caller's ``ttl`` matches the cached
  bucket's TTL; a mismatch raises :class:`ValueError`. Callers
  needing distinct TTLs MUST use distinct ``bucket_name`` values.
- **Background heartbeat, renewed by compare-and-swap.** A dedicated
  task renews the KV entry every ``heartbeat`` seconds so a long-running
  body stays the authoritative holder -- but only while the entry still
  carries THIS holder's token, and only at the revision just read. A
  blind ``put`` would let a holder that stalled past ``ttl`` (a blocked
  loop, a GC pause, a partition) overwrite the successor that acquired
  the expired key, leaving two holders. ``heartbeat`` must be strictly
  less than ``ttl``; the contextmanager raises ``ValueError`` otherwise.
- **The holder is told when it loses the lock.** The context manager
  yields a :class:`LockHold`. When the entry is gone or someone else's
  (:attr:`LockLossReason.EXPIRED` / :attr:`LockLossReason.TAKEN`), when
  renewals keep failing until the entry may have expired
  (:attr:`LockLossReason.RENEWAL_FAILED`), or when the holder has held
  past the maximum hold (:attr:`LockLossReason.MAX_HOLD`), ``hold.lost``
  is set. By default the body is also cancelled and the ``async with``
  raises :class:`LockLost`: a body that keeps writing after its lock is
  gone is the damage a lock exists to prevent, and a flag nobody reads
  prevents none of it. ``cancel_on_loss=False`` is for a body whose
  correctness does not rest on the lock (it only saves duplicate work);
  such a body can still read ``hold.lost``.
- **Clean cancellation on exit.** Normal exit and exception both
  cancel + await the heartbeat task before releasing the key. The
  ``finally`` block uses ``asyncio.gather(..., return_exceptions=True)``
  so a heartbeat death does not mask the body's own exception.
- **Graceful single-pod fallback.** ``client=None`` yields
  immediately without acquiring anything -- matches the existing
  behaviour for dev environments that do not run NATS.
- **Bucket name namespacing.** The default bucket ``"scheduler-locks"``
  rides through :meth:`NatsClient.kv_bucket` and picks up the
  client's ``nats_subject_namespace`` prefix automatically (the
  resulting bucket is ``{namespace}-scheduler-locks``).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import secrets
from datetime import timedelta
from enum import StrEnum
from typing import Final

from threetears.observe import get_logger

from threetears.nats.kv import KvCapable
from threetears.nats.errors import KvError

__all__ = ["LockHeld", "LockHold", "LockLossReason", "LockLost", "nats_distributed_lock"]


log = get_logger(__name__)


class LockHeld(Exception):
    """Raised by :func:`nats_distributed_lock` when another holder owns the key.

    Distinct from :class:`threetears.nats.KvError`, which signals a
    transport / bucket-creation failure -- callers can treat ``LockHeld``
    as the expected "another pod is already running this job" branch
    and re-raise / surface ``KvError`` separately.
    """


class LockLossReason(StrEnum):
    """Why a holder stopped holding a :func:`nats_distributed_lock` before its body finished.

    :cvar EXPIRED: the entry was gone at renewal -- it expired (the holder stalled past the
        TTL) and nobody has taken it yet
    :cvar TAKEN: the entry carries another holder's token, or another write landed between
        the renewal's read and its compare-and-swap -- someone else holds the lock now
    :cvar RENEWAL_FAILED: renewals kept failing (broker unreachable) until the entry may
        have expired, so the lock can no longer be vouched for
    :cvar MAX_HOLD: the holder kept the lock past the maximum hold and renewal stopped so
        the TTL could hand it on
    """

    EXPIRED = "expired"
    TAKEN = "taken"
    RENEWAL_FAILED = "renewal_failed"
    MAX_HOLD = "max_hold"


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


class _HoldState:
    """what the heartbeat and the context manager share about one hold.

    :ivar lost: set once, when the lock is lost
    :ivar reason: why, once lost
    :ivar exiting: the body has finished; a loss after this cancels nothing
    :ivar body_cancelled: the loss cancelled the body, and the exit owes an uncancel
    """

    __slots__ = ("body_cancelled", "exiting", "lost", "reason")

    def __init__(self) -> None:
        """start held."""
        self.lost = asyncio.Event()
        self.reason: LockLossReason | None = None
        self.exiting = False
        self.body_cancelled = False


class LockHold:
    """What the body of :func:`nats_distributed_lock` holds: the key, and whether it still does.

    ``async with nats_distributed_lock(nats, key) as hold:`` -- or ignore it; by default a
    lost lock interrupts the body and the ``async with`` raises :class:`LockLost`, so a body
    that never looks at the hold is still stopped. A body run with ``cancel_on_loss=False``
    checks it at its own safe points: ``hold.raise_if_lost()``, or ``await hold.lost.wait()``
    in a task of its own.
    """

    __slots__ = ("_key", "_state")

    def __init__(self, key: str, state: _HoldState) -> None:
        """bind the hold to its key and shared state.

        :param key: the lock key
        :ptype key: str
        :param state: the state the heartbeat updates
        :ptype state: _HoldState
        """
        self._key = key
        self._state = state

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
        return self._state.lost

    @property
    def lost_reason(self) -> LockLossReason | None:
        """why the lock was lost, or ``None`` while it is held.

        :return: the loss reason
        :rtype: LockLossReason | None
        """
        return self._state.reason

    def raise_if_lost(self) -> None:
        """raise :class:`LockLost` if the lock has been lost.

        :return: nothing
        :rtype: None
        :raises LockLost: when the lock has been lost
        """
        if self._state.reason is not None:
            raise LockLost(self._key, self._state.reason)


_DEFAULT_TTL: Final[timedelta] = timedelta(seconds=60)
_DEFAULT_HEARTBEAT: Final[timedelta] = timedelta(seconds=20)
_DEFAULT_BUCKET: Final[str] = "scheduler-locks"

#: How long one holder may keep renewing before the heartbeat gives up.
#:
#: A holder that DIES stops heartbeating and the TTL hands the lock on within
#: ``ttl``. A holder that WEDGES does not: its heartbeat task is healthy and
#: goes on renewing while the body makes no progress, so the lock is held for
#: as long as the process lives. That is how one stuck pod blocked every other
#: pod's tick in the 2026-08 incident -- the lock behaved exactly as designed
#: and the fleet starved anyway.
#:
#: Renewal therefore stops here, the holder is told (:attr:`LockLossReason.MAX_HOLD`),
#: and the TTL takes over. The default is sized far above any caller in this
#: workspace (scheduler ticks, a derived-collection rebuild, a backup) so a healthy
#: long job is never interrupted; a caller whose body legitimately runs longer, or
#: that wants a wedge noticed sooner, passes its own ``max_hold``.
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

    A background heartbeat task renews the KV entry every ``heartbeat``
    seconds, by compare-and-swap on this holder's token, so the lock stays
    alive for arbitrarily long bodies and a holder that has lost it never
    writes it back. On normal exit or exception the heartbeat is cancelled
    and the key is deleted if it is still this holder's. If the process
    dies the heartbeat stops, the TTL expires the key within ``ttl``
    seconds, and another claimer can win on the next attempt.

    The yielded :class:`LockHold` reports a loss. With ``cancel_on_loss``
    (the default) a loss also cancels the body at its next ``await`` and
    the ``async with`` raises :class:`LockLost` instead of the
    cancellation; a cancellation from anywhere else still propagates as
    :class:`asyncio.CancelledError`. The loss cannot be reported before
    the body next yields to the event loop -- a body stalled in
    synchronous code learns of it when it resumes.

    When ``client`` is ``None`` the context manager yields a hold that is
    never lost, without acquiring anything -- safe for single-pod dev
    environments without NATS available.

    :param client: connected NATS client, or ``None`` to no-op
    :ptype client: KvCapable | None
    :param key: lock key (per-resource identifier, e.g. ``"backup"``
        for a backup job or ``"agent_wake_tick"`` for the
        wake tick engine)
    :ptype key: str
    :param bucket_name: KV bucket suffix; the connected client's
        namespace is prefixed automatically. Defaults to
        ``"scheduler-locks"`` so existing prod state continues
        to bind to the same bucket post-extraction.
    :ptype bucket_name: str
    :param ttl: KV entry TTL; bounds the orphan-lock window after a
        holder dies between heartbeats. **Bucket-level: the first
        caller to materialise a given ``bucket_name`` pins the TTL
        for every key in that bucket**, because
        :meth:`NatsClient.kv_bucket` caches buckets and JetStream KV
        does not support per-key TTL without ``AllowMsgTTL`` on the
        stream (nats-server >= 2.11; not enabled by default on
        ``scheduler-locks``). Subsequent callers passing a different
        ``ttl`` against the same ``bucket_name`` raise
        :class:`ValueError`; use a distinct ``bucket_name`` to vary
        TTL.
    :ptype ttl: timedelta
    :param heartbeat: KV refresh interval; must be strictly less than
        ``ttl`` or the lock will expire under a live holder
    :ptype heartbeat: timedelta
    :param cancel_on_loss: cancel the body when the lock is lost and raise
        :class:`LockLost` from the ``async with``. ``False`` only reports the
        loss on the yielded hold -- for a body whose correctness does not
        rest on the lock and which must not be interrupted mid-way
    :ptype cancel_on_loss: bool
    :param max_hold: the longest this holder renews the lock; past it renewal stops, the
        hold reports :attr:`LockLossReason.MAX_HOLD` and the TTL hands the lock on, so a
        wedged body cannot keep every other claimer out for as long as its process lives.
        Six hours by default
    :ptype max_hold: timedelta
    :return: async iterator yielding the :class:`LockHold` while the lock is held
    :rtype: AsyncIterator[LockHold]
    :raises ValueError: when ``heartbeat >= ttl`` (invalid invariant), when ``max_hold``
        is negative, or when ``ttl`` does not match the bucket's already-cached TTL
    :raises LockHeld: when the key is already owned by another holder
    :raises LockLost: when the lock was lost while the body ran and
        ``cancel_on_loss`` is set
    :raises KvError: on transport / bucket failures (distinct from
        ``LockHeld``)
    """
    state = _HoldState()
    hold = LockHold(key, state)
    if client is None:
        yield hold
        return
    if heartbeat >= ttl:
        msg = f"heartbeat {heartbeat} must be less than ttl {ttl}"
        raise ValueError(msg)
    if max_hold < timedelta(0):
        msg = f"max_hold {max_hold} must not be negative"
        raise ValueError(msg)

    bucket = await client.kv_bucket(name=bucket_name, ttl=ttl)
    # JetStream KV bucket TTL is bucket-level + fixed-at-creation. The
    # client caches buckets by name (first-caller wins), so a second
    # caller passing a different ``ttl`` would silently inherit the
    # first caller's TTL and the apparent contract of *this* call
    # ("lock expires after `ttl` seconds") would be wrong. Make the
    # mismatch loud instead.
    if bucket.ttl is not None and bucket.ttl != ttl:
        msg = (
            f"nats_distributed_lock: bucket {bucket_name!r} was created "
            f"with ttl={bucket.ttl}; this call passed ttl={ttl}. KV "
            f"bucket TTL is bucket-level + fixed at creation -- use a "
            f"distinct bucket_name to vary TTL."
        )
        raise ValueError(msg)
    # This holder's identity, written as the entry's value. The release fences on
    # it rather than on a revision number: see the `finally` below for why a
    # revision cannot answer "is this still mine".
    #
    # A nonce rather than a uuid7, deliberately: this identifies one HOLD, not an
    # entity, so it is never stored, joined, or ordered, and the repo's
    # time-ordered-id convention has nothing to say about it. Hex so an operator
    # reading a stuck lock out of the bucket sees something legible.
    holder_token = secrets.token_hex(16).encode()
    loop = asyncio.get_running_loop()
    # Taken BEFORE the create is sent: the server stamps the entry no earlier than
    # this, so it cannot expire before ``acquired_at + ttl``.
    acquired_at = loop.time()
    acquired = await bucket.create(key=key, value=holder_token)
    if acquired is None:
        raise LockHeld(f"lock already held: {key}")

    body_task = asyncio.current_task()
    heartbeat_seconds = heartbeat.total_seconds()
    ttl_seconds = ttl.total_seconds()

    def _lose(reason: LockLossReason) -> None:
        """record the loss, and interrupt the body unless it has finished or opted out.

        :param reason: why the lock was lost
        :ptype reason: LockLossReason
        :return: nothing
        :rtype: None
        """
        state.reason = reason
        state.lost.set()
        if cancel_on_loss and not state.exiting and body_task is not None:
            state.body_cancelled = True
            body_task.cancel(f"nats_distributed_lock: lock lost ({reason.value}): {key}")

    async def _renew() -> LockLossReason | None:
        """renew the entry by compare-and-swap on this holder's token.

        :return: the loss this renewal discovered, or ``None`` when renewed
        :rtype: LockLossReason | None
        :raises KvError: when the broker cannot be reached
        """
        entry = await bucket.get_entry(key=key)
        result: LockLossReason | None = None
        if entry is None:
            result = LockLossReason.EXPIRED
        elif entry[0] != holder_token:
            result = LockLossReason.TAKEN
        elif await bucket.update(key=key, value=holder_token, revision=entry[1]) is None:
            # another write landed between the read and the swap: only a new holder writes
            # this key, so the fence refusing is the lock changing hands.
            result = LockLossReason.TAKEN
        return result

    async def _heartbeat() -> None:
        """Renew the entry until cancelled, until the lock is lost, or until held too long.

        ``CancelledError`` is the normal exit path (the contextmanager cancels us on body
        completion). Every other exit is a loss, reported through :func:`_lose`.

        **A renewal never writes blind.** It reads the entry and swaps it at the revision
        just read, and only while the entry carries this holder's token. A holder that
        stalled past the TTL wakes to find the key expired or re-acquired and reports the
        loss instead of overwriting the successor.

        **A failed renewal is retried while the entry cannot have expired.** One broker
        blip must not interrupt a healthy body: the entry was last written no earlier than
        the last renewal was sent, so it is this holder's until that moment plus ``ttl``.
        Once the next attempt would land past that, the lock can no longer be vouched for
        and the failure is a loss.

        **Renewal stops at ``max_hold``.** A holder that WEDGES keeps a perfectly
        healthy heartbeat task renewing a lock whose body is making no progress, which is
        how one stuck pod starved a whole fleet. Past the maximum hold this stops renewing,
        reports the loss, and lets the TTL hand the lock on -- loudly, at ERROR, because a
        lock withdrawn under a live body is a real event a human needs to see.
        """
        deadline = loop.time() + max_hold.total_seconds()
        last_renewal_sent = acquired_at
        log_extra = {"key": key, "bucket": bucket_name, "ttl_seconds": ttl_seconds}
        while True:
            await asyncio.sleep(heartbeat_seconds)
            if loop.time() >= deadline:
                log.error(
                    "nats_distributed_lock: holder has kept this lock past the maximum hold "
                    "and is still running; refusing to renew so the TTL can hand it on. The "
                    "body is wedged or far slower than this lock was sized for -- another "
                    "holder may now acquire it.",
                    extra={"extra_data": {**log_extra, "max_hold_seconds": max_hold.total_seconds()}},
                )
                _lose(LockLossReason.MAX_HOLD)
                return
            renewal_sent = loop.time()
            try:
                loss = await _renew()
            except Exception as exc:  # noqa: BLE001 - boundary: a renewal failure is retried or reported as a loss, never raised into the heartbeat's owner
                expired_by = last_renewal_sent + ttl_seconds
                if loop.time() + heartbeat_seconds < expired_by:
                    log.warning(
                        "nats_distributed_lock: renewal failed; retrying while the entry cannot have expired",
                        extra={"extra_data": {**log_extra, "error_type": type(exc).__name__, "error": str(exc)}},
                    )
                    continue
                log.warning(
                    "nats_distributed_lock: renewals failed until the entry may have expired; the lock is lost",
                    extra={"extra_data": {**log_extra, "error_type": type(exc).__name__, "error": str(exc)}},
                )
                _lose(LockLossReason.RENEWAL_FAILED)
                return
            if loss is not None:
                log.error(
                    "nats_distributed_lock: the lock was lost while its holder was still running -- the "
                    "holder stalled past the TTL or the entry was taken; not renewing it",
                    extra={"extra_data": {**log_extra, "reason": loss.value}},
                )
                _lose(loss)
                return
            last_renewal_sent = renewal_sent

    hb_task = asyncio.create_task(_heartbeat(), name=f"nats-lock-heartbeat:{key}")
    cancelling_at_entry = body_task.cancelling() if body_task is not None else 0
    try:
        yield hold
    except BaseException as exc:
        # The loss cancelled the body: take back exactly that cancellation, and turn it
        # into LockLost -- unless something else ALSO cancelled the task, whose
        # cancellation must still propagate. The same bookkeeping asyncio.timeout does.
        if state.body_cancelled and body_task is not None:
            state.body_cancelled = False
            remaining = body_task.uncancel()
            lost_the_lock = state.reason is not None and remaining <= cancelling_at_entry
            if isinstance(exc, asyncio.CancelledError) and lost_the_lock and state.reason is not None:
                raise LockLost(key, state.reason) from exc
        raise
    else:
        # The body swallowed the cancellation (or finished before it arrived): withdraw
        # it so it cannot fire at the caller's next await, and still report the loss.
        if state.body_cancelled and body_task is not None and state.reason is not None:
            state.body_cancelled = False
            body_task.uncancel()
            raise LockLost(key, state.reason)
    finally:
        state.exiting = True
        hb_task.cancel()
        # return_exceptions=True so heartbeat-death does not mask a
        # body exception that we are about to re-raise via the context
        # manager protocol.
        await asyncio.gather(hb_task, return_exceptions=True)
        try:
            # FENCED on IDENTITY, not on sequence. An unconditional delete is a correctness
            # bug whenever the lock did not survive the body: if the heartbeat died, or
            # stopped at the maximum hold, the TTL expires the key and another pod acquires
            # it -- and a plain delete then removes the SUCCESSOR's lock, handing the same
            # key to a third holder while the second still believes it owns it.
            #
            # Fencing on "the revision this holder last wrote" was the first answer and it
            # is not sound, because a holder can be BEHIND its own writes. The heartbeat
            # recorded a revision by assigning the result of its renewal, so a cancellation
            # delivered after the write landed but before that assignment -- a window one
            # network round trip wide -- left the holder one revision short of the entry it
            # owned. The fence then refused its owner's own release and the lock sat there
            # for its whole TTL while every other pod waited: the very outcome the fence
            # exists to prevent, reached from the other side.
            #
            # The token answers the question the revision was standing in for. The
            # heartbeat is already cancelled and awaited above, so nothing of ours can
            # write between this read and the delete, and the delete stays fenced on the
            # revision just read, so anything ELSE that writes in between still wins.
            entry = await bucket.get_entry(key=key)
            if entry is not None and entry[0] == holder_token:
                await bucket.delete(key=key, revision=entry[1])
        except KvError as exc:
            log.debug(
                "nats_distributed_lock: cleanup delete failed (key already gone, or the lock "
                "moved to another holder and the identity fence refused)",
                extra={"extra_data": {"key": key, "bucket": bucket_name, "error": str(exc)}},
            )
