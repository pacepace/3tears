"""KVLease — TTL-bounded distributed mutex over NATS JetStream KV.

generalizes the ownership-token + expiry-in-value pattern used
historically by the hub's login-lockout tracking (since migrated to
:class:`~aibots.hub.security.collections.LoginLockoutCollection`, a
durable three-tier ``BaseCollection`` — not a candidate for this
primitive, since lockout state must survive pod restarts and counts
failures rather than granting exclusive ownership) into a reusable
primitive for cross-pod coordination (workspace bind locks, leader
election, fencing tokens).

usage::

    lease = KVLease(nats_client, bucket_name="leases")
    handle = await lease.acquire("workspace/customer-x/path")
    async with handle:
        ...

to HOLD a lease across real work, renewed in the background and told when it is lost::

    held = await lease.hold("git-writer.story-1", ttl=timedelta(seconds=30), renew_every=timedelta(seconds=10))
    ...                        # watch held.lost / await held.until_lost()
    await held.release()       # or: async with held: ...

to hold keys in a bucket ANOTHER owner declared and handed over already bound, as readers that
test a key's existence see them (a writer's claim, a registration)::

    lease = KVLease(None, bucket=pointers, pod_id=replica, expire_entries=True)
    held = await lease.hold(key, ttl=ttl, renew_every=ttl / 3, retake=True)

``expire_entries`` gives each entry a NATS per-key TTL, so a holder that stops renewing loses the
KEY itself, not only its envelope's expiry; ``retake`` takes back an entry that went away (it lapsed
through an outage, or NATS lost the bucket) instead of reporting the lease lost.

``nats_client`` is the canonical
:class:`threetears.nats.kv.KvCapable` wrapper. the lease opens its
bucket via :meth:`KvCapable.kv_bucket` so all CAS / miss semantics
flow through :class:`KvBucketLike`'s typed return shape (``None`` on
conflict instead of raising :class:`KeyWrongLastSequenceError`).

all KV envelope payloads flow through
:func:`threetears.core.serialization.serialize_to_json` /
:func:`deserialize_from_json` so UUID, datetime, Decimal round-trip
correctly with the rest of the codebase.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import TYPE_CHECKING, Any, Final
from uuid import uuid7

from threetears.core.serialization import deserialize_from_json, json_datetime, serialize_to_json
from threetears.nats.errors import LockLossReason
from threetears.observe import get_logger

from threetears.core.coordination._owner_scope import owner_scoped_key, validated_key_scope

if TYPE_CHECKING:
    # From the submodule, not the package: these three are Protocols that
    # `threetears.nats` stopped re-exporting when its nats-py-backed surface went lazy.
    # Annotation-only, so the eager `kv` import here costs an L1 consumer nothing.
    from threetears.nats.kv import KvBucketLike, KvCapable

__all__ = [
    "HeldLease",
    "KVLease",
    "LeaseHandle",
    "LeaseLossReason",
    "LeaseLost",
    "LeaseTimeout",
    "LeaseUnavailable",
]

log = get_logger(__name__)

#: the longest a release waits for a renewal already in flight to be answered: one KV operation's own
#: ceiling (``threetears.nats.kv.KvTimings.op_timeout_seconds``, which bounds every KV round trip; a
#: test pins the two equal). A renewal unanswered by then is not coming back in time to matter, and
#: waiting longer -- the TTL, as a release once did -- stretches every owner's shutdown by that much.
RENEWAL_ANSWER_WAIT_SECONDS: Final[float] = 10.0

#: why a held lease was lost (:attr:`HeldLease.lost_reason`): the one vocabulary
#: :func:`~threetears.nats.nats_distributed_lock` reports too, so it is that enum, under the name
#: this module's callers read.
LeaseLossReason = LockLossReason


class LeaseUnavailable(LookupError):
    """raised by :meth:`KVLease.acquire` when fail-fast path finds lease held.

    caller requested ``max_wait_seconds == 0`` and key is currently owned by
    live holder. subclasses :class:`LookupError` so callers may catch broadly
    or narrowly.
    """


class LeaseTimeout(TimeoutError):
    """raised by :meth:`KVLease.acquire` when wait deadline elapses.

    caller's ``max_wait_seconds`` expired before lease became free.
    subclasses :class:`TimeoutError`.
    """


class LeaseLost(RuntimeError):
    """raised by refresh or release when ownership has silently changed.

    another holder took over (entry expired, got reclaimed) or a concurrent
    CAS advanced the revision underneath us. caller's handle is no longer
    valid; caller should abort whatever work the lease protected.

    :ivar reason: what the refresh found -- the entry gone (:attr:`LeaseLossReason.EXPIRED`) or
        another holder's (:attr:`LeaseLossReason.TAKEN`); ``None`` when the raiser did not say
    """

    def __init__(self, *args: object, reason: LeaseLossReason | None = None) -> None:
        """record the message and, when known, why the lease is gone.

        :param args: the exception's message arguments
        :ptype args: object
        :param reason: why the lease is gone, or ``None``
        :ptype reason: LeaseLossReason | None
        :return: None
        :rtype: None
        """
        super().__init__(*args)
        self.reason = reason


@dataclass
class _Envelope:
    """internal decoded KV value.

    :ivar holder: pod identifier that currently owns the lease
    :ivar date_expires: timezone-aware datetime past which entry is stale
    :ivar date_acquired: timezone-aware datetime when entry was first created
    """

    holder: str
    date_expires: datetime
    date_acquired: datetime


def _encode_envelope(holder: str, date_expires: datetime, date_acquired: datetime) -> bytes:
    """serialize holder and timestamps to JSON bytes for KV storage.

    routes through :func:`serialize_to_json` so encoding stays consistent
    with the rest of the codebase; each instant is written in
    :func:`~threetears.core.serialization.json_datetime`'s one stored form.

    :param holder: pod identifier owning the lease
    :ptype holder: str
    :param date_expires: timezone-aware datetime at which entry goes stale
    :ptype date_expires: datetime
    :param date_acquired: timezone-aware datetime lease was first acquired
    :ptype date_acquired: datetime
    :return: JSON bytes suitable for storing as KV value
    :rtype: bytes
    :raises ValueError: if either datetime is naive
    """
    payload: dict[str, Any] = {
        "holder": holder,
        "expires_at": json_datetime(date_expires, field="expires_at"),
        "acquired_at": json_datetime(date_acquired, field="acquired_at"),
    }
    return serialize_to_json(payload)


def _decode_envelope(value: bytes) -> _Envelope:
    """deserialize stored KV value bytes back to envelope dataclass.

    routes through :func:`deserialize_from_json`; datetimes are stored as
    ISO 8601 strings so they come back as plain ``str`` and are parsed
    here into timezone-aware ``datetime`` objects. an envelope written before
    the one stored form (``isoformat()``, no fixed fraction) parses the same.

    :param value: bytes payload as returned from KV bucket ``get`` call
    :ptype value: bytes
    :return: envelope dataclass with holder + parsed timestamps
    :rtype: _Envelope
    :raises ValueError: if payload is malformed or timestamps unparseable
    """
    if not value.lstrip().startswith(b"{"):
        # only a JSON object is an envelope. an older lock's holder token is hex, and an all-digit
        # one parses as a JSON number, which the decoder below would fail on with an AttributeError
        # nobody catches -- an unreadable entry must read as another holder's, never as a crash.
        raise ValueError("not a lease envelope: the value is not a JSON object")
    raw = deserialize_from_json(value, field_types={})
    holder = str(raw["holder"])
    date_expires = datetime.fromisoformat(str(raw["expires_at"]))
    date_acquired = datetime.fromisoformat(str(raw["acquired_at"]))
    return _Envelope(holder=holder, date_expires=date_expires, date_acquired=date_acquired)


def _decoded_envelope(value: bytes) -> _Envelope | None:
    """the envelope a stored value holds, or ``None`` when it is not one this lease wrote.

    a value it cannot read (another writer's format, or one written before a lease kept the key) is
    nobody this lease may reclaim, refresh or delete: it is treated as another holder's.

    :param value: the stored value
    :ptype value: bytes
    :return: the envelope, or ``None``
    :rtype: _Envelope | None
    """
    try:
        return _decode_envelope(value)
    except (ValueError, KeyError, TypeError) as exc:
        log.info("KVLease: an entry it cannot read is another holder's: %s: %s", type(exc).__name__, exc)
        return None


class _Holds:
    """the holds one :class:`KVLease` has handed out since it was last closed: told to end together,
    and counted until each has (:meth:`KVLease.close`). Each hold reports its own start and end."""

    def __init__(self) -> None:
        self.closing = asyncio.Event()
        self.ended = asyncio.Event()
        self.ended.set()
        self._live = 0

    def started(self) -> None:
        """one more hold renewing."""
        self._live += 1
        self.ended.clear()

    def finished(self) -> None:
        """one hold stopped renewing; the last one wakes :meth:`KVLease.close`."""
        self._live -= 1
        if self._live <= 0:
            self.ended.set()


class LeaseHandle:
    """opaque handle representing one successful lease acquisition.

    returned by :meth:`KVLease.acquire`. exposes :meth:`refresh` and
    :meth:`release` as well as async-context-manager sugar that releases
    on exit (so callers can write ``async with handle:``).

    :ivar key: KV key under which lease entry lives
    :ivar holder: pod identifier used when the lease was acquired
    :ivar revision: last-known KV revision for CAS refresh/release
    :ivar ttl_seconds: default TTL applied by :meth:`refresh` when caller
        does not supply one
    :ivar released: flag set once :meth:`release` runs to completion,
        making subsequent :meth:`release` calls a no-op
    :ivar in_flight_value: the value of this handle's write that has been sent and not yet
        answered, or ``None``. A write cancelled after the server applied it leaves the handle a
        revision behind its own entry; the release recognises that entry by this exact value --
        which no other write, by this holder or another, produces -- and deletes it rather than
        leave it to hold everyone off for its TTL
    """

    def __init__(
        self,
        lease: KVLease,
        key: str,
        holder: str,
        revision: int,
        ttl_seconds: int,
        date_expires: datetime | None = None,
    ) -> None:
        """bind handle to parent lease and successful-acquire result state.

        :param lease: parent lease that produced this handle
        :ptype lease: KVLease
        :param key: KV key under which lease entry lives
        :ptype key: str
        :param holder: pod identifier used during acquire
        :ptype holder: str
        :param revision: initial KV revision from create or CAS update
        :ptype revision: int
        :param ttl_seconds: default TTL for implicit refresh TTL reuse
        :ptype ttl_seconds: int
        :param date_expires: the expiry written into the entry's envelope -- the instant other pods
            treat it as stale
        :ptype date_expires: datetime | None
        :return: None
        :rtype: None
        """
        self._lease = lease
        self.key = key
        self.holder = holder
        self.revision = revision
        self.ttl_seconds = ttl_seconds
        self.date_expires = date_expires
        self.released = False
        self.in_flight_value: bytes | None = None

    async def refresh(self, ttl_seconds: int | None = None, *, adopt_own_revision: bool = False) -> None:
        """extend lease expiry; raise :class:`LeaseLost` on ownership change.

        fetches current entry, verifies holder still matches, then performs
        CAS update with new expiry. on revision mismatch, holder mismatch,
        or key absence, raises :class:`LeaseLost`.

        :param ttl_seconds: override TTL for this refresh; falls back to
            TTL supplied at acquire time when None
        :ptype ttl_seconds: int | None
        :param adopt_own_revision: when the entry is still this holder's at a revision the handle has
            not seen (one of its own writes landed and its reply did not), renew from that revision
            rather than call the lease lost
        :ptype adopt_own_revision: bool
        :return: None
        :rtype: None
        :raises LeaseLost: if ownership has changed or CAS failed
        """
        await self._lease.refresh_handle(self, ttl_seconds=ttl_seconds, adopt_own_revision=adopt_own_revision)

    async def retake(self) -> bool:
        """take the entry again when it is gone, as this holder, by create; never over another's entry.

        :return: whether the entry is this holder's again; ``False`` when the key is there (another
            holder's, or a write of this one the handle has not seen)
        :rtype: bool
        """
        return await self._lease.retake_handle(self)

    async def release(self) -> None:
        """delete KV entry if still owned; no-op on repeat calls.

        idempotent: once the ``released`` flag is set, subsequent calls
        return immediately. safe under contention — if another pod has
        stolen the lease, this method marks the handle released without
        touching the new holder's entry.

        :return: None
        :rtype: None
        """
        await self._lease.release_handle(self)

    async def __aenter__(self) -> LeaseHandle:
        """async-context-manager entry; returns self unchanged.

        the handle is already acquired before ``async with`` starts; the
        context manager exists only to guarantee release on exit.

        :return: this handle
        :rtype: LeaseHandle
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """async-context-manager exit; releases lease regardless of exception.

        :param exc_type: pending exception class or None
        :ptype exc_type: type[BaseException] | None
        :param exc: pending exception instance or None
        :ptype exc: BaseException | None
        :param tb: pending traceback or None
        :ptype tb: TracebackType | None
        :return: None
        :rtype: None
        """
        await self.release()


class HeldLease:
    """a lease this pod holds and renews in the background, and learns it has lost.

    returned by :meth:`KVLease.hold` (construct it only through that). renewal is the handle's
    compare-and-swap :meth:`LeaseHandle.refresh` on a background task, so a takeover surfaces as
    :class:`LeaseLost` rather than being overwritten. losing the lease is REPORTED, not raised: the
    holder is usually parked in work that has to be interrupted rather than a call that can return,
    so it watches :attr:`lost` (or awaits :meth:`until_lost`) and decides for itself. usable across
    spans no single ``async with`` covers (claimed in one request, released at shutdown) through
    :meth:`release`, or as an async context manager.

    loss is reported no later than the moment the entry could expire -- a timer is armed for that
    instant and pushed back by each successful renewal -- because another pod may take the key the
    moment it does. an entry is only ever assumed alive from the START of the renewal that extended
    it, never from when the reply arrived.

    a hold given a maximum (``max_hold`` on :meth:`KVLease.hold`) stops renewing once it has held
    that long and reports :attr:`LeaseLossReason.MAX_HOLD`, so a holder that WEDGES -- its renewal
    healthy, its work going nowhere -- cannot keep every other claimer out for as long as its
    process lives; the TTL then hands the key on.

    :ivar key: the KV key held
    :ivar lost: set once this pod is no longer the holder; never cleared -- a lease that was lost
        stays lost, because somebody else may already be acting on it
    """

    def __init__(
        self,
        handle: LeaseHandle,
        *,
        ttl_seconds: float,
        renew_every_seconds: float,
        expires_at: float,
        log_extra: Mapping[str, Any] | None = None,
        retake: bool = False,
        name: str = "the lease",
        renew_failure_level: int = logging.INFO,
        holds: _Holds | None = None,
        max_hold_seconds: float | None = None,
        on_lost: Callable[[LeaseLossReason], None] | None = None,
    ) -> None:
        """start renewing ``handle`` in the background. internal: use :meth:`KVLease.hold`.

        :param handle: the freshly acquired lease
        :ptype handle: LeaseHandle
        :param ttl_seconds: how long the entry outlives a missed renewal
        :ptype ttl_seconds: float
        :param renew_every_seconds: seconds between renewals; shorter than the ttl
        :ptype renew_every_seconds: float
        :param expires_at: the event-loop time at which the entry could first expire
        :ptype expires_at: float
        :param log_extra: context added to every log line about this lease
        :ptype log_extra: Mapping[str, Any] | None
        :param retake: take the entry again when it is gone, rather than report the lease lost (see
            :meth:`KVLease.hold`)
        :ptype retake: bool
        :param name: what the lease is, in its log lines
        :ptype name: str
        :param renew_failure_level: the level a renewal that failed inside the TTL is logged at
        :ptype renew_failure_level: int
        :param holds: the factory's current holds, which this one joins until it stops renewing
        :ptype holds: _Holds | None
        :param max_hold_seconds: the longest this lease is renewed, or ``None`` for no limit
        :ptype max_hold_seconds: float | None
        :param on_lost: called with the reason, once, the moment the lease is lost
        :ptype on_lost: Callable[[LeaseLossReason], None] | None
        :return: None
        :rtype: None
        """
        loop = asyncio.get_running_loop()
        self._handle = handle
        self._retake = retake
        self._name = name
        self._renew_failure_level = renew_failure_level
        self._closing = holds.closing if holds is not None else None
        self.key = handle.key
        self.lost = asyncio.Event()
        self._lost_reason: LeaseLossReason | None = None
        self._on_lost = on_lost
        self._hold_deadline = None if max_hold_seconds is None else loop.time() + max_hold_seconds
        self._ended = asyncio.Event()
        self._stop = asyncio.Event()
        self._ttl = ttl_seconds
        self._renew_every = renew_every_seconds
        # a renewal is due every interval, so one not answered within an interval is already late
        self._release_wait = min(renew_every_seconds, RENEWAL_ANSWER_WAIT_SECONDS)
        self._expires_at = expires_at
        self._log_extra = {"key": handle.key, **dict(log_extra or {})}
        self._released = False
        self._expiry = loop.call_at(expires_at, self._on_expired)
        self._renewal: asyncio.Task[None] | None = asyncio.create_task(self._renew(), name=f"kv-lease-renew:{self.key}")
        if holds is not None:
            holds.started()
            self._renewal.add_done_callback(lambda _task: holds.finished())

    @property
    def held(self) -> bool:
        """whether this pod may still act as the holder.

        :return: ``False`` once lost or released
        :rtype: bool
        """
        return not self.lost.is_set() and not self._released

    @property
    def lost_reason(self) -> LeaseLossReason | None:
        """why this lease was lost, or ``None`` while it is held (or after a release, which is no loss).

        :return: the loss reason
        :rtype: LeaseLossReason | None
        """
        return self._lost_reason

    async def until_lost(self) -> None:
        """block until this pod no longer holds the lease -- lost, or released.

        for racing the lease against work that would otherwise run on::

            await asyncio.wait(
                [asyncio.create_task(work()), asyncio.create_task(held.until_lost())],
                return_when=asyncio.FIRST_COMPLETED,
            )

        :attr:`lost` tells the two apart: a release is not a loss.

        :return: None
        :rtype: None
        """
        await self._ended.wait()

    def _mark_lost(self, message: str, reason: LeaseLossReason, *, level: int = logging.WARNING) -> None:
        """record the loss once, stop renewing, wake every waiter, and tell the holder.

        :param message: what to log
        :ptype message: str
        :param reason: why the lease is lost
        :ptype reason: LeaseLossReason
        :param level: the level to log it at
        :ptype level: int
        :return: None
        :rtype: None
        """
        if self.lost.is_set() or self._released:
            return
        log.log(level, message, extra={"extra_data": {**self._log_extra, "reason": reason.value}})
        self._lost_reason = reason
        self.lost.set()
        self._ended.set()
        self._stop.set()
        self._expiry.cancel()
        if self._on_lost is not None:
            self._on_lost(reason)

    def _on_expired(self) -> None:
        """the entry could have expired by now: another pod may hold the key.

        a lease that retakes its entry is not lost by a lapse alone: it says so, and the next renewal
        that reaches the bucket takes the entry back, or finds another holder on it and reports that.

        :return: None
        :rtype: None
        """
        if self._retake:
            log.warning(
                "KVLease: the lease went un-renewed past its TTL and may have lapsed; it is taken again "
                "once the bucket answers",
                extra={"extra_data": self._log_extra},
            )
            return
        self._mark_lost(
            "KVLease: the lease went un-renewed past its TTL and is assumed lost", LeaseLossReason.RENEWAL_FAILED
        )

    async def _renew(self) -> None:
        """renew until lost, released or cancelled.

        :class:`LeaseLost` is authoritative -- another holder is on the entry, or it is gone -- and
        marks the lease lost at once; a lease that retakes tries to take a GONE entry back first, and
        is lost only when another holder is on it. anything else (an unreachable bucket, a transport
        fault, a renewal that hangs) is not evidence of anything, so it is retried; the expiry timer
        decides when the lease could have lapsed, and each renewal is bounded by the time left before
        it (a renewal of a lease that retakes, once that time is spent, by its TTL).

        :return: None
        :rtype: None
        """
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            await self._pause()
            if self._closing is not None and self._closing.is_set() and not self._stop.is_set():
                await self._end_for_close()
                return
            if self._stop.is_set():
                break
            if self._hold_deadline is not None and loop.time() >= self._hold_deadline:
                self._mark_lost(
                    "KVLease: the holder has kept this lease past its maximum hold and is still running; "
                    "it is no longer renewed, so the TTL can hand it on. The work is wedged or far slower "
                    "than the lease was sized for -- another holder may now take it.",
                    LeaseLossReason.MAX_HOLD,
                    level=logging.ERROR,
                )
                return
            started = loop.time()
            remaining = self._expires_at - started
            if remaining <= 0 and not self._retake:
                break  # the expiry timer has fired, or is about to
            try:
                async with asyncio.timeout(remaining if remaining > 0 else self._ttl):
                    await self._renew_once()
            except LeaseLost as exc:
                reason = exc.reason if exc.reason is not None else LeaseLossReason.TAKEN
                self._mark_lost(
                    "KVLease: the lease's entry is gone -- it expired or was removed"
                    if reason is LeaseLossReason.EXPIRED
                    else "KVLease: another holder now owns the lease",
                    reason,
                )
                return
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- a renewal can fail for any transport reason; the expiry timer decides whether that has cost us the lease
                log.log(
                    self._renew_failure_level,
                    "KVLease: renewing %s failed; trying again while it is inside its TTL",
                    self._name,
                    extra={"extra_data": {**self._log_extra, "error": type(exc).__name__}},
                )
            else:
                self._expires_at = started + self._ttl
                self._expiry.cancel()
                self._expiry = loop.call_at(self._expires_at, self._on_expired)

    async def _pause(self) -> None:
        """wait one renewal interval, or less when this lease is released or its factory closes.

        :return: None
        :rtype: None
        """
        waits = {asyncio.ensure_future(self._stop.wait())}
        if self._closing is not None:
            waits.add(asyncio.ensure_future(self._closing.wait()))
        try:
            await asyncio.wait(waits, timeout=self._renew_every, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for waiting in waits:
                waiting.cancel()

    async def _end_for_close(self) -> None:
        """let the lease go because its factory closed: as :meth:`release`, from the renewal itself.

        :return: None
        :rtype: None
        """
        self._released = True
        self._ended.set()
        self._stop.set()
        self._expiry.cancel()
        self._renewal = None
        await self._delete_entry()

    async def _renew_once(self) -> None:
        """one renewal: the handle's compare-and-swap refresh, and for a lease that retakes, a take of
        the entry again when the refresh found it gone.

        :return: None
        :rtype: None
        :raises LeaseLost: another holder is on the entry, or (not retaking) the entry is gone
        """
        try:
            await self._handle.refresh(adopt_own_revision=self._retake)
        except LeaseLost:
            if not self._retake or not await self._handle.retake():
                raise
            log.warning("KVLease: the lease's entry was gone; it is taken again", extra={"extra_data": self._log_extra})

    async def release(self) -> None:
        """stop renewing and delete the entry if this pod still holds it; idempotent.

        a renewal in flight is given a short while to be answered -- one renewal interval, and never
        longer than one KV operation's own ceiling (:data:`RENEWAL_ANSWER_WAIT_SECONDS`) -- so the delete
        that follows is keyed on the revision the server really holds; past that the renewal is
        cancelled and the release goes on, so a broker that stopped answering cannot hold an owner's
        shutdown for a TTL. A cancelled renewal may still have landed: the entry is then deleted only
        when it is exactly that renewal's write (:attr:`LeaseHandle.in_flight_value`), and otherwise left
        to lapse by its TTL, which the release logs. releasing a
        LOST lease is a no-op on the entry (the delete is fenced on the holder), so it never frees the
        new holder's claim. best-effort: the likeliest reason a release fails is the unreachable bucket
        that cost the lease in the first place, and a cleanup error must not replace whatever ended the
        caller's work -- the TTL frees an entry nobody deleted.

        :return: None
        :rtype: None
        """
        if self._released:
            return
        self._released = True
        self._ended.set()
        self._stop.set()
        self._expiry.cancel()
        renewal, self._renewal = self._renewal, None
        try:
            if renewal is not None:
                done, _pending = await asyncio.wait([renewal], timeout=self._release_wait)
                if not done:
                    renewal.cancel()
                    await asyncio.wait([renewal])
        finally:
            # even when the releasing task is cancelled meanwhile (an owner stopping): an entry left
            # behind holds everyone else off for its whole TTL
            await self._delete_entry()

    async def _delete_entry(self) -> None:
        """delete the entry if this pod still holds it; never raises (see :meth:`release`).

        :return: None
        :rtype: None
        """
        try:
            await self._handle.release()
        except Exception:  # prawduct:allow prawduct/broad-except -- see above: the TTL frees the entry, and a cleanup failure must not replace the caller's outcome
            log.warning(
                "KVLease: could not release the lease; it will expire with its TTL",
                extra={"extra_data": self._log_extra},
                exc_info=True,
            )

    async def __aenter__(self) -> HeldLease:
        """async-context-manager entry; the lease is already held.

        :return: this held lease
        :rtype: HeldLease
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """release on exit, whatever the body did.

        :param exc_type: pending exception class or None
        :ptype exc_type: type[BaseException] | None
        :param exc: pending exception instance or None
        :ptype exc: BaseException | None
        :param tb: pending traceback or None
        :ptype tb: TracebackType | None
        :return: None
        :rtype: None
        """
        await self.release()


def _hold_seconds(ttl: timedelta | float, renew_every: timedelta | float) -> tuple[int, float]:
    """validate a hold's timing: a whole-second TTL of at least one, renewed more often than it expires.

    :param ttl: how long the entry outlives a missed renewal
    :ptype ttl: timedelta | float
    :param renew_every: seconds between renewals
    :ptype renew_every: timedelta | float
    :return: ``(ttl_seconds, renew_every_seconds)``
    :rtype: tuple[int, float]
    :raises ValueError: a sub-second or fractional TTL (the entry expresses whole seconds, and 0.5
        would truncate to an entry stale the instant it lands), a non-positive renewal, or a
        renewal not shorter than the TTL (the lease would lapse under its own holder)
    """
    ttl_s = ttl.total_seconds() if isinstance(ttl, timedelta) else float(ttl)
    renew_s = renew_every.total_seconds() if isinstance(renew_every, timedelta) else float(renew_every)
    if ttl_s < 1 or ttl_s != int(ttl_s):
        raise ValueError(f"ttl {ttl} must be a whole number of seconds and at least one second")
    if renew_s <= 0:
        raise ValueError(f"renew_every must be positive, got {renew_every}")
    if renew_s >= ttl_s:
        raise ValueError(
            f"renew_every {renew_every} must be shorter than ttl {ttl}, or the lease lapses under its holder"
        )
    return int(ttl_s), renew_s


class KVLease:
    """distributed mutex factory backed by NATS JetStream KV.

    one :class:`KVLease` instance may mint many :class:`LeaseHandle`\\ s,
    each keyed on a different KV key. holder identity (``pod_id``) is
    shared across all leases minted by one factory so one pod consistently
    owns every lease it acquires.
    """

    def __init__(
        self,
        nats_client: "KvCapable | None",
        bucket_name: str | None = None,
        pod_id: str | None = None,
        *,
        create_if_missing: bool = True,
        key_scope: str | None = None,
        bucket: "KvBucketLike | None" = None,
        expire_entries: bool = False,
    ) -> None:
        """configure factory; defer bucket creation until first acquire.

        ``bucket_name`` is a SUFFIX, not a full name: it is handed to
        :meth:`KvCapable.kv_bucket`, which layers the connection's own
        ``{namespace}-`` over it. Passing a name that already carries the
        namespace therefore produces it twice. The default is the constant
        ``"leases"`` for exactly that reason. default ``pod_id`` is
        ``f"pod-{uuid7().hex}"`` -- the WHOLE uuid7, unique per factory instance.
        the holder id is the fence on refresh and release, so it must name one
        factory; a uuid7's leading hex is its millisecond timestamp, and a prefix
        of it named every factory built in the same millisecond.

        :param nats_client: connected canonical
            :class:`threetears.nats.kv.KvCapable` wrapper; the lease
            opens its KV bucket through :meth:`KvCapable.kv_bucket`
        :ptype nats_client: KvCapable
        :param bucket_name: explicit bucket-name SUFFIX (the transport adds the
            namespace prefix); None uses the constant default
        :ptype bucket_name: str | None
        :param pod_id: explicit holder identifier; None auto-generates one
        :ptype pod_id: str | None
        :param create_if_missing: ``True`` (the default) DECLARES the bucket, creating it
            when absent; ``False`` only BINDS a bucket another identity declared, and never
            issues STREAM.CREATE -- for a process whose grant on the bucket is key-addressed
            only, where a refused create would cost the full JetStream deadline first
        :ptype create_if_missing: bool
        :param key_scope: the owner scope every lease key leads with (``{key_scope}.{key}``), for a
            bucket SHARED by many owners -- the platform's ``leases``, which every tool pod binds and
            in which each pod is granted only the keys under its own
            :func:`~threetears.nats.subject_permissions.kv_key_scope_for`. Replicas of one owner
            share the scope and so contend for one key. ``None`` keys by the caller's key alone,
            for a bucket this factory's owner has to itself
        :ptype key_scope: str | None
        :param bucket: a bucket already bound, in place of ``nats_client``: one another owner declared
            and handed over (a pod's pointer bucket, bound by its own name). The lease opens nothing,
            and its keys are exactly the names it is given (led by ``key_scope``, when set)
        :ptype bucket: KvBucketLike | None
        :param expire_entries: write every entry with a NATS per-key TTL equal to the lease's, so a
            holder that stops renewing loses the KEY, not only its envelope's expiry -- for a bucket
            whose readers test whether a key exists rather than decode it. The bucket must allow
            per-key TTLs
        :ptype expire_entries: bool
        :return: None
        :rtype: None
        :raises ValueError: when ``key_scope`` is not one literal subject token, or not exactly one of
            ``nats_client`` and ``bucket`` is given
        """
        if (nats_client is None) == (bucket is None):
            raise ValueError(
                "KVLease takes a nats_client to open its bucket, or a bucket already bound: one of the two"
            )
        self._key_scope = validated_key_scope(key_scope, primitive="KVLease")
        self._client = nats_client
        self._bucket_name = bucket_name if bucket_name is not None else self._default_bucket_name()
        self._pod_id = pod_id if pod_id is not None else f"pod-{uuid7().hex}"
        self._create_if_missing = create_if_missing
        self._bucket: "KvBucketLike | None" = bucket
        self._bucket_lock = asyncio.Lock()
        self._expire_entries = expire_entries
        self._holds = _Holds()

    @property
    def bucket_name(self) -> str:
        """return configured bucket name.

        :return: bucket name set at construction; for a lease handed its bucket, that bucket's name
        :rtype: str
        """
        if self._client is None and self._bucket is not None:
            return self._bucket.name
        return self._bucket_name

    @property
    def pod_id(self) -> str:
        """return holder identifier used for every acquire.

        :return: holder identifier
        :rtype: str
        """
        return self._pod_id

    @staticmethod
    def _default_bucket_name() -> str:
        """the bucket-name SUFFIX this lease opens by default.

        a SUFFIX, not a full name, because that is what
        :meth:`threetears.nats.kv.KvCapable.kv_bucket` takes: it layers the
        connection's own ``{namespace}-`` prefix over whatever it is handed. an
        earlier version of this returned ``f"{ns}_leases"``, reading the
        namespace itself and baking it in, which made the bucket that actually
        materialised ``{ns}-{ns}_leases`` -- the namespace twice. Nothing
        detected it because a KV grant that names a bucket no opener produces
        is not an error, it is a silent JetStream timeout on first use.

        The namespace therefore must NOT be read here. It is the transport's to
        apply, and applying it in both places is how it got applied twice.

        :return: the constant suffix ``"leases"``
        :rtype: str
        """
        return "leases"

    def stored_key(self, key: str) -> str:
        """the KV key a lease named ``key`` is stored under: led by the factory's owner scope, if any.

        For state kept beside a lease under the same grant (a run's pending request, say), which must
        land where the owner's keys do or a scoped grant refuses it.

        :param key: the name
        :ptype key: str
        :return: the stored key
        :rtype: str
        """
        return owner_scoped_key(self._key_scope, key)

    async def bucket(self) -> "KvBucketLike":
        """the bucket this factory's leases live in, opened as an acquire opens it.

        :return: the bucket
        :rtype: KvBucketLike
        """
        return await self._ensure_bucket()

    async def _ensure_bucket(self) -> "KvBucketLike":
        """open the bucket with history=1 on first call: declare it, or bind only when so configured.

        lazy, async-safe: an ``asyncio.Lock`` serializes first-call setup
        so two concurrent acquires do not race to create the same
        bucket. routes through :meth:`KvCapable.kv_bucket` which
        idempotently creates-or-binds. the wrapper auto-prefixes the
        bucket name with the connected client's namespace prefix; the
        lease's ``bucket_name`` is treated as the bucket suffix the
        wrapper layers on top of that prefix.

        :return: backing wrapper-typed KV bucket
        :rtype: KvBucketLike
        """
        if self._bucket is not None:
            return self._bucket
        async with self._bucket_lock:
            if self._bucket is None:
                assert self._client is not None  # a lease without a client was handed its bucket
                self._bucket = await self._client.kv_bucket(
                    name=self._bucket_name,
                    history=1,
                    create_if_missing=self._create_if_missing,
                )
                log.info(
                    "KVLease bound bucket %s (create_if_missing=%s)",
                    self._bucket_name,
                    self._create_if_missing,
                )
        return self._bucket

    async def close(self) -> None:
        """end every hold this factory has handed out: each stops renewing and deletes its entry if
        it is still its own, and this returns once all have. Never raises.

        For an owner that stops while holds it handed to callers may still be out (a caller that
        dropped one, or never reached its release): without it they renew for the life of the
        process. Holds taken after it are not affected.

        :return: None
        :rtype: None
        """
        holds, self._holds = self._holds, _Holds()
        holds.closing.set()
        await holds.ended.wait()

    def _entry_ttl(self, ttl_seconds: int) -> timedelta | None:
        """the per-key TTL an entry is written with: the lease's own when entries expire, else none.

        :param ttl_seconds: the lease's TTL
        :ptype ttl_seconds: int
        :return: the per-key TTL, or ``None``
        :rtype: timedelta | None
        """
        return timedelta(seconds=ttl_seconds) if self._expire_entries else None

    async def acquire(
        self,
        key: str,
        ttl_seconds: int = 30,
        max_wait_seconds: int = 60,
    ) -> LeaseHandle:
        """acquire lease on ``key``; block up to ``max_wait_seconds`` if held.

        algorithm:

        1. try ``bucket.create(key, payload)`` — wins on empty key.
        2. on key-exists, ``get`` current entry; if expiry is in the past,
           attempt CAS ``update`` to reclaim. success -> lease acquired.
        3. if still held by live holder and ``max_wait_seconds == 0``, raise
           :class:`LeaseUnavailable` immediately (no sleep).
        4. otherwise sleep ``min(1.0, remaining_time)`` and retry.
        5. on deadline elapsed, raise :class:`LeaseTimeout`.

        :param key: the lease's name; the KV key is this, led by the factory's ``key_scope`` when
            it has one (the handle carries the stored key)
        :ptype key: str
        :param ttl_seconds: seconds past acquisition at which entry goes stale
        :ptype ttl_seconds: int
        :param max_wait_seconds: total seconds caller is willing to block;
            0 disables blocking and forces fail-fast on contention
        :ptype max_wait_seconds: int
        :return: handle representing successful acquisition
        :rtype: LeaseHandle
        :raises LeaseUnavailable: if ``max_wait_seconds == 0`` and key is held
        :raises LeaseTimeout: if deadline elapses before lease becomes free
        """
        bucket = await self._ensure_bucket()
        key = owner_scoped_key(self._key_scope, key)
        deadline = datetime.now(UTC) + timedelta(seconds=max_wait_seconds)
        handle: LeaseHandle | None = None
        timed_out = False
        while handle is None and not timed_out:
            attempt = await self._try_once(bucket, key, ttl_seconds)
            if attempt is not None:
                handle = attempt
                break
            # key was held by a live holder; decide whether to wait
            now = datetime.now(UTC)
            remaining = (deadline - now).total_seconds()
            if max_wait_seconds == 0:
                raise LeaseUnavailable(f"lease {key!r} held by another pod; fail-fast requested")
            if remaining <= 0:
                timed_out = True
                break
            await asyncio.sleep(min(1.0, remaining))
        if handle is None:
            raise LeaseTimeout(f"lease {key!r} not acquired within {max_wait_seconds}s")
        return handle

    async def _try_once(
        self,
        bucket: "KvBucketLike",
        key: str,
        ttl_seconds: int,
    ) -> LeaseHandle | None:
        """single pass of create-or-reclaim-stale against the KV bucket.

        returns :class:`LeaseHandle` when lease is ours; returns ``None``
        when key is currently held by a live holder and caller should wait
        or fail-fast.

        :param bucket: backing wrapper KV bucket (from :meth:`_ensure_bucket`)
        :ptype bucket: KvBucketLike
        :param key: KV key under which lease entry lives
        :ptype key: str
        :param ttl_seconds: TTL to record in envelope when winning
        :ptype ttl_seconds: int
        :return: handle on successful acquire, None on live contention
        :rtype: LeaseHandle | None
        """
        now = datetime.now(UTC)
        date_expires = now + timedelta(seconds=ttl_seconds)
        payload = _encode_envelope(holder=self._pod_id, date_expires=date_expires, date_acquired=now)
        result: LeaseHandle | None = None
        # KvBucketLike.create returns the new revision on success or
        # ``None`` on CAS conflict (key already exists). no exception
        # to catch for the conflict path -- the wrapper hides
        # KeyWrongLastSequenceError behind the typed ``None`` return.
        revision = await bucket.create(key=key, value=payload, ttl=self._entry_ttl(ttl_seconds))
        if revision is not None:
            result = LeaseHandle(
                lease=self,
                key=key,
                holder=self._pod_id,
                revision=revision,
                ttl_seconds=ttl_seconds,
                date_expires=date_expires,
            )
        else:
            result = await self._maybe_reclaim_stale(bucket, key, ttl_seconds, now, date_expires)
        return result

    async def _maybe_reclaim_stale(
        self,
        bucket: "KvBucketLike",
        key: str,
        ttl_seconds: int,
        now: datetime,
        date_expires: datetime,
    ) -> LeaseHandle | None:
        """attempt CAS reclaim when existing entry is expired; else return None.

        :param bucket: backing wrapper KV bucket
        :ptype bucket: KvBucketLike
        :param key: KV key under which lease entry lives
        :ptype key: str
        :param ttl_seconds: TTL to record on successful reclaim
        :ptype ttl_seconds: int
        :param now: captured ``datetime.now(UTC)`` for this attempt
        :ptype now: datetime
        :param date_expires: pre-computed new expires_at for reclaim payload
        :ptype date_expires: datetime
        :return: handle when reclaim wins, None when entry is still live or
            CAS lost to another pod this round
        :rtype: LeaseHandle | None
        """
        result: LeaseHandle | None = None
        # KvBucketLike.get_entry returns (value, revision) on hit or
        # ``None`` on miss; the entry could vanish between the create
        # race and this read so a None return means "caller will
        # retry" same as if it had been a KeyNotFound raise.
        entry = await bucket.get_entry(key=key)
        if entry is None:
            return None
        value, revision = entry
        envelope = _decoded_envelope(value)
        if envelope is None or envelope.date_expires > now:
            # still held by a live holder
            return None
        payload = _encode_envelope(holder=self._pod_id, date_expires=date_expires, date_acquired=now)
        # KvBucketLike.update returns the new revision on success or
        # ``None`` on CAS conflict; another pod may have reclaimed the
        # stale entry between our get_entry and update.
        new_revision = await bucket.update(key=key, value=payload, revision=revision, ttl=self._entry_ttl(ttl_seconds))
        if new_revision is None:
            return None
        result = LeaseHandle(
            lease=self,
            key=key,
            holder=self._pod_id,
            revision=new_revision,
            ttl_seconds=ttl_seconds,
            date_expires=date_expires,
        )
        return result

    async def hold(
        self,
        key: str,
        *,
        ttl: timedelta | float,
        renew_every: timedelta | float,
        max_wait_seconds: int = 0,
        log_extra: Mapping[str, Any] | None = None,
        retake: bool = False,
        name: str = "the lease",
        renew_failure_level: int = logging.INFO,
        max_hold: timedelta | float | None = None,
        on_lost: Callable[[LeaseLossReason], None] | None = None,
    ) -> HeldLease:
        """acquire ``key`` and keep it renewed in the background until released or lost.

        the difference from :meth:`acquire` is the renewal loop and the loss signal, which every
        caller holding a lease across real work otherwise writes for itself. fails fast by default
        (``max_wait_seconds=0``): a caller that cannot hold the lease usually wants to say so now.

        :param key: KV key to hold
        :ptype key: str
        :param ttl: how long the entry outlives a missed renewal; whole seconds, at least one
        :ptype ttl: timedelta | float
        :param renew_every: how often to renew; shorter than ``ttl`` (a third leaves room for two
            missed renewals)
        :ptype renew_every: timedelta | float
        :param max_wait_seconds: how long to wait for a held key; 0 refuses at once
        :ptype max_wait_seconds: int
        :param log_extra: context for every log line about this lease (the key alone is often a
            digest an operator cannot map back to anything)
        :ptype log_extra: Mapping[str, Any] | None
        :param retake: when a renewal finds the entry GONE -- it lapsed through an outage, or NATS lost
            the bucket -- take it again rather than report the lease lost, and treat a lapse as a
            warning, not a loss. Only an entry another holder is on loses the lease. For a key that
            names its holder alone (a writer's claim on what it writes), where an absent entry
            means nobody else has it; never for a mutex others contend for, whose lapse may already
            have let another pod act
        :ptype retake: bool
        :param name: what the lease is, for its log lines ("the snapshot rebuild claim")
        :ptype name: str
        :param renew_failure_level: the level a renewal that failed while the entry is still inside its
            TTL is logged at; INFO by default (it is not yet evidence of anything), WARNING for an
            owner whose operators watch for it
        :ptype renew_failure_level: int
        :param max_hold: the longest the lease is renewed; past it renewal stops, the lease reports
            :attr:`LeaseLossReason.MAX_HOLD` and the TTL hands the key on, so a wedged holder cannot
            keep every other claimer out for as long as its process lives. ``None`` (the default)
            renews until released or lost
        :ptype max_hold: timedelta | float | None
        :param on_lost: called once, synchronously, with the reason the moment the lease is lost --
            for a holder that must interrupt its work then (cancel a task), not at its next look at
            :attr:`HeldLease.lost`. Runs on the renewal task or a loop callback, so it must not block
            or raise
        :ptype on_lost: Callable[[LeaseLossReason], None] | None
        :return: the held lease, already renewing
        :rtype: HeldLease
        :raises ValueError: timing that cannot hold (see :func:`_hold_seconds`), or a negative
            ``max_hold``
        :raises LeaseUnavailable: the key is held and ``max_wait_seconds`` is 0
        :raises LeaseTimeout: the key stayed held past ``max_wait_seconds``
        """
        ttl_seconds, renew_seconds = _hold_seconds(ttl, renew_every)
        max_hold_seconds = max_hold.total_seconds() if isinstance(max_hold, timedelta) else max_hold
        if max_hold_seconds is not None and max_hold_seconds < 0:
            raise ValueError(f"max_hold {max_hold} must not be negative")
        loop = asyncio.get_running_loop()
        handle = await self.acquire(key, ttl_seconds=ttl_seconds, max_wait_seconds=max_wait_seconds)
        # the entry expires when its envelope says -- the instant other pods treat it as stale, stamped
        # at the start of the winning attempt -- not ttl after acquire() returned, which would overstate
        # its life by a round trip.
        assert handle.date_expires is not None  # both acquire paths stamp it
        expires_at = loop.time() + (handle.date_expires - datetime.now(UTC)).total_seconds()
        return HeldLease(
            handle,
            ttl_seconds=ttl_seconds,
            renew_every_seconds=renew_seconds,
            expires_at=expires_at,
            log_extra=log_extra,
            retake=retake,
            name=name,
            renew_failure_level=renew_failure_level,
            holds=self._holds,
            max_hold_seconds=max_hold_seconds,
            on_lost=on_lost,
        )

    async def refresh_handle(
        self, handle: LeaseHandle, ttl_seconds: int | None, *, adopt_own_revision: bool = False
    ) -> None:
        """implementation of :meth:`LeaseHandle.refresh`.

        public because :class:`LeaseHandle` lives in a sibling class and
        forwards its :meth:`refresh` here. the refresh pipeline (fetch
        entry, verify holder, CAS-update) is owned by the lease object
        that holds the KV bucket; the handle is just the revision token.

        :param handle: handle requesting refresh
        :ptype handle: LeaseHandle
        :param ttl_seconds: override TTL; None reuses handle's recorded TTL
        :ptype ttl_seconds: int | None
        :param adopt_own_revision: see :meth:`LeaseHandle.refresh`
        :ptype adopt_own_revision: bool
        :return: None
        :rtype: None
        :raises LeaseLost: if ownership changed or CAS update failed
        """
        bucket = await self._ensure_bucket()
        entry = await bucket.get_entry(key=handle.key)
        if entry is None:
            raise LeaseLost(f"lease {handle.key!r} entry missing during refresh", reason=LeaseLossReason.EXPIRED)
        value, observed = entry
        envelope = _decoded_envelope(value)
        if envelope is None or envelope.holder != handle.holder:
            raise LeaseLost(
                f"lease {handle.key!r} holder changed: expected {handle.holder!r}, "
                f"found {'an entry it cannot read' if envelope is None else repr(envelope.holder)}",
                reason=LeaseLossReason.TAKEN,
            )
        if adopt_own_revision and observed != handle.revision:
            # still this holder's entry: one of its own writes landed while its reply was lost
            handle.revision = observed
        effective_ttl = ttl_seconds if ttl_seconds is not None else handle.ttl_seconds
        now = datetime.now(UTC)
        date_expires = now + timedelta(seconds=effective_ttl)
        payload = _encode_envelope(holder=handle.holder, date_expires=date_expires, date_acquired=now)
        # KvBucketLike.update returns ``None`` on revision mismatch,
        # mapping the previous KeyWrongLastSequenceError raise into a
        # typed conflict signal we surface as LeaseLost.
        handle.in_flight_value = payload
        new_revision = await bucket.update(
            key=handle.key, value=payload, revision=handle.revision, ttl=self._entry_ttl(effective_ttl)
        )
        handle.in_flight_value = None
        if new_revision is None:
            # only another holder's write moves the entry on between the read and the swap
            raise LeaseLost(f"lease {handle.key!r} revision advanced during refresh", reason=LeaseLossReason.TAKEN)
        handle.revision = new_revision
        handle.ttl_seconds = effective_ttl

    async def retake_handle(self, handle: LeaseHandle) -> bool:
        """implementation of :meth:`LeaseHandle.retake`, public for the reason :meth:`refresh_handle` is.

        :param handle: the handle whose entry is gone
        :ptype handle: LeaseHandle
        :return: whether the entry is this holder's again
        :rtype: bool
        """
        bucket = await self._ensure_bucket()
        now = datetime.now(UTC)
        date_expires = now + timedelta(seconds=handle.ttl_seconds)
        payload = _encode_envelope(holder=handle.holder, date_expires=date_expires, date_acquired=now)
        handle.in_flight_value = payload
        revision = await bucket.create(key=handle.key, value=payload, ttl=self._entry_ttl(handle.ttl_seconds))
        handle.in_flight_value = None
        if revision is None:
            return False
        handle.revision = revision
        handle.date_expires = date_expires
        return True

    async def release_handle(self, handle: LeaseHandle) -> None:
        """implementation of :meth:`LeaseHandle.release`; idempotent.

        public for the same reason as :meth:`refresh_handle` -- the
        handle forwards its :meth:`release` to the lease object so the
        lease can verify ownership before deleting the KV entry.

        :param handle: handle requesting release
        :ptype handle: LeaseHandle
        :return: None
        :rtype: None
        """
        if handle.released:
            return
        bucket = await self._ensure_bucket()
        entry = await bucket.get_entry(key=handle.key)
        if entry is None:
            handle.released = True
            return
        value, observed = entry
        envelope = _decoded_envelope(value)
        if envelope is None or envelope.holder != handle.holder:
            # someone else owns it now; not our entry to delete
            handle.released = True
            return
        revision = handle.revision
        if observed != revision and handle.in_flight_value is not None and value == handle.in_flight_value:
            # this handle's own renewal, applied by the server after the handle stopped waiting for
            # its answer: the entry is ours, one revision on from what the handle recorded
            revision = observed
        # CAS delete keyed on the handle's revision so a stale-after-
        # check delete cannot evict a successor's freshly reclaimed
        # entry. KvBucketLike.delete returns False on revision mismatch
        # which is treated as "raced with another writer; entry
        # effectively gone from our view" -- diagnostic-only here.
        deleted = await bucket.delete(key=handle.key, revision=revision)
        if not deleted:
            # a write landed after the read: another holder's, or a renewal of this one's that is not
            # recognisably its own -- either way not provably this holder's to delete
            log.info(
                "KVLease: the entry %s moved on before its release could delete it; it is left to lapse by its TTL",
                handle.key,
            )
        handle.released = True
