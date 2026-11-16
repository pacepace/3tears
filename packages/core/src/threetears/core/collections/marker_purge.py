"""purge the delete markers the collections L2 bucket holds, once no read can still be fenced on them.

Every delete in ``{ns}-collections`` leaves a marker, and the bucket has no expiry, so the marker stays: on cobalt on
2026-11-16 markers were 113,348 of the bucket's 119,598 keys on dev and 168,066 of 179,878 on prod (about 16 MB and
23 MB), gained over the 14 days since the bucket was created. The bucket refuses new keys at its bound (discard NEW),
so markers that never go would, in time, refuse real rows. :class:`RemovalMarkerPurger` removes the old ones: one
owner's pass, under a lease, walking the stream by sequence a page at a time
(:meth:`threetears.nats.NatsKvBucket.purge_aged_removal_markers`).

Why a marker must stay a while: it is a revision, and the L2 fences on revisions. A three-tier read takes the key's
latest revision (a marker's included, ``0`` for a key with no message) BEFORE its L3 query and seeds the row it read
at that revision (``BaseCollection._seed_l2``); a save takes it before its L3 write (``_l2_before_write``); an
L2-first commit applies at the revision it read. Each lands only if nothing happened to the key in between. Purging
a marker inside that window rewinds the key: a read that found no message (revision ``0``), while a writer committed
a newer row and deleted the key (a marker), seeds its older L3 row at ``0`` once the marker is gone, and every
replica serves the older row. So a marker is purged only once it is older than any such window can be:
:data:`L2_MARKER_PURGE_AGE`.

The windows, measured or bounded: one L3 round trip between the revision read and the write that depends on it
(the hub pool's ``command_timeout`` is 30 s; the L3 broker proxy's default statement timeout is 5 s,
``THREETEARS_NATS_PROXY_TIMEOUT_MS``). A pod's own pool may carry no statement timeout, and a process stalled (a
paused node, a starved CPU) holds its revision across the stall, so the age is set far past any bounded window
rather than at it. Watchers do not lengthen it: a key watch whose consumer went idle 30 s is re-created and replays
each key's latest message, and the only keys watched in this bucket are table write generations, which are never
deleted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final, Protocol

from threetears.observe import get_logger
from threetears.observe.metrics import counter, gauge, histogram

from threetears.core.coordination.lease import LeaseUnavailable

if TYPE_CHECKING:
    from collections.abc import Callable

    from threetears.nats import AgedMarkerPurge

    from threetears.core.coordination.lease import KVLease

__all__ = [
    "L2_MARKER_PURGE_AGE",
    "L2_MARKER_PURGE_INTERVAL",
    "MarkerPurgePass",
    "RemovalMarkerPurger",
]

log = get_logger(__name__)

#: how old a marker must be before it is purged. A day: about 2,900 times the longest bounded window a revision is
#: held across (the hub pool's 30 s ``command_timeout``), so only a process stalled for a day could still hold one.
#: What it costs is a day of markers (plus one interval) left in the bucket: on 2026-11-16 the bucket gained about
#: 8,100 a day on dev and 12,000 on prod (113,348 / 168,066 in 14 days, about 140 bytes each), so 1.1-1.7 MB, under
#: 0.4% of its 512 MiB bound.
L2_MARKER_PURGE_AGE: Final = timedelta(hours=24)

#: how often the owner runs a pass. A pass walks only the messages that aged past :data:`L2_MARKER_PURGE_AGE` since
#: the one before (it resumes at the sequence that one stopped at): an hour's worth, about 340-500 markers at the
#: 2026-11-16 rates, so the bucket holds at most a day and an hour of markers.
L2_MARKER_PURGE_INTERVAL: Final = timedelta(hours=1)

#: the messages one page reads: each is one ``DIRECT.GET`` reply of one entry, so a page never holds the event loop
#: longer than one entry's handling, and the lease is looked at between pages
_PAGE_MESSAGES: Final = 1_000

#: the most messages one pass reads before it stops and reports itself behind. The first pass over a bucket that has
#: never been purged walks every message older than the age (dev: 119,598, prod: 179,878 on 2026-11-16): at this
#: budget prod's backlog is gone in four hourly passes, and no pass runs long against the server's purge API.
_MAX_MESSAGES_PER_PASS: Final = 50_000

#: the lease one purger holds for a pass, and how long a dead holder's hold lasts. Renewed in the background, so a
#: pass may run past it; two passes at once would be safe (each purge is fenced on its marker's sequence), only
#: wasteful.
_LEASE_TTL: Final = timedelta(seconds=60)
_LEASE_RENEW_EVERY: Final = timedelta(seconds=20)

_PASSES = counter(
    "threetears_kv_marker_purge_passes_total",
    description=(
        "marker purge passes by outcome: done (caught up), behind (stopped at its budget), skipped (another "
        "replica held the lease), lost (the lease was lost mid-pass), failed"
    ),
    label_names=("bucket", "outcome"),
)
_PURGED = counter(
    "threetears_kv_markers_purged_total",
    description="delete and purge markers older than the purge age removed from a KV bucket",
    label_names=("bucket",),
)
_PASS_SECONDS = histogram(
    "threetears_kv_marker_purge_pass_seconds",
    description="how long one marker purge pass took, from taking the lease to its last page",
    label_names=("bucket",),
)
_OLDEST_AGE = gauge(
    "threetears_kv_marker_oldest_age_seconds",
    description=(
        "the age of the oldest marker this replica's last pass found past the purge age (0 when it found none or "
        "another replica held the lease): about the purge age plus one interval while the purge keeps up, more "
        "when it is behind"
    ),
    label_names=("bucket",),
)


class _MarkerAgingBucket(Protocol):
    """the slice of :class:`threetears.nats.NatsKvBucket` a purger needs."""

    @property
    def name(self) -> str:
        """the bucket's full name.

        :return: the name
        :rtype: str
        """
        ...

    async def date_created(self) -> datetime:
        """when the bucket's backing stream was created, read fresh.

        :return: the creation time
        :rtype: datetime
        """
        ...

    async def purge_aged_removal_markers(
        self, *, before: datetime, from_sequence: int = 1, max_messages: int, apply: bool = True
    ) -> AgedMarkerPurge:
        """purge one page of the markers stored before ``before``.

        :param before: the cutoff
        :ptype before: datetime
        :param from_sequence: where the page starts
        :ptype from_sequence: int
        :param max_messages: the most messages the page reads
        :ptype max_messages: int
        :param apply: purge; ``False`` only counts
        :ptype apply: bool
        :return: what the page did
        :rtype: AgedMarkerPurge
        """
        ...


@dataclass(frozen=True, slots=True)
class MarkerPurgePass:
    """what one pass did.

    :ivar purged: markers purged (without ``apply``, found)
    :ivar read: messages read
    :ivar oldest_marker_age: the age of the oldest marker found past the purge age; ``None`` when none was
    :ivar behind: the pass stopped at its budget with messages older than the age left to read
    :ivar skipped: another holder had the lease, so nothing was read
    :ivar lost: the lease was lost mid-pass, so the pass stopped early
    :ivar seconds: how long the pass took
    """

    purged: int
    read: int
    oldest_marker_age: timedelta | None
    behind: bool
    skipped: bool = False
    lost: bool = False
    seconds: float = 0.0


class RemovalMarkerPurger:
    """purge the markers in one KV bucket older than an age, a pass at a time, one replica at a time.

    A pass resumes at the sequence the one before stopped at: below it are only the values the walk kept, since a
    delete always appends, so an hourly pass reads only what aged past the cutoff in the hour. The position is this
    process's own; a pass on another replica, or after a restart, starts at the stream's first message and walks
    the older values once more. A bucket whose stream was created again (NATS lost it) restarts its sequences, so a
    pass that finds a new creation time starts over at the first message.

    :param bucket: the bucket, bound through its owner's connection (a stream purge is the owner's verb)
    :ptype bucket: _MarkerAgingBucket
    :param lease: one pass at a time across replicas; ``None`` runs without one (a one-shot command)
    :ptype lease: KVLease | None
    :param age: markers older than this are purged
    :ptype age: timedelta
    :param max_messages_per_pass: the most messages one pass reads
    :ptype max_messages_per_pass: int
    :param page_messages: the most messages one page reads
    :ptype page_messages: int
    :param clock: the current time, timezone-aware
    :ptype clock: Callable[[], datetime]
    """

    def __init__(
        self,
        bucket: _MarkerAgingBucket,
        *,
        lease: KVLease | None,
        age: timedelta = L2_MARKER_PURGE_AGE,
        max_messages_per_pass: int = _MAX_MESSAGES_PER_PASS,
        page_messages: int = _PAGE_MESSAGES,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """bind the bucket and the lease; no I/O.

        :param bucket: the bucket
        :ptype bucket: _MarkerAgingBucket
        :param lease: the pass lease, or ``None``
        :ptype lease: KVLease | None
        :param age: the purge age
        :ptype age: timedelta
        :param max_messages_per_pass: the pass budget
        :ptype max_messages_per_pass: int
        :param page_messages: the page size
        :ptype page_messages: int
        :param clock: the current time
        :ptype clock: Callable[[], datetime] | None
        :return: nothing
        :rtype: None
        :raises ValueError: when the age is not positive or a budget is under one
        """
        if age <= timedelta(0):
            raise ValueError(f"a marker purge age must be positive, got {age}")
        if max_messages_per_pass < 1 or page_messages < 1:
            raise ValueError("a marker purge reads at least one message a pass and a page")
        self._bucket = bucket
        self._lease = lease
        self._age = age
        self._max_messages = max_messages_per_pass
        self._page = page_messages
        self._clock = clock if clock is not None else (lambda: datetime.now(UTC))
        self._next_sequence = 1
        self._created: datetime | None = None

    async def run_pass(self, *, apply: bool = True) -> MarkerPurgePass:
        """purge (or, without ``apply``, count) the markers older than the age, under the lease when there is one.

        :param apply: purge; ``False`` only counts
        :ptype apply: bool
        :return: what the pass did
        :rtype: MarkerPurgePass
        :raises KvError: when the bucket cannot be read or a purge fails, after counting the pass as failed
        """
        bucket = self._bucket.name
        started = time.monotonic()
        try:
            if self._lease is None:
                done = await self._walk(apply=apply, lost=lambda: False)
            else:
                try:
                    held = await self._lease.hold(
                        f"kv-marker-purge.{bucket}",
                        ttl=_LEASE_TTL,
                        renew_every=_LEASE_RENEW_EVERY,
                        name="the KV marker purge lease",
                    )
                except LeaseUnavailable:
                    _PASSES.labels(bucket=bucket, outcome="skipped").inc()
                    # this replica found nothing this hour, so a max across replicas reads the holder's pass, not
                    # an hours-old one of a replica that held the lease before
                    _OLDEST_AGE.labels(bucket=bucket).set(0.0)
                    return MarkerPurgePass(purged=0, read=0, oldest_marker_age=None, behind=False, skipped=True)
                async with held:
                    done = await self._walk(apply=apply, lost=held.lost.is_set)
        except Exception:
            _PASSES.labels(bucket=bucket, outcome="failed").inc()
            raise
        seconds = time.monotonic() - started
        outcome = "lost" if done.lost else "behind" if done.behind else "done"
        _PASSES.labels(bucket=bucket, outcome=outcome).inc()
        _PASS_SECONDS.labels(bucket=bucket).observe(seconds)
        if apply:
            _PURGED.labels(bucket=bucket).inc(done.purged)
        age = done.oldest_marker_age
        _OLDEST_AGE.labels(bucket=bucket).set(0.0 if age is None else age.total_seconds())
        log.info(
            "KV marker purge pass %s",
            outcome,
            extra={
                "extra_data": {
                    "bucket": bucket,
                    "purged" if apply else "found": done.purged,
                    "read": done.read,
                    "oldest_marker_age_seconds": None if age is None else round(age.total_seconds()),
                    "seconds": round(seconds, 3),
                    "next_sequence": self._next_sequence,
                }
            },
        )
        return MarkerPurgePass(
            purged=done.purged,
            read=done.read,
            oldest_marker_age=age,
            behind=done.behind,
            lost=done.lost,
            seconds=seconds,
        )

    async def _walk(self, *, apply: bool, lost: Callable[[], bool]) -> MarkerPurgePass:
        """walk pages from the remembered sequence until caught up, out of budget, or the lease is lost.

        :param apply: purge; ``False`` only counts
        :ptype apply: bool
        :param lost: whether the lease was lost
        :ptype lost: Callable[[], bool]
        :return: what the walk did (its ``seconds`` unset)
        :rtype: MarkerPurgePass
        """
        created = await self._bucket.date_created()
        if created != self._created:
            # a stream created again starts its sequences over: a position in the old one means nothing here
            self._created = created
            self._next_sequence = 1
        now = self._clock()
        before = now - self._age
        # a dry run purges nothing, so its position is its own: a later pass reads the same markers again
        sequence = self._next_sequence
        purged = read = 0
        oldest: datetime | None = None
        while True:
            if lost():
                return MarkerPurgePass(purged, read, _age_of(oldest, now), behind=True, lost=True)
            budget = min(self._page, self._max_messages - read)
            if budget < 1:
                return MarkerPurgePass(purged, read, _age_of(oldest, now), behind=True)
            page = await self._bucket.purge_aged_removal_markers(
                before=before, from_sequence=sequence, max_messages=budget, apply=apply
            )
            sequence = page.next_sequence
            if apply:
                self._next_sequence = sequence
            purged += page.purged
            read += page.read
            if oldest is None:
                oldest = page.oldest_marker
            if page.done:
                return MarkerPurgePass(purged, read, _age_of(oldest, now), behind=False)


def _age_of(stored: datetime | None, now: datetime) -> timedelta | None:
    """how old a marker stored at ``stored`` is at ``now``.

    :param stored: when it was stored, or ``None`` for no marker
    :ptype stored: datetime | None
    :param now: the time of the pass
    :ptype now: datetime
    :return: its age, or ``None``
    :rtype: timedelta | None
    """
    return None if stored is None else now - stored
