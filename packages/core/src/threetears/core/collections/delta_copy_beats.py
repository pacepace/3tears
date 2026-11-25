"""each pod copy's position, published for the writer's fold: which delta files no copy can still need.

**Why** (``design-pod-delta-files.md`` decision 54). A fold that deletes its delta files (``retention="delete"``) once
deleted each file the moment a new full covered it, with no input about copies: on the 2026-11-24 replay copies lagging
their writer by a median 7.7 s found 536 files gone, each a lost file, and 91 whole-table reloads followed. So each copy
says how far it has applied each table, and the fold deletes a covered file only once every live copy is past it.

**The first rule: when in doubt, keep** (Pace, 2026-11-24: "really try to make sure we don't delete the deltas until all
the pods don't need it"). Every doubt keeps every file that round, never a guess:

- a copy registers a ``starting`` beat before it does anything else (:meth:`CopyBeats.register`): its position is
  unknown, so the fold deletes nothing while any copy is starting;
- a beat that does not decode, that lacks a table the fold deletes for, or whose epoch is of another incarnation than
  the file's, and a read of the beats that fails, each delete nothing that round (:meth:`CopyPositions.judge`);
- a copy counts as gone only once its beat is removed, and only by the same confirmed path that clears a dead writer's
  (:meth:`CopyBeats.recover_pass`: under a lease, a beat not renewed for the writers' stale time, deleted at the
  revision read, so a beat renewed meanwhile is never taken); no reader judges a beat by its age;
- the fold reads the beats again immediately before each delete, never once a round
  (:meth:`~threetears.core.collections.delta_fold.DeltaFolds`);
- a delete is idempotent (a file already gone is no error) and logged per file with the beats that allowed it.

**A beat** is one key a copy in the pod's own key scope of the platform's ``leases`` bucket,
``<pod scope>.delta-copies.<copy>``: the newest epoch the copy has applied per table (its marks), the moment it beat,
and whether it is still starting. A copy puts it at every beat interval (so a live copy's key is always fresh) and
whenever its marks change between beats; it deletes its key on a clean stop. A copy that lags only keeps files longer:
a beat behind the copy's true position never lets a file go early.

**No new grant.** The beats live in the pod's own key scope of the ``leases`` bucket, which every replica already
reads and writes through its lease factory (``KVLease.stored_key``), the same scope its write lease is held in: the
fold (the writer replica) reads its own pod's copies, never another pod's.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Final, Literal

from threetears.core.collections.generation import split_generation_token
from threetears.core.coordination.lease import KVLease, LeaseUnavailable
from threetears.nats.errors import NatsClientError
from threetears.observe import get_logger
from threetears.observe.metrics import counter, gauge

__all__ = ["COPY_BEAT_SCOPE", "CopyBeat", "CopyBeats", "CopyPositions", "CopyPositionsReader"]

log = get_logger(__name__)

#: the key segment, under the pod's own key scope, every copy's beat is kept under
COPY_BEAT_SCOPE: Final = "delta-copies"

#: how often a copy renews its beat: the writers' own interval (``lost_writes._BEAT_SECONDS``)
_BEAT_SECONDS: Final = 5.0

#: a beat not renewed for this long is a gone copy's, removed only by the recovery lease's holder: the writers' own stale
#: time (``lost_writes._STALE_SECONDS``), six beats missed, never a slow one (a beat is one KV put)
_STALE_SECONDS: Final = 30.0

#: the lease one replica holds for a pass of removing gone copies' beats, and how long a dead holder's hold lasts
_RECOVERY_LEASE: Final = "delta-copies-recovery"
_RECOVERY_LEASE_TTL_SECONDS: Final = 15

#: why the fold deletes nothing this round, or keeps a file (:meth:`CopyPositions.judge`)
Verdict = Literal["delete", "behind", "starting", "unreadable", "read_failed", "missing_table", "other_incarnation"]

_BEATS_READ = counter(
    "delta_copy_beats_reads_total",
    description="reads of the pod copies' beats by the writer's fold before a delete, by outcome: read, unreadable (a "
    "beat that does not decode: nothing deleted that round), read_failed (the bucket's read raised: nothing deleted)",
    label_names=("outcome",),
)
_GONE = counter(
    "delta_copy_beats_removed_total",
    description="gone copies' beats removed by the recovery lease's holder: each not renewed for the writers' stale time",
)
_LIVE = gauge(
    "delta_copy_beats_live",
    description="copy beats the recovery lease's holder read on its last pass, live (renewed within the stale time)",
)


@dataclass(frozen=True, slots=True)
class CopyBeat:
    """one copy's beat.

    :ivar copy: the copy's id, one NATS token
    :ivar epochs: each table's newest epoch the copy has applied (a generation token); a table it holds no mark of is
        absent, which keeps every file of it
    :ivar starting: the copy has registered and not yet loaded and followed: its position is unknown
    :ivar beat_at: when it beat, by its clock
    """

    copy: str
    epochs: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    starting: bool = False
    beat_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def encode(self) -> bytes:
        """the beat as its KV value.

        :return: the bytes
        :rtype: bytes
        """
        return json.dumps(
            {
                "copy": self.copy,
                "epochs": dict(sorted(self.epochs.items())),
                "starting": self.starting,
                "beat_at": self.beat_at.isoformat(),
            }
        ).encode()

    @classmethod
    def decode(cls, raw: bytes) -> CopyBeat:
        """a beat from its KV value.

        :param raw: the value
        :ptype raw: bytes
        :return: the beat
        :rtype: CopyBeat
        :raises ValueError: when the value is not a beat
        """
        try:
            body = json.loads(raw)
            epochs = {str(table): str(epoch) for table, epoch in dict(body["epochs"]).items()}
            beat_at = datetime.fromisoformat(str(body["beat_at"]))
            return cls(
                copy=str(body["copy"]),
                epochs=MappingProxyType(epochs),
                starting=bool(body["starting"]),
                beat_at=beat_at,
            )
        except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
            raise ValueError(f"not a copy beat: {raw[:200]!r}") from exc


@dataclass(frozen=True, slots=True)
class CopyPositions:
    """the copies' beats as one read found them, or why the read cannot be trusted.

    :ivar beats: each live copy's beat
    :ivar refused: why no delete may follow this read (``unreadable``, ``read_failed``); None when every beat read
    """

    beats: tuple[CopyBeat, ...] = ()
    refused: Verdict | None = None

    def judge(self, table: str, epoch: str) -> Verdict:
        """whether a delta file of ``table`` whose change is at ``epoch`` (or before it) may be deleted: ``delete`` once
        every live copy has applied it; ``behind`` keeps this file (a copy lags it); any other word keeps every file
        this round (module docstring, "when in doubt, keep").

        :param table: the table
        :ptype table: str
        :param epoch: the file's epoch, or a version at or after it (a generation token)
        :ptype epoch: str
        :return: the verdict
        :rtype: str
        """
        if self.refused is not None:
            return self.refused
        wanted = split_generation_token(epoch)
        if wanted is None:
            return "other_incarnation"
        behind = False
        for beat in self.beats:
            if beat.starting:
                return "starting"
            held = beat.epochs.get(table)
            if held is None:
                return "missing_table"
            at = split_generation_token(held)
            if at is None or at[0] != wanted[0]:
                return "other_incarnation"
            behind = behind or at[1] < wanted[1]
        return "behind" if behind else "delete"

    def said(self, table: str) -> list[dict[str, object]]:
        """the beats as a log line names them, for ``table``.

        :param table: the table
        :ptype table: str
        :return: each copy, its epoch of the table and when it beat
        :rtype: list[dict[str, object]]
        """
        return [
            {
                "copy": beat.copy,
                "epoch": beat.epochs.get(table),
                "starting": beat.starting,
                "at": beat.beat_at.isoformat(),
            }
            for beat in self.beats
        ]


class CopyPositionsReader:
    """what the writer's fold reads the copies' positions through (:class:`CopyBeats` is one)."""

    async def positions(self) -> CopyPositions:  # pragma: no cover - a protocol's body
        """the copies' beats now.

        :return: them, or why they cannot be trusted
        :rtype: CopyPositions
        """
        raise NotImplementedError


class CopyBeats(CopyPositionsReader):
    """a pod's copy beats in its own key scope of the ``leases`` bucket: a copy's beat (:meth:`register`,
    :meth:`beat`, :meth:`stop`), the fold's read (:meth:`positions`), and the removal of gone copies' beats under a
    lease (:meth:`recover_pass`).

    :param lease: the pod's lease factory over the platform's ``leases`` bucket, under the pod's own key scope
    :ptype lease: KVLease
    :param copy_id: this replica's copy id, one NATS token; None for a reader only (the fold, the recovery)
    :ptype copy_id: str | None
    :param stale_seconds: how long a beat goes unrenewed before the recovery removes it
    :ptype stale_seconds: float
    :param clock: wall-clock seconds since the epoch
    :ptype clock: Callable[[], float]
    :raises ValueError: when the copy id is not one token, or the stale time is not past a beat interval
    """

    def __init__(
        self,
        lease: KVLease,
        *,
        copy_id: str | None = None,
        stale_seconds: float = _STALE_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if copy_id is not None and not copy_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"a copy id is one NATS token: {copy_id!r}")
        if stale_seconds <= _BEAT_SECONDS:
            raise ValueError("a copy is stale only after more than one beat was missed")
        self._lease = lease
        self._copy = copy_id
        self._stale_seconds = stale_seconds
        self._clock = clock
        #: the marks the last beat put, replaced whole: a beat between intervals is put only when they changed
        self._last: Mapping[str, str] | None = None

    @property
    def beat_seconds(self) -> float:
        """how often a copy renews its beat.

        :return: seconds
        :rtype: float
        """
        return _BEAT_SECONDS

    def _key(self, copy: str) -> str:
        return self._lease.stored_key(f"{COPY_BEAT_SCOPE}.{copy}")

    def _own(self) -> str:
        if self._copy is None:
            raise ValueError("a reader of the copies' beats has no beat of its own")
        return self._copy

    def _now(self) -> datetime:
        return datetime.fromtimestamp(self._clock(), tz=UTC)

    async def register(self) -> None:
        """put this copy's ``starting`` beat, before it loads or follows anything: until its first :meth:`beat` its
        position is unknown, and the fold deletes nothing (module docstring).

        :return: nothing
        :rtype: None
        """
        copy = self._own()
        bucket = await self._lease.bucket()
        await bucket.put(key=self._key(copy), value=CopyBeat(copy=copy, starting=True, beat_at=self._now()).encode())
        self._last = None

    async def beat(self, epochs: Mapping[str, str | None], *, renew: bool = True) -> bool:
        """put this copy's beat: the newest epoch it has applied per table (a table with no mark is left out, which
        keeps every file of it). ``renew`` puts it whatever changed (the interval's beat, keeping the key fresh); else
        only when the marks changed since the last put.

        :param epochs: each table's mark
        :ptype epochs: Mapping[str, str | None]
        :param renew: put even when nothing changed
        :ptype renew: bool
        :return: whether it was put
        :rtype: bool
        """
        copy = self._own()
        held = MappingProxyType({table: epoch for table, epoch in sorted(epochs.items()) if epoch is not None})
        if not renew and self._last is not None and dict(self._last) == dict(held):
            return False
        bucket = await self._lease.bucket()
        await bucket.put(key=self._key(copy), value=CopyBeat(copy=copy, epochs=held, beat_at=self._now()).encode())
        self._last = held
        return True

    async def stop(self) -> None:
        """delete this copy's beat on a clean stop: it needs no file any more.

        :return: nothing
        :rtype: None
        """
        bucket = await self._lease.bucket()
        await bucket.delete(key=self._key(self._own()))
        self._last = None

    async def positions(self) -> CopyPositions:
        """every copy's beat now, read by key; a beat that does not decode, or a read that fails, refuses every delete
        that round (module docstring). Never judges a beat by its age: a key kept is a copy not yet gone.

        :return: the beats, or why they cannot be trusted
        :rtype: CopyPositions
        """
        try:
            bucket = await self._lease.bucket()
            keys = await bucket.list_keys(prefix=self._lease.stored_key(f"{COPY_BEAT_SCOPE}."))
            beats: list[CopyBeat] = []
            for key in sorted(keys):
                entry = await bucket.get_entry(key=key)
                if entry is None:
                    continue  # removed between the listing and the read: a copy gone meanwhile
                try:
                    beats.append(CopyBeat.decode(entry[0]))
                except ValueError as exc:
                    _BEATS_READ.labels(outcome="unreadable").inc()
                    log.warning(
                        "a copy beat does not decode: the fold deletes no delta file this round",
                        extra={"extra_data": {"key": key, "error": str(exc)}},
                    )
                    return CopyPositions(refused="unreadable")
        except (NatsClientError, OSError, TimeoutError) as exc:
            # a failed read keeps every file this round, said; the next delete reads again
            _BEATS_READ.labels(outcome="read_failed").inc()
            log.warning(
                "the copies' beats could not be read: the fold deletes no delta file this round",
                extra={"extra_data": {"error": repr(exc)}},
            )
            return CopyPositions(refused="read_failed")
        _BEATS_READ.labels(outcome="read").inc()
        return CopyPositions(beats=tuple(beats))

    async def recover_pass(self) -> tuple[str, ...]:
        """under the recovery lease, remove each copy's beat not renewed for the stale time, at the revision read (a
        beat renewed meanwhile is kept): the same confirmed path that clears a dead writer's beat
        (``lost_writes.WriteBackstop.recover_pass``). Run by every replica; one holds the lease a pass.

        :return: the copies whose beats were removed
        :rtype: tuple[str, ...]
        """
        try:
            handle = await self._lease.acquire(
                _RECOVERY_LEASE, ttl_seconds=_RECOVERY_LEASE_TTL_SECONDS, max_wait_seconds=0
            )
        except LeaseUnavailable:
            return ()
        try:
            bucket = await self._lease.bucket()
            now = self._now()
            removed: list[str] = []
            live = 0
            for key in sorted(await bucket.list_keys(prefix=self._lease.stored_key(f"{COPY_BEAT_SCOPE}."))):
                entry = await bucket.get_entry(key=key)
                if entry is None:
                    continue
                raw, revision = entry
                try:
                    beat = CopyBeat.decode(raw)
                except ValueError:
                    # never removed: an undecodable beat keeps every file until a person looks (module docstring)
                    log.warning("a copy beat does not decode; left as it is", extra={"extra_data": {"key": key}})
                    continue
                if (now - beat.beat_at).total_seconds() <= self._stale_seconds:
                    live += 1
                    continue
                await bucket.delete(key=key, revision=revision)
                removed.append(beat.copy)
                _GONE.inc()
                log.warning(
                    "a gone copy's beat was removed: the fold no longer keeps delta files for it",
                    extra={
                        "extra_data": {
                            "copy": beat.copy,
                            "beat_at": beat.beat_at.isoformat(),
                            "epochs": dict(beat.epochs),
                        }
                    },
                )
            _LIVE.set(live)
            return tuple(removed)
        finally:
            await handle.release()
