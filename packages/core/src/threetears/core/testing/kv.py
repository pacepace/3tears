"""in-memory fake of :class:`threetears.nats.NatsKvBucket` + :class:`~threetears.nats.NatsClient`.

published rather than kept per-repo: every consumer that touches KV was
writing its own double of this same narrow surface, and a double that
drifts from the wrapper is how a KV bug ships green.

**every operation yields to the event loop before it touches state.** a
double whose methods never await cannot interleave, so `asyncio.gather`
runs each call to completion in turn and a read-then-write store passes
the concurrency test a real broker would fail. that is not hypothetical:
it is how a non-atomic redemption and a dropped revision guard both
survived a full suite. mirrors the wrapper surface consumers actually
use:

- :meth:`FakeKvBucket.create` returns the new revision on success or
  ``None`` on CAS conflict (key already present).
- :meth:`FakeKvBucket.get` returns ``bytes | None``.
- :meth:`FakeKvBucket.get_entry` returns ``(bytes, revision) | None``.
- :meth:`FakeKvBucket.get_latest` returns ``(bytes | None, revision)``: the
  key's latest message, a deletion marker included, with revision ``0`` for
  a key that has none.
- :meth:`FakeKvBucket.update` returns the new revision on success or
  ``None`` on CAS conflict: it lands only when the key's latest message --
  live value, deletion marker, or none at all for revision ``0`` -- is at the
  expected revision, as the server's expected-last-subject-sequence does.
- :meth:`FakeKvBucket.delete` leaves a marker with its own revision, as a
  real delete publishes one; :meth:`FakeKvBucket.create` lands over it.
- :meth:`FakeKvBucket.delete` accepts an optional ``revision`` and
  returns ``True`` on success or absent key, ``False`` on CAS mismatch.
- :meth:`FakeKvBucket.date_created` reports when the bucket was created, and
  :meth:`FakeKvBucket.wipe` empties it, moves that time forward and restarts
  its revisions at 1, which is what a broker restart does to a memory-backed
  bucket once something has recreated it; :meth:`FakeKvBucket.vanish` leaves
  it absent until the next operation recreates it, as the real wrapper's
  self-heal does -- through a handle that may create it. Through a BIND-ONLY
  handle (a bucket named in ``declared_buckets``, or bound with
  ``ensure_kv_bucket(create_if_missing=False)``) the operation raises
  :class:`threetears.nats.KvBucketNotFoundError` instead and the bucket stays
  absent, as the real handle does once its wait for the declarer is spent.
- a bind-only open of an absent bucket (``kv_bucket`` / ``ensure_kv_bucket``
  with ``create_if_missing=False``) raises
  :class:`threetears.nats.KvBucketNotFoundError`, which is a ``KvError``, as
  the real client does.
- :meth:`FakeKvBucket.become_unreachable` makes every operation raise a given
  error, before it touches state, until :meth:`FakeKvBucket.become_reachable`:
  an outage that loses nothing, as distinct from a vanished bucket.
- :meth:`FakeKvBucket.watch_key` yields the key's latest message -- a value,
  or a deletion marker with ``value=None`` -- then every later write or
  delete of that key, as :meth:`threetears.nats.NatsKvBucket.watch_key`
  does. It yields :class:`threetears.nats.kv_watch.KvKeyUpdate`, the same
  type, so a consumer's test runs the code path production runs.
- :meth:`FakeNatsClient.add_reconnect_callback` registers a hook and
  :meth:`FakeNatsClient.reconnect` runs every hook, as a real reconnect does.
- :meth:`FakeNatsClient.ensure_kv_bucket` declares as the real client does, and remembers a
  declaration that may create, on memory or file storage; :meth:`FakeNatsClient.restart_broker`
  loses every bucket, puts back only the remembered ones, then runs the reconnect hooks.

the fake stores data in a plain dict keyed by bucket name so multiple
buckets created from the same client share no state. revision counter
is bucket-local and monotonic per incarnation: a wiped or vanished bucket
starts again at 1, as a recreated stream's sequence does.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Iterable
from dataclasses import dataclass
from typing import Any

from threetears.nats.errors import KvBucketNotFoundError, KvConfigMismatch
from threetears.nats.kv_watch import DEFAULT_KEY_WATCH_HEARTBEAT, DEFAULT_KEY_WATCH_RETRY, KvKeyUpdate
from threetears.observe import get_logger

__all__ = ["FakeKvBucket", "FakeNatsClient"]

log = get_logger(__name__)


class _YieldOnce:
    """suspend the coroutine once, handing control back to the event loop.

    deliberately NOT ``asyncio.sleep(0)``: several suites spy on ``asyncio.sleep`` to assert a
    code path does not back off, and a double that slept would be counted as the code under
    test sleeping.
    """

    def __await__(self) -> Generator[None, None, None]:
        yield


@dataclass
class _Entry:
    """internal storage entry.

    :ivar expires_at: the bucket-clock time the server would remove a per-entry-TTL entry, or
        ``None`` for an entry that lives as long as the bucket
    """

    value: bytes
    revision: int
    expires_at: timedelta | None = None


# parity-with: threetears.nats.kv.KvBucketLike
class FakeKvBucket:
    """in-memory fake mirroring :class:`threetears.nats.NatsKvBucket`.

    methods take kw-only args matching the wrapper surface so test
    fixtures exercise the same call shape production code uses.
    """

    def __init__(
        self,
        bucket_name: str,
        ttl: timedelta | None = None,
        storage: str = "memory",
        direct: bool | None = None,
        may_create: bool = True,
    ) -> None:
        """initialize empty fake bucket with zero revision counter.

        :param bucket_name: full bucket name (with namespace prefix)
        :ptype bucket_name: str
        :param ttl: the bucket TTL this bucket was opened with. Recorded and
            reported by :attr:`ttl` rather than applied -- the fake does not
            expire entries. It is carried because production code reads it back
            (``nats_distributed_lock`` compares the bucket's TTL against the one
            it asked for), and a double that cannot answer a question the real
            bucket answers is a double that hides the call.
        :ptype ttl: timedelta | None
        :param storage: the backing this bucket was opened with. Recorded and reported by
            :attr:`storage` for the same reason as ``ttl``: whether a caller asked for
            ``memory`` or ``file`` is a real difference in a clustered broker -- memory
            state is not reliably readable by another replica -- and a double that cannot
            answer it hides the choice
        :ptype storage: str
        :param direct: whether the bucket was opened with direct gets. Recorded and reported
            by :attr:`direct` for the same reason: a pod granted ``DIRECT.GET`` on one key
            and not ``STREAM.MSG.GET`` blocks on every read of a bucket opened without them
        :ptype direct: bool | None
        :param may_create: whether operations through this handle recreate the bucket once it has
            vanished, as a real handle opened with ``create_if_missing=True`` does. ``False`` is a
            bind-only handle, whose operations on a vanished bucket raise
            :class:`threetears.nats.KvBucketNotFoundError` instead; see :meth:`set_may_create`
        :ptype may_create: bool
        :return: None
        :rtype: None
        """
        self._bucket_name = bucket_name
        self._may_create = may_create
        self._ttl = ttl
        self._storage = storage
        self._direct = direct
        self._entries: dict[str, _Entry] = {}
        # the revision of each deleted key's marker: its latest message once its value is gone.
        self._markers: dict[str, int] = {}
        self._revision = 0
        self._date_created = datetime.now(UTC)
        # set by vanish(): the stream is gone until the next operation recreates it.
        self._vanished = False
        # set by become_unreachable(): every operation raises it until become_reachable().
        self._unreachable_error: Exception | None = None
        # a clock only this bucket reads, moved by advance_clock, so a test can make a per-entry
        # TTL lapse without sleeping.
        self._elapsed = timedelta(0)
        # one queue per open watch_key iterator, keyed by the key it watches.
        self._key_watchers: dict[str, list[asyncio.Queue[KvKeyUpdate]]] = {}

    async def _arrive(self) -> None:
        """what every operation does first: yield to the loop, fail if unreachable, heal if vanished.

        The yield is so ``gather()`` genuinely interleaves. An unreachable bucket raises before it
        touches any state, so nothing lands and nothing heals while it is down. The heal mirrors
        the real wrapper, which recreates a vanished stream on the next operation through a handle
        that may create it, so the recreated bucket's creation time is the moment of that
        operation. A bind-only handle may not: the real one re-binds, finds nothing, and raises
        once its wait for the declarer is spent, leaving the bucket absent.

        :return: None
        :rtype: None
        :raises Exception: the error :meth:`become_unreachable` was given, while it is set
        :raises KvBucketNotFoundError: the bucket has vanished and this handle is bind-only
        """
        await _YieldOnce()
        if self._unreachable_error is not None:
            raise self._unreachable_error
        if self._vanished and not self._may_create:
            raise KvBucketNotFoundError(
                f"KV bucket {self._bucket_name!r} does not exist, and this handle only binds it: its "
                f"declarer must create it again",
                bucket=self._bucket_name,
            )
        if self._vanished:
            self._vanished = False
            self._date_created = datetime.now(UTC)
            self._revision = 0
            self._markers.clear()

    def advance_clock(self, delta: timedelta) -> None:
        """move this bucket's clock forward, lapsing any per-entry TTL it passes.

        :param delta: how far to move the clock; must not be negative
        :ptype delta: timedelta
        :return: None
        :rtype: None
        :raises ValueError: when ``delta`` is negative
        """
        if delta < timedelta(0):
            raise ValueError("FakeKvBucket.advance_clock cannot move the clock backwards")
        self._elapsed += delta

    def _live(self, key: str) -> _Entry | None:
        """the entry under ``key``, or ``None`` once its per-entry TTL has lapsed.

        :param key: key to look up
        :ptype key: str
        :return: the live entry, or ``None``
        :rtype: _Entry | None
        """
        entry = self._entries.get(key)
        if entry is not None and entry.expires_at is not None and self._elapsed >= entry.expires_at:
            # the server removes a lapsed entry's message, leaving no marker behind.
            del self._entries[key]
            entry = None
        return entry

    def _latest_revision(self, key: str) -> int:
        """the revision of the key's latest message: its live value, its deletion marker, or 0.

        :param key: key to look up
        :ptype key: str
        :return: the revision an expected-last-subject-sequence write must name
        :rtype: int
        """
        entry = self._live(key)
        return entry.revision if entry is not None else self._markers.get(key, 0)

    def _write(self, key: str, value: bytes, ttl: timedelta | None) -> int:
        """append one message for ``key`` and return its revision.

        :param key: key to write
        :ptype key: str
        :param value: bytes payload
        :ptype value: bytes
        :param ttl: per-entry lifetime, or ``None``
        :ptype ttl: timedelta | None
        :return: the new revision
        :rtype: int
        """
        expires_at = self._expiry(ttl)
        self._revision += 1
        self._entries[key] = _Entry(value=value, revision=self._revision, expires_at=expires_at)
        self._markers.pop(key, None)
        self._notify(KvKeyUpdate(key=key, value=value, revision=self._revision))
        return self._revision

    def _mark_deleted(self, key: str) -> None:
        """append the deletion marker a real delete publishes, the key's latest message from now on.

        :param key: the deleted key
        :ptype key: str
        :return: None
        :rtype: None
        """
        self._revision += 1
        self._markers[key] = self._revision
        self._notify(KvKeyUpdate(key=key, value=None, revision=self._revision))

    def _notify(self, update: KvKeyUpdate) -> None:
        """hand one new message to every open watch of its key.

        :param update: the message just appended
        :ptype update: KvKeyUpdate
        :return: None
        :rtype: None
        """
        for queue in self._key_watchers.get(update.key, ()):
            queue.put_nowait(update)

    def _latest_message(self, key: str) -> KvKeyUpdate | None:
        """the key's latest message as a watch first delivers it, or ``None`` when it has none.

        :param key: key to look up
        :ptype key: str
        :return: the live value, the deletion marker, or ``None``
        :rtype: KvKeyUpdate | None
        """
        entry = self._live(key)
        marker = self._markers.get(key)
        latest: KvKeyUpdate | None = None
        if entry is not None:
            latest = KvKeyUpdate(key=key, value=entry.value, revision=entry.revision)
        elif marker is not None:
            latest = KvKeyUpdate(key=key, value=None, revision=marker)
        return latest

    async def watch_key(
        self,
        *,
        key: str,
        heartbeat: timedelta = DEFAULT_KEY_WATCH_HEARTBEAT,
        retry: timedelta = DEFAULT_KEY_WATCH_RETRY,
    ) -> AsyncGenerator[KvKeyUpdate]:
        """the key's latest message, then every later write or delete of it, until iteration stops.

        Mirrors :meth:`threetears.nats.NatsKvBucket.watch_key`: a delete arrives with
        ``value=None``, and a key with no message yields nothing until one is written. There is no
        consumer to lose in memory, so ``heartbeat`` and ``retry`` are accepted and unused.

        :param key: the key to watch; one literal key, never a wildcard
        :ptype key: str
        :param heartbeat: accepted for signature parity; unused
        :ptype heartbeat: timedelta
        :param retry: accepted for signature parity; unused
        :ptype retry: timedelta
        :return: the key's messages, in order
        :rtype: AsyncGenerator[KvKeyUpdate]
        :raises ValueError: when ``key`` is empty or not literal, as the real watch refuses it
        """
        del heartbeat, retry
        if not key or any(char in "*> \t\r\n" for char in key):
            raise ValueError(f"watch_key needs one literal key, got {key!r}")
        await self._arrive()
        queue: asyncio.Queue[KvKeyUpdate] = asyncio.Queue()
        # registered and read with no await between, so no write can fall between the two.
        self._key_watchers.setdefault(key, []).append(queue)
        try:
            latest = self._latest_message(key)
            if latest is not None:
                yield latest
            while True:
                yield await queue.get()
        finally:
            watchers = self._key_watchers.get(key, [])
            if queue in watchers:
                watchers.remove(queue)
            if not watchers:
                self._key_watchers.pop(key, None)

    def _expiry(self, ttl: timedelta | None) -> timedelta | None:
        """the bucket-clock removal time for an entry written now with ``ttl``.

        :param ttl: the per-entry lifetime, or ``None``
        :ptype ttl: timedelta | None
        :return: the removal time, or ``None``
        :rtype: timedelta | None
        :raises ValueError: when ``ttl`` is under one second, as the real wrapper refuses
        """
        if ttl is None:
            return None
        if ttl < timedelta(seconds=1):
            raise ValueError(f"a per-entry KV TTL must be at least one second, got {ttl}")
        return self._elapsed + ttl

    async def date_created(self) -> datetime:
        """when this bucket was created, or last wiped.

        :return: timezone-aware UTC creation time
        :rtype: datetime
        """
        await self._arrive()
        return self._date_created

    def keys(self) -> tuple[str, ...]:
        """every live key in the bucket, for a test asserting on what was stored.

        Public because the alternative is reaching into the fake's entries, which the underscore
        contract forbids across classes and which every consumer was otherwise doing.

        :return: the live keys, in insertion order
        :rtype: tuple[str, ...]
        """
        return tuple(key for key in tuple(self._entries) if self._live(key) is not None)

    async def list_keys(self, *, prefix: str = "") -> list[str]:
        """every live key starting with ``prefix``, as :meth:`threetears.nats.kv.NatsKvBucket.list_keys`.

        :param prefix: keep keys starting with this; ``""`` lists every key
        :ptype prefix: str
        :return: the live keys, in insertion order
        :rtype: list[str]
        :raises ValueError: when ``prefix`` carries a wildcard or whitespace, as the real one does
        """
        if any(char in prefix for char in ("*", ">", " ", "\t", "\r", "\n")):
            raise ValueError(f"list_keys needs a literal prefix, got {prefix!r}")
        await self._arrive()
        return [key for key in self.keys() if key.startswith(prefix)]

    def wipe(self, *, date_created: datetime | None = None) -> None:
        """empty the bucket, give it a new creation time and restart its revisions, as a broker restart does.

        Every handle a test holds keeps working afterwards and silently sees the empty
        bucket -- the same property the real wrapper has, and the one a wipe-detecting
        caller exists to handle.

        :param date_created: the new creation time; ``None`` uses now. Must be timezone-aware.
        :ptype date_created: datetime | None
        :return: None
        :rtype: None
        :raises ValueError: when ``date_created`` is timezone-naive
        """
        if date_created is not None and date_created.tzinfo is None:
            raise ValueError("FakeKvBucket.wipe requires a timezone-aware date_created")
        self._entries.clear()
        self._markers.clear()
        self._date_created = date_created if date_created is not None else datetime.now(UTC)
        self._vanished = False
        # the revision is the stream sequence, and a recreated stream starts it again.
        self._revision = 0

    def vanish(self) -> None:
        """lose the bucket the way a broker restart does, leaving it absent until next used.

        Where :meth:`wipe` models a bucket some other caller already recreated, this models the
        moment in between: the entries are gone, and the next operation through a handle that
        may create (:attr:`may_create`) recreates the bucket, taking that operation's moment as its
        creation time. That is what the real wrapper's self-heal does, and it is the difference
        between recreating a bucket when the broker comes back and recreating it whenever someone
        next happens to use it. Through a bind-only handle the operation raises
        :class:`threetears.nats.KvBucketNotFoundError` instead, until a declaration puts the bucket
        back, as the real bind-only handle does once its wait for the declarer is spent.

        :return: None
        :rtype: None
        """
        self._entries.clear()
        self._markers.clear()
        self._vanished = True

    @property
    def may_create(self) -> bool:
        """whether operations through this handle recreate the bucket once it has vanished.

        :return: ``True`` for a handle opened as the real ``create_if_missing=True`` one is
        :rtype: bool
        """
        return self._may_create

    def set_may_create(self, may_create: bool) -> None:
        """record how the client's one handle on this bucket is now opened.

        The real client caches one handle per bucket, and a declaration replaces it with one opened
        the way the declaration asked -- ``create_if_missing=True`` heals a vanished bucket, a bind
        does not. :class:`FakeNatsClient` calls this as its declarations do the same.

        :param may_create: ``True`` when the handle may recreate the bucket
        :ptype may_create: bool
        :return: None
        :rtype: None
        """
        self._may_create = may_create

    @property
    def is_vanished(self) -> bool:
        """whether the bucket is absent, lost to :meth:`vanish` and not yet recreated by an operation.

        :return: ``True`` while the bucket does not exist on the broker
        :rtype: bool
        """
        return self._vanished

    def reconcile(self, *, ttl: timedelta | None, direct: bool, storage: str | None = None) -> None:
        """take a declaration's reconciled fields, as a real declaration does.

        Entries are kept, except across a change of storage: JetStream cannot change a live stream's
        storage, so a declaration that owns its bucket deletes the stream and creates it again, empty.

        :param ttl: the declared bucket TTL; ``None`` means no expiry
        :ptype ttl: timedelta | None
        :param direct: the declared ``allow_direct`` value
        :ptype direct: bool
        :param storage: the declared storage, ``"memory"`` or ``"file"``; ``None`` keeps the live one
        :ptype storage: str | None
        :return: None
        :rtype: None
        """
        self._ttl = ttl
        self._direct = direct
        if storage is not None and storage != self._storage:
            self.wipe()
            self._storage = storage

    def become_unreachable(self, error: Exception) -> None:
        """make every operation raise ``error`` until :meth:`become_reachable`, as a lost broker does.

        Unlike :meth:`vanish`, nothing is lost: the bucket cannot be reached, so no write lands
        and no read answers, and when it is reachable again its entries and revisions are exactly
        as they were. This is the outage a coordination primitive must ride out -- a lease that
        gives up a claim over one failed renewal, or keeps one past its TTL because its renewals
        keep failing, is wrong in opposite directions, and only a bucket that fails on demand
        lets a test hold either.

        :param error: what each operation raises, as the real client would: a
            ``ConnectionError``, ``TimeoutError`` or the client's own error type
        :ptype error: Exception
        :return: None
        :rtype: None
        """
        self._unreachable_error = error

    def become_reachable(self) -> None:
        """end :meth:`become_unreachable`: operations reach the bucket again, its state intact.

        :return: None
        :rtype: None
        """
        self._unreachable_error = None

    @property
    def unreachable_error(self) -> Exception | None:
        """the error every operation raises while the bucket is unreachable; ``None`` when reachable.

        :return: the error, or ``None``
        :rtype: Exception | None
        """
        return self._unreachable_error

    @property
    def ttl(self) -> timedelta | None:
        """the TTL this bucket was opened with; ``None`` means no expiry.

        :return: the bucket TTL
        :rtype: timedelta | None
        """
        return self._ttl

    @property
    def storage(self) -> str:
        """the backing this bucket was opened with (``memory`` or ``file``).

        :return: storage name as the caller asked for it
        :rtype: str
        """
        return self._storage

    @property
    def direct(self) -> bool | None:
        """whether this bucket was opened with direct gets; ``None`` means the caller left the default.

        :return: the ``direct`` flag as the caller asked for it
        :rtype: bool | None
        """
        return self._direct

    @property
    def name(self) -> str:
        """fully-qualified bucket name.

        :return: bucket name
        :rtype: str
        """
        return self._bucket_name

    async def create(self, *, key: str, value: bytes, ttl: timedelta | None = None) -> int | None:
        """create-if-absent. returns new revision or ``None`` on conflict.

        :param key: key to insert
        :ptype key: str
        :param value: bytes payload
        :ptype value: bytes
        :param ttl: per-entry lifetime on this bucket's clock (see :meth:`advance_clock`), or ``None``
        :ptype ttl: timedelta | None
        :return: new revision number, or ``None`` if key already exists
        :rtype: int | None
        """
        await self._arrive()
        if self._live(key) is not None:
            return None
        return self._write(key, value, ttl)

    async def get(self, *, key: str) -> bytes | None:
        """get value bytes for key. returns ``None`` on miss.

        :param key: key to read
        :ptype key: str
        :return: stored bytes or ``None``
        :rtype: bytes | None
        """
        await self._arrive()
        entry = self._live(key)
        if entry is None:
            return None
        return entry.value

    async def get_entry(self, *, key: str) -> tuple[bytes, int] | None:
        """get value + revision tuple. returns ``None`` on miss.

        :param key: key to read
        :ptype key: str
        :return: ``(value, revision)`` tuple or ``None``
        :rtype: tuple[bytes, int] | None
        """
        await self._arrive()
        entry = self._live(key)
        if entry is None:
            return None
        return (entry.value, entry.revision)

    async def get_latest(self, *, key: str) -> tuple[bytes | None, int]:
        """the key's latest message: its value, and its revision even when it is a deletion.

        :param key: key to read
        :ptype key: str
        :return: ``(value, revision)`` for a live value, ``(None, marker revision)`` for a deleted
            key, ``(None, 0)`` for a key with no message
        :rtype: tuple[bytes | None, int]
        """
        await self._arrive()
        entry = self._live(key)
        if entry is None:
            return (None, self._markers.get(key, 0))
        return (entry.value, entry.revision)

    async def update(self, *, key: str, value: bytes, revision: int, ttl: timedelta | None = None) -> int | None:
        """CAS update. returns new revision or ``None`` on mismatch.

        :param key: key to update
        :ptype key: str
        :param value: new bytes payload
        :ptype value: bytes
        :param revision: expected current revision
        :ptype revision: int
        :param ttl: per-entry lifetime for the new entry on this bucket's clock, or ``None``
        :ptype ttl: timedelta | None
        :return: new revision, or ``None`` when the key's latest message is not at ``revision``
        :rtype: int | None
        """
        await self._arrive()
        if self._latest_revision(key) != revision:
            return None
        return self._write(key, value, ttl)

    async def delete(self, *, key: str, revision: int | None = None) -> bool:
        """delete a key, optionally guarded by a CAS revision.

        :param key: key to remove
        :ptype key: str
        :param revision: expected current revision; ``None`` skips CAS
        :ptype revision: int | None
        :return: ``True`` on success; ``False`` on CAS mismatch, INCLUDING when the key is
            already gone. an unguarded delete of an absent key is still ``True`` (idempotent).
        :rtype: bool
        """
        await self._arrive()
        entry = self._live(key)
        if entry is None:
            # A revision-guarded delete of a key that is no longer there LOST the race -- it
            # cannot have been the caller whose revision matched. Returning True here made
            # every concurrent redemption look like a winner, which is how a non-atomic claim
            # passed a concurrency test. An unguarded one still publishes a marker, as the real
            # server does: a delete is a message whether or not the key held anything.
            if revision is None:
                self._mark_deleted(key)
            return revision is None
        if revision is not None and entry.revision != revision:
            return False
        del self._entries[key]
        # a real delete publishes a marker, which is the key's latest message from now on.
        self._mark_deleted(key)
        return True

    async def put(self, *, key: str, value: bytes, ttl: timedelta | None = None) -> int:
        """unconditional write. returns new revision.

        :param key: key to write
        :ptype key: str
        :param value: bytes payload
        :ptype value: bytes
        :param ttl: a per-entry lifetime, honoured against this bucket's own clock (see
            :meth:`advance_clock`) exactly as :meth:`create` and :meth:`update` honour theirs
        :ptype ttl: timedelta | None
        :return: new revision number
        :rtype: int
        """
        await self._arrive()
        return self._write(key, value, ttl)


# parity-with: threetears.nats.kv.KvCapable
class FakeNatsClient:
    """fake NATS wrapper exposing :meth:`kv_bucket` returning :class:`FakeKvBucket`.

    matches the narrow surface KV consumers depend on. the bucket
    cache mirrors :class:`NatsClient`'s internal cache: repeat
    ``kv_bucket`` calls for the same name return the same instance.

    :meth:`publish` and :meth:`subscribe_typed` are here because every
    ``BaseCollection`` write publishes a cache invalidation: a fake with
    only ``kv_bucket`` cannot stand in for a collection's client at all,
    and each consumer was otherwise left to discover that and write its
    own. Published messages are delivered to whatever subscribed on the
    same subject through this instance and kept in
    :attr:`published`, so a test can assert on the broadcast or run a
    real listener against it.
    """

    def __init__(self, *, bucket_age: timedelta | None = None, declared_buckets: Iterable[str] = ()) -> None:
        """initialize with a bucket registry holding only what a declarer already created.

        :param bucket_age: how long ago every bucket this client creates reports having been
            created. ``None`` means now, which is the real client's behaviour for a fresh bucket.

            **What this exists for.** ``ReplayGuard`` refuses an artifact issued before its
            bucket was created plus the verifier's tolerance, because after a wipe it cannot
            rule out an earlier sighting. A fake bucket created at the instant the test mints
            its artifact is inside that window BY CONSTRUCTION, so every verifier test written
            the obvious way fails with a replay refusal that has nothing to do with replay. The
            same trap is documented for the first real call after a broker restart.

            A test about replay semantics should therefore ask for an ESTABLISHED bucket, and
            this is the one-line way to get one; a test about the watermark itself leaves this
            ``None`` and uses :meth:`FakeKvBucket.wipe` to place the creation time deliberately.
        :ptype bucket_age: timedelta | None
        :param declared_buckets: bucket names another identity has already declared, as the
            platform's hub declares every bucket a pod binds. A pod's primitives open bind-only
            (``create_if_missing=False``), and a bind of a bucket nobody declared raises
            :class:`threetears.nats.KvBucketNotFoundError` -- so a
            test of pod code over this fake names the buckets the hub would have declared,
            rather than every such test failing with an absent bucket that has nothing to do with
            what it tests. Each is created in the default shape (no TTL, memory storage)
        :ptype declared_buckets: Iterable[str]
        :return: None
        :rtype: None
        :raises ValueError: when ``bucket_age`` is negative
        """
        if bucket_age is not None and bucket_age < timedelta(0):
            raise ValueError(f"FakeNatsClient bucket_age must not be negative, got {bucket_age}")
        self._bucket_age = bucket_age
        self._buckets: dict[str, FakeKvBucket] = {}
        for name in declared_buckets:
            # declared by ANOTHER identity: this client only binds them, so it never recreates one
            self._buckets[name] = self._new_bucket(name=name, ttl=None, storage="memory", direct=None, may_create=False)
        self.published: list[Any] = []
        self._subscribers: dict[str, list[tuple[Any, Any]]] = {}
        self._reconnect_callbacks: list[Callable[[], Awaitable[None]]] = []
        # what the real client remembers: every bucket this client declared through
        # :meth:`ensure_kv_bucket` with a create, on memory or file storage, and so puts back after a
        # reconnect. names only -- the fake bucket carries its own config.
        self._remembered: set[str] = set()

    @property
    def remembered_declarations(self) -> frozenset[str]:
        """every bucket this client would create again after a reconnect, as the real client does.

        Exactly the buckets declared through :meth:`ensure_kv_bucket` with
        ``create_if_missing=True``, whatever their storage: a restart can lose file storage as well
        as memory (on Kubernetes the volume goes with the pod). An ordinary :meth:`kv_bucket` open
        and a bind-only declaration are never remembered, so a test can assert which form a
        declarer used rather than only that the bucket exists.

        :return: the remembered bucket names
        :rtype: frozenset[str]
        """
        return frozenset(self._remembered)

    def bucket_exists(self, name: str) -> bool:
        """whether the broker holds this bucket right now: created, and not lost to a restart since.

        :param name: bucket suffix, as passed to :meth:`kv_bucket`
        :ptype name: str
        :return: ``True`` when the bucket exists
        :rtype: bool
        """
        bucket = self._buckets.get(name)
        return bucket is not None and not bucket.is_vanished

    async def restart_broker(self) -> None:
        """lose every bucket to a broker restart, put back the remembered ones, then run the reconnect hooks.

        The real sequence on a restart that lost the broker's storage -- memory storage always, and
        file storage too when the volume went with the pod, as it can on Kubernetes, so the fake
        loses both: every bucket loses its entries; the client creates
        each declaration it remembers again, empty, before any hook it was given runs; every other
        bucket stays absent until an operation through a handle that may create it recreates it
        (the wrapper's self-heal, :meth:`FakeKvBucket.vanish`) -- through a bind-only handle the
        operation raises :class:`threetears.nats.KvBucketNotFoundError` until a declaration puts
        the bucket back. Entries are never put back -- republishing them is the declarer's job,
        and a test of that job runs it from a reconnect hook.

        :return: None
        :rtype: None
        """
        for name, bucket in self._buckets.items():
            if name in self._remembered:
                bucket.wipe()
            else:
                bucket.vanish()
        await self.reconnect()

    def add_reconnect_callback(self, callback: Callable[[], Awaitable[None]]) -> None:
        """register an async hook run after each reconnect, as the real client does.

        :param callback: an argument-less coroutine function
        :ptype callback: Callable[[], Awaitable[None]]
        :return: None
        :rtype: None
        """
        self._reconnect_callbacks.append(callback)

    async def reconnect(self) -> None:
        """run every reconnect hook in registration order, as the real client does after reconnecting.

        A hook that raises is logged and does not stop the others, matching the real dispatcher.

        :return: None
        :rtype: None
        """
        for callback in list(self._reconnect_callbacks):
            try:
                await callback()
            except Exception as exc:  # noqa: BLE001 -- one bad hook must not abort the others, as in the real client
                log.warning("reconnect callback failed: %s", exc)

    async def publish(self, *, subject: Any, message: Any, reply_to: Any = None) -> None:
        """record a message and deliver it to this client's subscribers on that subject.

        :param subject: the subject published to; stringified for the subscriber lookup
        :ptype subject: Any
        :param message: the typed envelope
        :ptype message: Any
        :param reply_to: ignored; present so the surface matches the real client
        :ptype reply_to: Any
        :return: None
        :rtype: None
        """
        del reply_to
        self.published.append(message)
        for callback, message_type in list(self._subscribers.get(str(subject), [])):
            await callback(message_type.model_validate_json(message.model_dump_json()))

    async def subscribe_typed(self, *, subject: Any, cb: Any, message_type: Any, **kwargs: Any) -> object:
        """register a typed subscriber, so a real listener can run against this fake.

        :param subject: the subject to subscribe to
        :ptype subject: Any
        :param cb: the async callback the listener supplies
        :ptype cb: Any
        :param message_type: the Pydantic envelope to validate into
        :ptype message_type: Any
        :param kwargs: ignored; the real client takes queue groups and durable names
        :ptype kwargs: Any
        :return: an opaque subscription handle
        :rtype: object
        """
        del kwargs
        entry = (cb, message_type)
        self._subscribers.setdefault(str(subject), []).append(entry)
        return (str(subject), entry)

    async def unsubscribe(self, subscription: Any) -> None:
        """drop the one subscription this handle names.

        Only that one: two L2-live registries in one process subscribe and stop independently,
        and a fake that cleared every subscriber could not express one listener stopping while
        another kept running -- so a test written against it passed or failed for reasons
        unrelated to the code under test.

        :param subscription: the handle :meth:`subscribe_typed` returned
        :ptype subscription: Any
        :return: None
        :rtype: None
        """
        if not isinstance(subscription, tuple) or len(subscription) != 2:
            return
        subject, entry = subscription
        entries = self._subscribers.get(subject)
        if entries is None or entry not in entries:
            return
        entries.remove(entry)
        if not entries:
            del self._subscribers[subject]

    async def kv_bucket(
        self,
        *,
        name: str,
        ttl: object | None = None,
        storage: str = "memory",
        create_if_missing: bool = True,
        history: int = 1,
        direct: bool | None = None,
    ) -> FakeKvBucket:
        """return existing bucket or create one. idempotent.

        :param name: bucket suffix; the fake skips the namespace
            prefix the real wrapper layers on top
        :ptype name: str
        :param ttl: recorded and reported by :attr:`FakeKvBucket.ttl`; not applied
        :ptype ttl: object | None
        :param storage: recorded and reported by :attr:`FakeKvBucket.storage`, so a test can
            witness which backing a caller asked for. Not otherwise applied -- the fake is
            always in-process
        :ptype storage: str
        :param create_if_missing: when ``False`` and bucket absent, raises
        :ptype create_if_missing: bool
        :param history: ignored by fake
        :ptype history: int
        :param direct: recorded and reported by :attr:`FakeKvBucket.direct`; not applied
        :ptype direct: bool | None
        :return: fake bucket; the cached handle when this client opened the bucket before, as the real
            client's cache returns it without asking the broker
        :rtype: FakeKvBucket
        :raises KvBucketNotFoundError: when ``create_if_missing=False`` and the bucket was never created
        """
        del history
        bucket = self._buckets.get(name)
        if bucket is None:
            if not create_if_missing:
                raise KvBucketNotFoundError(
                    f"KV bucket {name!r} does not exist, and this open only binds it", bucket=name
                )
            bucket = self._new_bucket(
                name=name, ttl=ttl if isinstance(ttl, timedelta) else None, storage=storage, direct=direct
            )
            self._buckets[name] = bucket
        return bucket

    async def ensure_kv_bucket(
        self,
        *,
        name: str,
        ttl: timedelta | None = None,
        storage: str = "memory",
        history: int = 1,
        direct: bool = True,
        create_if_missing: bool = True,
        owns_bucket: bool = False,
        drop_file_storage: bool = False,
    ) -> FakeKvBucket:
        """declare a bucket -- create it, or reconcile a live one in place -- or bind one somebody declared.

        Mirrors :meth:`threetears.nats.NatsClient.ensure_kv_bucket`: a declaration shares the one
        handle :meth:`kv_bucket` hands out, a declaration of a live bucket takes the declared
        ``direct`` with its entries kept -- and the declared TTL and storage only when its declarer
        owns the bucket (``owns_bucket``), as only then does the real one reconcile them, a changed
        storage emptying the bucket as the real recreate does, and a live FILE bucket refused unless
        ``drop_file_storage`` -- and a declaration that may create,
        whatever its storage, is remembered (:attr:`remembered_declarations`) and put back by
        :meth:`restart_broker`.

        :param name: bucket suffix; the fake skips the namespace prefix
        :ptype name: str
        :param ttl: the declared bucket TTL; recorded, not applied
        :ptype ttl: timedelta | None
        :param storage: ``"memory"`` or ``"file"``; recorded on a bucket this call creates
        :ptype storage: str
        :param history: ignored by fake
        :ptype history: int
        :param direct: the declared ``allow_direct`` value; recorded
        :ptype direct: bool
        :param create_if_missing: ``True`` declares; ``False`` binds and raises when the bucket is absent
        :ptype create_if_missing: bool
        :param owns_bucket: the declarer owns the bucket's whole shape, so a live bucket takes the
            declared TTL and storage; only with ``create_if_missing`` on memory storage
        :ptype owns_bucket: bool
        :param drop_file_storage: the owner may recreate a bucket live on file storage, emptying it;
            without it such a bucket is refused and left untouched. only with ``owns_bucket``
        :ptype drop_file_storage: bool
        :return: the bucket, the same instance every later open receives
        :rtype: FakeKvBucket
        :raises ValueError: when ``owns_bucket=True`` with ``create_if_missing=False`` or file storage,
            or ``drop_file_storage=True`` without ``owns_bucket``, as the real one
        :raises KvConfigMismatch: when an owner finds the bucket live on file storage without
            ``drop_file_storage``, as the real one; the bucket is left as it is
        :raises KvBucketNotFoundError: when ``create_if_missing=False`` and the bucket is absent --
            never created, or lost to :meth:`FakeKvBucket.vanish` -- since a declaration asks the
            broker rather than the client's cache
        """
        del history
        if drop_file_storage and not owns_bucket:
            raise ValueError(
                f"KV bucket {name!r}: drop_file_storage=True needs owns_bucket=True -- only the bucket's "
                f"owner may recreate it"
            )
        if owns_bucket and not create_if_missing:
            raise ValueError(
                f"KV bucket {name!r}: owns_bucket=True needs create_if_missing=True -- only the bucket's "
                f"declarer may own it, and a bind-only open declares nothing"
            )
        if owns_bucket and storage != "memory":
            raise ValueError(
                f"KV bucket {name!r}: owns_bucket=True needs storage='memory' -- an owner reconciles "
                f"storage drift by deleting the stream and its entries, which is safe only for a bucket whose "
                f"contents are ephemeral by design"
            )
        bucket = self._buckets.get(name)
        absent = bucket is None or bucket.is_vanished
        if absent and not create_if_missing:
            raise KvBucketNotFoundError(
                f"KV bucket {name!r} does not exist, and this declaration only binds it", bucket=name
            )
        if bucket is None:
            bucket = self._new_bucket(name=name, ttl=ttl, storage=storage, direct=direct)
            self._buckets[name] = bucket
        elif create_if_missing:
            # a declaration CREATES a lost bucket with its own shape; a live one keeps its TTL and
            # storage unless the declarer owns the bucket, as the real declaration reconciles them only then
            takes_shape = owns_bucket or bucket.is_vanished
            if (
                owns_bucket
                and not bucket.is_vanished
                and bucket.storage == "file"
                and storage != "file"
                and not drop_file_storage
            ):
                raise KvConfigMismatch(
                    f"KV bucket {name!r} is live on file storage and its owner declares it on {storage}; "
                    f"recreating it would drop entries a NATS restart would have kept, and this "
                    f"declaration was not given drop_file_storage=True. it is left as it is."
                )
            if bucket.is_vanished:
                # the declaration creates the lost bucket now, empty, as the real one creates its stream
                bucket.wipe()
            bucket.reconcile(
                ttl=ttl if takes_shape else bucket.ttl, direct=direct, storage=storage if takes_shape else None
            )
        # the real declaration replaces the client's one cached handle with one opened as it asked
        bucket.set_may_create(create_if_missing)
        if create_if_missing:
            self._remembered.add(name)
        return bucket

    def _new_bucket(
        self, *, name: str, ttl: timedelta | None, storage: str, direct: bool | None, may_create: bool = True
    ) -> FakeKvBucket:
        """create one fake bucket, aged by ``bucket_age`` when the client was given one.

        :param name: bucket name
        :ptype name: str
        :param ttl: recorded TTL
        :ptype ttl: timedelta | None
        :param storage: recorded storage
        :ptype storage: str
        :param direct: recorded direct-get flag
        :ptype direct: bool | None
        :param may_create: whether operations through the handle recreate a vanished bucket
        :ptype may_create: bool
        :return: the bucket
        :rtype: FakeKvBucket
        """
        bucket = FakeKvBucket(bucket_name=name, ttl=ttl, storage=storage, direct=direct, may_create=may_create)
        if self._bucket_age is not None:
            # `wipe` is how a creation time is placed, and on a bucket with no entries it
            # removes nothing -- so this ages the bucket without pretending anything was lost.
            bucket.wipe(date_created=datetime.now(UTC) - self._bucket_age)
        return bucket
