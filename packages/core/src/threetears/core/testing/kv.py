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
  self-heal does.
- :meth:`FakeKvBucket.watch_key` yields the key's latest message -- a value,
  or a deletion marker with ``value=None`` -- then every later write or
  delete of that key, as :meth:`threetears.nats.NatsKvBucket.watch_key`
  does. It yields :class:`threetears.nats.kv_watch.KvKeyUpdate`, the same
  type, so a consumer's test runs the code path production runs.
- :meth:`FakeNatsClient.add_reconnect_callback` registers a hook and
  :meth:`FakeNatsClient.reconnect` runs every hook, as a real reconnect does.

the fake stores data in a plain dict keyed by bucket name so multiple
buckets created from the same client share no state. revision counter
is bucket-local and monotonic per incarnation: a wiped or vanished bucket
starts again at 1, as a recreated stream's sequence does.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from dataclasses import dataclass
from typing import Any

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
        :return: None
        :rtype: None
        """
        self._bucket_name = bucket_name
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
        # a clock only this bucket reads, moved by advance_clock, so a test can make a per-entry
        # TTL lapse without sleeping.
        self._elapsed = timedelta(0)
        # one queue per open watch_key iterator, keyed by the key it watches.
        self._key_watchers: dict[str, list[asyncio.Queue[KvKeyUpdate]]] = {}

    async def _arrive(self) -> None:
        """what every operation does first: yield to the loop, then heal a vanished bucket.

        The yield is so ``gather()`` genuinely interleaves. The heal mirrors the real wrapper,
        which recreates a vanished stream on the next operation through any handle, so the
        recreated bucket's creation time is the moment of that operation.

        :return: None
        :rtype: None
        """
        await _YieldOnce()
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
        moment in between: the entries are gone, and the next operation through any handle
        recreates the bucket, taking that operation's moment as its creation time. That is what
        the real wrapper's self-heal does, and it is the difference between recreating a bucket
        when the broker comes back and recreating it whenever someone next happens to use it.

        :return: None
        :rtype: None
        """
        self._entries.clear()
        self._markers.clear()
        self._vanished = True

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

    def __init__(self, *, bucket_age: timedelta | None = None) -> None:
        """initialize with empty bucket registry.

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
        :return: None
        :rtype: None
        :raises ValueError: when ``bucket_age`` is negative
        """
        if bucket_age is not None and bucket_age < timedelta(0):
            raise ValueError(f"FakeNatsClient bucket_age must not be negative, got {bucket_age}")
        self._bucket_age = bucket_age
        self._buckets: dict[str, FakeKvBucket] = {}
        self.published: list[Any] = []
        self._subscribers: dict[str, list[tuple[Any, Any]]] = {}
        self._reconnect_callbacks: list[Callable[[], Awaitable[None]]] = []

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
        :return: fake bucket
        :rtype: FakeKvBucket
        :raises KeyError: when ``create_if_missing=False`` and bucket absent
        """
        del history
        bucket = self._buckets.get(name)
        if bucket is None:
            if not create_if_missing:
                raise KeyError(f"bucket {name!r} not found")
            bucket = FakeKvBucket(
                bucket_name=name,
                ttl=ttl if isinstance(ttl, timedelta) else None,
                storage=storage,
                direct=direct,
            )
            if self._bucket_age is not None:
                # `wipe` is how a creation time is placed, and on a bucket with no entries it
                # removes nothing -- so this ages the bucket without pretending anything was lost.
                bucket.wipe(date_created=datetime.now(UTC) - self._bucket_age)
            self._buckets[name] = bucket
        return bucket
