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
- :meth:`FakeKvBucket.update` returns the new revision on success or
  ``None`` on CAS conflict (revision mismatch or key absent).
- :meth:`FakeKvBucket.delete` accepts an optional ``revision`` and
  returns ``True`` on success or absent key, ``False`` on CAS mismatch.
- :meth:`FakeKvBucket.date_created` reports when the bucket was created, and
  :meth:`FakeKvBucket.wipe` empties it and moves that time forward, which is
  what a broker restart does to a memory-backed bucket.

the fake stores data in a plain dict keyed by bucket name so multiple
buckets created from the same client share no state. revision counter
is bucket-local and monotonic per bucket.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

__all__ = ["FakeKvBucket", "FakeNatsClient"]


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


class FakeKvBucket:
    """in-memory fake mirroring :class:`threetears.nats.NatsKvBucket`.

    methods take kw-only args matching the wrapper surface so test
    fixtures exercise the same call shape production code uses.
    """

    def __init__(self, bucket_name: str, ttl: timedelta | None = None, storage: str = "memory") -> None:
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
        :return: None
        :rtype: None
        """
        self._bucket_name = bucket_name
        self._ttl = ttl
        self._storage = storage
        self._entries: dict[str, _Entry] = {}
        self._revision = 0
        self._date_created = datetime.now(UTC)
        # a clock only this bucket reads, moved by advance_clock, so a test can make a per-entry
        # TTL lapse without sleeping.
        self._elapsed = timedelta(0)

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
            del self._entries[key]
            entry = None
        return entry

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
        await _YieldOnce()  # so gather() genuinely interleaves
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
        """empty the bucket and give it a new creation time, as a broker restart does.

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
        self._date_created = date_created if date_created is not None else datetime.now(UTC)

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
        await _YieldOnce()  # so gather() genuinely interleaves
        if self._live(key) is not None:
            return None
        self._revision += 1
        self._entries[key] = _Entry(value=value, revision=self._revision, expires_at=self._expiry(ttl))
        return self._revision

    async def get(self, *, key: str) -> bytes | None:
        """get value bytes for key. returns ``None`` on miss.

        :param key: key to read
        :ptype key: str
        :return: stored bytes or ``None``
        :rtype: bytes | None
        """
        await _YieldOnce()  # so gather() genuinely interleaves
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
        await _YieldOnce()  # so gather() genuinely interleaves
        entry = self._live(key)
        if entry is None:
            return None
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
        :return: new revision, or ``None`` on conflict / missing key
        :rtype: int | None
        """
        await _YieldOnce()  # so gather() genuinely interleaves
        entry = self._live(key)
        if entry is None or entry.revision != revision:
            return None
        self._revision += 1
        self._entries[key] = _Entry(value=value, revision=self._revision, expires_at=self._expiry(ttl))
        return self._revision

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
        await _YieldOnce()  # so gather() genuinely interleaves
        entry = self._live(key)
        if entry is None:
            # A revision-guarded delete of a key that is no longer there LOST the race -- it
            # cannot have been the caller whose revision matched. Returning True here made
            # every concurrent redemption look like a winner, which is how a non-atomic claim
            # passed a concurrency test.
            return revision is None
        if revision is not None and entry.revision != revision:
            return False
        del self._entries[key]
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
        await _YieldOnce()  # so gather() genuinely interleaves
        self._revision += 1
        self._entries[key] = _Entry(value=value, revision=self._revision, expires_at=self._expiry(ttl))
        return self._revision


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

    def __init__(self) -> None:
        """initialize with empty bucket registry.

        :return: None
        :rtype: None
        """
        self._buckets: dict[str, FakeKvBucket] = {}
        self.published: list[Any] = []
        self._subscribers: dict[str, list[tuple[Any, Any]]] = {}

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
            )
            self._buckets[name] = bucket
        return bucket
