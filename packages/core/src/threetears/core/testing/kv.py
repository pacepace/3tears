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

the fake stores data in a plain dict keyed by bucket name so multiple
buckets created from the same client share no state. revision counter
is bucket-local and monotonic per bucket.
"""

from __future__ import annotations

from datetime import timedelta

from collections.abc import Generator
from dataclasses import dataclass

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
    """internal storage entry."""

    value: bytes
    revision: int


class FakeKvBucket:
    """in-memory fake mirroring :class:`threetears.nats.NatsKvBucket`.

    methods take kw-only args matching the wrapper surface so test
    fixtures exercise the same call shape production code uses.
    """

    def __init__(self, bucket_name: str, ttl: timedelta | None = None) -> None:
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
        :return: None
        :rtype: None
        """
        self._bucket_name = bucket_name
        self._ttl = ttl
        self._entries: dict[str, _Entry] = {}
        self._revision = 0

    @property
    def ttl(self) -> timedelta | None:
        """the TTL this bucket was opened with; ``None`` means no expiry.

        :return: the bucket TTL
        :rtype: timedelta | None
        """
        return self._ttl

    @property
    def name(self) -> str:
        """fully-qualified bucket name.

        :return: bucket name
        :rtype: str
        """
        return self._bucket_name

    async def create(self, *, key: str, value: bytes) -> int | None:
        """create-if-absent. returns new revision or ``None`` on conflict.

        :param key: key to insert
        :ptype key: str
        :param value: bytes payload
        :ptype value: bytes
        :return: new revision number, or ``None`` if key already exists
        :rtype: int | None
        """
        await _YieldOnce()  # so gather() genuinely interleaves
        if key in self._entries:
            return None
        self._revision += 1
        self._entries[key] = _Entry(value=value, revision=self._revision)
        return self._revision

    async def get(self, *, key: str) -> bytes | None:
        """get value bytes for key. returns ``None`` on miss.

        :param key: key to read
        :ptype key: str
        :return: stored bytes or ``None``
        :rtype: bytes | None
        """
        await _YieldOnce()  # so gather() genuinely interleaves
        entry = self._entries.get(key)
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
        entry = self._entries.get(key)
        if entry is None:
            return None
        return (entry.value, entry.revision)

    async def update(self, *, key: str, value: bytes, revision: int) -> int | None:
        """CAS update. returns new revision or ``None`` on mismatch.

        :param key: key to update
        :ptype key: str
        :param value: new bytes payload
        :ptype value: bytes
        :param revision: expected current revision
        :ptype revision: int
        :return: new revision, or ``None`` on conflict / missing key
        :rtype: int | None
        """
        await _YieldOnce()  # so gather() genuinely interleaves
        entry = self._entries.get(key)
        if entry is None or entry.revision != revision:
            return None
        self._revision += 1
        self._entries[key] = _Entry(value=value, revision=self._revision)
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
        entry = self._entries.get(key)
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

    async def put(self, *, key: str, value: bytes) -> int:
        """unconditional write. returns new revision.

        :param key: key to write
        :ptype key: str
        :param value: bytes payload
        :ptype value: bytes
        :return: new revision number
        :rtype: int
        """
        await _YieldOnce()  # so gather() genuinely interleaves
        self._revision += 1
        self._entries[key] = _Entry(value=value, revision=self._revision)
        return self._revision


class FakeNatsClient:
    """fake NATS wrapper exposing :meth:`kv_bucket` returning :class:`FakeKvBucket`.

    matches the narrow surface KV consumers depend on. the bucket
    cache mirrors :class:`NatsClient`'s internal cache: repeat
    ``kv_bucket`` calls for the same name return the same instance.
    """

    def __init__(self) -> None:
        """initialize with empty bucket registry.

        :return: None
        :rtype: None
        """
        self._buckets: dict[str, FakeKvBucket] = {}

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
        :param storage: ignored by fake
        :ptype storage: str
        :param create_if_missing: when ``False`` and bucket absent, raises
        :ptype create_if_missing: bool
        :param history: ignored by fake
        :ptype history: int
        :return: fake bucket
        :rtype: FakeKvBucket
        :raises KeyError: when ``create_if_missing=False`` and bucket absent
        """
        del storage, history
        bucket = self._buckets.get(name)
        if bucket is None:
            if not create_if_missing:
                raise KeyError(f"bucket {name!r} not found")
            bucket = FakeKvBucket(bucket_name=name, ttl=ttl if isinstance(ttl, timedelta) else None)
            self._buckets[name] = bucket
        return bucket
