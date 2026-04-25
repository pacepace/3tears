"""in-memory fake of :class:`threetears.nats.NatsKvBucket` + :class:`NatsClient`.

mirrors the wrapper surface :class:`KVLease` actually consumes:

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

from dataclasses import dataclass


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

    def __init__(self, bucket_name: str) -> None:
        """initialize empty fake bucket with zero revision counter.

        :param bucket_name: full bucket name (with namespace prefix)
        :ptype bucket_name: str
        :return: None
        :rtype: None
        """
        self._bucket_name = bucket_name
        self._entries: dict[str, _Entry] = {}
        self._revision = 0

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
        :return: ``True`` on success or absent key, ``False`` on CAS mismatch
        :rtype: bool
        """
        entry = self._entries.get(key)
        if entry is None:
            return True
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
        self._revision += 1
        self._entries[key] = _Entry(value=value, revision=self._revision)
        return self._revision


class FakeNatsClient:
    """fake NATS wrapper exposing :meth:`kv_bucket` returning :class:`FakeKvBucket`.

    matches the narrow surface :class:`KVLease` depends on. the bucket
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
        storage: str = "file",
        create_if_missing: bool = True,
        history: int = 1,
    ) -> FakeKvBucket:
        """return existing bucket or create one. idempotent.

        :param name: bucket suffix; the fake skips the namespace
            prefix the real wrapper layers on top
        :ptype name: str
        :param ttl: ignored by fake
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
        del ttl, storage, history
        bucket = self._buckets.get(name)
        if bucket is None:
            if not create_if_missing:
                raise KeyError(f"bucket {name!r} not found")
            bucket = FakeKvBucket(bucket_name=name)
            self._buckets[name] = bucket
        return bucket
