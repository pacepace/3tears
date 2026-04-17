"""in-memory fake of nats-py KeyValue bucket + JetStream + Client.

mimics the subset of nats-py KV semantics used by
:class:`threetears.core.coordination.lease.KVLease`:

- ``create(key, value)`` rejects duplicate keys with
  :class:`nats.js.errors.KeyWrongLastSequenceError`.
- ``get(key)`` returns entry object with ``.value`` (bytes) and ``.revision``
  (int); raises :class:`nats.js.errors.KeyNotFoundError` when absent.
- ``update(key, value, last=rev)`` performs CAS on revision; raises
  :class:`nats.js.errors.KeyWrongLastSequenceError` on mismatch.
- ``delete(key, last=rev)`` performs CAS delete; raises
  :class:`nats.js.errors.KeyWrongLastSequenceError` on revision mismatch.

the fake stores data in a plain dict keyed by bucket name so multiple
buckets created from the same jetstream share no state. revision counter
is bucket-local and monotonic per bucket.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nats.js.errors import (
    BucketNotFoundError,
    KeyNotFoundError,
    KeyWrongLastSequenceError,
)


@dataclass
class FakeEntry:
    """entry object returned by :meth:`FakeKV.get`.

    :ivar value: stored bytes payload
    :ivar revision: monotonic revision number for CAS
    """

    value: bytes
    revision: int


class FakeKV:
    """in-memory fake of a single nats-py KeyValue bucket.

    tracks entries and a bucket-local monotonic revision counter so
    :class:`KVLease` CAS semantics can be exercised without real NATS.
    """

    def __init__(self, bucket_name: str) -> None:
        """initialize empty fake bucket with zero revision counter.

        :param bucket_name: name used to identify this bucket in logs
        :ptype bucket_name: str
        :return: None
        :rtype: None
        """
        self._bucket_name = bucket_name
        self._entries: dict[str, FakeEntry] = {}
        self._revision = 0

    @property
    def bucket_name(self) -> str:
        """return bucket name.

        :return: bucket name set at construction
        :rtype: str
        """
        return self._bucket_name

    async def create(self, key: str, value: bytes) -> int:
        """insert new key; reject if key already present.

        :param key: key under which to store value
        :ptype key: str
        :param value: bytes payload
        :ptype value: bytes
        :return: new revision assigned to entry
        :rtype: int
        :raises KeyWrongLastSequenceError: if key already exists
        """
        if key in self._entries:
            raise KeyWrongLastSequenceError(description=f"key {key!r} exists")
        self._revision += 1
        self._entries[key] = FakeEntry(value=value, revision=self._revision)
        return self._revision

    async def get(self, key: str) -> FakeEntry:
        """fetch entry for key.

        :param key: key to look up
        :ptype key: str
        :return: entry with value bytes and current revision
        :rtype: FakeEntry
        :raises KeyNotFoundError: if key is absent
        """
        entry = self._entries.get(key)
        if entry is None:
            raise KeyNotFoundError(message=f"key {key!r} not found")
        return entry

    async def update(self, key: str, value: bytes, last: int) -> int:
        """CAS update: replace value only if current revision matches ``last``.

        :param key: key to update
        :ptype key: str
        :param value: new bytes payload
        :ptype value: bytes
        :param last: revision caller believes is current
        :ptype last: int
        :return: new revision assigned to updated entry
        :rtype: int
        :raises KeyWrongLastSequenceError: if revision mismatch or key absent
        """
        entry = self._entries.get(key)
        if entry is None or entry.revision != last:
            raise KeyWrongLastSequenceError(description=f"revision mismatch on key {key!r}")
        self._revision += 1
        self._entries[key] = FakeEntry(value=value, revision=self._revision)
        return self._revision

    async def delete(self, key: str, last: int | None = None) -> bool:
        """CAS delete: remove entry only if revision matches (when ``last`` given).

        :param key: key to remove
        :ptype key: str
        :param last: revision caller believes is current; None disables CAS
        :ptype last: int | None
        :return: True on successful removal
        :rtype: bool
        :raises KeyWrongLastSequenceError: if revision mismatch (when ``last`` given)
        """
        entry = self._entries.get(key)
        if entry is None:
            raise KeyWrongLastSequenceError(description=f"key {key!r} not found for delete")
        if last is not None and entry.revision != last:
            raise KeyWrongLastSequenceError(description=f"revision mismatch on delete of key {key!r}")
        del self._entries[key]
        return True


class FakeJetStream:
    """fake JetStream context returning fake KV buckets on demand."""

    def __init__(self) -> None:
        """initialize with empty bucket registry.

        :return: None
        :rtype: None
        """
        self._buckets: dict[str, FakeKV] = {}

    async def key_value(self, bucket: str) -> FakeKV:
        """open existing bucket or raise if not yet created.

        :param bucket: bucket name
        :ptype bucket: str
        :return: existing fake KV bucket
        :rtype: FakeKV
        :raises BucketNotFoundError: if bucket has not been created
        """
        kv = self._buckets.get(bucket)
        if kv is None:
            raise BucketNotFoundError(description=f"bucket {bucket!r} not found")
        return kv

    async def create_key_value(self, bucket: str, history: int = 1, **_: Any) -> FakeKV:
        """create new fake KV bucket; idempotent — returns existing on re-create.

        :param bucket: bucket name
        :ptype bucket: str
        :param history: history depth (unused by fake)
        :ptype history: int
        :return: newly created or pre-existing fake KV bucket
        :rtype: FakeKV
        """
        kv = self._buckets.get(bucket)
        if kv is None:
            kv = FakeKV(bucket_name=bucket)
            self._buckets[bucket] = kv
        return kv


class FakeNatsClient:
    """fake NATS client exposing only :meth:`jetstream`.

    matches the narrow surface :class:`KVLease` depends on.
    """

    def __init__(self) -> None:
        """initialize with single backing fake JetStream.

        :return: None
        :rtype: None
        """
        self._js = FakeJetStream()

    def jetstream(self) -> FakeJetStream:
        """return backing fake JetStream context.

        :return: fake JetStream context
        :rtype: FakeJetStream
        """
        return self._js
