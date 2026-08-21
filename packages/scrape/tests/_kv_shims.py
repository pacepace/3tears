"""An in-memory JetStream KV, so the session-claim tests run a real ``KVLease``.

The alternative was faking ``KVLease`` itself, which would have tested this package's
orchestration against a mock of the very semantics it depends on -- compare-and-swap renewal,
holder identity, stale reclaim. Those are the properties a claim is built out of, so they are
the last thing to stub. Faking one layer lower means every assertion above runs the real lease.

Narrower than ``threetears.nats.NatsKvBucket``: only the operations ``KVLease`` actually calls.
A method it never reaches is a method whose fake behaviour nothing here would notice being
wrong.

DUPLICATION, recorded rather than hidden: ``packages/core/tests/unit/coordination/_fake_kv.py``
is the same idea, written for the lease's own tests, and it cannot be imported across package
test trees. The second copy is this one. A third is the point at which it should be promoted
into ``threetears.core.testing`` instead of written again.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Entry:
    """One stored value and the revision it was written at."""

    value: bytes
    revision: int


# parity-exempt: subset shim for nats.js.KeyValue carrying only the create/get_entry/update/delete surface KVLease calls while a session claim is held; put and get are deliberately absent so a claim renewed by an unconditional write fails here instead of passing
class FakeKvBucket:
    """In-memory stand-in for the KV bucket ``KVLease`` writes its entries to.

    Revisions are bucket-local and monotonic, which is what makes compare-and-swap mean
    anything: an update guarded by a revision fails once anybody else has written.
    """

    def __init__(self, bucket_name: str) -> None:
        self._bucket_name = bucket_name
        self._entries: dict[str, _Entry] = {}
        self._revision = 0
        #: Raised by every read while set, to model a bucket this pod cannot reach.
        self.unreachable: Exception | None = None

    @property
    def name(self) -> str:
        """The fully-qualified bucket name."""
        return self._bucket_name

    def _check_reachable(self) -> None:
        """Fail the way an unreachable bucket does, if the test has asked for one."""
        if self.unreachable is not None:
            raise self.unreachable

    async def create(self, *, key: str, value: bytes) -> int | None:
        """Create if absent; ``None`` when the key is already taken."""
        self._check_reachable()
        if key in self._entries:
            return None
        self._revision += 1
        self._entries[key] = _Entry(value=value, revision=self._revision)
        return self._revision

    async def get_entry(self, *, key: str) -> tuple[bytes, int] | None:
        """The stored value and its revision, or ``None`` when the key is absent."""
        self._check_reachable()
        entry = self._entries.get(key)
        return None if entry is None else (entry.value, entry.revision)

    async def update(self, *, key: str, value: bytes, revision: int) -> int | None:
        """Compare-and-swap; ``None`` when the revision has moved or the key has gone."""
        self._check_reachable()
        entry = self._entries.get(key)
        if entry is None or entry.revision != revision:
            return None
        self._revision += 1
        self._entries[key] = _Entry(value=value, revision=self._revision)
        return self._revision

    async def delete(self, *, key: str, revision: int | None = None) -> bool:
        """Delete, optionally guarded by a revision. Absent is success, not failure."""
        self._check_reachable()
        entry = self._entries.get(key)
        if entry is None:
            return True
        if revision is not None and entry.revision != revision:
            return False
        del self._entries[key]
        return True


# parity-exempt: subset shim for nats.aio.Client exposing kv_bucket alone, which is the entire surface KVLease reaches through; the full client's two dozen publish/subscribe/jetstream methods are unreachable from a lease
class FakeNatsClient:
    """Just enough NATS client for ``KVLease`` to open a bucket through.

    Buckets are cached by name, as the real wrapper caches them, so two lease factories sharing
    a client contend over the same entries -- which is the whole point when the test is about
    two pods.
    """

    def __init__(self) -> None:
        self.buckets: dict[str, FakeKvBucket] = {}

    async def kv_bucket(
        self,
        *,
        name: str,
        ttl: object | None = None,
        storage: str = "memory",
        create_if_missing: bool = True,
        history: int = 1,
    ) -> FakeKvBucket:
        """Return the named bucket, creating it on first ask."""
        del ttl, storage, create_if_missing, history
        bucket = self.buckets.get(name)
        if bucket is None:
            bucket = FakeKvBucket(bucket_name=name)
            self.buckets[name] = bucket
        return bucket
