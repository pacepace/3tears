"""unit tests for :class:`threetears.nats.NatsKvBucket`.

these tests substitute fakes for the underlying nats-py JetStream KV
api so the bucket wrapper logic (CAS semantics, error mapping,
namespace-prefix) can be exercised without a live broker. integration
tests against a real JetStream KV bucket live in tests/integration/.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from nats.js.errors import KeyNotFoundError, KeyWrongLastSequenceError

from threetears.nats import KvError, NatsKvBucket


class _FakeEntry:
    def __init__(self, value: bytes | None, revision: int | None) -> None:
        self.value = value
        self.revision = revision


class _FakeKv:
    """fake KeyValue handle storing entries in a dict."""

    def __init__(self) -> None:
        self.store: dict[str, tuple[bytes, int]] = {}
        self.next_revision = 0
        self.fail_next: BaseException | None = None

    def _maybe_fail(self) -> None:
        if self.fail_next is not None:
            exc = self.fail_next
            self.fail_next = None
            raise exc

    async def get(self, key: str) -> _FakeEntry:
        self._maybe_fail()
        entry = self.store.get(key)
        if entry is None:
            raise KeyNotFoundError()
        value, rev = entry
        return _FakeEntry(value=value, revision=rev)

    async def put(self, key: str, value: bytes) -> int:
        self._maybe_fail()
        self.next_revision += 1
        self.store[key] = (value, self.next_revision)
        return self.next_revision

    async def create(self, key: str, value: bytes) -> int:
        self._maybe_fail()
        if key in self.store:
            raise KeyWrongLastSequenceError()
        self.next_revision += 1
        self.store[key] = (value, self.next_revision)
        return self.next_revision

    async def update(self, key: str, value: bytes, revision: int) -> int:
        self._maybe_fail()
        existing = self.store.get(key)
        if existing is None or existing[1] != revision:
            raise KeyWrongLastSequenceError()
        self.next_revision += 1
        self.store[key] = (value, self.next_revision)
        return self.next_revision

    async def delete(self, key: str) -> None:
        self._maybe_fail()
        if key not in self.store:
            raise KeyNotFoundError()
        del self.store[key]


def _make_bucket() -> tuple[NatsKvBucket, _FakeKv]:
    """construct a NatsKvBucket backed by a fake KV handle."""
    kv = _FakeKv()
    bucket = NatsKvBucket(
        client=None,  # type: ignore[arg-type]
        full_name="aibots-tests",
        kv=kv,  # type: ignore[arg-type]
        ttl=timedelta(seconds=60),
    )
    return bucket, kv


class _FakeJetStream:
    """Returns a pre-seeded healed KV from create_key_value / key_value (re-open path)."""

    def __init__(self, healed_kv: _FakeKv) -> None:
        self._healed_kv = healed_kv

    async def create_key_value(self, _config: Any) -> _FakeKv:
        return self._healed_kv

    async def key_value(self, _name: str) -> _FakeKv:
        return self._healed_kv


class _FakeClient:
    """Minimal NatsClient stand-in whose jetstream re-open yields ``healed_kv``."""

    def __init__(self, healed_kv: _FakeKv) -> None:
        self._js = _FakeJetStream(healed_kv)

    def jetstream_context(self) -> _FakeJetStream:
        return self._js


def _make_self_healing_bucket(broken_kv: _FakeKv, healed_kv: _FakeKv) -> NatsKvBucket:
    """A bucket whose initial handle is ``broken_kv`` and whose re-open yields ``healed_kv``."""
    return NatsKvBucket(
        client=_FakeClient(healed_kv),  # type: ignore[arg-type]
        full_name="aibots-tests",
        kv=broken_kv,  # type: ignore[arg-type]
        ttl=timedelta(seconds=60),
    )


@pytest.mark.asyncio
async def test_get_returns_none_on_miss() -> None:
    bucket, _ = _make_bucket()
    assert await bucket.get(key="absent") is None


@pytest.mark.asyncio
async def test_get_returns_value() -> None:
    bucket, kv = _make_bucket()
    await kv.put("k", b"v")
    assert await bucket.get(key="k") == b"v"


@pytest.mark.asyncio
async def test_get_entry_returns_value_and_revision() -> None:
    bucket, kv = _make_bucket()
    rev = await kv.put("k", b"v")
    entry = await bucket.get_entry(key="k")
    assert entry == (b"v", rev)


@pytest.mark.asyncio
async def test_get_entry_returns_none_on_miss() -> None:
    bucket, _ = _make_bucket()
    assert await bucket.get_entry(key="absent") is None


@pytest.mark.asyncio
async def test_put_returns_new_revision() -> None:
    bucket, _ = _make_bucket()
    rev = await bucket.put(key="k", value=b"v1")
    assert rev > 0


@pytest.mark.asyncio
async def test_create_returns_revision_when_absent() -> None:
    bucket, _ = _make_bucket()
    rev = await bucket.create(key="k", value=b"v")
    assert rev is not None and rev > 0


@pytest.mark.asyncio
async def test_create_returns_none_on_conflict() -> None:
    bucket, _ = _make_bucket()
    await bucket.create(key="k", value=b"v")
    second = await bucket.create(key="k", value=b"v2")
    assert second is None


@pytest.mark.asyncio
async def test_update_cas_succeeds_with_correct_revision() -> None:
    bucket, _ = _make_bucket()
    rev1 = await bucket.put(key="k", value=b"v1")
    assert rev1 is not None
    rev2 = await bucket.update(key="k", value=b"v2", revision=rev1)
    assert rev2 is not None and rev2 != rev1


@pytest.mark.asyncio
async def test_update_cas_returns_none_on_revision_mismatch() -> None:
    bucket, _ = _make_bucket()
    await bucket.put(key="k", value=b"v1")
    result = await bucket.update(key="k", value=b"v2", revision=999)
    assert result is None


@pytest.mark.asyncio
async def test_delete_succeeds() -> None:
    bucket, _ = _make_bucket()
    await bucket.put(key="k", value=b"v")
    assert await bucket.delete(key="k") is True
    assert await bucket.get(key="k") is None


@pytest.mark.asyncio
async def test_delete_idempotent_on_missing_key() -> None:
    bucket, _ = _make_bucket()
    assert await bucket.delete(key="never-existed") is True


@pytest.mark.asyncio
async def test_get_wraps_transport_failure() -> None:
    bucket, kv = _make_bucket()
    kv.fail_next = RuntimeError("transport down")
    with pytest.raises(KvError):
        await bucket.get(key="k")


@pytest.mark.asyncio
async def test_put_wraps_transport_failure() -> None:
    bucket, kv = _make_bucket()
    kv.fail_next = RuntimeError("transport down")
    with pytest.raises(KvError):
        await bucket.put(key="k", value=b"v")


@pytest.mark.asyncio
async def test_create_self_heals_after_stream_vanishes() -> None:
    """A vanished stream (NATS restart on ephemeral storage) is re-opened + retried.

    Regression for the production wake outage: the cached bucket handle failed forever on
    "nats: no response from stream" until a process restart. The op now re-opens once and
    retries against the recreated bucket.
    """
    broken_kv = _FakeKv()
    broken_kv.fail_next = RuntimeError("nats: no response from stream")
    healed_kv = _FakeKv()
    bucket = _make_self_healing_bucket(broken_kv, healed_kv)

    rev = await bucket.create(key="agent_wake_tick", value=b"1")

    assert rev is not None and rev > 0
    assert "agent_wake_tick" in healed_kv.store  # the retry wrote to the recreated bucket


@pytest.mark.asyncio
async def test_put_self_heals_after_stream_vanishes() -> None:
    broken_kv = _FakeKv()
    broken_kv.fail_next = RuntimeError("nats: no response from stream")
    healed_kv = _FakeKv()
    bucket = _make_self_healing_bucket(broken_kv, healed_kv)

    rev = await bucket.put(key="k", value=b"v")

    assert rev > 0
    assert healed_kv.store["k"][0] == b"v"


@pytest.mark.asyncio
async def test_miss_does_not_trigger_reopen() -> None:
    """A normal KeyNotFound miss is control flow, NOT a transport failure -- no re-open."""
    kv = _FakeKv()
    healed_kv = _FakeKv()
    bucket = _make_self_healing_bucket(kv, healed_kv)

    assert await bucket.get(key="absent") is None
    # The handle was never swapped: a miss must not pay a re-open round trip.
    assert bucket._kv is kv  # noqa: SLF001 - asserting no self-heal on a normal miss


@pytest.mark.asyncio
async def test_persistent_failure_after_reopen_surfaces_kverror() -> None:
    """If the op still fails AFTER a re-open (NATS genuinely down), surface KvError."""
    broken_kv = _FakeKv()
    broken_kv.fail_next = RuntimeError("down")
    healed_kv = _FakeKv()
    healed_kv.fail_next = RuntimeError("still down")  # the retry fails too
    bucket = _make_self_healing_bucket(broken_kv, healed_kv)

    with pytest.raises(KvError):
        await bucket.put(key="k", value=b"v")


@pytest.mark.asyncio
async def test_bucket_name_property() -> None:
    bucket, _ = _make_bucket()
    assert bucket.name == "aibots-tests"


@pytest.mark.asyncio
async def test_ttl_property() -> None:
    bucket, _ = _make_bucket()
    assert bucket.ttl == timedelta(seconds=60)
