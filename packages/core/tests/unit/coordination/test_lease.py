"""tests for threetears.core.coordination.lease.KVLease.

covers acquire/refresh/release/timeout/contention/async-with against
fake NATS KV bucket (mirroring :class:`threetears.nats.NatsKvBucket`).
no real NATS process required.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from threetears.core.coordination.lease import (
    KVLease,
    LeaseHandle,
    LeaseLost,
    LeaseTimeout,
    LeaseUnavailable,
)
from threetears.core.serialization import deserialize_from_json, serialize_to_json

from ._fake_kv import FakeKvBucket, FakeNatsClient


async def _make_lease(
    pod_id: str = "pod-alpha",
    bucket_name: str = "test_leases",
) -> tuple[KVLease, FakeNatsClient]:
    """construct KVLease wired to fresh fake NATS client for test use.

    :param pod_id: explicit holder identifier
    :ptype pod_id: str
    :param bucket_name: bucket name to operate against
    :ptype bucket_name: str
    :return: tuple of configured lease and backing fake client
    :rtype: tuple[KVLease, FakeNatsClient]
    """
    client = FakeNatsClient()
    lease = KVLease(nats_client=client, bucket_name=bucket_name, pod_id=pod_id)  # type: ignore[arg-type]
    return lease, client


async def _bucket_for(client: FakeNatsClient, bucket_name: str) -> FakeKvBucket:
    """resolve fake bucket by name (test-side helper)."""
    return await client.kv_bucket(name=bucket_name)


def _decode_envelope(value: bytes) -> dict[str, Any]:
    """decode stored KV value bytes back to envelope dict.

    :param value: bytes payload as stored in KV bucket
    :ptype value: bytes
    :return: envelope dict with holder, expires_at, acquired_at
    :rtype: dict[str, Any]
    """
    return deserialize_from_json(value, field_types={})


class TestAcquireEmpty:
    """acquire on an empty bucket succeeds and records ownership."""

    async def test_acquire_returns_handle_on_empty_key(self) -> None:
        lease, _client = await _make_lease()
        handle = await lease.acquire("lock/a", ttl_seconds=30)
        assert isinstance(handle, LeaseHandle)
        assert handle.holder == "pod-alpha"
        assert handle.key == "lock/a"

    async def test_acquire_writes_holder_and_expiry_to_kv(self) -> None:
        lease, client = await _make_lease()
        before = datetime.now(UTC)
        await lease.acquire("lock/a", ttl_seconds=30)
        bucket = await _bucket_for(client, "test_leases")
        value = await bucket.get(key="lock/a")
        assert value is not None
        envelope = _decode_envelope(value)
        assert envelope["holder"] == "pod-alpha"
        expires_at = datetime.fromisoformat(envelope["expires_at"])
        acquired_at = datetime.fromisoformat(envelope["acquired_at"])
        assert expires_at > acquired_at
        assert (expires_at - acquired_at) >= timedelta(seconds=29)
        assert acquired_at >= before - timedelta(seconds=1)


class TestAcquireFailFast:
    """max_wait_seconds=0 fails fast without sleeping when key is held."""

    async def test_raises_lease_unavailable_immediately(self) -> None:
        holder_lease, client = await _make_lease(pod_id="pod-first")
        await holder_lease.acquire("lock/a", ttl_seconds=60)

        second = KVLease(nats_client=client, bucket_name="test_leases", pod_id="pod-second")  # type: ignore[arg-type]
        with pytest.raises(LeaseUnavailable):
            await second.acquire("lock/a", ttl_seconds=30, max_wait_seconds=0)

    async def test_fail_fast_does_not_sleep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        holder_lease, client = await _make_lease(pod_id="pod-first")
        await holder_lease.acquire("lock/a", ttl_seconds=60)

        sleeps: list[float] = []
        original_sleep = asyncio.sleep

        async def _spy_sleep(seconds: float) -> None:
            sleeps.append(seconds)
            await original_sleep(0)

        monkeypatch.setattr("threetears.core.coordination.lease.asyncio.sleep", _spy_sleep)
        second = KVLease(nats_client=client, bucket_name="test_leases", pod_id="pod-second")  # type: ignore[arg-type]
        with pytest.raises(LeaseUnavailable):
            await second.acquire("lock/a", ttl_seconds=30, max_wait_seconds=0)
        assert sleeps == []


class TestAcquireWaitThenSucceeds:
    """acquire waits while held then succeeds once prior holder releases."""

    async def test_waits_and_acquires_after_release(self) -> None:
        first_lease, client = await _make_lease(pod_id="pod-first")
        handle = await first_lease.acquire("lock/a", ttl_seconds=60)

        async def _release_after_delay() -> None:
            await asyncio.sleep(0.1)
            await handle.release()

        second = KVLease(nats_client=client, bucket_name="test_leases", pod_id="pod-second")  # type: ignore[arg-type]
        release_task = asyncio.create_task(_release_after_delay())
        try:
            second_handle = await second.acquire("lock/a", ttl_seconds=30, max_wait_seconds=5)
        finally:
            await release_task
        assert second_handle.holder == "pod-second"


class TestAcquireTimeout:
    """deadline elapses -> LeaseTimeout is raised."""

    async def test_raises_lease_timeout_when_deadline_passes(self) -> None:
        holder_lease, client = await _make_lease(pod_id="pod-first")
        await holder_lease.acquire("lock/a", ttl_seconds=60)

        second = KVLease(nats_client=client, bucket_name="test_leases", pod_id="pod-second")  # type: ignore[arg-type]
        with pytest.raises(LeaseTimeout):
            await second.acquire("lock/a", ttl_seconds=30, max_wait_seconds=1)


class TestRefresh:
    """refresh extends TTL and advances revision; ownership changes raise LeaseLost."""

    async def test_refresh_by_current_holder_advances_revision(self) -> None:
        lease, client = await _make_lease()
        handle = await lease.acquire("lock/a", ttl_seconds=30)
        original_revision = handle.revision
        bucket = await _bucket_for(client, "test_leases")
        value_before = await bucket.get(key="lock/a")
        assert value_before is not None
        envelope_before = _decode_envelope(value_before)
        expires_before = datetime.fromisoformat(envelope_before["expires_at"])

        await asyncio.sleep(0.01)
        await handle.refresh(ttl_seconds=120)

        assert handle.revision != original_revision
        value_after = await bucket.get(key="lock/a")
        assert value_after is not None
        envelope_after = _decode_envelope(value_after)
        expires_after = datetime.fromisoformat(envelope_after["expires_at"])
        assert expires_after > expires_before

    async def test_refresh_raises_lease_lost_on_holder_mismatch(self) -> None:
        first_lease, client = await _make_lease(pod_id="pod-first")
        handle = await first_lease.acquire("lock/a", ttl_seconds=30)

        # another pod steals the lease by deleting and recreating it
        bucket = await _bucket_for(client, "test_leases")
        deleted = await bucket.delete(key="lock/a", revision=handle.revision)
        assert deleted is True
        now = datetime.now(UTC)
        thief_envelope = serialize_to_json(
            {
                "holder": "pod-thief",
                "expires_at": (now + timedelta(seconds=30)).isoformat(),
                "acquired_at": now.isoformat(),
            }
        )
        await bucket.create(key="lock/a", value=thief_envelope)

        with pytest.raises(LeaseLost):
            await handle.refresh()

    async def test_refresh_raises_lease_lost_on_revision_mismatch(self) -> None:
        lease, client = await _make_lease()
        handle = await lease.acquire("lock/a", ttl_seconds=30)
        # holder line stays the same but a concurrent writer (same pod_id)
        # advances the revision underneath us, simulating a successor write.
        bucket = await _bucket_for(client, "test_leases")
        now = datetime.now(UTC)
        sneak = serialize_to_json(
            {
                "holder": "pod-alpha",
                "expires_at": (now + timedelta(seconds=30)).isoformat(),
                "acquired_at": now.isoformat(),
            }
        )
        new_revision = await bucket.update(key="lock/a", value=sneak, revision=handle.revision)
        assert new_revision is not None
        # handle.revision is now stale; refresh CAS must fail -> LeaseLost
        with pytest.raises(LeaseLost):
            await handle.refresh()


class TestRelease:
    """release is idempotent and safe when ownership has moved on."""

    async def test_release_removes_entry(self) -> None:
        lease, client = await _make_lease()
        handle = await lease.acquire("lock/a", ttl_seconds=30)
        await handle.release()
        bucket = await _bucket_for(client, "test_leases")
        assert await bucket.get(key="lock/a") is None

    async def test_release_is_idempotent(self) -> None:
        lease, _client = await _make_lease()
        handle = await lease.acquire("lock/a", ttl_seconds=30)
        await handle.release()
        # second call must not raise
        await handle.release()

    async def test_release_after_theft_is_noop(self) -> None:
        lease, client = await _make_lease(pod_id="pod-first")
        handle = await lease.acquire("lock/a", ttl_seconds=30)
        bucket = await _bucket_for(client, "test_leases")
        # simulate another pod replacing the entry
        await bucket.delete(key="lock/a", revision=handle.revision)
        now = datetime.now(UTC)
        thief = serialize_to_json(
            {
                "holder": "pod-thief",
                "expires_at": (now + timedelta(seconds=30)).isoformat(),
                "acquired_at": now.isoformat(),
            }
        )
        await bucket.create(key="lock/a", value=thief)
        # release must not raise; thief's entry must remain.
        await handle.release()
        value = await bucket.get(key="lock/a")
        assert value is not None
        envelope = _decode_envelope(value)
        assert envelope["holder"] == "pod-thief"


class TestAsyncWith:
    """LeaseHandle async-with releases on normal exit and on exception."""

    async def test_async_with_releases_on_normal_exit(self) -> None:
        lease, client = await _make_lease()
        handle = await lease.acquire("lock/a", ttl_seconds=30)
        async with handle:
            pass
        bucket = await _bucket_for(client, "test_leases")
        assert await bucket.get(key="lock/a") is None

    async def test_async_with_releases_on_exception(self) -> None:
        lease, client = await _make_lease()
        handle = await lease.acquire("lock/a", ttl_seconds=30)

        class _Boom(RuntimeError):
            """synthetic exception for async-with cleanup verification."""

        with pytest.raises(_Boom):
            async with handle:
                raise _Boom("oops")

        bucket = await _bucket_for(client, "test_leases")
        assert await bucket.get(key="lock/a") is None


class TestStaleLease:
    """stale lease (expires_at in past) is reclaimable via CAS."""

    async def test_stale_entry_reclaimed_via_cas(self) -> None:
        first_lease, client = await _make_lease(pod_id="pod-first")
        first_handle = await first_lease.acquire("lock/a", ttl_seconds=30)

        # rewrite entry with already-expired expires_at (preserve revision chain)
        bucket = await _bucket_for(client, "test_leases")
        past = datetime.now(UTC) - timedelta(seconds=120)
        expired = serialize_to_json(
            {
                "holder": "pod-first",
                "expires_at": past.isoformat(),
                "acquired_at": (past - timedelta(seconds=30)).isoformat(),
            }
        )
        new_revision = await bucket.update(key="lock/a", value=expired, revision=first_handle.revision)
        assert new_revision is not None

        second = KVLease(nats_client=client, bucket_name="test_leases", pod_id="pod-second")  # type: ignore[arg-type]
        handle = await second.acquire("lock/a", ttl_seconds=30, max_wait_seconds=0)
        assert handle.holder == "pod-second"
        value = await bucket.get(key="lock/a")
        assert value is not None
        envelope = _decode_envelope(value)
        assert envelope["holder"] == "pod-second"


class TestBucketDefaults:
    """default bucket name derives from FOURTEENAIBOTS_NATS_SUBJECT_NAMESPACE."""

    async def test_default_bucket_uses_namespace_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FOURTEENAIBOTS_NATS_SUBJECT_NAMESPACE", "prod14")
        client = FakeNatsClient()
        lease = KVLease(nats_client=client, pod_id="pod-alpha")  # type: ignore[arg-type]
        await lease.acquire("lock/a", ttl_seconds=30)
        # bucket should exist under "prod14_leases"
        bucket = await _bucket_for(client, "prod14_leases")
        value = await bucket.get(key="lock/a")
        assert value is not None
        envelope = _decode_envelope(value)
        assert envelope["holder"] == "pod-alpha"

    async def test_default_bucket_fallback_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FOURTEENAIBOTS_NATS_SUBJECT_NAMESPACE", raising=False)
        client = FakeNatsClient()
        lease = KVLease(nats_client=client, pod_id="pod-alpha")  # type: ignore[arg-type]
        await lease.acquire("lock/a", ttl_seconds=30)
        bucket = await _bucket_for(client, "leases")
        value = await bucket.get(key="lock/a")
        assert value is not None
        envelope = _decode_envelope(value)
        assert envelope["holder"] == "pod-alpha"

    async def test_default_pod_id_is_generated_when_omitted(self) -> None:
        client = FakeNatsClient()
        lease = KVLease(nats_client=client, bucket_name="test_leases")  # type: ignore[arg-type]
        handle = await lease.acquire("lock/a", ttl_seconds=30)
        assert handle.holder.startswith("pod-")
        assert len(handle.holder) == len("pod-") + 12
