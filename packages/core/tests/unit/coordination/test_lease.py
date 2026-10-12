"""tests for threetears.core.coordination.lease.KVLease.

covers acquire/refresh/release/timeout/contention/async-with against
fake NATS KV bucket (mirroring :class:`threetears.nats.NatsKvBucket`).
no real NATS process required.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest

from threetears.core.coordination import lease as lease_module
from threetears.core.coordination.lease import (
    KVLease,
    LeaseHandle,
    LeaseLost,
    LeaseTimeout,
    LeaseUnavailable,
)
from threetears.core.serialization import deserialize_from_json, json_datetime, serialize_to_json

from threetears.core.testing.kv import FakeKvBucket, FakeNatsClient


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

    async def test_the_envelope_stores_both_instants_in_the_one_stored_form(self) -> None:
        """``json_datetime``'s fixed-width form, as every storage tier writes it, not ``isoformat()``."""
        lease, client = await _make_lease()
        await lease.acquire("lock/a", ttl_seconds=30)
        value = await (await _bucket_for(client, "test_leases")).get(key="lock/a")
        assert value is not None
        envelope = _decode_envelope(value)
        for field in ("expires_at", "acquired_at"):
            stored = envelope[field]
            assert stored == json_datetime(datetime.fromisoformat(stored))


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
    """the default bucket name is a constant SUFFIX; the transport owns the prefix."""

    async def test_default_bucket_does_not_vary_with_the_namespace_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """the namespace must NOT reach this default, and that is the whole point.

        ``kv_bucket`` layers the connection's own ``{namespace}-`` over whatever it is
        handed. A default that read the namespace itself produced ``{ns}-{ns}_leases``
        -- the namespace twice -- and named a bucket that the KV grant minted for it
        did not cover, so the first acquire blocked to its deadline rather than raising.

        This asserts the property directly (same name under a set and an unset env) so
        a reintroduced env read fails here rather than in a deployment. Note the fake
        below deliberately skips the prefix the real wrapper applies, which is exactly
        why the original defect was invisible to every unit test of this class -- so the
        assertion is on the name PASSED to the transport, not on a composed one.
        """
        client = FakeNatsClient()
        monkeypatch.setenv("THREETEARS_NATS_SUBJECT_NAMESPACE", "prod14")
        lease = KVLease(nats_client=client, pod_id="pod-alpha")  # type: ignore[arg-type]
        await lease.acquire("lock/a", ttl_seconds=30)
        bucket = await _bucket_for(client, "leases")
        assert await bucket.get(key="lock/a") is not None
        assert "prod14" not in lease.bucket_name

        monkeypatch.delenv("THREETEARS_NATS_SUBJECT_NAMESPACE", raising=False)
        assert KVLease(nats_client=FakeNatsClient(), pod_id="p").bucket_name == lease.bucket_name  # type: ignore[arg-type]

    async def test_default_bucket_is_the_leases_suffix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("THREETEARS_NATS_SUBJECT_NAMESPACE", raising=False)
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
        assert len(handle.holder) == len("pod-") + 32

    async def test_two_factories_built_in_one_millisecond_are_two_holders(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """the holder id is the lease's fence on refresh and release, so it must name ONE factory.

        a uuid7 leads with its millisecond timestamp; a holder id cut from that head was shared by
        every factory built in the same millisecond -- two pods starting together would each
        pass the other's holder check.
        """
        same_millisecond = iter(
            [UUID("019470a8-b5c3-7def-8123-456789abcdef"), UUID("019470a8-b5c3-7a01-9fed-cba987654321")]
        )
        monkeypatch.setattr(lease_module, "uuid7", lambda: next(same_millisecond))
        client = FakeNatsClient()

        first = KVLease(nats_client=client, bucket_name="test_leases")  # type: ignore[arg-type]
        second = KVLease(nats_client=client, bucket_name="test_leases")  # type: ignore[arg-type]

        assert first.pod_id != second.pod_id


class TestOwnerScopedLeaseKeys:
    """a lease over a SHARED bucket keys every claim under its owner's scope.

    The platform's ``leases`` bucket is one bucket every tool pod binds, and each pod is granted only
    the keys under its own scope. Replicas of one pod share the scope and so contend for one key; a
    different pod's claim on the same name is a different key it cannot see.
    """

    _SCOPE = "tool_pod-01947100000070008000000000000001"

    @pytest.mark.asyncio
    async def test_a_claim_is_written_under_the_owners_scope(self) -> None:
        client = FakeNatsClient()
        lease = KVLease(client, bucket_name="leases", pod_id="replica-1", key_scope=self._SCOPE)  # type: ignore[arg-type]
        handle = await lease.acquire("session-digest", ttl_seconds=30, max_wait_seconds=0)
        assert handle.key == f"{self._SCOPE}.session-digest"
        bucket = await client.kv_bucket(name="leases")
        assert await bucket.get(key=f"{self._SCOPE}.session-digest") is not None
        assert await bucket.get(key="session-digest") is None
        await handle.release()
        assert await bucket.get(key=f"{self._SCOPE}.session-digest") is None

    @pytest.mark.asyncio
    async def test_replicas_of_one_owner_contend_for_one_key(self) -> None:
        client = FakeNatsClient()
        first = KVLease(client, bucket_name="leases", pod_id="replica-1", key_scope=self._SCOPE)  # type: ignore[arg-type]
        second = KVLease(client, bucket_name="leases", pod_id="replica-2", key_scope=self._SCOPE)  # type: ignore[arg-type]
        held = await first.acquire("s", ttl_seconds=30, max_wait_seconds=0)
        with pytest.raises(LeaseUnavailable):
            await second.acquire("s", ttl_seconds=30, max_wait_seconds=0)
        await held.release()

    @pytest.mark.parametrize("scope", ["", "a.b", "a*", ">"])
    def test_a_scope_that_is_not_one_literal_token_is_refused(self, scope: str) -> None:
        with pytest.raises(ValueError, match="key_scope"):
            KVLease(FakeNatsClient(), bucket_name="leases", key_scope=scope)  # type: ignore[arg-type]


class TestAnEntryInAnotherFormatIsAnotherHolders:
    """a value the lease did not write is somebody's claim: never reclaimed, refreshed or deleted, never a crash.

    The NATS lock wrote its holder's raw ``token_hex(16)`` before it ran on this lease, and during a
    deploy those entries share keys with lease envelopes. A hex token can parse as JSON -- all digits
    is a number, digits around an ``e`` an exponent -- so "unreadable" has to cover those too.
    """

    @pytest.mark.parametrize(
        "foreign",
        [
            b"9f86d081884c7d659a2feaa0c55ad015",
            b"12345678901234567890123456789012",
            b"1234e567890123456789012345678901",
            b"1",
            b'["pod-alpha"]',
        ],
    )
    async def test_acquire_reports_it_held(self, foreign: bytes) -> None:
        lease, client = await _make_lease()
        bucket = await _bucket_for(client, "test_leases")
        await bucket.create(key="lock/a", value=foreign)
        with pytest.raises(LeaseUnavailable):
            await lease.acquire("lock/a", ttl_seconds=30, max_wait_seconds=0)
        assert await bucket.get(key="lock/a") == foreign

    async def test_refresh_reports_it_taken_and_release_leaves_it(self) -> None:
        lease, client = await _make_lease()
        handle = await lease.acquire("lock/a", ttl_seconds=30)
        bucket = await _bucket_for(client, "test_leases")
        assert await bucket.delete(key="lock/a", revision=handle.revision)
        await bucket.create(key="lock/a", value=b"12345678901234567890123456789012")
        with pytest.raises(LeaseLost) as lost:
            await handle.refresh()
        assert lost.value.reason is lease_module.LeaseLossReason.TAKEN
        await handle.release()
        assert await bucket.get(key="lock/a") == b"12345678901234567890123456789012"


class TestLeaseLostSaysWhy:
    async def test_a_missing_entry_is_expired(self) -> None:
        lease, client = await _make_lease()
        handle = await lease.acquire("lock/a", ttl_seconds=30)
        assert await (await _bucket_for(client, "test_leases")).delete(key="lock/a", revision=handle.revision)
        with pytest.raises(LeaseLost) as lost:
            await handle.refresh()
        assert lost.value.reason is lease_module.LeaseLossReason.EXPIRED

    def test_a_lease_lost_raised_without_a_reason_still_constructs(self) -> None:
        assert LeaseLost("gone").reason is None


class TestAWriteAppliedButNeverAnswered:
    """the server applied a refresh whose answer never arrived: the handle is one revision behind its own entry."""

    async def _unanswered_refresh(self) -> tuple[LeaseHandle, FakeKvBucket]:
        lease, client = await _make_lease()
        handle = await lease.acquire("lock/a", ttl_seconds=30)
        bucket = await _bucket_for(client, "test_leases")
        real_update = bucket.update

        async def applied_then_lost(**kwargs: Any) -> int | None:
            await real_update(**kwargs)
            bucket.update = real_update  # type: ignore[method-assign]
            raise TimeoutError("the answer never came")

        bucket.update = applied_then_lost  # type: ignore[method-assign]
        with pytest.raises(TimeoutError):
            await handle.refresh()
        return handle, bucket

    async def test_the_next_refresh_renews_from_its_own_write(self) -> None:
        handle, bucket = await self._unanswered_refresh()
        await handle.refresh()
        entry = await bucket.get_entry(key="lock/a")
        assert entry is not None and entry[1] == handle.revision

    async def test_a_refresh_refused_for_another_reason_keeps_the_record_for_the_release(self) -> None:
        """a later refresh that fails must not forget the unanswered write: the release still needs it."""
        handle, bucket = await self._unanswered_refresh()
        real_get_entry = bucket.get_entry

        async def unreachable(**kwargs: Any) -> tuple[bytes, int] | None:
            raise ConnectionError("kv is unreachable")

        bucket.get_entry = unreachable  # type: ignore[method-assign]
        with pytest.raises(ConnectionError):
            await handle.refresh()
        bucket.get_entry = real_get_entry  # type: ignore[method-assign]
        await handle.release()
        assert await bucket.get(key="lock/a") is None

    async def _write_that_lands_late(self) -> tuple[LeaseHandle, FakeKvBucket]:
        """a refresh whose write is sent, never answered, and lands only as the next refresh swaps."""
        lease, client = await _make_lease()
        handle = await lease.acquire("lock/a", ttl_seconds=30)
        bucket = await _bucket_for(client, "test_leases")
        real_update = bucket.update
        sent: dict[str, Any] = {}

        async def lands_before_this_swap(**kwargs: Any) -> int | None:
            bucket.update = real_update  # type: ignore[method-assign]
            await real_update(**sent)
            return await real_update(**kwargs)

        async def sent_and_never_answered(**kwargs: Any) -> int | None:
            sent.update(kwargs)
            bucket.update = lands_before_this_swap  # type: ignore[method-assign]
            raise TimeoutError("the answer never came")

        bucket.update = sent_and_never_answered  # type: ignore[method-assign]
        with pytest.raises(TimeoutError):
            await handle.refresh()
        return handle, bucket

    async def test_a_write_that_lands_between_the_next_refreshs_read_and_its_swap_is_renewed_from(self) -> None:
        """the read saw the entry before the unanswered write; the swap was refused because it landed."""
        handle, bucket = await self._write_that_lands_late()
        await handle.refresh()
        entry = await bucket.get_entry(key="lock/a")
        assert entry is not None and entry[1] == handle.revision

    async def test_a_refused_swap_whose_re_read_fails_keeps_the_record_for_the_release(self) -> None:
        """the swap was refused and the entry could not be read again: the unanswered write is still the
        handle's own, and the release deletes it."""
        handle, bucket = await self._write_that_lands_late()
        real_get_entry = bucket.get_entry
        reads = 0

        async def second_read_fails(**kwargs: Any) -> tuple[bytes, int] | None:
            nonlocal reads
            reads += 1
            if reads == 2:
                raise ConnectionError("kv is unreachable")
            return await real_get_entry(**kwargs)

        bucket.get_entry = second_read_fails  # type: ignore[method-assign]
        with pytest.raises(ConnectionError):
            await handle.refresh()
        bucket.get_entry = real_get_entry  # type: ignore[method-assign]
        await handle.release()
        assert await bucket.get(key="lock/a") is None

    async def test_another_holders_entry_is_still_a_lost_lease(self) -> None:
        handle, bucket = await self._unanswered_refresh()
        entry = await bucket.get_entry(key="lock/a")
        assert entry is not None
        stolen = entry[0].replace(b"pod-alpha", b"pod-thief")
        assert await bucket.update(key="lock/a", value=stolen, revision=entry[1]) is not None
        with pytest.raises(LeaseLost):
            await handle.refresh()
        await handle.release()
        assert await bucket.get(key="lock/a") == stolen
