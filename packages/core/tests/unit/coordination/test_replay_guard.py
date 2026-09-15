"""tests for :class:`ReplayGuard`: shared, fail-closed single-use nonce protection.

The contract this pins:

- a fresh nonce records and returns True; the SAME nonce again returns False (replay);
- the cache is SHARED, not per-process: a nonce recorded by one guard instance is a replay for
  another instance over the same bucket (an in-process set would miss the cross-replica replay);
- it is FAIL-CLOSED: a KV transport failure propagates (never silently answers "fresh");
- a WIPED bucket fails closed too: an artifact issued before the bucket's current creation time,
  allowing for the configured clock skew, is refused even though its nonce is not recorded;
- the bucket is memory-backed and opened with the accept-window TTL so nonces self-expire;
- construction rejects a non-positive TTL and a negative clock skew; a naive issue time raises.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.core.coordination import ReplayGuard
from threetears.nats import KvError

from threetears.core.testing.kv import FakeNatsClient

_SKEW = timedelta(seconds=60)


def _later() -> datetime:
    """an issue time comfortably after any bucket the fake created during this test.

    :return: now plus twice the skew
    :rtype: datetime
    """
    return datetime.now(UTC) + 2 * _SKEW


def _guard(client: object, *, bucket_name: str = "pop_nonces", ttl_seconds: int = 120) -> ReplayGuard:
    """build a guard with the test skew.

    :param client: the KV-capable client double
    :ptype client: object
    :param bucket_name: bucket suffix
    :ptype bucket_name: str
    :param ttl_seconds: nonce TTL
    :ptype ttl_seconds: int
    :return: the guard
    :rtype: ReplayGuard
    """
    return ReplayGuard(client, bucket_name=bucket_name, ttl_seconds=ttl_seconds, max_clock_skew=_SKEW)  # type: ignore[arg-type]


@pytest.fixture
def client() -> FakeNatsClient:
    return FakeNatsClient()


class TestReplayGuard:
    @pytest.mark.asyncio
    async def test_fresh_nonce_is_recorded(self, client: FakeNatsClient) -> None:
        assert await _guard(client).record_unique("nonce-1", issued_at=_later()) is True

    @pytest.mark.asyncio
    async def test_same_nonce_twice_is_replay(self, client: FakeNatsClient) -> None:
        guard = _guard(client)
        assert await guard.record_unique("n", issued_at=_later()) is True
        assert await guard.record_unique("n", issued_at=_later()) is False

    @pytest.mark.asyncio
    async def test_distinct_nonces_each_fresh(self, client: FakeNatsClient) -> None:
        guard = _guard(client)
        assert await guard.record_unique("a", issued_at=_later()) is True
        assert await guard.record_unique("b", issued_at=_later()) is True

    @pytest.mark.asyncio
    async def test_replay_detected_across_instances_sharing_a_bucket(self, client: FakeNatsClient) -> None:
        # two guards == two replica processes against the SAME shared bucket; a nonce recorded by
        # one must be a replay for the other. an in-process set could never catch this.
        g1 = _guard(client, bucket_name="shared")
        g2 = _guard(client, bucket_name="shared")
        assert await g1.record_unique("x", issued_at=_later()) is True
        assert await g2.record_unique("x", issued_at=_later()) is False

    @pytest.mark.asyncio
    async def test_hashed_keys_keep_similar_nonces_distinct(self, client: FakeNatsClient) -> None:
        guard = _guard(client, bucket_name="b")
        assert await guard.record_unique("nonce-1", issued_at=_later()) is True
        assert await guard.record_unique("nonce-2", issued_at=_later()) is True
        assert await guard.record_unique("nonce-1", issued_at=_later()) is False  # the first is now a replay

    @pytest.mark.asyncio
    async def test_transport_failure_propagates_fail_closed(self) -> None:
        bucket = AsyncMock()
        bucket.create = AsyncMock(side_effect=KvError("kv down"))
        failing_client = AsyncMock()
        failing_client.kv_bucket = AsyncMock(return_value=bucket)
        guard = _guard(failing_client, bucket_name="b")
        with pytest.raises(KvError):
            await guard.record_unique("x", issued_at=_later())

    @pytest.mark.asyncio
    async def test_creation_time_read_failure_propagates_fail_closed(self) -> None:
        # the nonce was recorded, but without the creation time the guard cannot rule out a wipe:
        # that must deny, never admit.
        bucket = AsyncMock()
        bucket.create = AsyncMock(return_value=1)
        bucket.date_created = AsyncMock(side_effect=KvError("no creation time"))
        failing_client = AsyncMock()
        failing_client.kv_bucket = AsyncMock(return_value=bucket)
        guard = _guard(failing_client, bucket_name="b")
        with pytest.raises(KvError):
            await guard.record_unique("x", issued_at=_later())

    @pytest.mark.asyncio
    async def test_bucket_opened_memory_backed_with_the_accept_window_ttl(self) -> None:
        bucket = AsyncMock()
        bucket.create = AsyncMock(return_value=1)
        bucket.date_created = AsyncMock(return_value=datetime.now(UTC))
        spy_client = AsyncMock()
        spy_client.kv_bucket = AsyncMock(return_value=bucket)
        guard = _guard(spy_client, bucket_name="b", ttl_seconds=90)
        await guard.record_unique("x", issued_at=_later())
        kwargs = spy_client.kv_bucket.call_args.kwargs
        assert kwargs["ttl"] == timedelta(seconds=90)
        # NATS is L2 and memory-only; a wipe is detected by the creation-time check, not survived.
        assert kwargs.get("storage", "memory") == "memory"

    @pytest.mark.asyncio
    async def test_bucket_bound_once_across_calls(self) -> None:
        bucket = AsyncMock()
        bucket.create = AsyncMock(return_value=1)
        bucket.date_created = AsyncMock(return_value=datetime.now(UTC))
        spy_client = AsyncMock()
        spy_client.kv_bucket = AsyncMock(return_value=bucket)
        guard = _guard(spy_client, bucket_name="b")
        await guard.record_unique("a", issued_at=_later())
        await guard.record_unique("b", issued_at=_later())
        spy_client.kv_bucket.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_replay_is_refused_without_reading_the_creation_time(self) -> None:
        # a present nonce is already a refusal; paying a second round trip for it is waste.
        bucket = AsyncMock()
        bucket.create = AsyncMock(return_value=None)
        bucket.date_created = AsyncMock(return_value=datetime.now(UTC))
        spy_client = AsyncMock()
        spy_client.kv_bucket = AsyncMock(return_value=bucket)
        guard = _guard(spy_client, bucket_name="b")
        assert await guard.record_unique("x", issued_at=_later()) is False
        bucket.date_created.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_concurrent_record_of_same_nonce_has_one_winner(self, client: FakeNatsClient) -> None:
        # CAS create-if-absent guarantees exactly one "fresh" even when the same nonce is recorded
        # concurrently -- e.g. a replay racing the original.
        guard = _guard(client, bucket_name="race")
        issued_at = _later()
        results = await asyncio.gather(*[guard.record_unique("dup", issued_at=issued_at) for _ in range(8)])
        assert results.count(True) == 1
        assert results.count(False) == 7

    def test_non_positive_ttl_rejected(self) -> None:
        with pytest.raises(ValueError):
            ReplayGuard(MagicMock(), bucket_name="b", ttl_seconds=0, max_clock_skew=_SKEW)
        with pytest.raises(ValueError):
            ReplayGuard(MagicMock(), bucket_name="b", ttl_seconds=-5, max_clock_skew=_SKEW)

    def test_negative_clock_skew_rejected(self) -> None:
        with pytest.raises(ValueError):
            ReplayGuard(MagicMock(), bucket_name="b", ttl_seconds=60, max_clock_skew=timedelta(seconds=-1))

    @pytest.mark.asyncio
    async def test_naive_issued_at_rejected(self, client: FakeNatsClient) -> None:
        with pytest.raises(ValueError):
            await _guard(client).record_unique("x", issued_at=datetime.now())  # noqa: DTZ005 - the naive value is the case under test


class TestReplayGuardAfterAWipe:
    """a wiped bucket must not become a window in which a spent artifact is fresh again."""

    @pytest.mark.asyncio
    async def test_unseen_artifact_issued_before_the_wipe_is_refused(self, client: FakeNatsClient) -> None:
        # the guard cannot tell a never-seen artifact from one whose record the wipe erased, so
        # anything issued before the bucket's creation time is refused.
        bucket = await client.kv_bucket(name="pop_nonces")
        issued_at = datetime.now(UTC)
        bucket.wipe(date_created=issued_at + timedelta(seconds=1))
        assert await _guard(client).record_unique("unseen", issued_at=issued_at) is False

    @pytest.mark.asyncio
    async def test_a_replay_of_an_admitted_artifact_is_refused_across_a_wipe(self, client: FakeNatsClient) -> None:
        # the realistic sequence: the bucket exists, a proof is admitted, the broker restarts, and
        # the SAME proof is presented again through a handle that never saw an error.
        bucket = await client.kv_bucket(name="pop_nonces")
        created = datetime.now(UTC)
        bucket.wipe(date_created=created)
        guard = _guard(client)
        issued_at = created + 2 * _SKEW
        assert await guard.record_unique("proof-1", issued_at=issued_at) is True
        bucket.wipe(date_created=issued_at + timedelta(seconds=30))
        assert await guard.record_unique("proof-1", issued_at=issued_at) is False

    @pytest.mark.asyncio
    async def test_artifact_within_the_skew_of_the_wipe_is_refused(self, client: FakeNatsClient) -> None:
        # an issue time just after the creation time could still be a replay stamped by a clock
        # running ahead; inside the skew it must be refused.
        bucket = await client.kv_bucket(name="pop_nonces")
        created = datetime.now(UTC)
        bucket.wipe(date_created=created)
        guard = _guard(client)
        assert await guard.record_unique("n", issued_at=created + _SKEW - timedelta(seconds=1)) is False

    @pytest.mark.asyncio
    async def test_artifact_issued_after_the_wipe_plus_skew_is_fresh(self, client: FakeNatsClient) -> None:
        bucket = await client.kv_bucket(name="pop_nonces")
        created = datetime.now(UTC)
        bucket.wipe(date_created=created)
        guard = _guard(client)
        assert await guard.record_unique("n", issued_at=created + _SKEW) is True

    @pytest.mark.asyncio
    async def test_a_refused_artifact_stays_refused(self, client: FakeNatsClient) -> None:
        # the refusal records the nonce, so presenting the same artifact later -- once the skew
        # has passed -- is a plain replay, not a second chance.
        bucket = await client.kv_bucket(name="pop_nonces")
        created = datetime.now(UTC)
        bucket.wipe(date_created=created)
        guard = _guard(client)
        stamped = created + timedelta(seconds=1)
        assert await guard.record_unique("n", issued_at=stamped) is False
        assert await guard.record_unique("n", issued_at=created + 2 * _SKEW) is False
