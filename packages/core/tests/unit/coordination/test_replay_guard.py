"""tests for :class:`ReplayGuard`: shared, fail-closed single-use nonce protection.

The contract this pins:

- a fresh nonce records and returns True; the SAME nonce again returns False (replay);
- the cache is SHARED, not per-process: a nonce recorded by one guard instance is a replay for
  another instance over the same bucket (an in-process set would miss the cross-replica replay);
- it is FAIL-CLOSED: a KV transport failure propagates (never silently answers "fresh");
- the bucket is opened with the accept-window TTL so nonces self-expire;
- a non-positive TTL is rejected at construction (it would grow the bucket without bound).
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.core.coordination import ReplayGuard
from threetears.nats import KvError

from ._fake_kv import FakeNatsClient


@pytest.fixture
def client() -> FakeNatsClient:
    return FakeNatsClient()


class TestReplayGuard:
    @pytest.mark.asyncio
    async def test_fresh_nonce_is_recorded(self, client: FakeNatsClient) -> None:
        guard = ReplayGuard(client, bucket_name="pop_nonces", ttl_seconds=120)  # type: ignore[arg-type]
        assert await guard.record_unique("nonce-1") is True

    @pytest.mark.asyncio
    async def test_same_nonce_twice_is_replay(self, client: FakeNatsClient) -> None:
        guard = ReplayGuard(client, bucket_name="pop_nonces", ttl_seconds=120)  # type: ignore[arg-type]
        assert await guard.record_unique("n") is True
        assert await guard.record_unique("n") is False

    @pytest.mark.asyncio
    async def test_distinct_nonces_each_fresh(self, client: FakeNatsClient) -> None:
        guard = ReplayGuard(client, bucket_name="pop_nonces", ttl_seconds=120)  # type: ignore[arg-type]
        assert await guard.record_unique("a") is True
        assert await guard.record_unique("b") is True

    @pytest.mark.asyncio
    async def test_replay_detected_across_instances_sharing_a_bucket(
        self, client: FakeNatsClient
    ) -> None:
        # two guards == two replica processes against the SAME shared bucket; a nonce recorded by
        # one must be a replay for the other. an in-process set could never catch this.
        g1 = ReplayGuard(client, bucket_name="shared", ttl_seconds=120)  # type: ignore[arg-type]
        g2 = ReplayGuard(client, bucket_name="shared", ttl_seconds=120)  # type: ignore[arg-type]
        assert await g1.record_unique("x") is True
        assert await g2.record_unique("x") is False

    @pytest.mark.asyncio
    async def test_hashed_keys_keep_similar_nonces_distinct(self, client: FakeNatsClient) -> None:
        guard = ReplayGuard(client, bucket_name="b", ttl_seconds=120)  # type: ignore[arg-type]
        assert await guard.record_unique("nonce-1") is True
        assert await guard.record_unique("nonce-2") is True
        assert await guard.record_unique("nonce-1") is False  # the first is now a replay

    @pytest.mark.asyncio
    async def test_transport_failure_propagates_fail_closed(self) -> None:
        bucket = AsyncMock()
        bucket.create = AsyncMock(side_effect=KvError("kv down"))
        failing_client = AsyncMock()
        failing_client.kv_bucket = AsyncMock(return_value=bucket)
        guard = ReplayGuard(failing_client, bucket_name="b", ttl_seconds=120)
        with pytest.raises(KvError):
            await guard.record_unique("x")

    @pytest.mark.asyncio
    async def test_bucket_opened_with_the_accept_window_ttl(self) -> None:
        bucket = AsyncMock()
        bucket.create = AsyncMock(return_value=1)
        spy_client = AsyncMock()
        spy_client.kv_bucket = AsyncMock(return_value=bucket)
        guard = ReplayGuard(spy_client, bucket_name="b", ttl_seconds=90)
        await guard.record_unique("x")
        kwargs = spy_client.kv_bucket.call_args.kwargs
        assert kwargs["ttl"] == timedelta(seconds=90)
        # file storage so the cache survives a NATS restart (else self-heal recreates it empty and
        # admits replays as fresh for one window -- a fail-open hole).
        assert kwargs["storage"] == "file"

    @pytest.mark.asyncio
    async def test_bucket_bound_once_across_calls(self) -> None:
        bucket = AsyncMock()
        bucket.create = AsyncMock(return_value=1)
        spy_client = AsyncMock()
        spy_client.kv_bucket = AsyncMock(return_value=bucket)
        guard = ReplayGuard(spy_client, bucket_name="b", ttl_seconds=120)
        await guard.record_unique("a")
        await guard.record_unique("b")
        spy_client.kv_bucket.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_concurrent_record_of_same_nonce_has_one_winner(
        self, client: FakeNatsClient
    ) -> None:
        # CAS create-if-absent guarantees exactly one "fresh" even when the same nonce is recorded
        # concurrently -- e.g. a replay racing the original.
        guard = ReplayGuard(client, bucket_name="race", ttl_seconds=120)  # type: ignore[arg-type]
        results = await asyncio.gather(*[guard.record_unique("dup") for _ in range(8)])
        assert results.count(True) == 1
        assert results.count(False) == 7

    def test_non_positive_ttl_rejected(self) -> None:
        with pytest.raises(ValueError):
            ReplayGuard(MagicMock(), bucket_name="b", ttl_seconds=0)
        with pytest.raises(ValueError):
            ReplayGuard(MagicMock(), bucket_name="b", ttl_seconds=-5)
