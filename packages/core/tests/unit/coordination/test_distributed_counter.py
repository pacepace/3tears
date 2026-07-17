"""tests for :class:`DistributedCounter`: atomic increment/decrement counter over NATS KV.

the contract this pins:

- increment() creates the key at delta if absent, else atomically adds
  delta to the existing value;
- decrement() is a convenience for increment(key, delta=-delta);
- get() returns 0 for an absent key, else the current value;
- the primitive does not clamp at zero -- decrementing below zero is
  allowed (surfaces caller bugs rather than masking them);
- increment/decrement is atomic under concurrency: N concurrent
  incrementers against a fresh key sum to exactly N, no lost updates;
- independent keys never share counts.
"""

from __future__ import annotations

import asyncio

import pytest

from threetears.core.coordination import DistributedCounter, DistributedCounterConflict

from ._fake_kv import FakeNatsClient


@pytest.fixture
def client() -> FakeNatsClient:
    return FakeNatsClient()


class TestIncrement:
    @pytest.mark.asyncio
    async def test_increment_creates_key_at_delta_when_absent(self, client: FakeNatsClient) -> None:
        counter = DistributedCounter(client, bucket_name="c")  # type: ignore[arg-type]
        result = await counter.increment("k")
        assert result == 1

    @pytest.mark.asyncio
    async def test_increment_with_custom_delta(self, client: FakeNatsClient) -> None:
        counter = DistributedCounter(client, bucket_name="c")  # type: ignore[arg-type]
        result = await counter.increment("k", delta=5)
        assert result == 5

    @pytest.mark.asyncio
    async def test_repeated_increment_accumulates(self, client: FakeNatsClient) -> None:
        counter = DistributedCounter(client, bucket_name="c")  # type: ignore[arg-type]
        await counter.increment("k")
        await counter.increment("k")
        result = await counter.increment("k")
        assert result == 3

    @pytest.mark.asyncio
    async def test_independent_keys_do_not_share_counts(self, client: FakeNatsClient) -> None:
        counter = DistributedCounter(client, bucket_name="c")  # type: ignore[arg-type]
        await counter.increment("a")
        await counter.increment("a")
        await counter.increment("b")

        assert await counter.get("a") == 2
        assert await counter.get("b") == 1


class TestDecrement:
    @pytest.mark.asyncio
    async def test_decrement_subtracts_from_existing_value(self, client: FakeNatsClient) -> None:
        counter = DistributedCounter(client, bucket_name="c")  # type: ignore[arg-type]
        await counter.increment("k", delta=5)
        result = await counter.decrement("k")
        assert result == 4

    @pytest.mark.asyncio
    async def test_decrement_with_custom_delta(self, client: FakeNatsClient) -> None:
        counter = DistributedCounter(client, bucket_name="c")  # type: ignore[arg-type]
        await counter.increment("k", delta=10)
        result = await counter.decrement("k", delta=3)
        assert result == 7

    @pytest.mark.asyncio
    async def test_decrement_below_zero_is_not_clamped(self, client: FakeNatsClient) -> None:
        counter = DistributedCounter(client, bucket_name="c")  # type: ignore[arg-type]
        result = await counter.decrement("k")
        assert result == -1

    @pytest.mark.asyncio
    async def test_concurrent_increment_then_decrement_pattern_stays_race_free(self, client: FakeNatsClient) -> None:
        """mirrors RateLimitingCallback._check_concurrent_limit's own increment-then-compensate pattern."""
        counter = DistributedCounter(client, bucket_name="c")  # type: ignore[arg-type]
        limit = 5

        async def try_acquire_slot() -> bool:
            new_value = await counter.increment("inflight")
            if new_value > limit:
                await counter.decrement("inflight")
                return False
            return True

        results = await asyncio.gather(*(try_acquire_slot() for _ in range(12)))

        assert sum(1 for r in results if r) == limit
        assert await counter.get("inflight") == limit


class TestGet:
    @pytest.mark.asyncio
    async def test_get_returns_zero_for_absent_key(self, client: FakeNatsClient) -> None:
        counter = DistributedCounter(client, bucket_name="c")  # type: ignore[arg-type]
        assert await counter.get("never-seen") == 0

    @pytest.mark.asyncio
    async def test_get_returns_current_value(self, client: FakeNatsClient) -> None:
        counter = DistributedCounter(client, bucket_name="c")  # type: ignore[arg-type]
        await counter.increment("k", delta=7)
        assert await counter.get("k") == 7


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_increments_sum_exactly_no_lost_updates(self, client: FakeNatsClient) -> None:
        counter = DistributedCounter(client, bucket_name="c")  # type: ignore[arg-type]
        num_concurrent = 30

        await asyncio.gather(*(counter.increment("k") for _ in range(num_concurrent)))

        assert await counter.get("k") == num_concurrent

    @pytest.mark.asyncio
    async def test_concurrent_mixed_increment_decrement_nets_correctly(self, client: FakeNatsClient) -> None:
        counter = DistributedCounter(client, bucket_name="c")  # type: ignore[arg-type]

        await asyncio.gather(
            *(counter.increment("k") for _ in range(20)),
            *(counter.decrement("k") for _ in range(8)),
        )

        assert await counter.get("k") == 12


class TestBucketConfiguration:
    @pytest.mark.asyncio
    async def test_bucket_opened_with_configured_ttl(self) -> None:
        from datetime import timedelta
        from unittest.mock import AsyncMock

        bucket_mock = AsyncMock()
        bucket_mock.get_entry = AsyncMock(return_value=None)
        bucket_mock.create = AsyncMock(return_value=1)
        spy_client = AsyncMock()
        spy_client.kv_bucket = AsyncMock(return_value=bucket_mock)

        counter = DistributedCounter(spy_client, bucket_name="c", ttl=timedelta(minutes=2))
        await counter.increment("x")

        kwargs = spy_client.kv_bucket.call_args.kwargs
        assert kwargs["ttl"] == timedelta(minutes=2)

    @pytest.mark.asyncio
    async def test_default_ttl_is_none(self) -> None:
        from unittest.mock import AsyncMock

        bucket_mock = AsyncMock()
        bucket_mock.get_entry = AsyncMock(return_value=None)
        bucket_mock.create = AsyncMock(return_value=1)
        spy_client = AsyncMock()
        spy_client.kv_bucket = AsyncMock(return_value=bucket_mock)

        counter = DistributedCounter(spy_client, bucket_name="c")
        await counter.increment("x")

        kwargs = spy_client.kv_bucket.call_args.kwargs
        assert kwargs["ttl"] is None

    @pytest.mark.asyncio
    async def test_bucket_bound_once_across_calls(self) -> None:
        from unittest.mock import AsyncMock

        bucket_mock = AsyncMock()
        bucket_mock.get_entry = AsyncMock(return_value=None)
        bucket_mock.create = AsyncMock(return_value=1)
        spy_client = AsyncMock()
        spy_client.kv_bucket = AsyncMock(return_value=bucket_mock)

        counter = DistributedCounter(spy_client, bucket_name="c")
        await counter.increment("a")
        await counter.increment("b")
        spy_client.kv_bucket.assert_awaited_once()


class TestCasContention:
    @pytest.mark.asyncio
    async def test_increment_retries_on_cas_conflict_then_succeeds(self) -> None:
        from unittest.mock import AsyncMock

        bucket_mock = AsyncMock()
        bucket_mock.get_entry = AsyncMock(
            side_effect=[
                (b'{"value": 3}', 1),
                (b'{"value": 3}', 2),
            ]
        )
        bucket_mock.update = AsyncMock(side_effect=[None, 5])  # first CAS attempt loses, second wins
        client = AsyncMock()
        client.kv_bucket = AsyncMock(return_value=bucket_mock)

        counter = DistributedCounter(client, bucket_name="c")
        result = await counter.increment("k")

        assert result == 4
        assert bucket_mock.update.await_count == 2

    @pytest.mark.asyncio
    async def test_increment_raises_when_cas_retry_budget_exhausted(self) -> None:
        from unittest.mock import AsyncMock

        bucket_mock = AsyncMock()
        bucket_mock.get_entry = AsyncMock(return_value=(b'{"value": 3}', 1))
        bucket_mock.update = AsyncMock(return_value=None)  # every CAS attempt loses
        client = AsyncMock()
        client.kv_bucket = AsyncMock(return_value=bucket_mock)

        counter = DistributedCounter(client, bucket_name="c")
        with pytest.raises(DistributedCounterConflict):
            await counter.increment("k")

    @pytest.mark.asyncio
    async def test_create_retries_on_cas_conflict_for_fresh_key(self) -> None:
        from unittest.mock import AsyncMock

        bucket_mock = AsyncMock()
        # first get_entry misses (key looks absent), but another writer wins
        # the create race; second get_entry sees their write and we update.
        bucket_mock.get_entry = AsyncMock(side_effect=[None, (b'{"value": 1}', 1)])
        bucket_mock.create = AsyncMock(return_value=None)  # lost race to another creator
        bucket_mock.update = AsyncMock(return_value=2)
        client = AsyncMock()
        client.kv_bucket = AsyncMock(return_value=bucket_mock)

        counter = DistributedCounter(client, bucket_name="c")
        result = await counter.increment("k")

        assert result == 2
