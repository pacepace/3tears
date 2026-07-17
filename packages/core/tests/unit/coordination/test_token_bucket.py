"""tests for :class:`TokenBucket`: distributed token-bucket rate limiter over NATS KV.

the contract this pins:

- a fresh key starts at full capacity; a claim consumes tokens from it;
- refill is continuous (fractional elapsed time * refill_rate), capped at
  capacity;
- claim() never raises for "not enough tokens" -- it returns a
  TokenClaimResult with claimed=False and a computed retry_after_seconds;
- max_wait_seconds=0 (default) is non-blocking: single attempt, returns
  immediately;
- max_wait_seconds>0 blocks, retrying until claimed or the deadline elapses;
- claim is atomic under concurrency: N concurrent claimers against a bucket
  with exactly N tokens all succeed, no overcounting;
- independent keys never share tokens;
- claiming more tokens than capacity raises ValueError immediately (can
  never be satisfied).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from threetears.core.coordination import TokenBucket, TokenBucketConflict

from ._fake_kv import FakeNatsClient


@pytest.fixture
def client() -> FakeNatsClient:
    return FakeNatsClient()


class TestConstruction:
    def test_non_positive_refill_rate_raises(self, client: FakeNatsClient) -> None:
        with pytest.raises(ValueError, match="refill_rate"):
            TokenBucket(client, bucket_name="b", refill_rate=0, capacity=10)  # type: ignore[arg-type]

    def test_non_positive_capacity_raises(self, client: FakeNatsClient) -> None:
        with pytest.raises(ValueError, match="capacity"):
            TokenBucket(client, bucket_name="b", refill_rate=1.0, capacity=0)  # type: ignore[arg-type]


class TestClaim:
    @pytest.mark.asyncio
    async def test_fresh_key_claim_succeeds(self, client: FakeNatsClient) -> None:
        bucket = TokenBucket(client, bucket_name="b", refill_rate=1.0, capacity=5.0)  # type: ignore[arg-type]
        outcome = await bucket.claim("k")
        assert outcome.claimed is True
        assert outcome.tokens_remaining == 4.0
        assert outcome.retry_after_seconds == 0.0

    @pytest.mark.asyncio
    async def test_claim_exceeding_capacity_raises_value_error(self, client: FakeNatsClient) -> None:
        bucket = TokenBucket(client, bucket_name="b", refill_rate=1.0, capacity=5.0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="exceeds bucket capacity"):
            await bucket.claim("k", tokens=10.0)

    @pytest.mark.asyncio
    async def test_repeated_claims_drain_bucket(self, client: FakeNatsClient) -> None:
        bucket = TokenBucket(client, bucket_name="b", refill_rate=0.001, capacity=3.0)  # type: ignore[arg-type]
        first = await bucket.claim("k")
        second = await bucket.claim("k")
        third = await bucket.claim("k")
        fourth = await bucket.claim("k")

        assert [first.claimed, second.claimed, third.claimed] == [True, True, True]
        assert fourth.claimed is False
        assert fourth.tokens_remaining < 1.0

    @pytest.mark.asyncio
    async def test_non_blocking_claim_returns_false_without_waiting(self, client: FakeNatsClient) -> None:
        bucket = TokenBucket(client, bucket_name="b", refill_rate=0.001, capacity=1.0)  # type: ignore[arg-type]
        await bucket.claim("k")  # drain the single token

        start = time.monotonic()
        outcome = await bucket.claim("k")  # default max_wait_seconds=0.0
        elapsed = time.monotonic() - start

        assert outcome.claimed is False
        assert elapsed < 0.05  # returned immediately, did not sleep

    @pytest.mark.asyncio
    async def test_retry_after_seconds_reflects_shortfall_over_refill_rate(self, client: FakeNatsClient) -> None:
        bucket = TokenBucket(client, bucket_name="b", refill_rate=2.0, capacity=1.0)  # type: ignore[arg-type]
        await bucket.claim("k")  # drain to 0 tokens

        outcome = await bucket.claim("k")

        # shortfall is 1.0 token, refill_rate is 2.0/sec -> 0.5s to have enough
        assert outcome.retry_after_seconds == pytest.approx(0.5, abs=0.01)

    @pytest.mark.asyncio
    async def test_refill_over_elapsed_time_allows_later_claim(self, client: FakeNatsClient) -> None:
        bucket = TokenBucket(client, bucket_name="b", refill_rate=50.0, capacity=1.0)  # type: ignore[arg-type]
        await bucket.claim("k")  # drain to 0 tokens
        assert (await bucket.claim("k")).claimed is False

        await asyncio.sleep(0.05)  # 50 tokens/sec * 0.05s ~= 2.5 tokens refilled

        outcome = await bucket.claim("k")
        assert outcome.claimed is True

    @pytest.mark.asyncio
    async def test_refill_never_exceeds_capacity(self, client: FakeNatsClient) -> None:
        bucket = TokenBucket(client, bucket_name="b", refill_rate=1000.0, capacity=3.0)  # type: ignore[arg-type]
        await bucket.claim("k")  # tokens_remaining == 2.0
        await asyncio.sleep(0.05)  # would refill far past capacity if uncapped

        outcome = await bucket.claim("k")
        # capacity is 3.0; after this claim at most 2.0 tokens can remain
        assert outcome.tokens_remaining <= 2.0

    @pytest.mark.asyncio
    async def test_independent_keys_do_not_share_tokens(self, client: FakeNatsClient) -> None:
        bucket = TokenBucket(client, bucket_name="b", refill_rate=0.001, capacity=1.0)  # type: ignore[arg-type]
        await bucket.claim("a")  # drain key "a" only

        outcome_a = await bucket.claim("a")
        outcome_b = await bucket.claim("b")

        assert outcome_a.claimed is False
        assert outcome_b.claimed is True

    @pytest.mark.asyncio
    async def test_blocking_claim_waits_and_succeeds_after_refill(self, client: FakeNatsClient) -> None:
        bucket = TokenBucket(client, bucket_name="b", refill_rate=20.0, capacity=1.0)  # type: ignore[arg-type]
        await bucket.claim("k")  # drain to 0

        start = time.monotonic()
        outcome = await bucket.claim("k", max_wait_seconds=2.0)
        elapsed = time.monotonic() - start

        assert outcome.claimed is True
        assert elapsed < 2.0  # succeeded well before the deadline

    @pytest.mark.asyncio
    async def test_blocking_claim_returns_unclaimed_after_deadline(self, client: FakeNatsClient) -> None:
        bucket = TokenBucket(client, bucket_name="b", refill_rate=0.001, capacity=1.0)  # type: ignore[arg-type]
        await bucket.claim("k")  # drain to 0; refill_rate too slow to matter

        outcome = await bucket.claim("k", max_wait_seconds=0.1)

        assert outcome.claimed is False

    @pytest.mark.asyncio
    async def test_concurrent_claims_exactly_capacity_all_succeed_no_overcounting(self, client: FakeNatsClient) -> None:
        """N concurrent claimers against a bucket with exactly N tokens all succeed; a further claim fails."""
        bucket = TokenBucket(client, bucket_name="b", refill_rate=0.0001, capacity=20.0)  # type: ignore[arg-type]
        outcomes = await asyncio.gather(*(bucket.claim("k") for _ in range(20)))

        assert all(o.claimed for o in outcomes)

        one_more = await bucket.claim("k")
        assert one_more.claimed is False

    @pytest.mark.asyncio
    async def test_concurrent_claims_exceeding_capacity_some_rejected(self, client: FakeNatsClient) -> None:
        bucket = TokenBucket(client, bucket_name="b", refill_rate=0.0001, capacity=10.0)  # type: ignore[arg-type]
        outcomes = await asyncio.gather(*(bucket.claim("k") for _ in range(25)))

        claimed_count = sum(1 for o in outcomes if o.claimed)
        assert claimed_count == 10


class TestBucketConfiguration:
    @pytest.mark.asyncio
    async def test_bucket_opened_with_configured_kv_ttl(self) -> None:
        from datetime import timedelta
        from unittest.mock import AsyncMock

        bucket_mock = AsyncMock()
        bucket_mock.get_entry = AsyncMock(return_value=None)
        bucket_mock.create = AsyncMock(return_value=1)
        spy_client = AsyncMock()
        spy_client.kv_bucket = AsyncMock(return_value=bucket_mock)

        store = TokenBucket(spy_client, bucket_name="b", refill_rate=1.0, capacity=5.0, kv_ttl=timedelta(minutes=30))
        await store.claim("x")

        kwargs = spy_client.kv_bucket.call_args.kwargs
        assert kwargs["ttl"] == timedelta(minutes=30)

    @pytest.mark.asyncio
    async def test_default_kv_ttl_is_one_hour(self) -> None:
        from datetime import timedelta
        from unittest.mock import AsyncMock

        bucket_mock = AsyncMock()
        bucket_mock.get_entry = AsyncMock(return_value=None)
        bucket_mock.create = AsyncMock(return_value=1)
        spy_client = AsyncMock()
        spy_client.kv_bucket = AsyncMock(return_value=bucket_mock)

        store = TokenBucket(spy_client, bucket_name="b", refill_rate=1.0, capacity=5.0)
        await store.claim("x")

        kwargs = spy_client.kv_bucket.call_args.kwargs
        assert kwargs["ttl"] == timedelta(hours=1)

    @pytest.mark.asyncio
    async def test_bucket_bound_once_across_calls(self) -> None:
        from unittest.mock import AsyncMock

        bucket_mock = AsyncMock()
        bucket_mock.get_entry = AsyncMock(return_value=None)
        bucket_mock.create = AsyncMock(return_value=1)
        spy_client = AsyncMock()
        spy_client.kv_bucket = AsyncMock(return_value=bucket_mock)

        store = TokenBucket(spy_client, bucket_name="b", refill_rate=1.0, capacity=5.0)
        await store.claim("a")
        await store.claim("b")
        spy_client.kv_bucket.assert_awaited_once()


class TestCasContention:
    @pytest.mark.asyncio
    async def test_claim_retries_on_cas_conflict_then_succeeds(self) -> None:
        from unittest.mock import AsyncMock

        bucket_mock = AsyncMock()
        bucket_mock.get_entry = AsyncMock(
            side_effect=[
                (
                    b'{"tokens": 5.0, "last_refill": "2026-01-01T00:00:00+00:00"}',
                    1,
                ),
                (
                    b'{"tokens": 5.0, "last_refill": "2026-01-01T00:00:00+00:00"}',
                    2,
                ),
            ]
        )
        bucket_mock.update = AsyncMock(side_effect=[None, 3])  # first CAS attempt loses, second wins
        client = AsyncMock()
        client.kv_bucket = AsyncMock(return_value=bucket_mock)

        store = TokenBucket(client, bucket_name="b", refill_rate=1.0, capacity=10.0)
        outcome = await store.claim("k")

        assert outcome.claimed is True
        assert bucket_mock.update.await_count == 2

    @pytest.mark.asyncio
    async def test_claim_raises_when_cas_retry_budget_exhausted(self) -> None:
        from unittest.mock import AsyncMock

        bucket_mock = AsyncMock()
        bucket_mock.get_entry = AsyncMock(
            return_value=(
                b'{"tokens": 5.0, "last_refill": "2026-01-01T00:00:00+00:00"}',
                1,
            )
        )
        bucket_mock.update = AsyncMock(return_value=None)  # every CAS attempt loses
        client = AsyncMock()
        client.kv_bucket = AsyncMock(return_value=bucket_mock)

        store = TokenBucket(client, bucket_name="b", refill_rate=1.0, capacity=10.0)
        with pytest.raises(TokenBucketConflict):
            await store.claim("k")
