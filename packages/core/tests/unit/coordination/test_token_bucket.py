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

Time is driven, never waited for: every bucket here reads a :class:`_DrivenClock`, whose sleep
advances it. A refill or a deadline asserted against the wall clock depends on how fast the
machine happens to run -- a drained bucket at 50 tokens a second refills a whole token in 20ms,
which a loaded machine spends between two claims.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest

from threetears.core.coordination import TokenBucket, TokenBucketConflict

from threetears.core.testing.kv import FakeNatsClient


class _DrivenClock:
    """a UTC clock that moves only when a test, or a sleep, moves it."""

    def __init__(self) -> None:
        self.now = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
        self.slept: list[float] = []

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)

    async def sleep(self, seconds: float) -> None:
        """advance by ``seconds`` and yield once, as a real sleep would."""
        self.slept.append(seconds)
        self.advance(seconds)
        await asyncio.sleep(0)


@pytest.fixture
def client() -> FakeNatsClient:
    return FakeNatsClient()


@pytest.fixture
def clock() -> _DrivenClock:
    return _DrivenClock()


def _bucket(client: object, clock: _DrivenClock, *, refill_rate: float, capacity: float) -> TokenBucket:
    return TokenBucket(
        client,  # type: ignore[arg-type]
        bucket_name="b",
        refill_rate=refill_rate,
        capacity=capacity,
        clock=clock,
        sleep=clock.sleep,
    )


class TestConstruction:
    def test_non_positive_refill_rate_raises(self, client: FakeNatsClient) -> None:
        with pytest.raises(ValueError, match="refill_rate"):
            TokenBucket(client, bucket_name="b", refill_rate=0, capacity=10)

    def test_non_positive_capacity_raises(self, client: FakeNatsClient) -> None:
        with pytest.raises(ValueError, match="capacity"):
            TokenBucket(client, bucket_name="b", refill_rate=1.0, capacity=0)


class TestClaim:
    @pytest.mark.asyncio
    async def test_fresh_key_claim_succeeds(self, client: FakeNatsClient, clock: _DrivenClock) -> None:
        bucket = _bucket(client, clock, refill_rate=1.0, capacity=5.0)
        outcome = await bucket.claim("k")
        assert outcome.claimed is True
        assert outcome.tokens_remaining == 4.0
        assert outcome.retry_after_seconds == 0.0

    @pytest.mark.asyncio
    async def test_claim_exceeding_capacity_raises_value_error(
        self, client: FakeNatsClient, clock: _DrivenClock
    ) -> None:
        bucket = _bucket(client, clock, refill_rate=1.0, capacity=5.0)
        with pytest.raises(ValueError, match="exceeds bucket capacity"):
            await bucket.claim("k", tokens=10.0)

    @pytest.mark.asyncio
    async def test_repeated_claims_drain_bucket(self, client: FakeNatsClient, clock: _DrivenClock) -> None:
        bucket = _bucket(client, clock, refill_rate=0.001, capacity=3.0)
        first = await bucket.claim("k")
        second = await bucket.claim("k")
        third = await bucket.claim("k")
        fourth = await bucket.claim("k")

        assert [first.claimed, second.claimed, third.claimed] == [True, True, True]
        assert fourth.claimed is False
        assert fourth.tokens_remaining < 1.0

    @pytest.mark.asyncio
    async def test_non_blocking_claim_returns_false_without_waiting(
        self, client: FakeNatsClient, clock: _DrivenClock
    ) -> None:
        bucket = _bucket(client, clock, refill_rate=0.001, capacity=1.0)
        await bucket.claim("k")  # drain the single token

        outcome = await bucket.claim("k")  # default max_wait_seconds=0.0

        assert outcome.claimed is False
        assert clock.slept == [], "a non-blocking claim slept"

    @pytest.mark.asyncio
    async def test_retry_after_seconds_reflects_shortfall_over_refill_rate(
        self, client: FakeNatsClient, clock: _DrivenClock
    ) -> None:
        bucket = _bucket(client, clock, refill_rate=2.0, capacity=1.0)
        await bucket.claim("k")  # drain to 0 tokens

        outcome = await bucket.claim("k")

        # shortfall is 1.0 token, refill_rate is 2.0/sec -> 0.5s to have enough
        assert outcome.retry_after_seconds == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_refill_over_elapsed_time_allows_later_claim(
        self, client: FakeNatsClient, clock: _DrivenClock
    ) -> None:
        bucket = _bucket(client, clock, refill_rate=50.0, capacity=1.0)
        await bucket.claim("k")  # drain to 0 tokens
        assert (await bucket.claim("k")).claimed is False

        clock.advance(0.01)  # 50 tokens/sec * 0.01s = 0.5 tokens: not yet enough
        assert (await bucket.claim("k")).claimed is False

        clock.advance(0.01)  # 1.0 token refilled in all

        outcome = await bucket.claim("k")
        assert outcome.claimed is True

    @pytest.mark.asyncio
    async def test_refill_never_exceeds_capacity(self, client: FakeNatsClient, clock: _DrivenClock) -> None:
        bucket = _bucket(client, clock, refill_rate=1000.0, capacity=3.0)
        await bucket.claim("k")  # tokens_remaining == 2.0
        clock.advance(0.05)  # 50 tokens' worth: far past capacity if uncapped

        outcome = await bucket.claim("k")
        # refilled to capacity 3.0, not past it; this claim leaves exactly 2.0
        assert outcome.tokens_remaining == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_independent_keys_do_not_share_tokens(self, client: FakeNatsClient, clock: _DrivenClock) -> None:
        bucket = _bucket(client, clock, refill_rate=0.001, capacity=1.0)
        await bucket.claim("a")  # drain key "a" only

        outcome_a = await bucket.claim("a")
        outcome_b = await bucket.claim("b")

        assert outcome_a.claimed is False
        assert outcome_b.claimed is True

    @pytest.mark.asyncio
    async def test_blocking_claim_waits_and_succeeds_after_refill(
        self, client: FakeNatsClient, clock: _DrivenClock
    ) -> None:
        bucket = _bucket(client, clock, refill_rate=20.0, capacity=1.0)
        await bucket.claim("k")  # drain to 0

        outcome = await bucket.claim("k", max_wait_seconds=2.0)

        assert outcome.claimed is True
        # one token at 20/sec is 0.05s away: it slept exactly that, well inside the deadline
        assert clock.slept == [pytest.approx(0.05)]

    @pytest.mark.asyncio
    async def test_blocking_claim_returns_unclaimed_after_deadline(
        self, client: FakeNatsClient, clock: _DrivenClock
    ) -> None:
        bucket = _bucket(client, clock, refill_rate=0.001, capacity=1.0)
        await bucket.claim("k")  # drain to 0; refill_rate too slow to matter

        outcome = await bucket.claim("k", max_wait_seconds=0.1)

        assert outcome.claimed is False
        assert sum(clock.slept) == pytest.approx(0.1), "the claim did not wait out its whole deadline"

    @pytest.mark.asyncio
    async def test_concurrent_claims_exactly_capacity_all_succeed_no_overcounting(
        self, client: FakeNatsClient, clock: _DrivenClock
    ) -> None:
        """N concurrent claimers against a bucket with exactly N tokens all succeed; a further claim fails."""
        bucket = _bucket(client, clock, refill_rate=0.0001, capacity=20.0)
        outcomes = await asyncio.gather(*(bucket.claim("k") for _ in range(20)))

        assert all(o.claimed for o in outcomes)

        one_more = await bucket.claim("k")
        assert one_more.claimed is False

    @pytest.mark.asyncio
    async def test_concurrent_claims_exceeding_capacity_some_rejected(
        self, client: FakeNatsClient, clock: _DrivenClock
    ) -> None:
        bucket = _bucket(client, clock, refill_rate=0.0001, capacity=10.0)
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
    async def test_claim_retries_on_cas_conflict_then_succeeds(self, clock: _DrivenClock) -> None:
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

        store = _bucket(client, clock, refill_rate=1.0, capacity=10.0)
        outcome = await store.claim("k")

        assert outcome.claimed is True
        assert bucket_mock.update.await_count == 2

    @pytest.mark.asyncio
    async def test_claim_raises_when_cas_retry_budget_exhausted(self, clock: _DrivenClock) -> None:
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

        store = _bucket(client, clock, refill_rate=1.0, capacity=10.0)
        with pytest.raises(TokenBucketConflict):
            await store.claim("k")


class TestRefund:
    """Returning a turn taken for work that never happened.

    Without this, `claim` consumed and the only recovery was refill over time -- so a caller
    cancelled between claiming and doing the work held the bucket down for as long as the
    refill rate took to make it up. Invisible once, compounding under repeated cancellation.
    """

    @pytest.mark.asyncio
    async def test_a_refund_returns_the_tokens_it_was_given(self, client: FakeNatsClient, clock: _DrivenClock) -> None:
        bucket = _bucket(client, clock, refill_rate=0.0001, capacity=5.0)
        first = await bucket.claim("k", tokens=3.0)
        assert first.claimed

        remaining = await bucket.refund("k", tokens=3.0)

        assert remaining == pytest.approx(5.0, abs=0.01), "the claimed tokens did not come back"

    @pytest.mark.asyncio
    async def test_a_refund_cannot_mint_budget_above_capacity(
        self, client: FakeNatsClient, clock: _DrivenClock
    ) -> None:
        """Safe to call unconditionally from a handler that may not have claimed at all."""
        bucket = _bucket(client, clock, refill_rate=0.0001, capacity=5.0)
        await bucket.claim("k", tokens=1.0)

        await bucket.refund("k", tokens=1.0)
        remaining = await bucket.refund("k", tokens=1.0)

        assert remaining == pytest.approx(5.0, abs=0.01), "a second refund invented budget the bucket never had"

    @pytest.mark.asyncio
    async def test_a_refund_makes_a_denied_claim_succeed_again(
        self, client: FakeNatsClient, clock: _DrivenClock
    ) -> None:
        """The behaviour that matters: the next caller is not made to wait for refill."""
        bucket = _bucket(client, clock, refill_rate=0.0001, capacity=1.0)
        assert (await bucket.claim("k")).claimed
        assert not (await bucket.claim("k")).claimed, "the bucket should be empty"

        await bucket.refund("k")

        assert (await bucket.claim("k")).claimed, "the refunded turn was not usable by the next caller"

    @pytest.mark.asyncio
    async def test_a_refund_on_an_untouched_key_invents_nothing(
        self, client: FakeNatsClient, clock: _DrivenClock
    ) -> None:
        """No key means nothing was consumed from it, so there is nothing to give back."""
        bucket = _bucket(client, clock, refill_rate=1.0, capacity=5.0)

        assert await bucket.refund("never-claimed") == pytest.approx(5.0)
        assert (await bucket.claim("never-claimed", tokens=5.0)).claimed

    @pytest.mark.asyncio
    async def test_a_refund_never_raises_into_a_caller_that_is_unwinding(self) -> None:
        """It runs from an exception handler, so raising would lose the original error.

        A failed refund costs throughput that self-heals; an exception escaping here replaces a
        recoverable dip with a lost traceback. The KV is down at the client, so opening the
        bucket is what fails.
        """
        from unittest.mock import AsyncMock

        client = AsyncMock()
        client.kv_bucket = AsyncMock(side_effect=RuntimeError("kv is down"))
        bucket = TokenBucket(client, bucket_name="b", refill_rate=1.0, capacity=5.0)

        assert await bucket.refund("k") == -1.0, "a broken refund must report failure, not raise"
        client.kv_bucket.assert_awaited()

    @pytest.mark.asyncio
    async def test_a_refund_retries_a_lost_cas_race(self, clock: _DrivenClock) -> None:
        """Another claimer touching the key mid-refund must not lose the returned tokens.

        Mocked at the client, matching `TestCasContention` above, so the test reaches no
        private attribute to observe the retry.
        """
        from unittest.mock import AsyncMock

        # last_refill is the clock's own now, so continuous refill contributes nothing and the
        # assertion is about the refund. Dated 2026-01-01 the bucket has months of refill behind
        # it and sits at capacity regardless, which would have made this pass for the wrong
        # reason -- or, as it did, fail against a correct implementation.
        now = clock.now.isoformat()
        entry = (f'{{"tokens": 3.0, "last_refill": "{now}"}}'.encode(), 1)
        bucket_mock = AsyncMock()
        bucket_mock.get_entry = AsyncMock(side_effect=[entry, entry])
        bucket_mock.update = AsyncMock(side_effect=[None, 3])  # first CAS loses, second wins
        client = AsyncMock()
        client.kv_bucket = AsyncMock(return_value=bucket_mock)

        store = _bucket(client, clock, refill_rate=0.0001, capacity=10.0)
        remaining = await store.refund("k", tokens=2.0)

        assert bucket_mock.update.await_count == 2, "the lost CAS race was not retried"
        assert remaining == pytest.approx(5.0, abs=0.01), "the refund was dropped by the race"

    @pytest.mark.asyncio
    async def test_exhausting_the_cas_budget_reports_rather_than_raising(self, clock: _DrivenClock) -> None:
        """`claim` raises `TokenBucketConflict` here; `refund` deliberately does not.

        The contracts differ because the callers do: a claim that cannot be made must stop its
        caller, while a refund runs from a handler already unwinding -- raising there would
        replace a self-healing throughput dip with a lost original error.
        """
        from unittest.mock import AsyncMock

        now = clock.now.isoformat()
        entry = (f'{{"tokens": 3.0, "last_refill": "{now}"}}'.encode(), 1)
        bucket_mock = AsyncMock()
        bucket_mock.get_entry = AsyncMock(return_value=entry)
        bucket_mock.update = AsyncMock(return_value=None)  # every CAS attempt loses
        client = AsyncMock()
        client.kv_bucket = AsyncMock(return_value=bucket_mock)

        store = _bucket(client, clock, refill_rate=0.0001, capacity=10.0)

        assert await store.refund("k", tokens=2.0) == -1.0, "exhaustion must report, not raise"


class TestOwnerScopedTokenBucketKeys:
    """a token bucket over a SHARED KV bucket keys every bucket state under its owner's scope."""

    _SCOPE = "agent_pod-019470a8b5c37def81230000000000aa"

    async def test_a_claim_and_a_refund_land_under_the_scope(self) -> None:
        client = FakeNatsClient()
        bucket = TokenBucket(
            client,  # type: ignore[arg-type]
            bucket_name="ratelimits",
            refill_rate=1.0,
            capacity=2.0,
            key_scope=self._SCOPE,
        )
        assert (await bucket.claim("rate_limit.llm.global")).claimed is True
        kv = await client.kv_bucket(name="ratelimits")
        assert await kv.get(key=f"{self._SCOPE}.rate_limit.llm.global") is not None
        assert await kv.get(key="rate_limit.llm.global") is None
        await bucket.refund("rate_limit.llm.global")
        assert await kv.get(key="rate_limit.llm.global") is None

    @pytest.mark.parametrize("scope", ["", "a.b", "a*", ">"])
    def test_a_scope_that_is_not_one_literal_token_is_refused(self, scope: str) -> None:
        with pytest.raises(ValueError, match="key_scope"):
            TokenBucket(
                FakeNatsClient(),  # type: ignore[arg-type]
                bucket_name="ratelimits",
                refill_rate=1.0,
                capacity=2.0,
                key_scope=scope,
            )


class TestStoredForm:
    """the bucket's state is stored with its instant in the one form every storage tier writes."""

    @pytest.mark.asyncio
    async def test_last_refill_is_stored_fixed_width_with_its_offset(
        self, client: FakeNatsClient, clock: _DrivenClock
    ) -> None:
        await _bucket(client, clock, refill_rate=1.0, capacity=5.0).claim("k")

        value = await (await client.kv_bucket(name="b")).get(key="k")

        assert value is not None
        assert json.loads(value)["last_refill"] == "2026-09-29T12:00:00.000000+00:00"

    @pytest.mark.asyncio
    async def test_a_naive_clock_is_refused_naming_the_field(self, client: FakeNatsClient) -> None:
        naive = TokenBucket(
            client,  # type: ignore[arg-type]
            bucket_name="b",
            refill_rate=1.0,
            capacity=5.0,
            clock=lambda: datetime(2026, 9, 29, 12, 0),
        )
        with pytest.raises(ValueError, match="naive datetime in 'last_refill'"):
            await naive.claim("k")
