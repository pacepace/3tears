"""Choosing a key: sticky selection, moves, priority pull-back, skip versus move, period rollover, the
pool's own retries of a rate limit, concurrency, and pools sharing one store (KP-D, KP-E2, KP-F, KP-G4,
KP-G6, KP-H4)."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from threetears.models.key_pool import (
    InMemoryKeyPoolStore,
    KeyPool,
    KeyPoolLease,
    KeySpec,
    MoveReason,
    NoKeyAvailableError,
    Secret,
    VerdictScope,
)

from .key_pool_support import (
    MODEL,
    OTHER_MODEL,
    PROVIDER,
    FakeFlakyStore,
    ManualClock,
    RecordingSleep,
    exhausted,
    fail_on,
    forbidden,
    lines,
    make_pool,
    oversized,
    rate_limited,
    revoked,
    secret_for,
    serve,
    spec,
    tripping_registry,
)


class TestSticky:
    async def test_every_call_runs_on_the_active_key(self) -> None:
        pool = make_pool(spec("a"), spec("b"), spec("c"))
        assert [await serve(pool) for _ in range(5)] == ["a"] * 5

    async def test_a_dead_key_is_moved_off_once_and_later_calls_start_on_the_new_one(self) -> None:
        pool = make_pool(spec("a"), spec("b"), spec("c"))
        assert await fail_on(pool, exhausted()) == "a"
        assert [await serve(pool) for _ in range(3)] == ["b"] * 3
        assert pool.counters.moves == {("a", "b", MoveReason.EXHAUSTED): 1}

    async def test_the_next_key_wraps_around_in_order(self) -> None:
        pool = make_pool(spec("a"), spec("b"), spec("c"))
        await fail_on(pool, exhausted())
        await pool.reset("a")
        assert await fail_on(pool, exhausted()) == "b"
        assert await fail_on(pool, exhausted()) == "c"
        assert await serve(pool) == "a"

    async def test_the_success_path_is_two_store_operations(self) -> None:
        store = FakeFlakyStore()
        pool = make_pool(spec("a"), spec("b"), store=store)
        await serve(pool)
        store.calls.clear()
        for _ in range(3):
            await serve(pool)
        assert store.calls == ["reserve", "settle"] * 3

    async def test_the_cached_record_is_read_again_only_after_the_refresh_interval(self) -> None:
        clock = ManualClock()
        store = FakeFlakyStore()
        pool = make_pool(spec("a"), store=store, clock=clock)
        await serve(pool)
        store.calls.clear()
        clock.advance(seconds=4)
        await serve(pool)
        assert "read_record" not in store.calls
        clock.advance(seconds=1)
        await serve(pool)
        assert store.calls.count("read_record") == 1


class TestMoveLogging:
    @pytest.mark.parametrize(
        ("verdict", "level", "reason"),
        [
            (exhausted(), logging.WARNING, "exhausted"),
            (revoked(), logging.WARNING, "revoked"),
            (forbidden(), logging.WARNING, "forbidden"),
        ],
    )
    async def test_each_move_logs_once_naming_pool_labels_and_reason(
        self, caplog: pytest.LogCaptureFixture, verdict: object, level: int, reason: str
    ) -> None:
        caplog.set_level(logging.INFO, logger="threetears.models.key_pool")
        pool = make_pool(spec("a"), spec("b"))
        await fail_on(pool, verdict)  # type: ignore[arg-type]
        await serve(pool)
        moved = lines(caplog, level, "moved its active key")
        assert len(moved) == 1
        message = moved[0].getMessage()
        assert "'test-pool'" in message and "'a'" in message and "'b'" in message and reason in message

    @pytest.mark.parametrize(
        ("verdict", "reason"),
        [(exhausted(), "exhausted"), (revoked(), "revoked"), (forbidden(), "forbidden")],
    )
    async def test_the_last_key_running_out_with_no_move_is_logged_at_warning(
        self, caplog: pytest.LogCaptureFixture, verdict: object, reason: str
    ) -> None:
        caplog.set_level(logging.INFO, logger="threetears.models.key_pool")
        pool = make_pool(spec("a"), spec("b"))
        await fail_on(pool, exhausted(VerdictScope.KEY))  # moves a to b: the move's line is a's
        await fail_on(pool, verdict)  # type: ignore[arg-type]
        assert len(lines(caplog, logging.WARNING, f"key 'b' is now {reason}; the active key is 'b'")) == 1
        assert not lines(caplog, logging.WARNING, "key 'a' is now")

    async def test_a_key_that_was_not_active_is_logged_when_its_account_runs_out(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="threetears.models.key_pool")
        pool = make_pool(spec("a", account="acct"), spec("b"), spec("c", account="acct"))
        await fail_on(pool, exhausted(VerdictScope.ACCOUNT))  # a and c, sharing an account; a moves to b
        assert len(lines(caplog, logging.WARNING, "moved its active key from 'a' to 'b': exhausted")) == 1
        assert lines(caplog, logging.WARNING, "key 'c' is now exhausted; the active key is 'b'")
        assert not lines(caplog, logging.WARNING, "key 'a' is now")

    async def test_a_state_already_recorded_is_not_logged_again(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger="threetears.models.key_pool")
        pool = make_pool(spec("a"))
        first = await pool.acquire(MODEL, Decimal("0.01"))
        second = await pool.acquire(MODEL, Decimal("0.01"))
        await first.fail(revoked())
        await second.fail(revoked())  # in flight on a when it was revoked: nothing new to say
        assert len(lines(caplog, logging.WARNING, "key 'a' is now revoked")) == 1

    async def test_a_reset_move_has_its_own_reason_at_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """An operator's reset onto a key never reads, or counts, as the active key's exhaustion."""
        caplog.set_level(logging.INFO, logger="threetears.models.key_pool")
        pool = make_pool(spec("a"), spec("b"))
        await fail_on(pool, exhausted(VerdictScope.KEY))
        await fail_on(pool, exhausted(VerdictScope.KEY))  # b, the last key, is exhausted too: no move
        caplog.clear()
        await pool.reset("a")
        assert await serve(pool) == "a"
        assert len(lines(caplog, logging.INFO, "moved its active key from 'b' to 'a': operator_reset")) == 1
        assert not [record for record in caplog.records if record.levelno >= logging.WARNING]
        assert pool.counters.moves == {
            ("a", "b", MoveReason.EXHAUSTED): 1,
            ("b", "a", MoveReason.OPERATOR_RESET): 1,
        }

    async def test_a_breaker_trip_moves_at_info(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger="threetears.models.key_pool")
        registry, _ = tripping_registry()
        pool = make_pool(spec("a"), spec("b"), breaker_registry=registry)
        assert await serve(pool) == "a"
        registry.get(PROVIDER, credential=secret_for("a")).record_failure()
        assert await serve(pool) == "b"
        assert len(lines(caplog, logging.INFO, "from 'a' to 'b': tripped")) == 1


class TestPullBack:
    """With distinct priorities, the pool returns to the preferred key as soon as it can serve (KP-D5)."""

    async def test_after_the_reset_its_provider_named_passes(self) -> None:
        clock = ManualClock()
        pool = make_pool(spec("a", priority=1), spec("b", priority=2), clock=clock)
        await fail_on(pool, exhausted(VerdictScope.KEY, until=clock.now + timedelta(seconds=30)))
        assert await serve(pool) == "b"
        clock.advance(seconds=31)
        assert await serve(pool) == "a"
        assert pool.counters.moves[("b", "a", MoveReason.PULL_BACK)] == 1

    async def test_after_its_breaker_closes(self) -> None:
        registry, monotonic = tripping_registry()
        pool = make_pool(spec("a", priority=1), spec("b", priority=2), breaker_registry=registry)
        registry.get(PROVIDER, credential=secret_for("a")).record_failure()
        assert await serve(pool) == "b"
        monotonic.now += 31
        assert await serve(pool) == "a"

    async def test_after_an_operator_reset(self) -> None:
        pool = make_pool(spec("a", priority=1), spec("b", priority=2))
        await fail_on(pool, revoked())
        assert await serve(pool) == "b"
        await pool.reset("a")
        assert await serve(pool) == "a"
        assert pool.counters.moves[("b", "a", MoveReason.OPERATOR_RESET)] == 1

    async def test_after_a_period_rollover(self) -> None:
        clock = ManualClock()
        pool = make_pool(spec("a", priority=1), spec("b", priority=2), clock=clock)
        await fail_on(pool, exhausted())
        assert await serve(pool) == "b"
        clock.now = datetime(2026, 12, 1, tzinfo=UTC)
        assert await serve(pool) == "a"

    async def test_after_its_secret_changes(self) -> None:
        store = InMemoryKeyPoolStore()
        old = make_pool(spec("a", priority=1), spec("b", priority=2), store=store)
        await fail_on(old, revoked())
        assert await serve(old) == "b"
        replaced = make_pool(
            spec("a", priority=1, secret="sk-ant-api03-a-replacement"), spec("b", priority=2), store=store
        )
        assert await serve(replaced) == "a"

    async def test_not_with_equal_priorities(self) -> None:
        clock = ManualClock()
        registry, monotonic = tripping_registry()
        pool = make_pool(spec("a"), spec("b"), clock=clock, breaker_registry=registry)
        await fail_on(pool, exhausted(VerdictScope.KEY, until=clock.now + timedelta(seconds=30)))
        clock.advance(seconds=31)
        assert await serve(pool) == "b"
        await pool.reset("a")
        assert await serve(pool) == "b"
        registry.get(PROVIDER, credential=secret_for("b")).record_failure()
        assert await serve(pool) == "a"
        monotonic.now += 31
        assert await serve(pool) == "a"

    async def test_only_to_a_strictly_more_preferred_key(self) -> None:
        clock = ManualClock()
        pool = make_pool(spec("a", priority=1), spec("b", priority=2), spec("c", priority=2), clock=clock)
        await fail_on(pool, exhausted())
        until = clock.now + timedelta(seconds=30)
        assert await fail_on(pool, exhausted(VerdictScope.KEY, until=until)) == "b"
        assert await serve(pool) == "c"
        clock.advance(seconds=31)
        assert await serve(pool) == "c"


class TestSkipVersusMove:
    async def test_a_call_too_big_for_a_roomy_key_skips_it_for_that_call_only(self) -> None:
        pool = make_pool(spec("a", "1"), spec("b", "10"))
        lease = await pool.acquire(MODEL, Decimal("5"))
        assert lease.label == "b"
        await lease.settle(Decimal("5"))
        assert await serve(pool) == "a"
        assert pool.counters.room_refusals == {"a": 1}
        assert not pool.counters.moves

    async def test_a_key_below_headroom_is_moved_off(self) -> None:
        pool = make_pool(spec("a", "10"), spec("b", "10"))
        lease = await pool.acquire(MODEL, Decimal("9.95"))
        await lease.settle(Decimal("9.95"))
        assert await serve(pool, amount="0.06") == "b"
        assert pool.counters.moves == {("a", "b", MoveReason.LOW_ROOM): 1}
        assert await serve(pool) == "b"

    async def test_a_low_preferred_key_is_not_retried_on_every_call(self) -> None:
        store = FakeFlakyStore()
        pool = make_pool(spec("a", "10", priority=1), spec("b", "10", priority=2), store=store)
        lease = await pool.acquire(MODEL, Decimal("9.95"))
        await lease.settle(Decimal("9.95"))
        assert await serve(pool, amount="0.06") == "b"
        store.calls.clear()
        assert await serve(pool) == "b"
        assert store.calls == ["reserve", "settle"]

    async def test_a_move_never_lands_on_a_key_with_no_room(self) -> None:
        pool = make_pool(spec("a", "10"), spec("b", "10"), spec("c", "10"))
        # drain b while a stays active: this call has already tried a
        lease = await pool.acquire(MODEL, Decimal("9.99"), tried={"a"})
        assert lease.label == "b"
        await lease.settle(Decimal("9.99"))
        assert await fail_on(pool, exhausted()) == "a"
        assert await serve(pool) == "c"


class TestPeriodRollover:
    async def test_rollover_returns_to_the_first_key_with_equal_priorities(self) -> None:
        clock = ManualClock()
        pool = make_pool(spec("a"), spec("b"), clock=clock)
        await fail_on(pool, exhausted())
        assert await serve(pool) == "b"
        clock.now = datetime(2026, 12, 1, tzinfo=UTC)
        assert await serve(pool) == "a"
        assert pool.counters.moves[("b", "a", MoveReason.PERIOD_ROLLOVER)] == 1

    async def test_a_new_period_starts_at_zero_spend(self) -> None:
        clock = ManualClock()
        pool = make_pool(spec("a", "1"), clock=clock)
        lease = await pool.acquire(MODEL, Decimal("1"))
        await lease.settle(Decimal("1"))
        with pytest.raises(NoKeyAvailableError):
            await pool.acquire(MODEL, Decimal("0.5"))
        clock.now = datetime(2026, 12, 1, tzinfo=UTC)
        assert (await pool.acquire(MODEL, Decimal("0.5"))).label == "a"

    async def test_the_anchor_day_sets_the_rollover(self) -> None:
        clock = ManualClock(datetime(2026, 11, 26, tzinfo=UTC))
        pool = make_pool(spec("a"), spec("b"), clock=clock, anchor_day=15)
        await fail_on(pool, exhausted())
        clock.now = datetime(2026, 12, 14, 23, 59, tzinfo=UTC)
        assert await serve(pool) == "b"
        clock.now = datetime(2026, 12, 15, tzinfo=UTC)
        assert await serve(pool) == "a"
        assert (await pool.status()).period == "2026-12-15"

    async def test_a_reservation_settles_into_the_period_it_was_made_in(self) -> None:
        clock = ManualClock(datetime(2026, 11, 30, 23, 59, tzinfo=UTC))
        pool = make_pool(spec("a", "10"), clock=clock)
        lease = await pool.acquire(MODEL, Decimal("2"))
        clock.now = datetime(2026, 12, 1, 0, 1, tzinfo=UTC)
        await lease.settle(Decimal("2"))
        status = await pool.status()
        assert status.keys[0].spent == Decimal(0)
        assert lease.period == "2026-11"


def _openrouter_pool(clock: ManualClock, sleep: RecordingSleep, **settings: object) -> KeyPool:
    keys = [KeySpec(Secret("sk-or-v1-only-0123456789abcdef"), "only", Decimal("10"))]
    return KeyPool("openrouter", "or-pool", keys, InMemoryKeyPoolStore(), clock=clock, sleep=sleep, **settings)  # type: ignore[arg-type]


class TestPoolSideRateLimitRetries:
    """A rate limit means retry (owner, 2026-11-26). Where the provider's SDK never retries one
    (OpenRouter's), the pool retries the call on the same key twice, through its injected sleep."""

    async def test_twice_after_each_retry_after_then_no_more(self) -> None:
        clock = ManualClock()
        sleep = RecordingSleep(clock)
        pool = _openrouter_pool(clock, sleep)
        answers = [await pool.wait_to_retry(rate_limited(until=clock.now + timedelta(seconds=17)), n) for n in range(3)]
        assert answers == [True, True, False]
        assert sleep.waits == [17.0, 17.0]

    async def test_with_no_wait_named_it_backs_off(self) -> None:
        clock = ManualClock()
        sleep = RecordingSleep(clock)
        pool = _openrouter_pool(clock, sleep)
        assert [await pool.wait_to_retry(rate_limited(), n) for n in range(2)] == [True, True]
        assert sleep.waits == [0.5, 1.0]

    async def test_a_wait_never_passes_the_exhaustion_threshold(self) -> None:
        clock = ManualClock()
        sleep = RecordingSleep(clock)
        pool = _openrouter_pool(clock, sleep, rate_limit_exhaustion_wait=timedelta(seconds=40))
        assert await pool.wait_to_retry(rate_limited(until=clock.now + timedelta(minutes=5)), 0)
        assert sleep.waits == [40.0]

    async def test_a_wait_already_over_is_no_wait(self) -> None:
        clock = ManualClock()
        sleep = RecordingSleep(clock)
        pool = _openrouter_pool(clock, sleep)
        assert await pool.wait_to_retry(rate_limited(until=clock.now - timedelta(seconds=3)), 0)
        assert sleep.waits == [0.0]

    @pytest.mark.parametrize("verdict", [exhausted(), revoked(), forbidden(), oversized()])
    async def test_only_a_rate_limit_is_retried(self, verdict: object) -> None:
        clock = ManualClock()
        sleep = RecordingSleep(clock)
        assert not await _openrouter_pool(clock, sleep).wait_to_retry(verdict, 0)  # type: ignore[arg-type]
        assert sleep.waits == []

    async def test_a_provider_whose_sdk_retries_rate_limits_gets_no_more_from_the_pool(self) -> None:
        clock = ManualClock()
        sleep = RecordingSleep(clock)
        pool = make_pool(spec("a"), clock=clock, sleep=sleep)  # anthropic
        assert not await pool.wait_to_retry(rate_limited(until=clock.now + timedelta(seconds=1)), 0)
        assert sleep.waits == []

    async def test_a_negative_retry_count_is_refused(self) -> None:
        clock = ManualClock()
        with pytest.raises(ValueError, match="retries"):
            await _openrouter_pool(clock, RecordingSleep(clock)).wait_to_retry(rate_limited(), -1)

    @pytest.mark.parametrize("wait", [timedelta(0), timedelta(seconds=-1)])
    def test_the_exhaustion_threshold_must_be_positive(self, wait: timedelta) -> None:
        with pytest.raises(ValueError, match="rate_limit_exhaustion_wait"):
            make_pool(spec("a"), rate_limit_exhaustion_wait=wait)


class TestOneKeyPool:
    """One key goes through every path a larger pool does (KP-A2)."""

    async def test_serves_and_spends(self) -> None:
        pool = make_pool(spec("only", "1"))
        assert await serve(pool, amount="0.4") == "only"
        status = await pool.status()
        assert status.active == "only"
        assert status.keys[0].spent == Decimal("0.4")

    async def test_a_call_too_big_raises_naming_the_room(self) -> None:
        pool = make_pool(spec("only", "1"))
        with pytest.raises(NoKeyAvailableError, match=r"only: no room: 1 of 1"):
            await pool.acquire(MODEL, Decimal("2"))

    @pytest.mark.parametrize(
        ("verdict", "state"),
        [(exhausted(), "exhausted"), (revoked(), "revoked"), (forbidden(VerdictScope.KEY), "forbidden")],
    )
    async def test_a_dead_key_raises_and_a_reset_brings_it_back(self, verdict: object, state: str) -> None:
        pool = make_pool(spec("only"))
        await fail_on(pool, verdict)  # type: ignore[arg-type]
        with pytest.raises(NoKeyAvailableError, match=f"only: {state}"):
            await pool.acquire(MODEL, Decimal("0.01"))
        await pool.reset("only")
        assert await serve(pool) == "only"

    async def test_forbidden_for_one_model_still_serves_another(self) -> None:
        pool = make_pool(spec("only"))
        await fail_on(pool, forbidden(), model_id=MODEL)
        with pytest.raises(NoKeyAvailableError):
            await pool.acquire(MODEL, Decimal("0.01"))
        assert await serve(pool, model_id=OTHER_MODEL) == "only"

    async def test_a_rate_limit_never_takes_the_only_key_out(self) -> None:
        pool = make_pool(spec("only"))
        await fail_on(pool, rate_limited())
        assert await serve(pool) == "only"

    async def test_exhausted_until_a_named_reset_then_served(self) -> None:
        clock = ManualClock()
        pool = make_pool(spec("only"), clock=clock)
        await fail_on(pool, exhausted(until=clock.now + timedelta(minutes=30)))
        with pytest.raises(NoKeyAvailableError, match="only: exhausted") as caught:
            await pool.acquire(MODEL, Decimal("0.01"))
        assert caught.value.earliest == clock.now + timedelta(minutes=30)
        clock.advance(minutes=30)
        assert await serve(pool) == "only"

    async def test_a_tripped_breaker_raises_until_it_admits(self) -> None:
        registry, monotonic = tripping_registry()
        pool = make_pool(spec("only"), breaker_registry=registry)
        registry.get(PROVIDER, credential=secret_for("only")).record_failure()
        with pytest.raises(NoKeyAvailableError, match="only: tripped until"):
            await pool.acquire(MODEL, Decimal("0.01"))
        monotonic.now += 30
        assert await serve(pool) == "only"

    async def test_a_tried_key_is_not_tried_again(self) -> None:
        pool = make_pool(spec("only"))
        with pytest.raises(NoKeyAvailableError, match="only: tried"):
            await pool.acquire(MODEL, Decimal("0.01"), tried={"only"})

    async def test_the_plain_reservation_api(self) -> None:
        pool = make_pool(spec("only", "1"))
        lease = await pool.reserve(Decimal("0.5"))
        assert lease.model_id is None
        await lease.settle(Decimal("0.25"))
        assert (await pool.status()).keys[0].spent == Decimal("0.25")


class TestConcurrency:
    async def test_concurrent_acquires_never_take_a_key_past_its_budget(self) -> None:
        pool = make_pool(spec("only", "1"), store=FakeFlakyStore())

        async def one() -> KeyPoolLease | None:
            try:
                return await pool.acquire(MODEL, Decimal("0.1"))
            except NoKeyAvailableError:
                return None

        leases = await asyncio.gather(*(one() for _ in range(25)))
        held = [lease for lease in leases if lease is not None]
        assert len(held) == 10
        status = await pool.status()
        assert status.keys[0].reserved == Decimal("1.0")

    @settings(max_examples=40, deadline=None)
    @given(
        budgets=st.lists(st.integers(min_value=1, max_value=20), min_size=1, max_size=3),
        amounts=st.lists(st.integers(min_value=1, max_value=8), min_size=1, max_size=30),
    )
    def test_concurrent_reservations_across_two_pools_never_exceed_any_budget(
        self, budgets: list[int], amounts: list[int]
    ) -> None:
        asyncio.run(_reserve_concurrently(budgets, amounts))


async def _reserve_concurrently(budgets: list[int], amounts: list[int]) -> None:
    store = FakeFlakyStore()
    keys = [spec(f"k{index}", str(budget)) for index, budget in enumerate(budgets)]
    pools = [make_pool(*keys, store=store), make_pool(*keys, store=store)]

    async def one(index: int, amount: int) -> KeyPoolLease | None:
        try:
            return await pools[index % 2].acquire(MODEL, Decimal(amount))
        except NoKeyAvailableError:
            return None

    leases = await asyncio.gather(*(one(index, amount) for index, amount in enumerate(amounts)))
    held: dict[str, Decimal] = {}
    for lease in leases:
        if lease is not None:
            held[lease.label] = held.get(lease.label, Decimal(0)) + lease.amount
    for key in keys:
        assert held.get(key.label, Decimal(0)) <= key.budget
    for status in (await pools[0].status()).keys:
        assert status.reserved == held.get(status.label, Decimal(0))


class TestSharedStore:
    """Two pool instances on one store see each other's spend, state and active key (KP-G4)."""

    async def test_spend_is_shared(self) -> None:
        store = InMemoryKeyPoolStore()
        first, second = make_pool(spec("a", "1"), store=store), make_pool(spec("a", "1"), store=store)
        await serve(first, amount="0.75")
        assert (await second.status()).keys[0].spent == Decimal("0.75")
        with pytest.raises(NoKeyAvailableError):
            await second.acquire(MODEL, Decimal("0.5"))

    async def test_state_and_the_active_key_are_shared(self) -> None:
        store = InMemoryKeyPoolStore()
        clock = ManualClock()
        first = make_pool(spec("a"), spec("b"), store=store, clock=clock)
        second = make_pool(spec("a"), spec("b"), store=store, clock=clock)
        assert await serve(second) == "a"
        await fail_on(first, exhausted())
        clock.advance(seconds=5)
        assert await serve(second) == "b"
        status = await second.status()
        assert status.active == "b"
        assert status.keys[0].state == "exhausted"
        assert not second.counters.moves

    async def test_two_pools_moving_at_once_agree_and_the_loser_adopts(self) -> None:
        store = InMemoryKeyPoolStore()
        first = make_pool(spec("a"), spec("b"), spec("c"), store=store)
        second = make_pool(spec("a"), spec("b"), spec("c"), store=store)
        lease_one = await first.acquire(MODEL, Decimal("0.01"))
        lease_two = await second.acquire(MODEL, Decimal("0.01"))
        assert lease_one.label == lease_two.label == "a"
        await lease_one.fail(exhausted())
        await lease_two.fail(revoked())
        assert await serve(first) == await serve(second) == "b"
        assert first.counters.moves == {("a", "b", MoveReason.EXHAUSTED): 1}
        assert not second.counters.moves
        status = await second.status()
        assert status.keys[0].state == "revoked"
