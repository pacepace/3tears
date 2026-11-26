"""Key state: each verdict's state and lifetime, account scope, rate-limit holds, reservations and their
settlement, the orphan sweep, fingerprints, configured budgets, operator reset and ``status()``
(KP-A5, KP-B2, KP-C3 to C5, KP-E2 to E5, KP-K2)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from threetears.models.key_pool import (
    InMemoryKeyPoolStore,
    KeyState,
    MoveReason,
    NoKeyAvailableError,
    ReservationNotFound,
    VerdictScope,
)

from .key_pool_support import (
    MODEL,
    OTHER_MODEL,
    POOL,
    PROVIDER,
    FakeFlakyStore,
    ManualClock,
    exhausted,
    fail_on,
    forbidden,
    lines,
    make_pool,
    not_key_related,
    rate_limited,
    revoked,
    secret_for,
    serve,
    spec,
    tripping_registry,
)


async def _state(pool: object, label: str) -> KeyState:
    status = await pool.status()  # type: ignore[attr-defined]
    return next(key.state for key in status.keys if key.label == label)


async def _until(pool: object, label: str) -> datetime | None:
    status = await pool.status()  # type: ignore[attr-defined]
    return next(key.until for key in status.keys if key.label == label)


class TestLifetimes:
    """KP-C4: when each state ends."""

    async def test_exhausted_ends_at_the_period_rollover(self) -> None:
        clock = ManualClock()
        pool = make_pool(spec("a"), clock=clock)
        await fail_on(pool, exhausted())
        assert await _state(pool, "a") is KeyState.EXHAUSTED
        assert await _until(pool, "a") == datetime(2026, 12, 1, tzinfo=UTC)
        clock.now = datetime(2026, 12, 1, tzinfo=UTC)
        assert await _state(pool, "a") is KeyState.OK

    @pytest.mark.parametrize("verdict", [revoked(), forbidden(VerdictScope.KEY)])
    async def test_revoked_and_forbidden_outlive_the_rollover(self, verdict: object) -> None:
        clock = ManualClock()
        pool = make_pool(spec("a"), clock=clock)
        await fail_on(pool, verdict)  # type: ignore[arg-type]
        clock.now = datetime(2027, 3, 1, tzinfo=UTC)
        assert await _state(pool, "a") in (KeyState.REVOKED, KeyState.FORBIDDEN)

    @pytest.mark.parametrize("verdict", [exhausted(), revoked(), forbidden(VerdictScope.KEY)])
    async def test_every_stored_state_ends_at_an_operator_reset(self, verdict: object) -> None:
        pool = make_pool(spec("a"))
        await fail_on(pool, verdict)  # type: ignore[arg-type]
        await pool.reset("a")
        assert await _state(pool, "a") is KeyState.OK

    @pytest.mark.parametrize("verdict", [exhausted(), revoked(), forbidden(VerdictScope.KEY), rate_limited()])
    async def test_every_stored_state_ends_when_the_labels_secret_changes(self, verdict: object) -> None:
        store = InMemoryKeyPoolStore()
        old = make_pool(spec("a"), store=store)
        await fail_on(old, verdict)  # type: ignore[arg-type]
        new = make_pool(spec("a", secret="sk-ant-api03-a-rotated"), store=store)
        assert await _state(new, "a") is KeyState.OK
        assert await serve(new) == "a"

    async def test_a_rate_limit_ends_at_its_until(self) -> None:
        clock = ManualClock()
        pool = make_pool(spec("a"), clock=clock)
        await fail_on(pool, rate_limited(until=clock.now + timedelta(seconds=45)))
        clock.advance(seconds=44)
        assert await _state(pool, "a") is KeyState.RATE_LIMITED
        clock.advance(seconds=1)
        assert await _state(pool, "a") is KeyState.OK

    async def test_a_tripped_breaker_is_read_live_and_never_stored(self) -> None:
        store = InMemoryKeyPoolStore()
        registry, monotonic = tripping_registry()
        tripped = make_pool(spec("a"), store=store, breaker_registry=registry)
        other = make_pool(spec("a"), store=store)
        await serve(tripped)
        registry.get(PROVIDER, credential=secret_for("a")).record_failure()
        assert await _state(tripped, "a") is KeyState.TRIPPED
        assert await _state(other, "a") is KeyState.OK
        _, record = await store.read_record(f"key-pool/{PROVIDER}/{POOL}")
        assert "tripped" not in str(record)
        monotonic.now += 30
        assert await _state(tripped, "a") is KeyState.OK

    async def test_a_per_model_breaker_blocks_only_its_model(self) -> None:
        registry, _ = tripping_registry()
        pool = make_pool(spec("a"), breaker_registry=registry)
        registry.get(PROVIDER, credential=secret_for("a"), model=MODEL).record_failure()
        with pytest.raises(NoKeyAvailableError):
            await pool.acquire(MODEL, Decimal("0.01"))
        assert await serve(pool, model_id=OTHER_MODEL) == "a"


class TestAccountScope:
    """KP-C5: an account-scoped verdict applies to every key in the account."""

    async def test_an_exhausted_account_takes_every_key_in_it(self) -> None:
        pool = make_pool(spec("a", account="org-1"), spec("b", account="org-1"), spec("c", account="org-2"))
        await fail_on(pool, exhausted(VerdictScope.ACCOUNT))
        assert await _state(pool, "b") is KeyState.EXHAUSTED
        assert await serve(pool) == "c"
        assert pool.counters.moves == {("a", "c", MoveReason.EXHAUSTED): 1}

    async def test_a_key_scoped_verdict_stays_on_its_key(self) -> None:
        pool = make_pool(spec("a", account="org-1"), spec("b", account="org-1"))
        await fail_on(pool, exhausted(VerdictScope.KEY))
        assert await _state(pool, "b") is KeyState.OK
        assert await serve(pool) == "b"

    async def test_an_account_rate_limit_holds_every_key_in_it(self) -> None:
        pool = make_pool(spec("a", account="org"), spec("b", account="org"), spec("c"))
        await fail_on(pool, rate_limited(scope=VerdictScope.ACCOUNT))
        assert await _state(pool, "b") is KeyState.RATE_LIMITED
        assert await serve(pool) == "c"

    async def test_keys_with_no_account_are_each_their_own(self) -> None:
        pool = make_pool(spec("a"), spec("b"))
        await fail_on(pool, exhausted(VerdictScope.ACCOUNT))
        assert await _state(pool, "b") is KeyState.OK


class TestHolds:
    """KP-C3: a floor when the provider names no wait, doubling for a key held again soon, a cap."""

    async def test_the_floor_holds_a_key_the_provider_named_no_wait_for(self) -> None:
        clock = ManualClock()
        pool = make_pool(spec("a"), clock=clock)
        await fail_on(pool, rate_limited())
        assert await _until(pool, "a") == clock.now + timedelta(seconds=30)

    async def test_a_configured_floor(self) -> None:
        clock = ManualClock()
        pool = make_pool(spec("a"), clock=clock, rate_limit_floor=timedelta(seconds=5))
        await fail_on(pool, rate_limited())
        assert await _until(pool, "a") == clock.now + timedelta(seconds=5)

    async def test_the_providers_wait_is_kept_even_below_the_floor(self) -> None:
        clock = ManualClock()
        pool = make_pool(spec("a"), clock=clock)
        await fail_on(pool, rate_limited(until=clock.now + timedelta(seconds=3)))
        assert await _until(pool, "a") == clock.now + timedelta(seconds=3)

    async def test_a_key_held_again_within_the_window_is_held_twice_as_long_up_to_the_cap(self) -> None:
        clock = ManualClock()
        pool = make_pool(spec("a"), clock=clock)
        holds: list[float] = []
        for _ in range(8):
            await fail_on(pool, rate_limited())
            until = await _until(pool, "a")
            assert until is not None
            holds.append((until - clock.now).total_seconds())
            clock.now = until + timedelta(minutes=1)
        assert holds == [30, 60, 120, 240, 480, 900, 900, 900]

    async def test_a_key_held_again_after_the_window_starts_over_at_the_floor(self) -> None:
        clock = ManualClock()
        pool = make_pool(spec("a"), clock=clock)
        await fail_on(pool, rate_limited())
        clock.advance(seconds=30 + 11 * 60)
        await fail_on(pool, rate_limited())
        assert await _until(pool, "a") == clock.now + timedelta(seconds=30)

    async def test_a_providers_wait_past_the_cap_is_kept(self) -> None:
        clock = ManualClock()
        pool = make_pool(spec("a"), clock=clock)
        until = clock.now + timedelta(hours=1)
        await fail_on(pool, rate_limited(until=until))
        assert await _until(pool, "a") == until

    async def test_a_second_limit_during_a_hold_does_not_double_it(self) -> None:
        clock = ManualClock()
        store = InMemoryKeyPoolStore()
        first, second = make_pool(spec("a"), store=store, clock=clock), make_pool(spec("a"), store=store, clock=clock)
        one, two = await first.acquire(MODEL, Decimal("0.01")), await second.acquire(MODEL, Decimal("0.01"))
        await one.fail(rate_limited())
        await two.fail(rate_limited())
        assert await _until(second, "a") == clock.now + timedelta(seconds=30)

    async def test_a_key_and_model_hold_is_for_that_model_only(self) -> None:
        clock = ManualClock()
        pool = make_pool(spec("a"), clock=clock)
        await fail_on(pool, rate_limited(scope=VerdictScope.KEY_AND_MODEL), model_id=MODEL)
        with pytest.raises(NoKeyAvailableError):
            await pool.acquire(MODEL, Decimal("0.01"))
        assert await serve(pool, model_id=OTHER_MODEL) == "a"
        status = (await pool.status()).keys[0]
        assert status.state is KeyState.OK
        assert status.rate_limited_models == ((MODEL, clock.now + timedelta(seconds=30)),)

    async def test_forbidden_per_model_shows_in_status(self) -> None:
        pool = make_pool(spec("a"))
        await fail_on(pool, forbidden(), model_id=MODEL)
        status = (await pool.status()).keys[0]
        assert status.state is KeyState.OK
        assert status.forbidden_models == (MODEL,)


class TestMoney:
    """KP-E2 to E5: reserve, then settle, release or fail."""

    async def test_settle_replaces_the_reservation_with_the_cost(self) -> None:
        pool = make_pool(spec("a", "10"))
        lease = await pool.acquire(MODEL, Decimal("1"))
        assert (await pool.status()).keys[0].reserved == Decimal("1")
        await lease.settle(Decimal("0.3"))
        key = (await pool.status()).keys[0]
        assert (key.spent, key.reserved, key.remaining) == (Decimal("0.3"), Decimal(0), Decimal("9.7"))

    async def test_a_cost_above_the_reservation_is_recorded_as_it_is(self) -> None:
        pool = make_pool(spec("a", "10"))
        lease = await pool.acquire(MODEL, Decimal("1"))
        await lease.settle(Decimal("1.5"))
        assert (await pool.status()).keys[0].spent == Decimal("1.5")

    async def test_release_spends_nothing(self) -> None:
        pool = make_pool(spec("a", "10"))
        lease = await pool.acquire(MODEL, Decimal("1"))
        await lease.release()
        key = (await pool.status()).keys[0]
        assert (key.spent, key.reserved) == (Decimal(0), Decimal(0))

    @pytest.mark.parametrize(("billed", "spent"), [(True, Decimal("1")), (False, Decimal(0))])
    async def test_a_failure_settles_in_full_or_releases_by_whether_it_may_have_billed(
        self, billed: bool, spent: Decimal
    ) -> None:
        pool = make_pool(spec("a", "10"))
        lease = await pool.acquire(MODEL, Decimal("1"))
        await lease.fail(not_key_related(billed=billed))
        assert (await pool.status()).keys[0].spent == spent
        assert await _state(pool, "a") is KeyState.OK

    async def test_a_key_moving_failure_records_the_state_and_closes_the_money(self) -> None:
        pool = make_pool(spec("a", "10"), spec("b", "10"))
        lease = await pool.acquire(MODEL, Decimal("1"))
        await lease.fail(revoked())
        status = await pool.status()
        assert status.keys[0].state is KeyState.REVOKED
        assert (status.keys[0].spent, status.keys[0].reserved) == (Decimal(0), Decimal(0))
        assert status.active == "b"

    async def test_a_lease_closes_once(self) -> None:
        pool = make_pool(spec("a", "10"))
        lease = await pool.acquire(MODEL, Decimal("1"))
        await lease.settle(Decimal("1"))
        assert lease.closed
        with pytest.raises(ReservationNotFound):
            await lease.settle(Decimal("1"))
        with pytest.raises(ReservationNotFound):
            await lease.fail(revoked())
        assert await _state(pool, "a") is KeyState.OK

    async def test_reservation_ids_are_unique(self) -> None:
        pool = make_pool(spec("a", "10"))
        leases = [await pool.acquire(MODEL, Decimal("0.1")) for _ in range(20)]
        assert len({lease.reservation_id for lease in leases}) == 20

    async def test_a_reservation_expires_its_margin_after_the_deadline(self) -> None:
        clock = ManualClock()
        pool = make_pool(spec("a"), clock=clock, reservation_margin=timedelta(minutes=2))
        deadline = clock.now + timedelta(minutes=5)
        assert (await pool.acquire(MODEL, Decimal("0.1"), deadline=deadline)).expires_at == deadline + timedelta(
            minutes=2
        )
        assert (await pool.acquire(MODEL, Decimal("0.1"))).expires_at == clock.now + timedelta(minutes=2)

    @pytest.mark.parametrize("amount", [Decimal(-1), Decimal("NaN"), 1.0])
    async def test_an_invalid_amount_is_refused(self, amount: object) -> None:
        pool = make_pool(spec("a"))
        with pytest.raises(ValueError, match="amount"):
            await pool.acquire(MODEL, amount)  # type: ignore[arg-type]

    async def test_a_naive_deadline_is_refused(self) -> None:
        pool = make_pool(spec("a"))
        with pytest.raises(ValueError, match="deadline"):
            await pool.acquire(MODEL, Decimal("0.1"), deadline=datetime(2026, 11, 26, 13))


class TestOrphanSweep:
    """KP-E5: a reservation nobody closed is settled in full once it expires, and counted."""

    async def test_an_expired_reservation_is_settled_in_full_and_counted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = ManualClock()
        pool = make_pool(spec("a", "10"), clock=clock, reservation_margin=timedelta(minutes=1))
        await pool.acquire(MODEL, Decimal("2"))  # its caller crashed and never closed it
        clock.advance(minutes=2)
        await serve(pool)
        assert pool.counters.orphaned == {"a": 1}
        assert (await pool.status()).keys[0].spent == Decimal("2.01")
        assert len(lines(caplog, logging.WARNING, "orphaned reservation")) == 1

    async def test_a_late_settle_after_the_sweep_corrects_the_spend(self) -> None:
        clock = ManualClock()
        pool = make_pool(spec("a", "10"), clock=clock, reservation_margin=timedelta(minutes=1))
        slow = await pool.acquire(MODEL, Decimal("2"))
        clock.advance(minutes=2)
        await serve(pool)
        await slow.settle(Decimal("0.5"))
        assert (await pool.status()).keys[0].spent == Decimal("0.51")

    async def test_the_sweep_runs_at_a_bounded_cadence_not_on_every_call(self) -> None:
        clock = ManualClock()
        store = FakeFlakyStore()
        pool = make_pool(spec("a"), store=store, clock=clock, sweep_interval=timedelta(minutes=1))
        for _ in range(5):
            await serve(pool)
        assert store.calls.count("sweep_expired") == 1
        clock.advance(minutes=1)
        await serve(pool)
        assert store.calls.count("sweep_expired") == 2

    async def test_the_previous_periods_orphans_are_swept_too(self) -> None:
        clock = ManualClock(datetime(2026, 11, 30, 23, 55, tzinfo=UTC))
        pool = make_pool(spec("a", "10"), clock=clock, reservation_margin=timedelta(minutes=1))
        await pool.acquire(MODEL, Decimal("2"))
        clock.now = datetime(2026, 12, 1, 0, 5, tzinfo=UTC)
        await serve(pool)
        assert pool.counters.orphaned == {"a": 1}


class TestFingerprints:
    """KP-B2: a label holding a new secret starts from fresh state and zero spend."""

    async def test_a_new_secret_starts_at_zero_spend_and_the_old_keeps_its_own(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="threetears.models.key_pool")
        store = InMemoryKeyPoolStore()
        old = make_pool(spec("a", "10"), store=store)
        await serve(old, amount="4")
        new = make_pool(spec("a", "10", secret="sk-ant-api03-a-rotated"), store=store)
        assert (await new.status()).keys[0].spent == Decimal(0)
        assert (await old.status()).keys[0].spent == Decimal("4")
        assert len(lines(caplog, logging.INFO, "holds a new secret")) == 1

    async def test_the_same_secret_in_another_process_shares_its_state(self) -> None:
        store = InMemoryKeyPoolStore()
        first = make_pool(spec("a", "10"), store=store)
        await serve(first, amount="4")
        await fail_on(first, revoked())
        again = make_pool(spec("a", "10"), store=store)
        status = (await again.status()).keys[0]
        assert (status.state, status.spent) == (KeyState.REVOKED, Decimal("4"))

    async def test_a_secret_flip_flopping_between_processes_never_resets_spend(self) -> None:
        store = InMemoryKeyPoolStore()
        old = make_pool(spec("a", "10"), store=store)
        await serve(old, amount="4")
        new = make_pool(spec("a", "10", secret="sk-ant-api03-a-rotated"), store=store)
        await serve(new, amount="3")
        old_again = make_pool(spec("a", "10"), store=store)
        assert (await old_again.status()).keys[0].spent == Decimal("4")
        new_again = make_pool(spec("a", "10", secret="sk-ant-api03-a-rotated"), store=store)
        assert (await new_again.status()).keys[0].spent == Decimal("3")


class TestConfiguredBudget:
    """KP-A5: a different budget configured for the same key is logged."""

    async def test_a_different_budget_logs_a_warning_naming_the_label_and_both_amounts(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        store = InMemoryKeyPoolStore()
        await serve(make_pool(spec("a", "10"), store=store))
        await serve(make_pool(spec("a", "25"), store=store))
        warned = lines(caplog, logging.WARNING, "budget")
        assert len(warned) == 1
        message = warned[0].getMessage()
        assert "'a'" in message and "10" in message and "25" in message

    async def test_the_same_budget_logs_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        store = InMemoryKeyPoolStore()
        await serve(make_pool(spec("a", "10"), store=store))
        await serve(make_pool(spec("a", "10.00"), store=store))
        assert not lines(caplog, logging.WARNING, "budget")

    async def test_the_budget_each_process_configures_is_the_one_it_enforces(self) -> None:
        store = InMemoryKeyPoolStore()
        await serve(make_pool(spec("a", "10"), store=store), amount="3")
        small = make_pool(spec("a", "3"), store=store)
        with pytest.raises(NoKeyAvailableError):
            await small.acquire(MODEL, Decimal("0.5"))


class TestResetAndStatus:
    async def test_reset_of_an_unknown_label_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no key labelled"):
            await make_pool(spec("a")).reset("missing")

    async def test_reset_clears_holds_and_per_model_forbids(self) -> None:
        pool = make_pool(spec("a"))
        await fail_on(pool, forbidden(), model_id=MODEL)
        await fail_on(pool, rate_limited(), model_id=OTHER_MODEL)
        await pool.reset("a")
        status = (await pool.status()).keys[0]
        assert (status.state, status.forbidden_models, status.rate_limited_models) == (KeyState.OK, (), ())

    async def test_reset_resets_the_keys_breakers(self) -> None:
        registry, _ = tripping_registry()
        pool = make_pool(spec("a"), breaker_registry=registry)
        registry.get(PROVIDER, credential=secret_for("a")).record_failure()
        await pool.reset("a")
        assert await serve(pool) == "a"

    async def test_status_reports_every_key_in_preference_order(self) -> None:
        clock = ManualClock()
        pool = make_pool(spec("b", "5", priority=2), spec("a", "10", priority=1), clock=clock)
        await serve(pool, amount="1")
        await pool.acquire(MODEL, Decimal("2"))
        status = await pool.status()
        assert (status.pool, status.provider, status.period, status.active) == (POOL, PROVIDER, "2026-11", "a")
        first, second = status.keys
        assert (first.label, first.state, first.active) == ("a", KeyState.OK, True)
        assert (first.spent, first.reserved, first.remaining, first.budget) == (
            Decimal("1"),
            Decimal("2"),
            Decimal("7"),
            Decimal("10"),
        )
        assert (second.label, second.active, second.remaining) == ("b", False, Decimal("5"))

    async def test_status_reports_low_room(self) -> None:
        pool = make_pool(spec("a", "10"))
        await serve(pool, amount="9.95")
        status = (await pool.status()).keys[0]
        assert (status.state, status.until) == (KeyState.NO_ROOM, datetime(2026, 12, 1, tzinfo=UTC))
