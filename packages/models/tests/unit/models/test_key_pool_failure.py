"""When a pool can't serve: ``NoKeyAvailableError`` with its contents, log line and counter; a store
that is down failing the call closed; and a planted secret appearing nowhere (KP-I, KP-G5, KP-B)."""

from __future__ import annotations

import logging
import traceback
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from threetears.models.key_pool import (
    KeyPool,
    KeyPoolStoreError,
    KeyState,
    NoKeyAvailableError,
    ReservationNotFound,
    VerdictScope,
)

from .key_pool_support import (
    MODEL,
    POOL,
    PROVIDER,
    FakeFlakyStore,
    ManualClock,
    exhausted,
    fail_on,
    forbidden,
    lines,
    make_pool,
    rate_limited,
    revoked,
    serve,
    spec,
    tripping_registry,
)


async def _every_key_out(clock: ManualClock) -> tuple[KeyPool, NoKeyAvailableError]:
    """A six-key pool with each key out for a different reason, and the error the next call raises."""
    registry, _ = tripping_registry()
    pool = make_pool(
        spec("a"), spec("b"), spec("c"), spec("d"), spec("e"), spec("f", "1"), clock=clock, breaker_registry=registry
    )
    await fail_on(pool, exhausted(VerdictScope.KEY))
    await fail_on(pool, revoked())
    await fail_on(pool, forbidden())
    await fail_on(pool, rate_limited(until=clock.now + timedelta(seconds=20)))
    registry.get(PROVIDER, credential=pool.keys[4].secret.reveal()).record_failure()
    drained = await pool.acquire(MODEL, Decimal("1"))
    assert drained.label == "f"
    await drained.settle(Decimal("1"))
    with pytest.raises(NoKeyAvailableError) as caught:
        await pool.acquire(MODEL, Decimal("0.01"))
    return pool, caught.value


class TestNoKeyAvailable:
    async def test_it_names_every_key_and_its_state(self) -> None:
        clock = ManualClock()
        _, error = await _every_key_out(clock)
        assert [(key.label, key.state) for key in error.keys] == [
            ("a", KeyState.EXHAUSTED),
            ("b", KeyState.REVOKED),
            ("c", KeyState.FORBIDDEN),
            ("d", KeyState.RATE_LIMITED),
            ("e", KeyState.TRIPPED),
            ("f", KeyState.NO_ROOM),
        ]
        text = str(error)
        rate_until = (clock.now + timedelta(seconds=20)).isoformat()
        trip_until = (clock.now + timedelta(seconds=30)).isoformat()
        for part in (
            "a: exhausted",
            "b: revoked",
            "c: forbidden",
            f"d: rate_limited until {rate_until}",
            f"e: tripped until {trip_until}",
            "f: no room: 0 of 1",
            f"pool {POOL!r}",
            f"model {MODEL!r}",
        ):
            assert part in text

    async def test_it_names_the_earliest_time_a_key_could_serve_again(self) -> None:
        clock = ManualClock()
        _, error = await _every_key_out(clock)
        assert error.earliest == clock.now + timedelta(seconds=20)
        assert f"the earliest a key could serve again: {error.earliest.isoformat()}" in str(error)

    async def test_with_nothing_that_ends_on_its_own_the_earliest_is_unknown(self) -> None:
        pool = make_pool(spec("a"), spec("b"))
        await fail_on(pool, revoked())
        await fail_on(pool, revoked())
        with pytest.raises(NoKeyAvailableError, match="could serve again: unknown") as caught:
            await pool.acquire(MODEL, Decimal("0.01"))
        assert caught.value.earliest is None

    async def test_an_exhausted_key_could_serve_again_at_the_rollover(self) -> None:
        pool = make_pool(spec("a"))
        await fail_on(pool, exhausted())
        with pytest.raises(NoKeyAvailableError) as caught:
            await pool.acquire(MODEL, Decimal("0.01"))
        assert caught.value.earliest == datetime(2026, 12, 1, tzinfo=UTC)

    async def test_it_is_raised_with_no_reservation_held(self) -> None:
        pool = make_pool(spec("a"))
        await fail_on(pool, revoked())
        with pytest.raises(NoKeyAvailableError):
            await pool.acquire(MODEL, Decimal("0.01"))
        assert (await pool.status()).keys[0].reserved == Decimal(0)

    async def test_every_raise_is_counted_and_the_log_is_one_line_per_minute(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = ManualClock()
        pool = make_pool(spec("a"), clock=clock)
        await fail_on(pool, revoked())
        for _ in range(3):
            with pytest.raises(NoKeyAvailableError):
                await pool.acquire(MODEL, Decimal("0.01"))
            clock.advance(seconds=10)
        assert pool.counters.no_key_available == 3
        logged = lines(caplog, logging.ERROR, "no key in pool")
        assert len(logged) == 1
        assert "0 more not logged" in logged[0].getMessage()
        clock.advance(seconds=31)
        with pytest.raises(NoKeyAvailableError):
            await pool.acquire(MODEL, Decimal("0.01"))
        logged = lines(caplog, logging.ERROR, "no key in pool")
        assert len(logged) == 2
        assert "2 more not logged" in logged[1].getMessage()
        assert logged[1].extra_data["suppressed"] == 2  # type: ignore[attr-defined]
        assert pool.counters.no_key_available == 4

    async def test_a_raise_whose_states_changed_is_logged_inside_the_minute(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The last key going from short of room to exhausted is logged at once, so the newest line
        never shows a state the pool has left."""
        clock = ManualClock()
        pool = make_pool(spec("a"), spec("b", "1"), clock=clock)
        await fail_on(pool, revoked())
        await serve(pool, amount="0.6")
        for _ in range(2):  # too big for b's room; the second, with the same states, is suppressed
            with pytest.raises(NoKeyAvailableError):
                await pool.acquire(MODEL, Decimal("0.5"))
        clock.advance(seconds=10)
        await fail_on(pool, exhausted(VerdictScope.KEY))
        with pytest.raises(NoKeyAvailableError):
            await pool.acquire(MODEL, Decimal("0.01"))
        logged = lines(caplog, logging.ERROR, "no key in pool")
        assert len(logged) == 2
        assert "a: revoked; b: no room" in logged[0].getMessage()
        assert "a: revoked; b: exhausted" in logged[1].getMessage()
        assert "1 more not logged" in logged[1].getMessage()
        assert pool.counters.no_key_available == 3

    async def test_the_dedup_is_per_pool(self, caplog: pytest.LogCaptureFixture) -> None:
        clock = ManualClock()
        first, second = make_pool(spec("a"), clock=clock, name="one"), make_pool(spec("a"), clock=clock, name="two")
        for pool in (first, second):
            await fail_on(pool, revoked())
            with pytest.raises(NoKeyAvailableError):
                await pool.acquire(MODEL, Decimal("0.01"))
        assert len(lines(caplog, logging.ERROR, "no key in pool")) == 2


class TestStoreDown:
    """KP-G5: a store that can't be reached fails the call closed, never unbudgeted."""

    async def test_a_store_down_from_the_start(self, caplog: pytest.LogCaptureFixture) -> None:
        store = FakeFlakyStore()
        store.down = True
        pool = make_pool(spec("a"), store=store)
        with pytest.raises(KeyPoolStoreError) as caught:
            await pool.acquire(MODEL, Decimal("0.01"))
        assert isinstance(caught.value.__cause__, ConnectionError)
        assert pool.counters.store_failures == {"read_record": 1}
        assert len(lines(caplog, logging.ERROR, "fails closed")) == 1

    @pytest.mark.parametrize("operation", ["reserve", "sweep_expired", "read_record", "compare_and_set_record"])
    async def test_a_store_failing_mid_call_fails_the_acquire_closed(self, operation: str) -> None:
        clock = ManualClock()
        store = FakeFlakyStore()
        pool = make_pool(spec("a"), spec("b"), store=store, clock=clock)
        await serve(pool)
        clock.advance(minutes=2)  # the sweep and the record read are both due
        if operation == "compare_and_set_record":
            await fail_on(pool, rate_limited())
        store.failing[operation] = ConnectionError("unreachable")
        with pytest.raises(KeyPoolStoreError):
            await (fail_on(pool, exhausted()) if operation == "compare_and_set_record" else serve(pool))
        assert pool.counters.store_failures[operation] == 1

    async def test_a_settle_while_down_fails_and_the_reservation_stays_held(self) -> None:
        store = FakeFlakyStore()
        pool = make_pool(spec("a", "10"), store=store)
        lease = await pool.acquire(MODEL, Decimal("1"))
        store.down = True
        with pytest.raises(KeyPoolStoreError):
            await lease.settle(Decimal("0.5"))
        assert not lease.closed
        store.down = False
        assert (await pool.status()).keys[0].reserved == Decimal("1")
        await lease.settle(Decimal("0.5"))
        assert (await pool.status()).keys[0].spent == Decimal("0.5")

    async def test_status_while_down_raises(self) -> None:
        store = FakeFlakyStore()
        pool = make_pool(spec("a"), store=store)
        await serve(pool)
        store.down = True
        with pytest.raises(KeyPoolStoreError):
            await pool.status()

    async def test_a_bare_key_error_from_a_settle_is_a_store_failure_not_a_missing_reservation(self) -> None:
        store = FakeFlakyStore()
        pool = make_pool(spec("a"), store=store)
        lease = await pool.acquire(MODEL, Decimal("0.01"))
        store.failing["settle"] = KeyError("account")
        with pytest.raises(KeyPoolStoreError) as caught:
            await lease.settle(Decimal("0.01"))
        assert not isinstance(caught.value, LookupError)
        assert isinstance(caught.value.__cause__, KeyError)
        assert pool.counters.store_failures == {"settle": 1}

    async def test_a_stores_own_not_found_passes_through_as_the_contract_says(self) -> None:
        class LedgerNotFound(LookupError):
            pass

        store = FakeFlakyStore()
        pool = make_pool(spec("a"), store=store)
        lease = await pool.acquire(MODEL, Decimal("0.01"))
        store.failing["settle"] = LedgerNotFound("no such reservation")
        with pytest.raises(LedgerNotFound):
            await lease.settle(Decimal("0.01"))
        assert not pool.counters.store_failures

    async def test_a_lookup_error_from_a_reserve_is_a_store_failure(self) -> None:
        class LedgerNotFound(LookupError):
            pass

        store = FakeFlakyStore()
        pool = make_pool(spec("a"), store=store)
        await serve(pool)
        store.failing["reserve"] = LedgerNotFound("a reserve has no not-found")
        with pytest.raises(KeyPoolStoreError):
            await serve(pool)

    async def test_an_unreadable_record_is_a_store_failure(self) -> None:
        store = FakeFlakyStore()
        store.failing["read_record"] = ValueError("Expecting value: line 1 column 1")
        with pytest.raises(KeyPoolStoreError):
            await make_pool(spec("a"), store=store).acquire(MODEL, Decimal("0.01"))

    async def test_a_record_in_another_layout_is_refused_not_overwritten(self) -> None:
        store = FakeFlakyStore()
        await store.inner.compare_and_set_record(f"key-pool/{PROVIDER}/{POOL}", 0, {"layout": 99, "keys": {}})
        with pytest.raises(KeyPoolStoreError, match="layout"):
            await make_pool(spec("a"), store=store).acquire(MODEL, Decimal("0.01"))
        assert (await store.inner.read_record(f"key-pool/{PROVIDER}/{POOL}"))[1] == {"layout": 99, "keys": {}}


_SENTINEL = "sk-ant-api03-SENTINEL-1f2e3d4c5b6a"
_ROTATED = "sk-ant-api03-SENTINEL-rotated-99887766"


class TestSecrecy:
    """KP-B3 for the pool core: a planted secret, driven through every state and error path, appears
    in no log line, exception, ``status()``, store argument or store result; nor does its fingerprint."""

    async def test_the_sentinel_and_its_fingerprint_appear_nowhere(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.DEBUG)
        clock = ManualClock()
        store = FakeFlakyStore()
        registry, monotonic = tripping_registry()
        keys = (spec("s", "1", secret=_SENTINEL), spec("t", "1"))
        pool = make_pool(
            *keys, store=store, clock=clock, breaker_registry=registry, reservation_margin=timedelta(minutes=1)
        )
        rendered: list[str] = [repr(pool), repr(keys), str(keys)]

        async def expect(call: object, *errors: type[BaseException]) -> None:
            with pytest.raises(errors) as caught:
                await call  # type: ignore[misc]
            rendered.extend([str(caught.value), repr(caught.value), "".join(traceback.format_exception(caught.value))])

        lease = await pool.acquire(MODEL, Decimal("0.5"))
        rendered.append(repr(lease))
        await lease.settle(Decimal("0.5"))
        await expect(lease.settle(Decimal("0.1")), ReservationNotFound)
        big = await pool.acquire(MODEL, Decimal("0.6"))  # skips s for room, runs on t
        await big.release()
        await pool.acquire(MODEL, Decimal("0.49"))  # left open: the sweep orphans it
        clock.advance(minutes=2)
        for verdict in (rate_limited(), forbidden(), revoked(), exhausted()):
            await pool.reset("s")
            await pool.reset("t")
            await fail_on(pool, verdict)
        await pool.reset("s")
        await pool.reset("t")
        registry.get(PROVIDER, credential=_SENTINEL).record_failure()
        await serve(pool)
        await fail_on(pool, revoked())
        await expect(pool.acquire(MODEL, Decimal("0.01")), NoKeyAvailableError)
        monotonic.now += 60
        rendered.append(repr(await pool.status()))
        store.down = True
        await expect(pool.acquire(MODEL, Decimal("0.01")), KeyPoolStoreError)
        store.down = False
        rotated = make_pool(spec("s", "2", secret=_ROTATED), store=store, clock=clock)
        rendered.append(repr(await rotated.status()))

        _, record = await store.inner.read_record(f"key-pool/{PROVIDER}/{POOL}")
        assert record is not None
        fingerprints = [
            generation["fingerprint"]
            for entry in record["keys"].values()
            for generation in entry["generations"].values()
        ]
        assert len(fingerprints) == 3
        logged = [record_.getMessage() for record_ in caplog.records] + [
            repr(getattr(record_, "extra_data", None)) for record_ in caplog.records
        ]
        logged += [
            "".join(traceback.format_exception(*record_.exc_info)) for record_ in caplog.records if record_.exc_info
        ]
        assert any("moved its active key" in line for line in logged)
        assert any("orphaned" in line for line in logged)
        for secret in (_SENTINEL, _ROTATED, "SENTINEL"):
            for where, texts in (("logs", logged), ("rendered", rendered), ("store", store.seen)):
                assert not [text for text in texts if secret in text], f"{secret!r} leaked into {where}"
        for fingerprint in fingerprints:
            for where, texts in (("logs", logged), ("rendered", rendered)):
                assert not [text for text in texts if fingerprint in text], f"a fingerprint leaked into {where}"
            account_arguments = [text for text in store.seen if fingerprint in text and "fingerprint" not in text]
            assert not account_arguments, "a fingerprint leaked into an account name"
