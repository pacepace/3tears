"""tests for the key pool store: the ledger's reserve, settle, release and sweep, and versioned records."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from threetears.models.errors import ModelProviderError
from threetears.models.key_pool.store import (
    InMemoryKeyPoolStore,
    KeyPoolStore,
    KeyPoolStoreError,
    ReservationAlreadyOpen,
    ReservationNotFound,
)

_ACCOUNT = "pool/primary/2026-11"
_NOW = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
_LATER = _NOW + timedelta(minutes=5)


@pytest.fixture
def store() -> InMemoryKeyPoolStore:
    return InMemoryKeyPoolStore()


async def _spent(store: InMemoryKeyPoolStore, account: str = _ACCOUNT) -> Decimal:
    return (await store.snapshot([account]))[account]["spent"]


def test_the_in_memory_store_satisfies_the_protocol() -> None:
    assert isinstance(InMemoryKeyPoolStore(), KeyPoolStore)


def test_store_errors_are_not_provider_errors() -> None:
    """a store failure never reaches the provider, so it isn't reported as one."""
    assert not issubclass(KeyPoolStoreError, ModelProviderError)
    assert issubclass(ReservationNotFound, KeyPoolStoreError)
    assert issubclass(ReservationNotFound, LookupError)
    assert issubclass(ReservationAlreadyOpen, KeyPoolStoreError)
    assert issubclass(ReservationAlreadyOpen, ValueError)


class TestReserve:
    async def test_a_new_account_starts_at_zero(self, store: InMemoryKeyPoolStore) -> None:
        snap = (await store.snapshot([_ACCOUNT]))[_ACCOUNT]
        assert snap == {"spent": Decimal(0), "reserved": Decimal(0), "open_reservations": {}, "orphaned": {}}

    async def test_a_reservation_that_fits_is_held(self, store: InMemoryKeyPoolStore) -> None:
        assert await store.reserve(_ACCOUNT, "r1", Decimal("0.40"), Decimal("1.00"), _LATER)
        snap = (await store.snapshot([_ACCOUNT]))[_ACCOUNT]
        assert snap["reserved"] == Decimal("0.40")
        assert snap["spent"] == Decimal(0)
        assert snap["open_reservations"] == {"r1": {"amount": Decimal("0.40"), "expires_at": _LATER}}

    async def test_exactly_the_budget_fits(self, store: InMemoryKeyPoolStore) -> None:
        assert await store.reserve(_ACCOUNT, "r1", Decimal("0.60"), Decimal("1.00"), _LATER)
        assert await store.reserve(_ACCOUNT, "r2", Decimal("0.40"), Decimal("1.00"), _LATER)

    async def test_one_cent_over_is_refused_and_holds_nothing(self, store: InMemoryKeyPoolStore) -> None:
        assert await store.reserve(_ACCOUNT, "r1", Decimal("0.60"), Decimal("1.00"), _LATER)
        assert not await store.reserve(_ACCOUNT, "r2", Decimal("0.41"), Decimal("1.00"), _LATER)
        snap = (await store.snapshot([_ACCOUNT]))[_ACCOUNT]
        assert snap["reserved"] == Decimal("0.60")
        assert list(snap["open_reservations"]) == ["r1"]

    async def test_spend_counts_against_the_budget(self, store: InMemoryKeyPoolStore) -> None:
        assert await store.reserve(_ACCOUNT, "r1", Decimal("0.50"), Decimal("1.00"), _LATER)
        await store.settle(_ACCOUNT, "r1", Decimal("0.90"))
        assert not await store.reserve(_ACCOUNT, "r2", Decimal("0.20"), Decimal("1.00"), _LATER)
        assert await store.reserve(_ACCOUNT, "r3", Decimal("0.10"), Decimal("1.00"), _LATER)

    async def test_a_raised_budget_applies_at_once(self, store: InMemoryKeyPoolStore) -> None:
        assert await store.reserve(_ACCOUNT, "r1", Decimal("1.00"), Decimal("1.00"), _LATER)
        assert not await store.reserve(_ACCOUNT, "r2", Decimal("1.00"), Decimal("1.00"), _LATER)
        assert await store.reserve(_ACCOUNT, "r2", Decimal("1.00"), Decimal("2.00"), _LATER)

    async def test_accounts_are_independent(self, store: InMemoryKeyPoolStore) -> None:
        """the next period's account starts at zero however much the last one spent."""
        assert await store.reserve(_ACCOUNT, "r1", Decimal("1.00"), Decimal("1.00"), _LATER)
        await store.settle(_ACCOUNT, "r1", Decimal("1.00"))
        assert await store.reserve("pool/primary/2026-12", "r1", Decimal("1.00"), Decimal("1.00"), _LATER)

    async def test_reusing_an_open_id_raises(self, store: InMemoryKeyPoolStore) -> None:
        assert await store.reserve(_ACCOUNT, "r1", Decimal("0.10"), Decimal("1.00"), _LATER)
        with pytest.raises(ReservationAlreadyOpen) as raised:
            await store.reserve(_ACCOUNT, "r1", Decimal("0.10"), Decimal("1.00"), _LATER)
        assert raised.value.reservation_id == "r1"
        assert (await store.snapshot([_ACCOUNT]))[_ACCOUNT]["reserved"] == Decimal("0.10")

    async def test_reusing_an_orphaned_id_raises(self, store: InMemoryKeyPoolStore) -> None:
        assert await store.reserve(_ACCOUNT, "r1", Decimal("0.10"), Decimal("1.00"), _NOW)
        await store.sweep_expired([_ACCOUNT], _NOW)
        with pytest.raises(ReservationAlreadyOpen):
            await store.reserve(_ACCOUNT, "r1", Decimal("0.10"), Decimal("1.00"), _LATER)

    async def test_an_id_settled_and_closed_may_be_reserved_again(self, store: InMemoryKeyPoolStore) -> None:
        assert await store.reserve(_ACCOUNT, "r1", Decimal("0.10"), Decimal("1.00"), _LATER)
        await store.settle(_ACCOUNT, "r1", Decimal("0.10"))
        assert await store.reserve(_ACCOUNT, "r1", Decimal("0.10"), Decimal("1.00"), _LATER)

    @pytest.mark.parametrize(
        "amount",
        [Decimal("-0.01"), Decimal("NaN"), Decimal("Infinity"), 0.5],
    )
    async def test_invalid_money_raises(self, store: InMemoryKeyPoolStore, amount: Decimal) -> None:
        with pytest.raises(ValueError, match="amount"):
            await store.reserve(_ACCOUNT, "r1", amount, Decimal("1.00"), _LATER)
        with pytest.raises(ValueError, match="budget"):
            await store.reserve(_ACCOUNT, "r1", Decimal("0.10"), amount, _LATER)

    async def test_a_naive_expiry_raises(self, store: InMemoryKeyPoolStore) -> None:
        with pytest.raises(ValueError, match="expires_at"):
            await store.reserve(_ACCOUNT, "r1", Decimal("0.10"), Decimal("1.00"), datetime(2026, 11, 26))


class TestConcurrentReserve:
    async def test_concurrent_tasks_never_exceed_the_budget(self, store: InMemoryKeyPoolStore) -> None:
        """twenty calls of 0.30 against a budget of 1.00: exactly three fit."""
        budget = Decimal("1.00")
        results = await asyncio.gather(
            *(store.reserve(_ACCOUNT, f"r{i}", Decimal("0.30"), budget, _LATER) for i in range(20))
        )
        assert results.count(True) == 3
        snap = (await store.snapshot([_ACCOUNT]))[_ACCOUNT]
        assert snap["reserved"] == Decimal("0.90")
        assert len(snap["open_reservations"]) == 3

        await asyncio.gather(*(store.settle(_ACCOUNT, rid, Decimal("0.30")) for rid in snap["open_reservations"]))
        after = (await store.snapshot([_ACCOUNT]))[_ACCOUNT]
        assert after["spent"] == Decimal("0.90") <= budget
        assert after["reserved"] == Decimal(0)

    def test_concurrent_threads_never_exceed_the_budget(self, store: InMemoryKeyPoolStore) -> None:
        """a pool driven from worker threads, each with its own event loop, shares one ledger."""
        budget = Decimal("1.00")
        start = threading.Barrier(16)

        def reserve(i: int) -> bool:
            start.wait()
            return asyncio.run(store.reserve(_ACCOUNT, f"r{i}", Decimal("0.07"), budget, _LATER))

        with ThreadPoolExecutor(max_workers=16) as workers:
            results = list(workers.map(reserve, range(16)))
        assert results.count(True) == 14
        assert asyncio.run(store.snapshot([_ACCOUNT]))[_ACCOUNT]["reserved"] == Decimal("0.98")


class TestSettleAndRelease:
    async def test_settle_below_the_reservation_frees_the_difference(self, store: InMemoryKeyPoolStore) -> None:
        assert await store.reserve(_ACCOUNT, "r1", Decimal("0.80"), Decimal("1.00"), _LATER)
        await store.settle(_ACCOUNT, "r1", Decimal("0.25"))
        snap = (await store.snapshot([_ACCOUNT]))[_ACCOUNT]
        assert snap["spent"] == Decimal("0.25")
        assert snap["reserved"] == Decimal(0)
        assert snap["open_reservations"] == {}
        assert await store.reserve(_ACCOUNT, "r2", Decimal("0.75"), Decimal("1.00"), _LATER)

    async def test_settle_above_the_reservation_records_the_truth(self, store: InMemoryKeyPoolStore) -> None:
        """settlement may take an account past its budget: the estimate was low, and spend is what was billed."""
        assert await store.reserve(_ACCOUNT, "r1", Decimal("0.80"), Decimal("1.00"), _LATER)
        await store.settle(_ACCOUNT, "r1", Decimal("1.30"))
        assert await _spent(store) == Decimal("1.30")
        assert not await store.reserve(_ACCOUNT, "r2", Decimal(0), Decimal("1.00"), _LATER)

    async def test_release_holds_and_spends_nothing(self, store: InMemoryKeyPoolStore) -> None:
        assert await store.reserve(_ACCOUNT, "r1", Decimal("0.80"), Decimal("1.00"), _LATER)
        await store.release(_ACCOUNT, "r1")
        snap = (await store.snapshot([_ACCOUNT]))[_ACCOUNT]
        assert snap["spent"] == Decimal(0)
        assert snap["reserved"] == Decimal(0)
        assert snap["open_reservations"] == {}

    async def test_a_second_settle_raises(self, store: InMemoryKeyPoolStore) -> None:
        assert await store.reserve(_ACCOUNT, "r1", Decimal("0.80"), Decimal("1.00"), _LATER)
        await store.settle(_ACCOUNT, "r1", Decimal("0.50"))
        with pytest.raises(ReservationNotFound) as raised:
            await store.settle(_ACCOUNT, "r1", Decimal("0.50"))
        assert (raised.value.account, raised.value.reservation_id) == (_ACCOUNT, "r1")
        assert await _spent(store) == Decimal("0.50")

    async def test_release_after_settle_raises(self, store: InMemoryKeyPoolStore) -> None:
        assert await store.reserve(_ACCOUNT, "r1", Decimal("0.80"), Decimal("1.00"), _LATER)
        await store.settle(_ACCOUNT, "r1", Decimal("0.50"))
        with pytest.raises(ReservationNotFound):
            await store.release(_ACCOUNT, "r1")
        assert await _spent(store) == Decimal("0.50")

    async def test_a_second_release_raises(self, store: InMemoryKeyPoolStore) -> None:
        assert await store.reserve(_ACCOUNT, "r1", Decimal("0.80"), Decimal("1.00"), _LATER)
        await store.release(_ACCOUNT, "r1")
        with pytest.raises(ReservationNotFound):
            await store.release(_ACCOUNT, "r1")

    async def test_unknown_ids_raise(self, store: InMemoryKeyPoolStore) -> None:
        with pytest.raises(ReservationNotFound):
            await store.settle(_ACCOUNT, "never", Decimal("0.10"))
        with pytest.raises(ReservationNotFound):
            await store.release(_ACCOUNT, "never")

    async def test_an_id_on_another_account_is_unknown(self, store: InMemoryKeyPoolStore) -> None:
        assert await store.reserve(_ACCOUNT, "r1", Decimal("0.10"), Decimal("1.00"), _LATER)
        with pytest.raises(ReservationNotFound):
            await store.settle("pool/backup/2026-11", "r1", Decimal("0.10"))

    async def test_settling_with_invalid_money_raises_and_keeps_the_reservation(
        self, store: InMemoryKeyPoolStore
    ) -> None:
        assert await store.reserve(_ACCOUNT, "r1", Decimal("0.10"), Decimal("1.00"), _LATER)
        with pytest.raises(ValueError, match="amount"):
            await store.settle(_ACCOUNT, "r1", Decimal("-1"))
        assert list((await store.snapshot([_ACCOUNT]))[_ACCOUNT]["open_reservations"]) == ["r1"]


class TestSweepExpired:
    async def test_expired_reservations_are_spent_in_full_and_reported(self, store: InMemoryKeyPoolStore) -> None:
        other = "pool/backup/2026-11"
        assert await store.reserve(_ACCOUNT, "old", Decimal("0.30"), Decimal("1.00"), _NOW - timedelta(seconds=1))
        assert await store.reserve(_ACCOUNT, "due", Decimal("0.20"), Decimal("1.00"), _NOW)
        assert await store.reserve(_ACCOUNT, "live", Decimal("0.10"), Decimal("1.00"), _LATER)
        assert await store.reserve(other, "live", Decimal("0.10"), Decimal("1.00"), _LATER)

        orphaned = await store.sweep_expired([_ACCOUNT, other, "pool/unseen/2026-11"], _NOW)

        assert orphaned == {_ACCOUNT: ["old", "due"]}
        snap = (await store.snapshot([_ACCOUNT]))[_ACCOUNT]
        assert snap["spent"] == Decimal("0.50")
        assert snap["reserved"] == Decimal("0.10")
        assert list(snap["open_reservations"]) == ["live"]
        assert snap["orphaned"] == {"old": Decimal("0.30"), "due": Decimal("0.20")}

    async def test_a_second_sweep_reports_nothing_new(self, store: InMemoryKeyPoolStore) -> None:
        assert await store.reserve(_ACCOUNT, "old", Decimal("0.30"), Decimal("1.00"), _NOW)
        assert await store.sweep_expired([_ACCOUNT], _NOW) == {_ACCOUNT: ["old"]}
        assert await store.sweep_expired([_ACCOUNT], _LATER) == {}
        assert await _spent(store) == Decimal("0.30")

    async def test_only_named_accounts_are_swept(self, store: InMemoryKeyPoolStore) -> None:
        assert await store.reserve(_ACCOUNT, "old", Decimal("0.30"), Decimal("1.00"), _NOW)
        assert await store.sweep_expired(["pool/backup/2026-11"], _LATER) == {}
        assert (await store.snapshot([_ACCOUNT]))[_ACCOUNT]["reserved"] == Decimal("0.30")

    async def test_an_expired_unswept_reservation_still_holds_budget(self, store: InMemoryKeyPoolStore) -> None:
        assert await store.reserve(_ACCOUNT, "old", Decimal("0.90"), Decimal("1.00"), _NOW)
        assert not await store.reserve(_ACCOUNT, "r2", Decimal("0.20"), Decimal("1.00"), _LATER)

    async def test_a_late_settle_replaces_the_swept_amount(self, store: InMemoryKeyPoolStore) -> None:
        """a slow call outlives its expiry: its true cost replaces the sweep's estimate, once."""
        assert await store.reserve(_ACCOUNT, "slow", Decimal("0.80"), Decimal("1.00"), _NOW)
        await store.sweep_expired([_ACCOUNT], _NOW)
        assert await _spent(store) == Decimal("0.80")

        await store.settle(_ACCOUNT, "slow", Decimal("0.15"))

        snap = (await store.snapshot([_ACCOUNT]))[_ACCOUNT]
        assert snap["spent"] == Decimal("0.15")
        assert snap["orphaned"] == {}
        with pytest.raises(ReservationNotFound):
            await store.settle(_ACCOUNT, "slow", Decimal("0.15"))

    async def test_a_late_settle_above_the_reservation(self, store: InMemoryKeyPoolStore) -> None:
        assert await store.reserve(_ACCOUNT, "slow", Decimal("0.80"), Decimal("1.00"), _NOW)
        await store.sweep_expired([_ACCOUNT], _NOW)
        await store.settle(_ACCOUNT, "slow", Decimal("0.95"))
        assert await _spent(store) == Decimal("0.95")

    async def test_a_late_release_takes_the_swept_amount_back(self, store: InMemoryKeyPoolStore) -> None:
        assert await store.reserve(_ACCOUNT, "slow", Decimal("0.80"), Decimal("1.00"), _NOW)
        await store.sweep_expired([_ACCOUNT], _NOW)
        await store.release(_ACCOUNT, "slow")
        snap = (await store.snapshot([_ACCOUNT]))[_ACCOUNT]
        assert snap["spent"] == Decimal(0)
        assert snap["orphaned"] == {}
        with pytest.raises(ReservationNotFound):
            await store.release(_ACCOUNT, "slow")

    async def test_a_naive_now_raises(self, store: InMemoryKeyPoolStore) -> None:
        with pytest.raises(ValueError, match="now"):
            await store.sweep_expired([_ACCOUNT], datetime(2026, 11, 26))

    async def test_a_bare_account_string_is_refused(self, store: InMemoryKeyPoolStore) -> None:
        with pytest.raises(TypeError, match="accounts"):
            await store.sweep_expired(_ACCOUNT, _NOW)
        with pytest.raises(TypeError, match="accounts"):
            await store.snapshot(_ACCOUNT)


class TestRecords:
    async def test_an_absent_record_is_version_zero(self, store: InMemoryKeyPoolStore) -> None:
        assert await store.read_record("pool") == (0, None)

    async def test_first_write_from_absent(self, store: InMemoryKeyPoolStore) -> None:
        document = {"active": "primary", "keys": {"primary": {"state": "ok", "budget": "10.00"}}}
        assert await store.compare_and_set_record("pool", 0, document) == (True, 1, document)
        assert await store.read_record("pool") == (1, document)

    async def test_a_write_over_the_read_version_succeeds(self, store: InMemoryKeyPoolStore) -> None:
        await store.compare_and_set_record("pool", 0, {"active": "primary"})
        version, _ = await store.read_record("pool")
        assert await store.compare_and_set_record("pool", version, {"active": "backup"}) == (
            True,
            2,
            {"active": "backup"},
        )
        assert await store.read_record("pool") == (2, {"active": "backup"})

    async def test_a_conflict_returns_the_winner(self, store: InMemoryKeyPoolStore) -> None:
        """two processes move the active key off version 1; the loser is handed the winner's choice."""
        await store.compare_and_set_record("pool", 0, {"active": "a"})
        assert await store.compare_and_set_record("pool", 1, {"active": "b"}) == (True, 2, {"active": "b"})
        assert await store.compare_and_set_record("pool", 1, {"active": "c"}) == (False, 2, {"active": "b"})
        assert await store.read_record("pool") == (2, {"active": "b"})

    async def test_a_second_first_write_loses(self, store: InMemoryKeyPoolStore) -> None:
        await store.compare_and_set_record("pool", 0, {"active": "a"})
        assert await store.compare_and_set_record("pool", 0, {"active": "b"}) == (False, 1, {"active": "a"})

    async def test_a_version_ahead_of_an_absent_record_loses(self, store: InMemoryKeyPoolStore) -> None:
        assert await store.compare_and_set_record("pool", 3, {"active": "a"}) == (False, 0, None)

    async def test_records_are_independent_by_name(self, store: InMemoryKeyPoolStore) -> None:
        await store.compare_and_set_record("pool-a", 0, {"active": "a"})
        assert await store.read_record("pool-b") == (0, None)

    async def test_the_caller_holds_copies(self, store: InMemoryKeyPoolStore) -> None:
        document: dict[str, object] = {"keys": {"a": {"state": "ok"}}}
        _, _, written = await store.compare_and_set_record("pool", 0, document)
        document["keys"] = {}
        assert written is not None
        written["keys"] = {}
        _, read = await store.read_record("pool")
        assert read == {"keys": {"a": {"state": "ok"}}}
        assert read is not None
        read["keys"] = {}
        assert await store.read_record("pool") == (1, {"keys": {"a": {"state": "ok"}}})

    @pytest.mark.parametrize(
        "document",
        [
            {"budget": Decimal("1.00")},
            {"until": datetime(2026, 11, 26, tzinfo=UTC)},
            {"held": float("nan")},
            {1: "int key"},
            {"order": ("a", "b")},
        ],
    )
    async def test_a_document_that_isnt_json_safe_raises(
        self, store: InMemoryKeyPoolStore, document: dict[str, object]
    ) -> None:
        with pytest.raises(ValueError, match="JSON"):
            await store.compare_and_set_record("pool", 0, document)
        assert await store.read_record("pool") == (0, None)

    async def test_a_negative_version_raises(self, store: InMemoryKeyPoolStore) -> None:
        with pytest.raises(ValueError, match="expected_version"):
            await store.compare_and_set_record("pool", -1, {})
