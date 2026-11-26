"""tests for :class:`BudgetLedger`: a durable budget ledger and record store over the coordination tables.

The contract is ``threetears.models.key_pool.store.KeyPoolStore``'s, which the in-memory store there
keeps; this pins the same behaviour on the durable one:

- a reservation is admitted only while ``spent + reserved + amount <= budget``, and concurrent
  reservations -- on one ledger or on two replicas -- never together pass the budget;
- settle replaces a reservation with its actual cost, above or below; release drops it; an unknown or
  closed id is refused as a ``LookupError``;
- the expiry sweep settles expired reservations at their full amount, reports them orphaned in the
  order they were made, and a late settle or release corrects one exactly once;
- records: a first write, a conflict handing back the winner, a document that is not JSON refused;
- every change is in L3 before the call returns, so a broker wipe loses no spend;
- a registry with no L2 is refused: the compare-and-swap is the guarantee.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.l2_order import l2_order_of
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.coordination import BudgetLedger
from threetears.core.coordination.tables import CoordinationLedgerAccountsCollection, coordination_collection
from threetears.core.testing.kv import FakeNatsClient

_SCOPE = "ledger-principal"
_NOW = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
_LATER = _NOW + timedelta(minutes=5)


class _Nats(FakeNatsClient):
    """the shared collections bucket; no replica here subscribes to invalidations."""

    async def publish(self, *, subject: Any, message: Any, reply_to: Any = None) -> None:
        return None


class _Store:
    """an in-process L3: rows by table and key, written only over an older stored order."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.statements: list[str] = []
        self.refuse_writes: Exception | None = None

    async def fetch_one(self, table: str, pk: dict[str, Any], *, conn: Any = None) -> dict[str, Any] | None:
        del conn
        row = self.rows.get((table, pk["purpose"], pk["key"]))
        return dict(row) if row is not None else None

    async def upsert(self, table: str, data: dict[str, Any], **kwargs: Any) -> int:
        del kwargs
        if self.refuse_writes is not None:
            raise self.refuse_writes
        self.rows[(table, data["purpose"], data["key"])] = dict(data)
        return 1

    async def upsert_ordered(self, table: str, data: dict[str, Any], *, conn: Any = None) -> int:
        # the conditional write the SQL backend generates: land only over an older stored order.
        del conn
        stored = self.rows.get((table, data["purpose"], data["key"]))
        stored_order = None if stored is None else l2_order_of(stored)
        incoming = l2_order_of(data)
        if stored_order is not None and incoming is not None and stored_order >= incoming:
            return 0
        return await self.upsert(table, data)

    async def delete(self, table: str, pk: dict[str, Any], *, conn: Any = None) -> None:
        del conn
        self.rows.pop((table, pk["purpose"], pk["key"]), None)

    async def scan(self, table: str, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        del filters
        return [dict(row) for (name, _, _), row in self.rows.items() if name == table]

    async def execute(self, query: str, *params: Any, namespace: str | None = None) -> str:
        del params, namespace
        self.statements.append(query)
        return "DELETE 0"


def _registry(nats: _Nats | None, store: _Store | None) -> CollectionRegistry:
    l1 = SQLiteBackend(db_name=f"ledger_{uuid.uuid4().hex[:8]}")
    registry = CollectionRegistry()
    registry.configure(
        l1_backend=l1,
        l2_client=nats,
        l3_pool=store,  # type: ignore[arg-type]
        kv_key_scope=_SCOPE if nats is not None else None,
    )
    return registry


def _ledger(registry: CollectionRegistry | None = None, *, purpose: str = "pool") -> BudgetLedger:
    return BudgetLedger(registry or _registry(_Nats(), _Store()), purpose=purpose)


class TestReserve:
    @pytest.mark.asyncio
    async def test_a_reservation_that_fits_is_held(self) -> None:
        ledger = _ledger()
        assert await ledger.reserve("acct", "r1", Decimal("3.50"), Decimal("10"), _LATER) is True
        snap = (await ledger.snapshot(["acct"]))["acct"]
        assert snap["spent"] == Decimal(0)
        assert snap["reserved"] == Decimal("3.50")
        assert snap["open_reservations"] == {"r1": {"amount": Decimal("3.50"), "expires_at": _LATER}}
        assert snap["orphaned"] == {}

    @pytest.mark.asyncio
    async def test_a_reservation_exactly_at_the_budget_fits_and_one_cent_over_does_not(self) -> None:
        ledger = _ledger()
        assert await ledger.reserve("acct", "r1", Decimal("6.00"), Decimal("10.00"), _LATER)
        assert await ledger.reserve("acct", "r2", Decimal("4.01"), Decimal("10.00"), _LATER) is False
        assert await ledger.reserve("acct", "r3", Decimal("4.00"), Decimal("10.00"), _LATER) is True
        snap = (await ledger.snapshot(["acct"]))["acct"]
        assert list(snap["open_reservations"]) == ["r1", "r3"], "a refused reservation held something"

    @pytest.mark.asyncio
    async def test_concurrent_reservations_never_exceed_the_budget(self) -> None:
        ledger = _ledger()
        results = await asyncio.gather(
            *(ledger.reserve("acct", f"r{i}", Decimal(1), Decimal(4), _LATER) for i in range(10))
        )
        assert sum(results) == 4, f"admitted {sum(results)} of 10 against a budget that fits 4"
        assert (await ledger.snapshot(["acct"]))["acct"]["reserved"] == Decimal(4)

    @pytest.mark.asyncio
    async def test_concurrent_reservations_across_two_replicas_never_exceed_the_budget(self) -> None:
        # two registries over one broker and one database stand in for two processes
        nats, durable = _Nats(), _Store()
        first, second = _ledger(_registry(nats, durable)), _ledger(_registry(nats, durable))
        results = await asyncio.gather(
            *((first if i % 2 else second).reserve("acct", f"r{i}", Decimal(1), Decimal(5), _LATER) for i in range(12))
        )
        assert sum(results) == 5
        for ledger in (first, second):
            assert (await ledger.snapshot(["acct"]))["acct"]["reserved"] == Decimal(5)

    @pytest.mark.asyncio
    async def test_settled_spend_counts_against_the_next_reservation(self) -> None:
        ledger = _ledger()
        await ledger.reserve("acct", "r1", Decimal(5), Decimal(10), _LATER)
        await ledger.settle("acct", "r1", Decimal(8))
        assert await ledger.reserve("acct", "r2", Decimal(3), Decimal(10), _LATER) is False
        assert await ledger.reserve("acct", "r3", Decimal(2), Decimal(10), _LATER) is True

    @pytest.mark.asyncio
    async def test_an_expired_reservation_no_sweep_closed_still_counts(self) -> None:
        ledger = _ledger()
        await ledger.reserve("acct", "old", Decimal(5), Decimal(5), _NOW - timedelta(hours=1))
        assert await ledger.reserve("acct", "new", Decimal(1), Decimal(5), _LATER) is False

    @pytest.mark.asyncio
    async def test_the_budget_is_read_from_each_call_not_stored(self) -> None:
        ledger = _ledger()
        await ledger.reserve("acct", "r1", Decimal(5), Decimal(5), _LATER)
        assert await ledger.reserve("acct", "r2", Decimal(5), Decimal(5), _LATER) is False
        assert await ledger.reserve("acct", "r2", Decimal(5), Decimal(10), _LATER) is True

    @pytest.mark.asyncio
    async def test_reusing_an_open_id_is_refused(self) -> None:
        ledger = _ledger()
        await ledger.reserve("acct", "r1", Decimal(1), Decimal(10), _LATER)
        with pytest.raises(ValueError, match="already holds reservation 'r1'"):
            await ledger.reserve("acct", "r1", Decimal(1), Decimal(10), _LATER)

    @pytest.mark.asyncio
    async def test_reusing_an_orphaned_id_is_refused(self) -> None:
        ledger = _ledger()
        await ledger.reserve("acct", "r1", Decimal(1), Decimal(10), _NOW)
        await ledger.sweep_expired(["acct"], _NOW)
        with pytest.raises(ValueError, match="already holds"):
            await ledger.reserve("acct", "r1", Decimal(1), Decimal(10), _LATER)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("amount", "budget"),
        [
            (1.5, Decimal(10)),
            (Decimal(1), 10),
            (Decimal(-1), Decimal(10)),
            (Decimal("NaN"), Decimal(10)),
            (Decimal(1), Decimal("Infinity")),
        ],
    )
    async def test_money_that_is_not_a_finite_non_negative_decimal_is_refused(self, amount: Any, budget: Any) -> None:
        with pytest.raises(ValueError, match="must be"):
            await _ledger().reserve("acct", "r1", amount, budget, _LATER)

    @pytest.mark.asyncio
    async def test_a_naive_expiry_is_refused(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            await _ledger().reserve("acct", "r1", Decimal(1), Decimal(10), datetime(2026, 11, 26, 12))

    @pytest.mark.asyncio
    async def test_purposes_keep_their_accounts_apart(self) -> None:
        registry = _registry(_Nats(), _Store())
        pool, other = _ledger(registry, purpose="pool"), _ledger(registry, purpose="other")
        await pool.reserve("acct", "r1", Decimal(5), Decimal(5), _LATER)
        assert await other.reserve("acct", "r1", Decimal(5), Decimal(5), _LATER) is True


class TestSettleAndRelease:
    @pytest.mark.asyncio
    async def test_settle_above_the_reservation_records_the_truth(self) -> None:
        ledger = _ledger()
        await ledger.reserve("acct", "r1", Decimal("2.00"), Decimal(10), _LATER)
        await ledger.settle("acct", "r1", Decimal("2.75"))
        snap = (await ledger.snapshot(["acct"]))["acct"]
        assert (snap["spent"], snap["reserved"], snap["open_reservations"]) == (Decimal("2.75"), Decimal(0), {})

    @pytest.mark.asyncio
    async def test_settle_below_the_reservation_frees_the_difference(self) -> None:
        ledger = _ledger()
        await ledger.reserve("acct", "r1", Decimal(8), Decimal(10), _LATER)
        await ledger.settle("acct", "r1", Decimal("0.25"))
        assert (await ledger.snapshot(["acct"]))["acct"]["spent"] == Decimal("0.25")
        assert await ledger.reserve("acct", "r2", Decimal("9.75"), Decimal(10), _LATER) is True

    @pytest.mark.asyncio
    async def test_release_drops_the_reservation_with_no_spend(self) -> None:
        ledger = _ledger()
        await ledger.reserve("acct", "r1", Decimal(8), Decimal(10), _LATER)
        await ledger.release("acct", "r1")
        snap = (await ledger.snapshot(["acct"]))["acct"]
        assert (snap["spent"], snap["reserved"], snap["open_reservations"]) == (Decimal(0), Decimal(0), {})

    @pytest.mark.asyncio
    async def test_an_unknown_id_is_refused(self) -> None:
        ledger = _ledger()
        await ledger.reserve("acct", "r1", Decimal(1), Decimal(10), _LATER)
        with pytest.raises(LookupError, match="holds no reservation 'nope'"):
            await ledger.settle("acct", "nope", Decimal(1))
        with pytest.raises(LookupError):
            await ledger.release("acct", "nope")
        with pytest.raises(LookupError):
            await ledger.settle("never-seen", "r1", Decimal(1))

    @pytest.mark.asyncio
    async def test_a_second_settle_or_release_is_refused(self) -> None:
        ledger = _ledger()
        await ledger.reserve("acct", "r1", Decimal(1), Decimal(10), _LATER)
        await ledger.settle("acct", "r1", Decimal(1))
        with pytest.raises(LookupError):
            await ledger.settle("acct", "r1", Decimal(1))
        with pytest.raises(LookupError):
            await ledger.release("acct", "r1")
        assert (await ledger.snapshot(["acct"]))["acct"]["spent"] == Decimal(1), "a refused settle moved spend"

    @pytest.mark.asyncio
    async def test_a_negative_settle_is_refused(self) -> None:
        ledger = _ledger()
        await ledger.reserve("acct", "r1", Decimal(1), Decimal(10), _LATER)
        with pytest.raises(ValueError, match="not negative"):
            await ledger.settle("acct", "r1", Decimal(-1))


class TestSweep:
    @pytest.mark.asyncio
    async def test_the_sweep_settles_expired_reservations_in_full_and_reports_them_in_order(self) -> None:
        ledger = _ledger()
        await ledger.reserve("acct", "b", Decimal(2), Decimal(100), _NOW - timedelta(seconds=1))
        await ledger.reserve("acct", "live", Decimal(5), Decimal(100), _LATER)
        await ledger.reserve("acct", "a", Decimal(3), Decimal(100), _NOW)
        await ledger.reserve("other", "c", Decimal(1), Decimal(100), _LATER)

        swept = await ledger.sweep_expired(["acct", "other", "never-seen"], _NOW)

        assert swept == {"acct": ["b", "a"]}, "orphans reported out of reservation order, or for the wrong accounts"
        snap = (await ledger.snapshot(["acct"]))["acct"]
        assert snap["spent"] == Decimal(5)
        assert snap["reserved"] == Decimal(5)
        assert list(snap["open_reservations"]) == ["live"]
        assert snap["orphaned"] == {"b": Decimal(2), "a": Decimal(3)}
        assert await ledger.sweep_expired(["acct"], _NOW) == {}, "a second sweep orphaned something again"

    @pytest.mark.asyncio
    async def test_a_late_settle_replaces_the_swept_amount_once(self) -> None:
        ledger = _ledger()
        await ledger.reserve("acct", "r1", Decimal(10), Decimal(100), _NOW)
        await ledger.sweep_expired(["acct"], _NOW)
        await ledger.settle("acct", "r1", Decimal(4))
        snap = (await ledger.snapshot(["acct"]))["acct"]
        assert (snap["spent"], snap["orphaned"]) == (Decimal(4), {})
        with pytest.raises(LookupError):
            await ledger.settle("acct", "r1", Decimal(4))

    @pytest.mark.asyncio
    async def test_a_late_release_takes_the_swept_amount_back_out(self) -> None:
        ledger = _ledger()
        await ledger.reserve("acct", "r1", Decimal(10), Decimal(100), _NOW)
        await ledger.sweep_expired(["acct"], _NOW)
        await ledger.release("acct", "r1")
        snap = (await ledger.snapshot(["acct"]))["acct"]
        assert (snap["spent"], snap["orphaned"]) == (Decimal(0), {})

    @pytest.mark.asyncio
    async def test_a_naive_now_and_a_bare_account_string_are_refused(self) -> None:
        ledger = _ledger()
        with pytest.raises(ValueError, match="timezone-aware"):
            await ledger.sweep_expired(["acct"], datetime(2026, 11, 26))
        with pytest.raises(TypeError, match="not the one string"):
            await ledger.sweep_expired("acct", _NOW)
        with pytest.raises(TypeError, match="not the one string"):
            await ledger.snapshot("acct")


class TestSnapshot:
    @pytest.mark.asyncio
    async def test_an_unseen_account_reads_as_zero(self) -> None:
        assert await _ledger().snapshot(["new"]) == {
            "new": {"spent": Decimal(0), "reserved": Decimal(0), "open_reservations": {}, "orphaned": {}}
        }


class TestRecords:
    @pytest.mark.asyncio
    async def test_an_absent_record_reads_as_version_zero(self) -> None:
        assert await _ledger().read_record("active") == (0, None)

    @pytest.mark.asyncio
    async def test_a_first_write_then_a_second_raise_the_version(self) -> None:
        ledger = _ledger()
        doc = {"active": "key-a", "states": {"key-a": {"state": "ok", "budget": "50.00"}}, "n": [1, 2.5, None]}
        assert await ledger.compare_and_set_record("pool", 0, doc) == (True, 1, doc)
        assert await ledger.read_record("pool") == (1, doc)
        assert await ledger.compare_and_set_record("pool", 1, {"active": "key-b"}) == (True, 2, {"active": "key-b"})
        assert await ledger.read_record("pool") == (2, {"active": "key-b"})

    @pytest.mark.asyncio
    async def test_a_conflict_hands_back_the_winner(self) -> None:
        nats, durable = _Nats(), _Store()
        first, second = _ledger(_registry(nats, durable)), _ledger(_registry(nats, durable))
        await first.compare_and_set_record("pool", 0, {"active": "a"})
        assert await first.compare_and_set_record("pool", 1, {"active": "b"}) == (True, 2, {"active": "b"})
        assert await second.compare_and_set_record("pool", 1, {"active": "c"}) == (False, 2, {"active": "b"})
        assert await second.compare_and_set_record("absent", 3, {"x": 1}) == (False, 0, None)

    @pytest.mark.asyncio
    async def test_concurrent_first_writes_have_one_winner(self) -> None:
        ledger = _ledger()
        results = await asyncio.gather(*(ledger.compare_and_set_record("pool", 0, {"by": i}) for i in range(6)))
        winners = [result for result in results if result[0]]
        assert len(winners) == 1
        assert all(result[1:] == winners[0][1:] for result in results), "a loser was handed something not the winner"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "document",
        [
            {"t": (1, 2)},
            {1: "int key"},
            {"f": float("nan")},
            {"s": {1, 2}},
            {"d": Decimal("1.5")},
            ["not", "a", "dict"],
        ],
    )
    async def test_a_document_that_is_not_json_safe_is_refused(self, document: Any) -> None:
        ledger = _ledger()
        with pytest.raises(ValueError, match="document must"):
            await ledger.compare_and_set_record("pool", 0, document)
        assert await ledger.read_record("pool") == (0, None)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("version", [-1, True, 1.0])
    async def test_a_bad_expected_version_is_refused(self, version: Any) -> None:
        with pytest.raises(ValueError, match="expected_version"):
            await _ledger().compare_and_set_record("pool", version, {})


class TestDurability:
    @pytest.mark.asyncio
    async def test_every_change_is_in_l3_before_the_call_returns(self) -> None:
        nats, durable = _Nats(), _Store()
        ledger = _ledger(_registry(nats, durable))
        await ledger.reserve("acct", "r1", Decimal("1.10"), Decimal(10), _LATER)
        row = durable.rows[("coordination_ledger_accounts", "pool", "acct")]
        assert row["reserved"] == "1.10", "a synchronous ledger returned before L3 held the reservation"
        await ledger.compare_and_set_record("pool", 0, {"active": "a"})
        assert durable.rows[("coordination_documents", "pool", "pool")]["version"] == 1

    @pytest.mark.asyncio
    async def test_spend_survives_a_broker_wipe(self) -> None:
        nats, durable = _Nats(), _Store()
        ledger = _ledger(_registry(nats, durable))
        await ledger.reserve("acct", "r1", Decimal(6), Decimal(10), _LATER)
        await ledger.settle("acct", "r1", Decimal(7))
        await ledger.reserve("acct", "r2", Decimal(1), Decimal(10), _LATER)
        await ledger.compare_and_set_record("pool", 0, {"active": "a"})

        (await nats.kv_bucket(name="collections")).wipe()
        fresh = _ledger(_registry(nats, durable))

        snap = (await fresh.snapshot(["acct"]))["acct"]
        assert (snap["spent"], snap["reserved"]) == (Decimal(7), Decimal(1)), "a wipe lost spend"
        assert await fresh.reserve("acct", "r3", Decimal(3), Decimal(10), _LATER) is False
        await fresh.settle("acct", "r2", Decimal(1))
        assert (await fresh.snapshot(["acct"]))["acct"]["spent"] == Decimal(8)
        assert await fresh.read_record("pool") == (1, {"active": "a"})

    @pytest.mark.asyncio
    async def test_a_failed_persist_raises_and_holds_nothing(self) -> None:
        # the caller is told the reserve failed, so neither tier may keep answering with it
        nats, durable = _Nats(), _Store()
        ledger = _ledger(_registry(nats, durable))
        durable.refuse_writes = ConnectionError("database unreachable")
        with pytest.raises(ConnectionError):
            await ledger.reserve("acct", "r1", Decimal(5), Decimal(5), _LATER)
        durable.refuse_writes = None
        assert (await ledger.snapshot(["acct"]))["acct"]["reserved"] == Decimal(0)
        assert await ledger.reserve("acct", "r2", Decimal(5), Decimal(5), _LATER) is True

    @pytest.mark.asyncio
    async def test_the_expiry_sweep_leaves_the_never_expiring_tables_alone(self) -> None:
        durable = _Store()
        registry = _registry(_Nats(), durable)
        await _ledger(registry).reserve("acct", "r1", Decimal(1), Decimal(10), _LATER)
        accounts = coordination_collection(registry, CoordinationLedgerAccountsCollection)
        assert await accounts.sweep_expired() == 0
        assert not [s for s in durable.statements if "DELETE" in s], "a sweep ran over rows that never expire"


class TestConstruction:
    def test_a_registry_with_no_l2_is_refused(self) -> None:
        with pytest.raises(ValueError, match="needs an L2 client"):
            _ledger(_registry(None, _Store()))

    def test_an_empty_purpose_is_refused(self) -> None:
        with pytest.raises(ValueError, match="purpose"):
            _ledger(purpose="  ")
