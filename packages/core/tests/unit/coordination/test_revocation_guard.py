"""tests for the durable denylists: :class:`RevocationGuard` and :class:`RedemptionLedger`.

The revocation contract this pins (unchanged by the move off file-backed KV; only the wiring
changed, and the bucket-configuration cases became row-level ones):

- a recorded revocation reads back; an unrevoked key has no stored moment and denylists nothing;
- something that started BEFORE the revocation is blocked, at or AFTER it is unaffected;
- distinct keys are independent, and keys are hashed before storage;
- a second record overwrites the stored moment rather than being refused;
- reads and writes fail CLOSED: a storage failure propagates so the caller denies;
- a naive datetime is refused on both sides of the comparison;
- a revocation survives a broker wipe, which is what leaving file-backed KV bought;
- the common answer -- not revoked -- costs no L3 query after the first, per write generation.

The ledger contract:

- the first sighting of a key is fresh and every later one is reuse, across replicas;
- a spent key survives a broker wipe (the reason a 30-day jti ledger cannot be a ReplayGuard);
- distinct purposes and distinct keys do not collide.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.coordination import RedemptionLedger, RevocationGuard
from threetears.core.coordination.tables import CoordinationRevocationsCollection, coordination_collection
from threetears.core.exceptions import DataLayerUnavailableError
from threetears.core.testing.kv import FakeNatsClient
from threetears.nats import KvError

_SCOPE = "denylist-principal"

#: a revocation is remembered for a ttl measured FROM the revocation, so a moment far in the past
#: is an entry that expired before it was written. These tests stand near the wall clock for the
#: same reason production does.
_NOW = datetime.now(UTC).replace(microsecond=0)


class _Nats(FakeNatsClient):
    """the shared collections bucket; no replica here subscribes to invalidations."""

    async def publish(self, *, subject: Any, message: Any, reply_to: Any = None) -> None:
        return None


class _FailingNats(FakeNatsClient):
    """a client whose bucket refuses every operation, the way an unreachable broker does."""

    async def kv_bucket(self, **kwargs: Any) -> Any:
        del kwargs
        raise KvError("kv down")

    async def publish(self, *, subject: Any, message: Any, reply_to: Any = None) -> None:
        return None


class _Store:
    """an in-process L3, counting the reads a denylist check costs."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}
        self.fetches = 0

    async def fetch_one(self, table: str, pk: dict[str, Any], *, conn: Any = None) -> dict[str, Any] | None:
        del table, conn
        self.fetches += 1
        row = self.rows.get((pk["purpose"], pk["key"]))
        return dict(row) if row is not None else None

    async def upsert(self, table: str, data: dict[str, Any], **kwargs: Any) -> int:
        del table, kwargs
        self.rows[(data["purpose"], data["key"])] = dict(data)
        return 1

    async def delete(self, table: str, pk: dict[str, Any], *, conn: Any = None) -> None:
        del table, conn
        self.rows.pop((pk["purpose"], pk["key"]), None)

    async def scan(self, table: str, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        del table, filters
        return [dict(row) for row in self.rows.values()]

    async def execute(self, query: str, *params: Any, namespace: str | None = None) -> str:
        del query, params, namespace
        return "DELETE 0"


# parity-with: threetears.core.collections.generation.GenerationSource
class _FakeGenerations:
    """the write generation an absence is stamped with; one per fleet in these tests."""

    def __init__(self) -> None:
        self.count = 0

    async def current(self, table_name: str) -> str:
        del table_name
        return f"i:{self.count}"

    async def advance(self, table_name: str) -> None:
        del table_name
        self.count += 1


def _registry(
    nats: Any = None,
    store: _Store | None = None,
    generations: _FakeGenerations | None = None,
) -> CollectionRegistry:
    l1 = SQLiteBackend(db_name=f"denylist_{uuid.uuid4().hex[:8]}")
    registry = CollectionRegistry()
    registry.configure(
        l1_backend=l1,
        l2_client=nats,
        l3_pool=store,  # type: ignore[arg-type]
        kv_key_scope=_SCOPE if nats is not None else None,
    )
    registry.set_generation_source(generations or _FakeGenerations())
    return registry


def _guard(registry: CollectionRegistry | None = None, **kwargs: Any) -> RevocationGuard:
    return RevocationGuard(
        registry or _registry(_Nats(), _Store()),
        purpose=kwargs.pop("purpose", "standing"),
        ttl_seconds=kwargs.pop("ttl_seconds", 3600),
        **kwargs,
    )


def _ledger(registry: CollectionRegistry | None = None, **kwargs: Any) -> RedemptionLedger:
    return RedemptionLedger(
        registry or _registry(_Nats(), _Store()),
        purpose=kwargs.pop("purpose", "refresh_jti"),
        ttl_seconds=kwargs.pop("ttl_seconds", 30 * 86400),
        **kwargs,
    )


class TestRevocationGuard:
    @pytest.mark.asyncio
    async def test_record_and_read_back_round_trips(self) -> None:
        guard = _guard()
        await guard.record_revocation("sub:p1", revoked_at=_NOW)
        assert await guard.revoked_at("sub:p1") == _NOW

    @pytest.mark.asyncio
    async def test_unrevoked_key_has_no_stored_timestamp(self) -> None:
        assert await _guard().revoked_at("sub:never") is None

    @pytest.mark.asyncio
    async def test_unrevoked_key_is_not_revoked_before_anything(self) -> None:
        assert await _guard().is_revoked_before("sub:never", moment=_NOW) is False

    @pytest.mark.asyncio
    async def test_session_started_before_revocation_is_blocked(self) -> None:
        guard = _guard()
        await guard.record_revocation("sub:p1", revoked_at=_NOW)
        assert await guard.is_revoked_before("sub:p1", moment=_NOW - timedelta(minutes=5)) is True

    @pytest.mark.asyncio
    async def test_session_started_after_revocation_is_unaffected(self) -> None:
        # otherwise every future login to a once-revoked principal is permanently denylisted,
        # which is not what recovery or offboarding mean.
        guard = _guard()
        await guard.record_revocation("sub:p1", revoked_at=_NOW)
        assert await guard.is_revoked_before("sub:p1", moment=_NOW + timedelta(minutes=5)) is False

    @pytest.mark.asyncio
    async def test_session_started_exactly_at_revocation_is_unaffected(self) -> None:
        guard = _guard()
        await guard.record_revocation("sub:p1", revoked_at=_NOW)
        assert await guard.is_revoked_before("sub:p1", moment=_NOW) is False

    @pytest.mark.asyncio
    async def test_distinct_keys_do_not_affect_each_other(self) -> None:
        guard = _guard()
        await guard.record_revocation("sub:p1", revoked_at=_NOW)
        assert await guard.is_revoked_before("sub:p2", moment=_NOW - timedelta(days=1)) is False

    @pytest.mark.asyncio
    async def test_distinct_purposes_do_not_collide(self) -> None:
        registry = _registry(_Nats(), _Store())
        sessions = _guard(registry, purpose="sessions")
        customers = _guard(registry, purpose="customers")
        await sessions.record_revocation("id-1", revoked_at=_NOW)
        assert await customers.revoked_at("id-1") is None

    @pytest.mark.asyncio
    async def test_second_record_call_overwrites_the_stored_moment(self) -> None:
        # an operator re-running offboarding, or narrowing an earlier cutoff.
        guard = _guard()
        await guard.record_revocation("sub:p1", revoked_at=_NOW)
        # narrowed within the ttl: a cutoff moved further back than the ttl is an entry whose
        # blocked sessions have all ended anyway, which is the point of measuring from it.
        narrowed = _NOW - timedelta(minutes=5)
        await guard.record_revocation("sub:p1", revoked_at=narrowed)
        assert await guard.revoked_at("sub:p1") == narrowed

    @pytest.mark.asyncio
    async def test_hashed_keys_keep_similar_keys_distinct(self) -> None:
        store = _Store()
        guard = _guard(_registry(_Nats(), store))
        await guard.record_revocation("sub:p1", revoked_at=_NOW)
        await guard.record_revocation("sub:p10", revoked_at=_NOW + timedelta(minutes=1))
        assert len(store.rows) == 2
        assert all("sub:p1" not in key for _purpose, key in store.rows)

    @pytest.mark.asyncio
    async def test_a_revocation_survives_a_broker_wipe(self) -> None:
        # the whole reason this left a memory-backed cache tier: losing it makes a revoked
        # session valid again.
        nats, store, generations = _Nats(), _Store(), _FakeGenerations()
        guard = _guard(_registry(nats, store, generations))
        await guard.record_revocation("sub:p1", revoked_at=_NOW)
        (await nats.kv_bucket(name="collections")).wipe()
        fresh = _guard(_registry(nats, store, generations))
        assert await fresh.is_revoked_before("sub:p1", moment=_NOW - timedelta(minutes=5)) is True

    @pytest.mark.asyncio
    async def test_the_common_answer_reaches_l3_once_per_generation(self) -> None:
        # nearly every check is of a key nobody revoked, and that answer must not be a database
        # query every time.
        nats, store, generations = _Nats(), _Store(), _FakeGenerations()
        reader = _guard(_registry(nats, store, generations))
        peer = _guard(_registry(nats, store, generations))
        assert await reader.is_revoked_before("sub:quiet", moment=_NOW) is False
        for _ in range(5):
            assert await reader.is_revoked_before("sub:quiet", moment=_NOW) is False
        assert await peer.is_revoked_before("sub:quiet", moment=_NOW) is False
        assert store.fetches == 1, "a recorded absence was not reused"

    @pytest.mark.asyncio
    async def test_a_revocation_supersedes_a_recorded_absence(self) -> None:
        nats, store, generations = _Nats(), _Store(), _FakeGenerations()
        reader = _guard(_registry(nats, store, generations))
        writer = _guard(_registry(nats, store, generations))
        assert await reader.is_revoked_before("sub:p1", moment=_NOW - timedelta(minutes=5)) is False
        await writer.record_revocation("sub:p1", revoked_at=_NOW)
        assert await reader.is_revoked_before("sub:p1", moment=_NOW - timedelta(minutes=5)) is True, (
            "a recorded absence hid a revocation"
        )

    @pytest.mark.asyncio
    async def test_an_l2_outage_does_not_stop_a_revocation_being_recorded_or_read(self) -> None:
        # what fail-closed meant while L2 held the truth, and what it means now that L3 does.
        # With the broker unreachable, the write still commits to L3 and the check still reads it,
        # so the answer is right rather than merely refused.
        guard = _guard(_registry(_FailingNats(), _Store()))
        await guard.record_revocation("sub:p1", revoked_at=_NOW)
        assert await guard.is_revoked_before("sub:p1", moment=_NOW - timedelta(minutes=5)) is True

    @pytest.mark.asyncio
    async def test_an_l3_failure_propagates_so_the_caller_denies(self) -> None:
        # the failure that does matter: without L3 there is no answer, and "no answer" must never
        # read as "not revoked".
        store = _Store()
        guard = _guard(_registry(_Nats(), store))

        async def _refuse(*args: Any, **kwargs: Any) -> Any:
            raise DataLayerUnavailableError("L3 down")

        store.fetch_one = _refuse  # type: ignore[method-assign]
        with pytest.raises(DataLayerUnavailableError):
            await guard.is_revoked_before("sub:p1", moment=_NOW)

    @pytest.mark.asyncio
    async def test_naive_revoked_at_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            await _guard().record_revocation("sub:p1", revoked_at=datetime(2026, 1, 1, 12, 0))  # noqa: DTZ001

    @pytest.mark.asyncio
    async def test_naive_moment_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            await _guard().is_revoked_before("sub:p1", moment=datetime(2026, 1, 1, 12, 0))  # noqa: DTZ001

    @pytest.mark.asyncio
    async def test_the_entry_expires_a_ttl_after_the_revocation_not_the_write(self) -> None:
        # measured from the revocation: re-recording must not extend how long it is remembered
        # past the sessions it exists to block.
        store = _Store()
        guard = _guard(_registry(_Nats(), store), ttl_seconds=3600)
        await guard.record_revocation("sub:p1", revoked_at=_NOW)
        row = next(iter(store.rows.values()))
        assert row["expires_at"] == _NOW + timedelta(hours=1)

    @pytest.mark.asyncio
    async def test_recording_a_revocation_drives_this_table_s_sweep(self) -> None:
        # the sweep is per collection instance, so a sibling primitive sweeping the redemptions
        # table does nothing here. Without a driver of its own, expired revocation rows -- ttl
        # 400 days in identity -- would accumulate in L3 forever.
        store = _Store()
        registry = _registry(_Nats(), store)
        guard = _guard(registry)
        collection = coordination_collection(registry, CoordinationRevocationsCollection)
        swept: list[str] = []

        async def _record_sweep() -> int:
            swept.append(collection.table_name)
            return 0

        collection.sweep_expired_if_due = _record_sweep  # type: ignore[method-assign]
        await guard.record_revocation("sub:p1", revoked_at=_NOW)
        assert swept == ["coordination_revocations"], "nothing sweeps the revocations table"

    def test_a_non_positive_ttl_is_refused(self) -> None:
        with pytest.raises(ValueError, match="ttl_seconds"):
            _guard(ttl_seconds=0)

    def test_an_empty_purpose_is_refused(self) -> None:
        with pytest.raises(ValueError, match="purpose"):
            _guard(purpose="  ")


class TestRedemptionLedger:
    @pytest.mark.asyncio
    async def test_the_first_sighting_is_fresh_and_the_second_is_reuse(self) -> None:
        ledger = _ledger()
        assert await ledger.record_unique("jti-1") is True
        assert await ledger.record_unique("jti-1") is False

    @pytest.mark.asyncio
    async def test_distinct_keys_are_independent(self) -> None:
        ledger = _ledger()
        assert await ledger.record_unique("jti-1") is True
        assert await ledger.record_unique("jti-2") is True

    @pytest.mark.asyncio
    async def test_exactly_one_concurrent_caller_is_told_it_was_first(self) -> None:
        ledger = _ledger()
        results = await asyncio.gather(*[ledger.record_unique("jti-dup") for _ in range(8)])
        assert results.count(True) == 1
        assert results.count(False) == 7

    @pytest.mark.asyncio
    async def test_a_replica_that_never_saw_the_first_sighting_still_refuses_the_second(self) -> None:
        nats, store = _Nats(), _Store()
        first = _ledger(_registry(nats, store))
        second = _ledger(_registry(nats, store))
        assert await first.record_unique("jti-1") is True
        assert await second.record_unique("jti-1") is False

    @pytest.mark.asyncio
    async def test_a_spent_key_survives_a_broker_wipe(self) -> None:
        # the reason a 30-day jti ledger cannot be a ReplayGuard: a wipe must not make every
        # outstanding token spendable again, and must not refuse them all either.
        nats, store = _Nats(), _Store()
        ledger = _ledger(_registry(nats, store))
        assert await ledger.record_unique("jti-1") is True
        (await nats.kv_bucket(name="collections")).wipe()
        fresh = _ledger(_registry(nats, store))
        assert await fresh.record_unique("jti-1") is False, "a wipe made a spent token spendable"
        assert await fresh.record_unique("jti-never-seen") is True, "a wipe refused an unspent token"

    @pytest.mark.asyncio
    async def test_was_redeemed_reports_without_recording(self) -> None:
        ledger = _ledger()
        assert await ledger.was_redeemed("jti-1") is False
        assert await ledger.record_unique("jti-1") is True
        assert await ledger.was_redeemed("jti-1") is True

    @pytest.mark.asyncio
    async def test_distinct_purposes_do_not_collide(self) -> None:
        registry = _registry(_Nats(), _Store())
        refresh = _ledger(registry, purpose="refresh_jti")
        exchange = _ledger(registry, purpose="exchange_jti")
        assert await refresh.record_unique("shared-id") is True
        assert await exchange.record_unique("shared-id") is True

    @pytest.mark.asyncio
    async def test_the_key_is_hashed_before_storage(self) -> None:
        store = _Store()
        ledger = _ledger(_registry(_Nats(), store))
        await ledger.record_unique("jti-secret-value")
        assert all("jti-secret-value" not in key for _purpose, key in store.rows)

    @pytest.mark.asyncio
    async def test_the_entry_expires_a_ttl_after_it_was_recorded(self) -> None:
        store = _Store()
        ledger = _ledger(_registry(_Nats(), store), ttl_seconds=3600)
        before = datetime.now(UTC)
        await ledger.record_unique("jti-1")
        row = next(iter(store.rows.values()))
        assert before + timedelta(hours=1) <= row["expires_at"] <= datetime.now(UTC) + timedelta(hours=1)

    @pytest.mark.asyncio
    async def test_an_l2_outage_propagates_because_the_fence_is_gone(self) -> None:
        # unlike the revocation guard's read, a redemption's compare-and-swap IS the L2 fence:
        # without it two callers could both be told they were first, so the failure must reach
        # the caller rather than degrade.
        ledger = _ledger(_registry(_FailingNats(), _Store()))
        with pytest.raises(KvError):
            await ledger.record_unique("jti-1")

    def test_a_non_positive_ttl_is_refused(self) -> None:
        with pytest.raises(ValueError, match="ttl_seconds"):
            _ledger(ttl_seconds=0)

    def test_an_empty_purpose_is_refused(self) -> None:
        with pytest.raises(ValueError, match="purpose"):
            _ledger(purpose="")

    def test_a_registry_with_no_l2_is_refused(self) -> None:
        # without L2 the compare-and-swap degrades to a read-modify-write, and two replicas that
        # both read absent are both told they were first -- the one thing this primitive may not
        # do. A guarantee that quietly does not hold is worse than a process that will not start.
        with pytest.raises(ValueError, match="needs an L2 client"):
            _ledger(_registry(None, _Store()))
