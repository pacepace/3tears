"""tests for :class:`IdempotencyKeyStore`: claim-once-with-expiry over the claims table.

The contract this pins (unchanged by the move off file-backed KV; only the wiring changed, and
the bucket-configuration cases became row-level ones):

- a fresh key claims and returns status="claimed" with a pending record;
- the SAME key claimed again returns status="exists" with the original claimer's record;
- claim is atomic under concurrency: N concurrent claimers, exactly one gets "claimed";
- complete()/fail() transition a claimed key to a terminal state and store the result/error,
  retrying under compare-and-swap contention;
- complete()/fail() raise IdempotencyKeyNotFound for a key that was never claimed;
- get() returns None for an unknown key, and the current record otherwise;
- a claim carries the expiry its ttl implies, so an expired claim is claimable again;
- the claim survives a broker wipe once flushed, which is what leaving file-backed KV bought.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.coordination import (
    IdempotencyConflict,
    IdempotencyKeyNotFound,
    IdempotencyKeyStore,
)
from threetears.core.coordination.tables import CoordinationClaimsCollection, coordination_collection
from threetears.core.exceptions import ConcurrentModificationError
from threetears.core.testing.kv import FakeNatsClient

_SCOPE = "claims-principal"


class _Nats(FakeNatsClient):
    """the shared collections bucket; no replica here subscribes to invalidations."""

    async def publish(self, *, subject: Any, message: Any, reply_to: Any = None) -> None:
        return None


class _Store:
    """an in-process L3 for the durable half."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}

    async def fetch_one(self, table: str, pk: dict[str, Any], *, conn: Any = None) -> dict[str, Any] | None:
        del table, conn
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


def _registry(nats: _Nats | None = None, store: _Store | None = None) -> CollectionRegistry:
    l1 = SQLiteBackend(db_name=f"claims_{uuid.uuid4().hex[:8]}")
    registry = CollectionRegistry()
    registry.configure(
        l1_backend=l1,
        l2_client=nats,
        l3_pool=store,  # type: ignore[arg-type]
        kv_key_scope=_SCOPE if nats is not None else None,
    )
    return registry


def _store(registry: CollectionRegistry | None = None, **kwargs: Any) -> IdempotencyKeyStore:
    return IdempotencyKeyStore(registry or _registry(_Nats()), purpose=kwargs.pop("purpose", "jobs"), **kwargs)


def _claims(registry: CollectionRegistry) -> CoordinationClaimsCollection:
    return coordination_collection(registry, CoordinationClaimsCollection)


class TestClaim:
    @pytest.mark.asyncio
    async def test_fresh_key_is_claimed(self) -> None:
        store = _store()
        outcome = await store.claim("key-1")
        assert outcome.status == "claimed"
        assert outcome.record.key == "key-1"
        assert outcome.record.status == "pending"
        assert outcome.record.result is None
        assert outcome.record.error is None
        assert outcome.record.date_completed is None

    @pytest.mark.asyncio
    async def test_same_key_twice_second_call_sees_existing_claim(self) -> None:
        store = _store()
        first = await store.claim("key-1")
        second = await store.claim("key-1")
        assert first.status == "claimed"
        assert second.status == "exists"
        assert second.record.status == "pending"

    @pytest.mark.asyncio
    async def test_distinct_keys_each_claimed(self) -> None:
        store = _store()
        assert (await store.claim("a")).status == "claimed"
        assert (await store.claim("b")).status == "claimed"

    @pytest.mark.asyncio
    async def test_distinct_purposes_do_not_collide(self) -> None:
        registry = _registry(_Nats())
        exports = _store(registry, purpose="exports")
        imports = _store(registry, purpose="imports")
        assert (await exports.claim("same-id")).status == "claimed"
        assert (await imports.claim("same-id")).status == "claimed"

    @pytest.mark.asyncio
    async def test_concurrent_claim_of_same_key_has_one_winner(self) -> None:
        store = _store(purpose="race")
        outcomes = await asyncio.gather(*[store.claim("dup") for _ in range(8)])
        statuses = [o.status for o in outcomes]
        assert statuses.count("claimed") == 1
        assert statuses.count("exists") == 7

    @pytest.mark.asyncio
    async def test_a_second_claimer_across_replicas_sees_the_existing_claim(self) -> None:
        nats, durable = _Nats(), _Store()
        first = _store(_registry(nats, durable))
        second = _store(_registry(nats, durable))
        assert (await first.claim("key-1")).status == "claimed"
        assert (await second.claim("key-1")).status == "exists"

    @pytest.mark.asyncio
    async def test_exists_returns_existing_claimers_completed_result(self) -> None:
        store = _store()
        await store.claim("key-1")
        await store.complete("key-1", result=b"done")
        outcome = await store.claim("key-1")
        assert outcome.status == "exists"
        assert outcome.record.status == "completed"
        assert outcome.record.result == b"done"

    @pytest.mark.asyncio
    async def test_claim_stores_metadata(self) -> None:
        outcome = await _store().claim("key-1", metadata=b"request-hash-abc")
        assert outcome.record.metadata == b"request-hash-abc"

    @pytest.mark.asyncio
    async def test_claim_without_metadata_defaults_to_none(self) -> None:
        assert (await _store().claim("key-1")).record.metadata is None

    @pytest.mark.asyncio
    async def test_exists_outcome_returns_original_claimers_metadata(self) -> None:
        store = _store()
        await store.claim("key-1", metadata=b"request-hash-abc")
        outcome = await store.claim("key-1", metadata=b"different-hash")
        assert outcome.status == "exists"
        assert outcome.record.metadata == b"request-hash-abc"  # the original claimer's

    @pytest.mark.asyncio
    async def test_an_expired_claim_is_claimable_again(self) -> None:
        # the expiry is what stops one abandoned claim blocking an operation forever. It is the
        # row's own column, so every tier treats a passed claim as absent.
        store = _store(ttl=timedelta(milliseconds=100))
        assert (await store.claim("key-1")).status == "claimed"
        await asyncio.sleep(0.15)  # the row's expiry is read against the wall clock at every tier
        assert await store.get("key-1") is None
        assert (await store.claim("key-1")).status == "claimed", "an expired claim blocked the operation"

    @pytest.mark.asyncio
    async def test_a_claim_carries_the_expiry_its_ttl_implies(self) -> None:
        nats, durable = _Nats(), _Store()
        registry = _registry(nats, durable)
        store = _store(registry, ttl=timedelta(hours=6))
        await store.claim("key-1")
        await _claims(registry).aclose()  # write-behind: flush first
        row = next(iter(durable.rows.values()))
        assert row["expires_at"] - row["date_claimed"] == timedelta(hours=6)

    @pytest.mark.asyncio
    async def test_a_ttl_of_none_never_expires(self) -> None:
        nats, durable = _Nats(), _Store()
        registry = _registry(nats, durable)
        store = _store(registry, ttl=None)
        await store.claim("key-1")
        await _claims(registry).aclose()
        assert next(iter(durable.rows.values()))["expires_at"] is None

    @pytest.mark.asyncio
    async def test_a_flushed_claim_survives_a_broker_wipe(self) -> None:
        nats, durable = _Nats(), _Store()
        registry = _registry(nats, durable)
        store = _store(registry)
        await store.claim("key-1")
        await store.complete("key-1", result=b"done")
        await _claims(registry).aclose()
        (await nats.kv_bucket(name="collections")).wipe()
        outcome = await _store(_registry(nats, durable)).claim("key-1")
        assert outcome.status == "exists", "a wipe let a completed operation run a second time"
        assert outcome.record.result == b"done"


class TestComplete:
    @pytest.mark.asyncio
    async def test_complete_stores_result_and_terminal_state(self) -> None:
        store = _store()
        await store.claim("key-1")
        await store.complete("key-1", result=b"the-result")
        record = await store.get("key-1")
        assert record is not None
        assert record.status == "completed"
        assert record.result == b"the-result"
        assert record.error is None
        assert record.date_completed is not None

    @pytest.mark.asyncio
    async def test_complete_preserves_original_date_claimed(self) -> None:
        store = _store()
        original = (await store.claim("key-1")).record.date_claimed
        await store.complete("key-1", result=b"x")
        record = await store.get("key-1")
        assert record is not None
        assert record.date_claimed == original

    @pytest.mark.asyncio
    async def test_complete_preserves_claim_time_metadata(self) -> None:
        store = _store()
        await store.claim("key-1", metadata=b"request-hash-abc")
        await store.complete("key-1", result=b"x")
        record = await store.get("key-1")
        assert record is not None
        assert record.metadata == b"request-hash-abc"

    @pytest.mark.asyncio
    async def test_complete_unclaimed_key_raises(self) -> None:
        with pytest.raises(IdempotencyKeyNotFound):
            await _store().complete("never-claimed", result=b"x")

    @pytest.mark.asyncio
    async def test_complete_retries_under_contention_and_the_last_writer_wins(self) -> None:
        # two callers transitioning one claim is a caller-side bug, but the transition must still
        # be a compare-and-swap rather than a blind overwrite of whatever it read.
        store = _store()
        await store.claim("k")
        await asyncio.gather(store.complete("k", result=b"a"), store.fail("k", error="b"))
        record = await store.get("k")
        assert record is not None
        assert record.status in {"completed", "failed"}

    @pytest.mark.asyncio
    async def test_complete_raises_a_conflict_when_the_budget_is_exhausted(self) -> None:
        # a lost compare-and-swap budget is reported as this primitive's own error, so a caller
        # catching IdempotencyConflict does not also have to know the collection's.
        store = _store()
        await store.claim("k")

        async def _always_lose(*args: Any, **kwargs: Any) -> Any:
            raise ConcurrentModificationError("coordination_claims", ("jobs", "k"), datetime.now(UTC))

        store._collection.l2_cas_mutate = _always_lose  # type: ignore[method-assign]  # noqa: SLF001
        with pytest.raises(IdempotencyConflict):
            await store.complete("k", result=b"result")


class TestFail:
    @pytest.mark.asyncio
    async def test_fail_stores_error_and_terminal_state(self) -> None:
        store = _store()
        await store.claim("key-1")
        await store.fail("key-1", error="boom")
        record = await store.get("key-1")
        assert record is not None
        assert record.status == "failed"
        assert record.error == "boom"
        assert record.result is None

    @pytest.mark.asyncio
    async def test_fail_unclaimed_key_raises(self) -> None:
        with pytest.raises(IdempotencyKeyNotFound):
            await _store().fail("never-claimed", error="boom")


class TestGet:
    @pytest.mark.asyncio
    async def test_get_unknown_key_returns_none(self) -> None:
        assert await _store().get("unknown") is None

    @pytest.mark.asyncio
    async def test_get_pending_key_returns_pending_record(self) -> None:
        store = _store()
        await store.claim("key-1")
        record = await store.get("key-1")
        assert record is not None
        assert record.status == "pending"


class TestConstruction:
    def test_an_empty_purpose_is_refused(self) -> None:
        with pytest.raises(ValueError, match="purpose"):
            _store(purpose=" ")

    def test_a_non_positive_ttl_is_refused(self) -> None:
        with pytest.raises(ValueError, match="ttl"):
            _store(ttl=timedelta(0))

    @pytest.mark.asyncio
    async def test_a_claim_made_without_a_ttl_expires_in_24_hours(self) -> None:
        # asserted through a written row, not by reading the constant back: the default only
        # means anything if a claim carries it.
        nats, durable = _Nats(), _Store()
        registry = _registry(nats, durable)
        store = _store(registry)  # no ttl argument
        await store.claim("key-1")
        await _claims(registry).aclose()
        row = next(iter(durable.rows.values()))
        assert row["expires_at"] - row["date_claimed"] == timedelta(hours=24)

    def test_a_registry_with_no_l2_is_refused(self) -> None:
        # "claimed" versus "exists" is the compare-and-swap; without L2 two replicas that both
        # read absent would both do the work, which is what a claim exists to prevent.
        with pytest.raises(ValueError, match="needs an L2 client"):
            _store(_registry(None, _Store()))

    def test_every_store_over_the_table_shares_one_collection(self) -> None:
        registry = _registry(_Nats())
        first = _store(registry, purpose="a")
        second = _store(registry, purpose="b")
        assert first._collection is second._collection  # noqa: SLF001
