"""tests for :class:`CollectionReplayAnchor`: the durable record of when a ledger first existed.

The contract this pins:

- the FIRST caller records its own clock and gets it back; every later caller gets that same
  moment, not its own -- which is what makes the anchor a fact about the ledger rather than
  about whichever replica asked;
- two ledgers anchored through one collection do not share a row;
- the row carries NO expiry, so the expiry sweep cannot remove it. A swept anchor would read as
  a ledger that had never existed, which is the one wrong answer this class must not give;
- a storage failure propagates, because the guard treats "cannot tell" as "apply the watermark"
  and can only do that if the anchor raises rather than guessing;
- it refuses to be built over a registry with no L2 fence, for the reason a redemption does:
  two replicas booting together must not both be told they recorded the first anchor.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.coordination import CollectionReplayAnchor
from threetears.core.coordination.replay_anchor import ANCHOR_KEY
from threetears.core.coordination.tables import CoordinationRedemptionsCollection, coordination_collection
from threetears.core.exceptions import DataLayerUnavailableError
from threetears.core.testing.kv import FakeNatsClient

_SCOPE = "identity-core"


class _Store:
    """an in-memory L3 double, keyed the way the coordination tables are."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}
        self.fail_on_upsert = False

    async def fetch_one(self, table: str, pk: dict[str, Any], *, conn: Any = None) -> dict[str, Any] | None:
        del table, conn
        return self.rows.get((pk["purpose"], pk["key"]))

    async def upsert(self, table: str, data: dict[str, Any], **kwargs: Any) -> int:
        del table, kwargs
        if self.fail_on_upsert:
            raise DataLayerUnavailableError("l3 down")
        self.rows[(data["purpose"], data["key"])] = dict(data)
        return 1

    async def delete(self, table: str, pk: dict[str, Any], *, conn: Any = None) -> None:
        del conn
        self.rows.pop((pk["purpose"], pk["key"]), None)

    async def scan(self, table: str, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        del table, filters
        return list(self.rows.values())

    async def execute(self, query: str, *params: Any, namespace: str | None = None) -> str:
        del query, params, namespace
        return "DELETE 0"


class _FakeGenerations:
    """the write-generation source a negative-caching collection refuses to be built without."""

    def __init__(self) -> None:
        self.count = 0

    async def current(self, table_name: str) -> str:
        del table_name
        return f"gen-{self.count}"

    async def advance(self, table_name: str) -> None:
        del table_name
        self.count += 1


def _registry(nats: Any = None, store: _Store | None = None) -> CollectionRegistry:
    """a registry wired the way a consumer wires its coordination tables.

    :param nats: the L2 client, or ``None`` for a registry with no fence
    :ptype nats: Any
    :param store: the durable tier
    :ptype store: _Store | None
    :return: the configured registry
    :rtype: CollectionRegistry
    """
    registry = CollectionRegistry()
    registry.configure(
        l1_backend=SQLiteBackend(db_name=f"anchor_{uuid.uuid4().hex[:8]}"),
        l2_client=nats,
        l3_pool=store,  # type: ignore[arg-type]
        kv_key_scope=_SCOPE if nats is not None else None,
    )
    registry.set_generation_source(_FakeGenerations())
    return registry


class TestTheFirstCallerSetsTheBirthTime:
    @pytest.mark.asyncio
    async def test_the_first_call_records_and_returns_its_own_clock(self) -> None:
        anchor = CollectionReplayAnchor(_registry(FakeNatsClient(), _Store()))
        now = datetime.now(UTC)
        assert await anchor.first_existed("pop_nonces", now=now) == now

    @pytest.mark.asyncio
    async def test_a_later_caller_gets_the_first_callers_moment_not_its_own(self) -> None:
        # the property the guard depends on: a replica starting hours later must learn when the
        # LEDGER was born, or it would read its own start time as the ledger's and conclude that
        # every wipe was a first run.
        registry = _registry(FakeNatsClient(), _Store())
        first = CollectionReplayAnchor(registry)
        born = datetime.now(UTC) - timedelta(hours=6)
        await first.first_existed("pop_nonces", now=born)

        later = CollectionReplayAnchor(registry)
        assert await later.first_existed("pop_nonces", now=datetime.now(UTC)) == born

    @pytest.mark.asyncio
    async def test_concurrent_first_callers_agree_on_one_moment(self) -> None:
        # two replicas booting together: exactly one write wins, and both must read the winner's
        # moment rather than each keeping its own.
        registry = _registry(FakeNatsClient(), _Store())
        anchors = [CollectionReplayAnchor(registry) for _ in range(4)]
        results = await asyncio.gather(*(a.first_existed("pop_nonces", now=datetime.now(UTC)) for a in anchors))
        assert len(set(results)) == 1, f"replicas disagreed on the ledger's birth time: {results}"

    @pytest.mark.asyncio
    async def test_two_ledgers_do_not_share_an_anchor(self) -> None:
        registry = _registry(FakeNatsClient(), _Store())
        anchor = CollectionReplayAnchor(registry)
        dpop = datetime.now(UTC) - timedelta(hours=2)
        await anchor.first_existed("identity-dpop-nonces", now=dpop)

        assorted = datetime.now(UTC)
        assert await anchor.first_existed("proxy_assertion_nonces", now=assorted) == assorted


class TestTheAnchorOutlivesTheSweep:
    @pytest.mark.asyncio
    async def test_the_row_carries_no_expiry(self) -> None:
        # the sweep deletes only `expires_at IS NOT NULL`, so a NULL expiry is what keeps the
        # anchor permanent. A swept anchor reads as a ledger that never existed, which would
        # make the guard skip the watermark after a real wipe.
        store = _Store()
        anchor = CollectionReplayAnchor(_registry(FakeNatsClient(), store))
        await anchor.first_existed("pop_nonces", now=datetime.now(UTC))
        rows = [row for key, row in store.rows.items() if key[1] == ANCHOR_KEY]
        assert rows, "no anchor row was written"
        assert all(row["expires_at"] is None for row in rows), rows

    @pytest.mark.asyncio
    async def test_the_sweep_leaves_it_alone(self) -> None:
        store = _Store()
        registry = _registry(FakeNatsClient(), store)
        anchor = CollectionReplayAnchor(registry)
        born = datetime.now(UTC) - timedelta(days=30)
        await anchor.first_existed("pop_nonces", now=born)

        collection = coordination_collection(registry, CoordinationRedemptionsCollection, None)
        await collection.sweep_expired(now=datetime.now(UTC))

        assert await anchor.first_existed("pop_nonces", now=datetime.now(UTC)) == born


class TestTheAnchorFailsLoudly:
    @pytest.mark.asyncio
    async def test_a_storage_failure_propagates(self) -> None:
        # the guard's fallback is "apply the watermark", and it can only choose that if the
        # anchor raises. An anchor that guessed would silently retire the watermark.
        store = _Store()
        store.fail_on_upsert = True
        anchor = CollectionReplayAnchor(_registry(FakeNatsClient(), store))
        with pytest.raises(DataLayerUnavailableError):
            await anchor.first_existed("pop_nonces", now=datetime.now(UTC))

    def test_a_registry_with_no_fence_is_refused_at_construction(self) -> None:
        # same reasoning as a redemption: without a compare-and-swap against L2 two replicas can
        # both be told they wrote the first anchor, and the later one's clock would become the
        # ledger's birth time.
        with pytest.raises(Exception, match="CollectionReplayAnchor"):
            CollectionReplayAnchor(_registry(None, _Store()))
