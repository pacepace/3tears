"""An L2 failure a collection degrades on is counted, by reason, and logged without flooding.

Where a durable tier stands behind L2, a refused or unreadable L2 is a cache miss: the read answers
from L3 and the save lands in L3. That fall-through is correct, and used to be silent but for a
WARNING per operation -- a full bucket read exactly like a broker outage, and nothing counted
either. ``threetears_l2_degraded_total{table,op,reason}`` counts each one, which is what an alert
reads, and the WARNING is rate-limited so a sustained failure does not bury every other line.

Where L2 is the store (``l2_is_the_store``), there is nothing to fall through to: the failure is
counted and raised.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from prometheus_client import REGISTRY
from sqlalchemy import Column, Integer, MetaData, String, Table

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import l2_degrade
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.l2_degrade import degrade_l2
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.tier_reads import ReadTier, record_tier_read
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import L2WriteRefusedError
from threetears.core.testing.bridge import drain_and_shutdown_bridge
from threetears.nats import KvError, KvFullError, KvStoreExhaustedError, KvValueTooLargeError

_SCOPE = "test-principal"
_TABLE = "l2_degraded_rows"
_METRIC = "threetears_l2_degraded_total"


class _Row(BaseEntity):
    primary_key_field = "id"


class _Rows(BaseCollection[_Row]):
    """a three-tier collection whose L3 is a dict."""

    def __init__(self, registry: CollectionRegistry, nats_client: AsyncMock, l3_rows: dict[str, dict]) -> None:
        self._l3_rows = l3_rows
        super().__init__(
            registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""), nats_client
        )

    @property
    def table_name(self) -> str:
        return _TABLE

    @property
    def entity_class(self) -> type[_Row]:
        return _Row

    async def fetch_from_store(self, entity_id: object) -> dict | None:
        return self._l3_rows.get(str(entity_id))

    async def save_to_store(self, data: dict, original_timestamp: datetime | None = None) -> int:
        self._l3_rows[str(data["id"])] = dict(data)
        return 1

    async def delete_from_store(self, entity_id: object) -> None:
        self._l3_rows.pop(str(entity_id), None)

    def serialize(self, data: dict) -> bytes:
        return json.dumps(data, default=str).encode()

    def deserialize(self, data: bytes) -> dict:
        return json.loads(data)


class _StoreRows(_Rows):
    """the same collection with no durable tier behind L2."""

    l2_is_the_store: bool = True


def _bucket(failure: BaseException) -> AsyncMock:
    """a NATS client whose bucket fails every read and write with ``failure``."""
    bucket = AsyncMock()
    bucket.get = AsyncMock(side_effect=failure)
    bucket.get_latest = AsyncMock(side_effect=failure)
    bucket.put = AsyncMock(side_effect=failure)
    bucket.update = AsyncMock(side_effect=failure)
    bucket.delete = AsyncMock(side_effect=failure)
    nats = AsyncMock()
    nats.kv_bucket = AsyncMock(return_value=bucket)
    nats.publish = AsyncMock()
    nats.subscribe_typed = AsyncMock()
    return nats


@pytest.fixture()
def registry() -> CollectionRegistry:
    metadata = MetaData()
    Table(_TABLE, metadata, Column("id", String, primary_key=True), Column("score", Integer))
    backend = SQLiteBackend(db_name=f"test_l2_degraded_{uuid.uuid4().hex[:8]}")
    backend.initialize(metadata)
    reg = CollectionRegistry()
    reg.configure(l1_backend=backend, kv_key_scope=_SCOPE)
    yield reg
    drain_and_shutdown_bridge()
    backend.reset()


def _degraded(op: str, reason: str, table: str = _TABLE) -> float:
    return REGISTRY.get_sample_value(_METRIC, {"table": table, "op": op, "reason": reason}) or 0.0


def _full() -> KvFullError:
    return KvFullError("KV put failed: maximum bytes exceeded", bucket="KV_collections")


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class TestTheReasonIsTheFailuresKind:
    @pytest.mark.parametrize(
        ("error", "reason"),
        [
            (KvFullError("full", bucket="b"), "full"),
            (KvStoreExhaustedError("out of store", bucket="b"), "exhausted"),
            (KvValueTooLargeError("too big", bucket="b"), "too_large"),
            (KvError("connection closed"), "transport"),
        ],
    )
    def test_each_kind_is_counted_under_its_reason(self, error: KvError, reason: str) -> None:
        table = _unique("reasons")
        degrade_l2(error, table=table, op="put", what="L2 cache write failed")
        assert _degraded("put", reason, table=table) == 1

    def test_a_refusal_the_server_answered_for_another_cause_is_other(self) -> None:
        table = _unique("reasons")
        error = KvError("KV put failed: stream is offline")
        error.server_error_code = 10118
        degrade_l2(error, table=table, op="put", what="L2 cache write failed")
        assert _degraded("put", "other", table=table) == 1
        assert _degraded("put", "transport", table=table) == 0


class TestAFullBucketBehindADurableTier:
    async def test_the_read_answers_from_l3_and_counts_full(self, registry: CollectionRegistry) -> None:
        coll = _Rows(registry, _bucket(_full()), l3_rows={"e1": {"id": "e1", "score": 1}})
        before = _degraded("get", "full")

        entity = await coll.get("e1")

        assert entity is not None and entity.score == 1
        assert _degraded("get", "full") >= before + 1

    async def test_the_save_lands_in_l3_and_counts_full(self, registry: CollectionRegistry) -> None:
        l3_rows: dict[str, dict] = {}
        coll = _Rows(registry, _bucket(_full()), l3_rows=l3_rows)
        before = _degraded("put", "full")

        await coll.save_entity(coll.create({"id": "e1", "score": 1}))

        assert "e1" in l3_rows
        assert _degraded("put", "full") == before + 1


class TestAFullBucketThatIsTheStore:
    async def test_the_read_raises_and_counts_full(self, registry: CollectionRegistry) -> None:
        coll = _StoreRows(registry, _bucket(_full()), l3_rows={})
        before = _degraded("get", "full")

        with pytest.raises(KvFullError):
            await coll.get("e1")

        assert _degraded("get", "full") == before + 1

    async def test_the_save_raises_and_counts_full(self, registry: CollectionRegistry) -> None:
        coll = _StoreRows(registry, _bucket(_full()), l3_rows={})
        before = _degraded("put", "full")

        with pytest.raises(L2WriteRefusedError) as raised:
            await coll.save_entity(coll.create({"id": "e1", "score": 1}))

        assert isinstance(raised.value.__cause__, KvFullError)
        assert _degraded("put", "full") == before + 1


class TestATransportFailure:
    async def test_the_read_answers_from_l3_and_counts_transport(self, registry: CollectionRegistry) -> None:
        coll = _Rows(registry, _bucket(KvError("connection closed")), l3_rows={"e1": {"id": "e1", "score": 2}})
        before = _degraded("get", "transport")

        entity = await coll.get("e1")

        assert entity is not None and entity.score == 2
        assert _degraded("get", "transport") >= before + 1


class _Clock:
    """a monotonic clock a test moves by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


class TestTheLogLineIsRateLimited:
    def test_a_sustained_failure_logs_once_per_window_and_counts_every_time(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        table = _unique("rate_limited")
        with caplog.at_level(logging.WARNING, logger=l2_degrade.__name__):
            for _ in range(5):
                degrade_l2(_full(), table=table, op="put", what="L2 cache write failed")

        lines = [record for record in caplog.records if record.name == l2_degrade.__name__]
        assert len(lines) == 1
        assert _degraded("put", "full", table=table) == 5

    def test_the_next_window_logs_again_saying_how_many_it_held_back(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _Clock()
        monkeypatch.setattr(l2_degrade, "time", clock)
        table = _unique("rate_limited")
        with caplog.at_level(logging.WARNING, logger=l2_degrade.__name__):
            for _ in range(3):
                degrade_l2(_full(), table=table, op="put", what="L2 cache write failed")
            clock.now += 59.0
            degrade_l2(_full(), table=table, op="put", what="L2 cache write failed")
            clock.now += 2.0
            degrade_l2(_full(), table=table, op="put", what="L2 cache write failed")

        lines = [record for record in caplog.records if record.name == l2_degrade.__name__]
        assert len(lines) == 2
        data = lines[-1].extra_data  # type: ignore[attr-defined]
        assert (data["table"], data["op"], data["reason"], data["suppressed"]) == (table, "put", "full", 3)

    def test_each_reason_has_its_own_window(self, caplog: pytest.LogCaptureFixture) -> None:
        """a full bucket must not be hidden behind a transport line logged a moment before it."""
        table = _unique("rate_limited")
        with caplog.at_level(logging.WARNING, logger=l2_degrade.__name__):
            degrade_l2(KvError("connection closed"), table=table, op="put", what="L2 cache write failed")
            degrade_l2(_full(), table=table, op="put", what="L2 cache write failed")

        reasons = [
            record.extra_data["reason"]  # type: ignore[attr-defined]
            for record in caplog.records
            if record.name == l2_degrade.__name__
        ]
        assert reasons == ["transport", "full"]

    def test_where_l2_is_the_store_it_raises_and_counts(self) -> None:
        table = _unique("store")
        with pytest.raises(KvFullError):
            degrade_l2(_full(), table=table, op="get", what="L2 read failed", l2_is_the_store=True)
        assert _degraded("get", "full", table=table) == 1


class TestTierReadsAreAPrometheusCounter:
    def test_each_read_counts_on_its_table_and_tier(self) -> None:
        table = _unique("tier_reads")
        record_tier_read(table, ReadTier.L1)
        record_tier_read(table, ReadTier.L1)
        record_tier_read(table, ReadTier.L3)

        assert REGISTRY.get_sample_value("threetears_tier_reads_total", {"table": table, "tier": "l1"}) == 2
        assert REGISTRY.get_sample_value("threetears_tier_reads_total", {"table": table, "tier": "l3"}) == 1
