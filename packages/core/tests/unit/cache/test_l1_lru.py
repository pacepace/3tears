"""The SQLite L1's bound (threetears.core.cache.l1_lru): a byte budget per process, least-recently-used eviction of
EVICTABLE tables only, never of KEEP ones; an eviction is a miss that reads back from the tier below, never a wrong
answer."""

from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest
from prometheus_client import REGISTRY
from sqlalchemy import Column, MetaData, String, Table

from threetears.core.cache.l1_lru import L1Ledger, declare_l1_table, row_bytes
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.eviction import Declared, Policy, Tier, policies_declared
from threetears.core.testing.l1_tables import L1_EVICTION_INFO
from threetears.core.testing.loop_lateness import quietly

_CACHED = Declared(Policy.EVICTABLE, "lru test: rows cached on a read")
_KEPT = Declared(Policy.KEEP, "lru test: a table a reader treats as complete")
_BODY = "b" * 1000
#: the most the host's idle loop may be late over a timed comparison, as the other side-by-side cost tests take it
_HOST_MAY_BE_LATE = 0.020


def _metadata() -> MetaData:
    metadata = MetaData()
    Table(
        "lru_cached",
        metadata,
        Column("id", String, primary_key=True),
        Column("body", String),
        info={L1_EVICTION_INFO: _CACHED},
    )
    Table(
        "lru_kept",
        metadata,
        Column("id", String, primary_key=True),
        Column("body", String),
        info={L1_EVICTION_INFO: _KEPT},
    )
    return metadata


def _backend(tmp_path: Path, budget: int) -> tuple[SQLiteBackend, L1Ledger]:
    ledger = L1Ledger(budget, thread=False)
    backend = SQLiteBackend(db_name=f"lru_{uuid.uuid4().hex[:8]}", directory=tmp_path, budget=ledger)
    backend.initialize(_metadata())
    return backend, ledger


def _fill(backend: SQLiteBackend, table: str, keys: list[str]) -> None:
    for key in keys:
        backend.upsert(table, {"id": key, "body": _BODY})


def _held(backend: SQLiteBackend, table: str, keys: list[str]) -> list[str]:
    return [key for key in keys if backend.execute_query(f"SELECT 1 FROM {table} WHERE id = ?", (key,))]


class TestDeclarations:
    def test_a_table_without_a_policy_is_refused(self, tmp_path: Path) -> None:
        metadata = MetaData()
        Table(f"lru_undeclared_{uuid.uuid4().hex[:6]}", metadata, Column("id", String, primary_key=True))
        backend = SQLiteBackend(db_name="lru_refuse", directory=tmp_path, budget=L1Ledger(1 << 20, thread=False))
        with pytest.raises(LookupError, match="declares no eviction policy"):
            backend.initialize(metadata)
        assert backend.file_path is None, "nothing is created for a refused table"

    def test_every_declared_table_is_in_the_family_declarations(self, tmp_path: Path) -> None:
        backend, _ = _backend(tmp_path, 1 << 20)
        assert policies_declared(Tier.L1)["lru_cached"].policy is Policy.EVICTABLE
        assert policies_declared(Tier.L1)["lru_kept"].policy is Policy.KEEP
        backend.reset()

    def test_two_declarations_that_disagree_keep_the_table(self) -> None:
        name = f"lru_disagree_{uuid.uuid4().hex[:6]}"
        declare_l1_table(name, _CACHED)
        declare_l1_table(name, _KEPT)
        declare_l1_table(name, _CACHED)
        assert policies_declared(Tier.L1)[name].policy is Policy.KEEP


class TestTheBudget:
    def test_over_budget_evicts_the_least_recently_used_cached_rows(self, tmp_path: Path) -> None:
        backend, ledger = _backend(tmp_path, 60 * 1100)
        keys = [f"k{i:03d}" for i in range(50)]
        _fill(backend, "lru_cached", keys)
        # the first ten are read again: they are now the most recent
        for key in keys[:10]:
            assert backend.select_by_id("lru_cached", key) is not None
        _fill(backend, "lru_cached", [f"n{i:03d}" for i in range(30)])
        ledger.drain()
        held = _held(backend, "lru_cached", keys)
        assert set(keys[:10]) <= set(held), "a recently read row was evicted before older ones"
        assert keys[10] not in held, "the least recently used row was kept"
        assert len(held) < len(keys)
        backend.reset()

    def test_kept_rows_are_never_evicted(self, tmp_path: Path) -> None:
        backend, ledger = _backend(tmp_path, 60 * 1100)
        kept = [f"keep{i:03d}" for i in range(40)]
        _fill(backend, "lru_kept", kept)
        ledger.measure()
        _fill(backend, "lru_cached", [f"c{i:03d}" for i in range(100)])
        ledger.drain()
        ledger.measure()
        assert _held(backend, "lru_kept", kept) == kept
        backend.reset()

    def test_kept_rows_alone_past_the_budget_are_a_sizing_failure_never_an_eviction(self, tmp_path: Path) -> None:
        backend, ledger = _backend(tmp_path, 10 * 1100)
        kept = [f"keep{i:03d}" for i in range(40)]
        _fill(backend, "lru_kept", kept)
        labels = {"tier": "l1", "name": ""}
        before = REGISTRY.get_sample_value("threetears_cache_sizing_failures_total", labels) or 0.0
        ledger.measure()
        assert REGISTRY.get_sample_value("threetears_cache_sizing_failures_total", labels) == before + 1
        assert _held(backend, "lru_kept", kept) == kept
        backend.reset()

    def test_the_accounting_stays_within_a_third_of_what_dbstat_measures(self, tmp_path: Path) -> None:
        """the ledger decides by its own per-row estimate (no file-system read on a write); SQ2's sampler reports
        dbstat's pages. Drift between the two shows here, before it shows as a budget that holds the wrong bytes."""
        backend, ledger = _backend(tmp_path, 1 << 30)
        rows = [{"id": f"r{i:06d}", "body": "x" * (50 + (i * 37) % 2000)} for i in range(5000)]
        backend.upsert_many("lru_cached", rows)
        estimated = sum(row_bytes((row["id"], row["body"])) for row in rows)
        measured = backend.table_bytes()["lru_cached"]
        assert 0.67 < estimated / measured < 1.5, f"estimated {estimated} bytes, dbstat measured {measured}"
        assert ledger.status()["evictable_bytes"] == estimated
        backend.reset()

    @pytest.mark.loop_timing
    def test_a_read_pays_little_for_its_recency(self, tmp_path: Path) -> None:
        """a hit moves its key in memory, under a lock it never waits for: no write, no file-system call."""
        backend, _ = _backend(tmp_path, 1 << 30)
        keys = [f"r{i:05d}" for i in range(2000)]
        backend.upsert_many("lru_cached", [{"id": key, "body": "x"} for key in keys])
        backend.upsert_many("lru_kept", [{"id": key, "body": "x"} for key in keys])

        def per_read(table: str) -> float:
            started = time.perf_counter()
            for key in keys:
                backend.select_by_id(table, key)
            return (time.perf_counter() - started) / len(keys)

        def side_by_side() -> tuple[float, float]:
            # the two tables' rounds alternate and each side keeps its fastest: run one after the other, a burst of
            # crowding on a shared host landed on one side only (the pool, 2026-11-15: 43 us against 29 us, where a
            # quiet laptop measures about 1 us of recency on a 9 us read)
            kept_rounds, cached_rounds = [], []
            for _ in range(5):
                kept_rounds.append(per_read("lru_kept"))
                cached_rounds.append(per_read("lru_cached"))
            return min(kept_rounds), min(cached_rounds)

        # a host too busy to keep an idle loop on time says nothing about the code: retaken, else CouldNotMeasure
        kept, cached = quietly(side_by_side, host_may_be_late=_HOST_MAY_BE_LATE, threads=1)
        backend.reset()
        # measured 2026-11-15 on a laptop: about 1 us on a 9 us read
        assert cached < kept * 1.5, (
            f"a cached-table read took {cached * 1e6:.1f} us, a kept-table read {kept * 1e6:.1f} us"
        )


class _Entity(BaseEntity):
    primary_key_field = "id"


class _Rows(BaseCollection[_Entity]):
    """a read-through collection over an L3 dict, with no L2."""

    datetime_columns: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, registry: CollectionRegistry, rows: dict[str, dict[str, Any]]) -> None:
        self._rows = rows
        self.fetches = 0
        super().__init__(registry, DefaultCoreConfig(), nats_client=None)

    @property
    def table_name(self) -> str:
        return "lru_cached"

    @property
    def entity_class(self) -> type[_Entity]:
        return _Entity

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        self.fetches += 1
        row = self._rows.get(str(entity_id))
        return dict(row) if row is not None else None

    async def save_to_store(
        self, data: dict[str, Any], original_timestamp: datetime | None = None, *, conn: Any = None
    ) -> int:
        self._rows[str(data["id"])] = dict(data)
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        self._rows.pop(str(entity_id), None)

    def serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data).encode()

    def deserialize(self, data: bytes) -> dict[str, Any]:
        row: dict[str, Any] = json.loads(data)
        return row


class TestAnEvictionIsAMiss:
    async def test_an_evicted_row_reads_back_from_l3_and_refills(self, tmp_path: Path) -> None:
        backend, ledger = _backend(tmp_path, 20 * 1100)
        l3 = {f"k{i:03d}": {"id": f"k{i:03d}", "body": _BODY} for i in range(60)}
        registry = CollectionRegistry()
        registry.configure(l1_backend=backend, l3_pool=object(), kv_key_scope="lru-test")  # type: ignore[arg-type]
        rows = _Rows(registry, l3)
        for key in l3:
            assert (await rows.ensure(key)) == l3[key]
        ledger.drain()
        assert "k000" not in _held(backend, "lru_cached", ["k000"]), "the oldest row was not evicted"
        fetches = rows.fetches
        assert (await rows.ensure("k000")) == l3["k000"], "an evicted row read back wrong"
        assert rows.fetches == fetches + 1, "an evicted row was answered without reading L3"
        assert _held(backend, "lru_cached", ["k000"]) == ["k000"], "the read did not refill L1"
        backend.reset()

    def test_a_read_during_an_eviction_answers_the_row_or_a_miss_never_another(self, tmp_path: Path) -> None:
        backend, ledger = _backend(tmp_path, 1 << 30)
        keys = [f"k{i:05d}" for i in range(20_000)]
        backend.upsert_many("lru_cached", [{"id": key, "body": f"body-of-{key}"} for key in keys])
        ledger.set_budget(1)
        stop = threading.Event()
        wrong: list[tuple[str, Any]] = []

        def read() -> None:
            index = 0
            while not stop.is_set():
                key = keys[(index * 7919) % len(keys)]
                row = backend.select_by_id("lru_cached", key)
                if row is not None and row != {"id": key, "body": f"body-of-{key}"}:
                    wrong.append((key, row))
                index += 1

        reader = threading.Thread(target=read)
        reader.start()
        backend.upsert("lru_cached", {"id": "trigger", "body": "x"})
        ledger.drain()
        stop.set()
        reader.join(30)
        assert wrong == []
        assert _held(backend, "lru_cached", keys[:100]) == []
        backend.reset()
