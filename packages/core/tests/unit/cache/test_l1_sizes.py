"""The L1 size gauges (:mod:`threetears.core.cache.l1_sizes`): each table's rows and bytes, measured from SQLite's
pages, set on the one cache family's gauges for the tables whose eviction policy is declared.

Each test's tables have names of their own: the process's sampler (``L1_SIZES``) measures every L1 the process
watches, and a table's sizes are summed over the L1s holding one of its name."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterator, Mapping
from types import MappingProxyType

import pytest
from prometheus_client import REGISTRY
from sqlalchemy import Column, Integer, MetaData, String, Table

from threetears.core.cache import l1_sizes
from threetears.core.cache.l1_sizes import L1_SIZES, measure
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.eviction import Declared, Policy, Tier
from threetears.core.testing.l1_tables import L1_EVICTION_INFO, TEST_L1_TABLE


def _metadata(*tables: str) -> MetaData:
    md = MetaData()
    for table in tables:
        # every L1 table declares its policy to be created at all; the sampler's view of the declarations is the
        # test's own (_declare), so a table can still be undeclared there
        Table(
            table,
            md,
            Column("id", String, primary_key=True),
            Column("pod", String),
            Column("n", Integer),
            info={L1_EVICTION_INFO: TEST_L1_TABLE},
        )
    return md


def _l1(*tables: str) -> SQLiteBackend:
    backend = SQLiteBackend(db_name=f"test_l1_sizes_{uuid.uuid4().hex[:8]}")
    backend.initialize(_metadata(*tables))
    return backend


@pytest.fixture()
def l1() -> Iterator[SQLiteBackend]:
    """an L1 holding ``sz_held`` (300 rows, indexed by pod), ``sz_few`` (7 rows) and ``sz_empty``, watched by the
    process's sampler through a collection registry, as a process's L1 is."""
    backend = _l1("sz_held", "sz_few", "sz_empty")
    backend.execute_query("CREATE INDEX sz_held_by_pod ON sz_held (pod)")
    backend.upsert_many("sz_held", [{"id": f"r{i}", "pod": f"p{i % 5}" * 40, "n": i} for i in range(300)], "id")
    backend.upsert_many("sz_few", [{"id": f"f{i}", "pod": "p", "n": i} for i in range(7)], "id")
    CollectionRegistry().configure(l1_backend=backend)
    yield backend
    backend.reset()


def _value(metric: str, table: str, policy: str) -> float | None:
    return REGISTRY.get_sample_value(metric, {"tier": "l1", "name": table, "policy": policy})


def _declare(monkeypatch: pytest.MonkeyPatch, policies: Mapping[str, Declared]) -> None:
    # the sampler's view of the L1's declarations, kept off the process-wide ones a gate lists
    monkeypatch.setattr(
        l1_sizes, "policies_declared", lambda tier: MappingProxyType(dict(policies)) if tier is Tier.L1 else {}
    )


def test_rows_are_each_tables_count_and_bytes_hold_its_indexes(l1: SQLiteBackend) -> None:
    """Flips if a table's rows are not its ``count(*)``, or its bytes leave out an index's pages."""
    sizes = measure(l1.get_connection())
    for table in ("sz_held", "sz_few", "sz_empty"):
        [counted] = l1.execute_query(f'SELECT count(*) AS n FROM "{table}"')  # noqa: S608 -- the test's own tables
        assert sizes[table].rows == counted["n"], table
    pages = {
        row["name"]: row["pgsize"]
        for row in l1.execute_query("SELECT name, sum(pgsize) AS pgsize FROM dbstat GROUP BY name")
    }
    assert sizes["sz_held"].bytes == pages["sz_held"] + pages["sz_held_by_pod"] + pages["sqlite_autoindex_sz_held_1"]
    assert sizes["sz_empty"].rows == 0
    assert sizes["sz_empty"].bytes == pages["sz_empty"] + pages["sqlite_autoindex_sz_empty_1"]
    assert not any(table.startswith("sqlite_") for table in sizes)


def test_a_declared_tables_size_is_set_on_the_family(l1: SQLiteBackend, monkeypatch: pytest.MonkeyPatch) -> None:
    """Flips if a registry's L1 is not measured, a declared table's rows and bytes are not on
    threetears_cache_held_entries / _held_bytes (tier l1, its name, its policy), or an undeclared one is."""
    _declare(
        monkeypatch,
        {"sz_held": Declared(Policy.EVICTABLE, "a cache of reads"), "sz_few": Declared(Policy.KEEP, "held whole")},
    )
    measured = L1_SIZES.sample_once()
    assert _value("threetears_cache_held_entries", "sz_held", "evictable") == 300
    assert _value("threetears_cache_held_bytes", "sz_held", "evictable") == measured["sz_held"].bytes
    assert _value("threetears_cache_held_entries", "sz_few", "keep") == 7
    assert measured["sz_empty"].rows == 0
    for policy in ("evictable", "keep"):
        assert _value("threetears_cache_held_entries", "sz_empty", policy) is None


def test_a_table_gone_since_reads_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flips if a table no L1 holds any more keeps its last size on the gauge."""
    _declare(monkeypatch, {"sz_gone_soon": Declared(Policy.EVICTABLE, "a cache of reads")})
    backend = _l1("sz_gone_soon")
    backend.upsert_many("sz_gone_soon", [{"id": str(i), "pod": "p", "n": i} for i in range(4)], "id")
    CollectionRegistry().configure(l1_backend=backend)
    L1_SIZES.sample_once()
    assert _value("threetears_cache_held_entries", "sz_gone_soon", "evictable") == 4
    backend.reset()
    assert "sz_gone_soon" not in L1_SIZES.sample_once()
    assert _value("threetears_cache_held_entries", "sz_gone_soon", "evictable") == 0
    assert _value("threetears_cache_held_bytes", "sz_gone_soon", "evictable") == 0


def test_the_sampler_runs_on_its_own_thread(l1: SQLiteBackend) -> None:
    """Flips if a watched L1 is not sampled on a daemon thread of the sampler's own, off the caller's thread."""
    [sampler] = [thread for thread in threading.enumerate() if thread.name == "l1-sizes"]
    assert sampler.daemon and sampler.is_alive()
    assert sampler is not threading.current_thread()
