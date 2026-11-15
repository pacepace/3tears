"""A local write pays no more for the scan cache than it did before the cache read by a dependency key.

The contract. Every local write (and every broadcast heard) runs :meth:`ScanCache.drop_for_table` for its table. At a
real write rate -- a row upserted into a collection's L1 table and its eviction, back to back, while the cache holds
scans of other tables -- a write takes no longer than the same write did with the eviction as it was before
a957c69ba (one select of every entry and a JSON filter, rebuilt here as :func:`_before`) at its cheapest, over an
empty cache, and the eviction alone takes at most half of that one empty select.

Why: a957c69ba made the eviction a keyed read of the dependency rows, about 8.4 us a write on SQLite against about
2 us (R3, 2026-11-15; 69 us against 12 us on a 10-core host at load ~70, builder S2), every write paying it though
almost none is for a table a scan depends on. An eviction of a table no stored scan depends on now touches no L1
(``scan_cache._Dependents``; 0.4 us there). Counted, the same contract is in ``test_scan_cache.py``
(``TestAnEvictionOfATableNoScanDependsOnTouchesNoL1``).
"""

from __future__ import annotations

import json
import statistics
import time
from uuid import uuid4

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, Text

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.scan_cache import ScanCache, ScanCacheKey
from threetears.core.testing.loop_lateness import quietly

#: the writes one round times, and the rounds (after one to warm)
WRITES = 1_000
ROUNDS = 5

#: scans the cache holds, of tables the writes never touch (a pod's cached visibility scans)
HELD_SCANS = 50

#: the most a write with its eviction may take against the same write with the eviction before a957c69ba
WRITE_MULTIPLE = 1.0

#: the most an eviction of a table no scan depends on may take against the select every eviction ran before
EVICTION_MULTIPLE = 0.5

#: the most the host's own idle loop may be late over the rounds for them to judge the cache
HOST_MAY_BE_LATE = 0.020

_ROWS_TABLE = "written_rows"


def _l1() -> SQLiteBackend:
    """an L1 holding a collection-like table the writes go to.

    :return: the backend
    :rtype: SQLiteBackend
    """
    l1 = SQLiteBackend(db_name=f"write-cost-{uuid4().hex}")
    metadata = MetaData()
    Table(
        _ROWS_TABLE,
        metadata,
        Column("id", String, primary_key=True),
        Column("agent_id", String, nullable=False),
        Column("name", Text, nullable=False),
        Column("body", Text, nullable=False),
        Column("version", Integer, nullable=False),
    )
    l1.initialize(metadata)
    return l1


def _cache(l1: SQLiteBackend) -> ScanCache:
    """a scan cache over ``l1`` holding scans of other tables, its write table evicted once (as a pod's first write).

    :param l1: the backend
    :ptype l1: SQLiteBackend
    :return: the cache
    :rtype: ScanCache
    """
    cache = ScanCache(l1, trusted=lambda _tables: True)
    for i in range(HELD_SCANS):
        key = ScanCacheKey("concepts", f"user-{i}")
        cache.put(key, [{"id": str(i)}], token=cache.begin_read(("concepts", "role_assignments")))
    cache.drop_for_table(_ROWS_TABLE)
    return cache


def _before(l1: SQLiteBackend, table: str) -> int:
    """the eviction as it was before a957c69ba: every entry selected, its JSON list filtered, the matches deleted.

    :param l1: the backend
    :ptype l1: SQLiteBackend
    :param table: the table written
    :ptype table: str
    :return: how many entries were evicted
    :rtype: int
    """
    rows = l1.execute_query("SELECT key, depends_on FROM collection_scan_cache")
    keys = [row["key"] for row in rows if table in json.loads(row["depends_on"])]
    l1.delete_many_by_id("collection_scan_cache", keys, "key")
    return len(keys)


def _row(i: int) -> dict[str, object]:
    """one written row.

    :param i: its number
    :ptype i: int
    :return: the row
    :rtype: dict[str, object]
    """
    return {"id": f"row-{i % 500}", "agent_id": "agent-1", "name": f"name {i}", "body": "x" * 200, "version": i}


def _rounds() -> dict[str, list[float]]:
    """each round's writes with today's eviction, with the eviction before, and each eviction alone, in turn.

    :return: the seconds a write (or an eviction) took, by ``now``, ``before``, ``evict_now`` and ``evict_before``
    :rtype: dict[str, list[float]]
    """
    now_l1, before_l1 = _l1(), _l1()
    # the eviction before is judged at its cheapest: a cache holding nothing, one empty select a write
    cache, _ = _cache(now_l1), ScanCache(before_l1, trusted=lambda _tables: True)
    times: dict[str, list[float]] = {"now": [], "before": [], "evict_now": [], "evict_before": []}
    clock = time.perf_counter
    for round_ in range(ROUNDS + 1):
        # the two writes alternate, write by write and in turn first, so the host's drift falls on both alike
        now = before = 0.0
        for i in range(WRITES):
            row = _row(i)
            for side in (i % 2, 1 - i % 2):
                started = clock()
                if side:
                    now_l1.upsert(_ROWS_TABLE, row, "id")
                    cache.drop_for_table(_ROWS_TABLE)
                    now += clock() - started
                else:
                    before_l1.upsert(_ROWS_TABLE, row, "id")
                    _before(before_l1, _ROWS_TABLE)
                    before += clock() - started
        now, before = now / WRITES, before / WRITES
        started = clock()
        for _ in range(WRITES):
            cache.drop_for_table(_ROWS_TABLE)
        evict_now = (clock() - started) / WRITES
        started = clock()
        for _ in range(WRITES):
            _before(before_l1, _ROWS_TABLE)
        evict_before = (clock() - started) / WRITES
        if round_:
            times["now"].append(now)
            times["before"].append(before)
            times["evict_now"].append(evict_now)
            times["evict_before"].append(evict_before)
    return times


@pytest.mark.loop_timing
class TestALocalWritePaysNoMoreForTheScanCacheThanBefore:
    """timed: run apart and serially, as the ``loop_timing`` suite runs its tests; both sides are measured in turn,
    round by round, so whatever else the machine does falls on each alike."""

    def test_a_write_and_its_eviction_cost_no_more_than_before_the_keyed_eviction(self) -> None:
        times = quietly(_rounds, host_may_be_late=HOST_MAY_BE_LATE)
        now, before, evict_now, evict_before = (
            statistics.median(times[key]) for key in ("now", "before", "evict_now", "evict_before")
        )
        print(
            f"a write with its eviction: {now * 1e6:.2f} us now, {before * 1e6:.2f} us before; "
            f"the eviction alone: {evict_now * 1e6:.2f} us now, {evict_before * 1e6:.2f} us before"
        )
        assert now <= WRITE_MULTIPLE * before, f"a write took {now / before:.2f}x what it took before"
        assert evict_now <= EVICTION_MULTIPLE * evict_before, (
            f"an eviction of a table no scan depends on took {evict_now / evict_before:.2f}x the select before"
        )
