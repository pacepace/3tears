"""A table-wide drop of the L1 runs off the event loop (SQ1 audit item 4a; Pace said yes 2026-11-15).

It walked and deleted the whole table on the loop: 276 ms at 100k rows (scratchpad/sq1/bench1.log). Now the loop only
announces it, under a "dropping" count that makes every L1 read of the table a miss, and a thread deletes."""

from __future__ import annotations

import ast
import asyncio
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
from prometheus_client import REGISTRY
from sqlalchemy import MetaData

from threetears.core.cache.l1_lru import L1Ledger
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.base import PrefixNotLeadingError
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import STRING_TYPE, Column, TableSchema, collection_for_schema
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.calibrate import budget_problem
from threetears.core.testing.l1_tables import TEST_L1_TABLE
from threetears.core.testing.loop_lateness import turn_holds

#: the longest one turn of the loop may run while 100k rows are dropped: the announcement, and with a listener each
#: page of 1,000 keys it hears. On the loop the drop held it 276 ms (bench1.log); measured here about 2 ms.
_TURN_MAX_SECONDS = 0.05


def _cached_collection() -> tuple[Any, SQLiteBackend]:
    schema = TableSchema(
        name="drop_rows",
        primary_key="id",
        columns=[Column("id", STRING_TYPE), Column("payload", STRING_TYPE)],
        l1_eviction=TEST_L1_TABLE,
    )
    metadata = MetaData()
    schema.to_sqlalchemy_table(metadata)
    l1 = SQLiteBackend(db_name=f"drop_off_loop_{uuid.uuid4().hex[:8]}", budget=L1Ledger(1 << 34, thread=False))
    l1.initialize(metadata)
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, kv_key_scope="test-principal")
    return collection_for_schema(schema)(registry, DefaultCoreConfig(), None), l1


def _cache(l1: SQLiteBackend, count: int) -> list[str]:
    keys = [f"k{index:07d}" for index in range(count)]
    l1.upsert_many("drop_rows", [{"id": key, "payload": "x" * 200} for key in keys], "id")
    return keys


class TestNoRowIsServedOnceTheDropIsAnnounced:
    async def test_a_read_while_the_delete_runs_is_a_miss(self, monkeypatch: Any) -> None:
        collection, l1 = _cached_collection()
        keys = _cache(l1, 50)
        deleting, release = threading.Event(), threading.Event()
        delete_all = SQLiteBackend.delete_all

        def held(self: SQLiteBackend, table: str) -> int:
            deleting.set()
            assert release.wait(10)
            return delete_all(self, table)

        monkeypatch.setattr(SQLiteBackend, "delete_all", held)
        assert collection.drop_cached_table() is None
        while not deleting.is_set():
            await asyncio.sleep(0.001)
        # the rows are still in the file: the delete has not run, and none of them may be served
        assert l1.select_by_id("drop_rows", keys[0]) is not None
        assert [collection.exists_in_cache_sync(key) for key in keys] == [False] * len(keys)
        release.set()
        assert await collection.table_drops_settled() == len(keys)
        assert [collection.exists_in_cache_sync(key) for key in keys] == [False] * len(keys)
        assert l1.select_by_id("drop_rows", keys[0]) is None

    async def test_a_row_cached_after_the_drop_is_served_once_it_ends(self) -> None:
        collection, l1 = _cached_collection()
        _cache(l1, 10)
        collection.drop_cached_table()
        await collection.table_drops_settled()
        l1.upsert("drop_rows", {"id": "fresh", "payload": "new"}, "id")
        assert collection.exists_in_cache_sync("fresh")


# bounds its own time on the host (turn_holds): run apart, never beside other work (tests/timing_suite.py)
@pytest.mark.loop_timing
class TestTheDropHoldsNoTurnOfTheLoop:
    async def test_dropping_100k_rows_holds_no_turn_long(self) -> None:
        collection, l1 = _cached_collection()
        await asyncio.to_thread(_cache, l1, 100_000)
        with turn_holds() as holds:
            assert collection.drop_cached_table() is None
            assert await collection.table_drops_settled() == 100_000
        longest = max(seconds for seconds, _ in holds)
        assert longest < _TURN_MAX_SECONDS, f"a turn held the loop {longest * 1000:.1f} ms during the drop"
        assert l1.execute_query("SELECT count(*) AS n FROM drop_rows")[0]["n"] == 0

    async def test_with_a_listener_every_key_is_heard_and_no_turn_is_long(self) -> None:
        collection, l1 = _cached_collection()
        keys = await asyncio.to_thread(_cache, l1, 100_000)
        heard: list[Any] = []
        collection.add_l1_change_listener(heard.append)
        with turn_holds() as holds:
            collection.drop_cached_table()
            await collection.table_drops_settled()
        longest = max(seconds for seconds, _ in holds)
        assert sorted(heard) == keys
        assert longest < _TURN_MAX_SECONDS, f"a turn held the loop {longest * 1000:.1f} ms during the drop"


#: every package's sources, read for where the L1 table drop is reached from
_PACKAGES = Path(__file__).resolve().parents[4]


def _functions_naming(name: str) -> set[str]:
    """each source function (``<module>:<qualified name>``) naming ``name`` as an attribute or a string, its own
    definition aside."""
    found: set[str] = set()
    for path in _PACKAGES.glob("**/src/**/*.py"):
        text = path.read_text(encoding="utf-8")
        if name not in text:
            continue
        tree = ast.parse(text)

        def visit(node: ast.AST, scope: str, module: str = path.name) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    visit(child, f"{scope}.{child.name}" if scope else child.name)
                    continue
                named = (isinstance(child, ast.Attribute) and child.attr == name) or (
                    isinstance(child, ast.Constant) and child.value == name
                )
                if named:
                    found.add(f"{module}:{scope}")
                visit(child, scope)

        visit(tree, "")
    return found


def test_delete_all_is_reached_only_from_a_table_drop() -> None:
    """Flips if anything but a table drop reaches SQLiteBackend.delete_all, the L1 table's one whole DELETE (declared
    l1_table_drop, Pace 2026-11-15): only BaseCollection._drop_l1_table_on_a_thread names it, and only
    BaseCollection.drop_cached_table starts that."""
    assert callable(SQLiteBackend.delete_all)
    assert _functions_naming("delete_all") == {"base.py:BaseCollection._drop_l1_table_on_a_thread"}
    assert _functions_naming("_drop_l1_table_on_a_thread") == {"base.py:BaseCollection.drop_cached_table"}


def _written(l1: SQLiteBackend, sql: str, rows: list[tuple[str, str]] | None = None) -> float:
    """one statement in its own immediate transaction on the backend's connection, as the backend writes; its
    seconds."""
    conn = l1.get_connection()
    started = time.perf_counter()
    conn.execute("BEGIN IMMEDIATE")
    if rows is None:
        conn.execute(sql)
    else:
        conn.executemany(sql, rows)
    conn.execute("COMMIT")
    return time.perf_counter() - started


def test_delete_all_of_100k_rows_is_the_bare_delete() -> None:
    """Flips if SQLiteBackend.delete_all (the l1_table_drop, Pace 2026-11-15) takes over twice a bare DELETE of the
    same 100k rows in the same database, plus the ledger forgetting the table's rows: the declared whole delete must be
    SQLite's truncate and the ledger's one pop, nothing on top (no keyed walk). Measured at 14-17 ms; the keyed walk it
    replaces took 645-916 ms (scratchpad/sq4/HANDOFF-SQ4.md).

    Why the ledger's forget is budgeted on its own (``plus=``): delete_all ends with ``budget.emptied``, which pops the
    table's recency, a 100k-entry dict freed on the drop's thread, a cost of the rows' count in Python and none of
    SQLite's. The bare DELETE of an untracked table never pays it, so the calibration once compared the drop with a
    baseline lacking it; it held while the DELETE itself was slow (14-17 ms), and failed once it was not (2026-11-23,
    macOS arm64, Python 3.14: bare DELETE 1.5-2.7 ms, the ledger's forget alone 3.6-3.8 ms, delete_all 5.1-6.5 ms).
    The forget is measured here on its own, as the path pays it, and the DELETE keeps its 2x."""
    _collection, l1 = _cached_collection()
    _written(l1, 'CREATE TABLE "bare_rows" (id TEXT PRIMARY KEY, payload TEXT)')
    rows = [(f"k{index:07d}", "x" * 200) for index in range(100_000)]
    took, baseline, forget = [], [], []
    for _ in range(3):
        # the ledger forgetting 100k rows' recency, alone: the rows stay, and are noted again for the drop
        _cache(l1, 100_000)
        started = time.perf_counter()
        l1.budget.emptied(l1, "drop_rows")
        forget.append(time.perf_counter() - started)
        _cache(l1, 100_000)
        _written(l1, 'INSERT INTO "bare_rows" VALUES (?, ?)', rows)
        baseline.append(_written(l1, 'DELETE FROM "bare_rows"'))
        started = time.perf_counter()
        assert l1.delete_all("drop_rows") == 100_000
        took.append(time.perf_counter() - started)
    assert budget_problem("SQLiteBackend.delete_all of 100k rows", min(took), min(baseline), plus=min(forget)) is None


# --- a key prefix's drop (a load's or a clear's reach) runs off the loop too (0.95.0 copy audit, loop item) ---------

#: the longest one turn of the loop may run while a prefix holding 100k of L1's 200k rows is dropped. Measured
#: 2026-11-23 (macOS arm64, Python 3.14): the keys read by the prefix and evicted on the loop, 334.7 ms; on threads,
#: 2.8 ms. A prefix not leading the key is refused, never walked (decision 34)
_PREFIX_TURN_MAX_SECONDS = 0.05


class _PrefixRow(BaseEntity):
    """a row whose own id is ``id``, within the state its key leads with."""

    primary_key_field: str = "id"


def _prefixed_collection() -> tuple[Any, SQLiteBackend]:
    schema = TableSchema(
        name="prefix_rows",
        primary_key=("state", "id"),
        columns=[Column("state", STRING_TYPE), Column("id", STRING_TYPE), Column("payload", STRING_TYPE)],
        l1_eviction=TEST_L1_TABLE,
    )
    metadata = MetaData()
    schema.to_sqlalchemy_table(metadata)
    l1 = SQLiteBackend(db_name=f"prefix_off_loop_{uuid.uuid4().hex[:8]}", budget=L1Ledger(1 << 34, thread=False))
    l1.initialize(metadata)
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, kv_key_scope="test-principal")
    return collection_for_schema(schema, entity_class=_PrefixRow)(registry, DefaultCoreConfig(), None), l1


def _cache_states(l1: SQLiteBackend, per_state: int) -> None:
    rows = [
        {"state": state, "id": f"k{index:07d}", "payload": "x" * 200}
        for state in ("VA", "IL")
        for index in range(per_state)
    ]
    l1.upsert_many("prefix_rows", rows, ("state", "id"))


class TestAPrefixDropHoldsNoTurnOfTheLoop:
    # bounds its own time on the host (turn_holds): run apart, never beside other work (tests/timing_suite.py)
    @pytest.mark.loop_timing
    async def test_dropping_a_100k_row_prefix_holds_no_turn_long_and_leaves_the_other_state(self) -> None:
        """Flips if a prefix drop reads L1's keys under it, or evicts them, on the event loop (the loop answered
        nothing else for that long), or drops a row outside the prefix."""
        collection, l1 = _prefixed_collection()
        await asyncio.to_thread(_cache_states, l1, 100_000)
        heard: list[Any] = []
        collection.add_l1_change_listener(heard.append)
        with turn_holds() as holds:
            assert await collection.drop_cached_prefixes([{"state": "VA"}]) == 100_000
        longest = max(seconds for seconds, _ in holds)
        assert longest < _PREFIX_TURN_MAX_SECONDS, f"a turn held the loop {longest * 1000:.1f} ms during the drop"
        assert len(heard) == 100_000
        counted = l1.execute_query("SELECT state, count(*) AS n FROM prefix_rows GROUP BY state")
        assert {row["state"]: row["n"] for row in counted} == {"IL": 100_000}

    async def test_a_prefix_not_leading_the_key_is_refused_and_walks_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Flips if a slice clear whose prefix does not lead the key (an older writer's partition, ``{"id": ...}`` of a
        ``(state, id)`` key) is walked through L1's keys or dropped some other way instead of refused (decision 34):
        it must raise PrefixNotLeadingError naming the table and prefix, count the refusal, read no L1 key, fence and
        distrust nothing, and leave every cached row."""
        collection, l1 = _prefixed_collection()
        _cache_states(l1, 1_000)
        heard: list[Any] = []
        collection.add_l1_change_listener(heard.append)

        def walked(*_: Any, **__: Any) -> Any:
            raise AssertionError("L1 was read for a refused prefix")

        labels = {"table": "prefix_rows"}
        before = REGISTRY.get_sample_value("threetears_collection_prefix_drops_refused_total", labels) or 0.0
        with monkeypatch.context() as patched, pytest.raises(PrefixNotLeadingError) as refused:
            # L1's front door: every read of its keys asks the backend for its connection
            patched.setattr(l1, "get_connection", walked)
            await collection.drop_cached_prefixes([{"state": "VA"}, {"id": "k0000001"}])
        assert refused.value.table == "prefix_rows"
        assert refused.value.prefix == {"id": "k0000001"}
        assert REGISTRY.get_sample_value("threetears_collection_prefix_drops_refused_total", labels) == before + 1
        assert heard == []
        assert collection.distrusted_prefixes() == {}, "a refused clear distrusts nothing"
        counted = l1.execute_query("SELECT state, count(*) AS n FROM prefix_rows GROUP BY state")
        assert {row["state"]: row["n"] for row in counted} == {"IL": 1_000, "VA": 1_000}
