"""A coroutine settling many keys evicts them from L1 on a worker thread, never the event loop.

A commit of thousands of rows settles them in one L1 delete, and SQLite ran it on the event loop for its whole
length: every other request on the pod waited (cobalt prod reports pods, 2026-11-06, the loop watchdog's stack in
``cache/sqlite.delete_many_by_id``). The delete now runs on a thread, the keys' fence forms are made there too,
and the fence and every listener still hear every key, in the order they did.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
import uuid
from typing import Any

import pytest
from sqlalchemy import MetaData

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import STRING_TYPE, Column, TableSchema, collection_for_schema
from threetears.core.config import DefaultCoreConfig

#: past the count from which an eviction leaves the loop (64)
_MANY = 200


def _cached_collection() -> tuple[Any, SQLiteBackend]:
    """a collection over a real SQLite L1 holding its table, with no bus."""
    schema = TableSchema(
        name="rows", primary_key="id", columns=[Column("id", STRING_TYPE), Column("payload", STRING_TYPE)]
    )
    metadata = MetaData()
    schema.to_sqlalchemy_table(metadata)
    l1 = SQLiteBackend(db_name=f"evict_off_loop_{uuid.uuid4().hex[:8]}")
    l1.initialize(metadata)
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, kv_key_scope="test-principal")
    return collection_for_schema(schema)(registry, DefaultCoreConfig(), None), l1


def _keys(count: int) -> list[str]:
    return [f"k{index:07d}" for index in range(count)]


def _cache(l1: SQLiteBackend, keys: list[str], payload: str = "x") -> None:
    l1.upsert_many("rows", [{"id": key, "payload": payload} for key in keys], "id")


def _held(collection: Any, keys: list[str]) -> list[bool]:
    return [collection.exists_in_cache_sync(key) for key in keys]


@pytest.fixture
def deleting_threads(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """the thread each L1 delete of the collection's own rows ran on (not of its cached scans)."""
    threads: list[int] = []
    original = SQLiteBackend.delete_many_by_id

    def recording(self: SQLiteBackend, table: str, *args: Any, **kwargs: Any) -> None:
        if table == "rows":
            threads.append(threading.get_ident())
        original(self, table, *args, **kwargs)

    monkeypatch.setattr(SQLiteBackend, "delete_many_by_id", recording)
    return threads


class TestManyKeysLeaveL1OffTheLoop:
    async def test_settling_many_keys_deletes_on_a_worker_thread_and_every_key_leaves(
        self, deleting_threads: list[int]
    ) -> None:
        collection, l1 = _cached_collection()
        keys = _keys(_MANY)
        _cache(l1, keys)
        await collection.invalidate_cache_many(keys)
        assert deleting_threads and threading.get_ident() not in deleting_threads, "L1 was deleted on the loop"
        assert _held(collection, keys) == [False] * len(keys)

    async def test_a_few_keys_delete_on_the_loop(self, deleting_threads: list[int]) -> None:
        collection, l1 = _cached_collection()
        keys = _keys(5)
        _cache(l1, keys)
        await collection.invalidate_cache_many(keys)
        assert deleting_threads == [threading.get_ident()]
        assert _held(collection, keys) == [False] * len(keys)

    async def test_every_listener_hears_every_key_after_its_row_has_left(self) -> None:
        collection, l1 = _cached_collection()
        keys = _keys(_MANY)
        _cache(l1, keys)
        heard: list[tuple[Any, bool]] = []
        collection.add_l1_change_listener(lambda key: heard.append((key, collection.exists_in_cache_sync(key))))
        await collection.invalidate_cache_many(keys)
        assert heard == [(key, False) for key in keys]

    async def test_a_read_that_began_before_the_eviction_may_not_cache_what_it_read(self) -> None:
        collection, l1 = _cached_collection()
        keys = _keys(_MANY)
        _cache(l1, keys)
        ticket = collection.scan_ticket()
        await collection.invalidate_cache_many(keys)
        assert collection.write_to_cache_sync({"id": keys[0], "payload": "old"}, read_since=ticket) is False
        assert not collection.exists_in_cache_sync(keys[0])

    async def test_a_failed_delete_raises_unchanged_rolls_back_and_still_tells_every_listener(self) -> None:
        collection, l1 = _cached_collection()
        keys = _keys(_MANY)
        _cache(l1, keys)
        l1.get_connection().execute(
            f"CREATE TRIGGER refuse BEFORE DELETE ON rows WHEN OLD.id = '{keys[100]}' "
            "BEGIN SELECT RAISE(ABORT, 'refused'); END"
        )
        heard: list[Any] = []
        collection.add_l1_change_listener(heard.append)
        with pytest.raises(sqlite3.IntegrityError, match="refused"):
            await collection.invalidate_cache_many(keys)
        assert heard == keys
        # the thread's transaction rolled back: every row is still there, and the loop's next write is not held
        assert _held(collection, keys) == [True] * len(keys)
        l1.get_connection().execute("DROP TRIGGER refuse")
        _cache(l1, ["after"])
        assert collection.exists_in_cache_sync("after")

    async def test_a_cancel_waits_for_the_eviction_to_finish_and_is_raised_after(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        collection, l1 = _cached_collection()
        keys = _keys(_MANY)
        _cache(l1, keys)
        deleting = threading.Event()
        release = threading.Event()
        original = SQLiteBackend.delete_many_by_id

        def slow(self: SQLiteBackend, *args: Any, **kwargs: Any) -> None:
            deleting.set()
            release.wait(5)
            original(self, *args, **kwargs)

        monkeypatch.setattr(SQLiteBackend, "delete_many_by_id", slow)
        heard: list[tuple[Any, bool]] = []
        collection.add_l1_change_listener(lambda key: heard.append((key, collection.exists_in_cache_sync(key))))
        settling = asyncio.create_task(collection.invalidate_cache_many(keys))
        while not deleting.is_set():
            await asyncio.sleep(0.001)
        settling.cancel()
        await asyncio.sleep(0.02)
        assert not settling.done(), "the cancel did not wait for the delete under way"
        assert heard == [], "a listener heard a key before its row had left"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await settling
        assert heard == [(key, False) for key in keys]


async def _ticker(stop: asyncio.Event, late: list[float], every: float = 0.002) -> None:
    """sleep ``every`` over and over, keeping how much later than asked each wake came."""
    while not stop.is_set():
        asked = time.perf_counter()
        await asyncio.sleep(every)
        late.append(time.perf_counter() - asked - every)


@pytest.mark.loop_timing
class TestTheLoopRunsWhileManyKeysLeaveL1:
    async def test_a_ticker_is_never_held_while_100k_keys_leave_l1(self) -> None:
        collection, l1 = _cached_collection()
        keys = _keys(100_000)
        _cache(l1, keys, payload="x" * 200)
        # how long the delete alone takes here, so the bound below is in proportion to this machine
        _, timing_l1 = _cached_collection()
        _cache(timing_l1, keys, payload="x" * 200)
        started = time.perf_counter()
        timing_l1.delete_many_by_id("rows", [(key,) for key in keys], ("id",))
        one_delete = time.perf_counter() - started
        stop = asyncio.Event()
        late: list[float] = []
        ticking = asyncio.create_task(_ticker(stop, late))
        await asyncio.sleep(0.01)
        await collection.invalidate_cache_many(keys)
        stop.set()
        await ticking
        assert _held(collection, keys[:10]) == [False] * 10
        worst = max(late)
        assert worst < max(0.05, one_delete / 3), (
            f"the loop was held {worst * 1000:.0f} ms while 100k keys left L1 "
            f"(the delete alone takes {one_delete * 1000:.0f} ms)"
        )
