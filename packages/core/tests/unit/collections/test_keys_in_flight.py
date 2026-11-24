"""the keys in flight -- the L1 fence's and the write buffer's -- are immutable packed batches, never state a key.

Pace, 2026-11-23: ``_L1Fence._keys`` (a mutable record a key in flight) and ``WriteBuffer._buf`` / ``_in_flight`` (a
dict and a set a pending write) were per-key Python objects and mutable shared state. Each is now immutable values its
owner replaces whole: the fence a count, its writes in flight as one packed batch and its recorded changes as eras;
the buffer a tuple of sorted runs, each its keys as one packed string. These tests pin the behaviour both kept (who
may cache, what a drain claims and in what order, the claim's version guard), the shape (no dict or set, nothing
changed in place), and the cost bounds (an era's copy, the buffer's runs logarithmic in its keys).

Races: every method of both runs on the loop with no await between reading its state and replacing it (the buffer's
under its lock), so no interleaving of two coroutines could reach a half-made change before or after; the tests below
drive the interleavings a flush and a save, or a read and a write, really take, and pass on the old code as well.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

import pytest

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import packed_keys
from threetears.core.collections.base import _FENCE_ERA_CHANGES, _L1Fence
from threetears.core.collections.flush import WriteBuffer


class TestPackedKeys:
    def test_a_key_text_is_exact(self) -> None:
        """Flips if two different keys share a text: a part holding the separator, or a key split differently."""
        keys = [("a", "b"), ("a\x1fb",), ("a\x00b",), ("a", "\x1eb"), ("",), ("", ""), ("ab",), ()]
        texts = [packed_keys.key_text(key) for key in keys]
        assert len(set(texts)) == len(keys)
        assert not any(packed_keys.SEP in text for text in texts)

    def test_a_batch_names_its_keys_and_no_other(self) -> None:
        """Flips if a batch names a key it does not hold: one holding another's text inside it."""
        batch = packed_keys.packed([packed_keys.key_text(("ab", "c")), packed_keys.key_text(("x",))])
        assert packed_keys.names(batch, packed_keys.key_text(("ab", "c")))
        assert packed_keys.names(batch, packed_keys.key_text(("x",)))
        for other in (("a",), ("b", "c"), ("ab",), ("c",), ("x", "y")):
            assert not packed_keys.names(batch, packed_keys.key_text(other))


def _state_types(owner: object, names: tuple[str, ...]) -> set[type]:
    return {type(getattr(owner, name)) for name in names}


class TestTheFence:
    def test_the_fence_holds_no_dict_or_set(self) -> None:
        """Flips if the fence keeps a record a key: its state is counts, one packed batch and a tuple of eras."""
        fence = _L1Fence()
        tickets = [fence.begin((f"k{i}",), writing=i % 2 == 0) for i in range(100)]
        assert _state_types(fence, fence.__slots__) <= {int, str, tuple}
        for ticket in tickets:
            fence.end(ticket)

    def test_a_read_is_overtaken_by_a_write_of_its_key_only(self) -> None:
        fence = _L1Fence()
        read = fence.begin(("a",), writing=False)
        other = fence.begin(("b",), writing=False)
        write = fence.begin(("a",), writing=True)
        assert not fence.still_newest(read)
        assert fence.still_newest(other)
        assert fence.still_newest(write)
        for ticket in (read, other, write):
            fence.end(ticket)

    def test_a_read_taken_while_a_write_is_in_flight_is_contended_until_none_is(self) -> None:
        fence = _L1Fence()
        first = fence.begin(("a",), writing=True)
        second = fence.begin(("a",), writing=True)
        assert not fence.still_newest(first) and not fence.still_newest(second)
        fence.end(first)
        read = fence.begin(("a",), writing=False)
        assert not fence.still_newest(read)  # the second write is still in flight
        fence.end(read)
        fence.end(second)
        read = fence.begin(("a",), writing=False)
        assert fence.still_newest(read)
        fence.end(read)

    def test_an_eviction_batch_prefix_and_drop_reach_the_keys_they_name(self) -> None:
        fence = _L1Fence()
        tickets = {key: fence.begin(key, writing=False) for key in (("s1", "g1"), ("s1", "g2"), ("s2", "g1"))}
        fence.changed_many(packed_keys.packed([packed_keys.key_text(("s1", "g2"))]), 1)
        assert [fence.still_newest(t) for t in tickets.values()] == [True, False, True]
        fence.changed_under(((0, "s2"),))
        assert [fence.still_newest(t) for t in tickets.values()] == [True, False, False]
        fence.changed_all()
        assert not any(fence.still_newest(t) for t in tickets.values())
        for ticket in tickets.values():
            fence.end(ticket)

    def test_a_change_before_a_ticket_does_not_reach_it(self) -> None:
        fence = _L1Fence()
        holder = fence.begin(("x",), writing=False)
        fence.changed(("a",))
        read = fence.begin(("a",), writing=False)
        assert fence.still_newest(read)
        fence.end(read)
        fence.end(holder)

    def test_a_long_held_ticket_sees_a_change_many_eras_later_and_the_eras_go_with_it(self) -> None:
        """Flips if a change recorded in a later era is missed, or eras outlive the tickets that need them."""
        fence = _L1Fence()
        held = fence.begin(("a",), writing=False)
        for i in range(10 * _FENCE_ERA_CHANGES):
            fence.changed((f"other{i}",))
        assert fence.still_newest(held)
        assert fence.recorded == (_FENCE_ERA_CHANGES,) * 10  # an era's copy is bounded, so the changes run over eras
        fence.changed(("a",))
        assert not fence.still_newest(held)
        fence.end(held)
        assert fence.recorded == (0,)

    def test_with_nothing_held_a_change_is_only_counted(self) -> None:
        fence = _L1Fence()
        fence.changed(("a",))
        fence.changed_many(packed_keys.NO_KEYS, 33_786)
        assert fence.recorded == (0,)
        assert fence.begin_scan().seen == 33_787

    async def test_a_read_racing_a_write_on_the_loop_declines_to_cache(self) -> None:
        """Two coroutines: a read awaits between its ticket and its check while a write of the key begins and ends.
        Holds on the old code too: no await inside either fence call, so no interleaving reaches a half-made change."""
        fence = _L1Fence()
        read_may_cache: list[bool] = []

        async def read() -> None:
            with fence.watching(("a",), writing=False) as ticket:
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                read_may_cache.append(fence.still_newest(ticket))

        async def write() -> None:
            await asyncio.sleep(0)
            with fence.watching(("a",), writing=True):
                await asyncio.sleep(0)

        await asyncio.gather(read(), write())
        assert read_may_cache == [False]
        assert not fence.watched and not fence.begin_scan().contended


def _write(i: int) -> dict[str, Any]:
    return {"id": i}


class TestTheWriteBuffer:
    async def test_the_buffer_holds_no_dict_or_set_and_replaces_its_runs_whole(self) -> None:
        """Flips if the buffer keeps a dict or set a key, or changes a run in place."""
        buffer = WriteBuffer()
        for i in range(50):
            await buffer.add("t", i, _write(i))
        before = vars(buffer)["_runs"]
        held = [(run.keys, run.states) for run in before]
        await buffer.add("t", 3, _write(99))
        await buffer.drain()
        assert [(run.keys, run.states) for run in before] == held
        state = {name: value for name, value in vars(buffer).items() if name != "_lock"}
        assert not any(isinstance(value, (dict, set)) for value in state.values())

    async def test_a_drain_claims_in_the_order_keys_entered_and_a_rewrite_keeps_its_place(self) -> None:
        buffer = WriteBuffer()
        for key in ("c", "a", "b"):
            await buffer.add("t", key, {"v": 1})
        await buffer.add("t", "c", {"v": 2})
        claimed = await buffer.drain()
        assert [(w.entity_id, w.data["v"]) for w in claimed] == [("c", 2), ("a", 1), ("b", 1)]
        assert await buffer.re_enqueue("t", "a", {"v": 3}, retries=1)
        await buffer.add("t", "d", {"v": 1})
        assert [(w.entity_id, w.data["v"]) for w in await buffer.drain()] == [("a", 3), ("d", 1)]

    async def test_a_write_landing_during_a_flush_survives_its_ack_and_its_retry(self) -> None:
        buffer = WriteBuffer()
        await buffer.add("t", 1, _write(1))
        await buffer.add("t", 2, _write(2))
        claimed = await buffer.drain()
        assert len(claimed) == 2 and await buffer.drain() == []
        await buffer.add("t", 1, {"id": 1, "newer": True})
        await buffer.ack("t", 1)
        assert not await buffer.re_enqueue("t", 1, _write(1), retries=1)
        await buffer.ack("t", 2)
        assert buffer.pending_count() == 1
        assert [w.data for w in await buffer.drain()] == [{"id": 1, "newer": True}]

    async def test_remove_drops_pending_and_claimed_writes(self) -> None:
        buffer = WriteBuffer()
        await buffer.add("t", 1, _write(1))
        await buffer.drain()
        await buffer.add("t", 2, _write(2))
        assert await buffer.remove("t", 1) and await buffer.remove("t", 2)
        assert not await buffer.remove("t", 3)
        assert buffer.pending_count() == 0 and await buffer.drain() == []
        await buffer.ack("t", 1)  # its claim went with it
        assert buffer.pending_count() == 0

    async def test_a_restarts_durable_rows_are_claimed_once_and_acked(self) -> None:
        l1 = SQLiteBackend(":memory:")
        first = WriteBuffer(l1)
        await first.add("t", "a", {"id": "a"})
        await first.add("t", "b", {"id": "b"})
        after = WriteBuffer(l1)
        claimed = await after.drain()
        assert sorted(w.entity_id for w in claimed) == ["a", "b"]
        assert await after.drain() == [] and after.pending_count() == 0
        assert await after.re_enqueue("t", "a", {"id": "a", "retry": 1}, retries=1)
        await after.ack("t", "b")
        assert after.pending_count() == 1
        assert [w.entity_id for w in await after.drain()] == ["a"]

    async def test_the_runs_stay_logarithmic_and_a_drain_leaves_one(self) -> None:
        """Flips if a change copies every key (one run rebuilt whole) or runs pile up a key each."""
        buffer = WriteBuffer()
        count = 5_000
        for i in range(count):
            await buffer.add("t", i, _write(i))
            assert len(buffer.run_sizes()) <= 2 * math.log2(i + 2) + 2
        assert buffer.pending_count() == count
        claimed = await buffer.drain()
        assert buffer.run_sizes() == (count,) and len(claimed) == count
        for write in claimed:
            await buffer.ack(write.table_name, write.entity_id)
        assert buffer.pending_count() == 0
        await buffer.drain()
        assert buffer.run_sizes() == ()

    async def test_a_flush_and_saves_racing_on_the_loop_lose_no_newer_write(self) -> None:
        """Two coroutines: a flush drains and acks write by write, yielding between, while saves rewrite the same keys.
        Every newer write is still buffered after the flush. Holds on the old code too: each method runs under the
        buffer's lock with no await inside, so no interleaving reaches a half-made change."""
        buffer = WriteBuffer()
        for i in range(200):
            await buffer.add("t", i, {"id": i, "v": 0})

        async def flush() -> None:
            for write in await buffer.drain():
                await asyncio.sleep(0)
                await buffer.ack(write.table_name, write.entity_id)

        async def saves() -> None:
            for i in range(0, 200, 3):
                await asyncio.sleep(0)
                await buffer.add("t", i, {"id": i, "v": 1})

        await asyncio.gather(flush(), saves())
        left = await buffer.drain()
        assert left and all(write.data["v"] == 1 for write in left)
        assert buffer.pending_count() == len(left)


@pytest.mark.parametrize("count", [33_786])
def test_a_bulk_eviction_is_one_change_and_one_search_a_check(count: int) -> None:
    """At a commit's real size (33,786 keys, the WD2 bench): the eviction is one recorded change holding one batch, and a
    held ticket's check is one search of it."""
    fence = _L1Fence()
    held = [fence.begin((f"S{i:02d}", f"{i:07d}"), writing=False) for i in range(100)]
    keys = [packed_keys.key_text((f"S{i % 50:02d}", f"{i:07d}")) for i in range(count)]
    fence.changed_many(packed_keys.packed(keys), count)
    assert fence.recorded == (1,)
    # the held keys the eviction names: (S<i>, i) for i < 50, where i % 50 == i
    assert [not fence.still_newest(ticket) for ticket in held] == [i < 50 for i in range(100)]
    for ticket in held:
        fence.end(ticket)
