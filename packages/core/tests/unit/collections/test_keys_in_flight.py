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

import pytest

from threetears.core.collections import packed_keys
from threetears.core.collections.base import _FENCE_ERA_CHANGES, _L1Fence


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
