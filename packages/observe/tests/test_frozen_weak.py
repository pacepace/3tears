"""Read-only holders of weak references: each change builds a new holder and leaves the old one as it was, and no
holder keeps an object alive."""

from __future__ import annotations

import gc

from threetears.observe import FrozenWeakKeys, FrozenWeakSet, FrozenWeakValues


class _Thing:
    """weakly referable, hashed by identity."""


def test_a_weak_set_is_replaced_whole_and_holds_nothing_alive() -> None:
    first, second = _Thing(), _Thing()
    empty: FrozenWeakSet[_Thing] = FrozenWeakSet()
    one = empty.with_item(first)
    two = one.with_item(second)
    assert first not in empty and list(empty) == []
    assert first in one and second not in one
    assert list(two) == [first, second] and len(two) == 2
    assert two.with_item(first) is two
    gone = two.without_item(first)
    assert list(gone) == [second] and list(two) == [first, second]
    assert gone.without_item(first) is gone
    del second
    gc.collect()
    assert list(gone) == [] and len(two) == 1
    assert 3 not in two  # an object that cannot be held weakly is never held


def test_weak_keys_map_values_by_a_live_key_and_drop_dead_keys_on_the_next_build() -> None:
    key, other = _Thing(), _Thing()
    held: FrozenWeakKeys[_Thing, int] = FrozenWeakKeys().with_entry(key, 1)
    moved = held.with_entry(key, 2).with_entry(other, 3)
    assert held[key] == 1 and moved[key] == 2 and moved.get(other) == 3
    assert key in moved and moved.get(_Thing(), -1) == -1
    assert moved.get([1], -1) == -1 and [1] not in moved  # unhashable: never held
    assert dict(moved.without_key(key).items()) == {other: 3}
    assert moved.without_key(_Thing()) is moved
    del other
    gc.collect()
    assert len(moved) == 1
    assert dict(moved.with_entry(key, 4).items()) == {key: 4}


def test_weak_values_read_none_once_the_value_is_gone() -> None:
    value, other = _Thing(), _Thing()
    held: FrozenWeakValues[int, _Thing] = FrozenWeakValues().with_entry(1, value).with_entry(2, other)
    assert held.get(1) is value and held.get(3) is None
    assert list(held.values()) == [value, other]
    assert held.without_key(3) is held
    assert dict(held.without_key(1).items()) == {2: other}
    del value
    gc.collect()
    assert held.get(1) is None and len(held) == 1
