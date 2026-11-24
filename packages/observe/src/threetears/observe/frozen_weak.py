"""read-only holders of weak references, replaced whole: in place of ``weakref.WeakSet`` / ``WeakKeyDictionary`` /
``WeakValueDictionary``.

A weak container changes in place under a reader: an add or a discard by another coroutine or thread, or the GC dropping
an entry while it is iterated. Serving code's shared state is never changed in place (decisions 45/46,
:mod:`threetears.core.frozen_state`): its owner builds the next immutable value and assigns it whole. These are that
shape over weak references. Each holder keeps only weak references, so it never keeps an object alive; a dead one reads
as absent at once and is dropped from the next holder built (each ``with_*`` / ``without_*``).

Standard library only: every 3tears package depends on this one (``threetears.nats`` included, which cannot reach
:mod:`threetears.core`), so any of them may hold weak state here without a new edge.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import Any, overload

__all__ = ["FrozenWeakKeys", "FrozenWeakSet", "FrozenWeakValues"]


def _live[T](refs: Iterable[weakref.ref[T]]) -> Iterator[T]:
    """the objects ``refs`` still point at, in order.

    :param refs: the weak references
    :ptype refs: Iterable[weakref.ref]
    :return: the live objects
    :rtype: Iterator
    """
    for ref in refs:
        item = ref()
        if item is not None:
            yield item


class FrozenWeakSet[T]:
    """a read-only set of objects held weakly, in place of ``weakref.WeakSet``: :meth:`with_item` and
    :meth:`without_item` build the next one, which the owner assigns whole. Membership is the objects' own equality and
    hash, as a ``WeakSet``'s is, and costs one hash.
    """

    __slots__ = ("_refs",)

    def __init__(self, items: Iterable[T] = ()) -> None:
        """
        :param items: the objects (each hashable and weakly referable)
        :ptype items: Iterable
        """
        self._refs: Mapping[weakref.ref[T], None] = MappingProxyType(dict.fromkeys(weakref.ref(item) for item in items))

    def __contains__(self, item: object) -> bool:
        """whether ``item`` is held (an object handed in is alive, so a held one is).

        :param item: the object
        :ptype item: object
        :return: whether it is
        :rtype: bool
        """
        try:
            return weakref.ref(item) in self._refs
        except TypeError:
            return False

    def __iter__(self) -> Iterator[T]:
        """the live objects, oldest first.

        :return: them
        :rtype: Iterator
        """
        return _live(self._refs)

    def __len__(self) -> int:
        """how many held objects are alive.

        :return: the count
        :rtype: int
        """
        return sum(1 for _ in _live(self._refs))

    def with_item(self, item: T) -> FrozenWeakSet[T]:
        """this set with ``item``, dead references dropped (the same one when ``item`` is held already).

        :param item: the object
        :ptype item: Any
        :return: the next set
        :rtype: FrozenWeakSet
        """
        if item in self:
            return self
        return FrozenWeakSet((*_live(self._refs), item))

    def without_item(self, item: object) -> FrozenWeakSet[T]:
        """this set without ``item``, dead references dropped (the same one when ``item`` is not held).

        :param item: the object
        :ptype item: object
        :return: the next set
        :rtype: FrozenWeakSet
        """
        if item not in self:
            return self
        gone = weakref.ref(item)
        return FrozenWeakSet(held for ref in self._refs if ref != gone and (held := ref()) is not None)


class FrozenWeakKeys[K, V]:
    """a read-only mapping whose keys are held weakly, in place of ``weakref.WeakKeyDictionary``: an entry goes with
    its key's object. :meth:`with_entry` and :meth:`without_key` build the next one, which the owner assigns whole.
    """

    __slots__ = ("_entries",)

    def __init__(self, entries: Iterable[tuple[K, V]] = ()) -> None:
        """
        :param entries: the (key, value) pairs (each key hashable and weakly referable)
        :ptype entries: Iterable[tuple]
        """
        self._entries: Mapping[weakref.ref[Any], V] = MappingProxyType(
            {weakref.ref(key): value for key, value in entries}
        )

    @overload
    def get(self, key: object) -> V | None: ...

    @overload
    def get(self, key: object, default: V) -> V: ...

    def get(self, key: object, default: V | None = None) -> V | None:
        """``key``'s value while its object lives, else ``default`` (a key that cannot be held weakly, or hashed, is
        never held).

        :param key: the key's object
        :ptype key: object
        :param default: what an absent key reads as
        :ptype default: Any
        :return: the value
        :rtype: Any
        """
        try:
            return self._entries.get(weakref.ref(key), default)
        except TypeError:
            return default

    def __getitem__(self, key: K) -> V:
        """``key``'s value.

        :param key: the key's object
        :ptype key: Any
        :return: the value
        :rtype: Any
        :raises KeyError: the key is not held
        """
        return self._entries[weakref.ref(key)]

    def __contains__(self, key: object) -> bool:
        """whether ``key`` is held (a key that cannot be held weakly, or hashed, never is).

        :param key: the key's object
        :ptype key: object
        :return: whether it is
        :rtype: bool
        """
        try:
            return weakref.ref(key) in self._entries
        except TypeError:
            return False

    def items(self) -> Iterator[tuple[K, V]]:
        """the entries whose key still lives.

        :return: them
        :rtype: Iterator[tuple]
        """
        for ref, value in self._entries.items():
            key = ref()
            if key is not None:
                yield key, value

    def __len__(self) -> int:
        """how many entries' keys still live.

        :return: the count
        :rtype: int
        """
        return sum(1 for _ in self.items())

    def with_entry(self, key: K, value: V) -> FrozenWeakKeys[K, V]:
        """this mapping with ``key`` set to ``value``, dead keys dropped.

        :param key: the key's object
        :ptype key: Any
        :param value: its value
        :ptype value: Any
        :return: the next mapping
        :rtype: FrozenWeakKeys
        """
        return FrozenWeakKeys((*((held, was) for held, was in self.items() if held != key), (key, value)))

    def without_key(self, key: object) -> FrozenWeakKeys[K, V]:
        """this mapping without ``key``, dead keys dropped (the same one when ``key`` is not held).

        :param key: the key's object
        :ptype key: object
        :return: the next mapping
        :rtype: FrozenWeakKeys
        """
        if key not in self:
            return self
        return FrozenWeakKeys((held, was) for held, was in self.items() if held != key)


class FrozenWeakValues[K, V]:
    """a read-only mapping whose values are held weakly, in place of ``weakref.WeakValueDictionary``: an entry goes
    with its value's object. :meth:`with_entry` and :meth:`without_key` build the next one, which the owner assigns
    whole.
    """

    __slots__ = ("_entries",)

    def __init__(self, entries: Iterable[tuple[K, V]] = ()) -> None:
        """
        :param entries: the (key, value) pairs (each value weakly referable)
        :ptype entries: Iterable[tuple]
        """
        self._entries: Mapping[K, weakref.ref[V]] = MappingProxyType(
            {key: weakref.ref(value) for key, value in entries}
        )

    def get(self, key: K) -> V | None:
        """``key``'s value while it lives, else None.

        :param key: the key
        :ptype key: Any
        :return: the value
        :rtype: Any
        """
        ref = self._entries.get(key)
        return None if ref is None else ref()

    def items(self) -> Iterator[tuple[K, V]]:
        """the entries whose value still lives.

        :return: them
        :rtype: Iterator[tuple]
        """
        for key, ref in self._entries.items():
            value = ref()
            if value is not None:
                yield key, value

    def values(self) -> Iterator[V]:
        """the values that still live.

        :return: them
        :rtype: Iterator
        """
        return (value for _key, value in self.items())

    def __len__(self) -> int:
        """how many entries' values still live.

        :return: the count
        :rtype: int
        """
        return sum(1 for _ in self.items())

    def with_entry(self, key: K, value: V) -> FrozenWeakValues[K, V]:
        """this mapping with ``key`` set to ``value``, dead values dropped.

        :param key: the key
        :ptype key: Any
        :param value: its value's object
        :ptype value: Any
        :return: the next mapping
        :rtype: FrozenWeakValues
        """
        return FrozenWeakValues((*((held, was) for held, was in self.items() if held != key), (key, value)))

    def without_key(self, key: K) -> FrozenWeakValues[K, V]:
        """this mapping without ``key``, dead values dropped (the same one when ``key`` is not held).

        :param key: the key
        :ptype key: Any
        :return: the next mapping
        :rtype: FrozenWeakValues
        """
        if key not in self._entries:
            return self
        return FrozenWeakValues((held, was) for held, was in self.items() if held != key)
