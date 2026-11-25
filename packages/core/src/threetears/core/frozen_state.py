"""immutable bookkeeping for an owner's state: a value is never changed in place, it is replaced whole.

An owner (a copy follower, a fold, a registry) keeps small per-table or per-state bookkeeping beside its Arrow data:
which tables wait, why, at what mark. Each such value is a read-only snapshot (a :class:`~types.MappingProxyType`, a
``frozenset``, a tuple), and a change builds the next snapshot and assigns it in one statement. A reader holding the old
snapshot (a status, a coroutine across an await, a thread handed it) never sees it change under it. Each is per table
or per state, never per key (per-key state is Arrow), so rebuilding one is cheap.

The same shape over weak references (in place of ``weakref.WeakSet`` / ``WeakKeyDictionary`` /
``WeakValueDictionary``) is :mod:`threetears.observe.frozen_weak`, where every 3tears package can reach it.

Stdlib only: shared_arrow, security and the collections import it, so it imports no pyarrow, asyncpg or
threetears.core.collections (shared_arrow importing that package is a cycle; security runs without the arrow extra).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

__all__ = [
    "EMPTY",
    "LoggedMap",
    "frozen",
    "logged",
    "with_chunk",
    "with_entries",
    "with_entry",
    "with_newest",
    "without_entries",
    "without_entry",
]

#: the empty snapshot, shared: it never changes, so every owner may start from it
EMPTY: Final[Mapping[Any, Any]] = MappingProxyType({})


def frozen[K, V](entries: Mapping[K, V] | Iterable[tuple[K, V]] = ()) -> Mapping[K, V]:
    """a read-only snapshot of ``entries``.

    :param entries: a mapping, or its (key, value) pairs
    :ptype entries: Mapping | Iterable[tuple]
    :return: the snapshot
    :rtype: Mapping
    """
    return MappingProxyType(dict(entries))


def with_entry[K, V](mapping: Mapping[K, V], key: K, value: V) -> Mapping[K, V]:
    """``mapping`` with ``key`` set to ``value``, as a new read-only snapshot (a :class:`LoggedMap` stays one).

    :param mapping: the snapshot
    :ptype mapping: Mapping
    :param key: the key
    :ptype key: Any
    :param value: its value
    :ptype value: Any
    :return: the new snapshot
    :rtype: Mapping
    """
    if isinstance(mapping, LoggedMap):
        return mapping.set(key, value)
    return MappingProxyType({**mapping, key: value})


def with_newest[K, V](mapping: Mapping[K, V], key: K, value: V, *, limit: int | None = None) -> Mapping[K, V]:
    """``mapping`` with ``key`` set to ``value`` as its newest entry, the oldest dropped past ``limit`` (a least
    recently used order: the first entry is the oldest).

    :param mapping: the snapshot
    :ptype mapping: Mapping
    :param key: the key
    :ptype key: Any
    :param value: its value
    :ptype value: Any
    :param limit: the most entries kept; None keeps every one
    :ptype limit: int | None
    :return: the new snapshot
    :rtype: Mapping
    """
    kept = [(held, was) for held, was in mapping.items() if held != key]
    if limit is not None:
        kept = kept[max(0, len(kept) + 1 - limit) :]
    return MappingProxyType({**dict(kept), key: value})


def with_entries[K, V](mapping: Mapping[K, V], entries: Mapping[K, V]) -> Mapping[K, V]:
    """``mapping`` with every entry of ``entries`` set, as a new read-only snapshot.

    :param mapping: the snapshot
    :ptype mapping: Mapping
    :param entries: the entries to set
    :ptype entries: Mapping
    :return: the new snapshot
    :rtype: Mapping
    """
    return MappingProxyType({**mapping, **entries})


def without_entry[K, V](mapping: Mapping[K, V], key: K) -> Mapping[K, V]:
    """``mapping`` without ``key``, as a new read-only snapshot (the same one when it holds no ``key``; a
    :class:`LoggedMap` stays one).

    :param mapping: the snapshot
    :ptype mapping: Mapping
    :param key: the key
    :ptype key: Any
    :return: the new snapshot
    :rtype: Mapping
    """
    if key not in mapping:
        return mapping
    if isinstance(mapping, LoggedMap):
        return mapping.unset(key)
    return MappingProxyType({held: value for held, value in mapping.items() if held != key})


def without_entries[K, V](mapping: Mapping[K, V], keys: Iterable[K]) -> Mapping[K, V]:
    """``mapping`` without any of ``keys``, as a new read-only snapshot.

    :param mapping: the snapshot
    :ptype mapping: Mapping
    :param keys: the keys
    :ptype keys: Iterable
    :return: the new snapshot
    :rtype: Mapping
    """
    gone = frozenset(keys)
    if not gone & mapping.keys():
        return mapping
    return MappingProxyType({held: value for held, value in mapping.items() if held not in gone})


def with_chunk[T](chunks: tuple[tuple[T, ...], ...], chunk: tuple[T, ...]) -> tuple[tuple[T, ...], ...]:
    """``chunks`` with ``chunk`` after them, as a new tuple of tuples: an append-only log that grows with rows or
    messages, replaced whole and never changed in place. The newest chunks are merged while the one before is no larger
    (log-structured), so an entry appended one at a time is copied about log2(n) times over its life, never n, and the
    log holds about log2(n) chunks. Read it in order with ``itertools.chain.from_iterable``.

    :param chunks: the log so far, oldest chunk first
    :ptype chunks: tuple[tuple[T, ...], ...]
    :param chunk: the newest entries, in order
    :ptype chunk: tuple[T, ...]
    :return: the log with them (the same one when ``chunk`` is empty)
    :rtype: tuple[tuple[T, ...], ...]
    """
    if not chunk:
        return chunks
    while chunks and len(chunks[-1]) <= len(chunk):
        chunk = (*chunks[-1], *chunk)
        chunks = chunks[:-1]
    return (*chunks, chunk)


#: the largest chunk of a :class:`LoggedMap` merged into another: one this large stays as it is, so no change copies
#: more than about twice this many keys in one step. Merged without a bound, the change that made a map of 32,768 keys
#: one chunk copied every key (4.6 ms, two maps a fake bucket's write: the event loop held 9 ms by one write); with it a
#: lookup reads a chunk a 4,096 keys past the first, a dict lookup each
_MERGED_MAX: Final = 4096

#: the value of a key a newer chunk of a :class:`LoggedMap` removed
_REMOVED: Final = object()


@dataclass(frozen=True, slots=True)
class _Slot:
    """a key in one chunk of a :class:`LoggedMap`: its value (:data:`_REMOVED` once removed), whether it was set while
    the map held no such key (it then takes the last place, as a dict re-adding a removed key does), and whether a
    chunk older than the one it was written to may hold the key (a removal is kept only while one may)."""

    value: Any
    reborn: bool
    deep: bool


def _merged(older: Mapping[Any, _Slot], newer: Mapping[Any, _Slot], *, bottom: bool) -> Mapping[Any, _Slot]:
    """two adjacent chunks of a :class:`LoggedMap` as one, in a dict's order: a key set again keeps its place, a key
    re-added after a removal takes the last. A removal is dropped once no older chunk may hold the key.

    Both chunks are merged by the dict's own operations (C, no Python step a key); only the keys both hold are walked
    (a removal kept is of a key an older chunk holds): a merge into a chunk of 16,384 keys took 31 ms walking every
    key, as a refill's writes are each a new key."""
    both = older.keys() & newer.keys()
    moved = frozenset(key for key in both if newer[key].value is _REMOVED or newer[key].reborn)
    if moved:
        # a key removed or re-added in the newer chunk leaves its older place
        merged = {key: slot for key, slot in older.items() if key not in moved}
        merged.update(newer)
    else:
        merged = {**older, **newer}
    for key in both:
        slot, was = newer[key], older[key]
        if slot.value is _REMOVED and (bottom or not was.deep):
            # nothing older may hold the key any more: the removal has nothing left to hide
            del merged[key]
        elif slot.deep != was.deep or (not slot.reborn and was.reborn):
            # what is under the key, and (set again in place) its place and whether it was re-added there, are the
            # older chunk's
            merged[key] = _Slot(slot.value, slot.reborn or was.reborn, was.deep)
    return MappingProxyType(merged)


@dataclass(frozen=True, slots=True, eq=False)
class LoggedMap[K, V](Mapping[K, V]):
    """a read-only snapshot of a map that grows with rows or keys, set and unset a key at a time: the next snapshot
    shares the last one's chunks, so a change copies a key about log2(n) times over its life, never n.

    :func:`with_entry` and :func:`without_entry` keep one a ``LoggedMap``. Its chunks are merged while the one before
    is no larger (log-structured, as :func:`with_chunk`), so a lookup reads about log2(n) chunks, and its order is a
    dict's: a key set again keeps its place, one added (or re-added after its removal) takes the last.

    Why: a map replaced whole on every change (:func:`with_entry` of a ``MappingProxyType``) copies every key a change,
    and a fake NATS bucket taking a 33,786-row refill a row at a time held the event loop 15 ms a write by the end
    (the loop-stall guard, 2026-11-25).
    """

    #: the chunks, oldest first: each about half the size of the one before or less
    chunks: tuple[Mapping[K, _Slot], ...] = ()
    _size: int = 0

    def _slot(self, key: object) -> _Slot | None:
        for chunk in reversed(self.chunks):
            slot = chunk.get(key)  # type: ignore[arg-type]
            if slot is not None:
                return slot
        return None

    def __getitem__(self, key: K) -> V:
        slot = self._slot(key)
        if slot is None or slot.value is _REMOVED:
            raise KeyError(key)
        value: V = slot.value
        return value

    def __contains__(self, key: object) -> bool:
        slot = self._slot(key)
        return slot is not None and slot.value is not _REMOVED

    def __iter__(self) -> Iterator[K]:
        if len(self.chunks) == 1:
            return iter(self.chunks[0])
        whole: Mapping[K, _Slot] = MappingProxyType({})
        for chunk in self.chunks:
            whole = _merged(whole, chunk, bottom=not whole)
        return iter([key for key, slot in whole.items() if slot.value is not _REMOVED])

    def __len__(self) -> int:
        return self._size

    def _with(self, key: K, slot: _Slot, size: int) -> LoggedMap[K, V]:
        chunks = self.chunks
        chunk: Mapping[K, _Slot] = MappingProxyType({key: slot})
        while chunks and len(chunks[-1]) <= len(chunk) and len(chunk) < _MERGED_MAX:
            chunk = _merged(chunks[-1], chunk, bottom=len(chunks) == 1)
            chunks = chunks[:-1]
        return LoggedMap((*chunks, chunk), size)

    def set(self, key: K, value: V) -> LoggedMap[K, V]:
        """this map with ``key`` set to ``value``.

        :param key: the key
        :ptype key: Any
        :param value: its value
        :ptype value: Any
        :return: the new snapshot
        :rtype: LoggedMap
        """
        found = self._slot(key)
        held = found is not None and found.value is not _REMOVED
        return self._with(key, _Slot(value, not held, found is not None), self._size if held else self._size + 1)

    def unset(self, key: K) -> LoggedMap[K, V]:
        """this map without ``key`` (the same one when it holds no ``key``).

        :param key: the key
        :ptype key: Any
        :return: the new snapshot
        :rtype: LoggedMap
        """
        if key not in self:
            return self
        return self._with(key, _Slot(_REMOVED, False, True), self._size - 1)


def logged[K, V](entries: Mapping[K, V] | Iterable[tuple[K, V]] = ()) -> LoggedMap[K, V]:
    """a :class:`LoggedMap` of ``entries``.

    :param entries: a mapping, or its (key, value) pairs
    :ptype entries: Mapping | Iterable[tuple]
    :return: the snapshot
    :rtype: LoggedMap
    """
    whole = dict(entries)
    if not whole:
        return LoggedMap()
    return LoggedMap((MappingProxyType({key: _Slot(value, False, False) for key, value in whole.items()}),), len(whole))
