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


#: a key a newer chunk of a :class:`LoggedMap` removed
_GONE: Final = object()


@dataclass(frozen=True, slots=True)
class _Slot:
    """a key's value in one chunk of a :class:`LoggedMap`, and whether it was set while the map held no such key (it
    then takes the last place, as a dict re-adding a removed key does)."""

    value: Any
    reborn: bool


def _merged(older: Mapping[Any, Any], newer: Mapping[Any, Any], *, bottom: bool) -> Mapping[Any, Any]:
    """two adjacent chunks of a :class:`LoggedMap` as one, in a dict's order: a key set again keeps its place, a key
    re-added after a removal takes the last. At the bottom (no older chunk) a removal is dropped, and nothing is
    re-added past anything."""
    merged: dict[Any, Any] = {}
    for key, slot in older.items():
        if key in newer:
            newest = newer[key]
            if newest is _GONE or newest.reborn:
                continue  # removed, or re-added: placed from the newer chunk below
            slot = newest if slot is _GONE else _Slot(newest.value, slot.reborn)
        merged[key] = slot
    for key, slot in newer.items():
        if key not in merged:
            merged[key] = slot
    if bottom:
        merged = {key: _Slot(slot.value, False) for key, slot in merged.items() if slot is not _GONE}
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
    chunks: tuple[Mapping[K, Any], ...] = ()
    _size: int = 0

    def _slot(self, key: object) -> Any:
        for chunk in reversed(self.chunks):
            slot = chunk.get(key)
            if slot is not None:
                return slot
        return None

    def __getitem__(self, key: K) -> V:
        slot = self._slot(key)
        if slot is None or slot is _GONE:
            raise KeyError(key)
        value: V = slot.value
        return value

    def __contains__(self, key: object) -> bool:
        slot = self._slot(key)
        return slot is not None and slot is not _GONE

    def __iter__(self) -> Iterator[K]:
        if len(self.chunks) == 1:
            return iter(self.chunks[0])
        whole: Mapping[K, Any] = MappingProxyType({})
        for chunk in self.chunks:
            whole = _merged(whole, chunk, bottom=False)
        return iter([key for key, slot in whole.items() if slot is not _GONE])

    def __len__(self) -> int:
        return self._size

    def _with(self, key: K, slot: Any, size: int) -> LoggedMap[K, V]:
        chunks = self.chunks
        chunk: Mapping[K, Any] = MappingProxyType({key: slot})
        while chunks and len(chunks[-1]) <= len(chunk):
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
        held = key in self
        return self._with(key, _Slot(value, not held), self._size if held else self._size + 1)

    def unset(self, key: K) -> LoggedMap[K, V]:
        """this map without ``key`` (the same one when it holds no ``key``).

        :param key: the key
        :ptype key: Any
        :return: the new snapshot
        :rtype: LoggedMap
        """
        if key not in self:
            return self
        return self._with(key, _GONE, self._size - 1)


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
    return LoggedMap((MappingProxyType({key: _Slot(value, False) for key, value in whole.items()}),), len(whole))
