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

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, Final

__all__ = [
    "EMPTY",
    "frozen",
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
    """``mapping`` with ``key`` set to ``value``, as a new read-only snapshot.

    :param mapping: the snapshot
    :ptype mapping: Mapping
    :param key: the key
    :ptype key: Any
    :param value: its value
    :ptype value: Any
    :return: the new snapshot
    :rtype: Mapping
    """
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
    """``mapping`` without ``key``, as a new read-only snapshot (the same one when it holds no ``key``).

    :param mapping: the snapshot
    :ptype mapping: Mapping
    :param key: the key
    :ptype key: Any
    :return: the new snapshot
    :rtype: Mapping
    """
    if key not in mapping:
        return mapping
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
