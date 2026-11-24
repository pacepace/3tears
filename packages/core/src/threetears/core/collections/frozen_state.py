"""immutable bookkeeping for an owner's state: a value is never changed in place, it is replaced whole.

An owner (a copy follower, a fold, a registry) keeps small per-table or per-state bookkeeping beside its Arrow data:
which tables wait, why, at what mark. Each such value is a read-only snapshot (a :class:`~types.MappingProxyType`, a
``frozenset``, a tuple), and a change builds the next snapshot and assigns it in one statement. A reader holding the old
snapshot (a status, a coroutine across an await, a thread handed it) never sees it change under it. Each is per table
or per state, never per key (per-key state is Arrow), so rebuilding one is cheap.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType

__all__ = [
    "frozen",
    "with_entries",
    "with_entry",
    "without_entries",
    "without_entry",
]


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
