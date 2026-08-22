"""Data layer exceptions."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ConcurrentModificationError",
    "CorruptCacheEntry",
    "DataLayerUnavailableError",
    "InvalidL2ScopeError",
    "L2ScopeError",
    "L2ScopeNotConfiguredError",
]


class ConcurrentModificationError(Exception):
    """Raised when optimistic locking detects a concurrent modification."""

    def __init__(self, table_name: str, entity_id: Any, expected_timestamp: Any) -> None:
        self.table_name = table_name
        self.entity_id = entity_id
        self.expected_timestamp = expected_timestamp
        super().__init__(
            f"Concurrent modification on {table_name}:{entity_id} (expected date_updated={expected_timestamp})"
        )


class DataLayerUnavailableError(Exception):
    """Raised when persistence layer is unavailable."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class L2ScopeError(RuntimeError):
    """Base for the two ways a registry's L2 key scope can be wrong.

    **Deliberately not a subclass of** :class:`threetears.nats.errors.KvError`, and that is the
    whole reason it is its own hierarchy. Four of the five :meth:`BaseCollection.l2_key` call
    sites (``_get_from_l2`` / ``_save_to_l2`` / ``_delete_from_l2`` / ``delete_l2_entry``) sit
    inside ``except KvError`` handlers that degrade to a warning, so a ``KvError`` raised here
    would be swallowed and the fleet would run with L2 silently off -- the exact degradation
    the fail-loud decision exists to prevent. The fifth (``l2_cas_mutate``) deliberately does
    NOT degrade, because L2 is the source of truth there, so a ``KvError`` would additionally
    be inconsistent between the five: swallowed at four sites and propagating at the one where
    a missing scope matters most. A distinct type behaves identically at all five.
    """


class L2ScopeNotConfiguredError(L2ScopeError):
    """Raised when a registry holds an L2 client but no ``kv_key_scope``.

    The primary raise site is :meth:`CollectionRegistry.configure`, evaluated over merged
    registry state -- wiring time, where the process can still fail its startup. The backstop
    raise in :meth:`BaseCollection.l2_key` covers the ``nats_client=``-direct construction path,
    which never calls ``configure`` at all.
    """


class InvalidL2ScopeError(L2ScopeError):
    """Raised when a supplied ``kv_key_scope`` falls outside the scope grammar.

    The scope is the leading NATS subject TOKEN of ``$KV.{bucket}.{scope}.{table}.{body}``, so
    the grammar it is checked against (``threetears.nats.KV_KEY_SCOPE_GRAMMAR``) is stricter
    than the JetStream KV key grammar: a scope carrying ``.`` renders two tokens and silently
    stops matching the per-principal ``$KV.{bucket}.{scope}.>`` grant.
    """


class CorruptCacheEntry(Exception):
    """Raised when an L2 value cannot be decoded back into the types it claims to hold.

    Deliberately NOT a data-layer outage and not a caller error. L2 is a cache: an entry that
    will not decode is a corrupt cache entry, and the correct response to one is to stop
    serving it, not to fail the read. Every read path in :class:`BaseCollection` catches this
    and falls through to L3, which is authoritative.

    That is the whole reason it exists as its own type. The alternatives both looked reasonable
    and were both worse: letting the underlying ``ValueError`` propagate turns one poisoned key
    into a failed read that L1 or L3 could have served, and swallowing it to return the raw
    undecoded value hands the caller a string where it declared a ``datetime``, which fails far
    from here and usually at the database border.
    """

    def __init__(self, table_name: str, column: str, value: Any) -> None:
        self.table_name = table_name
        self.column = column
        self.value = value
        super().__init__(f"{table_name}.{column} holds a value that will not decode: {value!r}")
