"""Data layer exceptions."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ConcurrentModificationError",
    "CorruptCacheEntry",
    "DataLayerUnavailableError",
    "DataVersionNotReadyError",
    "DataVersionSupersededError",
    "GenerationUnavailableError",
    "InvalidL2ScopeError",
    "L2EpochRegressedError",
    "L2ScopeError",
    "L2ScopeNotConfiguredError",
]


class GenerationUnavailableError(Exception):
    """Raised when a table's write generation cannot be read or advanced.

    A reader treats it as "cannot trust a cached absence": it neither serves a negative-cache
    marker nor records one, and asks L3. A writer surfaces it, because a generation it failed to
    advance leaves older markers valid over the write it just committed.
    """


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


class DataVersionSupersededError(DataLayerUnavailableError):
    """Raised when the broker refuses a pod whose data version is older than its space's target.

    FATAL for the pod. Its space has been, or is being, upgraded to a later version of its table
    list, and a pod on the older list must not touch the new tables; a pod on the new version
    replaces it. The broker keeps refusing every request this pod sends, so there is nothing to
    retry. The backend hands this error to its ``on_superseded`` callback once before raising it,
    so the pod runtime -- not whichever caller happened to issue the refused query -- owns the
    exit.

    A subclass of :class:`DataLayerUnavailableError` because, to code that only needs to know the
    data layer is unusable, that is exactly what it is: an existing handler that treats an outage
    as an outage stays correct, while code that must tell the two apart catches this type.
    """


class DataVersionNotReadyError(DataLayerUnavailableError):
    """Raised when the broker refuses a pod whose data version IS its space's target, mid-upgrade.

    TRANSIENT. The pod is at the right version, but the upgrade that brings its space there has
    not finished, so the tables it expects may not exist yet. The pod waits for the upgrade's
    all-clear and retries; it must not exit, which is the whole difference from
    :class:`DataVersionSupersededError` and why the two are separate types.

    A subclass of :class:`DataLayerUnavailableError` for the reason given there: the data layer
    is, for now, unavailable to this pod.
    """


class L2EpochRegressedError(RuntimeError):
    """Raised when L3 holds a row ordered after anything the current L2 bucket can write.

    A compare-and-swap write is stored in L3 with the creation time of the L2 stream it won in,
    and L3 refuses any write ordered at or before what it holds. A stream created EARLIER than an
    order L3 already holds -- the broker's clock moved backwards across a restart -- would have
    every write it accepts refused in L3 as superseded while the caller was told it succeeded.
    :meth:`BaseCollection.l2_cas_mutate` raises this instead, before touching L2, so the loss is
    loud rather than silent.

    :ivar table_name: the collection's table
    :ivar entity_id: the row whose stored order is ahead
    :ivar stored_epoch: the epoch L3 holds
    :ivar bucket_epoch: the current bucket's creation time
    """

    def __init__(self, table_name: str, entity_id: Any, stored_epoch: Any, bucket_epoch: Any) -> None:
        self.table_name = table_name
        self.entity_id = entity_id
        self.stored_epoch = stored_epoch
        self.bucket_epoch = bucket_epoch
        super().__init__(
            f"{table_name}:{entity_id} is stored in L3 under L2 epoch {stored_epoch}, later than the current "
            f"bucket's creation time {bucket_epoch}; every write would be refused as superseded. The broker's "
            f"clock moved backwards across a restart -- correct it, or recreate the bucket once it is ahead"
        )


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
