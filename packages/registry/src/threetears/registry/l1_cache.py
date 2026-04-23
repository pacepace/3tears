"""L1 cache metadata + SQLiteBackend factory for the registry process.

namespace-task-01 phase 8.5l-3 introduces the first L1 tier inside the
registry process. prior to 8.5l-3 the registry held pod liveness in
an in-memory dict (``HeartbeatMonitor._pods``) with an optional raw
``SQLiteBackend`` mirror bolted on. 8.5l-3 retires that shape: the
canonical surface is now :class:`HeartbeatCollection` backed by L1
(this SQLite tier) + L2 (NATS KV cross-registry coherence); no L3
durability, as heartbeats are transient state that rebuilds within
seconds of a pod restart.

mirrors the hub (``aibots.hub.common.l1_cache``) and agent pod
(``aibots_agents.runtime.l1_cache``) patterns: one metadata object per
process, one factory that constructs + initializes a shared
:class:`SQLiteBackend`, one table per Collection-backed surface.
"""

from __future__ import annotations

from sqlalchemy import (
    Column,
    Integer,
    MetaData,
    String,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.observe import get_logger

__all__ = [
    "REGISTRY_L1_METADATA",
    "REGISTRY_L1_TABLE_NAMES",
    "create_registry_l1_backend",
    "pod_heartbeats_table",
]

log = get_logger(__name__)

REGISTRY_L1_METADATA = MetaData()


# ---------------------------------------------------------------------------
# pod_heartbeats (L1+L2 only; no L3 mirror)
# ---------------------------------------------------------------------------
#
# namespace-task-01 phase 8.5l-3: ``pod_heartbeats`` is the
# registry's first Collection-backed surface. each row captures one
# tool pod's liveness state: last heartbeat timestamp, status label,
# tools list, consecutive-miss counter. pod ids are opaque strings
# supplied by tool pods (not UUIDs); the ``String`` primary-key column
# matches the :class:`HeartbeatMessage.pod_id` wire shape.
#
# no L3 migration carries this table -- heartbeats are transient by
# construction. a pod that restarts re-emits its heartbeat within
# seconds and the subscriber repopulates both L1 and L2. durable
# historical ``pod_heartbeats`` rows in YugabyteDB would be dead
# weight: they are never queried outside the liveness window.
pod_heartbeats_table = Table(
    "pod_heartbeats",
    REGISTRY_L1_METADATA,
    Column("pod_id", String(255), primary_key=True),
    Column("date_last_heartbeat", TIMESTAMP, nullable=False),
    Column("tools", JSONB, nullable=False),
    Column("tools_count", Integer, nullable=False),
    Column("status", String(32), nullable=False),
    Column("consecutive_misses", Integer, nullable=False),
    Column("date_created", TIMESTAMP, nullable=False),
    Column("date_updated", TIMESTAMP, nullable=False),
)


REGISTRY_L1_TABLE_NAMES: frozenset[str] = frozenset(
    table.name for table in REGISTRY_L1_METADATA.tables.values()
)


def create_registry_l1_backend() -> SQLiteBackend:
    """create and initialize shared SQLiteBackend for the registry process.

    constructs a named in-memory SQLite database (``registry_l1_cache``)
    and mirrors every registry-process Collection table defined in
    :data:`REGISTRY_L1_METADATA`. the returned backend is ready to be
    passed into :meth:`CollectionRegistry.configure` as the default
    L1 tier.

    :return: initialized SQLiteBackend with every registry-process
        Collection table
    :rtype: SQLiteBackend
    """
    backend = SQLiteBackend(db_name="registry_l1_cache")
    backend.initialize(REGISTRY_L1_METADATA)
    log.info(
        "registry L1 cache initialized",
        extra={"extra_data": {"table_count": len(REGISTRY_L1_METADATA.tables)}},
    )
    return backend
