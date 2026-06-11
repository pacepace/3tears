"""L1 cache metadata + SQLiteBackend factory for the presence state layer.

channels-task-01 introduces the channels package's first L1 tier:
two Collection-backed surfaces behind the
:class:`~threetears.channels.presence.collection.PresenceCollection`.

- ``presence_connections`` — one row per live websocket connection
  (pk ``connection_id``): room membership, identity (user / customer /
  pod), and the ``date_last_heartbeat`` that drives self-heal.
- ``presence_rooms`` — one row per ``{customer}:{story}:{branch}:{file}``
  room (pk ``room_id``): the JSONB member ``connection_id`` set.

both are L1+L2 only (no L3 mirror): presence is transient by
construction. mirrors the registry's
:mod:`threetears.registry.l1_cache` pattern — one metadata object,
one factory that constructs + initializes a shared
:class:`~threetears.core.cache.sqlite.SQLiteBackend`, one table per
Collection-backed surface.
"""

from __future__ import annotations

from sqlalchemy import Column, MetaData, String, Table
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.observe import get_logger

__all__ = [
    "PRESENCE_L1_METADATA",
    "PRESENCE_L1_TABLE_NAMES",
    "create_presence_l1_backend",
    "presence_connections_table",
    "presence_rooms_table",
]

log = get_logger(__name__)

PRESENCE_L1_METADATA = MetaData()


# ---------------------------------------------------------------------------
# presence_connections (L1+L2 only; no L3 mirror)
# ---------------------------------------------------------------------------
#
# one row per live websocket connection. connection ids are opaque
# strings minted per-socket by the handler (not UUIDs); the ``String``
# primary key matches that shape. the row carries the membership
# pointer (``room_id``), the identity triple (``user_id`` /
# ``customer_id`` / ``pod_id``), and the heartbeat timestamp that the
# sweep compares against the liveness window. heartbeats refresh THIS
# row only, so the busy heartbeat path never contends on the
# room-index CAS.
presence_connections_table = Table(
    "presence_connections",
    PRESENCE_L1_METADATA,
    Column("connection_id", String(255), primary_key=True),
    Column("room_id", String(1024), nullable=False),
    Column("user_id", String(255), nullable=False),
    Column("pod_id", String(255), nullable=False),
    Column("customer_id", String(255), nullable=False),
    Column("date_last_heartbeat", TIMESTAMP(timezone=True), nullable=False),
    Column("date_created", TIMESTAMP(timezone=True), nullable=False),
    Column("date_updated", TIMESTAMP(timezone=True), nullable=False),
)


# ---------------------------------------------------------------------------
# presence_rooms (L1+L2 only; no L3 mirror)
# ---------------------------------------------------------------------------
#
# one row per room. the room id is the composite
# ``{customer}:{story}:{branch}:{file}`` string (colons and all — the
# Collection's ``l2_key`` override sanitizes them for the JetStream KV
# key grammar; the raw id round-trips through L1 + the invalidation
# envelope). ``members`` is the JSONB set of member connection ids,
# updated only on join/leave under optimistic-concurrency CAS.
presence_rooms_table = Table(
    "presence_rooms",
    PRESENCE_L1_METADATA,
    Column("room_id", String(1024), primary_key=True),
    Column("customer_id", String(255), nullable=False),
    Column("members", JSONB, nullable=False),
    Column("date_created", TIMESTAMP(timezone=True), nullable=False),
    Column("date_updated", TIMESTAMP(timezone=True), nullable=False),
)


PRESENCE_L1_TABLE_NAMES: frozenset[str] = frozenset(table.name for table in PRESENCE_L1_METADATA.tables.values())


def create_presence_l1_backend() -> SQLiteBackend:
    """create and initialize the shared SQLiteBackend for presence L1.

    constructs a named in-memory SQLite database
    (``channels_presence_l1_cache``) and mirrors every presence
    Collection table defined in :data:`PRESENCE_L1_METADATA`. the
    returned backend is ready to be passed into
    :meth:`~threetears.core.collections.registry.CollectionRegistry.configure`
    as the default L1 tier.

    :return: initialized SQLiteBackend with every presence Collection
        table
    :rtype: SQLiteBackend
    """
    backend = SQLiteBackend(db_name="channels_presence_l1_cache")
    backend.initialize(PRESENCE_L1_METADATA)
    log.info(
        "channels presence L1 cache initialized",
        extra={"extra_data": {"table_count": len(PRESENCE_L1_METADATA.tables)}},
    )
    return backend
