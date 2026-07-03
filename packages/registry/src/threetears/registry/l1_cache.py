"""L1 cache metadata + SQLiteBackend factory for the registry process.

namespace-task-01 phase 8.5l-3 introduces the first L1 tier inside the
registry process. prior to 8.5l-3 the registry held pod liveness in
an in-memory dict (``HeartbeatMonitor._pods``) with an optional raw
``SQLiteBackend`` mirror bolted on. 8.5l-3 retires that shape: the
canonical surface is now :class:`HeartbeatCollection` backed by L1
(this SQLite tier) + L2 (NATS KV cross-registry coherence); no L3
durability, as heartbeats are transient state that rebuilds within
seconds of a pod restart.

the registry-rbac task adds the five rbac mirror tables (namespaces +
groups / group_members / roles / role_assignments) so the
:class:`RbacEvaluatorAuthorizer` wired into the standalone registry
can warm an in-process L1 view of the rbac surface. column shapes
match :mod:`threetears.agent.acl` collection schemas one-for-one --
the agent SDK's
:data:`3tears_agents.runtime.l1_cache.AGENT_L1_METADATA` is the
sibling definition. duplicating the column lists here keeps the
3tears registry package free of an 3tears_agents dependency (the
registry must stay below the 3tears stack in the import graph).

mirrors the hub (``3tears.hub.common.l1_cache``) and agent pod
(``3tears_agents.runtime.l1_cache``) patterns: one metadata object per
process, one factory that constructs + initializes a shared
:class:`SQLiteBackend`, one table per Collection-backed surface.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    Integer,
    MetaData,
    String,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.observe import get_logger

__all__ = [
    "REGISTRY_L1_METADATA",
    "REGISTRY_L1_TABLE_NAMES",
    "create_registry_l1_backend",
    "group_members_table",
    "groups_table",
    "namespaces_table",
    "pod_heartbeats_table",
    "role_assignments_table",
    "roles_table",
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
    Column("date_last_heartbeat", TIMESTAMP(timezone=True), nullable=False),
    Column("tools", JSONB, nullable=False),
    Column("tools_count", Integer, nullable=False),
    Column("status", String(32), nullable=False),
    Column("consecutive_misses", Integer, nullable=False),
    Column("date_created", TIMESTAMP(timezone=True), nullable=False),
    Column("date_updated", TIMESTAMP(timezone=True), nullable=False),
)


# ---------------------------------------------------------------------------
# rbac mirror tables (column shapes copied verbatim from
# 3tears_agents.runtime.l1_cache; duplicated to keep the 3tears
# registry package free of any 3tears_agents dependency)
# ---------------------------------------------------------------------------
#
# the registry's :class:`RbacEvaluatorAuthorizer` evaluates two-sided
# tool-call decisions against a NATS-proxy NamespaceCollection (read
# of ``platform.namespaces``) plus four rbac metadata Collections
# (``groups`` / ``group_members`` / ``roles`` / ``role_assignments``).
# the canonical schemas live on the
# :mod:`threetears.agent.acl` collection classes; the L1 mirror here
# matches them column-for-column so reads round-trip through the
# in-process SQLite tier without ``column does not exist`` errors.
# v0.8.0 shard 04.6 PK column rename: namespaces.id -> namespace_id,
# groups.id -> group_id, roles.id -> role_id,
# role_assignments.id -> assignment_id. group_members.id was NOT
# renamed (the canonical primary_key is ``(group_id, id)``).
namespaces_table = Table(
    "namespaces",
    REGISTRY_L1_METADATA,
    Column("row_scope", String(20), primary_key=True, nullable=False),
    Column("namespace_id", UUID, primary_key=True),
    Column("name", String(255), nullable=False),
    Column("namespace_type", String(20), nullable=False),
    Column("owner_agent_id", UUID, nullable=True),
    Column("schema_name", String(100), nullable=True),
    Column("customer_id", UUID, nullable=True),
    Column("metadata", JSONB, nullable=True),
    Column("date_created", TIMESTAMP(timezone=True), nullable=False),
    Column("date_updated", TIMESTAMP(timezone=True), nullable=False),
)

groups_table = Table(
    "groups",
    REGISTRY_L1_METADATA,
    Column("row_scope", String(20), primary_key=True, nullable=False),
    Column("group_id", UUID, primary_key=True),
    Column("customer_id", UUID, nullable=True),
    Column("name", String(255), nullable=False),
    Column("description", Text, nullable=True),
    Column("date_created", TIMESTAMP(timezone=True), nullable=False),
    Column("date_updated", TIMESTAMP(timezone=True), nullable=False),
)

group_members_table = Table(
    "group_members",
    REGISTRY_L1_METADATA,
    # group_members.id was NOT renamed in v0.8.0.
    Column("id", UUID, primary_key=True),
    Column("group_id", UUID, primary_key=True, nullable=False),
    Column("member_type", String(10), nullable=False),
    Column("member_id", UUID, nullable=False),
    Column("customer_id", UUID, nullable=True),
    Column("date_added", TIMESTAMP(timezone=True), nullable=False),
    CheckConstraint("member_type IN ('user', 'agent')"),
)

roles_table = Table(
    "roles",
    REGISTRY_L1_METADATA,
    Column("role_id", UUID, primary_key=True),
    Column("name", String(255), nullable=False),
    Column("description", Text, nullable=False),
    Column("permissions", JSONB, nullable=False),
    Column("is_builtin", Boolean, nullable=False),
    Column("date_created", TIMESTAMP(timezone=True), nullable=False),
    Column("date_updated", TIMESTAMP(timezone=True), nullable=False),
)

role_assignments_table = Table(
    "role_assignments",
    REGISTRY_L1_METADATA,
    Column("row_scope", String(20), primary_key=True, nullable=False),
    Column("assignment_id", UUID, primary_key=True),
    Column("role_id", UUID, nullable=False),
    Column("group_id", UUID, nullable=False),
    Column("scope_type", String(16), nullable=False),
    Column("scope_namespace_id", UUID, nullable=True),
    Column("scope_namespace_type", String(255), nullable=True),
    Column("scope_customer_id", UUID, nullable=True),
    Column("granted_by", UUID, nullable=True),
    Column("date_granted", TIMESTAMP(timezone=True), nullable=False),
    Column("managed_by", String(64), nullable=True),
)


REGISTRY_L1_TABLE_NAMES: frozenset[str] = frozenset(table.name for table in REGISTRY_L1_METADATA.tables.values())


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
