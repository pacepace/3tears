"""The one rbac L1 mirror, generated from the canonical Collection schemas.

Every process that evaluates an rbac decision locally -- the hub, the registry, an agent pod --
holds an L1 SQLite mirror of the five tables the evaluator reads (``namespaces`` plus
``groups`` / ``group_members`` / ``roles`` / ``role_assignments``). Each of the three used to
declare that mirror by hand, and two of the three had fallen five columns behind
:attr:`NamespaceCollection.schema` (``tool_eligible``, ``skill_eligible``, ``face_api``,
``face_mcp``, ``face_platform_tool``). The failure is not a cache miss:
:meth:`BaseCollection.write_to_cache_sync` raises ``sqlite3.OperationalError: table namespaces
has no column named tool_eligible`` on the write, and the caller sees an authorize error.

So the mirror is EMITTED, never retyped. :meth:`TableSchema.to_sqlalchemy_table` already
performs the conversion and every rbac Collection is a :class:`SchemaBackedCollection`, so a
column added to a canonical schema reaches all three processes with no second edit. Collapsing
three hand-maintained copies into one hand-maintained copy would only have moved the drift a
column later.

Callers register onto their OWN ``MetaData`` (each process mirrors a different table set
beside the rbac five) and pass that metadata to :meth:`SQLiteBackend.initialize`.
"""

from __future__ import annotations

from typing import Any, cast

from sqlalchemy import MetaData, Table

from threetears.core.collections.schema_backed import SchemaBackedCollection

from threetears.agent.acl.collections import (
    GroupCollection,
    GroupMemberCollection,
    NamespaceCollection,
    RoleAssignmentCollection,
    RoleCollection,
)

__all__ = [
    "RBAC_L1_COLLECTIONS",
    "RBAC_L1_TABLE_NAMES",
    "register_rbac_l1_tables",
]


#: the canonical Collections whose schemas the mirror is generated from. the unified evaluator
#: reads all five on the authorize hot path (membership walks ``group_members`` -> ``groups``,
#: the grant walk adds ``role_assignments`` -> ``roles``, and the namespace lookup is the
#: entry point), so a process that mirrors four of them still round-trips to L3 on every call.
RBAC_L1_COLLECTIONS: tuple[type[SchemaBackedCollection[Any]], ...] = (
    NamespaceCollection,
    GroupCollection,
    GroupMemberCollection,
    RoleCollection,
    RoleAssignmentCollection,
)


#: table names the mirror covers. derived from the schemas rather than written out, for the
#: same reason the tables themselves are.
RBAC_L1_TABLE_NAMES: frozenset[str] = frozenset(cls.schema.name for cls in RBAC_L1_COLLECTIONS)


def register_rbac_l1_tables(metadata: MetaData) -> dict[str, Table]:
    """register the five rbac mirror tables on ``metadata`` and return them by name.

    idempotent, because :meth:`TableSchema.to_sqlalchemy_table` is: a table already present on
    ``metadata`` comes back unchanged instead of raising. that matters because the hub and the
    agent pod register onto a module-level ``MetaData`` that other tables share.

    :param metadata: SQLAlchemy metadata to attach the tables to; typically the caller's
        process-wide L1 metadata, passed on to :meth:`SQLiteBackend.initialize`
    :ptype metadata: MetaData
    :return: mapping of table name to registered :class:`Table`
    :rtype: dict[str, Table]
    :raises KeyError: if a canonical schema carries a column type with no SQLAlchemy mapping
    """
    return {cls.schema.name: cast(Table, cls.schema.to_sqlalchemy_table(metadata)) for cls in RBAC_L1_COLLECTIONS}
