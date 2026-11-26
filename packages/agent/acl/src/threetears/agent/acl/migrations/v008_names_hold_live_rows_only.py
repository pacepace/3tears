"""agent-acl v008: a group's and a role's name is unique among live rows only.

Append-only L3: a delete of a group or a role marks its row gone in place (v006, ``ROW_STAMPED``), and the row keeps
its name until quiet-time cleanup removes it, so "delete X, create X" was refused by the name's unique index. Pace
2026-11-25: unique names apply to live rows only. Each name index (v001) becomes ``... WHERE <its predicate> AND NOT
gone`` (:func:`~threetears.core.data.migrations.live_unique.make_unique_indexes_live_only`, the shared helper: built
beside, then swapped in under the same name), and a re-create makes a new row with a new id.

Left whole, each for its own reason:

- ``uq_groups_group_id``, ``uq_role_assignments_assignment_id``, ``uq_namespaces_namespace_id``: ids, never reused (and
  foreign keys' targets);
- ``uq_role_assignments_namespace_grant`` / ``uq_role_assignments_all_grant``: a grant's natural key, whose revoked
  (gone) row ``RoleAssignmentCollection.ensure_grant`` revives in place;
- ``uq_namespaces_schema_name_non_workspace``: a namespace is re-created on its deterministic id, which revives its gone
  row in place (``NamespaceCollection.ensure_namespace``), schema and all.

A deployment whose own migrations create these tables (the hub) runs no agent_acl migration: it makes its own name
indexes live-only with the same helper in a migration of its own. Idempotent; forward-only.
"""

from __future__ import annotations

from typing import Final

from threetears.core.data.migrations.live_unique import make_unique_indexes_live_only
from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["names_hold_live_rows_only"]

log = get_logger(__name__)

#: the name indexes v008 makes live-only, as v001 named them (FROZEN)
_LIVE_UNIQUE_INDEXES: Final[tuple[str, ...]] = (
    "uq_groups_platform_name",
    "uq_groups_customer_name",
    "uq_roles_platform_name",
    "uq_roles_customer_name",
)


async def names_hold_live_rows_only(store: DataStore) -> None:
    """make the group and role name indexes hold live rows only (v008).

    :param store: DataStore bound to the platform schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    changed = await make_unique_indexes_live_only(store, _LIVE_UNIQUE_INDEXES)
    log.info(
        "agent_acl v008: group and role names are unique among live rows", extra={"extra_data": {"changed": changed}}
    )
