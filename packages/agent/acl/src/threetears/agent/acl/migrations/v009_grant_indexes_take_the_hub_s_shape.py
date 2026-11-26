"""agent-acl v009: a role's grants and a scope type's grants read the hub's range indexes; v004's two are dropped.

v004 made ``idx_role_assignments_role`` (role_id, assignment_id) and ``idx_role_assignments_scope_type`` (scope_type,
assignment_id). The hub, which runs no agent_acl migration, made the same two in its own chain, then replaced them with
its v163 page indexes and drops them in its held contract step v1001. 3tears hints each read of the table to the index
its schema declares, by name, so on the hub a read hinted to one of v004's names named nothing (pg_hint_plan ignores a
missing index). The hub's shape is the better one, so 3tears takes it, under the hub's names:

- ``role_assignments_page_by_scope_type`` (scope_type ASC, assignment_id ASC) replaces ``idx_role_assignments_scope_type``:
  the same columns, range-sharded on YugabyteDB, where v004's is hashed on its first column, so a page across every
  grant by scope type reads its limit from the cursor rather than every grant (the hub's v163);
- ``role_assignments_page_by_role_scope_type`` (role_id, scope_type ASC, assignment_id ASC) replaces
  ``idx_role_assignments_role``: a role's grants (:meth:`RoleAssignmentCollection.list_by_role`, ``count_by_role``,
  a role delete's restricted count) read by its lead as before, and a role's grants of one scope type besides.

The new indexes are built first, beside the table (migrations run outside a transaction; YugabyteDB builds an index on a
populated table online), so a role's and a scope type's reads always have one, then v004's two are dropped. Not held as
the hub's v1001 is: v004 and this step ship in the same release (3tears v0.95.0), so no running pod plans through
v004's names. On a hub schema the hub's own v163 and v1001 have already done the same. Every statement is guarded, so
re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["grant_indexes_take_the_hub_s_shape"]

log = get_logger(__name__)

#: Every statement, in order: each new index before the one it replaces is dropped. Each is safe to run twice. FROZEN
_GRANT_INDEXES_DDL: tuple[str, ...] = (
    "-- purpose: packages/agent/acl/src/threetears/agent/acl/collections.py::RoleAssignmentCollection.list_by_scope_type\n"
    "-- plan_test: packages/agent/acl/tests/integration/test_index_plans.py::test_assignments_scope_type\n"
    "-- reason: the subtree and all grants any namespace may be reached by, the reverse access query's other half;"
    " range-sharded on YugabyteDB, the hub's v163 index of the same name\n"
    "CREATE INDEX IF NOT EXISTS role_assignments_page_by_scope_type"
    " ON role_assignments (scope_type ASC, assignment_id ASC)",
    "-- purpose: packages/agent/acl/src/threetears/agent/acl/collections.py::RoleAssignmentCollection.list_by_role;"
    " packages/agent/acl/src/threetears/agent/acl/collections.py::RoleAssignmentCollection.count_by_role\n"
    "-- plan_test: packages/agent/acl/tests/integration/test_index_plans.py::test_assignments_role\n"
    "-- reason: a role's update preview and its delete check read every assignment of the role; the hub's v163"
    " index of the same name\n"
    "CREATE INDEX IF NOT EXISTS role_assignments_page_by_role_scope_type"
    " ON role_assignments (role_id, scope_type ASC, assignment_id ASC)",
    "DROP INDEX IF EXISTS idx_role_assignments_scope_type",
    "DROP INDEX IF EXISTS idx_role_assignments_role",
)


async def grant_indexes_take_the_hub_s_shape(store: DataStore) -> None:
    """build the hub's scope-type and role grant indexes, then drop v004's two they replace (v009).

    :param store: the migration's data store
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    for statement in _GRANT_INDEXES_DDL:
        await store.execute(statement)
    log.info("agent-acl v009: the role and scope-type grant indexes take the hub's shape and names")
