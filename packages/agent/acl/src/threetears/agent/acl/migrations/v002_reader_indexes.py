"""
agent-acl v002: the indexes the evaluator's keyed reads use.

Each read by something other than a table's primary key goes through an index made for it, named as the
platform's own schema names it, so a deployment that already has one creates nothing:

- ``idx_group_members_reverse`` (member_type, member_id): the memberships naming a user, agent or group,
  :meth:`GroupMemberCollection._load_naming`, on every access evaluation;
- ``idx_role_assignments_group_scope`` (group_id, scope_type): every assignment a set of groups holds, the
  evaluator's grant load and a group delete's cascade;
- ``idx_namespaces_name`` (name): :meth:`NamespaceCollection.get_by_name`. Not unique here: a namespace's name
  is not unique in this package (v001).

Idempotent: every index checks first.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["create_acl_reader_indexes"]

log = get_logger(__name__)

#: Every statement, in order. Each is safe to run twice.
_ACL_READER_INDEXES_DDL: tuple[str, ...] = (
    "-- purpose: packages/agent/acl/src/threetears/agent/acl/collections.py::GroupMemberCollection._load_naming;"
    " packages/agent/acl/src/threetears/agent/acl/query_visibility.py::CALLER_VISIBLE_CUSTOMERS_SQL; packages/agent/acl/src/threetears/agent/acl/query_visibility.py::CALLER_VISIBLE_CUSTOMERS_OF_TYPE_SQL;"
    " src/aibots/hub/rbac/collections.py::HubGroupMemberCollection.remove_member_everywhere;"
    " src/aibots/hub/rbac/collections.py::HubGroupMemberCollection.repoint_member;"
    " src/aibots/hub/rbac/collections.py::HubGroupCollection.list_memberships_visible_to_user\n"
    "-- plan_test: packages/agent/acl/tests/integration/test_index_plans.py::test_members_reverse\n"
    "CREATE INDEX IF NOT EXISTS idx_group_members_reverse ON group_members (member_type, member_id)",
    "-- purpose: packages/agent/acl/src/threetears/agent/acl/collections.py::GroupCollection.read_cascade; packages/agent/acl/src/threetears/agent/acl/collections.py::RoleAssignmentCollection.load_for_groups; packages/agent/acl/src/threetears/agent/acl/collections.py::RoleAssignmentCollection.ensure_group_role_assignment;"
    " packages/agent/acl/src/threetears/agent/acl/collections.py::GRANT_WINNER_OF_GROUP_SQL; packages/agent/acl/src/threetears/agent/acl/collections.py::GRANTS_OF_GROUP_SCOPE_SQL;"
    " packages/agent/acl/src/threetears/agent/acl/collections.py::DELETE_GRANTS_OF_GROUP_SCOPE_SQL;"
    " packages/agent/acl/src/threetears/agent/acl/query_visibility.py::CALLER_VISIBLE_CUSTOMERS_SQL; packages/agent/acl/src/threetears/agent/acl/query_visibility.py::CALLER_VISIBLE_CUSTOMERS_OF_TYPE_SQL;"
    " packages/agent/acl/src/threetears/agent/acl/collections.py::RoleAssignmentCollection.list_by_group;"
    " packages/agent/acl/src/threetears/agent/acl/collections.py::RoleAssignmentCollection.count_by_group\n"
    "-- plan_test: packages/agent/acl/tests/integration/test_index_plans.py::test_assignments_group_scope\n"
    "CREATE INDEX IF NOT EXISTS idx_role_assignments_group_scope ON role_assignments (group_id, scope_type)",
    "-- purpose: packages/agent/acl/src/threetears/agent/acl/collections.py::NamespaceCollection.get_by_name;"
    " packages/agent/acl/src/threetears/agent/acl/collections.py::NAMESPACE_IDS_UNDER_NAME_SQL\n"
    "-- plan_test: packages/agent/acl/tests/integration/test_index_plans.py::test_namespaces_name\n"
    "-- reason: a namespace by its name, every grant and tool lookup that names one\n"
    "CREATE INDEX IF NOT EXISTS idx_namespaces_name ON namespaces (name)",
)


async def create_acl_reader_indexes(store: DataStore) -> None:
    """Create the indexes the evaluator's keyed reads use.

    :param store: the migration's data store
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    for statement in _ACL_READER_INDEXES_DDL:
        await store.execute(statement)
    log.info("agent-acl v002: reader indexes created")
