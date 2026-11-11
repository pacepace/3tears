"""
agent-acl v004: the indexes the access tables' other keyed reads use.

The keyed methods a platform reads the access tables by (a reverse access query, a dry run, a role or group
delete, a namespace's teardown, a tool provider's nodes) each go through an index made for it, named as the
platform's own schema names it, so a deployment that already has one creates nothing:

- ``idx_group_members_group_id`` (group_id): every member of a set of groups,
  :meth:`GroupMemberCollection.members_of_groups`;
- ``idx_role_assignments_scope_namespace`` (scope_namespace_id), partial on a namespace scope: a namespace's
  namespace-scope grants, :meth:`RoleAssignmentCollection.list_scoped_to_namespace`;
- ``idx_role_assignments_scope_type_customer`` (scope_customer_id, scope_namespace_type), partial on the
  type_customer scope: :meth:`RoleAssignmentCollection.list_of_type_customer`;
- ``idx_role_assignments_role`` (role_id, assignment_id): a role's assignments,
  :meth:`RoleAssignmentCollection.list_by_role` and ``count_by_role``;
- ``idx_role_assignments_scope_type`` (scope_type, assignment_id): every grant of one scope type (``subtree``,
  ``all``), :meth:`RoleAssignmentCollection.list_by_scope_type`;
- ``idx_namespaces_type`` (namespace_type): a type's namespaces, :meth:`NamespaceCollection.type_rows` and
  ``list_by_type``;
- ``idx_namespaces_tool_provider_pod`` (namespace_type, row_scope, (metadata->>'tool_pod_id')): a tool provider's
  nodes, by the pod that owns them or all of them, :meth:`NamespaceCollection.tool_provider_nodes`,
  ``tool_provider_ownership`` and ``tool_provider_names``.

The two new ones are built online beside the table (migrations run outside a transaction; YugabyteDB builds an
index on a populated table online), and nothing reads through them until they land. Idempotent: every index
checks first.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["create_acl_keyed_method_indexes"]

log = get_logger(__name__)

#: Every statement, in order. Each is safe to run twice.
_ACL_KEYED_METHOD_INDEXES_DDL: tuple[str, ...] = (
    "-- purpose: packages/agent/acl/src/threetears/agent/acl/collections.py::GroupMemberCollection.members_of_groups\n"
    "-- plan_test: packages/agent/acl/tests/integration/test_index_plans.py::test_members_of_groups\n"
    "-- reason: a group's members are read by group across the walk down from granting groups; the primary key\n"
    "-- leads with the group too, but the group is the table's partition, so a read of a group's every member\n"
    "-- names this index rather than a partition's whole key\n"
    "CREATE INDEX IF NOT EXISTS idx_group_members_group_id ON group_members (group_id)",
    "-- purpose: packages/agent/acl/src/threetears/agent/acl/collections.py::RoleAssignmentCollection.list_scoped_to_namespace\n"
    "-- plan_test: packages/agent/acl/tests/integration/test_index_plans.py::test_assignments_scope_namespace\n"
    "-- reason: a namespace's grants are read before it is torn down or rescoped, and its row cascades them\n"
    "CREATE INDEX IF NOT EXISTS idx_role_assignments_scope_namespace ON role_assignments (scope_namespace_id)"
    " WHERE scope_namespace_id IS NOT NULL",
    "-- purpose: packages/agent/acl/src/threetears/agent/acl/collections.py::RoleAssignmentCollection.list_of_type_customer\n"
    "-- plan_test: packages/agent/acl/tests/integration/test_index_plans.py::test_assignments_scope_type_customer\n"
    "-- reason: the type_customer grants on a customer's namespaces of one type, the reverse access query's half\n"
    "CREATE INDEX IF NOT EXISTS idx_role_assignments_scope_type_customer"
    " ON role_assignments (scope_customer_id, scope_namespace_type) WHERE scope_type = 'type_customer'",
    "-- purpose: packages/agent/acl/src/threetears/agent/acl/collections.py::RoleAssignmentCollection.list_by_role; packages/agent/acl/src/threetears/agent/acl/collections.py::RoleAssignmentCollection.count_by_role\n"
    "-- plan_test: packages/agent/acl/tests/integration/test_index_plans.py::test_assignments_role\n"
    "-- reason: a role's update preview and its delete check read every assignment of the role; none led with it\n"
    "CREATE INDEX IF NOT EXISTS idx_role_assignments_role ON role_assignments (role_id, assignment_id)",
    "-- purpose: packages/agent/acl/src/threetears/agent/acl/collections.py::RoleAssignmentCollection.list_by_scope_type\n"
    "-- plan_test: packages/agent/acl/tests/integration/test_index_plans.py::test_assignments_scope_type\n"
    "-- reason: the subtree and all grants any namespace may be reached by, the reverse access query's other half\n"
    "CREATE INDEX IF NOT EXISTS idx_role_assignments_scope_type ON role_assignments (scope_type, assignment_id)",
    "-- purpose: packages/agent/acl/src/threetears/agent/acl/collections.py::NamespaceCollection.type_rows; packages/agent/acl/src/threetears/agent/acl/collections.py::NamespaceCollection.list_by_type;"
    " packages/agent/acl/src/threetears/agent/acl/collections.py::NamespaceCollection.tool_provider_ownership; packages/agent/acl/src/threetears/agent/acl/collections.py::NamespaceCollection.tool_provider_names\n"
    "-- plan_test: packages/agent/acl/tests/integration/test_index_plans.py::test_namespaces_type\n"
    "-- reason: a type's namespaces (every model, every tool provider node) are read by type\n"
    "CREATE INDEX IF NOT EXISTS idx_namespaces_type ON namespaces (namespace_type)",
    "-- purpose: packages/agent/acl/src/threetears/agent/acl/collections.py::NamespaceCollection.tool_provider_nodes; packages/agent/acl/src/threetears/agent/acl/collections.py::NamespaceCollection.tool_provider_ownership;"
    " packages/agent/acl/src/threetears/agent/acl/collections.py::NamespaceCollection.tool_provider_names\n"
    "-- plan_test: packages/agent/acl/tests/integration/test_index_plans.py::test_namespaces_tool_provider_pod\n"
    "-- reason: every tool registration asks for a pod's provider nodes; the pod is a field of the row's metadata\n"
    "CREATE INDEX IF NOT EXISTS idx_namespaces_tool_provider_pod"
    " ON namespaces (namespace_type, row_scope, (metadata->>'tool_pod_id'))",
)


async def create_acl_keyed_method_indexes(store: DataStore) -> None:
    """Create the indexes the access tables' other keyed reads use.

    :param store: the migration's data store
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    for statement in _ACL_KEYED_METHOD_INDEXES_DDL:
        await store.execute(statement)
    log.info("agent-acl v004: keyed method indexes created")
