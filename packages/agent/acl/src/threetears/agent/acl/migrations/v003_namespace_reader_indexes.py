"""
agent-acl v003: the indexes the namespace lookups by owner, customer, owner namespace and schema use.

Each of these reads asked a column no index of this package covered, so it read every namespace row:

- ``idx_namespaces_owner_agent_id`` (owner_agent_id): an agent's own namespace,
  :meth:`NamespaceCollection.get_by_agent_id`, and an owned one by its triple,
  :meth:`NamespaceCollection.get_by_owner_and_customer`; the hub's discovery reads an agent's own namespaces by it
  too (``HubNamespaceCollection.list_granted_candidates``, its plan test the hub's
  ``test_namespace_index_plans.py::test_discovery_reads_its_candidates_from_the_agents_grants``). The platform's
  schema already names it so, so a deployment that has it creates nothing;
- ``idx_namespaces_customer_type`` (customer_id, namespace_type): a customer's namespaces of a type
  (:meth:`NamespaceCollection.find_by_type_and_customer`, :meth:`NamespaceCollection.list_ids_by_customer_and_type`)
  and an ownerless one by its triple, a platform row by its NULL customer;
- ``idx_namespaces_owner_namespace`` (owner_namespace): what a namespace owns, a teardown's walk,
  :meth:`NamespaceCollection.list_owned_by`;
- ``idx_namespaces_schema_name`` (schema_name): whether any row still names a schema,
  :meth:`NamespaceCollection.schema_in_use`. The unique ``uq_namespaces_schema_name_non_workspace`` cannot serve
  it: workspace rows share their agent's schema and are left out of it on purpose, and widening it would refuse
  them;
- ``idx_namespaces_name_c`` (name COLLATE "C"): the names beneath a node, a subtree grant's namespaces,
  :meth:`NamespaceCollection.list_ids_under_name`, read as the byte range ``[node + ".", node + "/")``; ``ASC``, so
  YugabyteDB range-shards it (a hashed first column cannot serve a range). New on the platform's schema too.

Idempotent: every index checks first.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["create_acl_namespace_reader_indexes"]

log = get_logger(__name__)

#: Every statement, in order. Each is safe to run twice.
_ACL_NAMESPACE_INDEXES_DDL: tuple[str, ...] = (
    "-- purpose: packages/agent/acl/src/threetears/agent/acl/collections.py::NamespaceCollection.get_by_agent_id;"
    " packages/agent/acl/src/threetears/agent/acl/collections.py::NamespaceCollection.get_by_owner_and_customer;"
    " src/aibots/hub/broker/namespaces.py::HubNamespaceCollection.list_granted_candidates\n"
    "-- plan_test: packages/agent/acl/tests/integration/test_index_plans.py::test_namespaces_owner_agent_id\n"
    "-- reason: an agent's own namespaces by its id, on every agent's provisioning and access check\n"
    "CREATE INDEX IF NOT EXISTS idx_namespaces_owner_agent_id ON namespaces (owner_agent_id)",
    "-- purpose: packages/agent/acl/src/threetears/agent/acl/collections.py::NamespaceCollection.find_by_type_and_customer;"
    " packages/agent/acl/src/threetears/agent/acl/collections.py::NamespaceCollection.list_ids_by_customer_and_type;"
    " packages/agent/acl/src/threetears/agent/acl/collections.py::NamespaceCollection.get_by_owner_and_customer;"
    " packages/agent/acl/src/threetears/agent/acl/collections.py::NamespaceCollection.get_by_owner_and_customer\n"
    "-- plan_test: packages/agent/acl/tests/integration/test_index_plans.py::test_namespaces_customer_type\n"
    "-- reason: a customer's namespaces of one type, the set a type_customer grant covers\n"
    "CREATE INDEX IF NOT EXISTS idx_namespaces_customer_type ON namespaces (customer_id, namespace_type)",
    "-- purpose: packages/agent/acl/src/threetears/agent/acl/collections.py::NamespaceCollection.list_owned_by\n"
    "-- plan_test: packages/agent/acl/tests/integration/test_index_plans.py::test_namespaces_owner_namespace\n"
    "-- reason: what one namespace owns, each step of a namespace teardown's walk\n"
    "CREATE INDEX IF NOT EXISTS idx_namespaces_owner_namespace ON namespaces (owner_namespace)",
    "-- purpose: packages/agent/acl/src/threetears/agent/acl/collections.py::NamespaceCollection.schema_in_use\n"
    "-- plan_test: packages/agent/acl/tests/integration/test_index_plans.py::test_namespaces_schema_name\n"
    "-- reason: whether any row of any type still names a schema, before it is dropped\n"
    "CREATE INDEX IF NOT EXISTS idx_namespaces_schema_name ON namespaces (schema_name)",
    "-- purpose: packages/agent/acl/src/threetears/agent/acl/collections.py::NAMESPACE_IDS_UNDER_NAME_SQL\n"
    "-- plan_test: packages/agent/acl/tests/integration/test_index_plans.py::test_namespaces_under_name\n"
    "-- reason: a subtree grant's namespaces, the byte range of names beneath its node\n"
    'CREATE INDEX IF NOT EXISTS idx_namespaces_name_c ON namespaces ((name COLLATE "C") ASC)',
)


async def create_acl_namespace_reader_indexes(store: DataStore) -> None:
    """Create the indexes the namespace lookups by owner, customer, owner namespace and schema use.

    :param store: the migration's data store
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    for statement in _ACL_NAMESPACE_INDEXES_DDL:
        await store.execute(statement)
    log.info("agent-acl v003: namespace reader indexes created")
