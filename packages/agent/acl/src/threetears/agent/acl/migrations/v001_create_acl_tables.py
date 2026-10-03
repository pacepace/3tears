"""
agent-acl v001: the five rbac tables the evaluator reads.

``namespaces``, ``groups``, ``group_members``, ``roles`` and
``role_assignments``, in the shape the Collections declare
(:mod:`threetears.agent.acl.collections`) and with the uniqueness the
evaluator and the idempotent writers rely on:

- a namespace's ``name`` is unique. ``owner_namespace`` is a name with no
  foreign key: an agent's own namespace row exists only where a registry
  registers agents, while the memory provisioner and identity name it
  everywhere;
  only workspace namespaces share a ``schema_name``; a platform namespace is
  one of the platform types;
- a group's ``name`` is unique across platform rows and per customer, so
  :meth:`GroupCollection.get_by_name` returns at most one row;
- a role's ``name`` is unique the same way, by ownership;
- a grant's natural key is unique -- ``(group_id, role_id,
  scope_namespace_id, managed_by)`` for a namespace grant, ``(group_id,
  role_id, managed_by)`` for an ``all`` grant -- so
  :meth:`RoleAssignmentCollection.ensure_group_role_assignment` is
  race-safe (its ``ON CONFLICT DO NOTHING`` absorbs the loser);
- ``managed_by`` is never empty: a row nobody stamped is ``manual``.

Until this package an application declared these tables itself; one that
did adopts this version by stamping it once its tables match.

The tool columns of ``namespaces`` (``tool_eligible``, the ``face_*``
flags) are ``agent_tools_platform``'s, which ALTERs this table and so
runs after this package.

Idempotent: every table, constraint and index checks first.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["ACL_TABLES_DDL", "create_acl_tables"]

log = get_logger(__name__)

#: Every statement, in order. Each is safe to run twice.
ACL_TABLES_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS namespaces (
        row_scope VARCHAR(20) NOT NULL,
        namespace_id UUID NOT NULL,
        name VARCHAR(255) NOT NULL,
        namespace_type VARCHAR(20) NOT NULL,
        owner_agent_id UUID,
        customer_id UUID,
        schema_name VARCHAR(100),
        metadata JSONB DEFAULT '{}'::jsonb,
        owner_namespace VARCHAR(255),
        date_created TIMESTAMPTZ NOT NULL,
        date_updated TIMESTAMPTZ NOT NULL,
        CONSTRAINT namespaces_pkey PRIMARY KEY (row_scope, namespace_id),
        CONSTRAINT uq_namespaces_namespace_id UNIQUE (namespace_id),
        CONSTRAINT namespaces_row_scope_check CHECK (row_scope IN ('platform', 'customer')),
        CONSTRAINT namespaces_row_scope_customer_check CHECK (
            (row_scope = 'platform' AND customer_id IS NULL
                AND namespace_type IN ('system', 'model', 'tool', 'tool_provider', 'shared', 'knowledge'))
            OR (row_scope = 'customer' AND customer_id IS NOT NULL)
        )
    )
    """,
    # a namespace is addressed by name (``get_by_name``, a subtree grant, an owner reference), so
    # the name is unique; only workspace rows may share a schema
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_namespaces_name ON namespaces (name)",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_namespaces_schema_name_non_workspace ON namespaces (schema_name) "
    "WHERE namespace_type <> 'workspace' AND schema_name IS NOT NULL",
    """
    CREATE TABLE IF NOT EXISTS groups (
        row_scope VARCHAR(20) NOT NULL,
        group_id UUID NOT NULL,
        customer_id UUID,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        date_created TIMESTAMPTZ NOT NULL DEFAULT now(),
        date_updated TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT groups_pkey PRIMARY KEY (row_scope, group_id),
        CONSTRAINT uq_groups_group_id UNIQUE (group_id),
        CONSTRAINT groups_row_scope_check CHECK (row_scope IN ('platform', 'customer')),
        CONSTRAINT groups_row_scope_customer_check CHECK ((row_scope = 'platform') = (customer_id IS NULL))
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_groups_platform_name ON groups (name) WHERE customer_id IS NULL",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_groups_customer_name ON groups (customer_id, name) "
    "WHERE customer_id IS NOT NULL",
    """
    CREATE TABLE IF NOT EXISTS group_members (
        id UUID NOT NULL,
        group_id UUID NOT NULL,
        member_type VARCHAR(10) NOT NULL,
        member_id UUID NOT NULL,
        customer_id UUID,
        date_added TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT group_members_pkey PRIMARY KEY (group_id, id),
        CONSTRAINT group_members_member_type_check CHECK (member_type IN ('user', 'agent')),
        CONSTRAINT group_members_group_fk FOREIGN KEY (group_id) REFERENCES groups (group_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS roles (
        role_id UUID NOT NULL,
        name VARCHAR(255) NOT NULL,
        description TEXT NOT NULL,
        permissions JSONB NOT NULL,
        is_builtin BOOLEAN NOT NULL DEFAULT false,
        customer_id UUID,
        date_created TIMESTAMPTZ NOT NULL DEFAULT now(),
        date_updated TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT roles_pkey PRIMARY KEY (role_id)
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_roles_platform_name ON roles (name) WHERE customer_id IS NULL",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_roles_customer_name ON roles (customer_id, name) "
    "WHERE customer_id IS NOT NULL",
    """
    CREATE TABLE IF NOT EXISTS role_assignments (
        row_scope VARCHAR(20) NOT NULL,
        assignment_id UUID NOT NULL,
        role_id UUID NOT NULL,
        group_id UUID NOT NULL,
        scope_type VARCHAR(16) NOT NULL,
        scope_namespace_id UUID,
        scope_namespace_type VARCHAR(255),
        scope_namespace_name VARCHAR(255),
        scope_customer_id UUID,
        granted_by UUID,
        date_granted TIMESTAMPTZ NOT NULL DEFAULT now(),
        managed_by VARCHAR(64) NOT NULL DEFAULT 'manual',
        CONSTRAINT role_assignments_pkey PRIMARY KEY (row_scope, assignment_id),
        CONSTRAINT uq_role_assignments_assignment_id UNIQUE (assignment_id),
        CONSTRAINT role_assignments_scope_type_check CHECK (scope_type IN ('namespace', 'type_customer', 'all')),
        CONSTRAINT role_assignments_group_fk FOREIGN KEY (group_id) REFERENCES groups (group_id) ON DELETE CASCADE,
        CONSTRAINT role_assignments_role_fk FOREIGN KEY (role_id) REFERENCES roles (role_id) ON DELETE RESTRICT,
        CONSTRAINT role_assignments_scope_namespace_fk FOREIGN KEY (scope_namespace_id)
            REFERENCES namespaces (namespace_id) ON DELETE CASCADE
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_role_assignments_namespace_grant "
    "ON role_assignments (group_id, role_id, scope_namespace_id, managed_by) WHERE scope_type = 'namespace'",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_role_assignments_all_grant "
    "ON role_assignments (group_id, role_id, managed_by) WHERE scope_type = 'all'",
)


async def create_acl_tables(store: DataStore) -> None:
    """Create the five rbac tables, their keys and their uniqueness.

    :param store: the migration's data store
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    for statement in ACL_TABLES_DDL:
        await store.execute(statement)
    log.info("agent-acl v001: rbac tables created")
