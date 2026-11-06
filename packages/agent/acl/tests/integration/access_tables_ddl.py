"""the access tables' DDL, for the integration tests that write them through the real collections."""

from __future__ import annotations

__all__ = ["ACCESS_TABLE_NAMES", "ACCESS_TABLES_DDL"]

ACCESS_TABLES_DDL = (
    """
    CREATE TABLE groups (
        row_scope varchar(8) NOT NULL,
        group_id uuid NOT NULL,
        customer_id uuid,
        name varchar(255) NOT NULL,
        description text,
        date_created timestamptz NOT NULL DEFAULT now(),
        date_updated timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (row_scope, group_id)
    )
    """,
    """
    CREATE TABLE group_members (
        id uuid NOT NULL,
        group_id uuid NOT NULL,
        member_type varchar(10) NOT NULL,
        member_id uuid NOT NULL,
        customer_id uuid,
        date_added timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (group_id, id)
    )
    """,
    """
    CREATE TABLE roles (
        role_id uuid PRIMARY KEY,
        name varchar(255) NOT NULL,
        description text NOT NULL DEFAULT '',
        permissions jsonb NOT NULL DEFAULT '{}',
        customer_id uuid,
        is_builtin boolean NOT NULL DEFAULT false,
        date_created timestamptz NOT NULL DEFAULT now(),
        date_updated timestamptz NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE role_assignments (
        row_scope varchar(8) NOT NULL,
        assignment_id uuid NOT NULL,
        role_id uuid NOT NULL,
        group_id uuid NOT NULL,
        scope_type varchar(16) NOT NULL,
        scope_namespace_id uuid,
        scope_namespace_type varchar(255),
        scope_customer_id uuid,
        scope_namespace_name varchar(255),
        granted_by uuid,
        date_granted timestamptz NOT NULL DEFAULT now(),
        managed_by varchar(32) NOT NULL DEFAULT 'manual',
        PRIMARY KEY (row_scope, assignment_id)
    )
    """,
    """
    CREATE TABLE namespaces (
        row_scope varchar(8) NOT NULL,
        namespace_id uuid NOT NULL UNIQUE,
        name varchar(255) NOT NULL,
        namespace_type varchar(20) NOT NULL,
        owner_agent_id uuid,
        owner_namespace varchar(255),
        customer_id uuid,
        schema_name varchar(100),
        metadata jsonb DEFAULT '{}'::jsonb,
        tool_eligible boolean NOT NULL DEFAULT true,
        skill_eligible boolean NOT NULL DEFAULT false,
        face_api boolean NOT NULL DEFAULT false,
        face_mcp boolean NOT NULL DEFAULT false,
        face_platform_tool boolean NOT NULL DEFAULT true,
        face_rest boolean NOT NULL DEFAULT false,
        face_rest_declaration jsonb,
        date_created timestamptz NOT NULL,
        date_updated timestamptz NOT NULL,
        PRIMARY KEY (row_scope, namespace_id)
    )
    """,
)
ACCESS_TABLE_NAMES = ("role_assignments", "roles", "group_members", "groups", "namespaces")
