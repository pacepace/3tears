"""workspace schema migrations.

migrations run against a DataStore bound to a per-agent YugabyteDB schema
via search_path. tables are therefore created unqualified; the schema is
supplied by the L3 layer at execution time.

the shard's sketch used a schema= kwarg on the version decorator -- the
real MigrationRunner.version(n) takes only an int. the migration body
receives the DataStore and executes SQL via store.execute().

typical use::

    from threetears.agent.workspace.migrations import register_workspace_migrations

    runner = register_workspace_migrations(store)
    await runner.apply()
"""

from __future__ import annotations

from threetears.core.data.migrations import MigrationRunner
from threetears.core.data.store import DataStore
from threetears.observe import get_logger

log = get_logger(__name__)


_CREATE_WORKSPACES_SQL = """
CREATE TABLE IF NOT EXISTS workspaces (
    id UUID PRIMARY KEY,
    agent_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    template_name VARCHAR(255),
    created_by UUID NOT NULL,
    current_version INTEGER NOT NULL DEFAULT 0,
    date_created TIMESTAMP NOT NULL,
    date_updated TIMESTAMP NOT NULL,
    CONSTRAINT uq_workspaces_agent_name UNIQUE (agent_id, name)
)
"""

_CREATE_WORKSPACE_FILES_SQL = """
CREATE TABLE IF NOT EXISTS workspace_files (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    relative_path VARCHAR(512) NOT NULL,
    content BYTEA NOT NULL,
    sha256 CHAR(64) NOT NULL,
    version INTEGER NOT NULL,
    date_updated TIMESTAMP NOT NULL,
    CONSTRAINT uq_workspace_files_path UNIQUE (workspace_id, relative_path)
)
"""

_CREATE_WORKSPACE_FILES_WORKSPACE_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_workspace_files_workspace ON workspace_files (workspace_id)"
)

_CREATE_WORKSPACE_FILE_VERSIONS_SQL = """
CREATE TABLE IF NOT EXISTS workspace_file_versions (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    relative_path VARCHAR(512) NOT NULL,
    version INTEGER NOT NULL,
    content BYTEA NOT NULL,
    sha256 CHAR(64) NOT NULL,
    action VARCHAR(32) NOT NULL,
    label VARCHAR(255),
    actor_id UUID NOT NULL,
    correlation_id UUID NOT NULL,
    date_created TIMESTAMP NOT NULL,
    CONSTRAINT uq_workspace_file_versions_triple UNIQUE (workspace_id, relative_path, version)
)
"""

_CREATE_WORKSPACE_FILE_VERSIONS_HISTORY_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_workspace_file_versions_history "
    "ON workspace_file_versions (workspace_id, date_created)"
)

_ADD_DATE_DELETED_COLUMN_SQL = "ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS date_deleted TIMESTAMP NULL"


async def create_workspace_tables(store: DataStore) -> None:
    """
    creates workspace, workspace_file, and workspace_file_version tables.

    migration runs in per-agent schema bound via search_path by the L3 layer;
    statements are therefore unqualified. idempotent via IF NOT EXISTS so
    replay during recovery does not fail.

    :param store: DataStore bound to per-agent schema
    :ptype store: DataStore
    """
    log.info("creating workspace tables")
    await store.execute(_CREATE_WORKSPACES_SQL)
    await store.execute(_CREATE_WORKSPACE_FILES_SQL)
    await store.execute(_CREATE_WORKSPACE_FILES_WORKSPACE_IDX_SQL)
    await store.execute(_CREATE_WORKSPACE_FILE_VERSIONS_SQL)
    await store.execute(_CREATE_WORKSPACE_FILE_VERSIONS_HISTORY_IDX_SQL)


async def add_date_deleted_column(store: DataStore) -> None:
    """
    adds nullable date_deleted column to workspaces for soft-delete support.

    idempotent via IF NOT EXISTS so replay during recovery does not fail
    even when the column already exists from a fresh-install path that
    chose to bake it into the v1 CREATE TABLE.

    :param store: DataStore bound to per-agent schema
    :ptype store: DataStore
    """
    log.info("adding date_deleted column to workspaces")
    await store.execute(_ADD_DATE_DELETED_COLUMN_SQL)


def register_workspace_migrations(store: DataStore) -> MigrationRunner:
    """
    builds MigrationRunner bound to store and registers workspace migrations.

    caller invokes runner.apply() to execute pending migrations. this module
    owns version assignment for the agent-workspace package; version 1 is
    the initial schema described in workspace-task-05; version 2 adds the
    date_deleted column to workspaces for soft-delete (workspace-task-10).

    :param store: DataStore bound to per-agent schema
    :ptype store: DataStore
    :return: configured MigrationRunner ready for apply()
    :rtype: MigrationRunner
    """
    runner = MigrationRunner(store)
    runner.version(1)(create_workspace_tables)
    runner.version(2)(add_date_deleted_column)
    return runner
