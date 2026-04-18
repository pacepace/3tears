"""
agent-workspace v001: create workspaces, workspace_files, workspace_file_versions.

migrations run against a DataStore bound to a per-agent YugabyteDB
schema via search_path. statements are therefore unqualified and
idempotent via ``CREATE TABLE IF NOT EXISTS`` so replay on recovery is
safe. this is the migration translated from the workspace-task-05
baseline shipped in the former package-local runner.
"""

from __future__ import annotations

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


async def create_workspace_tables(store: DataStore) -> None:
    """
    create workspaces, workspace_files, workspace_file_versions tables.

    migration runs in per-agent schema bound via search_path by the L3
    layer; statements are unqualified and idempotent via IF NOT EXISTS
    so replay during recovery does not fail.

    :param store: DataStore bound to per-agent schema
    :ptype store: DataStore
    """
    log.info("creating workspace tables")
    await store.execute(_CREATE_WORKSPACES_SQL)
    await store.execute(_CREATE_WORKSPACE_FILES_SQL)
    await store.execute(_CREATE_WORKSPACE_FILES_WORKSPACE_IDX_SQL)
    await store.execute(_CREATE_WORKSPACE_FILE_VERSIONS_SQL)
    await store.execute(_CREATE_WORKSPACE_FILE_VERSIONS_HISTORY_IDX_SQL)
