"""
agent-workspace v005: rename ``id`` PK columns to canonical
``<entity>_id`` shape across the workspace tables.

Rename triple, all under one migration so the FK constraints are
updated atomically by Postgres:

- ``workspaces.id`` -> ``workspaces.workspace_id``
- ``workspace_files.id`` -> ``workspace_files.file_id``
- ``workspace_file_versions.id`` -> ``workspace_file_versions.version_id``

Postgres updates the PK constraint, every index referencing the
renamed column, and every dependent FK constraint automatically on
``RENAME COLUMN``. The two FKs in this package's v001 migration are:

- ``workspace_files.workspace_id REFERENCES workspaces(id) ON DELETE CASCADE``
- ``workspace_file_versions.workspace_id REFERENCES workspaces(id) ON DELETE CASCADE``

Both point at ``workspaces(id)``. Renaming that column updates both
FK constraints in the same transaction; the constraints themselves
keep their auto-generated names.

v0.8.0 shard 04.6 rationale. Every other entity table in the 3tears
+ metallm stack uses the ``<entity>_id`` shape
(``memories.memory_id``, ``media.media_id``,
``context_items.context_id``, ``conversations.conversation_id`` (after
v007), etc.). The workspace tables shipped with bare-``id`` PK
columns; this migration closes the gap.

Idempotency. Each rename uses a guarded ``information_schema.columns``
DO block: if the source column exists and the target does not, the
rename runs; otherwise it no-ops. Replays on a fully-migrated schema
are safe.

Forward-only: 3tears migrations do not declare downgrades.

Revision ID: 005
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "rename_id_columns",
]

log = get_logger(__name__)


_RENAME_WORKSPACES_ID_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'workspaces'
          AND column_name = 'id'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'workspaces'
          AND column_name = 'workspace_id'
    ) THEN
        ALTER TABLE workspaces RENAME COLUMN id TO workspace_id;
        RAISE NOTICE 'v005: renamed workspaces.id -> workspaces.workspace_id';
    ELSE
        RAISE NOTICE 'v005: workspaces rename no-op';
    END IF;
END
$$
"""

_RENAME_WORKSPACE_FILES_ID_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'workspace_files'
          AND column_name = 'id'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'workspace_files'
          AND column_name = 'file_id'
    ) THEN
        ALTER TABLE workspace_files RENAME COLUMN id TO file_id;
        RAISE NOTICE 'v005: renamed workspace_files.id -> workspace_files.file_id';
    ELSE
        RAISE NOTICE 'v005: workspace_files rename no-op';
    END IF;
END
$$
"""

_RENAME_WORKSPACE_FILE_VERSIONS_ID_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'workspace_file_versions'
          AND column_name = 'id'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'workspace_file_versions'
          AND column_name = 'version_id'
    ) THEN
        ALTER TABLE workspace_file_versions RENAME COLUMN id TO version_id;
        RAISE NOTICE 'v005: renamed workspace_file_versions.id -> workspace_file_versions.version_id';
    ELSE
        RAISE NOTICE 'v005: workspace_file_versions rename no-op';
    END IF;
END
$$
"""


async def rename_id_columns(store: DataStore) -> None:
    """rename ``id`` PK columns to ``<entity>_id`` shape on the three
    workspace tables.

    runs in the per-agent schema set by the migration runner's
    ``search_path``. each rename is idempotent via a guarded
    ``information_schema`` DO block so replays on a fully-migrated
    schema are no-ops.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("v005: renaming workspaces.id / workspace_files.id / workspace_file_versions.id")
    await store.execute(_RENAME_WORKSPACES_ID_SQL)
    await store.execute(_RENAME_WORKSPACE_FILES_ID_SQL)
    await store.execute(_RENAME_WORKSPACE_FILE_VERSIONS_ID_SQL)
    log.info("v005: workspace column renames complete")
