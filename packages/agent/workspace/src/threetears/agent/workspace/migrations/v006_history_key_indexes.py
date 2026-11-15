"""agent-workspace v006: a workspace's and a path's history are indexed in their order, completed by the key.

``threetears.workspace.history`` reads one page newest first through :func:`threetears.core.keyed.keyset_page`,
ordered ``date_created DESC, version_id DESC``: ``version_id`` (the primary key) makes the order total and runs the
way ``date_created`` does, so one index walks the page. v001's ``idx_workspace_file_versions_history`` holds
``(workspace_id, date_created)`` alone: the page completed by the key read and sorted every journal row of the
workspace (YugabyteDB has no incremental sort), and a path's page read every version of the path and sorted them.

- ``idx_workspace_file_versions_history_key`` ``(workspace_id, date_created DESC, version_id DESC)``
  (:meth:`~threetears.agent.workspace.collections.WorkspaceFileVersionCollection.find_by_workspace`). It leads with
  v001's columns over the same rows, so it serves every read the old one did (a previous release's
  ``ORDER BY date_created DESC`` page mid-rollout walks it forward): v001's index is dropped here, after the new one
  is built (covering-drop ruling, 2026-11-14).
- ``idx_workspace_file_versions_path_history`` ``(workspace_id, relative_path, date_created DESC, version_id DESC)``
  (:meth:`~threetears.agent.workspace.collections.WorkspaceFileVersionCollection.find_by_workspace_and_path`); why not
  the ``UNIQUE (workspace_id, relative_path, version)`` read by ``version DESC`` is at the declaration.

Every statement is guarded, so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["add_history_key_indexes"]

log = get_logger(__name__)

_HISTORY_KEY_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_workspace_file_versions_history_key "
    "ON workspace_file_versions (workspace_id, date_created DESC, version_id DESC)"
)

_PATH_HISTORY_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_workspace_file_versions_path_history "
    "ON workspace_file_versions (workspace_id, relative_path, date_created DESC, version_id DESC)"
)

_DROP_HISTORY_INDEX_SQL = "DROP INDEX IF EXISTS idx_workspace_file_versions_history"


async def add_history_key_indexes(store: DataStore) -> None:
    """Build the two history indexes, then drop v001's ``idx_workspace_file_versions_history``.

    :param store: ``DataStore`` bound to the target agent schema via ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_HISTORY_KEY_INDEX_SQL)
    await store.execute(_PATH_HISTORY_INDEX_SQL)
    await store.execute(_DROP_HISTORY_INDEX_SQL)
    log.info("agent_workspace v006: a workspace's and a path's history are indexed in their order")
