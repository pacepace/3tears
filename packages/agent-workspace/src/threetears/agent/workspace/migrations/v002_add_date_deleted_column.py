"""
agent-workspace v002: add date_deleted column to workspaces.

adds a nullable timestamp column used by the workspace soft-delete flow
(workspace-task-10). idempotent via ``ADD COLUMN IF NOT EXISTS`` so
replay during recovery succeeds even when the column already exists in
a fresh install path that chose to bake it into the v1 CREATE TABLE.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "add_date_deleted_column",
]

log = get_logger(__name__)


_ADD_DATE_DELETED_COLUMN_SQL = "ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS date_deleted TIMESTAMP NULL"


async def add_date_deleted_column(store: DataStore) -> None:
    """
    add nullable date_deleted column to workspaces.

    :param store: DataStore bound to per-agent schema
    :ptype store: DataStore
    """
    log.info("adding date_deleted column to workspaces")
    await store.execute(_ADD_DATE_DELETED_COLUMN_SQL)
