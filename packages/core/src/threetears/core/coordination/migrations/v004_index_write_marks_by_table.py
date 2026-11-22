"""coordination v004: the write-mark queue's ``(table_name, id)`` index, now built no more.

Until 0.95 the hub's broker read, on every commit it answered, the marks of the tables the commit wrote, and this gave
the queue the index that read went through. The marks are gone (coordination v006); a database that ran v004 before
keeps its index for an older replica mid-roll, until the contract release drops the queue with it, and a database
created now needs none. The version stays so every database's ledger holds the same versions.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["index_write_marks_by_table"]

log = get_logger(__name__)


async def index_write_marks_by_table(store: DataStore) -> None:
    """nothing, from 0.95: the index no current read uses (the version kept for the ledger).

    :param store: DataStore bound to the target schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("coordination v004 builds no write-mark index from 0.95 on: the marks are gone")
