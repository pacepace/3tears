"""
conversations v013: a folder's conversations and one ref's conversations each read an index.

``ConversationsCollection.find_by_folder`` / ``count_by_folder`` read one folder's conversations under the agent,
newest first, and ``find_by_ref`` one ``conversation_ref``'s under the agent and a channel. No index led with those
pins, so each read every conversation of the agent and filtered. Two indexes, each led by ``agent_id`` (every read
binds it by equality; YugabyteDB's hash on the first column is a point and the rest range), then the read's pins,
then its order and the tie-break:

- ``idx_conv_agent_folder_created (agent_id, folder_id, date_created, conversation_id)``
- ``idx_conv_agent_ref_created (agent_id, channel_type, conversation_ref, date_created, conversation_id)``

The expand step of expand / contract: nothing is dropped here. Idempotent: ``CREATE INDEX IF NOT EXISTS``, one
statement at a time (YugabyteDB runs no DDL in a transaction). The purposes and plan tests are declared beside each
index on ``ConversationsCollection.schema``.

Forward-only: 3tears migrations do not declare downgrades.

Revision ID: 013
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["add_folder_and_ref_indexes"]

log = get_logger(__name__)

#: one literal statement per index, in the order they are built
_INDEXES_SQL: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_conv_agent_folder_created"
    " ON conversations (agent_id, folder_id, date_created, conversation_id)",
    "CREATE INDEX IF NOT EXISTS idx_conv_agent_ref_created"
    " ON conversations (agent_id, channel_type, conversation_ref, date_created, conversation_id)",
)


async def add_folder_and_ref_indexes(store: DataStore) -> None:
    """
    build the folder and ref indexes.

    runs in the per-agent schema set by the migration runner's
    ``search_path``; every statement is idempotent on its own.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("v013: building the conversations folder and ref indexes")
    for sql in _INDEXES_SQL:
        await store.execute(sql)
    log.info("v013: conversations folder and ref indexes complete")
