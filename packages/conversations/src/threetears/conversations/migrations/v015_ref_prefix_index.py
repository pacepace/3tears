"""
conversations v015: the refs starting with a prefix on a channel read an index, as a code-point range.

``ConversationsCollection.find_by_ref_prefix`` lists every conversation whose ``conversation_ref`` starts with a prefix
under the agent and a channel (a story's every object chat). It matched ``left(conversation_ref, char_length($3)) = $3``,
a filter on every conversation of the agent's channel. It now reads the prefix as a range in code-point order
(``conversation_ref COLLATE "C" >= $prefix AND < $successor``), which only an index holding the column in that
collation seeks: v013's ``idx_conv_agent_ref_created`` holds it in the database's own collation (where ``a/zzz`` may
sort past ``a0``), so it cannot.

- ``idx_conv_agent_ref_c (agent_id, channel_type, conversation_ref COLLATE "C" ASC)``: the agent (YugabyteDB's hash,
  a point), the channel, then the refs in byte order, range-sharded so a prefix is one range.

The expand step of expand / contract: nothing is dropped here (``find_by_ref``'s equality still reads v013's index).
Idempotent: ``CREATE INDEX IF NOT EXISTS``. The purpose and plan test are declared beside the index on
``ConversationsCollection.schema``.

Forward-only: 3tears migrations do not declare downgrades.

Revision ID: 015
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["add_ref_prefix_index"]

log = get_logger(__name__)

_REF_PREFIX_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_conv_agent_ref_c"
    ' ON conversations (agent_id, channel_type, conversation_ref COLLATE "C" ASC)'
)


async def add_ref_prefix_index(store: DataStore) -> None:
    """
    build the ref-prefix index.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_REF_PREFIX_INDEX_SQL)
    log.info("v015: conversations ref-prefix index complete")
