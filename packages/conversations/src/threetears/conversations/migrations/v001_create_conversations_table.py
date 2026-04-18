"""
conversations v001: create conversations table plus scoped-lookup indexes.

translated byte-equivalent from the former ``agent-memory`` v001
migration. DDL is unqualified so the L3 broker's ``search_path``
governs which schema gets the table; every statement is idempotent so
replay on recovery is safe.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

log = get_logger(__name__)


_CREATE_CONVERSATIONS_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY,
    agent_id UUID NOT NULL,
    customer_id UUID NOT NULL,
    user_id UUID NOT NULL,
    channel_type VARCHAR(50) NOT NULL,
    conversation_ref VARCHAR(500),
    status VARCHAR(20) NOT NULL,
    summary TEXT,
    date_created TIMESTAMP NOT NULL,
    date_updated TIMESTAMP NOT NULL,
    date_last_message TIMESTAMP,
    metadata JSONB
)
"""

_CREATE_CONV_USER_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_conv_user "
    "ON conversations (user_id, date_created)"
)

_CREATE_CONV_CUSTOMER_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_conv_customer "
    "ON conversations (customer_id, date_created)"
)

_CREATE_CONV_STATUS_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_conv_status ON conversations (status)"
)


async def create_conversations_table(store: DataStore) -> None:
    """
    create the ``conversations`` table and its three lookup indexes.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    """
    log.info("creating conversations table")
    await store.execute(_CREATE_CONVERSATIONS_SQL)
    await store.execute(_CREATE_CONV_USER_IDX_SQL)
    await store.execute(_CREATE_CONV_CUSTOMER_IDX_SQL)
    await store.execute(_CREATE_CONV_STATUS_IDX_SQL)
