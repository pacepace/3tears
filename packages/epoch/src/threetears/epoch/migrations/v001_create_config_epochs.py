"""epoch v001: create ``config_epochs`` table.

DDL is unqualified so the caller's ``search_path`` governs which
schema gets the table. every statement is idempotent so replay on
recovery is safe.

shape:

- ``subject_path TEXT PRIMARY KEY`` -- the namespaced NATS subject
  string (e.g. ``"metallm.capabilities.epoch"``); also the broadcast
  subject so wire identity matches row identity exactly
- ``epoch BIGINT NOT NULL DEFAULT 0`` -- strictly-monotonic counter,
  bumped via ``ON CONFLICT DO UPDATE SET epoch = epoch + 1``
- ``payload JSONB`` -- opaque hint forwarded to subscribers in the
  broadcast envelope; framework never inspects
- ``date_updated TIMESTAMPTZ NOT NULL DEFAULT now()`` -- aware-UTC
  timestamp, refreshed on every bump
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "create_config_epochs_table",
]

log = get_logger(__name__)


_CREATE_CONFIG_EPOCHS_SQL = """
CREATE TABLE IF NOT EXISTS config_epochs (
    subject_path TEXT PRIMARY KEY,
    epoch BIGINT NOT NULL DEFAULT 0,
    payload JSONB,
    date_updated TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


async def create_config_epochs_table(store: DataStore) -> None:
    """create the ``config_epochs`` table.

    :param store: DataStore bound to platform schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("creating config_epochs table")
    await store.execute(_CREATE_CONFIG_EPOCHS_SQL)
