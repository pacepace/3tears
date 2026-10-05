"""epoch v002: add ``config_epochs.previous_epoch``.

``previous_epoch`` is the epoch a durable subject held before its latest move by
:meth:`threetears.epoch.client.EpochClient.advance_to`: the value that move replaced,
``NULL`` before the subject has moved twice (an insert replaces nothing). A tile
version's previous generation is what the hub keeps serving while clients still hold
it, and versions may skip, so it is recorded rather than inferred as one below.

The column is nullable and added with ``IF NOT EXISTS``, so the statement is idempotent
and existing rows read ``NULL`` until their next move.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "add_previous_epoch_column",
]

log = get_logger(__name__)


_ADD_PREVIOUS_EPOCH_SQL = "ALTER TABLE config_epochs ADD COLUMN IF NOT EXISTS previous_epoch BIGINT"


async def add_previous_epoch_column(store: DataStore) -> None:
    """add the ``previous_epoch`` column to ``config_epochs``.

    :param store: DataStore bound to platform schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("adding config_epochs.previous_epoch")
    await store.execute(_ADD_PREVIOUS_EPOCH_SQL)
