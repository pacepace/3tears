"""conversations v014 (CONTRACT, held): the two indexes no read uses are dropped.

- ``idx_conv_customer`` (v001, ``(customer_id, date_created)``): no read in 3tears or a consumer pins a customer's
  conversations; a person's are read by ``idx_conv_user``, an agent's by v012's agent-led sort indexes.
- ``idx_conv_parent`` (v011, ``(parent_type, parent_id)``): nothing reads a conversation by what started it.

A pod of an older release may still plan through a dropped index mid-rollout, so the step is held:
:func:`~threetears.conversations.migrations.register` attaches it only when asked (``held_contract=True``: the test
schemas now, every deployment from the release after this one).

``DROP INDEX IF EXISTS``, so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["drop_unread_conversation_indexes"]

log = get_logger(__name__)

_DROP_CUSTOMER_INDEX_SQL = "DROP INDEX IF EXISTS idx_conv_customer"

_DROP_PARENT_INDEX_SQL = "DROP INDEX IF EXISTS idx_conv_parent"


async def drop_unread_conversation_indexes(store: DataStore) -> None:
    """Drop ``idx_conv_customer`` and ``idx_conv_parent`` (v014, contract).

    :param store: ``DataStore`` bound to the target agent schema via ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_DROP_CUSTOMER_INDEX_SQL)
    await store.execute(_DROP_PARENT_INDEX_SQL)
    log.info("conversations v014 (contract): idx_conv_customer and idx_conv_parent dropped")
