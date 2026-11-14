"""agent-wake v015 (CONTRACT, held): the indexes no read uses are dropped, and v014's foreign key is validated.

The contract step of v013's and v014's expand / contract:

- ``idx_wake_schedules_conv_status`` (v001): v013's partial ``idx_wake_schedules_conv_active`` serves a conversation's
  active schedules (the cap's count and the admin listing); a pause writes by the primary key.
- ``idx_webhook_subs_conv`` (v003): a conversation's subscriptions read the primary key's
  ``(conversation_id, subscription_id)`` prefix.
- ``idx_wake_fires_started_conversation`` (v007): nothing reads a fire by the conversation it started.
- ``wake_fires_schedule_conversation_fk`` (v014, added ``NOT VALID``): ``VALIDATE CONSTRAINT`` reads every fire once,
  beside the table. It runs only once a read-only count of mismatched fires (a fire whose conversation differs from its
  schedule's) is zero on the deployment.

A pod of an older release may still plan through a dropped index mid-rollout, so the step is held:
:func:`~threetears.agent.wake.migrations.register` attaches it only when asked (``held_contract=True``: the test
schemas now, every deployment from the release after the one that shipped v013 and v014).

``DROP INDEX IF EXISTS``; validating a validated key is a no-op. Re-running it changes nothing.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["drop_unread_indexes_and_validate_fire_key"]

log = get_logger(__name__)

_DROP_CONV_STATUS_INDEX_SQL = "DROP INDEX IF EXISTS idx_wake_schedules_conv_status"

_DROP_WEBHOOK_CONV_INDEX_SQL = "DROP INDEX IF EXISTS idx_webhook_subs_conv"

_DROP_STARTED_CONVERSATION_INDEX_SQL = "DROP INDEX IF EXISTS idx_wake_fires_started_conversation"

_VALIDATE_FIRE_KEY_SQL = "ALTER TABLE wake_fires VALIDATE CONSTRAINT wake_fires_schedule_conversation_fk"


async def drop_unread_indexes_and_validate_fire_key(store: DataStore) -> None:
    """Drop the three unread indexes and validate ``wake_fires_schedule_conversation_fk`` (v015, contract).

    :param store: ``DataStore`` bound to the target agent schema via ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_DROP_CONV_STATUS_INDEX_SQL)
    await store.execute(_DROP_WEBHOOK_CONV_INDEX_SQL)
    await store.execute(_DROP_STARTED_CONVERSATION_INDEX_SQL)
    await store.execute(_VALIDATE_FIRE_KEY_SQL)
    log.info("agent_wake v015 (contract): three unread indexes dropped; a fire's schedule key validated")
