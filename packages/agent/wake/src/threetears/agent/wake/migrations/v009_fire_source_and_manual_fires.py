"""agent-wake v009: a fire says what started it, and a wake can be run by hand.

- ``wake_fires.fire_source`` (``TEXT NOT NULL DEFAULT 'scheduled_tick'``): what
  started the fire. ``'scheduled_tick'`` for the tick, ``'webhook'`` for an
  inbound event, ``'manual'`` for a fire someone asked for outside the
  schedule. Rows written before this version are backfilled: a row with a
  ``webhook_subscription_id`` is ``'webhook'``, every other row
  ``'scheduled_tick'``.
- ``wake_fires.requested_by``: who asked for a manual fire, as the consumer
  names them. A CHECK requires it on a manual fire.
- ``agent_wake_schedules.last_manual_fire_at``: when the wake was last run by
  hand. The manual-fire limit is taken by one ``UPDATE`` on this column
  (:func:`~threetears.agent.wake.manual.claim_manual_fire`); it is never read
  from ``last_fired_at``, which the reschedule engine reads and a manual fire
  does not touch.

Every statement is guarded, so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["add_fire_source_and_manual_fires"]

log = get_logger(__name__)

_ADD_FIRE_SOURCE_SQL = """
ALTER TABLE wake_fires
    ADD COLUMN IF NOT EXISTS fire_source TEXT NOT NULL DEFAULT 'scheduled_tick'
"""

_ADD_REQUESTED_BY_SQL = """
ALTER TABLE wake_fires
    ADD COLUMN IF NOT EXISTS requested_by TEXT
"""

# Only rows still carrying the column default are touched, so a second run changes nothing.
_BACKFILL_WEBHOOK_SQL = """
UPDATE wake_fires
   SET fire_source = 'webhook'
 WHERE webhook_subscription_id IS NOT NULL
   AND fire_source = 'scheduled_tick'
"""


def _add_check_sql(name: str, expression: str) -> str:
    return f"""
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        WHERE c.conname = '{name}'
          AND t.relname = 'wake_fires'
          AND n.nspname = current_schema()
    ) THEN
        ALTER TABLE wake_fires ADD CONSTRAINT {name} CHECK ({expression});
    END IF;
END
$$
"""


_ADD_FIRE_SOURCE_CHECK_SQL = _add_check_sql(
    "wake_fires_fire_source_check",
    "fire_source IN ('scheduled_tick', 'webhook', 'manual')",
)

_ADD_REQUESTED_BY_CHECK_SQL = _add_check_sql(
    "wake_fires_manual_requested_by_check",
    "fire_source <> 'manual' OR requested_by IS NOT NULL",
)

_ADD_LAST_MANUAL_FIRE_SQL = """
ALTER TABLE agent_wake_schedules
    ADD COLUMN IF NOT EXISTS last_manual_fire_at TIMESTAMPTZ
"""


async def add_fire_source_and_manual_fires(store: DataStore) -> None:
    """Add ``fire_source`` and ``requested_by`` to fires, and ``last_manual_fire_at`` to wakes.

    :param store: ``DataStore`` bound to the target agent schema via ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("agent_wake v009: fire source, who asked for a manual fire, the manual-fire stamp")
    await store.execute(_ADD_FIRE_SOURCE_SQL)
    await store.execute(_ADD_REQUESTED_BY_SQL)
    await store.execute(_BACKFILL_WEBHOOK_SQL)
    await store.execute(_ADD_FIRE_SOURCE_CHECK_SQL)
    await store.execute(_ADD_REQUESTED_BY_CHECK_SQL)
    await store.execute(_ADD_LAST_MANUAL_FIRE_SQL)
