"""agent-wake v014: a schedule's fire lives in its schedule's conversation, by a foreign key.

``wake_fires`` is partitioned by ``conversation_id``, and a schedule's fire is written in the schedule's conversation:
the tick's claim returns the schedule row's ``(conversation_id, schedule_id)`` and the fire takes both, and nothing
updates a schedule's ``conversation_id``. ``wake_fires_schedule_conversation_fk`` makes that a declared fact:
``(conversation_id, schedule_id)`` references ``agent_wake_schedules``' primary key, so every fire of one schedule
holds that schedule's conversation. A read bound by ``schedule_id`` (unique on ``agent_wake_schedules``) reads one
conversation's fires, which the per-wake fire count
(:data:`~threetears.agent.wake.rate_limit._COUNT_WAKE_FIRES_SQL`) relies on. Deleting a schedule cascades, as
``wake_fires_schedule_fk`` does.

This is the EXPAND step: the key is added ``NOT VALID``, so it holds for every row written from now on and the add
does not read the existing rows. ``VALIDATE CONSTRAINT`` waits for the held CONTRACT migration, once a read-only
count of mismatched rows (a fire whose conversation differs from its schedule's) is zero on each deployment.

The statement is guarded, so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["add_fire_schedule_conversation_fk"]

log = get_logger(__name__)

_ADD_FK_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        WHERE c.conname = 'wake_fires_schedule_conversation_fk'
          AND t.relname = 'wake_fires'
          AND n.nspname = current_schema()
    ) THEN
        ALTER TABLE wake_fires
            ADD CONSTRAINT wake_fires_schedule_conversation_fk
            FOREIGN KEY (conversation_id, schedule_id)
            REFERENCES agent_wake_schedules (conversation_id, schedule_id)
            ON DELETE CASCADE
            NOT VALID;
    END IF;
END
$$
"""


async def add_fire_schedule_conversation_fk(store: DataStore) -> None:
    """Add ``wake_fires_schedule_conversation_fk``, not yet validated.

    :param store: ``DataStore`` bound to the target agent schema via ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_ADD_FK_SQL)
    log.info("agent_wake v014: a schedule's fires are tied to its conversation")
