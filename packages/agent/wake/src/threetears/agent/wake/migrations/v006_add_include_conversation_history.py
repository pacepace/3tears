"""agent-wake v006: add ``agent_wake_schedules.include_conversation_history``.

A wake schedule gains a boolean switch controlling whether a fire
carries the conversation's recent history into the LLM context. When
``true`` (the default, preserving the pre-flag behavior) the consumer
injects the recent conversation messages so the agent continues the
live thread; when ``false`` the wake fires without conversation history
(a self-directed run that still knows it is a wake and its origin).

This is independent of the attached skill's ``prompt_mode`` (persona
injection) -- the two switches compose, so all four combinations are
valid (e.g. ``replace`` + history = a neutral analyzer over the
conversation).

Idempotent via ``ADD COLUMN IF NOT EXISTS``. ``NOT NULL DEFAULT true``
backfills existing rows to the prior always-on behavior in one shot.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "add_include_conversation_history",
]

log = get_logger(__name__)


_ADD_COLUMN_SQL = """
ALTER TABLE agent_wake_schedules
    ADD COLUMN IF NOT EXISTS include_conversation_history
        BOOLEAN NOT NULL DEFAULT true
"""


async def add_include_conversation_history(store: DataStore) -> None:
    """Add the ``include_conversation_history`` column.

    :param store: ``DataStore`` bound to the target agent schema via
        ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info(
        "adding agent_wake_schedules.include_conversation_history "
        "(NOT NULL DEFAULT true) (v006)"
    )
    await store.execute(_ADD_COLUMN_SQL)
