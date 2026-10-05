"""agent-wake v008: a check that found nothing is a fire status of its own.

A consumer may run a check before a fire starts anything -- new mail, a
changed page -- and wake the agent only when it finds something. A check
that found nothing ran no turn, so it is neither ``'fired'`` nor
``'fired_silent'``, the two statuses the fire limits count: recorded as
either, a check every ten minutes would spend an agent's whole budget.
``'checked_quiet'`` records it, uncounted and not a failure.

PostgreSQL has no ALTER CONSTRAINT for a CHECK predicate; the constraint is
dropped and re-added under the same name, as v004 and v007 did. The pair is
idempotent.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["add_checked_quiet_status"]

log = get_logger(__name__)

_DROP_STATUS_CHECK_SQL = "ALTER TABLE wake_fires DROP CONSTRAINT IF EXISTS wake_fires_status_check"

_ADD_STATUS_CHECK_SQL = """
ALTER TABLE wake_fires
    ADD CONSTRAINT wake_fires_status_check
        CHECK (status IN (
            'dispatching',
            'fired',
            'fired_silent',
            'yielded',
            'skipped_busy',
            'skipped_rate_limit',
            'skipped_cap',
            'skipped_no_handler',
            'skipped_life_off',
            'checked_quiet',
            'failed'
        ))
"""


async def add_checked_quiet_status(store: DataStore) -> None:
    """Accept ``'checked_quiet'`` on ``wake_fires.status``.

    :param store: ``DataStore`` bound to the target agent schema via ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("agent_wake v008: checked_quiet fire status")
    await store.execute(_DROP_STATUS_CHECK_SQL)
    await store.execute(_ADD_STATUS_CHECK_SQL)
