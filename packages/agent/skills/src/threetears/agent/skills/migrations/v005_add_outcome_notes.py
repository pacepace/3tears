"""agent-skills v005: what the agent said about how a use went is kept.

Adds one nullable column to ``agent_skill_invocations``:

- ``outcome_notes TEXT`` -- the agent's line on why a use worked or failed,
  written by ``skill_report_outcome``. Before this the tool logged the line
  and dropped it. It is its own column because ``notes`` already holds the
  reason the agent gave when it invoked the skill, and one must not
  overwrite the other.

Idempotent: ``ADD COLUMN IF NOT EXISTS``, no DML. Existing rows get ``NULL``.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["add_outcome_notes"]

log = get_logger(__name__)


_ADD_OUTCOME_NOTES_SQL = "ALTER TABLE agent_skill_invocations ADD COLUMN IF NOT EXISTS outcome_notes TEXT"


async def add_outcome_notes(store: DataStore) -> None:
    """Add ``outcome_notes`` to ``agent_skill_invocations``.

    :param store: ``DataStore`` bound to the target agent schema via ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("adding agent_skill_invocations.outcome_notes (v005)")
    await store.execute(_ADD_OUTCOME_NOTES_SQL)
