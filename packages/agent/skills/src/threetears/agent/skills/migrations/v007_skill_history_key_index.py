"""agent-skills v007 (expand): a skill's history pages off an index holding its order completed by the key.

:meth:`~threetears.agent.skills.collections.AgentSkillInvocationCollection.list_for_skill` reads its page through
:func:`threetears.core.keyed.keyset_page`, ordered ``invoked_at DESC, invocation_id DESC``: ``invocation_id`` (the
table's ``UNIQUE`` key) makes the order total, so a cursor names one place. v002's ``idx_skill_invocations_skill_time``
(``agent_id, skill_id, invoked_at DESC``) holds the order without the key, so a page ordered by both reads and sorts
every use of the skill (YugabyteDB has no incremental sort); its ``OFFSET`` page read and threw away every use before
the page. ``idx_skill_invocations_skill_time_key`` holds both, in the page's directions.

It covers v002's index (the same leading columns, over the same rows), so v002's is dropped in this same step (Pace's
ruling, 2026-11-14): an old pod's ``ORDER BY invoked_at DESC`` page and the database's own lookup of a deleted skill's
uses (``agent_skill_invocations_skill_fk``) read the new one as they read the old. ``IF [NOT] EXISTS``, so re-running
it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["add_skill_history_key_index"]

log = get_logger(__name__)

_SKILL_HISTORY_KEY_INDEX_SQL = (
    "-- purpose: packages/agent/skills/src/threetears/agent/skills/collections.py::AgentSkillInvocationCollection.list_for_skill;"
    " fk:agent_skill_invocations_skill_fk\n"
    "-- plan_test: packages/agent/skills/tests/integration/test_index_plans.py::test_skill_invocations_skill_time_key\n"
    "CREATE INDEX IF NOT EXISTS idx_skill_invocations_skill_time_key "
    "ON agent_skill_invocations (agent_id, skill_id, invoked_at DESC, invocation_id DESC)"
)

_DROP_COVERED_INDEX_SQL = "DROP INDEX IF EXISTS idx_skill_invocations_skill_time"


async def add_skill_history_key_index(store: DataStore) -> None:
    """Build ``idx_skill_invocations_skill_time_key`` and drop the index it covers.

    :param store: ``DataStore`` bound to the target agent schema via ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_SKILL_HISTORY_KEY_INDEX_SQL)
    await store.execute(_DROP_COVERED_INDEX_SQL)
    log.info("agent_skill_invocations: a skill's history index holds its key, the one it covers dropped (v007)")
