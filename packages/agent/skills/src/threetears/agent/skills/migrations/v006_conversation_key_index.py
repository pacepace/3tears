"""agent-skills v006 (expand): a conversation's invocations page off an index holding their order completed by the key.

:meth:`~threetears.agent.skills.collections.AgentSkillInvocationCollection.list_for_conversation` reads its page
through :func:`threetears.core.keyed.keyset_page`, ordered ``invoked_at DESC, invocation_id``: ``invocation_id`` (the
table's ``UNIQUE`` key) makes the order total. ``idx_skill_invocations_conv`` (v002) holds ``invoked_at`` without the
key, so a page ordered by both reads and sorts every invocation of the conversation (YugabyteDB has no incremental
sort). ``idx_skill_invocations_conv_key`` holds both, in the page's directions.

Expand only: v002's index stays until the release's contract step drops it (old pods read it mid-rollout; the new
index serves their ``ORDER BY invoked_at DESC`` statement too). ``IF NOT EXISTS``, so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["add_conversation_key_index"]

log = get_logger(__name__)

_CONVERSATION_KEY_INDEX_SQL = (
    "-- purpose: packages/agent/skills/src/threetears/agent/skills/collections.py::AgentSkillInvocationCollection.list_for_conversation\n"
    "-- plan_test: packages/agent/skills/tests/integration/test_index_plans.py::test_skill_invocations_conv_key\n"
    "CREATE INDEX IF NOT EXISTS idx_skill_invocations_conv_key "
    "ON agent_skill_invocations (agent_id, conversation_id, invoked_at DESC, invocation_id ASC)"
)


async def add_conversation_key_index(store: DataStore) -> None:
    """Build ``idx_skill_invocations_conv_key``.

    :param store: ``DataStore`` bound to the target agent schema via ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_CONVERSATION_KEY_INDEX_SQL)
    log.info("agent_skill_invocations: a conversation's page index holds its key (v006)")
