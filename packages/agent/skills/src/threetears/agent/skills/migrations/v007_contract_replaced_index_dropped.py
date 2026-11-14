"""agent-skills v007 (CONTRACT, held): v002's conversation index, which v006's replaced, is dropped.

The contract step of v006's expand / contract. :meth:`~threetears.agent.skills.collections.AgentSkillInvocationCollection.list_for_conversation`
reads ``idx_skill_invocations_conv_key`` (v006: the conversation's page completed by the key); v002's
``idx_skill_invocations_conv`` holds the same order without the key and no read of this release uses it. A pod of an
older release may still plan through it mid-rollout, so the drop is held: :func:`~threetears.agent.skills.migrations.register`
attaches this version only when asked (``held_contract=True``: the test schemas now, every deployment from the release
after the one that shipped v006).

``DROP INDEX IF EXISTS``, so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["drop_replaced_conversation_index"]

log = get_logger(__name__)

_DROP_CONVERSATION_INDEX_SQL = "DROP INDEX IF EXISTS idx_skill_invocations_conv"


async def drop_replaced_conversation_index(store: DataStore) -> None:
    """Drop ``idx_skill_invocations_conv`` (v007, contract).

    :param store: ``DataStore`` bound to the target agent schema via ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_DROP_CONVERSATION_INDEX_SQL)
    log.info("agent_skill_invocations: v002's conversation index dropped, v006's serves its page (v007, contract)")
