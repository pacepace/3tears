"""agent-skills v002: create ``agent_skill_invocations`` table.

One row per skill load (wake-driven or explicit ``skill_invoke``).
Partition column ``agent_id``; composite primary key
``(agent_id, invocation_id)``; standalone ``UNIQUE (invocation_id)``
so cross-package FKs can reference the bare id.

The composite FK
``(agent_id, skill_id) REFERENCES agent_skills(agent_id, skill_id)
ON DELETE CASCADE`` ensures deleting a skill removes its history
synchronously -- analytics rows for a deleted skill are noise, not
audit, so cascade-on-delete is the right primitive (per PLACEMENT
§1.1 disposition).

``message_id`` is deliberately NOT FK'd to ``messages``: the messages
table is consumer-owned (a consumer has it; future consumers may
differ) and rows may be hard-deleted. The invocation history must
survive a message deletion, so the column is a plain UUID with no
FK constraint.

CHECK constraints:

- ``invocation_source IN ('wake', 'invoke')`` -- enum-by-app.
- ``outcome IS NULL OR outcome IN ('success', 'failure')`` -- NULL is
  valid (no marker matched in the assistant's response).
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "create_agent_skill_invocations",
]

log = get_logger(__name__)


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS agent_skill_invocations (
    agent_id          UUID         NOT NULL,
    invocation_id     UUID         NOT NULL,
    skill_id          UUID         NOT NULL,
    user_id           UUID         NOT NULL,
    conversation_id   UUID         NOT NULL,
    message_id        UUID,
    invocation_source TEXT         NOT NULL,
    invoked_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
    outcome           TEXT,
    outcome_source    TEXT,
    notes             TEXT,
    PRIMARY KEY (agent_id, invocation_id),
    UNIQUE (invocation_id),
    CONSTRAINT agent_skill_invocations_skill_fk
        FOREIGN KEY (agent_id, skill_id)
        REFERENCES agent_skills(agent_id, skill_id)
        ON DELETE CASCADE,
    CONSTRAINT agent_skill_invocations_source_check
        CHECK (invocation_source IN ('wake', 'invoke')),
    CONSTRAINT agent_skill_invocations_outcome_check
        CHECK (outcome IS NULL OR outcome IN ('success', 'failure'))
)
"""

_CREATE_SKILL_TIME_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_skill_invocations_skill_time "
    "ON agent_skill_invocations (agent_id, skill_id, invoked_at DESC)"
)

_CREATE_CONV_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_skill_invocations_conv "
    "ON agent_skill_invocations (agent_id, conversation_id, invoked_at DESC)"
)


async def create_agent_skill_invocations(store: DataStore) -> None:
    """Create ``agent_skill_invocations`` table + indexes.

    :param store: ``DataStore`` bound to the target agent schema via
        ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("creating agent_skill_invocations table (v002)")
    await store.execute(_CREATE_TABLE_SQL)
    await store.execute(_CREATE_SKILL_TIME_INDEX_SQL)
    await store.execute(_CREATE_CONV_INDEX_SQL)
