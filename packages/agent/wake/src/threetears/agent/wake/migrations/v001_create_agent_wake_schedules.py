"""agent-wake v001: create ``agent_wake_schedules`` table + indexes.

agent-wake shard 01. Partition column is ``conversation_id``;
composite primary key is ``(conversation_id, schedule_id)``.
Standalone ``UNIQUE (schedule_id)`` lets cross-package FKs reference
the bare column without partition knowledge -- specifically
``wake_fires.schedule_id`` declared in v002.

Nullable ``skill_id UUID REFERENCES agent_skills(skill_id) ON DELETE
SET NULL`` -- single attached skill per wake (PLACEMENT §1.1).
``ON DELETE SET NULL`` so deleting the skill leaves the schedule
active but unbound. Relies on the cross-package standalone
``UNIQUE (skill_id)`` constraint declared in agent-skills v001.

CHECK constraints:

- ``schedule_type`` enum-by-app (no DB CHECK; PLACEMENT anti-pattern
  notes the type list is app-evolvable and the agent-tools layer
  validates the type / config pairing).
- ``execution_mode IN ('inline', 'spawn')`` -- CHECK-pinned.
- ``status IN ('active', 'paused', 'expired')`` -- CHECK-pinned.
- ``missed_fire_policy IN ('coalesce', 'catch_up')`` -- CHECK-pinned
  (added per PLACEMENT §1.7 in the 2026-05-19 revision).
- ``delivery_target IN ('conversation', 'email')`` -- CHECK-pinned.

Self-FK ``context_from_schedule_id UUID REFERENCES
agent_wake_schedules(schedule_id) ON DELETE SET NULL`` -- single-hop,
same-conversation only (PLACEMENT §1.6). App-layer cycle detection
lives in shard 04.

NO FK on ``conversation_id``: the 3tears ``conversations`` table has
composite PK ``(agent_id, conversation_id)`` with no standalone
``UNIQUE (conversation_id)``, so a single-column FK is not legal.
Same precedent as ``context_items.conversation_id`` (agent-tools
v003) and ``agent_skill_invocations.conversation_id`` (agent-skills
v002). The package's :func:`register` declares
``depends_on=("conversations", ...)`` so the migration runner
orders this after the conversations migrations, but no DB-level FK
exists.

every statement is idempotent so re-running this migration on a
schema that already has the table is a no-op
(``CREATE TABLE IF NOT EXISTS`` / ``CREATE INDEX IF NOT EXISTS``).

Anti-pattern reminders (PLACEMENT shard-01 body):

- No ``gen_random_uuid()`` default on ``schedule_id`` -- UUIDs are
  uuid7 allocated app-side via ``uuid_utils.uuid7()``.
- No CHECK on ``schedule_config`` shape -- JSONB shape varies per
  ``schedule_type``; validation lives in the agent-tools layer.
- ``name`` is nullable -- agents often create schedules without
  naming them.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "create_agent_wake_schedules",
]

log = get_logger(__name__)


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS agent_wake_schedules (
    conversation_id           UUID         NOT NULL,
    schedule_id               UUID         NOT NULL,
    user_id                   UUID         NOT NULL,
    agent_id                  UUID         NOT NULL,
    skill_id                  UUID,
    schedule_type             TEXT         NOT NULL,
    schedule_config           JSONB        NOT NULL DEFAULT '{}'::jsonb,
    task_prompt               TEXT,
    execution_mode            TEXT         NOT NULL DEFAULT 'inline',
    status                    TEXT         NOT NULL DEFAULT 'active',
    next_fire_at              TIMESTAMPTZ,
    last_fired_at             TIMESTAMPTZ,
    name                      TEXT,
    missed_fire_policy        TEXT         NOT NULL DEFAULT 'coalesce',
    context_from_schedule_id  UUID,
    delivery_target           TEXT         NOT NULL DEFAULT 'conversation',
    delivery_config           JSONB        NOT NULL DEFAULT '{}'::jsonb,
    date_created              TIMESTAMPTZ  NOT NULL DEFAULT now(),
    date_updated              TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (conversation_id, schedule_id),
    UNIQUE (schedule_id),
    CONSTRAINT agent_wake_schedules_skill_fk
        FOREIGN KEY (skill_id) REFERENCES agent_skills(skill_id)
        ON DELETE SET NULL,
    CONSTRAINT agent_wake_schedules_context_from_fk
        FOREIGN KEY (context_from_schedule_id)
        REFERENCES agent_wake_schedules(schedule_id)
        ON DELETE SET NULL,
    CONSTRAINT agent_wake_schedules_execution_mode_check
        CHECK (execution_mode IN ('inline', 'spawn')),
    CONSTRAINT agent_wake_schedules_status_check
        CHECK (status IN ('active', 'paused', 'expired')),
    CONSTRAINT agent_wake_schedules_missed_fire_policy_check
        CHECK (missed_fire_policy IN ('coalesce', 'catch_up')),
    CONSTRAINT agent_wake_schedules_delivery_target_check
        CHECK (delivery_target IN ('conversation', 'email'))
)
"""


# Partial index on the tick-engine's hot query path
# (``WHERE status = 'active' AND next_fire_at <= now``). Partial cuts
# the index footprint roughly by 1 - (active_count / total_count); for
# wakes that's typically 0.9+ savings because expired one-shots
# accumulate.
_CREATE_NEXT_FIRE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_wake_schedules_next_fire "
    "ON agent_wake_schedules (next_fire_at) "
    "WHERE status = 'active' AND next_fire_at IS NOT NULL"
)


# Partial index for per-conv cap enforcement (count active per conv).
_CREATE_CONV_STATUS_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_wake_schedules_conv_status ON agent_wake_schedules (conversation_id, status)"
)


# Lookup index for user-scoped admin views.
_CREATE_USER_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_wake_schedules_user ON agent_wake_schedules (user_id)"


# Partial index for context_from resolution (sparse column).
_CREATE_CONTEXT_FROM_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_wake_schedules_context_from "
    "ON agent_wake_schedules (context_from_schedule_id) "
    "WHERE context_from_schedule_id IS NOT NULL"
)


async def create_agent_wake_schedules(store: DataStore) -> None:
    """Create ``agent_wake_schedules`` table + indexes.

    :param store: ``DataStore`` bound to the target agent schema via
        ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("creating agent_wake_schedules table (v001)")
    await store.execute(_CREATE_TABLE_SQL)
    await store.execute(_CREATE_NEXT_FIRE_INDEX_SQL)
    await store.execute(_CREATE_CONV_STATUS_INDEX_SQL)
    await store.execute(_CREATE_USER_INDEX_SQL)
    await store.execute(_CREATE_CONTEXT_FROM_INDEX_SQL)
