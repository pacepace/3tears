"""agent-wake v007: agent-scoped lookup, protected wakes, fire links, the Life gate.

- ``agent_wake_schedules.protected`` (``BOOLEAN NOT NULL DEFAULT false``)
  and a trigger that makes a protected wake the exception in storage: it
  cannot be deleted, paused, expired, retyped or unprotected, and its
  ``schedule_config`` changes only inside a transaction that opened the
  gate. :func:`~threetears.agent.wake.protected.update_protected` and
  :func:`~threetears.agent.wake.protected.delete_protected` open it with
  ``set_config('threetears.wake_protected_gate', ..., true)``, which lasts
  until the transaction ends. A CHECK keeps a protected wake off the two
  one-shot types, whose single fire would expire it.
- ``idx_wake_schedules_agent_status`` and ``idx_webhook_subs_agent``
  serve the agent-scoped lookups, which span conversations.
- ``wake_fires.started_conversation_id``: the conversation a fire started.
  Set in the transaction that creates that conversation, before the model
  runs, so a fire the reaper later fails is still linked to it. No FK, for
  the reason v001 gives for ``conversation_id``.
- ``wake_fires.status`` accepts ``'skipped_life_off'``: the consumer's
  permit callback answered "not now".
- ``execution_mode`` defaults to ``'spawn'``: every fire starts a new
  conversation, and ``'inline'`` is accepted only on rows written before
  this version, until their consumer moves them.

Every statement is guarded, so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "PROTECTED_GATE_SETTING",
    "agent_scope_protection_and_fire_links",
]

log = get_logger(__name__)


#: The transaction-local setting the protection trigger reads.
PROTECTED_GATE_SETTING = "threetears.wake_protected_gate"


_ADD_PROTECTED_SQL = """
ALTER TABLE agent_wake_schedules
    ADD COLUMN IF NOT EXISTS protected BOOLEAN NOT NULL DEFAULT false
"""

_ADD_PROTECTED_TYPE_CHECK_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        WHERE c.conname = 'agent_wake_schedules_protected_type_check'
          AND t.relname = 'agent_wake_schedules'
          AND n.nspname = current_schema()
    ) THEN
        ALTER TABLE agent_wake_schedules
            ADD CONSTRAINT agent_wake_schedules_protected_type_check
            CHECK (NOT protected OR schedule_type NOT IN ('one_shot_at', 'relative_delay'));
    END IF;
END
$$
"""

_GUARD_FUNCTION_SQL = f"""
CREATE OR REPLACE FUNCTION agent_wake_schedules_guard_protected() RETURNS trigger AS $$
DECLARE
    gate text := coalesce(current_setting('{PROTECTED_GATE_SETTING}', true), '');
BEGIN
    IF TG_OP = 'DELETE' THEN
        IF OLD.protected AND gate <> 'delete' THEN
            RAISE EXCEPTION 'wake % is protected and cannot be deleted', OLD.schedule_id
                USING ERRCODE = 'check_violation';
        END IF;
        RETURN OLD;
    END IF;
    IF NEW.protected IS DISTINCT FROM OLD.protected THEN
        RAISE EXCEPTION 'wake %: protected is set when a wake is created and never changes', OLD.schedule_id
            USING ERRCODE = 'check_violation';
    END IF;
    IF OLD.protected THEN
        IF NEW.status <> 'active' THEN
            RAISE EXCEPTION 'wake % is protected and cannot be paused or expired', OLD.schedule_id
                USING ERRCODE = 'check_violation';
        END IF;
        IF NEW.schedule_type <> OLD.schedule_type THEN
            RAISE EXCEPTION 'wake % is protected and cannot change type', OLD.schedule_id
                USING ERRCODE = 'check_violation';
        END IF;
        IF NEW.schedule_config IS DISTINCT FROM OLD.schedule_config AND gate <> 'update' THEN
            RAISE EXCEPTION 'wake % is protected; its schedule changes only through update_protected', OLD.schedule_id
                USING ERRCODE = 'check_violation';
        END IF;
    END IF;
    RETURN NEW;
END
$$ LANGUAGE plpgsql
"""

_GUARD_TRIGGER_SQL = """
CREATE OR REPLACE TRIGGER agent_wake_schedules_guard_protected
    BEFORE UPDATE OR DELETE ON agent_wake_schedules
    FOR EACH ROW EXECUTE FUNCTION agent_wake_schedules_guard_protected()
"""

_AGENT_STATUS_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_wake_schedules_agent_status ON agent_wake_schedules (agent_id, status)"
)

_SUBSCRIPTION_AGENT_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_webhook_subs_agent ON webhook_subscriptions (agent_id)"

_ADD_STARTED_CONVERSATION_SQL = """
ALTER TABLE wake_fires
    ADD COLUMN IF NOT EXISTS started_conversation_id UUID
"""

_STARTED_CONVERSATION_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_wake_fires_started_conversation "
    "ON wake_fires (started_conversation_id) WHERE started_conversation_id IS NOT NULL"
)

# PostgreSQL has no ALTER CONSTRAINT for a CHECK predicate; drop and re-add
# under the same name, as v004 did.
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
            'failed'
        ))
"""

_SPAWN_DEFAULTS_SQL = (
    "ALTER TABLE agent_wake_schedules ALTER COLUMN execution_mode SET DEFAULT 'spawn'",
    "ALTER TABLE webhook_subscriptions ALTER COLUMN execution_mode SET DEFAULT 'spawn'",
)


async def agent_scope_protection_and_fire_links(store: DataStore) -> None:
    """Add protection, the agent indexes, the fire link and the Life-gate status.

    :param store: ``DataStore`` bound to the target agent schema via
        ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("agent_wake v007: protected wakes, agent indexes, fire links, skipped_life_off")
    await store.execute(_ADD_PROTECTED_SQL)
    await store.execute(_ADD_PROTECTED_TYPE_CHECK_SQL)
    await store.execute(_GUARD_FUNCTION_SQL)
    await store.execute(_GUARD_TRIGGER_SQL)
    await store.execute(_AGENT_STATUS_INDEX_SQL)
    await store.execute(_SUBSCRIPTION_AGENT_INDEX_SQL)
    await store.execute(_ADD_STARTED_CONVERSATION_SQL)
    await store.execute(_STARTED_CONVERSATION_INDEX_SQL)
    await store.execute(_DROP_STATUS_CHECK_SQL)
    await store.execute(_ADD_STATUS_CHECK_SQL)
    for statement in _SPAWN_DEFAULTS_SQL:
        await store.execute(statement)
