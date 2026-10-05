"""
agent-audit v001: the ``audit_events`` table and its four indexes.

These are the statements the persisting consumer has always run; they live
here now, frozen with this version, and
:data:`threetears.agent.audit.persist.AUDIT_EVENTS_DDL` reads them, so the
runner and a consumer that ensures its own table run one definition.
Idempotent: every statement checks first.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["AUDIT_EVENTS_V001_DDL", "create_audit_events"]

log = get_logger(__name__)

#: The statements, frozen with this version. Idempotent (``IF NOT EXISTS``).
AUDIT_EVENTS_V001_DDL: tuple[str, ...] = (
    "CREATE TABLE IF NOT EXISTS audit_events ("
    "id UUID PRIMARY KEY, "
    "timestamp TIMESTAMPTZ NOT NULL, "
    "event_type TEXT NOT NULL, "
    "action TEXT NOT NULL, "
    "outcome TEXT NOT NULL DEFAULT 'success', "
    "actor_user_id UUID, "
    "acting_as_principal_id UUID, "
    "calling_agent_id UUID, "
    "owner_agent_id UUID, "
    "customer_id UUID, "
    "resource_namespace_id UUID, "
    "resource_namespace_type TEXT, "
    "correlation_id UUID NOT NULL, "
    "conversation_id UUID, "
    "details JSONB NOT NULL DEFAULT '{}', "
    "ip_address TEXT)",
    # an existing table (created before a column existed) gains every column the insert names beyond the
    # key and the four required fields, with the CREATE's own type and default
    *(
        f"ALTER TABLE audit_events ADD COLUMN IF NOT EXISTS {column}"
        for column in (
            "outcome TEXT NOT NULL DEFAULT 'success'",
            "actor_user_id UUID",
            "acting_as_principal_id UUID",
            "calling_agent_id UUID",
            "owner_agent_id UUID",
            "customer_id UUID",
            "resource_namespace_id UUID",
            "resource_namespace_type TEXT",
            "conversation_id UUID",
            "details JSONB NOT NULL DEFAULT '{}'",
            "ip_address TEXT",
        )
    ),
    "CREATE INDEX IF NOT EXISTS idx_audit_events_time ON audit_events (timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_audit_events_customer_time ON audit_events (customer_id, timestamp DESC)",
    "CREATE INDEX IF NOT EXISTS idx_audit_events_type ON audit_events (event_type)",
    "CREATE INDEX IF NOT EXISTS idx_audit_events_actor ON audit_events (actor_user_id)",
)


async def create_audit_events(store: DataStore) -> None:
    """Create ``audit_events`` and its indexes.

    :param store: the migration's data store
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    for statement in AUDIT_EVENTS_V001_DDL:
        await store.execute(statement)
    log.info("agent-audit v001: audit_events created")
