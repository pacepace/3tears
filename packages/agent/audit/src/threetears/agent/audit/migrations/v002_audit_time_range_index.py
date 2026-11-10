"""
agent-audit v002: retention's index range-sharded, built beside the old one, which is then dropped.

``idx_audit_events_time`` (v001) indexes ``timestamp`` with no order declared, so YugabyteDB hashes it: the prune's
``timestamp < $1 LIMIT $2`` seeks no range there and reads every event, a table that only grows between prunes.
``idx_audit_events_time_asc`` declares the column ``ASC``, so YugabyteDB range-shards it and a batch reads about its
own rows; Postgres reads both the same way. The new index is built first, so the prune always has one, then the old
is dropped. Idempotent: every statement checks first.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["AUDIT_EVENTS_V002_DDL", "range_shard_audit_time_index"]

log = get_logger(__name__)

#: The statements, frozen with this version, in order: the new index before the old one goes.
AUDIT_EVENTS_V002_DDL: tuple[str, ...] = (
    "-- purpose: packages/agent/audit/src/threetears/agent/audit/persist.py::prune_audit_events\n"
    "-- plan_test: packages/agent/audit/tests/integration/test_index_plans.py::test_audit_events_time\n"
    "-- reason: audit_events carries four indexes: retention's prune, an erasure by actor, and two the hub's views read\n"
    "CREATE INDEX IF NOT EXISTS idx_audit_events_time_asc ON audit_events (timestamp ASC)",
    "DROP INDEX IF EXISTS idx_audit_events_time",
)


async def range_shard_audit_time_index(store: DataStore) -> None:
    """Build ``idx_audit_events_time_asc`` and drop ``idx_audit_events_time``.

    :param store: the migration's data store
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    for statement in AUDIT_EVENTS_V002_DDL:
        await store.execute(statement)
    log.info("agent-audit v002: retention's index range-sharded")
