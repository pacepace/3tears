"""
agent-workspace v003: backfill platform.namespaces for every workspace row.

workspace-task-19 (WS-ACL-11) makes every workspace a platform-level
namespace via shared primary key: ``workspaces.id == namespaces.id``
and ``namespaces.namespace_type = 'workspace'``. v003 heals existing
per-agent schemas so every row in ``<agent_schema>.workspaces`` is
paired with a matching row in ``platform.namespaces``.

**Scope:** this runs in the per-agent schema via search_path. it is
one of the rare agent-scope migrations that deliberately writes into
the platform schema — a one-time data translation, not a cross-package
reach in the "hot path" sense the migration guide warns against.
future ``workspace_create`` flows insert the namespace row inside the
same transaction as the workspace row; this migration heals pre-task-19
history so both inserts converge on the same invariant.

**Per-schema identity:** ``current_schema()`` returns the agent's own
schema name because the L3 broker binds ``search_path`` before the
runner calls this function. we stamp that value into
``namespaces.schema_name`` so non-owner queries against a shared
workspace route back to this agent via the broker's namespace machinery.

**Customer linkage:** the v014 platform migration populates
``agents.customer_id`` as a hard NOT NULL. we join through
``platform.agents`` to resolve the customer every namespace row carries.

**Namespace name:** the ``name`` column on ``platform.namespaces`` is
UNIQUE. we avoid collisions with existing ``agent.<uuid>`` names by
prefixing with ``workspace.``: the full namespace name becomes
``workspace.<workspace_uuid>``. this is the form workspace tools will
ask the broker to route against.

**Idempotency:** the ``INSERT ... ON CONFLICT (id) DO NOTHING`` clause
guarantees replay safety. if a future ``workspace_create`` has already
inserted the namespace row for a given workspace, this migration skips
it silently. the migration also self-skips workspaces whose customer
cannot be resolved via ``agents.customer_id`` (should not happen after
v014, but the WHERE-clause avoids NULL-constraint violations on any
edge case).
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["workspace_namespace_backfill"]

log = get_logger(__name__)


_BACKFILL_NAMESPACES_FROM_WORKSPACES_SQL = """
INSERT INTO platform.namespaces (
    id,
    name,
    namespace_type,
    owner_agent_id,
    schema_name,
    customer_id,
    metadata,
    date_created,
    date_updated
)
SELECT
    w.id,
    'workspace.' || w.id::text AS name,
    'workspace' AS namespace_type,
    w.agent_id AS owner_agent_id,
    current_schema() AS schema_name,
    a.customer_id,
    jsonb_build_object('workspace_name', w.name) AS metadata,
    w.date_created,
    w.date_updated
  FROM workspaces w
  JOIN platform.agents a ON a.id = w.agent_id
 WHERE w.date_deleted IS NULL
ON CONFLICT (id) DO NOTHING
"""


async def workspace_namespace_backfill(store: DataStore) -> None:
    """
    insert one platform.namespaces row per live workspace in this schema.

    writes into ``platform.namespaces`` from the per-agent schema bound
    to ``search_path``. idempotent via ``ON CONFLICT (id) DO NOTHING`` so
    replay during recovery — or concurrent ``workspace_create`` paths
    that landed the namespace row first — is safe.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    """
    log.info("backfilling platform.namespaces rows for live workspaces")
    await store.execute(_BACKFILL_NAMESPACES_FROM_WORKSPACES_SQL)
