"""mcp v002 (CONTRACT, held): v001's two lookup indexes, which no read uses, are dropped.

Every read of ``mcp_tool_grants`` is by its key: one grant by ``grant_id``, a page of grants after a ``grant_id``, the
authorizer's cold fill and then each changed grant by key (:class:`~threetears.mcp.rbac.McpToolGrantCollection`).
Nothing reads a principal's grants (``idx_mcp_tool_grants_principal``, ``(principal_id, permission)``) or a tool's
(``idx_mcp_tool_grants_tool``, ``(tool_name)``) through the database. A pod of an older release may still plan through
them mid-rollout, so the step is held: :func:`~threetears.mcp.migrations.register` attaches it only when asked
(``held_contract=True``: the test schemas now, every deployment from the release after this one).

``DROP INDEX IF EXISTS``, so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["drop_lookup_indexes"]

log = get_logger(__name__)

_DROP_PRINCIPAL_INDEX_SQL = "DROP INDEX IF EXISTS idx_mcp_tool_grants_principal"

_DROP_TOOL_INDEX_SQL = "DROP INDEX IF EXISTS idx_mcp_tool_grants_tool"


async def drop_lookup_indexes(store: DataStore) -> None:
    """Drop ``idx_mcp_tool_grants_principal`` and ``idx_mcp_tool_grants_tool`` (v002, contract).

    :param store: DataStore bound to the platform schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_DROP_PRINCIPAL_INDEX_SQL)
    await store.execute(_DROP_TOOL_INDEX_SQL)
    log.info("mcp v002 (contract): the two unread grant lookup indexes dropped")
