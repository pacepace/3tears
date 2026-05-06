"""mcp v001: create ``mcp_tool_grants`` table.

DDL is unqualified so the caller's ``search_path`` governs which
schema gets the table. every statement is idempotent so replay on
recovery is safe.

shape:

- ``grant_id UUID PRIMARY KEY`` -- surrogate PK, addressable for
  revocation (``DELETE WHERE grant_id = $1``)
- ``principal_type TEXT NOT NULL`` -- ``"user"`` / ``"group"`` /
  ``"role"`` (no enum constraint at the DB layer; the framework
  enforces the literal at the Python border)
- ``principal_id UUID NOT NULL`` -- the user / group / role UUID
- ``tool_name TEXT NOT NULL`` -- target tool name (matches
  :attr:`McpTool.name`)
- ``permission TEXT NOT NULL`` -- platform-format permission string
- ``date_created TIMESTAMPTZ NOT NULL DEFAULT now()`` -- aware-UTC

mutations to this table MUST be followed by a bump of
:func:`Subjects.mcp_rbac_epoch` so sibling pods reload their
authorizer cache. that bump is the caller's responsibility (REST
endpoint, admin script, etc.) -- the migration does not foreclose
the bump pattern beyond providing the row store.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "create_mcp_tool_grants_table",
]

log = get_logger(__name__)


_CREATE_MCP_TOOL_GRANTS_SQL = """
CREATE TABLE IF NOT EXISTS mcp_tool_grants (
    grant_id UUID PRIMARY KEY,
    principal_type TEXT NOT NULL,
    principal_id UUID NOT NULL,
    tool_name TEXT NOT NULL,
    permission TEXT NOT NULL,
    date_created TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

_CREATE_PRINCIPAL_LOOKUP_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_mcp_tool_grants_principal "
    "ON mcp_tool_grants (principal_id, permission)"
)

_CREATE_TOOL_LOOKUP_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_mcp_tool_grants_tool "
    "ON mcp_tool_grants (tool_name)"
)


async def create_mcp_tool_grants_table(store: DataStore) -> None:
    """create the ``mcp_tool_grants`` table and lookup indexes.

    :param store: DataStore bound to platform schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("creating mcp_tool_grants table")
    await store.execute(_CREATE_MCP_TOOL_GRANTS_SQL)
    await store.execute(_CREATE_PRINCIPAL_LOOKUP_INDEX_SQL)
    await store.execute(_CREATE_TOOL_LOOKUP_INDEX_SQL)
