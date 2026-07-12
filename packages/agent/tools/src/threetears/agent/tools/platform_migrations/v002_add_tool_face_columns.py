"""agent-tools-platform v002: add face_api + face_mcp + face_platform_tool columns.

gu-task-02b: extend the platform-shared ``namespaces`` table so every
``tool``-type row carries the authored FACE declaration emitted from the
tool's :class:`~threetears.agent.tools.base_tool.TearsTool` subclass
(the ``face_*`` cvars gu-task-02 stamped onto every
``ToolManifestEntry``). The hub ``ToolNamespaceEmitter`` reads those
manifest fields and stamps them onto the namespace row via its existing
``ON CONFLICT (id) DO UPDATE`` upsert.

DDL is unqualified so the caller's ``search_path`` governs which schema
gets the columns. every statement is idempotent (``ADD COLUMN IF NOT
EXISTS``) so replay on recovery is safe.

shape:

- ``face_platform_tool BOOLEAN NOT NULL DEFAULT TRUE`` -- whether the
  tool is exposed as a first-class platform tool. Defaults TRUE so
  pre-face rows keep their pre-face "platform tool" shape.
- ``face_api BOOLEAN NOT NULL DEFAULT FALSE`` -- whether the tool is
  exposed on the API face. Defaults FALSE (explicit opt-in).
- ``face_mcp BOOLEAN NOT NULL DEFAULT FALSE`` -- whether the tool is
  exported on the MCP face (the column gu-task-25's MCP export reads).
  Defaults FALSE (explicit opt-in).

the defaults ARE the backwards compatibility: a pre-face namespace row
(and a pre-face manifest) reads as "platform-tool only," identical to
today, with no backfill and no alias.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "add_tool_face_columns",
]

log = get_logger(__name__)


_ADD_FACE_PLATFORM_TOOL_COLUMN_SQL = (
    "ALTER TABLE namespaces ADD COLUMN IF NOT EXISTS face_platform_tool BOOLEAN NOT NULL DEFAULT TRUE"
)

_ADD_FACE_API_COLUMN_SQL = "ALTER TABLE namespaces ADD COLUMN IF NOT EXISTS face_api BOOLEAN NOT NULL DEFAULT FALSE"

_ADD_FACE_MCP_COLUMN_SQL = "ALTER TABLE namespaces ADD COLUMN IF NOT EXISTS face_mcp BOOLEAN NOT NULL DEFAULT FALSE"


async def add_tool_face_columns(store: DataStore) -> None:
    """add ``face_platform_tool`` + ``face_api`` + ``face_mcp`` BOOLEAN columns.

    :param store: DataStore bound to platform schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("adding face_platform_tool + face_api + face_mcp columns to namespaces")
    await store.execute(_ADD_FACE_PLATFORM_TOOL_COLUMN_SQL)
    await store.execute(_ADD_FACE_API_COLUMN_SQL)
    await store.execute(_ADD_FACE_MCP_COLUMN_SQL)
