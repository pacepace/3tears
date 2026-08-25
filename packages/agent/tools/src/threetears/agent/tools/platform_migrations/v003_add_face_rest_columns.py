"""agent-tools-platform v003: add face_rest + face_rest_declaration columns.

extends the platform-shared ``namespaces`` table with the FOURTH reach face a
tool may declare. the first three are plain booleans added by
``v002_add_tool_face_columns`` because their address is derived; REST is not,
so it needs the authored declaration alongside the flag.

the columns ship HERE, in this package, rather than in a consumer: the face
columns are one column family on one table, and splitting them across two
repositories would put half the family on a second release train -- so the
serving shard would block on a 3tears release it does not budget for.

DDL is unqualified so the caller's ``search_path`` governs which schema gets
the columns. every statement is idempotent (``ADD COLUMN IF NOT EXISTS``) so
replay on recovery is safe.

shape:

- ``face_rest BOOLEAN NOT NULL DEFAULT FALSE`` -- whether the tool is served
  as a REST resource. Defaults FALSE (explicit opt-in), matching ``face_api``
  and ``face_mcp``. it is DERIVED, not authored: the manifest carries the
  declaration and nothing else, and the consumer writing this row sets the
  boolean from "is there a declaration". a manifest field beside the
  declaration would be a second place to say the same thing.
- ``face_rest_declaration JSONB`` -- the authored
  :class:`~threetears.agent.tools.http_operation.RestAffordance` (method, path
  template, derived placeholders, cache posture). NULLABLE with no default:
  NULL means "no REST face", which is exactly what ``face_rest FALSE`` means,
  and the two are written together.

the defaults ARE the backwards compatibility: a pre-REST namespace row (and a
pre-REST manifest) reads as "no REST face", identical to today, with no
backfill and no alias.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "add_face_rest_columns",
]

log = get_logger(__name__)


_ADD_FACE_REST_COLUMN_SQL = "ALTER TABLE namespaces ADD COLUMN IF NOT EXISTS face_rest BOOLEAN NOT NULL DEFAULT FALSE"

_ADD_FACE_REST_DECLARATION_COLUMN_SQL = "ALTER TABLE namespaces ADD COLUMN IF NOT EXISTS face_rest_declaration JSONB"


async def add_face_rest_columns(store: DataStore) -> None:
    """add ``face_rest`` BOOLEAN + ``face_rest_declaration`` JSONB columns.

    :param store: DataStore bound to platform schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("adding face_rest + face_rest_declaration columns to namespaces")
    await store.execute(_ADD_FACE_REST_COLUMN_SQL)
    await store.execute(_ADD_FACE_REST_DECLARATION_COLUMN_SQL)
