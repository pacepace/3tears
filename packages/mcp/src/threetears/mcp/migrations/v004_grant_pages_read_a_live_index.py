"""mcp v004: a page of the tool grants reads a live-only index by ``grant_id``.

Append-only L3: a revoke marks its grant gone in place (v003, ``ROW_STAMPED``), and the row stays until quiet-time
cleanup removes it. A shared keyed read of a stamped table answers live rows alone (``threetears.core.keyed.live_read``
adds ``NOT gone``), so an operator's page of grants walked through the primary key read every revoked grant between its
cursor and its limit, and dropped it. ``mcp_tool_grants_live_by_id`` (grant_id ASC) WHERE NOT gone holds live grants
only, in the page's order (no-whole-reads, coordinator 2026-11-26: the live guard is a filter a page may walk past,
fixed by an index, never allowed); ``ASC``, so YugabyteDB range-shards it and a page reads from its cursor.

Built beside the table (migrations run outside a transaction; YugabyteDB builds an index on a populated table online).
Always attached, as v003 is: it only adds. Idempotent; forward-only.
"""

from __future__ import annotations

from typing import Final

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["grant_pages_read_a_live_index"]

log = get_logger(__name__)

#: the statement, safe to run twice (FROZEN)
_LIVE_PAGE_INDEX_DDL: Final[str] = (
    "-- purpose: packages/mcp/src/threetears/mcp/rbac.py::McpToolGrantCollection.list_grants_page\n"
    "-- plan_test: packages/mcp/tests/integration/test_index_plans.py::test_grants_live_page\n"
    "-- reason: an operator's live grants a page at a time, never through the revoked grants until cleanup\n"
    "CREATE INDEX IF NOT EXISTS mcp_tool_grants_live_by_id ON mcp_tool_grants (grant_id ASC) WHERE NOT gone"
)


async def grant_pages_read_a_live_index(store: DataStore) -> None:
    """create the tool grants' live-only page index (v004).

    :param store: DataStore bound to the platform schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_LIVE_PAGE_INDEX_DDL)
    log.info("mcp v004: a page of the tool grants reads its live-only index")
