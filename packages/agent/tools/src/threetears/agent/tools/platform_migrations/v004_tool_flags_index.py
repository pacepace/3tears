"""agent-tools-platform v004: index the tool namespaces a catalog flag shows.

The default tool surface and the skills catalog each list the tool namespaces whose flag is set
(:meth:`~threetears.agent.acl.collections.NamespaceCollection.list_tool_eligible_namespaces` and
``list_skill_eligible_tool_namespaces``), one written-out statement per flag column. Without an index each
read every platform and customer namespace, agents and workspaces included. This one keys them by
``namespace_type`` over the rows either flag holds, so both statements read the flagged tools alone
(a statement's ``tool_eligible = TRUE`` implies the index's ``tool_eligible OR skill_eligible``).

DDL is unqualified so the caller's ``search_path`` governs which schema gets the index. Idempotent
(``IF NOT EXISTS``), so replay on recovery is safe.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["add_tool_flags_index"]

log = get_logger(__name__)

_TOOL_FLAGS_INDEX_SQL = (
    "-- purpose: packages/agent/acl/src/threetears/agent/acl/collections.py::TOOL_ELIGIBLE_NAMESPACES_SQL;"
    " packages/agent/acl/src/threetears/agent/acl/collections.py::SKILL_ELIGIBLE_NAMESPACES_SQL\n"
    "-- plan_test: packages/agent/acl/tests/integration/test_index_plans.py::test_namespaces_tool_flags\n"
    "-- reason: the tool namespaces the default tool surface and the skills catalog list, by their flag\n"
    "CREATE INDEX IF NOT EXISTS idx_namespaces_tool_flags ON namespaces (namespace_type)"
    " WHERE tool_eligible OR skill_eligible"
)


async def add_tool_flags_index(store: DataStore) -> None:
    """build ``idx_namespaces_tool_flags``.

    :param store: DataStore bound to platform schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("indexing the flagged tool namespaces")
    await store.execute(_TOOL_FLAGS_INDEX_SQL)
