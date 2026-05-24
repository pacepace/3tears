"""agent-tools-platform v001: add tool_eligible + skill_eligible columns.

agent-tools-eligibility shard 01: extend the platform-shared
``namespaces`` table so every ``tool``-type row carries the
authored visibility declaration emitted from the tool's
:class:`~threetears.agent.tools.base_tool.TearsTool` subclass.

DDL is unqualified so the caller's ``search_path`` governs which
schema gets the column. every statement is idempotent (``ADD COLUMN
IF NOT EXISTS``) so replay on recovery is safe.

shape:

- ``tool_eligible BOOLEAN NOT NULL DEFAULT TRUE`` -- whether the
  tool belongs in the agent's default tool surface (the set
  ``list_tool_eligible_tools_for_actor`` returns). Defaults TRUE
  so pre-shard rows keep their pre-shard "always visible" shape.
- ``skill_eligible BOOLEAN NOT NULL DEFAULT FALSE`` -- whether the
  tool is discoverable via the skills catalog
  (``list_skill_eligible_tools``). Defaults FALSE so no
  pre-shard row leaks into the skills catalog without an explicit
  opt-in from the tool's subclass.

eligibility is *visibility* layered over ACL: a tool with
``tool_eligible=False`` is still callable by an actor who holds the
``tool.call`` ACL grant if a skill surfaces it via ``tool_additions``.
The two flags govern default presentation; they never bypass
authorization.

per-customer overrides are explicitly out of scope (shard TE-09);
admin-side per-customer policy lives in the ACL grant layer.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "add_tool_eligibility_columns",
]

log = get_logger(__name__)


_ADD_TOOL_ELIGIBLE_COLUMN_SQL = (
    "ALTER TABLE namespaces ADD COLUMN IF NOT EXISTS tool_eligible BOOLEAN NOT NULL DEFAULT TRUE"
)

_ADD_SKILL_ELIGIBLE_COLUMN_SQL = (
    "ALTER TABLE namespaces ADD COLUMN IF NOT EXISTS skill_eligible BOOLEAN NOT NULL DEFAULT FALSE"
)


async def add_tool_eligibility_columns(store: DataStore) -> None:
    """add ``tool_eligible`` + ``skill_eligible`` BOOLEAN columns.

    :param store: DataStore bound to platform schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("adding tool_eligible + skill_eligible columns to namespaces")
    await store.execute(_ADD_TOOL_ELIGIBLE_COLUMN_SQL)
    await store.execute(_ADD_SKILL_ELIGIBLE_COLUMN_SQL)
