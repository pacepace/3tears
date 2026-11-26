"""agent-acl v010: each page of a stamped acl table reads a live-only index shaped like the page.

Append-only L3: a delete marks a row gone in place (v006, ``ROW_STAMPED``), and the row stays until quiet-time cleanup
removes it. A shared keyed read of a stamped table answers live rows alone (``threetears.core.keyed.live_read`` adds
``NOT gone``), so a page walked through the primary key read every gone row between its cursor and its limit, and
dropped it: a group's memberships after a churn of adds and removes, every grant a backfill pages. Each page now has an
index holding live rows only, ordered as it pages (no-whole-reads, coordinator 2026-11-26: the live guard is a filter
a page may walk past, fixed by an index, never allowed):

- ``group_members_live_by_group`` (group_id, id ASC) WHERE NOT gone: a group's members a page at a time
  (``GroupMemberCollection.page_by_group``) and a group delete's cascade read (``GroupCollection.read_cascade``);
- ``groups_live_by_id`` (group_id ASC) WHERE NOT gone: every group, across both row scopes, by id
  (``GroupCollection.page_all``);
- ``role_assignments_live_by_id`` (assignment_id ASC) WHERE NOT gone: every grant by id, the customer backfill's pages
  (``_customer_backfill_plan``).

The ordered column is declared ``ASC``, so YugabyteDB range-shards it and a page reads from its cursor. Built beside the
tables (migrations run outside a transaction; YugabyteDB builds an index on a populated table online). A deployment
whose own migrations create these tables (the hub) runs no agent_acl migration: it creates the same indexes, by these
names, in a migration of its own, so 3tears' hints name an index its schema holds. Idempotent; forward-only.
"""

from __future__ import annotations

from typing import Final

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["live_pages_read_live_indexes"]

log = get_logger(__name__)

#: every statement, in order, each safe to run twice (FROZEN)
_LIVE_PAGE_INDEXES_DDL: Final[tuple[str, ...]] = (
    "-- purpose: packages/agent/acl/src/threetears/agent/acl/collections.py::GroupMemberCollection.page_by_group;"
    " packages/agent/acl/src/threetears/agent/acl/collections.py::GroupCollection.read_cascade\n"
    "-- plan_test: packages/agent/acl/tests/integration/test_index_plans.py::test_group_members_live_page\n"
    "-- reason: a group's live members a page at a time, never through the gone memberships a churn leaves until"
    " cleanup\n"
    "CREATE INDEX IF NOT EXISTS group_members_live_by_group ON group_members (group_id, id ASC) WHERE NOT gone",
    "-- purpose: packages/agent/acl/src/threetears/agent/acl/collections.py::GroupCollection.page_all\n"
    "-- plan_test: packages/agent/acl/tests/integration/test_index_plans.py::test_groups_live_page\n"
    "-- reason: every live group by id, an operator's listing, never through the deleted groups until cleanup\n"
    "CREATE INDEX IF NOT EXISTS groups_live_by_id ON groups (group_id ASC) WHERE NOT gone",
    "-- purpose: packages/agent/acl/src/threetears/agent/acl/collections.py::_customer_backfill_plan\n"
    "-- plan_test: packages/agent/acl/tests/integration/test_index_plans.py::test_role_assignments_live_page\n"
    "-- reason: every live grant by id, the customer backfill's pages, never through the revoked grants until"
    " cleanup\n"
    "CREATE INDEX IF NOT EXISTS role_assignments_live_by_id ON role_assignments (assignment_id ASC) WHERE NOT gone",
)


async def live_pages_read_live_indexes(store: DataStore) -> None:
    """create the live-only page indexes of the stamped acl tables (v010).

    :param store: DataStore bound to the platform schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    for statement in _LIVE_PAGE_INDEXES_DDL:
        await store.execute(statement)
    log.info("agent-acl v010: the group, membership and grant pages read live-only indexes")
