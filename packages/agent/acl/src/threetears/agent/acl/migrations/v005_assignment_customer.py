"""
agent-acl v005: a grant records its customer (the expand step).

``role_assignments`` gains ``customer_id UUID``: the customer whose grant the row is, set by every writer from this
release (``RoleAssignmentCollection.create`` derives it from the scope; the ensure reads a namespace grant's
namespace's customer by key): a ``namespace`` grant's namespace's customer (NULL for a platform namespace), a
``type_customer`` grant's ``scope_customer_id``, NULL for ``all`` and ``subtree``.

This is the expand step of a rolling upgrade (expand / migrate / contract). The column is nullable and unchecked:
while a deployment rolls, a pod of the previous release still writes grants without it, and a CHECK would refuse
them. The grants written before it are filled by
:meth:`~threetears.agent.acl.collections.RoleAssignmentCollection.backfill_customers`, page by page by the grant's
key, which the deploying application runs as an operator step after every pod runs this release. The contract
release, once a whole walk writes nothing, holds the column to the grant's scope by a CHECK and moves readers onto
it.

``ADD COLUMN IF NOT EXISTS``: a replay adds nothing. DDL only (YugabyteDB's DDL/DML separation). Forward-only:
3tears migrations do not declare downgrades.

Revision ID: 005
"""

from __future__ import annotations

from typing import Final

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["add_assignment_customer"]

log = get_logger(__name__)

_ADD_COLUMN_SQL: Final[str] = "ALTER TABLE role_assignments ADD COLUMN IF NOT EXISTS customer_id UUID"


async def add_assignment_customer(store: DataStore) -> None:
    """add ``role_assignments.customer_id`` (nullable, unchecked: the expand step).

    :param store: the migration's data store, bound to the platform schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_ADD_COLUMN_SQL)
    log.info("agent-acl v005: role_assignments.customer_id added; RoleAssignmentCollection.backfill_customers fills it")
