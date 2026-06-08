"""SQL WHERE clause generator for caller-visibility filtering (closes #37).

every list endpoint that previously branched on caller type to choose
``customer_id`` query parameters now drives a single SQL query whose
WHERE clause filters rows by the caller's RBAC grants. endpoints stop
knowing about super_admin / customer_admin / scope branching entirely;
the SQL fragment generated here encodes the visibility decision
declaratively.

architecture:

a row at customer_id ``X`` is visible to caller user ``$U`` iff
``$U`` is a member of any group with at least one ``role_assignments``
row whose scope admits ``X``:

- ``scope_type='all'`` admits every customer
- ``scope_type='type_customer'`` with ``scope_customer_id=X`` admits ``X``
- ``scope_type='namespace'`` with ``scope_namespace_id`` whose
  namespace's ``customer_id=X`` admits ``X``

the generator emits an ``EXISTS`` subquery joining
``role_assignments`` + ``group_members`` + ``namespaces``, parameterised
by the caller's user_id. composes additively with other WHERE clauses
via the ``param_offset`` knob (caller passes the next $N to allocate).

the result IS a cache-bypass per CLAUDE.md cache primitive rule -- the
generator builds raw SQL the consuming collection runs through its
``l3_pool.fetch`` directly. the visibility evaluation cannot be
expressed through a by-pk Collection ``get`` because (a) it is
inherently a JOIN across rbac tables and (b) the L1 SQLite mirror
does not carry ``role_assignments`` / ``group_members`` rows.

callers append the marker comment ``# cache-bypass: rbac-visibility
filtered list``  on every site invoking this helper so the cache-
primitive enforcement walker recognizes the pattern.

shared across the hub broker and every agent pod: this is security SQL,
and a verbatim duplicate across the repo boundary is a data-leak drift
hazard, so the hub and the agent SDK import this ONE copy.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from threetears.observe import get_logger

__all__ = [
    "caller_visible_customer_clause",
]

log = get_logger(__name__)


def caller_visible_customer_clause(
    *,
    user_id: UUID,
    customer_id_column: str,
    param_offset: int = 1,
) -> tuple[str, list[Any]]:
    """build a WHERE clause + bind params restricting rows to caller-visible customers.

    composes additively with other WHERE predicates: returns a SQL
    fragment that callers wrap in their full statement, plus the
    parameter list to extend the bind sequence with. the
    ``param_offset`` argument names the FIRST $N placeholder this
    fragment will allocate; the helper allocates exactly one bind
    (``$param_offset = caller_user_id``).

    rows admitted when the caller has any role assignment whose
    scope covers the row's customer:

    - scope_type='all' (platform admin) -> every customer admitted
    - scope_type='type_customer' AND scope_customer_id = row's
      customer_id -> that customer admitted
    - scope_type='namespace' AND namespace's customer_id matches the
      row's customer_id -> that customer admitted

    a customer admin assigned to multiple customers' admin namespaces
    sees ALL their authorized customers' rows (not just their JWT
    home customer); platform admins see every customer's rows.

    :param user_id: caller user UUID (``auth["user_id"]``)
    :ptype user_id: UUID
    :param customer_id_column: SQL identifier for the row's
        ``customer_id`` column from the caller's perspective. examples:
        ``"groups.customer_id"``, ``"g.customer_id"``,
        ``"api_keys.customer_id"``. embedded verbatim in the EXISTS
        subquery so callers must pass an identifier they trust (no
        user input).
    :ptype customer_id_column: str
    :param param_offset: 1-based position of the FIRST ``$N``
        placeholder this fragment uses. callers building a multi-clause
        WHERE start at 1; callers that have already allocated N binds
        pass ``N + 1``.
    :ptype param_offset: int
    :return: tuple ``(sql_fragment, bind_params)``; the fragment is a
        single boolean SQL expression suitable for use as a complete
        WHERE clause or composition with ``AND`` / ``OR``; bind_params
        is a list with exactly one element (the caller user_id) the
        caller appends to its own param sequence.
    :rtype: tuple[str, list[Any]]
    """
    # v0.8.0 PK column rename: namespaces.id -> namespaces.namespace_id;
    # the JOIN condition follows.
    fragment = f"""EXISTS (
        SELECT 1
          FROM role_assignments ra
          JOIN group_members gm ON gm.group_id = ra.group_id
          LEFT JOIN namespaces ns ON ns.namespace_id = ra.scope_namespace_id
         WHERE gm.member_type = 'user'
           AND gm.member_id = ${param_offset}
           AND (
                ra.scope_type = 'all'
             OR (ra.scope_type = 'type_customer'
                 AND ra.scope_customer_id = {customer_id_column})
             OR (ra.scope_type = 'namespace'
                 AND ns.customer_id = {customer_id_column})
           )
    )"""
    return fragment, [user_id]
