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
    "customer_scope_visibility_clause",
    "three_scope_visibility_clause",
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


def customer_scope_visibility_clause(
    *,
    user_id: UUID,
    customer_id_column: str,
    param_offset: int = 1,
) -> tuple[str, list[Any]]:
    """build the customer-scope read wrap: platform rows OR caller-visible ones.

    the customer-scope read rule for a row carrying a nullable
    ``customer_id``: the row is visible iff it is platform-scope
    (``customer_id IS NULL``) OR its customer is one the caller's RBAC
    grants admit. the customer-side decision is delegated verbatim to
    :func:`caller_visible_customer_clause` so the EXISTS subquery is never
    re-implemented; this wrap only adds the platform-scope ``OR`` arm.

    admin-review lists where a user-scope row is NOT private to its author
    (e.g. promotion-request lists, where a ``customer_admin`` must see every
    requester's pending row) compose THIS clause; entity lists that DO want
    user-privacy compose :func:`three_scope_visibility_clause` instead.

    :param user_id: caller user UUID (``auth["user_id"]``)
    :ptype user_id: UUID
    :param customer_id_column: SQL identifier for the row's ``customer_id``
        column (e.g. ``"pe.customer_id"``); embedded verbatim, callers pass
        a trusted identifier (never user input)
    :ptype customer_id_column: str
    :param param_offset: 1-based position of the FIRST ``$N`` placeholder
        this fragment allocates; allocates exactly one bind (caller user_id)
    :ptype param_offset: int
    :return: tuple ``(sql_fragment, bind_params)``; ``bind_params`` has
        exactly one element (the caller user_id)
    :rtype: tuple[str, list[Any]]
    """
    customer_clause, params = caller_visible_customer_clause(
        user_id=user_id,
        customer_id_column=customer_id_column,
        param_offset=param_offset,
    )
    fragment = f"({customer_id_column} IS NULL OR ({customer_clause}))"
    return fragment, params


def three_scope_visibility_clause(
    *,
    user_id: UUID,
    customer_id_column: str,
    user_id_column: str,
    param_offset: int = 1,
) -> tuple[str, list[Any]]:
    """build the full three-scope-ladder entity-list read rule (D10).

    a row carrying nullable ``customer_id`` + ``user_id`` (the three-scope
    ladder: platform / customer / user) is visible to the caller iff it
    passes BOTH halves:

    - customer-scope (:func:`customer_scope_visibility_clause`): platform
      row OR caller-visible customer row, and
    - user-privacy: the row is platform/customer-scope (``user_id IS NULL``)
      OR the caller's own (``user_id = caller``) — a user-scope row is
      admitted only to its owner, so a peer in the same customer never sees
      it even though the customer grant admits the row's customer.

    the user predicate REUSES the single bind the customer clause already
    allocates: a user-scope row's owner IS the caller the EXISTS subquery
    binds at ``$param_offset``, so both halves reference the same placeholder
    and the clause returns exactly ONE bind. every user-scopable knowledge
    entity list (playbook entries, concepts, eval cases, and any future
    type) composes this clause; no list site hand-rolls the wrapping.

    :param user_id: caller user UUID (``auth["user_id"]``)
    :ptype user_id: UUID
    :param customer_id_column: SQL identifier for the row's ``customer_id``
        column (e.g. ``"pe.customer_id"``); embedded verbatim
    :ptype customer_id_column: str
    :param user_id_column: SQL identifier for the row's ``user_id`` column
        (e.g. ``"pe.user_id"``); embedded verbatim into the privacy predicate
    :ptype user_id_column: str
    :param param_offset: 1-based position of the FIRST ``$N`` placeholder
        this fragment allocates; allocates exactly one bind (caller user_id),
        referenced by BOTH the EXISTS membership match and the user predicate
    :ptype param_offset: int
    :return: tuple ``(sql_fragment, bind_params)``; ``bind_params`` has
        exactly one element (the caller user_id), reused across both halves
    :rtype: tuple[str, list[Any]]
    """
    customer_scope, params = customer_scope_visibility_clause(
        user_id=user_id,
        customer_id_column=customer_id_column,
        param_offset=param_offset,
    )
    user_predicate = (
        f"({user_id_column} IS NULL OR {user_id_column} = ${param_offset})"
    )
    fragment = f"{customer_scope} AND {user_predicate}"
    return fragment, params
