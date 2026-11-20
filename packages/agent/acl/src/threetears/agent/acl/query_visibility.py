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
row whose scope admits ``X`` and whose ROLE grants an action where that
scope needs one.

"grants an action on a type" is :func:`threetears.agent.acl.types.permissions_act_on`:
among the actions :meth:`threetears.agent.acl.Role.actions_for` unions from the
type's bucket and the ``*`` wildcard bucket, one that is the type's OWN action
(named ``<type>.``, or a verb naming no type, ``read`` or
``read_file_matching:<glob>``; spelled for SQL by
:func:`threetears.agent.acl.types.action_acts_on_sql`). a bucket's non-emptiness
is not enough: ``{"customer": ["group.read"]}`` holds an action on groups, and bound
on a customer namespace it once opened every list of that customer (Pace's ruling
2026-11-20, option A).

- ``scope_type='all'`` admits every customer, when the role grants an
  action on the LISTED type. a fleet-wide grant reaches every customer's
  rows only of the types its role acts on; without this, a narrow role
  (one action on one system namespace) bound at ``all`` showed its holder
  every customer's rows of every type.
- ``scope_type='type_customer'`` with ``scope_customer_id=X`` AND
  ``scope_namespace_type`` matching the LISTED entity's namespace type
  admits ``X`` for rows of that type only, when the role grants an action
  on that type. a ``type_customer`` grant is scoped to ONE namespace type
  within the customer (see :meth:`threetears.agent.acl.RoleAssignment.covers`);
  dropping the type predicate would let a grant on one type (e.g.
  ``datasource``) expose every other type's rows in the same customer.
- ``scope_type='namespace'`` with ``scope_namespace_id`` whose
  namespace's ``customer_id=X`` admits ``X``, when the role grants an
  action on THAT namespace's own type. a namespace grant is membership in
  its customer: a customer admin's grant on the customer namespace opens
  every list of that customer, which is the admin model the hub's routes
  gate on. a role bound where it grants nothing is no grant, and admits
  nothing.
- ``scope_type='subtree'`` admits NO customer here, deliberately. that
  scope names a node in the namespace-NAME space, and the shapes it is
  written against -- ``tools.*`` rows -- carry ``customer_id IS NULL``,
  so it says nothing about which customer's rows a caller may LIST.
  admitting a customer off a subtree grant would widen row visibility
  on the strength of a grant that never mentioned one.

every arm sits behind the evaluator's cross-customer wall: a grant admits
a row of customer ``X`` only when every membership edge from the caller to
the granting group, the granting group and the role name no customer or
``X`` (the evaluator's ``_filter_memberships`` and ``_walk_assignments``).
visibility is never broader than the evaluator: a customer-B group's grant
on customer A's namespace admits nothing of A, and a fleet-wide grant
reached through customer C's rows admits C's rows alone.

the generator emits an ``EXISTS`` subquery joining
``role_assignments`` + ``roles`` + ``group_members`` + ``groups`` + ``namespaces``,
parameterised by the caller's user_id. composes additively with other WHERE
clauses via the ``param_offset`` knob (caller passes the next $N to allocate).

the subquery's own aliases -- ``ra``, ``vis_role``, ``vis_group``, ``gm`` and ``gm2``...,
``ns``, ``vis_bucket``, ``vis_action`` -- shadow a caller's alias of the same name inside
it, so the caller's ``customer_id_column`` must not name one. the role
join is ``vis_role`` rather than ``r`` because callers alias their own
listed table ``r`` (``dataset_runs r``): a short alias here would turn
``r.customer_id`` into the role's NULL customer and admit nothing.

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

from threetears.agent.acl.types import MAX_GROUP_MEMBERSHIP_DEPTH, WILDCARD_RESOURCE_TYPE, action_acts_on_sql

__all__ = [
    "caller_visible_customer_clause",
    "customer_scope_visibility_clause",
    "three_scope_visibility_clause",
]

log = get_logger(__name__)


def _caller_membership_predicate(param_offset: int, customer_sql: str) -> str:
    """render the caller-reaches-this-group predicate, nesting included.

    a caller reaches a granting group DIRECTLY, or through group nesting
    (group-in-group membership) as deep as the evaluator walks: a chain of
    up to :data:`MAX_GROUP_MEMBERSHIP_DEPTH` groups from the caller's own
    group to the granting one, so ``MAX_GROUP_MEMBERSHIP_DEPTH - 1`` group
    edges. the evaluator resolves nested grants at authorization time, so a
    visibility clause that stopped short of it would hide rows a caller is
    authorized to reach, and one that went past it would show rows the
    caller cannot reach -- the two must name the same rows. one private
    helper so every public clause builder emits the identical predicate and
    cannot drift.

    the predicate walks DOWN from the granting group (the ``gm`` alias the
    callers join on ``ra.group_id``), one nested ``EXISTS`` per edge, each
    keyed on ``group_members.group_id`` -- the leading column of the
    table's primary key -- so every hop is a keyed lookup rather than a
    scan. it is generated from the constant, never written per depth; the
    nested aliases are ``gm2``, ``gm3``, ... so a caller composing the
    clause keeps clear of the ``gm`` prefix. every membership row on the
    chain passes the evaluator's cross-customer wall (:func:`_customer_wall`),
    as each edge of its walk does.

    :param param_offset: the ``$N`` placeholder bound to the caller's user id
    :ptype param_offset: int
    :param customer_sql: the listed row's customer column, a TRUSTED identifier
    :ptype customer_sql: str
    :return: boolean SQL expression over the ``gm`` join alias
    :rtype: str
    """
    return _member_reaches_caller("gm", MAX_GROUP_MEMBERSHIP_DEPTH - 1, param_offset, customer_sql)


def _role_grants_type_predicate(type_sql: str) -> str:
    """render the granting role carries an action of a type's own on it: :func:`permissions_act_on` in SQL.

    over the ``vis_role`` alias (the ``roles`` row joined on ``ra.role_id``): an action under the
    type's bucket, or under the ``*`` wildcard bucket (:meth:`threetears.agent.acl.Role.actions_for`'s
    union), that :func:`action_acts_on_sql` the type -- named for it, or naming no type. the role
    collection decodes a ``permissions`` value stored as a JSON string (a double-encoded write) the
    same as an object, and skips a bucket that is not a list, so the predicate does both; a clause
    that read only the object form would hide rows the evaluator grants. a bucket that is not an array
    is read as an empty one rather than filtered beside it: SQL does not short-circuit ``AND``, and
    ``jsonb_array_elements_text`` raises on a scalar. a JSON ``null`` element reads as ``'null'``, a
    verb naming no type, as :func:`decode_role_permissions` reads it (``'None'``), so the two agree.

    :param type_sql: SQL naming the type: a ``$N`` placeholder bound to the listed
        type, or a trusted column reference such as ``ns.namespace_type``; never user input
    :ptype type_sql: str
    :return: boolean SQL expression over the ``vis_role`` join alias
    :rtype: str
    """
    permissions = "(CASE WHEN jsonb_typeof(vis_role.permissions) = 'string' THEN (vis_role.permissions #>> '{}')::jsonb ELSE vis_role.permissions END)"
    wildcard = WILDCARD_RESOURCE_TYPE.replace("'", "''")
    acts = action_acts_on_sql("COALESCE(vis_action.action, 'null')", type_sql)
    return f"""EXISTS (
                SELECT 1
                  FROM jsonb_each(
                        CASE WHEN jsonb_typeof({permissions}) = 'object' THEN {permissions} ELSE '{{}}'::jsonb END
                       ) vis_bucket
                 CROSS JOIN LATERAL jsonb_array_elements_text(
                        CASE WHEN jsonb_typeof(vis_bucket.value) = 'array' THEN vis_bucket.value ELSE '[]'::jsonb END
                       ) AS vis_action(action)
                 WHERE vis_bucket.key IN ({type_sql}, '{wildcard}')
                   AND {acts}
            )"""


def _customer_wall(alias: str, customer_sql: str) -> str:
    """the row ``alias`` names no customer, or the listed row's: the evaluator's cross-customer wall on one row.

    the evaluator drops a membership row (``_filter_memberships``, on every edge of the walk), a granting group and a
    customer-authored role (``_walk_assignments``) that name a customer other than the namespace's; a row naming none
    (a platform row) passes everywhere. every arm here admits a row only through a namespace of the row's own customer,
    so the namespace's customer is the listed row's.

    :param alias: the joined row's alias: a ``group_members``, ``groups`` or ``roles`` row
    :ptype alias: str
    :param customer_sql: the listed row's customer column, a TRUSTED identifier
    :ptype customer_sql: str
    :return: boolean SQL expression over ``alias``
    :rtype: str
    """
    return f"({alias}.customer_id IS NULL OR {alias}.customer_id = {customer_sql})"


def _member_reaches_caller(alias: str, edges_left: int, param_offset: int, customer_sql: str) -> str:
    """the membership row ``alias`` names the caller, directly or through up to ``edges_left`` group edges, every
    edge inside the cross-customer wall (:func:`_customer_wall`).

    :param alias: the ``group_members`` alias whose row is tested
    :ptype alias: str
    :param edges_left: group edges the walk may still follow below this row
    :ptype edges_left: int
    :param param_offset: the ``$N`` placeholder bound to the caller's user id
    :ptype param_offset: int
    :param customer_sql: the listed row's customer column, a TRUSTED identifier
    :ptype customer_sql: str
    :return: boolean SQL expression over ``alias``
    :rtype: str
    """
    wall = _customer_wall(alias, customer_sql)
    direct = f"({alias}.member_type = 'user' AND {alias}.member_id = ${param_offset})"
    if edges_left <= 0:
        return f"({wall} AND {direct})"
    child = f"gm{MAX_GROUP_MEMBERSHIP_DEPTH - edges_left + 1}"
    nested = _member_reaches_caller(child, edges_left - 1, param_offset, customer_sql)
    return f"""({wall} AND ({direct}
             OR ({alias}.member_type = 'group' AND EXISTS (
                    SELECT 1
                      FROM group_members {child}
                     WHERE {child}.group_id = {alias}.member_id
                       AND {nested}
                ))))"""


def caller_visible_customer_clause(
    *,
    user_id: UUID,
    customer_id_column: str,
    scope_namespace_type: str,
    param_offset: int = 1,
) -> tuple[str, list[Any]]:
    """build a WHERE clause + bind params restricting rows to caller-visible customers.

    composes additively with other WHERE predicates: returns a SQL
    fragment that callers wrap in their full statement, plus the
    parameter list to extend the bind sequence with. the
    ``param_offset`` argument names the FIRST $N placeholder this
    fragment will allocate; the helper allocates exactly TWO binds
    (``$param_offset = caller_user_id`` and
    ``$(param_offset + 1) = scope_namespace_type``).

    rows admitted when the caller has any role assignment whose scope
    covers the row's customer and whose role grants an action of the
    type's own (its type's bucket or the ``*`` bucket, as
    :meth:`Role.actions_for` reads it, each action counted only on the
    type it names or, naming none, any type: :func:`permissions_act_on`)
    where that scope needs one:

    - scope_type='all' (platform admin), role grants on
      ``scope_namespace_type`` -> every customer admitted
    - scope_type='type_customer' AND scope_customer_id = row's
      customer_id AND scope_namespace_type = ``scope_namespace_type``,
      role grants on that type -> that customer admitted for rows of the
      listed type only. a ``type_customer`` grant is scoped to ONE
      namespace type within the customer, so the type predicate keeps a
      grant on one type (e.g. ``datasource``) from exposing rows of every
      OTHER type in the same customer. mirrors :meth:`RoleAssignment.covers`.
    - scope_type='namespace' AND namespace's customer_id matches the
      row's customer_id, role grants on the namespace's own type -> that
      customer admitted

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
    :param scope_namespace_type: namespace type that governs the entity
        being listed (e.g. ``"api_key"``, ``"tool"``, ``"knowledge"``,
        ``"customer"``). bound as a value, NOT embedded; the ``all`` and
        ``type_customer`` arms admit only through a role granting an
        action on this type, and it constrains the ``type_customer`` arm
        so a per-type customer grant admits only rows of its own type.
    :ptype scope_namespace_type: str
    :param param_offset: 1-based position of the FIRST ``$N``
        placeholder this fragment uses. callers building a multi-clause
        WHERE start at 1; callers that have already allocated N binds
        pass ``N + 1``.
    :ptype param_offset: int
    :return: tuple ``(sql_fragment, bind_params)``; the fragment is a
        single boolean SQL expression suitable for use as a complete
        WHERE clause or composition with ``AND`` / ``OR``; bind_params
        is a list with two elements (the caller user_id and the
        namespace type) the caller appends to its own param sequence.
    :rtype: tuple[str, list[Any]]
    """
    # v0.8.0 PK column rename: namespaces.id -> namespaces.namespace_id;
    # the JOIN condition follows.
    listed_grant = _role_grants_type_predicate(f"${param_offset + 1}")
    fragment = f"""EXISTS (
        SELECT 1
          FROM role_assignments ra
          JOIN roles vis_role ON vis_role.role_id = ra.role_id
          JOIN group_members gm ON gm.group_id = ra.group_id
          LEFT JOIN groups vis_group ON vis_group.group_id = ra.group_id
          LEFT JOIN namespaces ns ON ns.namespace_id = ra.scope_namespace_id
         WHERE {_caller_membership_predicate(param_offset, customer_id_column)}
           AND {_customer_wall("vis_group", customer_id_column)}
           AND {_customer_wall("vis_role", customer_id_column)}
           AND (
                (ra.scope_type = 'all'
                 AND {listed_grant})
             OR (ra.scope_type = 'type_customer'
                 AND ra.scope_customer_id = {customer_id_column}
                 AND ra.scope_namespace_type = ${param_offset + 1}
                 AND {listed_grant})
             OR (ra.scope_type = 'namespace'
                 AND ns.customer_id = {customer_id_column}
                 AND {_role_grants_type_predicate("ns.namespace_type")})
           )
    )"""
    return fragment, [user_id, scope_namespace_type]


def customer_scope_visibility_clause(
    *,
    user_id: UUID,
    customer_id_column: str,
    scope_namespace_type: str,
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
    :param scope_namespace_type: namespace type governing the listed
        entity; forwarded to :func:`caller_visible_customer_clause` so the
        ``type_customer`` arm is type-constrained (e.g. ``"knowledge"``)
    :ptype scope_namespace_type: str
    :param param_offset: 1-based position of the FIRST ``$N`` placeholder
        this fragment allocates; allocates two binds (caller user_id and
        namespace type)
    :ptype param_offset: int
    :return: tuple ``(sql_fragment, bind_params)``; ``bind_params`` has
        two elements (the caller user_id and the namespace type)
    :rtype: tuple[str, list[Any]]
    """
    customer_clause, params = caller_visible_customer_clause(
        user_id=user_id,
        customer_id_column=customer_id_column,
        scope_namespace_type=scope_namespace_type,
        param_offset=param_offset,
    )
    fragment = f"({customer_id_column} IS NULL OR ({customer_clause}))"
    return fragment, params


def three_scope_visibility_clause(
    *,
    user_id: UUID,
    customer_id_column: str,
    user_id_column: str,
    scope_namespace_type: str,
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

    the user predicate REUSES the caller-user bind the customer clause
    allocates at ``$param_offset``: a user-scope row's owner IS the caller
    the EXISTS subquery binds there, so both halves reference the same
    placeholder. the customer clause additionally binds the namespace type
    at ``$(param_offset + 1)``, so the clause returns TWO binds. every
    user-scopable knowledge entity list (playbook entries, concepts, eval
    cases, and any future type) composes this clause; no list site
    hand-rolls the wrapping.

    :param user_id: caller user UUID (``auth["user_id"]``)
    :ptype user_id: UUID
    :param customer_id_column: SQL identifier for the row's ``customer_id``
        column (e.g. ``"pe.customer_id"``); embedded verbatim
    :ptype customer_id_column: str
    :param user_id_column: SQL identifier for the row's ``user_id`` column
        (e.g. ``"pe.user_id"``); embedded verbatim into the privacy predicate
    :ptype user_id_column: str
    :param scope_namespace_type: namespace type governing the listed
        entity; forwarded to the customer clause so the ``type_customer``
        arm is type-constrained (e.g. ``"knowledge"``)
    :ptype scope_namespace_type: str
    :param param_offset: 1-based position of the FIRST ``$N`` placeholder
        this fragment allocates; allocates two binds (caller user_id at
        ``$param_offset``, referenced by BOTH the EXISTS membership match
        and the user predicate; namespace type at ``$(param_offset + 1)``)
    :ptype param_offset: int
    :return: tuple ``(sql_fragment, bind_params)``; ``bind_params`` has
        two elements (the caller user_id, reused across both halves, and
        the namespace type)
    :rtype: tuple[str, list[Any]]
    """
    customer_scope, params = customer_scope_visibility_clause(
        user_id=user_id,
        customer_id_column=customer_id_column,
        scope_namespace_type=scope_namespace_type,
        param_offset=param_offset,
    )
    user_predicate = f"({user_id_column} IS NULL OR {user_id_column} = ${param_offset})"
    fragment = f"{customer_scope} AND {user_predicate}"
    return fragment, params
