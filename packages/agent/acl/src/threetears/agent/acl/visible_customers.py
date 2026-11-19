"""the customers a caller's grants cover, read once: what a listing pages within, instead of an EXISTS a row.

A listing of one entity type used to admit each row through a correlated EXISTS over the caller's grants
(:func:`threetears.agent.acl.caller_visible_customer_clause`), run for every row the page scanned. The customers the
caller can see are resolved once instead, by one statement of keyed lookups: the caller's memberships, the groups
they nest in and those groups' grants (:func:`visible_customers` for the listing's namespace type, the same set the per-row clause
admits), and the page reads its rows by key with ``customer_id = ANY(<those customers>)`` and its
``LIMIT``. A platform-wide grant (``scope_type='all'``) whose role acts on the listed type narrows nothing (its
fragment is ``TRUE``).

**A grant counts only through its role**, as the evaluator reads it (:meth:`threetears.agent.acl.Role.actions_for`,
:func:`threetears.agent.acl.types.permissions_act_on`): the role binding a grant is read beside it, and a grant whose
role grants no action where its scope needs one is no grant. Read without its role, any assignment at scope
``all`` -- a one-action system role, or a role with no action at all -- made every admin listing fleet-wide (the leak
3tears #1276 closed in :func:`~threetears.agent.acl.caller_visible_customer_clause`, which this set must equal).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final
from uuid import UUID

from threetears.core.keyed import AnyOf, Const, Fragment, Joined, Sort, keyset_rows, read_reached
from threetears.core.sql_fragments import quote_identifier

from threetears.agent.acl.types import MAX_GROUP_MEMBERSHIP_DEPTH, decode_role_permissions, permissions_act_on

__all__ = ["VISIBLE_LIST_PAGE", "VisibleCustomers", "visible_customers", "visible_rows"]

#: the rows one statement of a whole visible listing reads (:func:`visible_rows`)
VISIBLE_LIST_PAGE: Final[int] = 500


@dataclass(frozen=True, slots=True)
class VisibleCustomers:
    """the customers a caller's grants cover for one namespace type: every customer, or these.

    :ivar is_all: a platform-wide grant covers every customer
    :ivar customer_ids: the customers covered, when not every one
    """

    is_all: bool
    customer_ids: tuple[UUID, ...]

    @property
    def bound(self) -> list[UUID] | None:
        """the customers as a statement binds them beside a ``$n::uuid[] IS NULL OR col = ANY($n::uuid[])`` literal:
        None (every row) for a platform-wide grant.

        :return: the customers, or None for every customer
        :rtype: list[UUID] | None
        """
        return None if self.is_all else list(self.customer_ids)

    @property
    def sees_none(self) -> bool:
        """whether the grants cover no customer at all: a listing of customers' rows is empty, unread.

        :return: True when no grant covers any customer
        :rtype: bool
        """
        return not self.is_all and not self.customer_ids

    def fragment(self, column: str, *, platform_rows: bool = False) -> Fragment:
        """the visibility fragment a 3tears keyed read takes for ``column`` (``keyset_page(visible=[...])``).

        Always a fragment, so a listing writes its ``visible=`` as a literal list the no-whole-reads walk reads:
        ``TRUE`` for a platform-wide grant (every row), the customers bound as ``ANY``, or, for grants that cover no
        customer, ``FALSE`` (no row; PostgreSQL plans the constant as a one-time filter and reads nothing) -- unless
        platform-shared rows are seen by every caller, when only those pass.

        :param column: the row's customer column, a TRUSTED identifier as the read aliases it (``t."customer_id"``)
        :ptype column: str
        :param platform_rows: a row with no customer (platform-shared) is seen by every caller
        :ptype platform_rows: bool
        :return: the fragment
        :rtype: Fragment
        """
        if self.is_all:
            return _EVERY_ROW
        if not self.customer_ids and not platform_rows:
            return _NO_ROW
        return Fragment(_within(column, "$1", platform_rows=platform_rows), (list(self.customer_ids),))


#: the fragment every row passes: a platform-wide grant narrows nothing
_EVERY_ROW: Final[Fragment] = Fragment("TRUE", ())

#: the fragment no row passes: grants that cover no customer, where platform rows are not seen by all
_NO_ROW: Final[Fragment] = Fragment("FALSE", ())


def _within(column: str, placeholder: str, *, platform_rows: bool) -> str:
    """``column`` is one of the customers bound at ``placeholder``, or, with ``platform_rows``, no customer.

    :param column: the TRUSTED customer column
    :ptype column: str
    :param placeholder: the customers' placeholder
    :ptype placeholder: str
    :param platform_rows: admit a row with no customer
    :ptype platform_rows: bool
    :return: the condition
    :rtype: str
    """
    within = f"{column} = ANY({placeholder}::uuid[])"
    return f"({column} IS NULL OR {within})" if platform_rows else within


#: the walk's discriminators, fixed by this code and written into its statement as literals (:class:`Const`): bound,
#: PostgreSQL never cached the walk's generic plan (costed 510 against the custom 391) and re-planned it every call,
#: ~75-100 us each, 2.3x the hand-written statement on the exclusive runner (Pace's ruling 2026-11-16; IT2 diagnosis).
#: As literals (2026-11-16, local, the speed test 20 runs alone): PostgreSQL 0.99-1.29x the old statement (was
#: 1.43-1.54x locally), YugabyteDB 0.87-0.91x; the plan test test_the_visible_set_caches_its_generic_plan holds it
_USER: Final[Const] = Const("user")
_GROUP: Final[Const] = Const("group")

#: the partitions a namespace is written in: a namespace grant's namespace is read by its id in either
_NAMESPACE_SCOPES: Final[tuple[str, ...]] = ("platform", "customer")


async def visible_customers(pool: Any, *, user_id: UUID, namespace_type: str | None) -> VisibleCustomers:
    """the customers ``user_id``'s grants cover for ``namespace_type``: the caller's groups and their grants, by key, in
    one statement.

    The same set :func:`threetears.agent.acl.caller_visible_customer_clause` admits a row of for that type, each
    grant through the role it binds (:func:`~threetears.agent.acl.types.permissions_act_on`, the evaluator's rule):
    a platform-wide grant whose role acts on the type, a ``type_customer`` grant on the type whose role acts on it, or
    a grant on a namespace of the customer whose role acts on that namespace's own type (a customer admin's grant on
    the customer namespace opens every list of that customer). ``namespace_type`` None (a listing of no one type): a
    platform-wide grant counts when its role grants any action at all, a ``type_customer`` grant when its role acts
    on the grant's own type; a role with no action is no grant anywhere. The groups
    are the caller's own (its memberships, ``idx_group_members_reverse``) and those they nest in, a level each, to
    :data:`MAX_GROUP_MEMBERSHIP_DEPTH` groups from the caller (the evaluator's walk); each reached group's grants are
    looked up by its key (``idx_role_assignments_group_scope``), and a namespace grant's namespace by its id
    (:func:`~threetears.core.keyed.read_reached` with ``depth`` and ``joins``: the walk capped at the nesting and the
    lookups beside it, one round trip).

    WHY one statement (2026-11-15, D5): read a level a read and the grants a read more, the set cost 3 round trips
    where the hand-written reached-CTE it replaced cost one: PostgreSQL, 5,000 groups / 20,000 members / 10,000
    grants, a caller in two groups (one nested), best of 30 interleaved: 1.18-1.26 ms against 0.38-0.42 ms
    (scratchpad d5/lat/pg.log). As one capped walk with its lookups, the same world: PostgreSQL 0.55-0.69 ms against
    the old statement's 0.44-0.51 ms (1.23-1.36x), YugabyteDB 1.64-2.09 ms against 1.44-1.82 ms (1.14-1.15x)
    (scratchpad d5/speed2.log, 2026-11-15). The bar is twice the old statement, held by
    ``packages/agent/acl/tests/integration/test_visible_customers_speed_live.py``.

    :param pool: anything with an asyncpg-shaped ``fetch``
    :ptype pool: Any
    :param user_id: the caller
    :ptype user_id: UUID
    :param namespace_type: the namespace type that governs the listed entity: a ``type_customer`` grant adds its
        customer only on that type; None admits one on any type
    :ptype namespace_type: str | None
    :return: the customers
    :rtype: VisibleCustomers
    """
    # cache-bypass: the caller's memberships, the groups they nest in, and every reached group's grants, by key
    rows = await read_reached(
        pool,
        "group_members",
        start={"member_type": _USER, "member_id": user_id},
        on="member_id",
        to="group_id",
        by={"member_type": _GROUP},
        columns=("group_id",),
        depth=MAX_GROUP_MEMBERSHIP_DEPTH,
        joins=[
            Joined(
                "role_assignments",
                on={"group_id": "group_id"},
                columns={
                    "scope_type": "scope_type",
                    "scope_customer_id": "scope_customer_id",
                    "scope_namespace_type": "scope_namespace_type",
                    "scope_namespace_id": "scope_namespace_id",
                    "role_id": "role_id",
                },
                left=True,
            ),
            # the role each grant binds, by its key: a grant counts only where its role acts
            Joined("roles", on={"role_id": "role_id"}, columns={"role_permissions": "permissions"}, left=True),
            Joined(
                "namespaces",
                on={"namespace_id": "scope_namespace_id"},
                by={"row_scope": AnyOf(_NAMESPACE_SCOPES, cast=None)},
                columns={"namespace_customer_id": "customer_id", "namespace_type": "namespace_type"},
                left=True,
            ),
        ],
    )
    is_all = False
    customers: set[UUID] = set()
    for row in rows:
        scope = row["scope_type"]
        if scope is None:
            continue
        permissions = decode_role_permissions(row["role_permissions"])
        if scope == "all":
            is_all = is_all or permissions_act_on(permissions, namespace_type)
        elif scope == "type_customer":
            granted = row["scope_namespace_type"]
            if (
                row["scope_customer_id"] is not None
                and granted is not None
                and (namespace_type is None or granted == namespace_type)
                and permissions_act_on(permissions, granted)
            ):
                customers.add(row["scope_customer_id"])
        elif scope == "namespace":
            if row["namespace_customer_id"] is not None and permissions_act_on(permissions, row["namespace_type"]):
                customers.add(row["namespace_customer_id"])
    return VisibleCustomers(is_all=is_all, customer_ids=tuple(sorted(customers, key=str)))


async def visible_rows(
    conn: Any,
    table: str,
    *,
    columns: Sequence[str],
    user_id: UUID,
    namespace_type: str,
    customer_column: str,
    order: Sequence[Sort],
    user_column: str | None = None,
    where: Sequence[Fragment] = (),
    page: int = VISIBLE_LIST_PAGE,
) -> list[dict[str, Any]]:
    """every row of ``table`` the caller sees, in ``order``: a whole listing read a page at a time, by key.

    The caller's customers are read once (:func:`visible_customers`); the rows are then read page after page through
    :func:`threetears.core.keyed.keyset_rows`, each statement one page of ``page`` rows past the last one's row,
    filtered on those customers. A platform row (no customer) is seen by every caller, as
    :func:`threetears.agent.acl.customer_scope_visibility_clause` admits it. With ``user_column``, a row owned by
    another user is not seen, whatever the grant (:func:`threetears.agent.acl.three_scope_visibility_clause`'s
    privacy half).

    :param conn: anything with an asyncpg-shaped ``fetch`` and ``fetchrow``
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param columns: the columns read, TRUSTED identifiers, each sort column among them
    :ptype columns: Sequence[str]
    :param user_id: the caller
    :ptype user_id: UUID
    :param namespace_type: the namespace type that governs the listed entity
    :ptype namespace_type: str
    :param customer_column: the row's customer column, a TRUSTED identifier
    :ptype customer_column: str
    :param order: the sort, its last column the key (unique), as :func:`~threetears.core.keyed.keyset_page` takes it
    :ptype order: Sequence[Sort]
    :param user_column: the row's owning-user column, when a user's own rows are private to them
    :ptype user_column: str | None
    :param where: the listing's own filters, as :func:`~threetears.core.keyed.keyset_page` takes them
    :ptype where: Sequence[Fragment]
    :param page: the rows a statement reads
    :ptype page: int
    :return: the rows, in ``order``
    :rtype: list[dict[str, Any]]
    """
    visible = await visible_customers(conn, user_id=user_id, namespace_type=namespace_type)
    clauses: list[Fragment] = [visible.fragment(f"t.{quote_identifier(customer_column)}", platform_rows=True)]
    if user_column is not None:
        owner = f"t.{quote_identifier(user_column)}"
        clauses.append(Fragment(f"({owner} IS NULL OR {owner} = $1)", (user_id,)))
    return await keyset_rows(
        conn,
        table,
        columns=columns,
        by={},
        order=order,
        page=page,
        visible=clauses,
        where=where,
    )
