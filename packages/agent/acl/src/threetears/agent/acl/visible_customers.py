"""the customers a caller's grants cover, read once: what a listing pages within, instead of an EXISTS a row.

A listing of one entity type used to admit each row through a correlated EXISTS over the caller's grants
(:func:`threetears.agent.acl.caller_visible_customer_clause`), run for every row the page scanned. The customers the
caller can see are resolved once instead, by keyed reads of the caller's memberships, the groups they nest in and
those groups' grants (:func:`visible_customers` for the listing's namespace type, the same set the per-row clause
admits), and the page reads its rows by key with ``customer_id = ANY(<those customers>)`` and its
``LIMIT``. A platform-wide grant (``scope_type='all'``) narrows nothing (its fragment is ``TRUE``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final
from uuid import UUID

from threetears.core.keyed import AnyOf, Fragment, Sort, keyset_rows, read_by
from threetears.core.sql_fragments import quote_identifier

from threetears.agent.acl.types import MAX_GROUP_MEMBERSHIP_DEPTH

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


#: the partitions a namespace is written in: a namespace grant's namespace is read by its id in either
_NAMESPACE_SCOPES: Final[tuple[str, ...]] = ("platform", "customer")


async def visible_customers(pool: Any, *, user_id: UUID, namespace_type: str | None) -> VisibleCustomers:
    """the customers ``user_id``'s grants cover for ``namespace_type``: the caller's groups, then their grants, by key.

    The same set :func:`threetears.agent.acl.caller_visible_customer_clause` admits a row of for that type: a
    platform-wide grant, a ``type_customer`` grant on the type, or a grant on a namespace of the customer. The groups
    are the caller's own (its memberships, ``idx_group_members_reverse``) and those they nest in, a level a read, to
    :data:`MAX_GROUP_MEMBERSHIP_DEPTH` groups from the caller (the evaluator's walk); then every reached group's grants
    in one read (``idx_role_assignments_group_scope``), and a namespace grant's namespace by its id. Reads: one for the
    caller's groups, one a further level that finds a group (none past the depth), one for the grants, one for the
    namespaces when a namespace grant is among them.

    Why keyed reads, not the one statement it was: that statement (a MATERIALIZED ``reached`` CTE written out to the
    deepest nesting, its grants by a LATERAL lookup) was raw SQL no helper wrote. Each read here is one index's keyed
    lookup; the levels stop at the first that finds no new group, so a caller whose groups nest in nothing (most)
    costs three reads where the statement was one.

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
    # cache-bypass: the caller's memberships, by the member's key (idx_group_members_reverse)
    level = {
        row["group_id"]
        for row in await read_by(
            pool, "group_members", by={"member_type": "user", "member_id": user_id}, columns=("group_id",)
        )
    }
    reached = set(level)
    for _ in range(MAX_GROUP_MEMBERSHIP_DEPTH - 1):
        if not level:
            break
        # cache-bypass: the groups this level's groups are members of, by the member's key; each group is walked
        # once (its first, shortest reach), so a cycle ends and the set is the groups within the depth
        parents = {
            row["group_id"]
            for row in await read_by(
                pool,
                "group_members",
                by={"member_type": "group", "member_id": AnyOf(sorted(level, key=str), cast="uuid")},
                columns=("group_id",),
            )
        }
        level = parents - reached
        reached |= level
    is_all = False
    customers: set[UUID] = set()
    if reached:
        # cache-bypass: every reached group's grants, by group (idx_role_assignments_group_scope), both partitions
        grants = await read_by(
            pool,
            "role_assignments",
            by={"group_id": AnyOf(sorted(reached, key=str), cast="uuid")},
            columns=("scope_type", "scope_customer_id", "scope_namespace_type", "scope_namespace_id"),
        )
        is_all = any(grant["scope_type"] == "all" for grant in grants)
        customers = {
            grant["scope_customer_id"]
            for grant in grants
            if grant["scope_type"] == "type_customer"
            and grant["scope_customer_id"] is not None
            and (namespace_type is None or grant["scope_namespace_type"] == namespace_type)
        }
        granted = {
            grant["scope_namespace_id"]
            for grant in grants
            if grant["scope_type"] == "namespace" and grant["scope_namespace_id"] is not None
        }
        if granted:
            # cache-bypass: a namespace grant's namespace, by its id in either partition (namespaces_pkey)
            customers |= {
                row["customer_id"]
                for row in await read_by(
                    pool,
                    "namespaces",
                    by={
                        "row_scope": AnyOf(_NAMESPACE_SCOPES, cast=None),
                        "namespace_id": AnyOf(sorted(granted, key=str), cast="uuid"),
                    },
                    columns=("customer_id",),
                )
                if row["customer_id"] is not None
            }
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
