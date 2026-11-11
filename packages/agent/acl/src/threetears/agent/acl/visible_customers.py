"""the customers a caller's grants cover, read once: what a listing pages within, instead of an EXISTS a row.

A listing of one entity type used to admit each row through a correlated EXISTS over the caller's grants
(:func:`threetears.agent.acl.caller_visible_customer_clause`), run for every row the page scanned. The customers the
caller can see are resolved once instead, by the caller's memberships through the grants' own indexes
(:func:`threetears.agent.acl.caller_visible_customers_query` for the listing's namespace type, the same set the
per-row clause admits), and the page reads its rows by key with ``customer_id = ANY(<those customers>)`` and its
``LIMIT``. A platform-wide grant (``scope_type='all'``) narrows nothing (its fragment is ``TRUE``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final
from uuid import UUID

from threetears.core.keyed import Fragment, Sort, keyset_page
from threetears.core.sql_fragments import quote_identifier

from threetears.agent.acl.query_visibility import caller_visible_customers_query

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


async def visible_customers(pool: Any, *, user_id: UUID, namespace_type: str) -> VisibleCustomers:
    """the customers ``user_id``'s grants cover for ``namespace_type``: one read of the caller's grants, by key.

    The same set :func:`threetears.agent.acl.caller_visible_customer_clause` admits a row of for that type: a
    platform-wide grant, a ``type_customer`` grant on the type, or a grant on a namespace of the customer.

    :param pool: anything with an asyncpg-shaped ``fetchrow``
    :ptype pool: Any
    :param user_id: the caller
    :ptype user_id: UUID
    :param namespace_type: the namespace type that governs the listed entity
    :ptype namespace_type: str
    :return: the customers
    :rtype: VisibleCustomers
    """
    sql, params = caller_visible_customers_query(user_id=user_id, scope_namespace_type=namespace_type)
    # cache-bypass: the caller's grants, by the caller's memberships (the grants' own indexes); one row
    row = await pool.fetchrow(sql, *params)
    return VisibleCustomers(is_all=bool(row["is_all"]), customer_ids=tuple(sorted(row["customer_ids"] or (), key=str)))


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
    :func:`threetears.core.keyed.keyset_page`, each statement one page of ``page`` rows past the last one's cursor,
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
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        found = await keyset_page(
            conn,
            table,
            columns=columns,
            by={},
            order=order,
            limit=page,
            cursor=cursor,
            visible=clauses,
            where=where,
        )
        rows.extend(found.rows)
        if found.next_cursor is None:
            return rows
        cursor = found.next_cursor
