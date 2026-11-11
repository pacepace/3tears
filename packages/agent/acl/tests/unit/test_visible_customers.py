"""unit tests for the caller's visible-customer set and the visible listing read on it.

:func:`visible_customers` reads the set once (``caller_visible_customers_query`` for one namespace type);
:func:`visible_rows` reads a whole listing a page a statement on that set. These pin what each statement binds and
admits: a platform-wide grant adds no customer condition, a set of customers admits platform rows beside them, a
user column keeps another user's rows out whatever the grant, and the pages run until one comes back short. The
rows a real database returns for each caller are proven equal to the per-row clauses' by the consumers' equivalence
tests (the hub's knowledge listings).
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from threetears.core.keyed import Sort

from threetears.agent.acl import VisibleCustomers, visible_customers, visible_rows

_ORDER = (Sort("date_created", cast="timestamptz"), Sort("id", cast="uuid"))


class _Conn:
    """a connection double: the grants row it is given, and pages of rows it hands out in turn."""

    def __init__(self, grants: dict[str, Any], pages: list[list[dict[str, Any]]]) -> None:
        """record what the reads answer.

        :param grants: the grants read's one row
        :ptype grants: dict[str, Any]
        :param pages: each page read's rows, in turn
        :ptype pages: list[list[dict[str, Any]]]
        """
        self.grants = grants
        self.pages = list(pages)
        self.grant_reads: list[tuple[str, tuple[Any, ...]]] = []
        self.page_reads: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchrow(self, sql: str, *params: Any) -> dict[str, Any]:
        """answer the grants read.

        :param sql: the statement
        :ptype sql: str
        :param params: its parameters
        :ptype params: Any
        :return: the grants row
        :rtype: dict[str, Any]
        """
        self.grant_reads.append((sql, params))
        return self.grants

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """answer a page read with the next page.

        :param sql: the statement
        :ptype sql: str
        :param params: its parameters
        :ptype params: Any
        :return: the page's rows
        :rtype: list[dict[str, Any]]
        """
        if sql.startswith("SELECT version()"):
            # the pager asks once which database it pages, as a pool is asked: a PostgreSQL one
            return [{"v": "PostgreSQL 16.15"}]
        self.page_reads.append((sql, params))
        return self.pages.pop(0) if self.pages else []


def _rows(count: int, start: int = 0) -> list[dict[str, Any]]:
    """rows a page returns, each with the sort's columns.

    :param count: how many
    :ptype count: int
    :param start: the first row's number
    :ptype start: int
    :return: the rows
    :rtype: list[dict[str, Any]]
    """
    return [
        {"id": UUID(int=start + index), "date_created": f"2026-11-10T00:00:{start + index:02d}+00:00"}
        for index in range(count)
    ]


def test_the_set_binds_its_customers_and_none_for_every_customer() -> None:
    """``bound`` is the customers, or None for a platform-wide grant; ``sees_none`` only for no customer at all."""
    customer = uuid4()
    assert VisibleCustomers(is_all=False, customer_ids=(customer,)).bound == [customer]
    assert VisibleCustomers(is_all=True, customer_ids=()).bound is None
    assert VisibleCustomers(is_all=False, customer_ids=()).sees_none
    assert not VisibleCustomers(is_all=True, customer_ids=()).sees_none


def test_the_fragment_admits_platform_rows_only_when_asked() -> None:
    """a set's fragment is ``= ANY`` its customers, with ``IS NULL OR`` for platform rows; every customer has none."""
    customer = uuid4()
    visible = VisibleCustomers(is_all=False, customer_ids=(customer,))
    plain = visible.fragment('t."customer_id"')
    shared = visible.fragment('t."customer_id"', platform_rows=True)
    assert plain is not None and plain.sql == 't."customer_id" = ANY($1::uuid[])'
    assert plain.params == ([customer],)
    assert shared is not None and shared.sql == '(t."customer_id" IS NULL OR t."customer_id" = ANY($1::uuid[]))'
    assert VisibleCustomers(is_all=True, customer_ids=()).fragment('t."customer_id"').sql == "TRUE"


def test_the_fragment_is_total_every_row_no_row_or_the_set() -> None:
    """Flips if a fragment is ever None (a literal ``visible=[...]`` could not hold it): a platform-wide grant is
    ``TRUE`` with no parameter, grants covering no customer ``FALSE`` -- or, platform rows seen by all, only those."""
    every = VisibleCustomers(is_all=True, customer_ids=()).fragment('t."customer_id"', platform_rows=True)
    assert (every.sql, every.params) == ("TRUE", ())
    none = VisibleCustomers(is_all=False, customer_ids=())
    assert (none.fragment('t."customer_id"').sql, none.fragment('t."customer_id"').params) == ("FALSE", ())
    shared = none.fragment('t."customer_id"', platform_rows=True)
    assert shared.sql == '(t."customer_id" IS NULL OR t."customer_id" = ANY($1::uuid[]))'
    assert shared.params == ([],)


async def test_the_set_is_read_once_for_the_namespace_type() -> None:
    """the grants read binds the caller and the listing's namespace type, and its customers come back sorted."""
    caller = uuid4()
    first, second = UUID(int=2), UUID(int=1)
    conn = _Conn({"is_all": False, "customer_ids": [first, second]}, [])

    visible = await visible_customers(conn, user_id=caller, namespace_type="knowledge")

    assert visible == VisibleCustomers(is_all=False, customer_ids=(second, first))
    assert [params for _, params in conn.grant_reads] == [(caller, "knowledge")]


async def test_a_platform_wide_grant_reads_every_customer_but_not_another_users_rows() -> None:
    """every customer: no customer condition; the user column still binds the caller."""
    caller = uuid4()
    conn = _Conn({"is_all": True, "customer_ids": []}, [_rows(2)])

    rows = await visible_rows(
        conn,
        "eval_sets",
        columns=["id", "date_created"],
        user_id=caller,
        namespace_type="knowledge",
        customer_column="customer_id",
        user_column="user_id",
        order=_ORDER,
        page=5,
    )

    assert rows == _rows(2)
    [(sql, params)] = conn.page_reads
    assert "ANY(" not in sql
    assert '(t."user_id" IS NULL OR t."user_id" = $1)' in sql
    assert params == (caller,)
    assert sql.endswith("LIMIT 6")


async def test_a_set_of_customers_admits_platform_rows_and_pages_to_a_short_page() -> None:
    """the customers bind beside platform rows; a full page (page + 1 rows) reads on past its last row until a page
    comes back short."""
    caller, customer = uuid4(), uuid4()
    conn = _Conn({"is_all": False, "customer_ids": [customer]}, [_rows(3), _rows(1, start=3)])

    rows = await visible_rows(
        conn,
        "promotion_requests",
        columns=["id", "date_created"],
        user_id=caller,
        namespace_type="knowledge",
        customer_column="source_customer_id",
        order=_ORDER,
        page=2,
    )

    assert rows == [*_rows(3), *_rows(1, start=3)]
    first_sql, first_params = conn.page_reads[0]
    assert '(t."source_customer_id" IS NULL OR t."source_customer_id" = ANY($1::uuid[]))' in first_sql
    assert "user_id" not in first_sql
    assert first_params == ([customer],)
    second_sql, second_params = conn.page_reads[1]
    assert len(conn.page_reads) == 2
    assert len(second_params) == 3, "the second page binds the last row's two sort values and the customers"
    assert second_params[-1] == [customer]
    assert "ANY($3::uuid[])" in second_sql
