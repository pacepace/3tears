"""``read_all`` returns the whole relation or raises. Never a prefix.

``DatasourceQueryResult.truncated`` is necessary and NOT sufficient, and these
tests exist because a caller checking it faithfully on every page still lost
rows.

THE TRAP. Keyset paging steps past the last key it saw. When the key is unique
only by promise -- and a warehouse enforces nothing, including a declared primary
key -- duplicate keys make the cursor step OVER the duplicates. The next page
returns empty with ``truncated`` false, which is byte-identical to a clean
finish. A hand-written helper returned 3 of 8 rows and reported success, on a
caller that had dutifully checked ``truncated`` every time.

THE SIGNAL. The PREVIOUS page said truncated. An empty page after a truncated one
cannot mean "reached the end" -- there were more rows a moment ago -- so it means
the cursor skipped them or a writer deleted them. Both are an incomplete read.

Reported by the delivery session, who spiked all three shapes against a fake hub
and found that OFFSET is the one that looks right in testing and loses a row in
production when anything is deleted mid-read.
"""

from __future__ import annotations

from typing import Any

import pytest
from uuid import uuid7

from threetears.datasources.query_client import (
    DatasourceQueryResult,
    IncompleteReadError,
    read_all,
)


# parity-with: threetears.datasources.query_client.DatasourceQueryClient
class _FakeClient:
    """a query client that replays scripted pages and records the SQL it was given.

    Not a mock of the wire: the point is the PAGING logic above the wire, so the
    pages are scripted and the SQL is captured for assertions about the
    predicate.
    """

    def __init__(self, pages: list[DatasourceQueryResult]) -> None:
        """
        :param pages: the pages to return, in order
        :ptype pages: list[DatasourceQueryResult]
        """
        self._pages = list(pages)
        self.statements: list[str] = []
        self.params: list[list[Any]] = []

    def forwarded_identity_token(self) -> str:
        """the token the real client forwards on every query.

        Present because the parity gate requires it, and it is not a formality:
        ``read_all`` issues N requests rather than one, so a client whose token
        provider expires mid-read fails partway. Declaring the method keeps this
        fake honest about the surface it stands in for.

        :return: a placeholder token; the paging logic never inspects it
        :rtype: str
        """
        return "fake-identity-token"

    async def query(
        self,
        datasource_name: str,
        query: str,
        *,
        params: list[Any] | None = None,
        **_: Any,
    ) -> DatasourceQueryResult:
        """
        :param datasource_name: unused; the fake serves one datasource
        :ptype datasource_name: str
        :param query: the sql under test
        :ptype query: str
        :param params: bound parameters
        :ptype params: list[Any] | None
        :return: the next scripted page
        :rtype: DatasourceQueryResult
        """
        self.statements.append(query)
        self.params.append(list(params or []))
        if not self._pages:
            return _page([], truncated=False)
        return self._pages.pop(0)


def _page(rows: list[dict[str, Any]], *, truncated: bool) -> DatasourceQueryResult:
    """build a result page.

    :param rows: the page's rows
    :ptype rows: list[dict[str, Any]]
    :param truncated: whether the hub cut this page at its cap
    :ptype truncated: bool
    :return: the page
    :rtype: DatasourceQueryResult
    """
    return DatasourceQueryResult(
        rows=rows,
        row_count=len(rows),
        truncated=truncated,
        correlation_id=uuid7(),
    )


def _row(jurisdiction: str, period: str) -> dict[str, Any]:
    """one row of a published-files-shaped relation.

    :param jurisdiction: first key column
    :ptype jurisdiction: str
    :param period: second key column
    :ptype period: str
    :return: the row
    :rtype: dict[str, Any]
    """
    return {"jurisdiction": jurisdiction, "period": period}


_KEY = ("jurisdiction", "period")
_COLUMNS = ("jurisdiction", "period")


async def _read(client: _FakeClient, **kwargs: Any) -> list[dict[str, Any]]:
    """drive ``read_all`` against the fake with the standard shape.

    :param client: the scripted client
    :ptype client: _FakeClient
    :param kwargs: overrides passed through
    :ptype kwargs: Any
    :return: the rows read
    :rtype: list[dict[str, Any]]
    """
    return await read_all(
        client,  # type: ignore[arg-type]
        "published-files",
        columns=_COLUMNS,
        relation="evd_delivery.published_files",
        key=_KEY,
        **kwargs,
    )


class TestACleanReadReturnsEveryRow:
    """the happy path, including the boundary that a naive guard gets wrong."""

    @pytest.mark.asyncio
    async def test_a_single_untruncated_page_is_the_whole_relation(self) -> None:
        """
        :return: nothing
        :rtype: None
        """
        client = _FakeClient([_page([_row("ca", "2026-01"), _row("ny", "2026-01")], truncated=False)])

        rows = await _read(client, page_size=10)

        assert rows == [_row("ca", "2026-01"), _row("ny", "2026-01")]

    @pytest.mark.asyncio
    async def test_pages_are_concatenated_in_order(self) -> None:
        """
        :return: nothing
        :rtype: None
        """
        client = _FakeClient(
            [
                _page([_row("ca", "2026-01")], truncated=True),
                _page([_row("ny", "2026-01")], truncated=True),
                _page([_row("tx", "2026-01")], truncated=False),
            ]
        )

        rows = await _read(client, page_size=1)

        assert [r["jurisdiction"] for r in rows] == ["ca", "ny", "tx"]

    @pytest.mark.asyncio
    async def test_a_final_page_that_is_not_truncated_ends_the_read(self) -> None:
        """A last page reporting ``truncated=False`` is a clean finish, and the
        empty page after a NON-truncated one must not be treated as a fault.

        :return: nothing
        :rtype: None
        """
        client = _FakeClient(
            [
                _page([_row("ca", "2026-01")], truncated=False),
                _page([], truncated=False),
            ]
        )

        rows = await _read(client, page_size=1)

        assert len(rows) == 1


class TestADuplicateKeyIsCaughtRatherThanLost:
    """the defect that produced 3 of 8 rows and reported success."""

    @pytest.mark.asyncio
    async def test_an_empty_page_after_a_truncated_one_raises(self) -> None:
        """The cursor stepped over duplicates: there were more rows a moment ago,
        so an empty page cannot be the end.

        :return: nothing
        :rtype: None
        """
        client = _FakeClient(
            [
                _page([_row("ca", "2026-01")], truncated=True),
                _page([], truncated=False),
            ]
        )

        with pytest.raises(IncompleteReadError, match="empty page after a truncated one"):
            await _read(client, page_size=1)

    @pytest.mark.asyncio
    async def test_the_message_says_the_rows_read_are_not_the_whole_relation(self) -> None:
        """A caller that logs and continues must not be able to read the message
        as a warning.

        :return: nothing
        :rtype: None
        """
        client = _FakeClient(
            [
                _page([_row("ca", "2026-01")], truncated=True),
                _page([], truncated=False),
            ]
        )

        with pytest.raises(IncompleteReadError) as excinfo:
            await _read(client, page_size=1)

        assert "NOT the whole relation" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_a_cursor_that_cannot_advance_raises(self) -> None:
        """Every row in the page carries the same key, so paging cannot progress
        and the remaining rows are unreachable. Without this the loop spins.

        :return: nothing
        :rtype: None
        """
        client = _FakeClient(
            [
                _page([_row("ca", "2026-01")], truncated=True),
                _page([_row("ca", "2026-01")], truncated=True),
            ]
        )

        with pytest.raises(IncompleteReadError, match="did not advance"):
            await _read(client, page_size=1)


class TestThePredicateIsPortableAndBound:
    """nested OR, because row constructors are not portable across the backends."""

    @pytest.mark.asyncio
    async def test_the_first_page_carries_no_where_clause(self) -> None:
        """
        :return: nothing
        :rtype: None
        """
        client = _FakeClient([_page([_row("ca", "2026-01")], truncated=False)])

        await _read(client, page_size=10)

        assert "WHERE" not in client.statements[0]

    @pytest.mark.asyncio
    async def test_the_second_page_uses_nested_or_not_a_row_constructor(self) -> None:
        """
        :return: nothing
        :rtype: None
        """
        client = _FakeClient(
            [
                _page([_row("ca", "2026-01")], truncated=True),
                _page([], truncated=False),
            ]
        )

        with pytest.raises(IncompleteReadError):
            await _read(client, page_size=1)

        predicate = client.statements[1]
        assert "(jurisdiction > ?) OR (jurisdiction = ? AND period > ?)" in predicate
        assert "(jurisdiction, period) >" not in predicate

    @pytest.mark.asyncio
    async def test_cursor_values_are_bound_not_interpolated(self) -> None:
        """
        :return: nothing
        :rtype: None
        """
        client = _FakeClient(
            [
                _page([_row("ca", "2026-01")], truncated=True),
                _page([], truncated=False),
            ]
        )

        with pytest.raises(IncompleteReadError):
            await _read(client, page_size=1)

        assert client.params[1] == ["ca", "ca", "2026-01"]
        assert "ca" not in client.statements[1]

    @pytest.mark.asyncio
    async def test_the_page_size_reaches_the_sql_as_a_limit(self) -> None:
        """The hub's cap bounds what crosses the bus, not what the warehouse
        returns -- the hub materializes the whole result and slices. Without a
        LIMIT here every page drags the remaining relation into hub memory.

        :return: nothing
        :rtype: None
        """
        client = _FakeClient([_page([_row("ca", "2026-01")], truncated=False)])

        await _read(client, page_size=250)

        assert "LIMIT 250" in client.statements[0]


class TestArgumentsThatCannotDescribeACompleteRead:
    """refused at the door rather than producing an unverifiable result."""

    @pytest.mark.asyncio
    async def test_an_empty_key_is_refused(self) -> None:
        """
        :return: nothing
        :rtype: None
        """
        with pytest.raises(ValueError, match="key must not be empty"):
            await read_all(
                _FakeClient([]),  # type: ignore[arg-type]
                "published-files",
                columns=_COLUMNS,
                relation="t",
                key=(),
            )

    @pytest.mark.asyncio
    async def test_a_runaway_read_is_bounded(self) -> None:
        """An unbounded read against a growing relation never terminates.

        :return: nothing
        :rtype: None
        """
        client = _FakeClient([_page([_row(f"j{i}", "2026-01")], truncated=True) for i in range(10)])

        with pytest.raises(IncompleteReadError, match="still reading after"):
            await _read(client, page_size=1, max_pages=3)
