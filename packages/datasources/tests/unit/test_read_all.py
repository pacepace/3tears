"""``read_all`` returns the whole relation or raises. Never a prefix.

HOW THIS WENT WRONG TWICE, because the tests are shaped by it.

The guard was built on ``DatasourceQueryResult.truncated``. The hub computes
``truncated = total > MAX_RESULT_ROWS`` over what the query returned, and
``read_all`` sends ``LIMIT page_size``. So ``total <= page_size``, and for EVERY
page size at or under the cap that comparison is unsatisfiable: truncated is
permanently false and the guard was unreachable. Not defaulted past. Unreachable.

``truncated`` answers "did the hub CUT an unbounded result". It was read here as
"are there more rows", which is a different question the hub was never asked.

The first round of tests missed it entirely because they SCRIPTED truncated as a
fixture value -- ``_page([...], truncated=True)`` -- and asserted the guard fired
when handed it. A fake agreeing with the code instead of the world. The second
round then "fixed" the default without touching the arithmetic, which is what a
boundary-value reading of the bug produces.

So this fake PAGES A REAL DATASET. It recovers the cursor from the bound
parameters, applies the keyset comparison itself, honours the LIMIT it is given,
and always reports ``truncated=False`` exactly as the hub does under a LIMIT.
Nothing about has-more is scripted: if the implementation stops asking for a
sentinel row, these tests fail.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid7

import pytest

from threetears.datasources.query_client import (
    DatasourceQueryResult,
    IncompleteReadError,
    read_all,
)

_KEY = ("jurisdiction", "period")
_COLUMNS = ("jurisdiction", "period")


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


def _limit_of(sql: str) -> int:
    """read the LIMIT the implementation asked for.

    :param sql: the emitted statement
    :ptype sql: str
    :return: the limit
    :rtype: int
    """
    return int(sql.rsplit("LIMIT", 1)[1].strip())


# parity-with: threetears.datasources.query_client.DatasourceQueryClient
class _PagingWarehouse:
    """a fake that actually pages, rather than replaying scripted answers.

    The cursor is recovered from the BOUND PARAMETERS rather than by parsing the
    predicate: ``_keyset_predicate`` appends each cursor column last within its
    clause, so the final ``len(key)`` parameters are the cursor. That keeps the
    fake honest about the contract without making it a SQL parser.
    """

    def __init__(self, dataset: list[dict[str, Any]]) -> None:
        """
        :param dataset: the relation, in any order; the fake sorts by key
        :ptype dataset: list[dict[str, Any]]
        """
        self._rows = sorted(dataset, key=lambda r: tuple(r[c] for c in _KEY))
        self.statements: list[str] = []
        self.params: list[list[Any]] = []

    def forwarded_identity_token(self) -> str:
        """
        :return: a placeholder; the paging logic never inspects it
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
        """serve one keyset page from the dataset.

        :param datasource_name: unused; the fake serves one relation
        :ptype datasource_name: str
        :param query: the statement under test
        :ptype query: str
        :param params: bound parameters, ending with the cursor
        :ptype params: list[Any] | None
        :return: the page
        :rtype: DatasourceQueryResult
        """
        self.statements.append(query)
        bound = list(params or [])
        self.params.append(bound)

        candidates = self._rows
        if bound:
            cursor = tuple(bound[-len(_KEY) :])
            candidates = [r for r in self._rows if tuple(r[c] for c in _KEY) > cursor]

        served = candidates[: _limit_of(query)]
        return DatasourceQueryResult(
            rows=served,
            row_count=len(served),
            # The hub sets this only for an UNBOUNDED query it had to cut. Under a
            # LIMIT it is always False, which is the defect these tests pin: the
            # implementation must not depend on it.
            truncated=False,
            correlation_id=uuid7(),
        )


async def _read(warehouse: _PagingWarehouse, **kwargs: Any) -> list[dict[str, Any]]:
    """drive ``read_all`` against the fake.

    :param warehouse: the paging fake
    :ptype warehouse: _PagingWarehouse
    :param kwargs: overrides passed through
    :ptype kwargs: Any
    :return: the rows read
    :rtype: list[dict[str, Any]]
    """
    return await read_all(
        warehouse,  # type: ignore[arg-type]
        "published-files",
        columns=_COLUMNS,
        relation="evd_delivery.published_files",
        key=_KEY,
        **kwargs,
    )


class TestAUniqueKeyReadsCompletely:
    """the contract, over real paging rather than scripted pages."""

    @pytest.mark.asyncio
    async def test_every_row_comes_back_across_many_pages(self) -> None:
        """
        :return: nothing
        :rtype: None
        """
        dataset = [_row(f"s{i:02d}", "2026-01") for i in range(25)]
        warehouse = _PagingWarehouse(dataset)

        rows = await _read(warehouse, page_size=4)

        assert len(rows) == 25
        assert [r["jurisdiction"] for r in rows] == sorted(r["jurisdiction"] for r in dataset)

    @pytest.mark.asyncio
    async def test_a_relation_smaller_than_one_page_reads_in_one_call(self) -> None:
        """
        :return: nothing
        :rtype: None
        """
        warehouse = _PagingWarehouse([_row("ca", "2026-01"), _row("ny", "2026-01")])

        rows = await _read(warehouse, page_size=10)

        assert len(rows) == 2
        assert len(warehouse.statements) == 1

    @pytest.mark.asyncio
    async def test_an_empty_relation_is_not_an_error(self) -> None:
        """
        :return: nothing
        :rtype: None
        """
        assert await _read(_PagingWarehouse([]), page_size=10) == []

    @pytest.mark.asyncio
    async def test_a_relation_that_is_an_exact_multiple_of_the_page_terminates(self) -> None:
        """The boundary a has-more signal most often gets wrong: a full last page
        looks identical to a page with more behind it unless a sentinel separates
        them.

        :return: nothing
        :rtype: None
        """
        warehouse = _PagingWarehouse([_row(f"s{i:02d}", "2026-01") for i in range(10)])

        rows = await _read(warehouse, page_size=5)

        assert len(rows) == 10


class TestADuplicateKeyRaisesRatherThanSilentlyDropping:
    """the 3-of-8 shape, walked through real page arithmetic.

    This is the failure the function exists for, and the one both earlier test
    rounds could not see. A duplicated key straddles a page boundary, the
    predicate steps past it, and rows sharing that key are never returned.
    """

    @pytest.mark.asyncio
    async def test_a_duplicate_straddling_a_page_boundary_raises(self) -> None:
        """
        :return: nothing
        :rtype: None
        """
        dataset = [
            _row("ca", "2026-01"),
            _row("ca", "2026-01"),
            _row("ca", "2026-01"),
            _row("ny", "2026-01"),
            _row("ny", "2026-01"),
            _row("tx", "2026-01"),
            _row("tx", "2026-01"),
            _row("tx", "2026-01"),
        ]

        with pytest.raises(IncompleteReadError):
            await _read(_PagingWarehouse(dataset), page_size=3)

    @pytest.mark.asyncio
    async def test_it_never_returns_a_short_read_as_a_complete_one(self) -> None:
        """The whole contract in one assertion: either every row, or a raise.

        :return: nothing
        :rtype: None
        """
        dataset = [_row("ca", "2026-01")] * 5 + [_row("ny", "2026-01")] * 5
        warehouse = _PagingWarehouse(dataset)

        try:
            rows = await _read(warehouse, page_size=3)
        except IncompleteReadError:
            return
        assert len(rows) == len(dataset), "returned a prefix instead of raising"

    @pytest.mark.asyncio
    async def test_a_page_of_identical_keys_raises_on_the_cursor(self) -> None:
        """Every row in a page sharing a key: the cursor cannot move.

        :return: nothing
        :rtype: None
        """
        warehouse = _PagingWarehouse([_row("ca", "2026-01")] * 12)

        # Caught by the empty-page guard rather than the cursor guard: page one
        # keeps three of the four identical rows, the cursor moves to that key,
        # and the next page is empty while the sentinel said more existed. Either
        # raise is correct; asserting the specific one pins which guard fires and
        # would hide the other silently going dead.
        with pytest.raises(IncompleteReadError, match="empty page after a page that had more"):
            await _read(warehouse, page_size=3)


class TestTheHasMoreSignalIsTheSentinelAndNotTruncated:
    """pins the mechanism, because the previous one was structurally dead."""

    @pytest.mark.asyncio
    async def test_the_query_asks_for_one_row_more_than_the_page(self) -> None:
        """The extra row IS the signal.

        :return: nothing
        :rtype: None
        """
        warehouse = _PagingWarehouse([_row(f"s{i:02d}", "2026-01") for i in range(3)])

        await _read(warehouse, page_size=7)

        assert _limit_of(warehouse.statements[0]) == 8

    @pytest.mark.asyncio
    async def test_the_sentinel_row_is_not_handed_to_the_caller(self) -> None:
        """It is a signal, not data. Returning it would duplicate a row across
        two pages.

        :return: nothing
        :rtype: None
        """
        dataset = [_row(f"s{i:02d}", "2026-01") for i in range(9)]
        warehouse = _PagingWarehouse(dataset)

        rows = await _read(warehouse, page_size=4)

        assert len(rows) == 9
        assert len({(r["jurisdiction"], r["period"]) for r in rows}) == 9

    @pytest.mark.asyncio
    async def test_the_read_does_not_depend_on_truncated(self) -> None:
        """The fake always reports ``truncated=False``, exactly as the hub does
        under a LIMIT. A correct implementation is unaffected; the versions
        shipped in 0.41.0 and 0.41.1 depended on a flag that could never be true.

        :return: nothing
        :rtype: None
        """
        warehouse = _PagingWarehouse([_row(f"s{i:02d}", "2026-01") for i in range(20)])

        rows = await _read(warehouse, page_size=6)

        assert len(rows) == 20


class TestThePredicateIsPortableAndBound:
    """nested OR, because row constructors are not portable across backends."""

    @pytest.mark.asyncio
    async def test_the_first_page_carries_no_where_clause(self) -> None:
        """
        :return: nothing
        :rtype: None
        """
        warehouse = _PagingWarehouse([_row("ca", "2026-01")])

        await _read(warehouse, page_size=10)

        assert "WHERE" not in warehouse.statements[0]

    @pytest.mark.asyncio
    async def test_the_next_page_uses_nested_or_not_a_row_constructor(self) -> None:
        """
        :return: nothing
        :rtype: None
        """
        warehouse = _PagingWarehouse([_row(f"s{i:02d}", "2026-01") for i in range(6)])

        await _read(warehouse, page_size=2)

        predicate = warehouse.statements[1]
        assert "(jurisdiction > ?) OR (jurisdiction = ? AND period > ?)" in predicate
        assert "(jurisdiction, period) >" not in predicate

    @pytest.mark.asyncio
    async def test_cursor_values_are_bound_not_interpolated(self) -> None:
        """
        :return: nothing
        :rtype: None
        """
        warehouse = _PagingWarehouse([_row(f"s{i:02d}", "2026-01") for i in range(6)])

        await _read(warehouse, page_size=2)

        assert warehouse.params[1] == ["s01", "s01", "2026-01"]
        assert "s01" not in warehouse.statements[1]


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
                _PagingWarehouse([]),  # type: ignore[arg-type]
                "published-files",
                columns=_COLUMNS,
                relation="t",
                key=(),
            )

    @pytest.mark.asyncio
    async def test_a_page_size_at_the_cap_is_refused_so_the_sentinel_fits(self) -> None:
        """``page_size + 1`` must stay under the hub's cap, or the cap eats the
        sentinel and "more rows exist" becomes "that was the last page".

        :return: nothing
        :rtype: None
        """
        with pytest.raises(ValueError, match="UNDER the hub's row cap"):
            await _read(_PagingWarehouse([]), page_size=1000)

    @pytest.mark.asyncio
    async def test_the_refusal_explains_the_sentinel(self) -> None:
        """A caller told only "too big" picks one smaller and keeps the bug.

        :return: nothing
        :rtype: None
        """
        with pytest.raises(ValueError) as excinfo:
            await _read(_PagingWarehouse([]), page_size=2000)

        assert "sentinel" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_the_default_leaves_room_for_the_sentinel(self) -> None:
        """
        :return: nothing
        :rtype: None
        """
        import inspect

        from threetears.datasources.query_client import _HUB_ROW_CAP

        default = inspect.signature(read_all).parameters["page_size"].default

        assert default + 1 <= _HUB_ROW_CAP

    @pytest.mark.asyncio
    async def test_a_runaway_read_is_bounded(self) -> None:
        """
        :return: nothing
        :rtype: None
        """
        warehouse = _PagingWarehouse([_row(f"s{i:03d}", "2026-01") for i in range(50)])

        with pytest.raises(IncompleteReadError, match="still reading after"):
            await _read(warehouse, page_size=1, max_pages=3)
