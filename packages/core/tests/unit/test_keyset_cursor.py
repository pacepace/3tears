"""The keyset pager's cursor: one condition both Postgres and YugabyteDB seek by, the same rows either way."""

from __future__ import annotations

from typing import Any

from threetears.core.backends.protocol import L3Page
from threetears.core.keyset import read_keyset_pages


class _Recorded:
    """answers each page from rows held here, as a database would, and keeps each statement."""

    def __init__(self, rows: list[dict[str, Any]], key: tuple[str, ...]) -> None:
        self.rows = rows
        self.key = key
        self.statements: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        self.statements.append((sql, params))
        limit = int(sql.rsplit("LIMIT ", 1)[1])
        if "> (" not in sql:
            return self.rows[:limit]
        cursor = tuple(params[-len(self.key) :])
        return [row for row in self.rows if tuple(row[c] for c in self.key) > cursor][:limit]


class TestTheCursor:
    async def test_a_multi_column_cursor_is_the_bound_the_row_comparison_and_its_expansion(self) -> None:
        """Flips if a page past its cursor stops naming the leading bound YugabyteDB seeks by, or the expansion it filters by."""
        rows = [{"a": a, "b": b, "c": c} for a in range(3) for b in range(3) for c in range(3)]
        recorded = _Recorded(rows, ("a", "b", "c"))

        read = await read_keyset_pages(recorded, "SELECT * FROM t", ("a", "b", "c"), page_size=10)

        assert read.rows == rows
        sql, params = recorded.statements[1]
        assert sql == (
            'SELECT * FROM t WHERE "a" >= $1 AND ("a", "b", "c") > ($1, $2, $3) '
            'AND ("a" > $1 OR ("a" = $1 AND ("b" > $2 OR ("b" = $2 AND ("c" > $3))))) '
            'ORDER BY "a", "b", "c" LIMIT 11'
        )
        assert params == (1, 0, 0)

    async def test_a_one_column_cursor_is_one_comparison(self) -> None:
        """Flips if a one-column key's cursor grows conditions it needs none of."""
        recorded = _Recorded([{"a": n, "b": 0, "c": 0} for n in range(5)], ("a",))

        await read_keyset_pages(recorded, "SELECT * FROM t", ("a",), page_size=2, where={"b": 0})

        assert 'WHERE "b" = $1 AND ("a") > ($2) ORDER BY "a" LIMIT 3' in recorded.statements[1][0]


class _ByteCapped:
    """a broker that cuts each reply at ``cap`` bytes, a row costing ``row_bytes``, and announces the cap; or, with
    ``announce`` off, an older one that says nothing of it."""

    def __init__(self, rows: list[dict[str, Any]], cap: int, row_bytes: int, *, announce: bool = True) -> None:
        self.rows = rows
        self.cap = cap
        self.announce = announce
        self.row_bytes = row_bytes
        self.asked: list[int] = []
        self.cut = 0

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        raise AssertionError("a byte-capped transport is read a page at a time")

    async def fetch_page(self, sql: str, *params: Any) -> L3Page:
        limit = int(sql.rsplit("LIMIT ", 1)[1])
        self.asked.append(limit)
        after = [row for row in self.rows if "> (" not in sql or (row["a"],) > (params[-1],)][:limit]
        fits = self.cap // self.row_bytes
        truncated = len(after) > fits
        self.cut += truncated
        answered = after[:fits]
        return L3Page(
            rows=answered,
            truncated=truncated,
            reply_bytes=len(answered) * self.row_bytes,
            reply_cap=self.cap if self.announce else None,
        )


class TestAPageSizedByBytes:
    async def test_pages_grow_to_the_budget_and_a_cut_page_loses_no_row(self) -> None:
        """Flips if a byte-capped read stays at its first page's rows, or a page the transport cut loses its tail."""
        rows = [{"a": n} for n in range(10_000)]
        # 100 bytes a row under a 50,000-byte cap: 500 fit; the first page asks for 2,000 and is cut
        transport = _ByteCapped(rows, cap=50_000, row_bytes=100)

        read = await read_keyset_pages(transport, "SELECT * FROM t", ("a",), page_size=2_000)

        assert read.rows == rows
        assert transport.cut >= 1
        # sized to 80% of the cap after the first page: 400 rows a page, never cut again
        assert transport.asked[1:3] == [401, 401]

    async def test_a_broker_that_announces_no_cap_is_paged_by_rows(self) -> None:
        """Flips if a page from an older broker, which announces no cap, is resized from its bytes."""
        rows = [{"a": n} for n in range(2_500)]
        transport = _ByteCapped(rows, cap=10**9, row_bytes=100, announce=False)

        read = await read_keyset_pages(transport, "SELECT * FROM t", ("a",), page_size=999)

        assert read.rows == rows
        assert set(transport.asked) == {1000}
