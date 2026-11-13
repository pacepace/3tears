"""A keyset read of a hashed lead: which database the L3 answers from decides whether the read walks the lead's values.

The walk's plans are proven on YugabyteDB (``tests/integration/test_keyset_hashed_lead_live.py``); these pin which
reads walk (YugabyteDB, a lead named hashed, no filter pinning it), that every other read keeps the cursor's form, and
what the rail's proxy is asked to tell (``hub_capabilities``: an older hub names no dialect, and is Postgres).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import duckdb
import pytest

from threetears.core.backends.dialect import l3_on_yugabyte
from threetears.core.keyed import ThreadedDuckDB
from threetears.core.keyset import read_keyset_pages


class _Rail:
    """the rail's proxy as a pod sees it: the capabilities its broker names, and the rows each statement answers."""

    def __init__(self, capabilities: frozenset[str], rows: list[dict[str, Any]]) -> None:
        self.capabilities = capabilities
        self.rows = rows
        self.statements: list[str] = []
        self.asked = 0

    async def hub_capabilities(self) -> frozenset[str]:
        self.asked += 1
        return self.capabilities

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        self.statements.append(sql)
        limit = int(sql.rsplit("LIMIT ", 1)[1])
        if sql.startswith("SELECT DISTINCT"):
            leads = sorted({row["state"] for row in self.rows})
            if " > $" in sql or " >= $" in sql:
                past = params[-1]
                leads = [lead for lead in leads if lead > past or (" >= $" in sql and lead == past)]
            return [{"state": lead} for lead in leads[:limit]]
        order = [column.strip('"') for column in sql.rsplit("ORDER BY ", 1)[1].split(" LIMIT")[0].split(", ")]
        rows = list(self.rows)
        used = 0
        if 'WHERE "state" = $1' in sql or 'WHERE ("state" = $1)' in sql:
            rows, used = [row for row in rows if row["state"] == params[0]], 1
        elif '"state" = ANY($1::text[])' in sql:
            rows, used = [row for row in rows if row["state"] in params[0]], 1
        if "> (" in sql:
            cursor = tuple(params[used : used + len(order)])
            rows = [row for row in rows if tuple(row[column] for column in order) > cursor]
        return sorted(rows, key=lambda row: tuple(row[column] for column in order))[:limit]


def _rows() -> list[dict[str, Any]]:
    return [{"state": state, "geo": geo} for state in ("IL", "TX", "VA") for geo in range(7)]


class TestWhichReadsWalkTheLead:
    async def test_a_hub_on_yugabytedb_walks_a_hashed_lead_value_by_value(self) -> None:
        """Flips if a read of a lead named hashed, from a hub whose broker names ``dialect:yugabyte``, pages by the
        cursor alone, or if the walk loses, repeats or reorders a row."""
        rail = _Rail(frozenset({"dialect:yugabyte"}), _rows())

        read = await read_keyset_pages(rail, "SELECT * FROM t", ("state", "geo"), page_size=5, lead_array="text[]")

        assert read.rows == _rows()
        assert rail.statements[0].startswith('SELECT DISTINCT "state" FROM (SELECT * FROM t) AS "_leads"')
        assert not any("> (" in sql and '"state" =' not in sql for sql in rail.statements[1:]), rail.statements
        assert any('"state" = $1' in sql for sql in rail.statements), rail.statements

    async def test_an_older_hub_or_postgres_keeps_the_cursor_s_form(self) -> None:
        """Flips if a hub naming no dialect (one built before it) or ``dialect:postgres`` is walked as YugabyteDB."""
        for capabilities in (frozenset(), frozenset({"dialect:postgres"})):
            rail = _Rail(capabilities, _rows())

            read = await read_keyset_pages(rail, "SELECT * FROM t", ("state", "geo"), page_size=5, lead_array="text[]")

            assert read.rows == _rows()
            assert not any(sql.startswith("SELECT DISTINCT") for sql in rail.statements), rail.statements

    async def test_a_lead_no_one_names_hashed_or_a_filter_pins_is_never_walked_nor_asked_about(self) -> None:
        """Flips if a read whose lead is not named hashed, whose key is one column, or whose filter pins the lead asks
        the hub what it runs on, or walks."""
        rail = _Rail(frozenset({"dialect:yugabyte"}), _rows())

        await read_keyset_pages(rail, "SELECT * FROM t", ("state", "geo"), page_size=5)
        await read_keyset_pages(rail, "SELECT * FROM t", ("geo",), page_size=50, lead_array="bigint[]")
        await read_keyset_pages(
            rail, "SELECT * FROM t", ("state", "geo"), where={"state": "TX"}, page_size=5, lead_array="text[]"
        )

        assert rail.asked == 0
        assert not any(sql.startswith("SELECT DISTINCT") for sql in rail.statements), rail.statements

    async def test_a_walk_past_a_saved_key_reads_exactly_the_rows_after_it(self) -> None:
        """Flips if a walk resumed past a key inside a lead value skips or repeats a row."""
        rows = _rows()
        rail = _Rail(frozenset({"dialect:yugabyte"}), rows)

        read = await read_keyset_pages(
            rail, "SELECT * FROM t", ("state", "geo"), page_size=5, after=("TX", 3), lead_array="text[]"
        )

        assert read.rows == rows[rows.index({"state": "TX", "geo": 3}) + 1 :]


class _Recorded:
    """an in-process DuckDB L3 keeping each statement it was sent."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._inner = ThreadedDuckDB(conn)
        self.statements: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        self.statements.append((sql, params))
        return await self._inner.fetch(sql, *params)


class _RecordedOnYugabyte(_Recorded):
    """the same L3, as a hub whose broker names ``dialect:yugabyte`` (its rows still come from DuckDB)."""

    async def hub_capabilities(self) -> frozenset[str]:
        return frozenset({"dialect:yugabyte"})


@pytest.fixture
def races() -> Iterator[duckdb.DuckDBPyConnection]:
    """a table keyed (race, geo): 12 races of 7 rows each."""
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t (race VARCHAR, geo BIGINT, votes BIGINT, PRIMARY KEY (race, geo))")
    conn.execute("INSERT INTO t SELECT 'r' || lpad(r::VARCHAR, 2, '0'), g, r * g FROM range(12) a(r), range(7) b(g)")
    yield conn
    conn.close()


def _pinned(conn: duckdb.DuckDBPyConnection, values: list[str]) -> list[tuple[Any, ...]]:
    return [
        tuple(row)
        for row in conn.execute("SELECT * FROM t WHERE list_contains(?, race) ORDER BY race, geo", [values]).fetchall()
    ]


class TestALeadASetFilterPins:
    """hub #53: on YugabyteDB, ``where_any`` naming a hashed key lead walks the filter's values, never paging the whole
    set by the cursor (which re-reads and sorts every row of the set on every page). An ordered lead pages by the
    cursor: its index answers the set in key order."""

    _VALUES = ["r09", "r02", None, "r02", "r05", "r77"]

    async def test_the_rows_come_in_key_order_each_page_pinning_its_own_values(
        self, races: duckdb.DuckDBPyConnection
    ) -> None:
        """Flips if a page past a cursor reads with the whole set (``= ANY`` of every value), or the walk loses,
        repeats or reorders a row, or reads a NULL or absent value's rows."""
        l3 = _RecordedOnYugabyte(races)

        read = await read_keyset_pages(
            l3,
            "SELECT * FROM t",
            ("race", "geo"),
            where_any={"race": ("text[]", self._VALUES)},
            page_size=5,
            lead_array="text[]",
        )

        assert [tuple(row.values()) for row in read.rows] == _pinned(races, ["r09", "r02", "r05"])
        first, *pages = l3.statements
        assert "unnest($1::text[])" in first[0] and " FROM t" not in first[0], first
        assert pages, l3.statements
        whole = [str(params[0]) for _, params in pages if params and params[0] == self._VALUES]
        assert not whole, f"a page read the whole set: {whole}"
        assert all('"race" = $1' in sql or '"race" = ANY($1::text[])' in sql for sql, _ in pages), pages

    async def test_a_pinned_read_past_a_saved_key_reads_exactly_the_rows_after_it(
        self, races: duckdb.DuckDBPyConnection
    ) -> None:
        """Flips if a set-pinned read resumed past a key inside a value skips or repeats a row."""
        ordered = _pinned(races, ["r09", "r02", "r05"])

        read = await read_keyset_pages(
            _RecordedOnYugabyte(races),
            "SELECT * FROM t",
            ("race", "geo"),
            where_any={"race": ("text[]", self._VALUES)},
            page_size=5,
            after=ordered[10][:2],
            lead_array="text[]",
        )

        assert [tuple(row.values()) for row in read.rows] == ordered[11:]

    async def test_a_set_of_no_values_is_one_empty_page(self, races: duckdb.DuckDBPyConnection) -> None:
        """Flips if a filter pinning the lead to nothing reads a row, or hands over no page."""
        read = await read_keyset_pages(
            _RecordedOnYugabyte(races),
            "SELECT * FROM t",
            ("race", "geo"),
            where_any={"race": ("text[]", [])},
            page_size=5,
            lead_array="text[]",
        )

        assert read.rows == []

    @pytest.mark.parametrize(
        ("l3", "lead_array"), [(_Recorded, "text[]"), (_RecordedOnYugabyte, None)], ids=["postgres", "ordered-lead"]
    )
    async def test_an_ordered_lead_reads_a_set_that_fits_a_page_in_one_statement(
        self, races: duckdb.DuckDBPyConnection, l3: type[_Recorded], lead_array: str | None
    ) -> None:
        """Flips if a set pinning an ordered lead (Postgres, or a lead not declared hashed) is walked: its index
        answers ``lead = ANY`` in key order, and a set that fits a page is one statement (the copy follower's batched
        prefix read), never one for the values' order and more for their rows."""
        recorded = l3(races)

        read = await read_keyset_pages(
            recorded,
            "SELECT * FROM t",
            ("race", "geo"),
            where_any={"race": ("text[]", ["r05", "r02"])},
            page_size=20,
            lead_array=lead_array,
        )

        assert [tuple(row.values()) for row in read.rows] == _pinned(races, ["r02", "r05"])
        reads = [sql for sql, _ in recorded.statements if "version()" not in sql]
        assert len(reads) == 1 and "unnest" not in reads[0], recorded.statements

    @pytest.mark.parametrize(
        ("l3", "row_seek"), [(_Recorded, True), (_RecordedOnYugabyte, False)], ids=["postgres", "yugabyte-ordered-lead"]
    )
    async def test_a_set_pinned_read_on_postgres_names_its_cursor_by_the_row_comparison_alone(
        self, races: duckdb.DuckDBPyConnection, l3: type[_Recorded], row_seek: bool
    ) -> None:
        """Flips if a set-pinned read on Postgres names its cursor three ways (beside ``lead = ANY`` the leading
        bound and the expansion shrink Postgres's estimate until it bitmap-scans and sorts every row left in the
        set: 85,019 rows scanned for 30,029 on Postgres 16), or if YugabyteDB's ordered-lead read loses the leading
        bound it seeks by. Either way every row comes once, in key order."""
        recorded = l3(races)

        read = await read_keyset_pages(
            recorded,
            "SELECT * FROM t",
            ("race", "geo"),
            where_any={"race": ("text[]", self._VALUES)},
            page_size=5,
        )

        assert [tuple(row.values()) for row in read.rows] == _pinned(races, ["r09", "r02", "r05"])
        pages = [sql for sql, _ in recorded.statements if "version()" not in sql][1:]
        assert pages, recorded.statements
        assert all('("race", "geo") > ($2, $3)' in sql for sql in pages), pages
        three_ways = [sql for sql in pages if '"race" >= $2' in sql or " OR (" in sql]
        assert (not three_ways) if row_seek else (three_ways == pages), pages


class TestWhichDatabaseAnL3Is:
    async def test_the_rail_is_asked_its_broker_s_dialect_and_a_pool_its_version(self) -> None:
        """Flips if the rail's proxy is not asked its broker's capabilities, or a pool's ``version()`` is misread."""
        rail = _Rail(frozenset({"dialect:yugabyte"}), [])
        assert await l3_on_yugabyte(rail)
        assert rail.asked == 1
        assert not rail.statements

        class _Pool:
            def __init__(self, version: str) -> None:
                self.version = version

            async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
                assert sql == "SELECT version() AS v"
                return [{"v": self.version}]

        assert await l3_on_yugabyte(_Pool("PostgreSQL 15.12-YB-2026.1.0.0-b0 on aarch64"))
        assert not await l3_on_yugabyte(_Pool("PostgreSQL 16.4 on aarch64-unknown-linux-gnu"))
