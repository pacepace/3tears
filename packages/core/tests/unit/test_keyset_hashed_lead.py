"""A keyset read of a hashed lead: which database the L3 answers from decides whether the read walks the lead's values.

The walk's plans are proven on YugabyteDB (``tests/integration/test_keyset_hashed_lead_live.py``); these pin which
reads walk (YugabyteDB, a lead named hashed, no filter pinning it), that every other read keeps the cursor's form, and
what the rail's proxy is asked to tell (``hub_capabilities``: an older hub names no dialect, and is Postgres).
"""

from __future__ import annotations

from typing import Any

from threetears.core.backends.dialect import l3_on_yugabyte
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
        if 'WHERE "state" = $1' in sql:
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
