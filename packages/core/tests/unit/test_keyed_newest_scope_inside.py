"""a versioned table's pin on its declared scope, when its key does not name the scope, is judged INSIDE the newest
relation's ``DISTINCT ON``, as a pin on its key is (``VersionedTable.fixed``).

Every version of such a table's key holds the key's scope (a delete's gone marker carries the scope of every key it
deletes, decision 27, and a restore finds a key's versions by it), so a pin on the scope holds of every version of a
key alike: judged before each key's newest version is chosen, it reads only the scope's versions (by the scope's
index) and answers the same rows as judged after. Judged after (the read before this), PostgreSQL never pushes it into
the ``DISTINCT ON`` (it is no ``DISTINCT ON`` column), and a state's count or a state's geographies read every state's
versions (reports' race-keyed live tables, 2026-11-26).

The table is reports-shaped: keyed by ``(race_id, geo_id)``, its scope ``state`` in its rows alone. Correctness runs on
an in-process DuckDB behind a reader that names no dialect, so every helper takes it for L3. The plans on PostgreSQL and
YugabyteDB are the integration pass's (``test_newest_scope_pin_plans_live``).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import duckdb
import pyarrow as pa
import pytest

from threetears.core.collections.complete_copy import read_l3_arrow, read_l3_rows
from threetears.core.collections.schema_backed import INT_TYPE, STRING_TYPE, Column, TableSchema
from threetears.core.ddl.row_epoch import row_versions
from threetears.core.keyed import KeyBatch, Sort, ThreadedDuckDB, counts_by, keyset_page, read_by, read_keyed
from threetears.core.keyed import rows as keyed_rows
from threetears.core.keyset import keyset_arrow_pages
from threetears.core.testing.l1_tables import TEST_L1_TABLE
from threetears.interfaces.keyed.keys import (
    VersionedTable,
    key_pins_within,
    newest_page_statement,
    newest_relation,
    versioned_table,
)

_TABLE = "scope_pin_results"

_COLUMNS = [
    Column("race_id", INT_TYPE),
    Column("geo_id", STRING_TYPE),
    Column("state", STRING_TYPE),
    Column("v", INT_TYPE),
]

_RESULTS = TableSchema(
    name=_TABLE,
    primary_key=("race_id", "geo_id"),
    columns=_COLUMNS,
    row_epoch=row_versions("state", scope_prefix="state:"),
    l1_eviction=TEST_L1_TABLE,
)

_DDL = (
    "(race_id INTEGER, geo_id VARCHAR, state VARCHAR, v INTEGER, epoch BIGINT, gone BOOLEAN NOT NULL DEFAULT false, "
    "PRIMARY KEY (race_id, geo_id, epoch))"
)

#: each key's newest live version, (race_id, geo_id) -> v: race 1 is PA's, race 2 OH's
_NEWEST = {(1, "a"): 2, (2, "a"): 5}


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    """race 1 (PA): geography ``a`` two versions (v 1 then 2), ``b`` a version then its gone marker (carrying PA);
    race 2 (OH): geography ``a`` one version."""
    conn = duckdb.connect()
    conn.execute(f"CREATE TABLE {_TABLE} {_DDL}")
    conn.execute(
        "CREATE TABLE threetears_run_markers (table_name VARCHAR, scope VARCHAR, prefix VARCHAR, epoch BIGINT)"
    )
    conn.execute(
        f"INSERT INTO {_TABLE} (race_id, geo_id, state, v, epoch, gone) VALUES "
        "(1, 'a', 'PA', 1, 10, false), (1, 'a', 'PA', 2, 20, false), "
        "(1, 'b', 'PA', 7, 10, false), (1, 'b', 'PA', NULL, 30, true), "
        "(2, 'a', 'OH', 5, 15, false)"
    )
    yield conn
    conn.close()


class _L3:
    """an L3 reader answering from DuckDB, naming no dialect; it keeps every statement it was sent."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._inner = ThreadedDuckDB(conn)
        self.statements: list[str] = []

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self.statements.append(query)
        return await self._inner.fetch(query, *params)

    async def fetchrow(self, query: str, *params: Any) -> Any:
        self.statements.append(query)
        rows = await self._inner.fetch(query, *params)
        return rows[0] if rows else None


def _chosen(statement: str) -> str:
    """the newest relation's ``DISTINCT ON`` select of ``statement``: what is judged before each key's newest version
    is chosen."""
    start = statement.index("SELECT DISTINCT ON")
    return statement[start : statement.index(") AS n", start)]


def _read(rows: Any) -> dict[tuple[int, str], int]:
    """the rows read, (race_id, geo_id) -> v, each key once."""
    answered: dict[tuple[int, str], int] = {}
    for row in rows:
        key = (int(row["race_id"]), row["geo_id"])
        assert key not in answered, f"{key} read twice"
        answered[key] = int(row["v"])
    return answered


class TestTheDeclaration:
    def test_a_scope_past_the_key_is_declared_fixed(self) -> None:
        """Flips if a versioned table whose key does not name its scope registers no fixed scope, so its pins on the
        scope are judged after the newest version is chosen."""
        declared = versioned_table(_TABLE, l3=True)
        assert declared is not None
        assert declared.key == ("race_id", "geo_id")
        assert declared.fixed == ("state",)
        assert declared.markers is None

    def test_a_versioned_table_declares_no_moveable_scope(self) -> None:
        """Flips if a versioned table may declare its scope moveable: then a key's versions could hold two scopes, and
        its scope is no fixed column (the declaration keeps a moveable one out all the same)."""
        with pytest.raises(ValueError, match="not ROW_STAMPED"):
            TableSchema(
                name="scope_pin_moveable",
                primary_key=("race_id", "geo_id"),
                columns=_COLUMNS,
                row_epoch=row_versions("state"),
                moveable=("state",),
                l1_eviction=TEST_L1_TABLE,
            )

    def test_a_fixed_column_is_never_the_key_or_its_flags(self) -> None:
        """Flips if a declaration naming a key column, the epoch or the gone flag as fixed is taken."""
        with pytest.raises(ValueError, match="past the key"):
            VersionedTable("t", ("race_id",), "epoch", "gone", fixed=("race_id",))


class TestTheSplit:
    def test_key_pins_within_takes_the_fixed_scope_and_leaves_the_rest(self) -> None:
        """Flips if the scope pin is left outside, or a column past the key and the scope goes inside."""
        fixed = VersionedTable("t", ("race_id", "geo_id"), "epoch", "gone", fixed=("state",))
        within, values = key_pins_within(fixed, {"state": "PA", "v": 2, "race_id": 1})
        assert within == ['t."state" = $1', 't."race_id" = $2']
        assert values == ["PA", 1]

    def test_a_scope_not_declared_fixed_stays_outside(self) -> None:
        """Flips if a scope no declaration fixes (a moveable one would be such) is judged inside: a key that moved
        would answer its older version of the pinned scope."""
        moveable = VersionedTable("t", ("race_id", "geo_id"), "epoch", "gone")
        assert key_pins_within(moveable, {"state": "PA"}) == ([], [])
        built = newest_page_statement(
            moveable,
            "t",
            columns=None,
            by={"state": "PA"},
            order=[Sort("race_id"), Sort("geo_id")],
            keys=10,
            past=None,
            by_row=True,
        )
        assert built is not None
        assert '"state"' not in _chosen(built[0])

    def test_a_page_statement_judges_the_fixed_scope_inside(self) -> None:
        """Flips if a key-ordered page leaves the fixed scope's pin outside its DISTINCT ON."""
        fixed = VersionedTable("t", ("race_id", "geo_id"), "epoch", "gone", fixed=("state",))
        built = newest_page_statement(
            fixed,
            "t",
            columns=None,
            by={"state": "PA"},
            order=[Sort("race_id"), Sort("geo_id")],
            keys=10,
            past=None,
            by_row=True,
        )
        assert built is not None
        assert 't."state" = $1' in _chosen(built[0])


class TestReadsJudgeTheScopeInside:
    async def test_counts_by_a_state(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a state's count reads its relation with the state pin outside (every state's versions read), or
        counts an older version or a gone key."""
        reader = _L3(db)
        assert await counts_by(reader, _TABLE, column="geo_id", by={"state": "PA"}) == {"a": 1}
        assert '"state"' in _chosen(reader.statements[-1])

    async def test_a_column_past_the_key_and_scope_stays_outside(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a pin on a column past the key and the scope is judged inside (an older version holding it would
        be read: race 1 a's v = 1 is its older version)."""
        reader = _L3(db)
        assert await counts_by(reader, _TABLE, column="geo_id", by={"state": "PA", "v": 1}) == {}
        chosen = _chosen(reader.statements[-1])
        assert '"state"' in chosen and '"v"' not in chosen

    async def test_held_in_by_state_and_geography(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a state's geographies read (reports' held_in: read_l3_rows, its where and where_any) leaves the
        state pin outside its pages' DISTINCT ON, or answers an older version or a gone key."""
        reader = _L3(db)
        where_any = {"geo_id": ("text[]", ["a", "b"])}
        names = [column.name for column in _COLUMNS]
        rows = await read_l3_rows(
            reader, _TABLE, names, ["race_id", "geo_id"], where={"state": "PA"}, where_any=where_any
        )
        assert _read(rows) == {(1, "a"): 2}
        chosen = _chosen(reader.statements[-1])
        assert '"state" = $1' in chosen and '"geo_id" = ANY(' in chosen
        arrow = await read_l3_arrow(
            reader, _TABLE, _COLUMNS, ["race_id", "geo_id"], where={"state": "PA"}, where_any=where_any
        )
        assert _read(arrow.to_pylist()) == {(1, "a"): 2}
        assert '"state" = $1' in _chosen(reader.statements[-1])

    async def test_a_keyset_page_of_a_state(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a key-ordered page of a state leaves the state pin outside its DISTINCT ON."""
        reader = _L3(db)
        page = await keyset_page(
            reader,
            _TABLE,
            columns=["race_id", "geo_id", "v"],
            by={"state": "PA"},
            order=[Sort("race_id", cast="integer"), Sort("geo_id")],
            limit=10,
        )
        assert _read(page.rows) == {(1, "a"): 2}
        assert '"state"' in _chosen(reader.statements[-1])

    async def test_a_point_read_of_a_key_and_its_state(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a point read pinning the key and its state leaves the state outside, or answers an older version
        or a gone key."""
        reader = _L3(db)
        rows = await read_by(reader, _TABLE, by={"race_id": 1, "geo_id": "a", "state": "PA"}, columns=["v"])
        assert [int(row["v"]) for row in rows] == [2]
        assert '"state"' in _chosen(reader.statements[-1])
        assert await read_by(reader, _TABLE, by={"race_id": 1, "geo_id": "b", "state": "PA"}, columns=["v"]) == []

    async def test_a_keyed_batch_with_a_state_pin(
        self, db: duckdb.DuckDBPyConnection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Flips if a keyed batch (YugabyteDB's batched DISTINCT ON) pinning a state leaves it outside."""

        async def yugabyte(conn: Any) -> bool:
            return True

        monkeypatch.setattr(keyed_rows, "reads_yugabyte", yugabyte)
        reader = _L3(db)
        keys = KeyBatch(("race_id", "geo_id"), [[1, 1, 2], ["a", "b", "a"]], ("integer", "text"))
        rows = await read_keyed(reader, _TABLE, keys=keys, by={"state": "PA"}, columns=["race_id", "geo_id", "v"])
        assert _read(rows) == {(1, "a"): 2}
        assert '"state"' in _chosen(reader.statements[-1])


#: the trials of the property, each a fresh table
_TRIALS = 100


class _Draws:
    """a fixed stream of pseudo-random draws (a linear congruential generator, its seed passed in): the property's
    tables are the same every run, so the test's result is cached by its code alone (no ``random`` draw)."""

    def __init__(self, seed: int) -> None:
        self._state = seed

    def below(self, bound: int) -> int:
        """a draw in ``[0, bound)``."""
        self._state = (self._state * 6364136223846793005 + 1442695040888963407) % 2**64
        return (self._state >> 33) % bound


def _versions(draws: _Draws, *, one_scope: bool) -> pa.Table:
    """a drawn versioned table's rows: keys ``(race_id, geo_id)``, each with 1-5 versions (about a quarter of them gone
    markers) at distinct epochs. ``one_scope``: every version of a key holds the key's one state (the contract); else a
    version may hold another."""
    states = ["PA", "OH", "VA"]
    rows: list[tuple[int, str, str, int | None, int, bool]] = []
    epoch = 0
    for race in range(1 + draws.below(8)):
        for geo in ["a", "b", "c", "d"][: 1 + draws.below(4)]:
            home = states[draws.below(3)]
            for _ in range(1 + draws.below(5)):
                state = home if one_scope else states[draws.below(3)]
                gone = draws.below(4) == 0
                # epochs distinct and in no key's order: a key's versions interleave with every other key's
                epoch += 1 + draws.below(50)
                rows.append((race, geo, state, None if gone else draws.below(4), epoch, gone))
    names = ["race_id", "geo_id", "state", "v", "epoch", "gone"]
    shuffled = sorted(rows, key=lambda row: (row[4] * 7919) % 10_007)
    return pa.table({name: [row[at] for row in shuffled] for at, name in enumerate(names)})


def _both_ways(conn: duckdb.DuckDBPyConnection, state: str, where_geo: list[str] | None) -> tuple[set[Any], set[Any]]:
    """the newest live versions of ``state`` (and of the geographies ``where_geo``), the state pin judged inside the
    DISTINCT ON and judged after."""
    table = VersionedTable("prop", ("race_id", "geo_id"), "epoch", "gone")
    geo = ['t."geo_id" = ANY($2::text[])'] if where_geo is not None else []
    params: list[Any] = [state] if where_geo is None else [state, where_geo]
    inside = newest_relation(table, "prop", within=['t."state" = $1', *geo])
    after = newest_relation(table, "prop", within=geo)
    select = "SELECT race_id, geo_id, state, v, epoch FROM ({}) AS r{}"
    first = conn.execute(select.format(inside, ""), params).fetchall()
    second = conn.execute(select.format(after, ' WHERE r."state" = $1'), params).fetchall()
    return set(first), set(second)


class TestThePinInsideIsSafe:
    def test_judged_inside_or_after_a_key_of_one_scope_reads_the_same(self) -> None:
        """the claim the split rests on, over random tables whose keys each hold one scope: a state pin judged inside
        the DISTINCT ON answers the same newest live versions as judged after. Flips if any trial differs."""
        draws = _Draws(20261126)
        for trial in range(_TRIALS):
            conn = duckdb.connect()
            versions = _versions(draws, one_scope=True)
            conn.register("versions", versions)
            conn.execute("CREATE TABLE prop AS SELECT * FROM versions")
            for state in ("PA", "OH", "VA"):
                for geos in (None, ["a", "c"]):
                    inside, after = _both_ways(conn, state, geos)
                    assert inside == after, f"trial {trial}, {state}, {geos}: {inside} != {after}"
            conn.close()

    def test_a_key_whose_versions_hold_two_scopes_breaks_it(self) -> None:
        """why the contract matters: a key whose older version held the pinned state and whose newest holds another
        is read inside (its older version) and not after. Flips if the property's tables could not tell the two
        apart, so the property above would prove nothing."""
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE prop (race_id INTEGER, geo_id VARCHAR, state VARCHAR, v INTEGER, epoch BIGINT, gone BOOLEAN)"
        )
        conn.execute("INSERT INTO prop VALUES (1, 'a', 'PA', 1, 10, false), (1, 'a', 'OH', 2, 20, false)")
        inside, after = _both_ways(conn, "PA", None)
        assert inside == {(1, "a", "PA", 1, 10)} and after == set()
        conn.close()
        draws = _Draws(20261126)
        differed = 0
        for _ in range(_TRIALS):
            conn = duckdb.connect()
            conn.register("versions", _versions(draws, one_scope=False))
            conn.execute("CREATE TABLE prop AS SELECT * FROM versions")
            differed += any(
                _both_ways(conn, state, None)[0] != _both_ways(conn, state, None)[1] for state in ("PA", "OH")
            )
            conn.close()
        assert differed > _TRIALS // 2


class TestAPageInItsOwnOrder:
    """a keyset read of a versioned table paging in an order of its own (a copy's whole read of a state: the state,
    then the state index's columns, then the rest of the key), not the key's: each page's ``LIMIT`` inside the
    ``DISTINCT ON`` must cut the keys in the page's order, the one its cursor steps in."""

    @staticmethod
    def _night(conn: duckdb.DuckDBPyConnection) -> dict[tuple[int, str], int]:
        """40 keys, two geographies of 20 races, the races' states alternating, each key two versions (the newer
        kept): the key's order (race, geography) and the page's (state, geography, race) differ at every step."""
        values = []
        for race in range(20):
            state = "PA" if race % 2 == 0 else "OH"
            for geo in ("a", "b"):
                values += [f"({race}, '{geo}', '{state}', {race}, 10, false)"]
                values += [f"({race}, '{geo}', '{state}', {race + 100}, 20, false)"]
        conn.execute(f"INSERT INTO {_TABLE} (race_id, geo_id, state, v, epoch, gone) VALUES {', '.join(values)}")
        return {(race, geo): race + 100 for race in range(20) for geo in ("a", "b")}

    async def test_every_key_is_read_once_paging_by_the_scope_then_the_key(self) -> None:
        """Flips if a page's LIMIT cuts the keys in the key's order while its cursor steps in its own: the second page
        skipped every key past the first's last in the page's order but not the key's (reports' copy held 1,550 of a
        state's 2,400 geography rows, 2026-11-26)."""
        conn = duckdb.connect()
        conn.execute(f"CREATE TABLE {_TABLE} {_DDL}")
        conn.execute(
            "CREATE TABLE threetears_run_markers (table_name VARCHAR, scope VARCHAR, prefix VARCHAR, epoch BIGINT)"
        )
        try:
            expected = self._night(conn)
            reader = _L3(conn)
            types = {"race_id": pa.int32(), "geo_id": pa.string(), "state": pa.string(), "v": pa.int32()}
            pages = [
                page
                async for page in keyset_arrow_pages(
                    reader,
                    f"SELECT race_id, geo_id, state, v FROM {_TABLE}",
                    ["state", "geo_id", "race_id"],
                    types=types,
                    page_size=3,
                    table=_TABLE,
                )
            ]
            assert _read(pa.concat_tables(pages).to_pylist()) == expected
            assert all('DISTINCT ON (t."state", t."geo_id", t."race_id")' in sql for sql in reader.statements[1:])
        finally:
            conn.close()

    def test_an_order_past_the_key_and_the_fixed_columns_is_refused(self) -> None:
        """Flips if a newest read may cut its keys by a column a key's versions may hold differently (two rows of one
        key read), or by an order missing a key column (one key's versions read as two)."""
        fixed = VersionedTable("t", ("race_id", "geo_id"), "epoch", "gone", fixed=("state",))
        with pytest.raises(ValueError, match="orders by its key"):
            newest_relation(fixed, "t", limit="10", order=("v", "race_id", "geo_id"))
        with pytest.raises(ValueError, match="orders by its key"):
            newest_relation(fixed, "t", limit="10", order=("state", "race_id"))
        assert 'DISTINCT ON (t."state", t."race_id", t."geo_id")' in newest_relation(
            fixed, "t", limit="10", order=("state", "race_id", "geo_id")
        )
