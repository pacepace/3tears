"""``read_all`` with set and typed filters, and ``read_counted``, which proves a read against a count.

The warehouse here RUNS the statements: every page and every fingerprint is executed by SQLite over
a real table, after the ``$N`` placeholders are renumbered for it. So a filter that is written wrong,
bound in the wrong order, or bound as the wrong type changes what comes back, rather than agreeing
with a fake that parses what it expects to see.

The relation is shaped like the ENR reports' LSF projections: a run's rows by contest, geography
and quantity, read by a set of integer run ids, two sets of text values and one state, with rows
that cannot be keyed (a NULL contest or geography) left out.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from uuid import uuid7

import pytest

from threetears.datasources.drivers.sql_fragments import build_filter
from threetears.datasources.query_client import (
    DatasourceQueryResult,
    IncompleteReadError,
    RelationFingerprintRequest,
    RelationFingerprintResult,
    read_all,
    read_counted,
)

_RELATION = "projections"
_COLUMNS = ("state_code", "run_id", "contest", "geo_level", "geo_id", "quantity", "value")
_KEY = ("run_id", "contest", "geo_level", "geo_id", "quantity")


def _rows() -> list[dict[str, Any]]:
    """three runs over two states, every geography level and quantity, plus rows that cannot be keyed.

    :return: the relation's rows
    :rtype: list[dict[str, Any]]
    """
    rows: list[dict[str, Any]] = []
    for run_id in (101, 102, 103):
        for state in ("TX", "DE"):
            for contest in ("pres", "sen"):
                for geo_level, geo_ids in (("state", ("00",)), ("cd", ("01", "02")), ("county", ("c1",))):
                    for geo_id in geo_ids:
                        for quantity in ("dem_2way", "counted_dem_2way", "turnout"):
                            rows.append(
                                {
                                    "state_code": state,
                                    "run_id": run_id,
                                    "contest": contest,
                                    "geo_level": geo_level,
                                    "geo_id": geo_id,
                                    "quantity": quantity,
                                    "value": float(run_id),
                                }
                            )
        rows.append(
            {
                "state_code": "TX",
                "run_id": run_id,
                "contest": None,
                "geo_level": "state",
                "geo_id": "00",
                "quantity": "dem_2way",
                "value": 0.0,
            }
        )
    return rows


def _sqlite(sql: str) -> str:
    """a canonical ``$N`` statement as SQLite numbers its parameters (``?N``).

    :param sql: the statement
    :ptype sql: str
    :return: the statement for SQLite
    :rtype: str
    """
    return re.sub(r"\$(\d+)", r"?\1", sql)


# parity-with: threetears.datasources.query_client.DatasourceQueryClient
class _SqlWarehouse:
    """a warehouse that executes what it is sent, over one SQLite table.

    :param rows: the relation's rows
    :ptype rows: list[dict[str, Any]]
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._db = sqlite3.connect(":memory:")
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            f"CREATE TABLE {_RELATION} (state_code TEXT, run_id INTEGER, contest TEXT, geo_level TEXT, "
            "geo_id TEXT, quantity TEXT, value REAL)"
        )
        self.insert(rows)
        self.pages: list[str] = []
        self.page_params: list[list[Any]] = []
        self.fingerprints: list[dict[str, Any]] = []
        self.after_fingerprint: Callable[[_SqlWarehouse], None] | None = None

    def insert(self, rows: list[dict[str, Any]]) -> None:
        """add rows, as a writer would.

        :param rows: the rows
        :ptype rows: list[dict[str, Any]]
        :return: nothing
        :rtype: None
        """
        self._db.executemany(
            f"INSERT INTO {_RELATION} VALUES (:state_code, :run_id, :contest, :geo_level, :geo_id, :quantity, :value)",
            rows,
        )

    def forwarded_identity_token(self) -> str:
        """
        :return: a placeholder the reads never inspect
        :rtype: str
        """
        return "token"

    async def query(
        self, datasource_name: str, query: str, *, params: list[Any] | None = None, **_: Any
    ) -> DatasourceQueryResult:
        """run one page.

        :return: its rows
        :rtype: DatasourceQueryResult
        """
        bound = list(params or [])
        self.pages.append(query)
        self.page_params.append(bound)
        rows = [dict(row) for row in self._db.execute(_sqlite(query), bound).fetchall()]
        return DatasourceQueryResult(rows=rows, row_count=len(rows), truncated=False, correlation_id=uuid7())

    def _fingerprint(
        self, key: Sequence[str], where: Mapping[str, Any], where_in: Mapping[str, Sequence[Any]], group_by: str | None
    ) -> dict[Any, RelationFingerprintResult]:
        # the drivers' own filter builder, as the hub's fingerprint would use it
        filters, values = build_filter(where, where_in)
        grouping = group_by or "1"
        statement = f"SELECT {grouping} AS g, {', '.join(key)} FROM {_RELATION}{filters} ORDER BY {', '.join(key)}"  # noqa: S608
        groups: dict[Any, list[str]] = {}
        for row in self._db.execute(_sqlite(statement), values).fetchall():
            groups.setdefault(row["g"], []).append("\x1f".join(str(row[c]) for c in key))
        return {
            group: RelationFingerprintResult(
                row_count=len(keys), digest=hashlib.sha256("\x1e".join(keys).encode()).hexdigest()
            )
            for group, keys in groups.items()
        }

    async def relation_fingerprint(
        self, datasource_name: str, *, relation: str, key: Sequence[str], where: Any = None, **_: Any
    ) -> RelationFingerprintResult:
        """fingerprint the rows the equality filters keep.

        :return: the fingerprint
        :rtype: RelationFingerprintResult
        """
        self.fingerprints.append({"where": dict(where or {}), "group_by": None})
        whole = self._fingerprint(key, dict(where or {}), {}, None)
        result = whole.get(1, RelationFingerprintResult(row_count=0, digest=""))
        if self.after_fingerprint is not None:
            self.after_fingerprint(self)
        return result

    async def relation_fingerprint_groups(
        self,
        datasource_name: str,
        *,
        relation: str,
        key: Sequence[str],
        group_by: str,
        where: Any = None,
        where_in: Any = None,
        **_: Any,
    ) -> dict[Any, RelationFingerprintResult]:
        """fingerprint each group of the rows the filters keep.

        :return: group -> fingerprint
        :rtype: dict[Any, RelationFingerprintResult]
        """
        self.fingerprints.append({"where": dict(where or {}), "where_in": dict(where_in or {}), "group_by": group_by})
        groups = self._fingerprint(key, dict(where or {}), dict(where_in or {}), group_by)
        if self.after_fingerprint is not None:
            self.after_fingerprint(self)
        return groups


def _expected(
    rows: list[dict[str, Any]], runs: set[int], levels: set[str], quantities: set[str], state: str
) -> list[dict[str, Any]]:
    kept = [
        r
        for r in rows
        if r["run_id"] in runs
        and r["geo_level"] in levels
        and r["quantity"] in quantities
        and r["state_code"] == state
        and r["contest"] is not None
        and r["geo_id"] is not None
    ]
    return sorted(kept, key=lambda r: tuple(r[c] for c in _KEY))


_SETS: Mapping[str, Sequence[Any]] = {
    "run_id": [101, 103],
    "geo_level": ["state", "cd"],
    "quantity": ["dem_2way", "counted_dem_2way"],
}


class TestReadAllTakesSetAndTypedFilters:
    """``where_in`` and integer values name the part read; the grouped fingerprint proves it."""

    async def test_every_row_of_the_set_filtered_part_comes_back_across_pages(self) -> None:
        rows = [r for r in _rows() if r["contest"] is not None]
        warehouse = _SqlWarehouse(rows)

        read = await read_all(
            warehouse,  # type: ignore[arg-type]
            "lsf",
            columns=_COLUMNS,
            relation=_RELATION,
            key=_KEY,
            where={"state_code": "TX"},
            where_in=_SETS,
            page_size=5,
        )

        assert read == _expected(rows, {101, 103}, {"state", "cd"}, {"dem_2way", "counted_dem_2way"}, "TX")
        assert len(warehouse.pages) > 2
        # the run ids travel as integers: a strictly typed engine (asyncpg) refuses text for a bigint
        assert all(type(value) is int for value in warehouse.page_params[0][1:3])
        # proven by the grouped fingerprint, grouped by the first set-filtered column, before and after
        assert [f["group_by"] for f in warehouse.fingerprints] == ["run_id", "run_id"]
        assert warehouse.fingerprints[0]["where_in"]["run_id"] == (101, 103)

    async def test_an_integer_equality_filter_is_bound_as_an_integer(self) -> None:
        rows = [r for r in _rows() if r["contest"] is not None]
        warehouse = _SqlWarehouse(rows)

        read = await read_all(
            warehouse,  # type: ignore[arg-type]
            "lsf",
            columns=_COLUMNS,
            relation=_RELATION,
            key=_KEY,
            where={"run_id": 102, "state_code": "DE"},
            page_size=7,
        )

        assert {r["run_id"] for r in read} == {102}
        assert len(read) == len([r for r in rows if r["run_id"] == 102 and r["state_code"] == "DE"])
        assert warehouse.fingerprints[0] == {"where": {"run_id": 102, "state_code": "DE"}, "group_by": None}

    async def test_a_group_that_changed_under_the_read_is_refused(self) -> None:
        rows = [r for r in _rows() if r["contest"] is not None]
        warehouse = _SqlWarehouse(rows)
        added = {**rows[0], "state_code": "TX", "run_id": 101, "geo_id": "99", "geo_level": "cd"}

        def write_once(w: _SqlWarehouse) -> None:
            w.after_fingerprint = None
            w.insert([added])

        warehouse.after_fingerprint = write_once

        with pytest.raises(IncompleteReadError):
            await read_all(
                warehouse,  # type: ignore[arg-type]
                "lsf",
                columns=_COLUMNS,
                relation=_RELATION,
                key=_KEY,
                where={"state_code": "TX"},
                where_in=_SETS,
                page_size=5,
            )

    async def test_an_empty_set_reads_nothing_and_asks_nothing(self) -> None:
        warehouse = _SqlWarehouse(_rows())

        read = await read_all(
            warehouse,  # type: ignore[arg-type]
            "lsf",
            columns=_COLUMNS,
            relation=_RELATION,
            key=_KEY,
            where_in={"run_id": []},
        )

        assert read == []
        assert warehouse.pages == [] and warehouse.fingerprints == []

    @pytest.mark.parametrize(
        ("where", "where_in"),
        [({"run_id": True}, None), ({"run_id": 1.5}, None), (None, {"run_id": [101, None]}), (None, {"state": "TX"})],
    )
    async def test_a_value_that_is_neither_text_nor_an_integer_is_refused(self, where: Any, where_in: Any) -> None:
        warehouse = _SqlWarehouse(_rows())

        with pytest.raises(ValueError):
            await read_all(
                warehouse,  # type: ignore[arg-type]
                "lsf",
                columns=_COLUMNS,
                relation=_RELATION,
                key=_KEY,
                where=where,
                where_in=where_in,
            )
        assert warehouse.pages == []


class TestReadCountedProvesTheRowsAgainstTheCount:
    """``read_counted`` pages the part and returns it only when it holds exactly the rows counted."""

    async def _read(self, warehouse: _SqlWarehouse, rows: int, **options: Any) -> list[dict[str, Any]]:
        return await read_counted(
            warehouse,  # type: ignore[arg-type]
            "lsf",
            columns=_COLUMNS,
            relation=_RELATION,
            key=_KEY,
            rows=rows,
            **{
                "where": {"state_code": "TX"},
                "where_in": _SETS,
                "not_null": ("contest", "geo_id"),
                "page_size": 4,
                **options,
            },
        )

    async def test_the_counted_rows_come_back_across_pages_with_the_unkeyable_rows_left_out(self) -> None:
        rows = _rows()
        warehouse = _SqlWarehouse(rows)
        expected = _expected(rows, {101, 103}, {"state", "cd"}, {"dem_2way", "counted_dem_2way"}, "TX")

        read = await self._read(warehouse, len(expected))

        assert read == expected
        assert len(warehouse.pages) > 2
        assert "contest IS NOT NULL AND geo_id IS NOT NULL" in warehouse.pages[0]
        # no fingerprint: the caller's count is the proof
        assert warehouse.fingerprints == []
        # every placeholder of a later page names a parameter, the filters' first and the cursor's after
        later, params = warehouse.pages[1], warehouse.page_params[1]
        assert sorted({int(n) for n in re.findall(r"\$(\d+)", later)}) == list(range(1, len(params) + 1))

    async def test_a_read_short_of_the_count_raises(self) -> None:
        rows = _rows()
        warehouse = _SqlWarehouse(rows)
        expected = _expected(rows, {101, 103}, {"state", "cd"}, {"dem_2way", "counted_dem_2way"}, "TX")

        with pytest.raises(IncompleteReadError, match=f"read {len(expected)} rows of the {len(expected) + 1}"):
            await self._read(warehouse, len(expected) + 1)

    async def test_a_read_past_the_count_raises(self) -> None:
        rows = _rows()
        warehouse = _SqlWarehouse(rows)
        expected = _expected(rows, {101, 103}, {"state", "cd"}, {"dem_2way", "counted_dem_2way"}, "TX")

        with pytest.raises(IncompleteReadError, match="more rows than counted"):
            await self._read(warehouse, len(expected) - 1)

    async def test_without_the_null_filter_a_row_the_key_cannot_page_past_fails_the_count(self) -> None:
        rows = _rows()
        warehouse = _SqlWarehouse(rows)
        expected = _expected(rows, {101, 103}, {"state", "cd"}, {"dem_2way", "counted_dem_2way"}, "TX")

        # the count leaves the NULL-contest rows out; a read that does not is not the counted part
        with pytest.raises(IncompleteReadError):
            await self._read(warehouse, len(expected), not_null=())

    async def test_an_empty_set_is_zero_rows_and_no_statement(self) -> None:
        warehouse = _SqlWarehouse(_rows())

        assert await self._read(warehouse, 0, where_in={"run_id": []}) == []
        assert warehouse.pages == []
        with pytest.raises(IncompleteReadError):
            await self._read(warehouse, 3, where_in={"run_id": []})

    async def test_a_negative_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            await self._read(_SqlWarehouse(_rows()), -1)


def test_a_typed_filter_travels_as_the_type_it_is() -> None:
    request = RelationFingerprintRequest(
        relation="lsf.projections",
        key_columns=["run_id"],
        where={"run_id": 101, "state_code": "TX"},
        where_in={"run_id": [101, 102]},
        group_by="run_id",
    )
    wire = request.model_dump_json()

    # an integer stays a JSON number: a hub whose model takes text only refuses it rather than
    # coercing it, so a typed filter fails closed against a hub that predates it
    assert '"run_id":101' in wire and '"run_id":[101,102]' in wire
    back = RelationFingerprintRequest.model_validate_json(wire)
    assert back.where == {"run_id": 101, "state_code": "TX"}
    assert back.where_in == {"run_id": [101, 102]}
