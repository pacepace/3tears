"""A full or cold read selects a row's hash beside its columns: read_all, read_parts and read_partitions take a
:class:`~threetears.core.keyed.Selected` (``row_hash_selected(..., alias=None)``), fingerprint only the columns read
as written, and answer the same rows, each hash the one the database computes for it.

The stand-in hub runs every statement on an in-process DuckDB and fingerprints a part over the key columns it is
asked for, refusing any name that is not a plain identifier, as the hub's request model does.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any
from uuid import uuid4

import duckdb
import pytest

from threetears.core.keyed import (
    ROW_HASH_COLUMN,
    Dialect,
    HashColumn,
    HashKind,
    Selected,
    row_hash_selected,
    row_hash_sql,
)
from threetears.datasources.partitioned_read import read_partitions, read_parts
from threetears.datasources.query_client import DatasourceQueryResult, RelationFingerprintResult, read_all

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HASHED = [HashColumn("name"), HashColumn("votes", HashKind.INTEGER)]
_COLUMNS = ["state", "geo", "name", "votes"]


class _DuckHub:
    """a hub over DuckDB: queries run as asked; fingerprints digest the asked key columns of the named part."""

    def __init__(self, db: duckdb.DuckDBPyConnection) -> None:
        self._db = db
        self.fingerprinted: list[list[str]] = []

    def forwarded_identity_token(self) -> str:
        return "token"

    def _checked(self, key: Sequence[str]) -> str:
        names = list(key)
        if not names or not all(isinstance(name, str) and _IDENTIFIER.match(name) for name in names):
            raise ValueError(f"key_columns {names!r} are not plain SQL identifiers")
        self.fingerprinted.append(names)
        return ", ".join(names)

    async def query(
        self, datasource: str, query: str, params: list[Any] | None = None, *, user_identity_token: str | None = None
    ) -> DatasourceQueryResult:
        cursor = self._db.execute(query, list(params or []))
        names = [column[0] for column in cursor.description or []]
        rows = [dict(zip(names, found, strict=True)) for found in cursor.fetchall()]
        return DatasourceQueryResult(rows=rows, row_count=len(rows), truncated=False, correlation_id=uuid4())

    def _print(self, relation: str, columns: str, where: Mapping[str, Any]) -> RelationFingerprintResult:
        conditions = " AND ".join(f"{column} = ${index}" for index, column in enumerate(where, 1)) or "TRUE"
        found = self._db.execute(
            f"SELECT {columns} FROM {relation} WHERE {conditions} ORDER BY {columns}",  # noqa: S608 - test's own names
            list(where.values()),
        ).fetchall()
        return RelationFingerprintResult(row_count=len(found), digest=hashlib.sha256(repr(found).encode()).hexdigest())

    async def relation_fingerprint(
        self, datasource: str, *, relation: str, key: Sequence[str], where: Mapping[str, Any] | None = None, **_: Any
    ) -> RelationFingerprintResult:
        return self._print(relation, self._checked(key), dict(where or {}))

    async def relation_fingerprint_groups(
        self,
        datasource: str,
        *,
        relation: str,
        key: Sequence[str],
        group_by: str,
        where: Mapping[str, Any] | None = None,
        where_in: Mapping[str, Sequence[Any]] | None = None,
        **_: Any,
    ) -> dict[str | None, RelationFingerprintResult]:
        columns = self._checked(key)
        groups = [found for (found,) in self._db.execute(f"SELECT DISTINCT {group_by} FROM {relation}").fetchall()]  # noqa: S608
        wanted = set((where_in or {}).get(group_by, groups))
        return {
            group: self._print(relation, columns, {**dict(where or {}), group_by: group})
            for group in groups
            if group in wanted
        }


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE results (state VARCHAR, geo BIGINT, name VARCHAR, votes BIGINT, PRIMARY KEY (state, geo))"
    )
    conn.execute(
        "INSERT INTO results SELECT CASE i % 3 WHEN 0 THEN 'PA' WHEN 1 THEN 'OH' ELSE 'VA' END, i, "
        "CASE WHEN i % 7 = 0 THEN NULL ELSE 'n' || i END, CASE WHEN i % 5 = 0 THEN NULL ELSE i * 10 END "
        "FROM range(2500) r(i)"
    )
    yield conn
    conn.close()


def _expected(db: duckdb.DuckDBPyConnection, state: str | None = None) -> list[tuple[Any, ...]]:
    scope = "" if state is None else f" WHERE state = '{state}'"
    return sorted(
        db.execute(
            f"SELECT state, geo, name, votes, {row_hash_sql(_HASHED, Dialect.DUCKDB, alias=None)} FROM results{scope}"  # noqa: S608
        ).fetchall()
    )


def _held(rows: Sequence[Mapping[str, Any]]) -> list[tuple[Any, ...]]:
    return sorted(tuple(row[column] for column in [*_COLUMNS, ROW_HASH_COLUMN]) for row in rows)


_SELECTED: list[str | Selected] = [*_COLUMNS, row_hash_selected(_HASHED, Dialect.DUCKDB, alias=None)]


@pytest.mark.parametrize("prove", [False, True])
async def test_a_full_read_selects_each_row_s_hash(db: duckdb.DuckDBPyConnection, prove: bool) -> None:
    """Flips if read_all refuses a Selected column, hands it to the fingerprint, or answers other rows or hashes
    than the plain statement does."""
    hub = _DuckHub(db)
    rows = await read_all(
        hub,  # type: ignore[arg-type]
        "stand-in",
        columns=_SELECTED,
        relation="results",
        key=["state", "geo"],
        where={"state": "PA"},
        page_size=300,
        prove=prove,
    )
    assert _held(rows) == _expected(db, "PA")
    assert all(names == ["state", "geo"] for names in hub.fingerprinted) and bool(hub.fingerprinted) == prove


async def test_parts_read_whole_select_each_row_s_hash(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if read_parts refuses a Selected column or answers other rows or hashes than the plain statement."""
    parts = await read_parts(
        _DuckHub(db),  # type: ignore[arg-type]
        "stand-in",
        columns=_SELECTED,
        relation="results",
        key=["geo"],
        parts=[{"state": "OH"}, {"state": "VA"}],
        page_size=300,
    )
    assert [_held(rows) for rows in parts] == [_expected(db, "OH"), _expected(db, "VA")]


@pytest.mark.parametrize("named", [True, False])
async def test_a_read_by_partitions_selects_each_row_s_hash(db: duckdb.DuckDBPyConnection, named: bool) -> None:
    """Flips if read_partitions (named parts by key, or the fingerprinted proof) refuses a Selected column, hands it
    to a fingerprint, or answers other rows or hashes than the plain statement."""
    hub = _DuckHub(db)
    read = {
        part: rows
        async for part, rows in read_partitions(
            hub,  # type: ignore[arg-type]
            "stand-in",
            columns=_SELECTED,
            relation="results",
            key=["geo"],
            partition_by="state",
            parts=["OH", "PA", "VA"] if named else None,
            page_size=300,
        )
    }
    assert {part: _held(rows) for part, rows in read.items()} == {
        state: _expected(db, state) for state in ("OH", "PA", "VA")
    }
    assert all(names == _COLUMNS for names in hub.fingerprinted) and bool(hub.fingerprinted) != named
