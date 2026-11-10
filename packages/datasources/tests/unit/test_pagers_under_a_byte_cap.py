"""every keyset pager returns the whole relation when the hub cuts its replies by their bytes.

A hub with a byte cap (``max_reply_bytes``) answers a page with the rows whose JSON fits, ``truncated`` set: the page
comes back short of its sentinel, so the row count alone reads like the relation's end. The stand-in here runs each
statement on DuckDB and cuts the answer as the hub's ``RowBudget`` does (the first row always, then each row while
the JSON array fits). For each pager (``read_all``, ``read_counted``, ``read_parts``, ``read_partitions``), under caps
from a few rows to one row a reply, the rows returned are the rows an uncut hub returns.
"""

from __future__ import annotations

import json
from typing import Any

import duckdb
import pytest
from uuid_utils.compat import uuid7

from threetears.datasources.partitioned_read import read_partitions, read_parts
from threetears.datasources.query_client import DatasourceQueryResult, IncompleteReadError, read_all, read_counted

_STATES = ["AK", "AL", "AZ"]


def _rows() -> list[tuple[str, int, str]]:
    """each state's 40 races, each race's note a different length (so a reply's rows are of uneven bytes)."""
    return [(state, race, "x" * (7 * race % 31)) for state in _STATES for race in range(40)]


class _ByteCappedHub:
    """the read rail over a DuckDB relation, every reply cut to ``max_bytes`` of JSON rows (None: never cut)."""

    def __init__(self, max_bytes: int | None) -> None:
        self.max_bytes = max_bytes
        self.cut = 0
        self._db = duckdb.connect()
        self._db.execute("CREATE SCHEMA s")
        self._db.execute("CREATE TABLE s.results (state VARCHAR, race INTEGER, note VARCHAR)")
        self._db.executemany("INSERT INTO s.results VALUES (?, ?, ?)", _rows())

    def forwarded_identity_token(self) -> str:
        return "fake-identity-token"

    async def query(
        self, datasource_name: str, query: str, params: list[Any] | None = None, **_: Any
    ) -> DatasourceQueryResult:
        cursor = self._db.execute(query, list(params or []))
        names = [column[0] for column in cursor.description]
        rows = [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
        kept, truncated = self._cut(rows)
        return DatasourceQueryResult(
            rows=kept,
            row_count=len(kept),
            truncated=truncated,
            correlation_id=uuid7(),
            max_reply_bytes=self.max_bytes,
        )

    def _cut(self, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
        """the rows whose JSON array fits the cap, the first always, as the hub's RowBudget keeps them."""
        if self.max_bytes is None:
            return rows, False
        kept: list[dict[str, Any]] = []
        size = 2
        for row in rows:
            grown = size + len(json.dumps(row)) + (1 if kept else 0)
            if kept and grown > self.max_bytes:
                self.cut += 1
                return kept, True
            kept.append(row)
            size = grown
        return kept, False


def _sorted(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return sorted((row["state"], row["race"], row["note"]) for row in rows)


_EVERY_ROW = sorted(_rows())
_COLUMNS = ["state", "race", "note"]
_KEY = ["state", "race"]


@pytest.mark.parametrize("max_bytes", [None, 2000, 400, 120, 60])
async def test_read_all_reads_every_row_through_cut_replies(max_bytes: int | None) -> None:
    """Flips if a page the hub cut by its bytes is taken as the relation's end (a prefix returned as the whole)."""
    hub = _ByteCappedHub(max_bytes)
    rows = await read_all(hub, "w", columns=_COLUMNS, relation="s.results", key=_KEY, page_size=50)  # type: ignore[arg-type]
    assert _sorted(rows) == _EVERY_ROW
    assert max_bytes is None or max_bytes >= 2000 or hub.cut > 0


@pytest.mark.parametrize("max_bytes", [400, 60])
async def test_read_counted_reads_every_row_through_cut_replies(max_bytes: int) -> None:
    """Flips if a counted read stops at a cut reply (its count would then refuse a read that is in fact whole)."""
    hub = _ByteCappedHub(max_bytes)
    rows = await read_counted(
        hub,  # type: ignore[arg-type]
        "w",
        columns=_COLUMNS,
        relation="s.results",
        key=["race"],
        rows=40,
        where={"state": "AZ"},
        page_size=50,
    )
    assert _sorted(rows) == [row for row in _EVERY_ROW if row[0] == "AZ"]


@pytest.mark.parametrize("max_bytes", [400, 60])
async def test_read_parts_reads_every_part_through_cut_replies(max_bytes: int) -> None:
    """Flips if a part read by keyset pages comes back short when the hub cuts its replies."""
    hub = _ByteCappedHub(max_bytes)
    parts = await read_parts(
        hub,  # type: ignore[arg-type]
        "w",
        columns=_COLUMNS,
        relation="s.results",
        key=["race"],
        parts=[{"state": state} for state in _STATES],
        page_size=50,
    )
    assert _sorted([row for part in parts for row in part]) == _EVERY_ROW


@pytest.mark.parametrize("max_bytes", [400, 60])
async def test_read_partitions_reads_every_page_through_cut_replies(max_bytes: int) -> None:
    """Flips if a page between two page starts comes back as a prefix when the hub cuts its reply."""
    hub = _ByteCappedHub(max_bytes)
    got: list[dict[str, Any]] = []
    async for _, rows in read_partitions(
        hub,  # type: ignore[arg-type]
        "w",
        columns=_COLUMNS,
        relation="s.results",
        key=["state", "race"],
        partition_by="state",
        parts=_STATES,
        page_size=10,
        counts={state: 40 for state in _STATES},
    ):
        got.extend(rows)
    assert _sorted(got) == _EVERY_ROW


async def test_a_key_group_wider_than_one_reply_raises_rather_than_dropping_rows() -> None:
    """Flips if a reply cut within one key value is returned as if it held the value's every row."""
    hub = _ByteCappedHub(60)
    with pytest.raises(IncompleteReadError, match="wider than one reply"):
        await read_all(hub, "w", columns=_COLUMNS, relation="s.results", key=["state"], page_size=50)  # type: ignore[arg-type]
