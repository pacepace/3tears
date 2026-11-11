"""every keyset pager returns the whole relation when the hub cuts its replies by their bytes.

A hub with a byte cap (``max_reply_bytes``) answers a page with the rows whose JSON fits, ``truncated`` set: the page
comes back short of its sentinel, so the row count alone reads like the relation's end. The stand-in here runs each
statement on DuckDB and cuts the answer as the hub's broker does (the first row always, then each row while the
rows' pydantic JSON, compact UTF-8, fits). For each pager (``read_all``, ``read_counted``, ``read_parts``, ``read_partitions``), under caps
from a few rows to one row a reply, the rows returned are the rows an uncut hub returns.
"""

from __future__ import annotations

import re
from typing import Any

import duckdb
import pydantic_core
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
            grown = size + len(pydantic_core.to_json(row)) + (1 if kept else 0)
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


class _WideByteCappedHub(_ByteCappedHub):
    """a byte-capped hub over 10,000 rows, counting the statements it answers and the rows each asked (its LIMIT)."""

    def __init__(self, max_bytes: int | None) -> None:
        super().__init__(max_bytes)
        self.statements = 0
        self.asked: list[int] = []
        self._db.execute("DELETE FROM s.results")
        self._db.executemany(
            "INSERT INTO s.results VALUES (?, ?, ?)",
            [(_STATES[index % 3], index, "x" * 40) for index in range(10_000)],
        )

    async def query(
        self, datasource_name: str, query: str, params: list[Any] | None = None, **kwargs: Any
    ) -> DatasourceQueryResult:
        self.statements += 1
        limit = re.search(r"LIMIT\s+(\d+)", query)
        self.asked.append(int(limit.group(1)) if limit else -1)
        return await super().query(datasource_name, query, params, **kwargs)


def _a_reply_carries(hub: _WideByteCappedHub) -> int:
    """the rows of the relation one reply of ``hub`` carries, at the width the hub measures them."""
    rows = [{"state": _STATES[index % 3], "race": index, "note": "x" * 40} for index in range(10_000)]
    assert hub.max_bytes is not None
    return hub.max_bytes * len(rows) // len(pydantic_core.to_json(rows))


async def test_pages_after_the_first_fill_the_reply_at_the_width_the_hub_measures() -> None:
    """Flips if a page is sized at a width other than the hub's own (pydantic's compact JSON) or fills much less
    than the reply: every page after the first asks at least 90% of what one reply carries, and none is cut."""
    hub = _WideByteCappedHub(max_bytes=400_000)

    got = await read_all(hub, "ds", columns=_COLUMNS, relation="s.results", key=["race"])  # type: ignore[arg-type]

    assert len(got) == 10_000
    carries = _a_reply_carries(hub)
    assert all(asked >= 0.9 * carries for asked in hub.asked[1:-1]), (carries, hub.asked)
    assert hub.cut == 0, hub.asked


async def test_a_read_after_an_announced_cap_sizes_its_first_page_from_it() -> None:
    """Flips if the first page of a read on a client a byte-capped hub has answered asks the fixed page size again:
    under a cap one reply holds the whole relation in, the first read takes its first page and one sized page, and
    the second asks what one reply carried at once and reads it whole in one statement."""
    hub = _WideByteCappedHub(max_bytes=2_000_000)
    await read_all(hub, "ds", columns=_COLUMNS, relation="s.results", key=["race"])  # type: ignore[arg-type]
    assert hub.statements == 2, hub.asked
    hub.statements = 0
    hub.asked.clear()

    got = await read_all(hub, "ds", columns=_COLUMNS, relation="s.results", key=["race"])  # type: ignore[arg-type]

    assert sorted(row["race"] for row in got) == list(range(10_000))
    assert hub.asked[0] >= 0.9 * _a_reply_carries(hub), hub.asked
    assert hub.statements == 1, hub.asked


async def test_a_hub_that_drops_its_byte_cap_is_read_whole_at_the_asked_size() -> None:
    """Flips if a remembered byte-cap page size is sent to a hub that now cuts by rows (its row cap would eat the
    sentinel) or refused: the read starts over at the asked size and returns every row."""
    hub = _WideByteCappedHub(max_bytes=400_000)
    await read_all(hub, "ds", columns=_COLUMNS, relation="s.results", key=["race"])  # type: ignore[arg-type]
    hub.max_bytes = None
    hub.asked.clear()

    got = await read_all(hub, "ds", columns=_COLUMNS, relation="s.results", key=["race"])  # type: ignore[arg-type]

    assert sorted(row["race"] for row in got) == list(range(10_000))
    assert all(asked <= 1_000 for asked in hub.asked[1:]), hub.asked


async def test_a_byte_capped_hub_is_read_in_pages_as_large_as_its_replies() -> None:
    """the pager sizes its pages from the announced cap: every row of 10,000, in far fewer statements than the
    row cap's 900-row pages take (12)."""
    hub = _WideByteCappedHub(max_bytes=400_000)

    got = await read_all(hub, "ds", columns=_COLUMNS, relation="s.results", key=["race"])  # type: ignore[arg-type]

    assert len(got) == 10_000
    assert sorted(row["race"] for row in got) == list(range(10_000))
    assert hub.statements <= 5, hub.statements


async def test_a_page_at_the_row_cap_is_taken_by_a_byte_capped_hub() -> None:
    hub = _WideByteCappedHub(max_bytes=400_000)

    got = await read_all(
        hub,  # type: ignore[arg-type]
        "ds",
        columns=_COLUMNS,
        relation="s.results",
        key=["race"],
        page_size=2_000,
    )

    assert len(got) == 10_000


async def test_a_page_at_the_row_cap_is_still_refused_by_a_hub_with_no_byte_cap() -> None:
    hub = _ByteCappedHub(max_bytes=None)

    with pytest.raises(ValueError, match="UNDER the hub's row cap"):
        await read_all(hub, "ds", columns=_COLUMNS, relation="s.results", key=["state", "race"], page_size=1_000)  # type: ignore[arg-type]
