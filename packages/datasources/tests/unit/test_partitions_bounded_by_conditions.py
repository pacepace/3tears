"""A partitioned read bounded by conditions holds the rows its count saw, however many are appended during the read.

A count taken before the read (a change statement's ``count(*)`` and ``max(stamp)``, in one statement) holds the rows
as they were. Read whole later, a part of an append-only relation holds the rows appended since too, and the read is
refused: the election replay of 2026-11-21 refused every cold start of the AP snapshots all night so. Bounded by the
stamp the count saw (``where``), the read holds exactly the counted rows; the rows appended after are the next load's,
past the mark. The stand-in runs each statement on DuckDB, appends rows between the page starts and the pages, and
cuts replies by their bytes as the hub does.
"""

from __future__ import annotations

from typing import Any

import duckdb
import pydantic_core
import pytest
from threetears.core.keyed import Fragment
from uuid_utils.compat import uuid7

from threetears.datasources.partitioned_read import read_partitions, read_partitions_arrow
from threetears.datasources.query_client import DatasourceQueryResult, IncompleteReadError

_STATES = ["AK", "AL", "AZ"]
_COLUMNS = ["state", "race", "uploaded"]
_KEY = ["race"]
_COUNTED_UP_TO = "2026-11-04 01:00:00"
#: rows at or before the stamp the count saw, and rows whose stamp is NULL, which the count held too
_BOUND = Fragment(
    sql="(uploaded IS NULL OR CAST(uploaded AS TIMESTAMP) <= CAST($1 AS TIMESTAMP))", params=(_COUNTED_UP_TO,)
)


class _AppendingHub:
    """the read rail over an append-only DuckDB relation: an upload lands after the read's page starts are asked."""

    def __init__(self, max_bytes: int | None = None) -> None:
        self.max_bytes = max_bytes
        self.appended = False
        self._db = duckdb.connect()
        self._db.execute("CREATE SCHEMA s")
        self._db.execute("CREATE TABLE s.snapshots (state VARCHAR, race INTEGER, uploaded VARCHAR)")
        rows: list[tuple[str, int, str | None]] = [
            (state, race, "2026-11-04 00:30:00") for state in _STATES for race in range(30)
        ]
        rows += [(state, 100, None) for state in _STATES]
        self._db.executemany("INSERT INTO s.snapshots VALUES (?, ?, ?)", rows)

    def forwarded_identity_token(self) -> str:
        return "fake-identity-token"

    def counts(self) -> dict[str, int]:
        """each state's rows, as the change statement counted them before the read."""
        found = self._db.execute("SELECT state, count(*) FROM s.snapshots GROUP BY state").fetchall()
        return {str(state): int(n) for state, n in found}

    def held(self, state: str) -> list[tuple[Any, ...]]:
        return self._db.execute(
            "SELECT state, race, uploaded FROM s.snapshots WHERE state = $1 ORDER BY race", [state]
        ).fetchall()

    async def query(
        self, datasource_name: str, query: str, params: list[Any] | None = None, **_: Any
    ) -> DatasourceQueryResult:
        del datasource_name
        cursor = self._db.execute(query, list(params or []))
        names = [column[0] for column in cursor.description]
        rows = [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
        if "ROW_NUMBER()" in query and not self.appended:
            # the next upload lands between the page starts and the pages: a later stamp, keys among the old ones
            self.appended = True
            self._db.executemany(
                "INSERT INTO s.snapshots VALUES (?, ?, ?)",
                [(state, race, "2026-11-04 01:05:00") for state in _STATES for race in range(200, 230)],
            )
        kept, truncated = self._cut(rows)
        return DatasourceQueryResult(
            rows=kept, row_count=len(kept), truncated=truncated, correlation_id=uuid7(), max_reply_bytes=self.max_bytes
        )

    def _cut(self, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
        if self.max_bytes is None:
            return rows, False
        kept: list[dict[str, Any]] = []
        size = 2
        for row in rows:
            grown = size + len(pydantic_core.to_json(row)) + (1 if kept else 0)
            if kept and grown > self.max_bytes:
                return kept, True
            kept.append(row)
            size = grown
        return kept, False


async def _read(hub: _AppendingHub, counts: dict[str, int], where: tuple[Fragment, ...]) -> dict[str, list[Any]]:
    read: dict[str, list[Any]] = {}
    async for state, rows in read_partitions(
        hub,  # type: ignore[arg-type]
        "w",
        columns=_COLUMNS,
        relation="s.snapshots",
        key=_KEY,
        partition_by="state",
        parts=_STATES,
        counts=counts,
        page_size=7,
        where=where,
    ):
        read[state] = rows
    return read


@pytest.mark.parametrize("max_bytes", [None, 300])
async def test_a_read_bounded_by_the_counted_stamp_holds_the_counted_rows_while_an_upload_lands(
    max_bytes: int | None,
) -> None:
    """Flips if the bound is not in every statement of the read (the page starts, the pages, a cut page's reads)."""
    hub = _AppendingHub(max_bytes)
    counts = hub.counts()
    read = await _read(hub, counts, (_BOUND,))
    assert hub.appended
    for state in _STATES:
        rows = read[state]
        assert len(rows) == counts[state] == 31
        assert [(r["state"], r["race"], r["uploaded"]) for r in rows] == [
            row for row in hub.held(state) if row[2] is None or row[2] <= _COUNTED_UP_TO
        ]


async def test_the_same_read_unbounded_is_refused_once_an_upload_lands() -> None:
    """Flips if this stand-in stops showing the race the bound closes: the whole part read after an upload."""
    hub = _AppendingHub()
    with pytest.raises(IncompleteReadError, match="counted 31 rows and the read returned 61"):
        await _read(hub, hub.counts(), ())


async def test_a_bound_needs_named_parts_proven_by_their_counts() -> None:
    """Flips if a bounded read is taken where the fingerprints, which name no condition, would prove it."""
    hub = _AppendingHub()
    with pytest.raises(ValueError, match="names its parts"):
        async for _ in read_partitions_arrow(
            hub,  # type: ignore[arg-type]
            "w",
            columns=_COLUMNS,
            relation="s.snapshots",
            key=_KEY,
            partition_by="state",
            types={},
            where=(_BOUND,),
        ):
            pass
