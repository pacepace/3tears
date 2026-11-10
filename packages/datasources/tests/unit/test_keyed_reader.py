"""DatasourceReader: the keyed helpers read a warehouse datasource through the query client, a cut answer refused."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.core.keyed import AnyOf, counts_by, read_past_marks
from threetears.datasources.keyed_reader import DatasourceAnswerCut, DatasourceReader
from threetears.datasources.query_client import DatasourceQueryResult


class _Client:
    """a query client answering canned rows, keeping each query it was asked."""

    def __init__(self, rows: list[dict[str, Any]], *, truncated: bool = False) -> None:
        self._rows = rows
        self._truncated = truncated
        self.asked: list[tuple[str, str, list[Any]]] = []
        self.tokens: list[str | None] = []

    async def query(
        self, datasource: str, query: str, params: list[Any] | None = None, *, user_identity_token: str | None = None
    ) -> DatasourceQueryResult:
        self.asked.append((datasource, query, list(params or [])))
        self.tokens.append(user_identity_token)
        return DatasourceQueryResult(
            rows=self._rows, row_count=len(self._rows), truncated=self._truncated, correlation_id=uuid4()
        )


async def test_a_keyed_count_runs_on_the_warehouse_with_its_values_listed() -> None:
    """Flips if the reader drops the datasource or the parameters, or a warehouse count binds an array."""
    client = _Client([{"relation": 0, "counted": 7, "held": 3}])
    reader = DatasourceReader(client, "redshift")  # type: ignore[arg-type]
    counted = await counts_by(reader, "enr.lsf_runs", column="run_id", values=AnyOf([7, 8], as_list=True))
    assert counted == {7: 3}
    ((datasource, query, params),) = client.asked
    assert datasource == "redshift" and params == [7, 8]
    assert '"enr"."lsf_runs"' in query and '"run_id" IN ($1, $2)' in query


async def test_a_cut_answer_is_refused() -> None:
    """Flips if an answer the hub cut at its row cap is taken as every row."""
    reader = DatasourceReader(_Client([{"x": 1}], truncated=True), "redshift")  # type: ignore[arg-type]
    with pytest.raises(DatasourceAnswerCut):
        await reader.fetch("SELECT 1")


async def test_a_read_past_marks_runs_on_the_warehouse_its_stamps_decoded() -> None:
    """Flips if a mark crosses the rail other than in its JSON form, a stamp comes back undecoded (so the next
    mark is text, or the comparison fails), the user's assertion is dropped, or the marks do not advance."""
    mark = datetime(2026, 11, 3, 20, tzinfo=UTC)
    client = _Client(
        [
            {"race_id": "r1", "updated_at": "2026-11-03T21:00:00+00:00"},
            {"race_id": "r2", "updated_at": None},
        ]
    )
    reader = DatasourceReader(
        client,  # type: ignore[arg-type]
        "redshift",
        user_identity_token="tok",
        decode={"updated_at": datetime.fromisoformat},
    )
    moved = await read_past_marks(reader, "enr.results", columns=["race_id", "updated_at"], marks={"updated_at": mark})
    assert moved.marks == {"updated_at": datetime(2026, 11, 3, 21, tzinfo=UTC)}
    assert [row["race_id"] for row in moved.rows] == ["r1", "r2"]
    ((_, query, params),) = client.asked
    assert params == ["2026-11-03T20:00:00+00:00"] and '"updated_at" > $1' in query
    assert client.tokens == ["tok"]


async def test_a_uuid_key_crosses_the_rail_as_text() -> None:
    """Flips if a UUID parameter is sent as an object the rail's JSON cannot carry."""
    client = _Client([])
    key = UUID(int=7)
    await DatasourceReader(client, "redshift").fetch("SELECT 1 WHERE k = $1", key)  # type: ignore[arg-type]
    assert client.asked[0][2] == [str(key)]
