"""DatasourceReader: the keyed helpers read a warehouse datasource through the query client, a cut answer refused."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from threetears.core.keyed import AnyOf, counts_by
from threetears.datasources.keyed_reader import DatasourceAnswerCut, DatasourceReader
from threetears.datasources.query_client import DatasourceQueryResult


class _Client:
    """a query client answering canned rows, keeping each query it was asked."""

    def __init__(self, rows: list[dict[str, Any]], *, truncated: bool = False) -> None:
        self._rows = rows
        self._truncated = truncated
        self.asked: list[tuple[str, str, list[Any]]] = []

    async def query(self, datasource: str, query: str, params: list[Any] | None = None) -> DatasourceQueryResult:
        self.asked.append((datasource, query, list(params or [])))
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
