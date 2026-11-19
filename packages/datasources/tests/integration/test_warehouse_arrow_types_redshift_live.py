"""live proof of ``warehouse_arrow_types`` on a real Redshift: a late-binding view's columns (``WITH NO SCHEMA
BINDING``, which dbt's ``+bind: false`` makes of every view) are answered, in ordinal order, each as its JSON form,
where ``information_schema.columns`` holds none of them.

Creates one table and two views over it (late-binding and bound) under unique names in ``public``, the schema the
login may create in, and drops all three after, whatever the test's outcome. Gated on its own environment-named
connection, as the fetch_within proof is (``live_redshift_config``, in this directory's conftest); it skips (fails in
CI) when any part is missing.

run locally, from the repo root, with the password injected by reference:

.. code-block:: bash

    THREETEARS_LIVE_REDSHIFT_HOST=<host> THREETEARS_LIVE_REDSHIFT_DATABASE=<db> \\
    THREETEARS_LIVE_REDSHIFT_USER=<user> \\
    THREETEARS_LIVE_REDSHIFT_PASSWORD=op://<vault>/<item>/password \\
      op run -- ./scripts/test.sh datasources \\
      packages/datasources/tests/integration/test_warehouse_arrow_types_redshift_live.py -v -s
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any
from uuid import uuid4, uuid7

import pyarrow as pa
import pytest

from threetears.core.keyed import Dialect
from threetears.datasources.arrow_rows import warehouse_arrow_types
from threetears.datasources.config import RedshiftConnectionConfig
from threetears.datasources.drivers.redshift_driver import RedshiftDriver
from threetears.datasources.query_client import DatasourceQueryResult

pytestmark = [pytest.mark.integration, pytest.mark.live, pytest.mark.secrets]

#: each column's Arrow type as the rail's JSON reply carries it, in the table's order
_TYPES = {
    "race": pa.string(),
    "votes": pa.int64(),
    "total": pa.int64(),
    "margin": pa.string(),
    "share": pa.float64(),
    "called": pa.bool_(),
    "called_on": pa.string(),
    "called_at": pa.string(),
    "polls_close": pa.string(),
    "notes": pa.string(),
}


class _Client:
    """the hub's query ask, answered by the driver."""

    def __init__(self, driver: RedshiftDriver) -> None:
        self._driver = driver
        self.queries: list[str] = []

    async def query(self, datasource_name: str, sql: str, params: Sequence[Any] = ()) -> DatasourceQueryResult:
        self.queries.append(sql)
        rows = await self._driver.fetch(sql, *params)
        return DatasourceQueryResult(rows=rows, row_count=len(rows), truncated=False, correlation_id=uuid7())


@pytest.fixture
async def warehouse(live_redshift_config: RedshiftConnectionConfig) -> AsyncIterator[tuple[_Client, str]]:
    """a table, a late-binding view and a bound view over it, under one unique stem in ``public``; all dropped after.

    :param live_redshift_config: the environment-named cluster
    :ptype live_redshift_config: RedshiftConnectionConfig
    :return: the client and the stem
    :rtype: AsyncIterator[tuple[_Client, str]]
    """
    driver = RedshiftDriver(live_redshift_config, datasource_name="live-arrow-types")
    stem = f"tf2_lbv_{uuid4().hex[:10]}"
    try:
        await driver.execute(
            f"CREATE TABLE public.{stem} (race varchar(20), votes integer, total bigint, margin numeric(10,2), "
            "share double precision, called boolean, called_on date, called_at timestamptz, polls_close timetz, "
            "notes varchar(max))"
        )
        await driver.execute(f"CREATE VIEW public.{stem}_lbv AS SELECT * FROM public.{stem} WITH NO SCHEMA BINDING")
        await driver.execute(f"CREATE VIEW public.{stem}_bound AS SELECT * FROM public.{stem}")
        yield _Client(driver), stem
    finally:
        try:
            for statement in (
                f"DROP VIEW IF EXISTS public.{stem}_lbv",
                f"DROP VIEW IF EXISTS public.{stem}_bound",
                f"DROP TABLE IF EXISTS public.{stem}",
            ):
                await driver.execute(statement)
            left = await driver.fetch(
                "SELECT table_name FROM svv_tables WHERE table_schema = 'public' AND table_name LIKE $1", f"{stem}%"
            )
            assert left == [], f"left behind: {left}"
        finally:
            await driver.close()


@pytest.mark.parametrize("suffix", ["_lbv", "_bound", ""])
async def test_a_late_binding_view_s_columns_are_read_as_its_table_s(
    warehouse: tuple[_Client, str], suffix: str
) -> None:
    """Flips if a late-binding view's columns are read from a catalogue that holds none of them
    (``information_schema.columns``), or any relation's are answered in another order or form than its table's."""
    client, stem = warehouse

    every = await warehouse_arrow_types(
        client,  # type: ignore[arg-type]
        "w",
        f"public.{stem}{suffix}",
        None,
        dialect=Dialect.REDSHIFT,
    )
    named = await warehouse_arrow_types(
        client,  # type: ignore[arg-type]
        "w",
        f"public.{stem}{suffix}",
        ["polls_close", "race"],
        dialect=Dialect.REDSHIFT,
    )

    assert every == _TYPES
    assert list(every) == list(_TYPES)
    assert named == {"polls_close": pa.string(), "race": pa.string()}
    print(f"\nLIVE warehouse_arrow_types public.{stem}{suffix}: {len(every)} columns via {client.queries[0]!r}")
