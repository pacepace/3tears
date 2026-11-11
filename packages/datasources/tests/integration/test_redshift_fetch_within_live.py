"""live proof of :meth:`RedshiftDriver.fetch_within` on a real cluster: a cursor declared with bind parameters, read
``FETCH FORWARD`` a page at a time up to the cut, and the connection reused after the cut.

Read-only and touching no table: a ``generate_series`` the leader answers, inside the checkout's own transaction,
so it reads no warehouse data and creates or writes nothing.

Gated on its OWN connection, never the central-reporting production user the other live tests log in as: every
part of it is named by the environment, and it skips (fails in CI) when any part is missing. The password is a
reference (``env://THREETEARS_LIVE_REDSHIFT_PASSWORD``), resolved only by the driver, never printed. The gate logs
in once per session (:func:`~tests.unit.helpers.redshift_live_gate.gated`), so a wrong password costs one refused
login; its fixture, ``live_redshift_config``, lives beside ``redshift_config`` in this directory's conftest, the one
place a live test's connection is built.

run locally, from the repo root, with the password injected by reference:

.. code-block:: bash

    THREETEARS_LIVE_REDSHIFT_HOST=<host> THREETEARS_LIVE_REDSHIFT_DATABASE=<db> \\
    THREETEARS_LIVE_REDSHIFT_USER=<user> \\
    THREETEARS_LIVE_REDSHIFT_PASSWORD=op://<vault>/<item>/password \\
      op run -- uv run pytest packages/datasources/tests/integration/test_redshift_fetch_within_live.py -v -s
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from typing import Any

import pytest

from threetears.datasources.config import RedshiftConnectionConfig
from threetears.datasources.drivers import redshift_driver as redshift_driver_module
from threetears.datasources.drivers.redshift_driver import RedshiftDriver

pytestmark = [pytest.mark.integration, pytest.mark.live]

#: the rows a page reads in these tests
_PAGE_ROWS = 100

#: a read with two bind parameters that touches no table: the series' first and last values
_SQL = "SELECT generate_series($1::int, $2::int) AS n"


def _encode(rows: list[dict[str, Any]]) -> bytes:
    """
    :param rows: rows
    :ptype rows: list[dict[str, Any]]
    :return: their JSON array
    :rtype: bytes
    """
    return json.dumps(rows, default=str, separators=(",", ":")).encode()


@pytest.fixture
def pages(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    """count the ``FETCH FORWARD`` pages a read sends, and the rows each asked for.

    :param monkeypatch: pytest's patcher
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: the rows each page asked for, in order
    :rtype: Iterator[list[int]]
    """
    sent: list[int] = []
    real = redshift_driver_module._page_within_sync  # noqa: SLF001

    def _counted(cursor: Any, budget: Any, wanted: int) -> bool:
        sent.append(wanted)
        return real(cursor, budget, wanted)

    monkeypatch.setattr(redshift_driver_module, "_page_within_sync", _counted)
    yield sent


async def _pid(driver: RedshiftDriver) -> int:
    """
    :param driver: the driver
    :ptype driver: RedshiftDriver
    :return: the backend its cached connection holds
    :rtype: int
    """
    rows = await driver.fetch("SELECT pg_backend_pid() AS pid")
    return int(rows[0]["pid"])


async def test_a_cursor_with_bind_parameters_reads_pages_to_the_cut_and_keeps_its_connection(
    live_redshift_config: RedshiftConnectionConfig, pages: list[int]
) -> None:
    driver = RedshiftDriver(live_redshift_config, datasource_name="live-fetch-within")
    try:
        before = await _pid(driver)
        whole = await driver.fetch(_SQL, 1, 2000)
        assert [row["n"] for row in whole] == list(range(1, 2001))
        cap = len(_encode(whole[: int(2.5 * _PAGE_ROWS)]))

        got = await driver.fetch_within(_SQL, 1, 2000, max_bytes=cap, encode=_encode, page_rows=_PAGE_ROWS)

        kept = len(got.rows)
        assert got.truncated
        assert got.rows == whole[:kept], "the bind parameters reached the declared cursor: the same rows, in order"
        assert got.size <= cap and len(got.json_array()) == got.size
        assert len(pages) <= math.ceil(kept / _PAGE_ROWS) + 1, (kept, pages)
        after = await _pid(driver)
        assert after == before, "the connection read past the cut went back to the cache, not evicted"
        print(
            f"\nLIVE fetch_within: whole={len(whole)} cap={cap}B kept={kept} pages={pages} pid reused={after == before}"
        )
    finally:
        await driver.close()


async def test_a_row_bound_read_stops_at_its_rows_with_bind_parameters(
    live_redshift_config: RedshiftConnectionConfig, pages: list[int]
) -> None:
    driver = RedshiftDriver(live_redshift_config, datasource_name="live-fetch-within")
    try:
        before = await _pid(driver)
        got = await driver.fetch_within(
            _SQL, 7, 2000, max_bytes=10_000_000, encode=_encode, max_rows=150, page_rows=_PAGE_ROWS
        )

        assert len(got.rows) == 150 and got.truncated
        assert [row["n"] for row in got.rows] == list(range(7, 157)), "the bind parameters reached the cursor"
        assert len(pages) <= math.ceil(150 / _PAGE_ROWS) + 1, pages
        assert await _pid(driver) == before
        print(f"\nLIVE fetch_within row bound: kept=150 pages={pages}")
    finally:
        await driver.close()
