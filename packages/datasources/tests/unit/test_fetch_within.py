"""A read cut by the bytes of its encoded rows (:meth:`Driver.fetch_within`): where it cuts, what it reads, and
what it leaves of the connection.

The budget's rule (:class:`RowBudget`) is checked on its own; the default every driver inherits keeps the rule on
what it kept; and the Redshift driver, against a double of a Redshift session that serves a declared cursor's
rows ``FETCH FORWARD`` at a time, reads no page past the cut and leaves its cached connection to the next read
after a cut and after a cancel while a page is read.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from threetears.datasources.config import RedshiftConnectionConfig, SnowflakeConnectionConfig
from threetears.datasources.bounded_rows import RowBudget, start_budget
from threetears.datasources.drivers.redshift_driver import RedshiftDriver
from threetears.datasources.drivers.snowflake_driver import SnowflakeDriver
from threetears.datasources.entities import DataSourceType
from threetears.datasources.query_client import DatasourceQueryResponse

_PASSWORD_ENV = "TEST_FETCH_WITHIN_PW"


def _encode(rows: list[dict[str, Any]]) -> bytes:
    """the callers' encoder in these tests: rows to a compact JSON array."""
    return json.dumps(rows, separators=(",", ":")).encode()


def _array_size(rows: list[dict[str, Any]]) -> int:
    return len(json.dumps(rows, separators=(",", ":")).encode())


# -- the budget ---------------------------------------------------------------------------------------------


def test_the_budget_keeps_the_rows_whose_array_fits_and_says_it_cut() -> None:
    rows = [{"n": index, "pad": "x" * 20} for index in range(100)]
    cap = _array_size(rows[:37])
    budget = start_budget(max_bytes=cap, encode=_encode, max_rows=None, page_rows=10)
    assert budget.take(rows[:50]) is False
    result = budget.result()
    assert result.rows == rows[:37]
    assert result.truncated is True
    # the reply is the encodings spliced, exactly the array's bytes, never a second encode
    assert result.json_array() == json.dumps(rows[:37], separators=(",", ":")).encode()
    assert result.size == len(result.json_array()) == cap


def test_the_budget_encodes_a_page_once_and_only_the_crossing_page_row_by_row() -> None:
    encoded: list[int] = []

    def counting(rows: list[dict[str, Any]]) -> bytes:
        encoded.append(len(rows))
        return _encode(rows)

    rows = [{"n": index} for index in range(50)]
    budget = RowBudget(_array_size(rows[:25]), counting, None)
    assert budget.take(rows[:10]) is True
    assert budget.take(rows[10:20]) is True
    assert budget.take(rows[20:30]) is False
    assert budget.result().rows == rows[:25]
    # two whole pages, the crossing page whole, then its rows one at a time up to the first that did not fit
    assert encoded == [10, 10, 10, 1, 1, 1, 1, 1, 1]
    assert budget.take(rows[30:40]) is False
    assert len(encoded) == 9, "nothing past the cut is encoded"


def test_after_the_first_page_a_page_is_sized_from_the_bytes_left() -> None:
    """the first page is the small one; the next fits whole with half a page to spare; the last reads one row past
    the expected cut, so only it is measured a row at a time."""
    rows = [{"n": index, "pad": "x" * 20} for index in range(10_000)]
    budget = RowBudget(_array_size(rows[:4_000]), _encode, None)
    assert budget.wanted(100) == 100
    assert budget.take(rows[:100]) is True
    second = budget.wanted(100)
    assert 3_800 <= second <= 3_900, second
    assert budget.take(rows[100 : 100 + second]) is True
    third = budget.wanted(100)
    assert third <= 101, third
    assert budget.take(rows[100 + second : 100 + second + third]) is False
    assert len(budget.result().rows) == 4_000
    capped = RowBudget(10_000_000, _encode, 150)
    capped.take(rows[:100])
    assert capped.wanted(100) == 51, "the row bound still caps a sized page"


def test_a_result_that_fits_is_whole_and_not_truncated() -> None:
    rows = [{"n": index} for index in range(5)]
    budget = RowBudget(10_000, _encode, None)
    assert budget.take(rows) is True
    assert (budget.result().rows, budget.result().truncated) == (rows, False)


def test_a_row_wider_than_the_cap_comes_back_alone_and_says_so() -> None:
    rows = [{"wide": "y" * 500}, {"n": 1}]
    budget = RowBudget(64, _encode, None)
    budget.take(rows)
    result = budget.result()
    assert result.rows == rows[:1]
    assert result.truncated is True
    assert result.size > 64


def test_the_row_bound_still_bounds_and_one_row_past_it_is_how_a_cut_is_told() -> None:
    rows = [{"n": index} for index in range(10)]
    exact = RowBudget(10_000, _encode, 10)
    assert exact.wanted(500) == 11
    exact.take(rows)
    assert (len(exact.result().rows), exact.result().truncated) == (10, False)
    longer = RowBudget(10_000, _encode, 4)
    longer.take(rows)
    assert (longer.result().rows, longer.result().truncated) == (rows[:4], True)


@pytest.mark.parametrize(
    ("max_bytes", "max_rows", "page_rows", "match"),
    [(1, None, 10, "max_bytes"), (100, 0, 10, "max_rows"), (100, None, 0, "page_rows")],
)
def test_bounds_out_of_range_are_refused(max_bytes: int, max_rows: int | None, page_rows: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        start_budget(max_bytes=max_bytes, encode=_encode, max_rows=max_rows, page_rows=page_rows)


# -- the default every driver inherits ------------------------------------------------------------------------


async def test_a_driver_that_cannot_page_still_keeps_no_more_than_fits() -> None:
    class _ReadsEverything(SnowflakeDriver):
        async def fetch(self, sql: str, *params: Any, timeout_seconds: int | None = None) -> list[dict[str, Any]]:
            return [{"n": index} for index in range(100)]

    reader = _ReadsEverything(
        SnowflakeConnectionConfig(
            datasource_type=DataSourceType.SNOWFLAKE, account="a", warehouse="w", user="u", password_ref="env://X"
        )
    )
    cap = _array_size([{"n": index} for index in range(30)])
    within = await reader.fetch_within("SELECT n FROM t", max_bytes=cap, encode=_encode)
    assert (len(within.rows), within.truncated) == (30, True)
    bounded = await reader.fetch_within("SELECT n FROM t", max_bytes=10_000, encode=_encode, max_rows=5)
    assert (len(bounded.rows), bounded.truncated) == (5, True)


# -- Redshift, against a double of its session ------------------------------------------------------------------


class _SessionDouble:
    """a Redshift session: one cursor serving a declared cursor's rows ``FETCH FORWARD`` at a time.

    :ivar statements: every statement executed, in order
    :ivar fetches: how many ``FETCH FORWARD`` pages were served
    :ivar served: how many rows those pages carried over the wire
    :ivar block_on: a statement prefix whose execution waits on ``release`` (to cancel a read inside it)
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self._offset = 0
        self._pending: list[tuple[Any, ...]] = []
        self.statements: list[str] = []
        self.fetches = 0
        self.served = 0
        self.closed = False
        self.rollbacks = 0
        self.block_on: str | None = None
        self.entered = threading.Event()
        self.release = threading.Event()
        self.description: list[tuple[str, Any]] = [(name, None) for name in rows[0]] if rows else []

    # the connection's surface
    def cursor(self) -> _SessionDouble:
        return self

    def rollback(self) -> None:
        self.rollbacks += 1
        self._offset = 0

    def commit(self) -> None:
        """nothing to commit in a read."""

    # the cursor's surface
    def execute(self, sql: str, params: Any = None) -> None:
        self.statements.append(sql)
        if self.block_on is not None and sql.startswith(self.block_on):
            self.entered.set()
            self.release.wait(5)
        self._pending = []
        if sql.startswith("DECLARE"):
            self._offset = 0
        elif sql.startswith("FETCH FORWARD"):
            self.fetches += 1
            wanted = int(sql.split()[2])
            page = self._rows[self._offset : self._offset + wanted]
            self._offset += len(page)
            self.served += len(page)
            self._pending = [tuple(row.values()) for row in page]

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._pending

    def fetchone(self) -> tuple[Any, ...]:
        return (4242,)

    def close(self) -> None:
        """a cursor's close is local; the session's close is :meth:`close_session`."""

    @property
    def arraysize(self) -> int:
        return 1000

    @arraysize.setter
    def arraysize(self, value: int) -> None:
        del value


class _ConnectionDouble:
    """the connection ``redshift_connector.connect`` returns: its session's surface, and a close of its own."""

    def __init__(self, session: _SessionDouble) -> None:
        self.session = session
        self.closed = False
        # the socket the driver lifts its login timeout from and sets keepalive on
        self._usock = MagicMock(name="socket")

    def cursor(self) -> _SessionDouble:
        return self.session

    def rollback(self) -> None:
        self.session.rollback()

    def commit(self) -> None:
        self.session.commit()

    def close(self) -> None:
        self.closed = True


def _redshift_config() -> RedshiftConnectionConfig:
    return RedshiftConnectionConfig(
        datasource_type=DataSourceType.REDSHIFT,
        host="rs.example.com",
        port=5439,
        database="analytics",
        username="rs_user",
        password_ref=f"env://{_PASSWORD_ENV}",
        executor_max_workers=2,
        connection_cache_size=2,
        query_timeout_seconds=60,
    )


@pytest.fixture(autouse=True)
def _password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_PASSWORD_ENV, "unit-test-password")


_ROWS = [{"n": index, "name": f"precinct-{index:05d}"} for index in range(10_000)]


@pytest.mark.parametrize("kept", [250, 4_000])
async def test_redshift_reads_no_page_past_the_cut(kept: int) -> None:
    """the wire bound: at most two FETCHes past the first, whatever the cap, and of the rows past the cut only the
    one that tells the cut crosses the wire."""
    session = _SessionDouble(_ROWS)
    connection = _ConnectionDouble(session)
    cap = _array_size(_ROWS[:kept])
    with patch("threetears.datasources.drivers.redshift_driver.redshift_connector.connect", return_value=connection):
        driver = RedshiftDriver(_redshift_config())
        within = await driver.fetch_within(
            "SELECT n, name FROM precincts;", max_bytes=cap, encode=_encode, page_rows=100, timeout_seconds=30
        )
        await driver.close()
    assert within.rows == _ROWS[:kept]
    assert within.truncated is True
    assert session.fetches <= 3, session.fetches
    assert session.served == kept + 1, "only the row that tells the cut crossed the wire past it"
    declared = [sql for sql in session.statements if sql.startswith("DECLARE")]
    assert declared == ["DECLARE threetears_fetch_within CURSOR FOR SELECT n, name FROM precincts"]
    # the statement deadline is set in the cursor's transaction, just before the statement runs
    declared_at = session.statements.index(declared[0])
    assert session.statements[declared_at - 1] == "SET LOCAL statement_timeout TO 30000"


async def test_redshift_reads_a_result_that_fits_whole_and_says_it_was_not_cut() -> None:
    session = _SessionDouble(_ROWS[:120])
    connection = _ConnectionDouble(session)
    with patch("threetears.datasources.drivers.redshift_driver.redshift_connector.connect", return_value=connection):
        driver = RedshiftDriver(_redshift_config())
        within = await driver.fetch_within("SELECT n, name FROM t", max_bytes=10_000_000, encode=_encode, page_rows=50)
        await driver.close()
    assert (within.rows, within.truncated) == (_ROWS[:120], False)
    # the first page, then a page sized from the bytes left, which holds the rest: a short page ends the read
    assert session.fetches == 2


async def test_redshift_honours_the_row_bound() -> None:
    session = _SessionDouble(_ROWS)
    connection = _ConnectionDouble(session)
    with patch("threetears.datasources.drivers.redshift_driver.redshift_connector.connect", return_value=connection):
        driver = RedshiftDriver(_redshift_config())
        within = await driver.fetch_within(
            "SELECT n, name FROM t", max_bytes=10_000_000, encode=_encode, max_rows=7, page_rows=100
        )
        await driver.close()
    assert (within.rows, within.truncated) == (_ROWS[:7], True)
    assert [sql for sql in session.statements if sql.startswith("FETCH")] == [
        "FETCH FORWARD 8 FROM threetears_fetch_within"
    ]


async def test_redshift_reuses_its_cached_connection_after_a_cut() -> None:
    session = _SessionDouble(_ROWS)
    connection = _ConnectionDouble(session)
    with patch(
        "threetears.datasources.drivers.redshift_driver.redshift_connector.connect", return_value=connection
    ) as connect:
        driver = RedshiftDriver(_redshift_config())
        first = await driver.fetch_within("SELECT n, name FROM t", max_bytes=2_000, encode=_encode, page_rows=10)
        second = await driver.fetch_within("SELECT n, name FROM t", max_bytes=2_000, encode=_encode, page_rows=10)
        assert connect.call_count == 1
        await driver.close()
    assert first.rows == second.rows
    assert first.truncated is True
    # the release path closed the cut read's transaction (and its cursor) before the connection was cached
    assert session.rollbacks >= 1


async def test_redshift_reuses_its_cached_connection_after_a_cancel_while_a_page_is_read() -> None:
    session = _SessionDouble(_ROWS)
    connection = _ConnectionDouble(session)
    session.block_on = "FETCH FORWARD"
    with patch(
        "threetears.datasources.drivers.redshift_driver.redshift_connector.connect", return_value=connection
    ) as connect:
        driver = RedshiftDriver(_redshift_config())
        read = asyncio.create_task(
            driver.fetch_within("SELECT n, name FROM t", max_bytes=2_000, encode=_encode, page_rows=10)
        )
        await asyncio.to_thread(session.entered.wait, 5)
        read.cancel()
        session.release.set()
        with pytest.raises(asyncio.CancelledError):
            await read
        assert connection.closed is False
        session.block_on = None
        again = await driver.fetch_within("SELECT n, name FROM t", max_bytes=2_000, encode=_encode, page_rows=10)
        assert connect.call_count == 1
        await driver.close()
    assert again.truncated is True


async def test_redshift_aborts_a_statement_cancelled_while_it_runs() -> None:
    """a cancel inside the DECLARE (the statement itself) is the read path's usual cancel: the session closes."""
    session = _SessionDouble(_ROWS)
    connection = _ConnectionDouble(session)
    session.block_on = "DECLARE"
    with patch("threetears.datasources.drivers.redshift_driver.redshift_connector.connect", return_value=connection):
        driver = RedshiftDriver(_redshift_config())
        read = asyncio.create_task(driver.fetch_within("SELECT n, name FROM t", max_bytes=2_000, encode=_encode))
        await asyncio.to_thread(session.entered.wait, 5)
        read.cancel()
        with pytest.raises(asyncio.CancelledError):
            await read
        session.release.set()
        assert connection.closed is True
        await driver.close()


# -- the reply carrying them ----------------------------------------------------------------------------------


def test_a_reply_with_spliced_rows_reads_as_the_model_would_have_written_it() -> None:
    rows = [{"n": 1, "name": "a"}, {"n": 2, "name": "b"}]
    budget = RowBudget(10_000, _encode, None)
    budget.take(rows)
    within = budget.result()
    reply = DatasourceQueryResponse(
        success=True, row_count=2, truncated=False, max_reply_bytes=1_000_000, correlation_id=uuid4()
    )
    spliced = reply.json_with_rows(within.json_array())
    assert DatasourceQueryResponse.model_validate_json(spliced) == reply.model_copy(update={"rows": rows})


def test_a_reply_cut_at_rows_announces_no_byte_cap() -> None:
    """expand-only: an old caller, and a hub cutting by rows, see the reply they always saw."""
    assert DatasourceQueryResponse(success=True).max_reply_bytes is None
    assert DatasourceQueryResponse.model_validate_json(b'{"success":true,"rows":[]}').max_reply_bytes is None
