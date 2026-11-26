"""A datasource read's rows built straight into its reply's Arrow table (:mod:`threetears.datasources.reply_arrow`).

The table must be exactly the one the pod's parse of the JSON reply of the same rows made (the hub's JSON reply:
``json_safe_row``, pydantic's JSON, Arrow's JSON reader), types included, and refuse what that parse refused; it must
be built with no Python call a row; and a byte-capped read keeps the longest prefix whose buffers fit, as the
Redshift driver's pages read it.
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import Callable
from datetime import UTC, date, datetime, time, timezone, timedelta
from decimal import Decimal
from types import FrameType
from typing import Any
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pydantic_core
import pytest

from threetears.core.backends.arrow_reply import ReplyNotArrowError, reply_rows
from threetears.datasources.config import RedshiftConnectionConfig, SnowflakeConnectionConfig
from threetears.datasources.drivers.redshift_driver import RedshiftDriver
from threetears.datasources.drivers.snowflake_driver import SnowflakeDriver
from threetears.datasources.entities import DataSourceType
from threetears.datasources.reply_arrow import ArrowBudget, records_table, rows_table, start_arrow_budget

_PASSWORD_ENV = "TEST_REPLY_ARROW_PW"


def _json_safe(row: dict[str, Any]) -> dict[str, Any]:
    """the hub's ``json_safe_row``: bytes as bare hex, everything else as it is."""
    return {key: (value.hex() if isinstance(value, bytes | bytearray) else value) for key, value in row.items()}


def _as_json_reply(names: list[str], rows: list[tuple[Any, ...]], schema: pa.Schema) -> pa.Table:
    """the table the pod made of the JSON reply of these rows, before: dicts, json_safe_row, pydantic, Arrow's reader."""
    dicts = [_json_safe(dict(zip(names, row, strict=True))) for row in rows]
    reply = b"".join((b'{"success":true,"rows":', pydantic_core.to_json(dicts), b"}"))
    parsed: pa.Table = reply_rows(reply, {field.name: field.type for field in schema}, bytes_mark=False).rows
    return parsed


_NAMES = ["t", "i", "f", "b", "d", "day", "naive", "aware", "clock", "u", "raw", "none", "mixed_none"]

_SCHEMA = pa.schema(
    [
        ("t", pa.string()),
        ("i", pa.int64()),
        ("f", pa.float64()),
        ("b", pa.bool_()),
        ("d", pa.string()),
        ("day", pa.string()),
        ("naive", pa.string()),
        ("aware", pa.string()),
        ("clock", pa.string()),
        ("u", pa.string()),
        ("raw", pa.string()),
        ("none", pa.int64()),
        ("mixed_none", pa.float64()),
    ]
)


def _row(index: int) -> tuple[Any, ...]:
    return (
        f'precinct "{index}" é\n' if index % 7 else None,
        index * 1_000_003 - 50 if index % 5 else None,
        index / 3 if index % 11 else (float("nan") if index % 2 else None),
        index % 2 == 0,
        Decimal(f"{index}.{index % 100:02d}") if index % 3 else Decimal("1E+2"),
        date(2026, 11, 1) + timedelta(days=index % 30),
        datetime(2026, 11, 3, 20, 0, index % 60, (index * 7) % 1_000_000),
        datetime(2026, 11, 4, 4, 41, index % 60, tzinfo=UTC)
        if index % 4
        else datetime(2026, 11, 4, 4, 41, 7, tzinfo=timezone(timedelta(hours=-5))),
        time(index % 24, index % 60, index % 60),
        uuid.UUID(int=index),
        bytes([index % 256, 255, 0]) if index % 6 else None,
        None,
        float(index) if index % 2 else None,
    )


_ROWS = [_row(index) for index in range(1, 400)]


def _same(built: pa.Table, before: pa.Table) -> list[str]:
    """the columns two tables differ in, types included (NaN equal to NaN, which ``equals`` does not hold)."""
    differ = [] if built.schema.equals(before.schema) else ["<schema>"]
    for name in built.column_names:
        left, right = built.column(name).to_pylist(), before.column(name).to_pylist()
        if [("nan" if value != value else value) for value in left] != [  # noqa: PLR0124 - NaN is the one value unequal to itself
            ("nan" if value != value else value)
            for value in right  # noqa: PLR0124
        ]:
            differ.append(name)
    return differ


def test_the_table_is_the_one_the_json_reply_read_as_types_included() -> None:
    built = rows_table(_NAMES, _ROWS, _SCHEMA)
    before = _as_json_reply(_NAMES, _ROWS, _SCHEMA)
    assert built.schema.equals(before.schema)
    assert _same(built, before) == []
    assert built.column("f").null_count == before.column("f").null_count


def test_an_integer_column_read_as_a_float_is_the_floats_the_json_reply_read() -> None:
    schema = pa.schema([("n", pa.float64())])
    rows = [(1,), (None,), (2**53 + 1,)]
    assert rows_table(["n"], rows, schema).equals(_as_json_reply(["n"], rows, schema))


def test_an_empty_result_is_the_empty_table_of_the_types() -> None:
    built = rows_table(_NAMES, [], _SCHEMA)
    assert built.num_rows == 0
    assert built.schema.equals(_SCHEMA)


def test_a_column_not_selected_is_null_and_one_not_asked_for_is_left_out_as_the_json_reply_read_them() -> None:
    names = ["extra", "n", "n"]
    rows = [("x", 1, 2), ("y", 3, 4)]
    schema = pa.schema([("n", pa.int64()), ("absent", pa.string())])
    built = rows_table(names, rows, schema)
    # a name selected twice: the last, as a row's dict kept it
    assert built.column("n").to_pylist() == [2, 4]
    assert built.column("absent").null_count == 2
    assert built.equals(_as_json_reply(["extra", "n"], [("x", 2), ("y", 4)], schema))


@pytest.mark.parametrize(
    ("values", "kind"),
    [
        ([1, 2], pa.string()),
        ([1.5, 2.0], pa.int64()),
        ([1.0], pa.int64()),
        ([True, False], pa.int64()),
        ([Decimal("1.5")], pa.float64()),
        ([b"\x00\xff"], pa.int64()),
        (["1"], pa.bool_()),
        ([1], pa.bool_()),
        ([2**70], pa.int64()),
    ],
)
def test_values_the_json_reply_refused_are_refused_naming_the_column_never_a_value(
    values: list[Any], kind: Any
) -> None:
    schema = pa.schema([("c", kind)])
    rows = [(value,) for value in values]
    with pytest.raises(ReplyNotArrowError):
        _as_json_reply(["c"], rows, schema)
    with pytest.raises(ReplyNotArrowError, match="column 'c'") as refused:
        rows_table(["c"], rows, schema)
    for value in values:
        assert f"{value}" not in f"{refused.value}".replace("'c'", "").replace(f"{kind}", "")


def test_records_with_keys_build_the_same_table_as_rows() -> None:
    mappings = [dict(zip(_NAMES, row, strict=True)) for row in _ROWS]
    assert _same(records_table(mappings, _SCHEMA), rows_table(_NAMES, _ROWS, _SCHEMA)) == []


def _python_calls(build: Callable[[], Any]) -> int:
    """how many Python functions ``build`` calls (C functions are not counted): what a row at a time would show."""
    calls = 0

    def profile(frame: FrameType, event: str, arg: Any) -> None:
        nonlocal calls
        del frame, arg
        if event == "call":
            calls += 1

    sys.setprofile(profile)
    try:
        build()
    finally:
        sys.setprofile(None)
    return calls


def test_a_hundred_thousand_rows_are_built_with_no_python_call_a_row() -> None:
    """the build is a column at a time, in C: its Python calls do not grow with the rows (the JSON reply's
    ``json_safe_row`` made one a row)."""
    small = [_row(index) for index in range(1, 1_001)]
    large = [_row(index) for index in range(1, 100_001)]
    few = _python_calls(lambda: rows_table(_NAMES, small, _SCHEMA))
    many = _python_calls(lambda: rows_table(_NAMES, large, _SCHEMA))
    # the same few hundred calls at a thousand rows and a hundred thousand (a call or two of the interpreter's own
    # either way), never one a row
    assert abs(many - few) <= 10, (few, many)
    assert many < 1_000, many
    # and the JSON reply's way makes one a row, which is what this replaces
    assert _python_calls(lambda: [_json_safe(dict(zip(_NAMES, row, strict=True))) for row in large]) > 100_000


# -- the byte cap ---------------------------------------------------------------------------------------------


_NARROW = pa.schema([("n", pa.int64()), ("name", pa.string())])
_PAGE_ROWS = [(index, f"precinct-{index:05d}") for index in range(10_000)]


def _bytes_of(count: int) -> int:
    return int(rows_table(["n", "name"], _PAGE_ROWS[:count], _NARROW).nbytes)


def test_the_budget_keeps_the_longest_prefix_that_fits_and_says_it_cut() -> None:
    cap = _bytes_of(3_737)
    budget = start_arrow_budget(schema=_NARROW, max_bytes=cap, max_rows=None, page_rows=100)
    assert budget.take(rows_table(["n", "name"], _PAGE_ROWS[:1_000], _NARROW)) is True
    assert budget.take(rows_table(["n", "name"], _PAGE_ROWS[1_000:5_000], _NARROW)) is False
    result = budget.result()
    assert result.table.num_rows == 3_737
    assert result.truncated is True
    assert result.size <= cap
    assert result.table.column("n").to_pylist() == list(range(3_737))
    # nothing past the cut is kept
    assert budget.take(rows_table(["n", "name"], _PAGE_ROWS[5_000:6_000], _NARROW)) is False
    assert budget.result().table.num_rows == 3_737


def test_the_budget_sizes_its_pages_from_the_bytes_left() -> None:
    budget = ArrowBudget(_NARROW, _bytes_of(4_000), None)
    assert budget.wanted(100) == 100
    budget.take(rows_table(["n", "name"], _PAGE_ROWS[:100], _NARROW))
    second = budget.wanted(100)
    assert 3_600 <= second <= 3_900, second
    capped = ArrowBudget(_NARROW, 10_000_000, 150)
    capped.take(rows_table(["n", "name"], _PAGE_ROWS[:100], _NARROW))
    assert capped.wanted(100) == 51


def test_a_row_wider_than_the_cap_comes_back_alone_and_the_row_bound_still_bounds() -> None:
    wide = pa.schema([("w", pa.string())])
    budget = start_arrow_budget(schema=wide, max_bytes=64, max_rows=None, page_rows=10)
    budget.take(rows_table(["w"], [("y" * 500,), ("z",)], wide))
    assert (budget.result().table.num_rows, budget.result().truncated) == (1, True)
    bounded = start_arrow_budget(schema=_NARROW, max_bytes=10_000_000, max_rows=4, page_rows=10)
    bounded.take(rows_table(["n", "name"], _PAGE_ROWS[:10], _NARROW))
    assert (bounded.result().table.num_rows, bounded.result().truncated) == (4, True)


@pytest.mark.parametrize(
    ("max_bytes", "max_rows", "page_rows", "match"),
    [(0, None, 10, "max_bytes"), (100, 0, 10, "max_rows"), (100, None, 0, "page_rows")],
)
def test_bounds_out_of_range_are_refused(max_bytes: int, max_rows: int | None, page_rows: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        start_arrow_budget(schema=_NARROW, max_bytes=max_bytes, max_rows=max_rows, page_rows=page_rows)


# -- the drivers ----------------------------------------------------------------------------------------------


async def test_a_driver_that_cannot_page_builds_its_rows_and_keeps_what_fits() -> None:
    class _ReadsEverything(SnowflakeDriver):
        async def fetch(self, sql: str, *params: Any, timeout_seconds: int | None = None) -> list[dict[str, Any]]:
            return [{"n": n, "name": name} for n, name in _PAGE_ROWS[:100]]

    reader = _ReadsEverything(
        SnowflakeConnectionConfig(
            datasource_type=DataSourceType.SNOWFLAKE, account="a", warehouse="w", user="u", password_ref="env://X"
        )
    )
    within = await reader.fetch_arrow_within("SELECT n, name FROM t", schema=_NARROW, max_bytes=_bytes_of(30))
    assert (within.table.num_rows, within.truncated) == (30, True)
    at_most = await reader.fetch_arrow_at_most("SELECT n, name FROM t", schema=_NARROW, max_rows=5)
    assert at_most.equals(rows_table(["n", "name"], _PAGE_ROWS[:5], _NARROW))


class _Session:
    """a Redshift session: a plain statement's rows (``fetchmany``), or a declared cursor's ``FETCH FORWARD`` pages."""

    def __init__(self, rows: list[tuple[Any, ...]], names: list[str]) -> None:
        self._rows = rows
        self._offset = 0
        self._pending: list[tuple[Any, ...]] = []
        self.statements: list[str] = []
        self.served = 0
        self.description = [(name, None) for name in names]

    def cursor(self) -> _Session:
        return self

    def rollback(self) -> None:
        self._offset = 0

    def commit(self) -> None:
        """nothing to commit in a read."""

    def execute(self, sql: str, params: Any = None) -> None:
        del params
        self.statements.append(sql)
        self._pending = []
        if sql.startswith("DECLARE"):
            self._offset = 0
        elif sql.startswith("FETCH FORWARD"):
            wanted = int(sql.split()[2])
            self._pending = self._rows[self._offset : self._offset + wanted]
            self._offset += len(self._pending)
            self.served += len(self._pending)
        elif sql.startswith("SELECT"):
            self._pending = list(self._rows)

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._pending

    def fetchone(self) -> tuple[Any, ...]:
        return (4242,)

    def fetchmany(self, count: int) -> list[tuple[Any, ...]]:
        self.served += min(count, len(self._pending))
        return self._pending[:count]

    def close(self) -> None:
        """a cursor's close is local."""


class _Connection:
    def __init__(self, session: _Session) -> None:
        self.session = session
        self.closed = False
        self._usock = MagicMock(name="socket")

    def cursor(self) -> _Session:
        return self.session

    def rollback(self) -> None:
        self.session.rollback()

    def commit(self) -> None:
        """nothing to commit."""

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


async def test_redshift_pages_a_byte_capped_read_into_arrow_and_reads_no_page_past_the_cut() -> None:
    session = _Session(_PAGE_ROWS, ["n", "name"])
    with patch(
        "threetears.datasources.drivers.redshift_driver.redshift_connector.connect", return_value=_Connection(session)
    ):
        driver = RedshiftDriver(_redshift_config())
        within = await driver.fetch_arrow_within(
            "SELECT n, name FROM precincts;",
            schema=_NARROW,
            max_bytes=_bytes_of(4_000),
            page_rows=100,
            timeout_seconds=30,
        )
        await driver.close()
    assert within.table.num_rows == 4_000
    assert within.truncated is True
    assert within.table.equals(rows_table(["n", "name"], _PAGE_ROWS[:4_000], _NARROW))
    # the first page, one sized to fit, then at most two that close on the cut
    assert len([sql for sql in session.statements if sql.startswith("FETCH")]) <= 4
    assert session.served <= 4_000 + 102, "the crossing page is about a page past the cut, never the result"
    assert "DECLARE threetears_fetch_within CURSOR FOR SELECT n, name FROM precincts" in session.statements


async def test_redshift_reads_at_most_its_bound_into_arrow() -> None:
    session = _Session(_PAGE_ROWS[:50], ["n", "name"])
    with patch(
        "threetears.datasources.drivers.redshift_driver.redshift_connector.connect", return_value=_Connection(session)
    ):
        driver = RedshiftDriver(_redshift_config())
        table = await driver.fetch_arrow_at_most("SELECT n, name FROM t", schema=_NARROW, max_rows=11)
        await driver.close()
    assert table.equals(rows_table(["n", "name"], _PAGE_ROWS[:11], _NARROW))
