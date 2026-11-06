"""An L3 rail read as Arrow: the rows ``fetch`` answers, parsed from the reply's bytes off the interpreter lock.

``NatsProxyL3Backend.fetch_arrow`` hands a reply's bytes to Arrow's JSON reader on a thread, so a reply's rows are
never Python objects on the event loop (nor anywhere): what a pod hands on to a worker process or a DuckDB table
crosses as columns. The contract is ``fetch``'s rows: ``fetch_arrow(...).to_pylist() == fetch(...)`` for every column
type the rail answers as one Arrow type (:func:`~threetears.core.collections.complete_copy.rail_arrow_types`), and a
reply that cannot be read so is refused, never answered otherwise. ``read_l3_arrow`` pages a whole table as
``read_l3_rows`` does, with the same statements.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

pa = pytest.importorskip("pyarrow")

from pydantic import BaseModel  # noqa: E402

from threetears.core.backends import arrow_reply  # noqa: E402
from threetears.core.backends import nats_proxy as nats_proxy_module  # noqa: E402
from threetears.core.backends.arrow_reply import ReplyNotArrowError, reply_rows  # noqa: E402
from threetears.core.backends.nats_proxy import NatsProxyL3Backend  # noqa: E402
from threetears.core.backends.protocol import ArrowL3Reader  # noqa: E402
from threetears.core.collections.complete_copy import rail_arrow_types, read_l3_arrow, read_l3_rows  # noqa: E402
from threetears.core.collections.schema_backed import (  # noqa: E402
    BIGINT_TYPE,
    BOOL_TYPE,
    BYTES_TYPE,
    DATETIMETZ_TYPE,
    INT_TYPE,
    JSONB_TYPE,
    NUMERIC_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
)
from threetears.core.exceptions import DataLayerUnavailableError  # noqa: E402


class _Reply(BaseModel):
    """the broker's ``l3.query`` reply, encoded as the hub encodes it (``model_dump_json``)."""

    success: bool = True
    rows: list[dict[str, Any]] = []
    truncated: bool = False
    duration_ms: int = 1


def _broker(rows: list[dict[str, Any]]) -> bytes:
    """the bytes the broker answers ``rows`` with: a ``Decimal`` as its digits, a datetime as ISO text."""
    return _Reply(rows=rows).model_dump_json().encode()


_COLUMNS = [
    Column("id", BIGINT_TYPE),
    Column("name", STRING_TYPE, nullable=True),
    Column("ref", UUID_TYPE, nullable=True),
    Column("share", NUMERIC_TYPE, nullable=True, precision=12, scale=4),
    Column("counted", INT_TYPE, nullable=True),
    Column("called", BOOL_TYPE, nullable=True),
    Column("reported_at", DATETIMETZ_TYPE, nullable=True),
    Column("date_updated", DATETIMETZ_TYPE, nullable=True),
]

_ROWS = [
    {
        "id": 1,
        "name": "Travis",
        "ref": UUID("3b62f72b-81ea-4142-aeb2-33ddcae51b69"),
        "share": Decimal("0.5000"),
        "counted": 12,
        "called": True,
        "reported_at": datetime(2026, 11, 3, 20, 1, 2, 345678, tzinfo=UTC),
        "date_updated": datetime(2026, 11, 3, 21, tzinfo=UTC),
    },
    {
        "id": 2,
        "name": None,
        "ref": None,
        "share": None,
        "counted": None,
        "called": None,
        "reported_at": None,
        "date_updated": None,
    },
]


def _proxy(*replies: bytes) -> tuple[NatsProxyL3Backend, AsyncMock]:
    nc = MagicMock()
    nc.request_raw = AsyncMock(side_effect=list(replies))
    return (
        NatsProxyL3Backend(nats_client=nc, namespace_prefix="test", agent_id="a", identity_token=lambda: "t"),
        nc.request_raw,
    )


async def test_fetch_arrow_answers_the_rows_fetch_answers() -> None:
    reply = _broker(_ROWS)
    proxy, _ = _proxy(reply, reply)

    rows = await proxy.fetch("SELECT * FROM t")
    table = await proxy.fetch_arrow("SELECT * FROM t", types=rail_arrow_types(_COLUMNS))

    assert table.num_rows == 2
    assert table.to_pylist() == rows
    # the column types are the reply's own: text where fetch answers text, an instant for a date_ column
    assert table.schema.field("share").type == pa.string()
    assert table.schema.field("reported_at").type == pa.string()
    assert table.schema.field("date_updated").type == pa.timestamp("us", tz="UTC")


async def test_fetch_arrow_parses_the_reply_on_another_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """Arrow's reader lets go of the interpreter lock while it parses; on a thread, the loop runs beside it."""
    threads: list[int] = []
    parse = arrow_reply.reply_rows

    def recording(reply: bytes, types: Any) -> Any:
        threads.append(threading.get_ident())
        return parse(reply, types)

    monkeypatch.setattr(nats_proxy_module, "reply_rows", recording)
    proxy, _ = _proxy(_broker(_ROWS))
    await proxy.fetch_arrow("SELECT * FROM t", types=rail_arrow_types(_COLUMNS))
    assert threads and threads[0] != threading.get_ident()


async def test_a_refused_statement_raises_what_fetch_raises() -> None:
    refused = json.dumps({"success": False, "error_code": "QUERY_FAILED", "error_message": "no such table"}).encode()
    proxy, _ = _proxy(refused, refused)
    with pytest.raises(DataLayerUnavailableError) as by_fetch:
        await proxy.fetch("SELECT * FROM t")
    with pytest.raises(DataLayerUnavailableError) as by_arrow:
        await proxy.fetch_arrow("SELECT * FROM t", types=rail_arrow_types(_COLUMNS))
    assert str(by_arrow.value) == str(by_fetch.value)


async def test_fetch_arrow_reads_only() -> None:
    proxy, request = _proxy()
    with pytest.raises(ValueError, match="SELECT"):
        await proxy.fetch_arrow("DELETE FROM t", types={})
    request.assert_not_called()


def test_a_value_not_of_its_columns_type_is_refused() -> None:
    reply = _Reply(rows=[{"name": 7}]).model_dump_json().encode()
    with pytest.raises(ReplyNotArrowError):
        reply_rows(reply, {"name": pa.string()})


def test_text_carrying_the_bytes_mark_is_refused() -> None:
    """fetch answers ``\\x...`` text as bytes, which a text column cannot hold."""
    reply = _Reply(rows=[{"name": "\\x6162"}]).model_dump_json().encode()
    with pytest.raises(ReplyNotArrowError, match="bytes mark"):
        reply_rows(reply, {"name": pa.string()})


def test_an_empty_reply_is_an_empty_table_of_the_types_asked_for() -> None:
    parsed = reply_rows(_broker([]), rail_arrow_types(_COLUMNS))
    assert parsed.success and parsed.rows.num_rows == 0
    assert parsed.rows.column_names == [column.name for column in _COLUMNS]


@pytest.mark.parametrize(
    "column",
    [Column("doc", JSONB_TYPE), Column("blob", BYTES_TYPE), Column("date_label", STRING_TYPE)],
)
def test_a_column_fetch_answers_otherwise_than_its_json_has_no_rail_type(column: Column) -> None:
    with pytest.raises(ValueError, match="read it with fetch"):
        rail_arrow_types([column])


class _Rail:
    """an L3 rail over one table in memory, answering both reads alike: ``fetch`` and ``fetch_arrow`` decode
    the same broker reply, each statement paged by its keyset and ``LIMIT`` as the broker would."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = sorted(rows, key=lambda row: row["id"])
        self.statements: list[tuple[str, tuple[Any, ...]]] = []

    def _answer(self, sql: str, params: tuple[Any, ...]) -> bytes:
        self.statements.append((sql, params))
        limit = int(sql.rsplit("LIMIT ", 1)[1])
        after = params[0] if params else None
        kept = [row for row in self.rows if after is None or row["id"] > after][:limit]
        return _broker(kept)

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        proxy, _ = _proxy(self._answer(sql, params))
        return await proxy.fetch(sql)

    async def fetch_arrow(self, sql: str, *params: Any, types: Any) -> Any:
        proxy, _ = _proxy(self._answer(sql, params))
        return await proxy.fetch_arrow(sql, types=types)


async def test_read_l3_arrow_pages_a_table_as_read_l3_rows_does() -> None:
    made = [{**_ROWS[index % 2], "id": index} for index in range(2_345)]
    by_rows, by_arrow = _Rail(made), _Rail(made)

    rows = await read_l3_rows(by_rows, "t", [column.name for column in _COLUMNS], ("id",), page_size=1000)
    table = await read_l3_arrow(by_arrow, "t", _COLUMNS, ("id",), page_size=1000)

    assert table.num_rows == len(made)
    assert table.to_pylist() == rows
    assert by_arrow.statements == by_rows.statements and len(by_arrow.statements) == 3


class _Direct(_Rail):
    """a direct pool over the same rows: ``fetch`` answers asyncpg's values (a ``Decimal``, a datetime, a UUID), and
    it has no ``fetch_arrow``."""

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        self.statements.append((sql, params))
        limit = int(sql.rsplit("LIMIT ", 1)[1])
        after = params[0] if params else None
        return [dict(row) for row in self.rows if after is None or row["id"] > after][:limit]

    fetch_arrow = None  # type: ignore[assignment]


async def test_an_l3_without_arrow_is_read_as_the_rails_arrow() -> None:
    """a direct pool's rows are made the table the rail's reply of them reads as, so a reader holding either L3
    gets the same table."""
    made = [{**_ROWS[index % 2], "id": index} for index in range(1_500)]
    direct = _Direct(made)
    assert not isinstance(direct, ArrowL3Reader)
    from_direct = await read_l3_arrow(direct, "t", _COLUMNS, ("id",), page_size=1000)
    from_rail = await read_l3_arrow(_Rail(made), "t", _COLUMNS, ("id",), page_size=1000)
    assert from_direct.equals(from_rail)


def test_the_rails_proxy_is_an_arrow_reader() -> None:
    proxy, _ = _proxy()
    assert isinstance(proxy, ArrowL3Reader)


class _Statements:
    """an L3 that records each statement and answers no row."""

    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        self.statements.append((sql, params))
        return []


async def test_a_set_filter_binds_every_value_as_one_array_in_both_reads() -> None:
    by_rows, by_arrow = _Statements(), _Statements()
    where = {"name": "Travis"}
    where_any = {"counted": ("bigint[]", [1, 2, 3])}
    await read_l3_rows(by_rows, "t", [c.name for c in _COLUMNS], ("id",), where=where, where_any=where_any)
    await read_l3_arrow(by_arrow, "t", _COLUMNS, ("id",), where=where, where_any=where_any)
    assert by_rows.statements == by_arrow.statements
    ((sql, params),) = by_rows.statements
    assert '"name" = $1 AND "counted" = ANY($2::bigint[]) ORDER BY "id" LIMIT 1000' in sql
    assert params == ("Travis", [1, 2, 3])
