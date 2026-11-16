"""DatasourceQueryClient.query_arrow: a datasource read's rows as one Arrow table, whichever form the hub answers in.

A hub that answers Arrow sends the rows as one IPC stream behind the reply's envelope (``L3ARROW1``, the ``l3.query``
rail's framing, read by the one parser of both rails); an older hub ignores the ask and answers JSON, which Arrow's own
reader parses into the same table. Only ``query_arrow`` carries the two fields: every other ask is byte for byte what
it was.
"""

from __future__ import annotations

import base64
import json
import struct
from datetime import timedelta
from typing import Any
from uuid import uuid7

import pyarrow as pa
import pytest

from threetears.core.backends.arrow_reply import ReplyNotArrowError, arrow_schema_text
from threetears.datasources.arrow_rows import warehouse_arrow_types
from threetears.datasources.query_client import (
    DATASOURCE_BUSY,
    DatasourceQueryClient,
    DatasourceQueryError,
    DatasourceQueryRequest,
    DatasourceQueryResponse,
    DatasourceQueryResult,
)
from threetears.nats.subjects import Subject, set_default_namespace

_TOKEN = "header.payload.signature"
_TYPES = {"race": pa.string(), "votes": pa.int64(), "share": pa.float64(), "called": pa.bool_(), "margin": pa.string()}
_ROWS: list[dict[str, Any]] = [
    {"race": "VA-01", "votes": 120, "share": 0.5, "called": True, "margin": "12.50"},
    {"race": "VA-02", "votes": 7, "share": 0.25, "called": False, "margin": None},
]


@pytest.fixture(autouse=True)
def _namespace() -> None:
    set_default_namespace("t")


def _json_reply(rows: list[dict[str, Any]], **fields: Any) -> bytes:
    """the reply an older hub sends: the JSON envelope with its rows."""
    reply = DatasourceQueryResponse(success=True, rows=rows, row_count=len(rows), correlation_id=uuid7(), **fields)
    return reply.model_dump_json().encode()


def _arrow_reply(rows: list[dict[str, Any]], *, rows_bytes: int, **fields: Any) -> bytes:
    """the reply a hub that answers Arrow sends: the magic, the envelope's length, the envelope, the rows' IPC."""
    reply = DatasourceQueryResponse(success=True, row_count=len(rows), correlation_id=uuid7(), **fields)
    envelope = json.loads(reply.model_dump_json(exclude={"rows"}))
    envelope["rows_bytes"] = rows_bytes
    head = json.dumps(envelope).encode()
    table = pa.Table.from_pylist(rows, schema=pa.schema(list(_TYPES.items())))
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return b"".join((b"L3ARROW1", struct.pack("!I", len(head)), head, sink.getvalue().to_pybytes()))


class _RawNats:
    """answers each raw request with the next canned reply, recording what was sent."""

    def __init__(self, *replies: bytes) -> None:
        self._replies = list(replies)
        self.sent: list[dict[str, Any]] = []
        self.typed: list[DatasourceQueryRequest] = []

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta) -> bytes:
        self.sent.append(json.loads(payload))
        return self._replies.pop(0)

    async def request(self, *, subject: Subject, message: Any, response_type: Any, timeout: timedelta) -> Any:
        self.typed.append(message)
        return response_type.model_validate_json(self._replies.pop(0))


def _client(nats: _RawNats) -> DatasourceQueryClient:
    return DatasourceQueryClient(nats, identity_token=lambda: _TOKEN)  # type: ignore[arg-type]


async def test_an_arrow_reply_is_read_as_its_table_with_its_cut_and_cap() -> None:
    """Flips if an Arrow reply's table, its cut, its cap or the bytes the cap measured its rows by are lost."""
    nats = _RawNats(_arrow_reply(_ROWS, rows_bytes=300, truncated=True, max_reply_bytes=4096))

    answer = await _client(nats).query_arrow("ds", "SELECT race, votes, share, called, margin FROM t", types=_TYPES)

    assert answer.rows.to_pylist() == _ROWS
    assert answer.rows.schema == pa.schema(list(_TYPES.items()))
    assert (answer.truncated, answer.max_reply_bytes) == (True, 4096)
    assert answer.rows_bytes > 300


async def test_a_json_reply_is_read_as_the_rows_query_answers() -> None:
    """Flips if an older hub's JSON reply is read as other rows than query reads from the same bytes."""
    reply = _json_reply(_ROWS, max_reply_bytes=4096)
    nats = _RawNats(reply, reply)
    client = _client(nats)

    answer = await client.query_arrow("ds", "SELECT race, votes, share, called, margin FROM t", types=_TYPES)
    rows = await client.query("ds", "SELECT race, votes, share, called, margin FROM t")

    assert answer.rows.to_pylist() == rows.rows == _ROWS
    assert (answer.truncated, answer.max_reply_bytes, answer.rows_bytes) == (False, 4096, len(reply))


async def test_a_json_value_not_of_its_type_is_refused() -> None:
    """Flips if a number in a column asked as text is answered as some other value than query's."""
    nats = _RawNats(_json_reply([{**_ROWS[0], "race": 1}]))

    with pytest.raises(ReplyNotArrowError):
        await _client(nats).query_arrow("ds", "SELECT race, votes, share, called, margin FROM t", types=_TYPES)


async def test_a_refusal_raises_the_hubs_code() -> None:
    refusal = DatasourceQueryResponse(success=False, error_code="ACCESS_DENIED", error_message="no grant")
    nats = _RawNats(refusal.model_dump_json().encode())

    with pytest.raises(DatasourceQueryError) as raised:
        await _client(nats).query_arrow("ds", "SELECT 1 AS race", types={"race": pa.int64()})

    assert raised.value.error_code == "ACCESS_DENIED"


async def test_a_busy_hub_is_asked_again(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("threetears.datasources.query_client.BUSY_BACKOFF_SECONDS", 0.0)
    busy = DatasourceQueryResponse(success=False, error_code=DATASOURCE_BUSY, error_message="busy")
    nats = _RawNats(busy.model_dump_json().encode(), _json_reply(_ROWS))

    answer = await _client(nats).query_arrow("ds", "SELECT race, votes, share, called, margin FROM t", types=_TYPES)

    assert answer.rows.num_rows == 2
    assert len(nats.sent) == 2


async def test_only_query_arrow_carries_the_arrow_fields() -> None:
    """Flips if a JSON ask carries reply_format or arrow_schema (its bytes would change for every hub), or an Arrow
    ask does not."""
    nats = _RawNats(_json_reply(_ROWS), _json_reply(_ROWS))
    client = _client(nats)

    await client.query_arrow("ds", "SELECT race, votes, share, called, margin FROM t", types=_TYPES)
    await client.query("ds", "SELECT race, votes, share, called, margin FROM t")

    assert nats.sent[0]["reply_format"] == "arrow"
    assert nats.sent[0]["arrow_schema"] == arrow_schema_text(_TYPES)
    plain = json.loads(nats.typed[0].model_dump_json())
    assert "reply_format" not in plain and "arrow_schema" not in plain


def test_the_schema_sent_reads_back_as_the_types() -> None:
    sent = arrow_schema_text(_TYPES)
    schema = pa.ipc.read_schema(pa.py_buffer(base64.b64decode(sent, validate=True)))
    assert schema == pa.schema(list(_TYPES.items()))


def test_an_arrow_ask_names_its_schema_and_is_a_query() -> None:
    """Flips if a request asking Arrow with no schema, or Arrow for an ask that is no query, is let through."""
    with pytest.raises(ValueError, match="arrow"):
        DatasourceQueryRequest(correlation_id=uuid7(), identity_token=_TOKEN, query="SELECT 1", reply_format="arrow")  # type: ignore[arg-type]


async def test_the_warehouse_types_are_the_forms_its_columns_reply_in() -> None:
    """Flips if a column's Arrow type is not its warehouse type's JSON form (a numeric as text, an integer as a
    number), or a column the catalogue lacks is given a type."""

    class _Catalogue:
        async def query(self, datasource_name: str, query: str, params: list[Any] | None = None) -> Any:
            assert params == ["reporting", "results"]
            rows = [
                {"column_name": "race", "data_type": "character varying"},
                {"column_name": "candidate_id", "data_type": "integer"},
                {"column_name": "margin", "data_type": "numeric"},
                {"column_name": "share", "data_type": "double precision"},
                {"column_name": "called_at", "data_type": "timestamp without time zone"},
                {"column_name": "called", "data_type": "boolean"},
            ]
            return DatasourceQueryResult(rows=rows, row_count=len(rows), truncated=False, correlation_id=uuid7())

    types = await warehouse_arrow_types(
        _Catalogue(),  # type: ignore[arg-type]
        "ds",
        "reporting.results",
        ["race", "candidate_id", "margin", "share", "called_at", "called"],
        extra={"row_hash": pa.string()},
    )

    assert types == {
        "race": pa.string(),
        "candidate_id": pa.int64(),
        "margin": pa.string(),
        "share": pa.float64(),
        "called_at": pa.string(),
        "called": pa.bool_(),
        "row_hash": pa.string(),
    }
    with pytest.raises(ValueError, match="no column"):
        await warehouse_arrow_types(_Catalogue(), "ds", "reporting.results", ["absent"])  # type: ignore[arg-type]
