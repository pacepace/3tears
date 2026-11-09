"""An L3 rail read answered in Arrow: the hub sends a read's rows as an Arrow IPC stream behind a JSON envelope.

``fetch_arrow`` and ``fetch_arrow_page`` ask for it (``reply_format: "arrow"`` and the schema of their ``types`` as
``arrow_schema``); every other read and write asks for nothing new. A reply beginning ``L3ARROW1`` carries its
envelope's length (4 bytes, big-endian), the envelope (the JSON reply without ``rows``) and the IPC stream; anything
else is a JSON reply, read as before -- an older hub ignores the two fields and answers JSON. The contract the hub
keeps: an Arrow reply's table is what ``reply_rows`` makes of the JSON reply of the same rows, so a pod gets one table
whichever the hub sent.
"""

from __future__ import annotations

import base64
import json
import struct
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

pa = pytest.importorskip("pyarrow")

from threetears.core.backends import arrow_reply  # noqa: E402
from threetears.core.backends import nats_proxy as nats_proxy_module  # noqa: E402
from threetears.core.backends.arrow_reply import ReplyNotArrowError, reply_rows  # noqa: E402
from threetears.core.backends.nats_proxy import NatsProxyL3Backend  # noqa: E402
from threetears.core.backends.protocol import L3AnswerCutError, L3Page  # noqa: E402
from threetears.core.collections.complete_copy import rail_arrow_types  # noqa: E402
from threetears.core.collections.schema_backed import (  # noqa: E402
    BIGINT_TYPE,
    BOOL_TYPE,
    DATETIMETZ_TYPE,
    NUMERIC_TYPE,
    STRING_TYPE,
    Column,
)
from threetears.core.exceptions import DataLayerUnavailableError  # noqa: E402
from threetears.core.keyset import _next_rows, keyset_arrow_pages  # noqa: E402

_COLUMNS = [
    Column("id", BIGINT_TYPE),
    Column("name", STRING_TYPE, nullable=True),
    Column("share", NUMERIC_TYPE, nullable=True, precision=12, scale=4),
    Column("called", BOOL_TYPE, nullable=True),
    Column("date_updated", DATETIMETZ_TYPE, nullable=True),
]
_TYPES = rail_arrow_types(_COLUMNS)

_ROWS: list[dict[str, Any]] = [
    {"id": 1, "name": "Travis", "share": "0.5000", "called": True, "date_updated": "2026-11-03T21:00:00Z"},
    {"id": 2, "name": None, "share": None, "called": None, "date_updated": None},
    {"id": 3, "name": "Harris", "share": "0.1250", "called": False, "date_updated": "2026-11-04T01:02:03.456Z"},
]


def _envelope(**fields: Any) -> dict[str, Any]:
    """an ``l3.query`` reply's fields beside its rows, as the hub writes them."""
    envelope: dict[str, Any] = {
        "success": True,
        "row_count": 0,
        "duration_ms": 1,
        "truncated": False,
        "max_reply_bytes": None,
        "generations": None,
        "generations_failed": None,
        "error_code": None,
        "error_message": None,
    }
    envelope.update(fields)
    return envelope


def _json_reply(rows: list[dict[str, Any]], **fields: Any) -> bytes:
    """the JSON reply of ``rows``, as an older hub (or any non-Arrow answer) sends it."""
    return json.dumps({**_envelope(row_count=len(rows), **fields), "rows": rows}, separators=(",", ":")).encode()


def _arrow_reply(rows: list[dict[str, Any]], *, rows_json_bytes: bool = True, **fields: Any) -> bytes:
    """the Arrow reply of ``rows``, built as the hub builds it: ``reply_rows`` over the rows' JSON, written as IPC."""
    rows_json = json.dumps(rows, separators=(",", ":")).encode()
    table = reply_rows(b'{"success":true,"rows":' + rows_json + b"}", _TYPES).rows
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    said = _envelope(row_count=len(rows), **fields)
    if rows_json_bytes:
        said["rows_json_bytes"] = len(rows_json)
    envelope = json.dumps(said, separators=(",", ":")).encode()
    stream: bytes = sink.getvalue().to_pybytes()
    return b"L3ARROW1" + struct.pack("!I", len(envelope)) + envelope + stream


def _proxy(*replies: bytes) -> tuple[NatsProxyL3Backend, AsyncMock]:
    nc = MagicMock()
    nc.request_raw = AsyncMock(side_effect=list(replies))
    return (
        NatsProxyL3Backend(nats_client=nc, namespace_prefix="test", agent_id="a", identity_token=lambda: "t"),
        nc.request_raw,
    )


def _sent(request: AsyncMock, index: int = -1) -> dict[str, Any]:
    """the ``l3.query`` payload of one request the proxy sent."""
    return json.loads(request.await_args_list[index].kwargs["payload"])  # type: ignore[no-any-return]


# (a) an Arrow reply
async def test_an_arrow_reply_is_read_as_its_table() -> None:
    proxy, _ = _proxy(_arrow_reply(_ROWS))
    table = await proxy.fetch_arrow("SELECT * FROM t", types=_TYPES)
    assert table.to_pylist() == reply_rows(_json_reply(_ROWS), _TYPES).rows.to_pylist()
    assert table.schema == pa.schema([pa.field(name, kind) for name, kind in _TYPES.items()])


async def test_an_arrow_reply_cut_at_the_cap_raises_as_json_does() -> None:
    proxy, _ = _proxy(_arrow_reply(_ROWS, truncated=True), _json_reply(_ROWS, truncated=True))
    with pytest.raises(L3AnswerCutError, match="cut the answer at 3 rows"):
        await proxy.fetch_arrow("SELECT * FROM t", types=_TYPES)
    with pytest.raises(L3AnswerCutError, match="cut the answer at 3 rows"):
        await proxy.fetch_arrow("SELECT * FROM t", types=_TYPES)


async def test_an_arrow_page_says_its_cut_and_cap() -> None:
    proxy, _ = _proxy(_arrow_reply(_ROWS, truncated=True, max_reply_bytes=4096))
    page = await proxy.fetch_arrow_page("SELECT * FROM t LIMIT 10", types=_TYPES)
    assert page.truncated is True
    assert page.reply_cap == 4096
    assert page.rows.num_rows == 3


async def test_an_arrow_page_is_sized_by_the_bytes_its_json_would_take() -> None:
    """the hub cuts by its JSON reply's bytes: an Arrow page is measured so, never by its own smaller bytes."""
    json_reply = _json_reply(_ROWS, max_reply_bytes=4096)
    proxy, _ = _proxy(_arrow_reply(_ROWS, max_reply_bytes=4096), json_reply)
    by_arrow = await proxy.fetch_arrow_page("SELECT * FROM t LIMIT 10", types=_TYPES)
    by_json = await proxy.fetch_arrow_page("SELECT * FROM t LIMIT 10", types=_TYPES)
    # the Arrow envelope says rows_json_bytes, which the JSON reply's own envelope does not: within that key
    assert abs(by_arrow.reply_bytes - by_json.reply_bytes) <= len(b',"rows_json_bytes":') + 4


async def test_an_arrow_page_that_does_not_say_its_json_bytes_keeps_its_rows() -> None:
    """a hub mid-roll may send Arrow without its rows' JSON bytes: the page is taken as having met the cap, and the
    next asks for as many rows as this one, never more by the Arrow's smaller bytes."""
    proxy, _ = _proxy(_arrow_reply(_ROWS, rows_json_bytes=False, max_reply_bytes=4096))
    page = await proxy.fetch_arrow_page("SELECT * FROM t LIMIT 10", types=_TYPES)
    assert page.reply_bytes == 0
    assert _next_rows(page, 3, 10) == 10


async def test_an_arrow_reply_is_read_on_another_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    threads: list[tuple[int, bool]] = []
    read = arrow_reply.read_reply

    def recording(reply: bytes, types: Any, json_rows: Any) -> Any:
        threads.append((threading.get_ident(), reply.startswith(b"L3ARROW1")))
        return read(reply, types, json_rows)

    monkeypatch.setattr(nats_proxy_module, "read_reply", recording)
    proxy, _ = _proxy(_arrow_reply(_ROWS))
    await proxy.fetch_arrow("SELECT * FROM t", types=_TYPES)
    assert threads and threads[0][0] != threading.get_ident() and threads[0][1]


async def test_an_arrow_reply_of_other_types_is_refused() -> None:
    proxy, _ = _proxy(_arrow_reply(_ROWS))
    with pytest.raises(ReplyNotArrowError, match="not the types asked for"):
        await proxy.fetch_arrow("SELECT * FROM t", types={**_TYPES, "id": pa.string()})


async def test_a_failed_arrow_envelope_raises_what_fetch_raises() -> None:
    failed = {"success": False, "error_code": "QUERY_FAILED", "error_message": "no such table"}
    proxy, _ = _proxy(_json_reply([], **failed))
    with pytest.raises(DataLayerUnavailableError) as by_fetch:
        await proxy.fetch("SELECT * FROM t")
    envelope = json.dumps(_envelope(**failed)).encode()
    arrow_failed = b"L3ARROW1" + struct.pack("!I", len(envelope)) + envelope
    proxy, _ = _proxy(arrow_failed)
    with pytest.raises(DataLayerUnavailableError) as by_arrow:
        await proxy.fetch_arrow("SELECT * FROM t", types=_TYPES)
    assert str(by_arrow.value) == str(by_fetch.value)


# (b) a JSON reply, from an older hub or one that could not answer Arrow
async def test_a_json_reply_is_read_as_today() -> None:
    proxy, _ = _proxy(_json_reply(_ROWS, max_reply_bytes=4096))
    page = await proxy.fetch_arrow_page("SELECT * FROM t LIMIT 10", types=_TYPES)
    reply = _json_reply(_ROWS, max_reply_bytes=4096)
    assert page.rows.equals(reply_rows(reply, _TYPES).rows)
    assert page.reply_bytes == len(reply)
    assert page.reply_cap == 4096


async def test_a_json_reply_not_of_the_types_raises_as_today() -> None:
    proxy, _ = _proxy(_json_reply([{**_ROWS[0], "id": "one"}]))
    with pytest.raises(ReplyNotArrowError):
        await proxy.fetch_arrow("SELECT * FROM t", types=_TYPES)


# (c) only the Arrow reads ask for Arrow
async def test_only_the_arrow_reads_carry_the_arrow_fields() -> None:
    proxy, request = _proxy(
        _json_reply(_ROWS), _json_reply(_ROWS), _json_reply(_ROWS), _json_reply(_ROWS), _json_reply([])
    )
    await proxy.fetch_arrow("SELECT * FROM t", types=_TYPES)
    await proxy.fetch_arrow_page("SELECT * FROM t LIMIT 10", types=_TYPES)
    await proxy.fetch("SELECT * FROM t")
    await proxy.fetchrow("SELECT * FROM t")
    await proxy.execute("DELETE FROM t")
    for index in (0, 1):
        sent = _sent(request, index)
        assert sent["reply_format"] == "arrow"
        assert isinstance(sent["arrow_schema"], str)
    for index in (2, 3, 4):
        sent = _sent(request, index)
        assert "reply_format" not in sent and "arrow_schema" not in sent


# (d) the schema the client sends is the one the hub reads
def test_the_schema_sent_reads_back_as_the_types() -> None:
    sent = arrow_reply.arrow_schema_text(_TYPES)
    schema = pa.ipc.read_schema(pa.py_buffer(base64.b64decode(sent, validate=True)))
    assert schema == pa.schema([pa.field(name, kind) for name, kind in _TYPES.items()])


# equivalence: an Arrow reply's table is reply_rows over the JSON reply of the same rows
@pytest.mark.parametrize("rows", [_ROWS, [], [_ROWS[1]] * 50])
async def test_an_arrow_reply_equals_its_json_replys_rows(rows: list[dict[str, Any]]) -> None:
    proxy, _ = _proxy(_arrow_reply(rows), _json_reply(rows))
    by_arrow = await proxy.fetch_arrow("SELECT * FROM t", types=_TYPES)
    by_json = await proxy.fetch_arrow("SELECT * FROM t", types=_TYPES)
    assert by_arrow.equals(reply_rows(_json_reply(rows), _TYPES).rows)
    assert by_arrow.equals(by_json)


def test_a_page_says_reply_bytes_as_the_pager_reads_them() -> None:
    assert _next_rows(L3Page(rows=None, truncated=False, reply_bytes=0, reply_cap=4096), 3, 10) == 10


async def test_the_keyset_pager_reads_a_table_in_arrow_pages() -> None:
    """``keyset_arrow_pages`` over the rail's proxy (as a copy follower reads) asks every page for Arrow, and reads the
    Arrow replies' rows in order, past each page's last key."""
    first, second = [{**row, "id": index} for index, row in enumerate(_ROWS * 2)][:4], [{**_ROWS[0], "id": 9}]
    proxy, request = _proxy(
        _arrow_reply(first, truncated=True, max_reply_bytes=1 << 20), _arrow_reply(second, max_reply_bytes=1 << 20)
    )
    pages = [
        page
        async for page in keyset_arrow_pages(
            proxy, f"SELECT {', '.join(_TYPES)} FROM t", ["id"], types=_TYPES, page_size=10
        )
    ]
    assert [row["id"] for page in pages for row in page.to_pylist()] == [0, 1, 2, 3, 9]
    assert all(_sent(request, index)["reply_format"] == "arrow" for index in range(2))
    assert _sent(request, 1)["params"] == [3]
