"""A table loaded from files: the ``l3.load_files`` request, and the SQL store's load on a direct pool.

What the pod sends is the contract the hub's broker implements, so these pin the request body field
for field (the subject, every field, how a constant crosses the rail), how long the pod waits for
the reply, and how a reply is read (a refusal raises as every L3 reply's does, a key already held as
the asyncpg error a direct pool raises, a write's generations are kept, a count other than the
files' is refused). The direct pool's load is pinned on real parquet files read through ``file://``.
"""

from __future__ import annotations

import io
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import asyncpg
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from threetears.nats import Subject
from threetears.nats.errors import RequestError

from threetears.core.backends.broker_generation import BrokerGenerationSource
from threetears.core.backends.nats_proxy import (
    CONSTRAINT_VIOLATION_ERROR_CODE,
    NatsProxyL3Backend,
)
from threetears.core.backends.protocol import (
    LOAD_FILES_MAX_WAIT_SECONDS,
    FileLoad,
    LoadFile,
    LoadingStore,
    LoadResult,
)
from threetears.core.backends.sql import SqlL3Backend, bound_request_connection
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    INT_TYPE,
    JSONB_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    TableSchema,
)
from threetears.core.exceptions import DataLayerUnavailableError

SCHEMA = TableSchema(
    name="lsf_rows",
    primary_key=("race", "geo"),
    columns=[
        Column("race", STRING_TYPE),
        Column("geo", STRING_TYPE),
        Column("votes", INT_TYPE, nullable=True),
        Column("source_id", UUID_TYPE, nullable=True),
        Column("loaded_at", DATETIMETZ_TYPE, nullable=True),
        Column("detail", JSONB_TYPE, nullable=True),
    ],
)

_WHEN = datetime(2026, 10, 22, 12, 30, tzinfo=UTC)
_SOURCE = UUID("019d9a00-0000-7000-8000-000000000001")
_FILES = [
    LoadFile(url="https://bucket.s3.amazonaws.com/exports/lsf/0000_part_00.parquet?sig=a", rows=120_000),
    LoadFile(url="https://bucket.s3.amazonaws.com/exports/lsf/0001_part_00.parquet?sig=b", rows=0),
]


class _Broker:
    """the bus a proxy sends on, keeping every request and the wait it gave each, answering each with the next reply."""

    def __init__(self, *replies: dict[str, Any]) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.timeouts: list[timedelta | None] = []
        self._replies = list(replies)

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta | None = None) -> bytes:
        self.timeouts.append(timeout)
        self.requests.append((subject.path, json.loads(payload)))
        reply = self._replies.pop(0) if self._replies else _ok()
        return json.dumps(reply).encode("utf-8")

    def proxy(self) -> NatsProxyL3Backend:
        nats = MagicMock()
        nats.request_raw = AsyncMock(side_effect=self.request_raw)
        return NatsProxyL3Backend(
            nats_client=nats,
            namespace_prefix="test",
            agent_id="agent-123",
            identity_token=lambda: "test-identity-token",
        )


def _ok(**fields: Any) -> dict[str, Any]:
    return {
        "success": True,
        "row_count": 120_000,
        "files": [{"index": 0, "rows_loaded": 120_000}, {"index": 1, "rows_loaded": 0}],
        **fields,
    }


async def _load(proxy: NatsProxyL3Backend, **kwargs: Any) -> LoadResult:
    arguments: dict[str, Any] = {
        "columns": ["geo", "votes", "source_id"],
        "constants": {"race": "r1", "loaded_at": _WHEN, "batch": _SOURCE},
        "rows_per_transaction": 50_000,
        "timeout_ms": 60_000,
        **kwargs,
    }
    return await proxy.load_files("lsf_rows", _FILES, **arguments)


# -- l3.load_files --------------------------------------------------------------------------------


async def test_a_load_sends_one_request_carrying_every_field_of_the_contract() -> None:
    broker = _Broker()
    await _load(broker.proxy())
    [(subject, body)] = broker.requests
    assert subject == "test.l3.load_files"
    assert set(body) == {
        "correlation_id",
        "identity_token",
        "namespace",
        "table",
        "columns",
        "constants",
        "files",
        "format",
        "rows_per_transaction",
        "timeout_ms",
    }
    assert body["identity_token"] == "test-identity-token"
    assert body["namespace"] == "agents.agent-123"
    assert body["table"] == "lsf_rows"
    assert body["columns"] == ["geo", "votes", "source_id"]
    assert body["files"] == [{"url": file.url, "rows": file.rows} for file in _FILES]
    assert body["format"] == "parquet"
    assert body["rows_per_transaction"] == 50_000
    assert body["timeout_ms"] == 60_000


async def test_constants_cross_the_rail_as_a_statement_s_parameters_do() -> None:
    broker = _Broker()
    proxy = broker.proxy()
    await _load(proxy)
    constants = broker.requests[0][1]["constants"]
    assert constants == {"race": "r1", "loaded_at": "2026-10-22T12:30:00+00:00", "batch": str(_SOURCE)}
    await proxy.execute("SELECT $1, $2, $3", "r1", _WHEN, _SOURCE)
    assert broker.requests[-1][1]["params"] == list(constants.values())


async def test_a_load_binds_a_namespace_it_is_given_as_a_query_does() -> None:
    broker = _Broker()
    await _load(broker.proxy(), namespace="tools.reports")
    assert broker.requests[0][1]["namespace"] == "tools.reports"


async def test_the_reply_is_read_as_the_rows_written_every_file_s_and_each_file_s() -> None:
    result = await _load(_Broker().proxy())
    assert result == LoadResult(
        row_count=120_000, files=(FileLoad(index=0, rows_loaded=120_000), FileLoad(index=1, rows_loaded=0))
    )


async def test_the_pod_waits_out_every_transaction_the_files_need() -> None:
    # 120 000 rows at 50 000 a transaction is three transactions; the empty file is still fetched,
    # so one more: four, at 60 s each, and the two seconds' grace every L3 request has
    broker = _Broker()
    await _load(broker.proxy())
    assert broker.timeouts == [timedelta(seconds=4 * 60 + 2)]


@pytest.mark.parametrize(
    ("rows", "per_transaction", "timeout_ms", "wait"),
    [
        ([10], 10, 1000, 3.0),
        ([11], 10, 1000, 4.0),
        ([0], 10, 1000, 3.0),
        ([25, 5], 10, 2000, 2 * 4 + 2.0),
        ([], 10, 1000, 3.0),
    ],
)
async def test_the_wait_is_the_bound_on_a_transaction_times_the_transactions(
    rows: list[int], per_transaction: int, timeout_ms: int, wait: float
) -> None:
    files = [LoadFile(url=f"https://x/{index}", rows=count) for index, count in enumerate(rows)]
    broker = _Broker(
        _ok(row_count=sum(rows), files=[{"index": index, "rows_loaded": count} for index, count in enumerate(rows)])
    )
    await broker.proxy().load_files(
        "lsf_rows",
        files,
        columns=["geo"],
        constants={},
        rows_per_transaction=per_transaction,
        timeout_ms=timeout_ms,
    )
    assert broker.timeouts == [timedelta(seconds=wait)]


async def test_the_wait_never_passes_the_cap_however_many_rows_the_files_hold() -> None:
    # 15.5 million rows at the defaults is 310 transactions of 300 s: a day's wait for one reply
    files = [LoadFile(url="https://x/0", rows=15_500_000)]
    broker = _Broker(_ok(row_count=15_500_000, files=[{"index": 0, "rows_loaded": 15_500_000}]))
    await broker.proxy().load_files(
        "lsf_rows", files, columns=["geo"], constants={}, rows_per_transaction=50_000, timeout_ms=300_000
    )
    assert broker.timeouts == [timedelta(seconds=LOAD_FILES_MAX_WAIT_SECONDS)]


async def test_a_load_logs_its_start_and_wait_and_a_reply_that_never_came(caplog: pytest.LogCaptureFixture) -> None:
    nats = MagicMock()
    nats.request_raw = AsyncMock(side_effect=RequestError("timed out"))
    proxy = NatsProxyL3Backend(
        nats_client=nats, namespace_prefix="test", agent_id="agent-123", identity_token=lambda: "token"
    )
    with caplog.at_level("INFO"), pytest.raises(DataLayerUnavailableError, match="NATS request failed"):
        await _load(proxy)
    [sent] = [record for record in caplog.records if record.getMessage().startswith("l3.load_files sent")]
    assert sent.extra_data == {  # type: ignore[attr-defined]
        "table": "lsf_rows",
        "namespace": "agents.agent-123",
        "files": 2,
        "rows": 120_000,
        "wait_s": 4 * 60 + 2,
    }
    [lost] = [record for record in caplog.records if "got no reply" in record.getMessage()]
    assert lost.levelname == "ERROR"
    assert "may still be loading" in lost.getMessage()


async def test_any_other_request_still_waits_the_backend_s_own_bound() -> None:
    broker = _Broker({"success": True, "rows": []})
    await broker.proxy().fetch("SELECT 1")
    assert broker.timeouts == [timedelta(seconds=5 + 2)]


async def test_a_refused_load_raises_as_every_l3_refusal_does() -> None:
    broker = _Broker({"success": False, "error_code": "FORBIDDEN", "error_message": "not a declared table"})
    with pytest.raises(DataLayerUnavailableError, match="L3 load_files failed: FORBIDDEN"):
        await _load(broker.proxy())


async def test_a_load_naming_a_key_already_held_raises_the_unique_violation_a_direct_pool_raises() -> None:
    broker = _Broker(
        {
            "success": False,
            "error_code": CONSTRAINT_VIOLATION_ERROR_CODE,
            "error_message": "duplicate key value violates unique constraint",
            "sqlstate": "23505",
            "constraint_name": "lsf_rows_pkey",
            "table_name": "lsf_rows",
        }
    )
    with pytest.raises(asyncpg.UniqueViolationError):
        await _load(broker.proxy())


async def test_the_generation_a_load_s_reply_names_is_handed_to_the_advance_after_it() -> None:
    await _load(_Broker(_ok(generations={"lsf_rows": "inc:12"})).proxy())
    assert await BrokerGenerationSource().advance("lsf_rows") == "inc:12"


async def test_a_reply_counting_other_than_the_files_hold_is_refused() -> None:
    broker = _Broker(_ok(row_count=119_999))
    with pytest.raises(DataLayerUnavailableError, match="answered 119999 rows; its files hold 120000"):
        await _load(broker.proxy())


@pytest.mark.parametrize(
    "reply",
    [_ok(row_count=None), _ok(files={"0": 1}), _ok(files=[{"index": 0, "rows_loaded": "many"}]), _ok(files=[3])],
    ids=["no-count", "files-not-a-list", "count-not-a-count", "file-not-an-object"],
)
async def test_a_reply_whose_counts_are_not_counts_is_refused(reply: dict[str, Any]) -> None:
    with pytest.raises(DataLayerUnavailableError, match="L3 load_files answered"):
        await _load(_Broker(reply).proxy())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rows_per_transaction": 0}, "rows_per_transaction must be at least 1"),
        ({"timeout_ms": 0}, "timeout_ms must be at least 1"),
        ({"columns": []}, "none were named"),
    ],
)
async def test_a_load_that_cannot_run_is_refused_before_anything_is_sent(kwargs: dict[str, Any], match: str) -> None:
    broker = _Broker()
    with pytest.raises(ValueError, match=match):
        await _load(broker.proxy(), **kwargs)
    assert broker.requests == []


async def test_a_sql_store_over_the_rail_loads_through_the_broker() -> None:
    broker = _Broker()
    store = SqlL3Backend(broker.proxy())
    store.register_schema("lsf_rows", SCHEMA)
    result = await store.load_files(
        "lsf_rows",
        _FILES,
        columns=["geo", "votes"],
        constants={"race": "r1"},
        rows_per_transaction=50_000,
        timeout_ms=60_000,
    )
    assert result.row_count == 120_000
    assert [subject for subject, _ in broker.requests] == ["test.l3.load_files"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"columns": ["geo", "nowhere"]}, "has no column 'nowhere'"),
        ({"columns": ["geo", "race"]}, "named twice"),
        ({"columns": ["votes"]}, "every key column; 'geo'"),
        ({"columns": []}, "none were named"),
        ({"rows_per_transaction": 0}, "rows_per_transaction must be at least 1"),
        ({"timeout_ms": 0}, "timeout_ms must be at least 1"),
    ],
)
async def test_a_sql_store_over_the_rail_refuses_what_a_direct_pool_refuses_before_anything_is_sent(
    kwargs: dict[str, Any], match: str
) -> None:
    broker = _Broker()
    store = SqlL3Backend(broker.proxy())
    store.register_schema("lsf_rows", SCHEMA)
    arguments: dict[str, Any] = {
        "columns": ["geo", "votes"],
        "constants": {"race": "r1"},
        "rows_per_transaction": 50_000,
        "timeout_ms": 60_000,
        **kwargs,
    }
    with pytest.raises(ValueError, match=match):
        await store.load_files("lsf_rows", _FILES, **arguments)
    assert broker.requests == []


async def test_a_sql_store_over_the_rail_needs_the_table_s_schema() -> None:
    broker = _Broker()
    with pytest.raises(ValueError, match="schema of 'lsf_rows' registered"):
        await SqlL3Backend(broker.proxy()).load_files(
            "lsf_rows", _FILES, columns=["geo"], constants={"race": "r1"}, rows_per_transaction=1, timeout_ms=1
        )
    assert broker.requests == []


def test_every_sql_backend_and_the_proxy_can_load_files() -> None:
    assert isinstance(SqlL3Backend(object()), LoadingStore)
    assert isinstance(_Broker().proxy(), LoadingStore)


@pytest.mark.parametrize(("url", "rows"), [("", 1), ("https://x", -1), ("https://x", True)])
def test_a_load_file_with_no_url_or_no_count_is_refused(url: str, rows: Any) -> None:
    with pytest.raises(ValueError, match="LoadFile"):
        LoadFile(url=url, rows=rows)


# -- the SQL store's load on a direct pool --------------------------------------------------------


class _Conn:
    """an asyncpg-shaped connection keeping each copy and whether its transaction committed."""

    def __init__(self, pool: _Pool) -> None:
        self._pool = pool

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        copies: list[tuple[str, list[tuple[Any, ...]], list[str]]] = []
        self._pool.open = copies
        yield
        self._pool.committed.append(copies)

    async def copy_records_to_table(self, table_name: str, *, records: Any, columns: Any) -> str:
        if self._pool.fail_on is not None and any(record[0] == self._pool.fail_on for record in records):
            raise asyncpg.UniqueViolationError("duplicate key value violates unique constraint")
        self._pool.open.append((table_name, list(records), list(columns)))
        return f"COPY {len(records)}"


class _Pool:
    """an asyncpg-shaped pool: each committed transaction's copies, in order."""

    def __init__(self) -> None:
        self.committed: list[list[tuple[str, list[tuple[Any, ...]], list[str]]]] = []
        self.open: list[tuple[str, list[tuple[Any, ...]], list[str]]] = []
        self.fail_on: str | None = None

    @asynccontextmanager
    async def _acquire(self) -> AsyncIterator[_Conn]:
        yield _Conn(self)

    def acquire(self) -> Any:
        return self._acquire()


def _store(pool: _Pool) -> SqlL3Backend:
    store = SqlL3Backend(pool)
    store.register_schema("lsf_rows", SCHEMA)
    return store


def _parquet_file(path: Path, rows: list[dict[str, Any]]) -> LoadFile:
    sink = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), sink)
    path.write_bytes(sink.getvalue())
    return LoadFile(url=path.as_uri(), rows=len(rows))


def _geo_rows(*geos: str) -> list[dict[str, Any]]:
    return [
        {"geo": geo, "votes": index, "source_id": str(_SOURCE), "extra": "ignored"} for index, geo in enumerate(geos)
    ]


async def _direct(store: SqlL3Backend, files: list[LoadFile], **kwargs: Any) -> LoadResult:
    arguments: dict[str, Any] = {
        "columns": ["geo", "votes", "source_id"],
        "constants": {"race": "r1", "loaded_at": _WHEN},
        "rows_per_transaction": 2,
        "timeout_ms": 10_000,
        **kwargs,
    }
    return await store.load_files("lsf_rows", files, **arguments)


async def test_a_direct_load_copies_each_file_ordered_by_key_in_transactions_of_the_size_asked(
    tmp_path: Path,
) -> None:
    pool = _Pool()
    files = [
        _parquet_file(tmp_path / "a.parquet", _geo_rows("g3", "g1", "g2")),
        _parquet_file(tmp_path / "b.parquet", _geo_rows("g9")),
    ]
    result = await _direct(_store(pool), files)
    assert result == LoadResult(row_count=4, files=(FileLoad(0, 3), FileLoad(1, 1)))
    assert [[len(records) for _, records, _ in copies] for copies in pool.committed] == [[2], [1], [1]]
    [(table, records, columns)] = pool.committed[0]
    assert table == "lsf_rows"
    assert columns == ["geo", "votes", "source_id", "race", "loaded_at"]
    # the file's columns, then the constants; ordered by the key (race, geo); a uuid as an insert binds it
    assert records == [("g1", 1, _SOURCE, "r1", _WHEN), ("g2", 2, _SOURCE, "r1", _WHEN)]
    assert pool.committed[1][0][1] == [("g3", 0, _SOURCE, "r1", _WHEN)]


async def test_a_key_already_held_stops_the_load_at_that_transaction(tmp_path: Path) -> None:
    pool = _Pool()
    pool.fail_on = "g3"
    files = [_parquet_file(tmp_path / "a.parquet", _geo_rows("g1", "g2", "g3", "g4"))]
    with pytest.raises(asyncpg.UniqueViolationError):
        await _direct(_store(pool), files)
    assert [[record[0] for record in copies[0][1]] for copies in pool.committed] == [["g1", "g2"]]


async def test_a_file_holding_other_than_its_declared_rows_is_refused_before_its_rows_are_written(
    tmp_path: Path,
) -> None:
    pool = _Pool()
    file = _parquet_file(tmp_path / "a.parquet", _geo_rows("g1", "g2"))
    with pytest.raises(ValueError, match="file 0 holds 2 rows; the load declares 3"):
        await _direct(_store(pool), [LoadFile(url=file.url, rows=3)])
    assert pool.committed == []


async def test_a_file_lacking_a_column_is_refused(tmp_path: Path) -> None:
    pool = _Pool()
    file = _parquet_file(tmp_path / "a.parquet", [{"geo": "g1"}])
    with pytest.raises(ValueError, match="file 0: the file holds no column 'votes'"):
        await _direct(_store(pool), [file])
    assert pool.committed == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"columns": ["geo", "nowhere"]}, "has no column 'nowhere'"),
        ({"columns": ["geo", "race"]}, "named twice"),
        ({"columns": ["votes"]}, "every key column; 'geo'"),
        ({"columns": ["geo", "detail"]}, "cannot write the jsonb column 'detail'"),
        ({"columns": []}, "none were named"),
        ({"rows_per_transaction": 0}, "rows_per_transaction must be at least 1"),
        ({"timeout_ms": 0}, "timeout_ms must be at least 1"),
    ],
)
async def test_a_load_that_cannot_run_is_refused_before_any_file_is_fetched(kwargs: dict[str, Any], match: str) -> None:
    pool = _Pool()
    nowhere = LoadFile(url="file:///nowhere/at/all.parquet", rows=1)
    with pytest.raises(ValueError, match=match):
        await _direct(_store(pool), [nowhere], **kwargs)
    assert pool.committed == []


async def test_a_direct_load_needs_the_table_s_schema() -> None:
    with pytest.raises(ValueError, match="schema of 'lsf_rows' registered"):
        await _direct(SqlL3Backend(_Pool()), [])


async def test_a_load_is_refused_inside_a_request_s_bound_transaction() -> None:
    async with bound_request_connection(object()):
        with pytest.raises(ValueError, match="commits its own transactions"):
            await _direct(_store(_Pool()), [])


async def test_a_url_of_another_scheme_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not 'ftp'"):
        await _direct(_store(_Pool()), [LoadFile(url="ftp://host/a.parquet", rows=1)])


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/a.parquet",
        "http://internal.service.cluster.local/a.parquet",
        "http://bucket.s3.amazonaws.com/a.parquet",
    ],
)
async def test_plain_http_off_this_machine_is_refused_before_any_file_is_fetched(tmp_path: Path, url: str) -> None:
    pool = _Pool()
    good = _parquet_file(tmp_path / "a.parquet", _geo_rows("g1"))
    with pytest.raises(ValueError, match="plain http only from this machine"):
        await _direct(_store(pool), [good, LoadFile(url=url, rows=1)])
    assert pool.committed == []


async def test_a_file_url_a_filesystem_store_presigns_is_loaded(tmp_path: Path) -> None:
    pool = _Pool()
    result = await _direct(_store(pool), [_parquet_file(tmp_path / "a.parquet", _geo_rows("g1"))])
    assert result.row_count == 1
