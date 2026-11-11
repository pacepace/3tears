"""DatasourceReader: the keyed helpers read a warehouse datasource through the query client, a cut answer refused."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import duckdb
import pydantic_core
import pytest

from threetears.core.keyed import (
    ROW_HASH_COLUMN,
    AnyOf,
    Dialect,
    HashColumn,
    HashKind,
    KeyBatch,
    Selected,
    Sort,
    counts_by,
    dialect_of,
    read_changed,
    read_keyed,
    read_past_marks,
    row_hash_sql,
)
from threetears.datasources.entities import DataSourceType
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


async def test_a_selected_expression_reaches_the_warehouse_as_written() -> None:
    """Flips if a Selected column is quoted as one identifier on its way to the warehouse, or loses its name."""
    client = _Client([{"run_id": 7, "copied_at": "2026-11-03 20:00:00.123"}])
    reader = DatasourceReader(client, "redshift")  # type: ignore[arg-type]
    rows = await read_keyed(
        reader,
        "enr.lsf_runs",
        keys=KeyBatch(("run_id",), [(7,)], as_list=True),
        columns=["run_id", Selected("CAST(t.copied_at AS VARCHAR)", "copied_at")],
    )
    assert rows == [{"run_id": 7, "copied_at": "2026-11-03 20:00:00.123"}]
    ((_, query, _),) = client.asked
    assert 'SELECT t."run_id", CAST(t.copied_at AS VARCHAR) AS "copied_at" FROM' in query


class _Scripted(_Client):
    """a query client answering each query with the next canned answer."""

    def __init__(self, answers: list[list[dict[str, Any]]]) -> None:
        super().__init__([])
        self._answers = answers

    async def query(
        self, datasource: str, query: str, params: list[Any] | None = None, *, user_identity_token: str | None = None
    ) -> DatasourceQueryResult:
        self._rows = self._answers[len(self.asked)]
        return await super().query(datasource, query, params, user_identity_token=user_identity_token)


async def test_the_reader_names_its_engine_s_dialect_and_refuses_an_unnamed_one() -> None:
    """Flips if a datasource's engine does not decide the hash's spelling (a Redshift reader spelled as
    PostgreSQL's, or asked over the rail), or a reader naming no engine, or one no hash is spelled for, is taken."""
    client = _Client([])
    assert await dialect_of(DatasourceReader(client, "r", engine=DataSourceType.REDSHIFT)) is Dialect.REDSHIFT  # type: ignore[arg-type]
    assert await dialect_of(DatasourceReader(client, "p", engine=DataSourceType.POSTGRES)) is Dialect.POSTGRES  # type: ignore[arg-type]
    for reader in (DatasourceReader(client, "s", engine=DataSourceType.SNOWFLAKE), DatasourceReader(client, "u")):  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="no row hash is spelled for"):
            await dialect_of(reader)
    assert client.asked == []


async def test_a_changed_rows_read_runs_on_the_warehouse_through_the_rail() -> None:
    """Flips if read_changed over a Redshift datasource spells its hash other than Redshift's, binds its keys as
    arrays, asks the rail for the engine, or reads in full a row whose hash the caller holds."""
    hashed = [HashColumn("name"), HashColumn("votes", HashKind.INTEGER)]
    client = _Scripted(
        [
            [
                {"state": "PA", "geo": 1, ROW_HASH_COLUMN: "a" * 32},
                {"state": "PA", "geo": 2, ROW_HASH_COLUMN: "c" * 32},
            ],
            [{"state": "PA", "geo": 2, "name": "n2", "votes": 7, ROW_HASH_COLUMN: "c" * 32}],
        ]
    )
    reader = DatasourceReader(client, "redshift", engine=DataSourceType.REDSHIFT)  # type: ignore[arg-type]
    changed = await read_changed(
        reader,
        "enr.results",
        key=["state", Sort("geo", cast="bigint")],
        columns=["state", "geo", "name", "votes"],
        held={("PA", 1): "a" * 32, ("PA", 2): "b" * 32, ("PA", 3): "d" * 32},
        hash_columns=hashed,
        by={"state": "PA"},
    )
    assert [row["geo"] for row in changed.rows] == [2] and changed.gone_keys == [("PA", 3)]
    (_, scope, _), (_, keyed, params) = client.asked
    assert row_hash_sql(hashed, Dialect.REDSHIFT) in scope and " LIMIT " in scope
    assert '(t."state", t."geo") IN (($1, $2))' in keyed and "unnest" not in keyed
    assert params[:2] == ["PA", 2]


class _ByteCappedHub:
    """a stand-in hub over an in-process DuckDB: it runs each query and cuts its reply at ``cap`` bytes of compact
    JSON (saying so, ``truncated``), announcing the cap on every reply, as the hub's broker does."""

    def __init__(self, db: duckdb.DuckDBPyConnection, cap: int) -> None:
        self._db = db
        self._cap = cap
        self.asked: list[str] = []
        self.cuts = 0

    async def query(
        self, datasource: str, query: str, params: list[Any] | None = None, *, user_identity_token: str | None = None
    ) -> DatasourceQueryResult:
        self.asked.append(query)
        cursor = self._db.execute(query, list(params or []))
        names = [column[0] for column in cursor.description or []]
        rows = [dict(zip(names, found, strict=True)) for found in cursor.fetchall()]
        kept: list[dict[str, Any]] = []
        size = 2
        for row in rows:
            size += len(pydantic_core.to_json(row)) + 1
            if size > self._cap:
                self.cuts += 1
                break
            kept.append(row)
        return DatasourceQueryResult(
            rows=kept,
            row_count=len(kept),
            truncated=len(kept) < len(rows),
            correlation_id=uuid4(),
            max_reply_bytes=self._cap,
        )


class _DuckReader(DatasourceReader):
    """a datasource reader whose stand-in hub answers in DuckDB's dialect."""

    @property
    def sql_dialect(self) -> Dialect | str:
        return Dialect.DUCKDB


async def test_a_changed_rows_read_reads_on_past_replies_the_hub_cut_by_bytes() -> None:
    """Flips if phase 1 over a datasource raises on (or stops at) a reply the hub cut by its bytes, or loses a key
    at a cut: every key of 5,000 comes back, so only the changed rows are read and only the gone keys named."""
    db = duckdb.connect()
    db.execute("CREATE TABLE results (state VARCHAR, geo BIGINT, name VARCHAR, PRIMARY KEY (state, geo))")
    db.execute("INSERT INTO results SELECT 'PA', i, 'n' || i FROM range(5000) r(i)")
    db.execute("INSERT INTO results SELECT 'OH', i, 'n' || i FROM range(100) r(i)")
    hashed = [HashColumn("name")]
    held: dict[tuple[Any, ...], str | None] = {
        ("PA", geo): found
        for geo, found in db.execute(
            f"SELECT geo, {row_hash_sql(hashed, Dialect.DUCKDB)} FROM results AS t WHERE state = 'PA'"
        ).fetchall()
    }
    db.execute("UPDATE results SET name = 'moved' WHERE state = 'PA' AND geo % 100 = 7")
    db.execute("DELETE FROM results WHERE state = 'PA' AND geo IN (11, 4999)")
    hub = _ByteCappedHub(db, cap=16_000)
    reader = _DuckReader(hub, "stand-in")  # type: ignore[arg-type]
    changed = await read_changed(
        reader,
        "results",
        key=["state", Sort("geo", cast="bigint")],
        columns=["state", "geo", "name"],
        held=held,
        hash_columns=hashed,
        by={"state": "PA"},
    )
    assert hub.cuts > 0
    assert sorted(row["geo"] for row in changed.rows) == list(range(7, 5000, 100))
    assert all(row["name"] == "moved" for row in changed.rows)
    assert sorted(changed.gone_keys) == [("PA", 11), ("PA", 4999)]


async def test_a_backfill_of_wide_rows_reads_on_past_replies_the_hub_cut_by_bytes() -> None:
    """Flips if phase 2 (read_keyed by the moved keys) raises on a batch the hub cut by its reply's bytes, takes the
    cut reply as every row, or pays a cut on every batch after the first (the fit it found is kept for the read)."""
    db = duckdb.connect()
    db.execute("CREATE TABLE geos (state VARCHAR, geo BIGINT, shape VARCHAR, PRIMARY KEY (state, geo))")
    db.execute("INSERT INTO geos SELECT 'TX', i, repeat(chr(CAST(65 + (i % 26) AS INTEGER)), 2000) || i FROM range(400) r(i)")
    hashed = [HashColumn("shape")]
    hub = _ByteCappedHub(db, cap=64_000)
    reader = _DuckReader(hub, "stand-in")  # type: ignore[arg-type]
    changed = await read_changed(
        reader,
        "geos",
        key=["state", Sort("geo", cast="bigint")],
        columns=["state", "geo", "shape"],
        held=dict.fromkeys((("TX", geo) for geo in range(400)), None),  # held with no hash: every key wanted
        hash_columns=hashed,
        by={"state": "TX"},
    )
    expected = db.execute(
        f"SELECT state, geo, shape, {row_hash_sql(hashed, Dialect.DUCKDB)} FROM geos AS t ORDER BY geo"
    ).fetchall()
    assert sorted((row["state"], row["geo"], row["shape"], row[ROW_HASH_COLUMN]) for row in changed.rows) == expected
    assert changed.gone_keys == []
    keyed = [query for query in hub.asked if " LIMIT " not in query]
    assert hub.cuts > 0
    # 400 keys of ~2 KB rows under a 64 KB reply: about 25 keys a reply; the first batch halves to that fit (4 cut
    # statements), every later batch is sent at it (16 whole ones), never halved from 400 again
    assert len(keyed) <= 16 + 4 + 2, len(keyed)


async def test_one_key_wider_than_a_reply_is_refused_not_split() -> None:
    """Flips if a cut answer for one key is retried, swallowed, or taken as that key's rows."""
    reader = DatasourceReader(_Client([{"a": 1}], truncated=True), "redshift")  # type: ignore[arg-type]
    with pytest.raises(DatasourceAnswerCut):
        await read_keyed(reader, "t", keys=KeyBatch(("a",), [(1,)], as_list=True), columns=["a"])
