"""DatasourceQueryClient.query_arrow: a datasource read's rows as one Arrow table, whichever form the hub answers in.

A hub that answers Arrow sends the rows as one IPC stream behind the reply's envelope (``L3ARROW1``, the ``l3.query``
rail's framing, read by the one parser of both rails); an older hub ignores the ask and answers JSON, which Arrow's own
reader parses into the same table. Only ``query_arrow`` carries the two fields: every other ask is byte for byte what
it was.
"""

from __future__ import annotations

import base64
import json
import re
import struct
from datetime import timedelta
from typing import Any
from uuid import uuid7

import pyarrow as pa
import pytest

from threetears.core.backends.arrow_reply import ReplyNotArrowError, arrow_schema_text
from threetears.core.keyed import Dialect
from threetears.datasources.arrow_rows import relation_columns_sql, warehouse_arrow_types, warehouse_arrow_types_of
from threetears.datasources.query_client import (
    DATASOURCE_BUSY,
    REPLY_NOT_ARROW,
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


@pytest.mark.parametrize("framed", [False, True], ids=["json", "arrow-envelope"])
async def test_the_hubs_refusal_of_a_value_not_of_its_type_raises_as_the_json_parse_did(framed: bool) -> None:
    """Flips if the hub's own refusal of a column's type (it builds the table from the warehouse's values) reaches a
    caller as anything but ReplyNotArrowError, which a reader answers by reading its types again."""
    refusal = DatasourceQueryResponse(
        success=False,
        error_code=REPLY_NOT_ARROW,
        error_message="column 'race': its values are not of the type asked for",
    )
    nats = _RawNats(_arrow_refusal(refusal) if framed else refusal.model_dump_json().encode())

    with pytest.raises(ReplyNotArrowError, match="column 'race'"):
        await _client(nats).query_arrow("ds", "SELECT race FROM t", types={"race": pa.int64()})


def _arrow_refusal(refusal: DatasourceQueryResponse) -> bytes:
    """a refusal as a hub answering an Arrow ask frames it: the magic, the envelope's length, the envelope."""
    head = refusal.model_dump_json(exclude={"rows"}).encode()
    return b"".join((b"L3ARROW1", struct.pack("!I", len(head)), head))


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
                {"table_schema": "reporting", "table_name": "results", "column_name": name, "data_type": kind}
                for name, kind in (
                    ("race", "character varying"),
                    ("candidate_id", "integer"),
                    ("margin", "numeric"),
                    ("share", "double precision"),
                    ("called_at", "timestamp without time zone"),
                    ("called", "boolean"),
                )
            ]
            return DatasourceQueryResult(rows=rows, row_count=len(rows), truncated=False, correlation_id=uuid7())

    types = await warehouse_arrow_types(
        _Catalogue(),  # type: ignore[arg-type]
        "ds",
        "reporting.results",
        ["race", "candidate_id", "margin", "share", "called_at", "called"],
        dialect=Dialect.POSTGRES,
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
        await warehouse_arrow_types(
            _Catalogue(),  # type: ignore[arg-type]
            "ds",
            "reporting.results",
            ["absent"],
            dialect=Dialect.POSTGRES,
        )


async def test_every_column_s_type_is_read_in_one_ask_and_a_column_of_no_json_form_is_left_out() -> None:
    """Flips if a caller asking every column (columns=None, one catalogue read for all its reads of a relation) is
    refused for a column of a type with no one JSON form it never reads, or is answered another column's type."""

    class _Catalogue:
        asks = 0

        async def query(self, datasource_name: str, query: str, params: list[Any] | None = None) -> Any:
            _Catalogue.asks += 1
            rows = [
                {"table_schema": "reporting", "table_name": "results", "column_name": name, "data_type": kind}
                for name, kind in (("race", "character varying"), ("payload", "super"), ("margin", "numeric"))
            ]
            return DatasourceQueryResult(rows=rows, row_count=len(rows), truncated=False, correlation_id=uuid7())

    types = await warehouse_arrow_types(
        _Catalogue(),  # type: ignore[arg-type]
        "ds",
        "reporting.results",
        None,
        dialect=Dialect.REDSHIFT,
        extra={"row_hash": pa.string()},
    )

    assert types == {"race": pa.string(), "margin": pa.string(), "row_hash": pa.string()}
    assert _Catalogue.asks == 1


class _TwoCatalogues:
    """a Redshift warehouse's two catalogues of one table and the late-binding view over it (``WITH NO SCHEMA
    BINDING``, as dbt's ``+bind: false`` makes every view): ``information_schema.columns`` holds the table's columns
    and none of the view's; ``SVV_COLUMNS`` holds both, spelling a table's ``timetz`` column ``timetz`` (both read
    on the warehouse, 2026-11-19). Each statement is answered from the catalogue it names, for every relation its
    parameters pair (schema, name), and recorded."""

    _TABLE = [
        {"column_name": "race", "data_type": "character varying"},
        {"column_name": "votes", "data_type": "integer"},
        {"column_name": "margin", "data_type": "numeric"},
        {"column_name": "called_at", "data_type": "timestamp with time zone"},
        {"column_name": "polls_close", "data_type": "time with time zone"},
    ]

    def __init__(self) -> None:
        self.asked: list[tuple[str, list[Any] | None]] = []

    async def query(self, datasource_name: str, query: str, params: list[Any] | None = None) -> Any:
        self.asked.append((query, params))
        assert params is not None
        rows: list[dict[str, Any]] = []
        for relation in _asked_relations(query, params):
            if "information_schema.columns" in query:
                held = self._TABLE if relation == ("reporting", "results_20261119") else []
            elif "svv_columns" in query.lower():
                held = {
                    ("reporting", "results_20261119"): [
                        {**row, "data_type": "timetz"} if row["column_name"] == "polls_close" else row
                        for row in self._TABLE
                    ],
                    ("reporting", "results"): self._TABLE,
                }.get(relation, [])
            else:
                raise AssertionError(f"not a catalogue read: {query}")
            rows += [{"table_schema": relation[0], "table_name": relation[1], **row} for row in held]
        return DatasourceQueryResult(rows=rows, row_count=len(rows), truncated=False, correlation_id=uuid7())


#: the one-relation catalogue read on Redshift: a batch of one (:func:`warehouse_arrow_types_of`'s statement)
_ONE_REDSHIFT_RELATION = (
    "SELECT table_schema, table_name, column_name, data_type FROM svv_columns "
    "WHERE table_schema IN ($1) AND table_name IN ($2) ORDER BY table_schema, table_name, ordinal_position"
)

#: the same, on every other engine
_ONE_STANDARD_RELATION = (
    "SELECT table_schema, table_name, column_name, data_type FROM information_schema.columns "
    "WHERE table_schema IN ($1) AND table_name IN ($2) ORDER BY table_schema, table_name, ordinal_position"
)


def _asked_relations(query: str, params: list[Any]) -> list[tuple[str, str]]:
    """every (schema, name) a batch catalogue statement's two IN lists pair, as the catalogue would match them.

    :param query: the statement
    :ptype query: str
    :param params: its parameters: the schemas, then the names
    :ptype params: list[Any]
    :return: the pairs
    :rtype: list[tuple[str, str]]
    """
    schemas = len(re.findall(r"\$\d+", query.split("table_schema IN (", 1)[1].split(")", 1)[0]))
    return [(schema, name) for schema in params[:schemas] for name in params[schemas:]]


_RESULTS_TYPES = {
    "race": pa.string(),
    "votes": pa.int64(),
    "margin": pa.string(),
    "called_at": pa.string(),
    "polls_close": pa.string(),
}


@pytest.mark.parametrize("columns", [None, list(_RESULTS_TYPES)])
async def test_a_redshift_late_binding_view_s_columns_are_read_from_the_catalogue_that_holds_them(
    columns: list[str] | None,
) -> None:
    """Flips if a Redshift relation's columns are read from ``information_schema.columns``, which holds no column of a
    late-binding view: the read of every column answers nothing and a named column is refused, so a datasource
    whose tables became views over dated tables is refused every load."""
    catalogue = _TwoCatalogues()

    types = await warehouse_arrow_types(
        catalogue,  # type: ignore[arg-type]
        "ds",
        "reporting.results",
        columns,
        dialect=Dialect.REDSHIFT,
    )

    assert types == _RESULTS_TYPES
    assert list(types) == list(_RESULTS_TYPES), "in the catalogue's ordinal order"
    assert len(catalogue.asked) == 1
    query, params = catalogue.asked[0]
    assert params == ["reporting", "results"]
    assert query == _ONE_REDSHIFT_RELATION


async def test_a_redshift_table_s_timetz_column_is_text_as_its_view_s_is() -> None:
    """Flips if ``SVV_COLUMNS``' spelling of a table's ``time with time zone`` column (``timetz``) has no JSON form:
    the column is left out of every column's read and refused when named, where its view's is read as text."""
    types = await warehouse_arrow_types(
        _TwoCatalogues(),  # type: ignore[arg-type]
        "ds",
        "reporting.results_20261119",
        ["polls_close"],
        dialect=Dialect.REDSHIFT,
    )

    assert types == {"polls_close": pa.string()}


@pytest.mark.parametrize("dialect", [Dialect.POSTGRES, Dialect.YUGABYTE, Dialect.DUCKDB])
async def test_a_relation_on_another_engine_is_read_from_information_schema_as_before(dialect: Dialect) -> None:
    """Flips if a PostgreSQL, YugabyteDB or DuckDB relation's columns are read other than from
    ``information_schema.columns`` (those engines hold no ``SVV_COLUMNS``), or by another statement than before."""
    catalogue = _TwoCatalogues()

    types = await warehouse_arrow_types(
        catalogue,  # type: ignore[arg-type]
        "ds",
        "reporting.results_20261119",
        None,
        dialect=dialect,
    )

    assert types == _RESULTS_TYPES
    assert catalogue.asked == [(_ONE_STANDARD_RELATION, ["reporting", "results_20261119"])]


@pytest.mark.parametrize(("dialect", "schema"), [(Dialect.DUCKDB, "main"), (Dialect.POSTGRES, "public")])
async def test_a_bare_name_is_asked_in_the_engines_own_default_schema(dialect: Dialect, schema: str) -> None:
    """Flips if a bare relation name is asked in another schema than the one its engine resolves it in (``main`` on
    DuckDB, ``public`` on PostgreSQL): the catalogue would answer no column of a relation that exists."""
    catalogue = _TwoCatalogues()

    await warehouse_arrow_types(
        catalogue,  # type: ignore[arg-type]
        "ds",
        "results_20261119",
        None,
        dialect=dialect,
    )

    assert catalogue.asked == [(_ONE_STANDARD_RELATION, [schema, "results_20261119"])]


async def test_many_relations_types_are_read_in_one_catalogue_statement_each_as_it_would_be_alone() -> None:
    """Flips if a batch of relations is asked one statement a relation (Redshift's leader answers each catalogue
    statement in ~25 s whatever its scope: 46 relations one by one, median 23.9 s each; all 46 in one statement,
    16.2 s; the ots warehouse, 2026-11-27), as pairs ORed (153.3 s for the same 46), if a relation its IN lists pair
    but no one asked is answered (a schema of one relation with the name of another), or if any relation's answer
    differs from the one-relation read's (the
    late-binding view's columns, the table's timetz as text, a relation the catalogue lacks answering ``extra``
    alone). A repeated relation is asked once; no relation is no ask."""
    catalogue = _TwoCatalogues()
    # "elsewhere.results_20261119" asked beside them: the IN lists pair "reporting" with "results_20261119", a
    # relation the catalogue holds that no one asked
    relations = ["reporting.results", "elsewhere.results_20261119", "reporting.absent", "reporting.results"]

    many = await warehouse_arrow_types_of(
        catalogue,  # type: ignore[arg-type]
        "ds",
        relations,
        dialect=Dialect.REDSHIFT,
        extra={"row_hash": pa.string()},
    )

    assert catalogue.asked == [
        (
            "SELECT table_schema, table_name, column_name, data_type FROM svv_columns "
            "WHERE table_schema IN ($1, $2) AND table_name IN ($3, $4, $5) "
            "ORDER BY table_schema, table_name, ordinal_position",
            ["reporting", "elsewhere", "results", "results_20261119", "absent"],
        )
    ]
    assert list(many) == ["reporting.results", "elsewhere.results_20261119", "reporting.absent"]
    assert many["elsewhere.results_20261119"] == {"row_hash": pa.string()}, "no relation asked holds a column there"
    for relation, types in many.items():
        alone = await warehouse_arrow_types(
            _TwoCatalogues(),  # type: ignore[arg-type]
            "ds",
            relation,
            None,
            dialect=Dialect.REDSHIFT,
            extra={"row_hash": pa.string()},
        )
        assert types == alone and list(types) == list(alone), relation
    assert many["reporting.absent"] == {"row_hash": pa.string()}
    assert list(many["reporting.results"]) == [*_RESULTS_TYPES, "row_hash"]

    nothing = _TwoCatalogues()
    assert await warehouse_arrow_types_of(nothing, "ds", [], dialect=Dialect.REDSHIFT) == {}  # type: ignore[arg-type]
    assert nothing.asked == []


async def test_a_batch_past_the_wire_protocols_parameter_count_is_split_there_and_only_there() -> None:
    """Flips if a batch binds more parameters than the wire protocol counts (32767: each distinct schema and name
    once), so the statement is refused, or is split before that limit (one more ~25 s statement on Redshift for
    nothing)."""
    catalogue = _TwoCatalogues()
    # one schema and 32767 names: 32767 parameters for the first 32766 relations, then one statement more
    relations = [f"reporting.t{index}" for index in range(32767)]

    many = await warehouse_arrow_types_of(catalogue, "ds", relations, dialect=Dialect.POSTGRES)  # type: ignore[arg-type]

    assert [len(params or ()) for _, params in catalogue.asked] == [32767, 2]
    assert catalogue.asked[1] == (_ONE_STANDARD_RELATION, ["reporting", "t32766"])
    assert len(many) == len(relations)


@pytest.mark.parametrize("named", ["redshift", "postgres", "unnamed"])
async def test_a_dialect_named_by_a_plain_string_is_refused_before_any_ask(named: str) -> None:
    """Flips if a plain string passes for a dialect (a reader's ``sql_dialect`` for an engine with none, or an engine's
    name): ``"redshift"`` would read another catalogue than ``Dialect.REDSHIFT`` names, or a load would run on a guess.
    Narrowed by ``dialect_of`` first, the caller gets a :class:`Dialect` or its refusal."""
    catalogue = _TwoCatalogues()

    with pytest.raises(AssertionError):
        await warehouse_arrow_types(
            catalogue,  # type: ignore[arg-type]
            "ds",
            "reporting.results",
            None,
            dialect=named,  # type: ignore[arg-type]
        )

    assert catalogue.asked == []


def test_a_relations_columns_statement_written_as_literals_reads_the_dialects_catalogue() -> None:
    """Flips if a caller whose query binds nothing (an operator's check holding one login's ``fetch``) cannot ask the
    catalogue that holds a late-binding view's columns: on Redshift it must read SVV_COLUMNS, elsewhere
    information_schema.columns, with the relation's schema and name written in as quoted literals."""
    assert relation_columns_sql("reporting_prod.report_enr_x", dialect=Dialect.REDSHIFT) == (
        "SELECT column_name, data_type FROM svv_columns "
        "WHERE table_schema = 'reporting_prod' AND table_name = 'report_enr_x' ORDER BY ordinal_position"
    )
    for dialect in (Dialect.POSTGRES, Dialect.YUGABYTE):
        assert relation_columns_sql("t", dialect=dialect) == (
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = 't' ORDER BY ordinal_position"
        )


def test_a_bare_duckdb_name_reads_duckdbs_own_default_schema() -> None:
    """Flips if a bare name on DuckDB is looked up in ``public``: DuckDB resolves it in ``main``, so the lookup would
    find no column of a relation that exists."""
    assert relation_columns_sql("t", dialect=Dialect.DUCKDB) == (
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = 'main' AND table_name = 't' ORDER BY ordinal_position"
    )


def test_a_quote_or_a_placeholder_in_a_name_stays_inside_its_own_literal() -> None:
    """Flips if a name's quote is not doubled, or a ``$2`` inside the schema is replaced by the table's literal: both
    are legal identifier text, and either would break the statement or read another relation."""
    assert relation_columns_sql("s.o'brien", dialect=Dialect.POSTGRES) == (
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = 's' AND table_name = 'o''brien' ORDER BY ordinal_position"
    )
    assert relation_columns_sql("rpt$2x.t", dialect=Dialect.REDSHIFT) == (
        "SELECT column_name, data_type FROM svv_columns "
        "WHERE table_schema = 'rpt$2x' AND table_name = 't' ORDER BY ordinal_position"
    )
    assert relation_columns_sql("$2.t", dialect=Dialect.REDSHIFT) == (
        "SELECT column_name, data_type FROM svv_columns "
        "WHERE table_schema = '$2' AND table_name = 't' ORDER BY ordinal_position"
    )


@pytest.mark.parametrize("relation", ["s\\x.t", "s.t\\", "s\x00.t", "s.t\x00"])
def test_a_name_no_quoted_literal_carries_is_refused_naming_the_relation(relation: str) -> None:
    """Flips if a backslash or a NUL reaches the statement (a backslash escapes with standard_conforming_strings off;
    a NUL ends the string), or the refusal names anything but the relation asked."""
    with pytest.raises(ValueError, match="the relation's (schema|name) holds a backslash or a NUL"):
        relation_columns_sql(relation, dialect=Dialect.REDSHIFT)


@pytest.mark.parametrize("named", ["redshift", "postgres"])
def test_a_literal_columns_statement_refuses_a_plain_string_dialect(named: str) -> None:
    """Flips if a plain string passes for a dialect: only a :class:`Dialect` member names a catalogue."""
    with pytest.raises(AssertionError):
        relation_columns_sql("s.t", dialect=named)  # type: ignore[arg-type]
