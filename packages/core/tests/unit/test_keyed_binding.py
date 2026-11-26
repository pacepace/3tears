"""Arrow mode's key binding (:mod:`threetears.core.keyed.binding`): each key column ONE text written in Arrow's
kernels, read back as the values it holds, binding the rows the plain lists bind, with no key a Python object on the
way in."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import tracemalloc
import uuid
from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime
from typing import Any

import duckdb
import pyarrow as pa
import pytest

from threetears.core.keyed import (
    Dialect,
    Increment,
    InlineDuckDB,
    KeyBatch,
    ThreadedDuckDB,
    delete_keyed,
    read_keyed,
    read_keyed_arrow,
    run_inline,
    update_keyed,
)
from threetears.core.keyed import binding
from threetears.core.keyed.binding import binder_for
from threetears.interfaces.keyed.rows import PLAIN_KEYS, key_chunks, read_keyed_statements

#: a batch whose column arrives as Arrow: a serving path's keys, bound in Arrow mode
_ARROW_KEYS = KeyBatch(("k",), [pa.array(["a"])])

#: the fewest keys the binder makes distinct in its worker process (``binding._INLINE_DISTINCT_KEYS``), read by name
_INLINE_DISTINCT_KEYS: int = getattr(binding, "_INLINE_DISTINCT_KEYS")

#: keys a real-size read names: a page of a night's changed rows
_REAL_KEYS = 50_000
_STATES, _GEOS, _IDS = 50, 200, 30


#: a process refusing pyarrow and NumPy, binding a batch of plain columns: prints ``plain`` once bound as plain lists
_NO_ARROW_BINDS_PLAIN = """
import importlib.abc, sys
class _Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("pyarrow", "numpy"):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None
sys.meta_path.insert(0, _Refuse())
from threetears.core.keyed import KeyBatch
from threetears.core.keyed.binding import binder_for
from threetears.interfaces.keyed.rows import PLAIN_KEYS
assert binder_for(object(), KeyBatch(("k",), [["a"]])) is PLAIN_KEYS
print("plain")
"""


class _DuckListReader:
    """a reader naming DuckDB: its binder writes a key column as the array's literal in DuckDB's list form."""

    sql_dialect = Dialect.DUCKDB


def _arrow_literal(values: Any, brackets: bool = False) -> str:
    """an Arrow column's literal as the front door binds it: a reader naming DuckDB's (``[...]``), or PostgreSQL's
    (``{...}``) for a column its split text cannot carry (one holding every delimiter it tries)."""
    bound, kind = binder_for(_DuckListReader() if brackets else object(), _ARROW_KEYS).array(values, "text", False)
    assert kind == "literal", (kind, bound)
    return str(bound)


def _duck_list(conn: duckdb.DuckDBPyConnection, literal: str, cast: str) -> list[Any]:
    """the values DuckDB reads a list literal as."""
    return list(conn.execute(f"SELECT $1::text::{cast}[]", [literal]).fetchone()[0])  # type: ignore[index]


class TestTheLiteral:
    def test_a_literal_reads_back_as_the_values_it_was_written_from(self) -> None:
        """Flips if a value holding a quote, a backslash, a comma, a brace, the word NULL or nothing is read back as
        anything but itself, or a null as anything but a null; the PostgreSQL form is the same body in braces."""
        # the last value holds every code point a split text is joined by, so PostgreSQL's binder writes the literal
        every_delimiter = "".join(chr(code) for code in range(24, 32))
        values = ['a"b', "c\\d", "e,f", "{g}", "NULL", "", "Peña ✓ 選挙", None, " lead and trail ", every_delimiter]
        conn = duckdb.connect()
        assert _duck_list(conn, _arrow_literal(pa.array(values), brackets=True), "text") == values
        assert _arrow_literal(pa.array(values)) == "{" + _arrow_literal(pa.array(values), brackets=True)[1:-1] + "}"
        assert _arrow_literal(pa.array([], pa.string()), brackets=True) == "[]"

    def test_a_uuid_a_date_a_time_and_a_number_are_written_as_their_types_read_them(self) -> None:
        """Flips if a UUID (Arrow's UUID type, sliced, a null among them) is not written as its 32 digits, or a date,
        a zoned time or an integer is not read back by its column's type as itself."""
        ids = [uuid.uuid4() for _ in range(5)]
        sliced = pa.array([ids[0], None, *ids[1:]], pa.uuid()).slice(1)
        conn = duckdb.connect()
        assert _duck_list(conn, _arrow_literal(sliced, brackets=True), "uuid") == [None, *ids[1:]]
        days = [date(2026, 11, 3), date(2024, 2, 29)]
        assert _duck_list(conn, _arrow_literal(pa.array(days), brackets=True), "date") == days
        stamps = [datetime(2026, 11, 3, 22, 30, 4, 123456, tzinfo=UTC)]
        read = _duck_list(conn, _arrow_literal(pa.array(stamps), brackets=True), "timestamptz")
        assert [stamp.astimezone(UTC) for stamp in read] == stamps
        assert _duck_list(conn, _arrow_literal(pa.array([-3, 0, 2**62]), brackets=True), "bigint") == [-3, 0, 2**62]


class _Sent:
    """a reader keeping what it is sent, answering no rows: asyncpg's (``native``) or the rail's."""

    def __init__(self, native: bool) -> None:
        self.binds_native_arrays = native
        self.sent: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *params: Any) -> list[Any]:
        self.sent.append((query, params))
        return []

    async def execute(self, query: str, *params: Any) -> str:
        self.sent.append((query, params))
        return "UPDATE 0"


class TestArrowModeBindsOneTextAColumn:
    @pytest.mark.parametrize("native", [True, False], ids=["asyncpg", "rail"])
    async def test_each_key_column_is_one_text_split_back_as_a_whole(self, native: bool) -> None:
        """Flips if Arrow mode binds a key column as anything but one text (a list, a value a key), or the statement
        makes its array other than by an immutable split cast as a whole to the column's type (a custom plan's
        constant): one column ``= ANY``, several joined as rows, a lead shared bounded to its values (made once a
        statement), a null-safe batch, and a write."""
        sent = _Sent(native)
        ids = [uuid.uuid4() for _ in range(3)]
        await read_keyed(sent, "t", keys=KeyBatch(("id",), [pa.array(ids, pa.uuid())], ("uuid",)), columns=["v"])
        sql, params = sent.sent[-1]
        assert 't."id" = ANY(string_to_array($1::text, chr(44))::uuid[])' in sql, sql
        assert params == (",".join(map(str, ids)),)
        scattered = [pa.array([f"S,{i}" for i in range(10)]), pa.array(range(10), pa.int64())]
        await read_keyed(sent, "t", keys=KeyBatch(("a", "b"), scattered, ("text", "bigint")), columns=["v"])
        sql, params = sent.sent[-1]
        assert (
            "IN (SELECT unnest(string_to_array($1::text, chr(31))) AS k0, "
            "unnest(string_to_array($2::text, chr(44))::bigint[]) AS k1)"
        ) in sql, sql
        assert "= ANY(" not in sql and params[0] == "\x1f".join(f"S,{i}" for i in range(10))
        lead = [pa.array(["S1", "S1"]), pa.array([1, 2], pa.int64())]
        await read_keyed(sent, "t", keys=KeyBatch(("a", "b"), lead, ("text", "bigint")), columns=["v"])
        sql, params = sent.sent[-1]
        # the bounds beside the keys made once a statement (an InitPlan), never a row at a time in a generic plan
        assert (
            't."a" = ANY((SELECT string_to_array($3::text, chr(31)))::text[]) '
            'AND t."b" = ANY((SELECT string_to_array($4::text, chr(44))::bigint[])::bigint[])'
        ) in sql, sql
        assert params[2:] == ("S1", "1,2")
        # a null-safe batch's NULL bucket is pinned IS NULL (null_split), so its split text carries no NULL
        nulls = KeyBatch(("scope",), [pa.array(["a", None])], null_safe=True)
        await read_keyed(sent, "t", keys=nulls, columns=["v"])
        sql, params = sent.sent[-1]
        assert sql == (
            'SELECT t."v" FROM "t" AS t WHERE t."scope" = ANY(string_to_array($1::text, chr(31))) '
            'UNION ALL SELECT t."v" FROM "t" AS t WHERE t."scope" IS NULL'
        ), sql
        assert params == ("a",)
        # a NULL in a batch's text is written as a code point of its own, split back as NULL
        await read_keyed(sent, "t", keys=KeyBatch(("scope",), [pa.array(["a", None])]), columns=["v"])
        sql, params = sent.sent[-1]
        assert "string_to_array($1::text, chr(31), chr(30))" in sql and params == ("a\x1f\x1e",)
        await update_keyed(
            sent, "t", values={"v": Increment(1)}, keys=KeyBatch(("id",), [pa.array(ids, pa.uuid())], ("uuid",))
        )
        sql, params = sent.sent[-1]
        assert '"id" = ANY(string_to_array($2::text, chr(44))::uuid[])' in sql and isinstance(params[1], str), sql

    def test_a_text_column_no_delimiter_fits_is_bound_as_its_literal(self) -> None:
        """Flips if a text column holding every candidate delimiter, or one empty text alone (which a split reads as
        no value), is bound other than as its array literal."""
        binder = binder_for(object(), _ARROW_KEYS)
        every = "".join(chr(code) for code in range(24, 32))
        bound, kind = binder.array(pa.array([every, "x"]), "text", False)
        assert kind == "literal" and bound.startswith("{")
        assert binder.array(pa.array([""]), "text", False) == ('{""}', "literal")

    async def test_an_in_process_duckdb_is_bound_the_arrow_array_itself(self) -> None:
        """Flips if a read on an in-process DuckDB is sent a PostgreSQL literal (it parses none) or Python values
        rather than the Arrow array of its texts (run_bound sets it as a DuckDB variable)."""
        conn = duckdb.connect()
        binder = binder_for(InlineDuckDB(conn), _ARROW_KEYS)
        bound, kind = binder.array(pa.array(["a", "b"]), "text", False)
        assert kind == "text" and isinstance(bound, pa.Array)
        assert binder_for(ThreadedDuckDB(conn), _ARROW_KEYS) is binder

    def test_the_binder_follows_the_connection_the_columns_and_whether_arrow_imports(self) -> None:
        """Flips if a reader naming DuckDB is sent PostgreSQL's literal, a batch of plain columns (a key list program
        logic holds) is made Arrow where pyarrow imports (Pace's option A: it read 2,000 UUIDs at 1.3x the native bind
        so), one arriving as Arrow is not, or a process with no Arrow binds other than plain lists."""

        class _DuckReader:
            sql_dialect = Dialect.DUCKDB

        bound, kind = binder_for(_DuckReader(), _ARROW_KEYS).array(pa.array(["a"]), "text", False)
        assert (bound, kind) == ('["a"]', "literal")
        plain = KeyBatch(("id",), [[uuid.uuid4(), uuid.uuid4()]], ("uuid",))
        assert binder_for(object(), plain) is PLAIN_KEYS and binder_for(object(), None) is PLAIN_KEYS
        mixed = KeyBatch(("a", "b"), [pa.array(["x"]), [1]], ("text", "bigint"))
        assert binder_for(object(), mixed) is binder_for(object(), _ARROW_KEYS) is not PLAIN_KEYS
        # a process where pyarrow does not import binds plain lists
        done = subprocess.run(
            [sys.executable, "-c", _NO_ARROW_BINDS_PLAIN], capture_output=True, text=True, check=False
        )
        assert done.returncode == 0 and done.stdout.strip() == "plain", done.stderr

    def test_the_plain_binder_refuses_an_arrow_column(self) -> None:
        """Flips if the pure builders' plain binder takes an Arrow column value by value (a Python object a key)."""
        with pytest.raises(TypeError, match="binds through threetears.core.keyed"):
            read_keyed_statements("t", keys=KeyBatch(("a",), [pa.array(["k"])]), columns=["v"], binder=PLAIN_KEYS)


@pytest.fixture
def real() -> Iterator[duckdb.DuckDBPyConnection]:
    """the results table at real size (300,000 rows keyed ``(state, geo, id)``) and 200,000 rows keyed by a UUID."""
    conn = duckdb.connect()
    conn.execute(
        f"""CREATE TABLE r AS SELECT 'S' || lpad(s::text, 2, '0') AS state, lpad(g::text, 3, '0') AS geo,
        i::BIGINT AS id, (s * g * i)::BIGINT AS v
        FROM range(1, {_STATES + 1}) a(s), range(1, {_GEOS + 1}) b(g), range(1, {_IDS + 1}) c(i)"""
    )
    conn.execute("CREATE TABLE u AS SELECT uuid() AS id, g::BIGINT AS v FROM range(200000) t(g)")
    yield conn
    conn.close()


def _picks(conn: duckdb.DuckDBPyConnection) -> tuple[list[list[Any]], list[uuid.UUID]]:
    """real-size keys, every tenth named twice: composite as columns, and UUIDs."""
    rows = conn.execute(f"SELECT state, geo, id FROM r USING SAMPLE {_REAL_KEYS} ROWS (reservoir, 7)").fetchall()
    rows += rows[::10]
    ids = [row[0] for row in conn.execute(f"SELECT id FROM u USING SAMPLE {_REAL_KEYS} ROWS (reservoir, 7)").fetchall()]
    return [list(column) for column in zip(*rows, strict=True)], ids + ids[::10]


def _sorted(rows: Sequence[Any]) -> list[tuple[Any, ...]]:
    return sorted(tuple(row.values()) for row in rows)


async def _read_both(
    conn: duckdb.DuckDBPyConnection, composite: list[list[Any]], ids: list[uuid.UUID], arrow: bool
) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]], int]:
    reader = ThreadedDuckDB(conn)
    casts = ("text", "text", "bigint")
    columns = [pa.array(composite[0]), pa.array(composite[1]), pa.array(composite[2], pa.int64())]
    as_arrow = [*columns] if arrow else composite
    rows = await read_keyed(reader, "r", keys=KeyBatch(("state", "geo", "id"), as_arrow, casts), columns=["v", "id"])
    keyed = [pa.array(ids, pa.uuid())] if arrow else [ids]
    by_id = await read_keyed(reader, "u", keys=KeyBatch(("id",), keyed, ("uuid",)), columns=["id", "v"])
    tables = await read_keyed_arrow(
        reader, "r", keys=KeyBatch(("state", "geo", "id"), as_arrow, casts), types={"v": pa.int64()}
    )
    # 50,000 rows a side sorted as tuples, on a thread: on the loop, with the writes below, it held it past the
    # loop-stall guard's limit (threetears.core.testing.loop_stall)
    return await asyncio.to_thread(lambda: (_sorted(rows), _sorted(by_id), sum(table.num_rows for table in tables)))


class TestArrowModeReadsWhatPlainModeReads:
    async def test_real_size_reads_and_writes_name_the_same_rows(self, real: duckdb.DuckDBPyConnection) -> None:
        """Flips if Arrow mode reads, or writes, other rows than the plain lists name for 50,000 keys (a repeat among
        them, composite and UUID-keyed), on DuckDB in process. The keys are picked, and the InlineDuckDB writes run
        (synchronous, for a worker process's own connection), on a thread: the loop only awaits."""
        composite, ids = await asyncio.to_thread(_picks, real)
        arrow_read = await _read_both(real, composite, ids, arrow=True)
        # plain lists bind as plain lists whether or not pyarrow imports (binder_for)
        plain_read = await _read_both(real, composite, ids, arrow=False)
        assert arrow_read == plain_read
        assert len(arrow_read[0]) == len(arrow_read[1]) == arrow_read[2] == _REAL_KEYS
        reader = InlineDuckDB(real)
        columns = [pa.array(composite[0]), pa.array(composite[1]), pa.array(composite[2], pa.int64())]
        keys = KeyBatch(("state", "geo", "id"), columns, ("text", "text", "bigint"))
        # run_inline drives a helper with no event loop, as a worker thread does: here on one, not on the test's loop
        written = await asyncio.to_thread(
            run_inline, update_keyed(reader, "r", values={"v": Increment(1)}, keys=keys, batch=10_000)
        )
        assert written.count == _REAL_KEYS
        deleted = await asyncio.to_thread(run_inline, delete_keyed(reader, "r", keys=keys, batch=10_000))
        assert deleted.count == _REAL_KEYS
        assert real.execute("SELECT count(*) FROM r").fetchone() == (_STATES * _GEOS * _IDS - _REAL_KEYS,)


class TestNoKeyIsAPythonObjectOnTheWayIn:
    def test_real_size_binding_allocates_the_literals_alone(self, real: duckdb.DuckDBPyConnection) -> None:
        """Flips if Arrow mode makes a Python object a key on the way in (a key's tuple, a value's text, a list of
        them): what it allocates is bounded by the literals it binds, where one Python text a key is several times
        that. 50,000 keys of UUIDs and of a composite key, every statement rendered and held."""
        composite, ids = _picks(real)
        cases = [
            (KeyBatch(("id",), [pa.array(ids, pa.uuid())], ("uuid",)), 1),
            (
                KeyBatch(
                    ("state", "geo", "id"),
                    [pa.array(composite[0]), pa.array(composite[1]), pa.array(composite[2], pa.int64())],
                    ("text", "text", "bigint"),
                ),
                3,
            ),
        ]
        literal_binder = binder_for(object(), _ARROW_KEYS)
        for keys, width in cases:
            read_keyed_statements("t", keys=keys, columns=["v"], batch=10_000, binder=literal_binder)  # warm
            tracemalloc.start()
            try:
                statements = read_keyed_statements("t", keys=keys, columns=["v"], batch=10_000, binder=literal_binder)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            binds = [param for _, params in statements for param in params[:width]]
            assert all(isinstance(bind, str) for bind in binds)
            literal_bytes = sum(len(bind.encode()) for bind in binds)
            bound = 1.25 * literal_bytes + 256 * 1024
            assert peak <= bound, f"{width} column(s): {peak} bytes allocated binding {literal_bytes} of literals"
            # the bound's teeth: the same keys as one Python value each (what a list bind makes) are over it
            tracemalloc.start()
            try:
                held = [column.to_pylist() for column in keys.keys]
                _, per_key = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            assert len(held[0]) == len(keys) and per_key > bound


def _repeating_batches(count: int) -> list[KeyBatch]:
    """batches of ``count`` keys, each named twice (a text column, a UUID column, a composite key), first-named order
    not sorted."""
    half = count // 2
    order = [(i * 7_919) % half for i in range(half)]
    named = order + order[::-1]
    ids = [uuid.UUID(int=at + 1) for at in range(half)]
    return [
        KeyBatch(("k",), [pa.chunked_array([pa.array([f"g{at}" for at in named])])]),
        KeyBatch(("id",), [pa.array([ids[at].bytes for at in named], pa.uuid())], ("uuid",)),
        KeyBatch(
            ("state", "geo"),
            [pa.array([f"S{at % 50}" for at in named]), pa.array([at for at in named], pa.int64())],
            ("text", "bigint"),
        ),
    ]


class TestALargeBatchIsMadeDistinctOffTheLoop:
    """A copy's page of 600,000 keys made distinct on the loop held it 0.45 s (``_unique`` under
    ``read_keyed_arrow``, the loop-stall guard on brooks-5950x and brooks-nas, 2026-11-25): a batch of
    ``_INLINE_DISTINCT_KEYS`` keys or more is made distinct in a worker process."""

    async def test_a_real_size_batch_is_never_made_distinct_on_the_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Flips if a keyed read of 50,000 Arrow keys finds their distinct keys in the serving process (the old code:
        ``binder.distinct`` on the loop, inside ``key_chunks``); a small batch still does, on the loop."""
        assert _REAL_KEYS >= _INLINE_DISTINCT_KEYS
        in_process: list[int] = []
        # the binder's one hash kernel a column, watched by name: a call in the worker process is not seen
        name = "_unique"
        unique = getattr(binding, name)

        def recorded(values: Any) -> Any:
            in_process.append(len(values))
            return unique(values)

        monkeypatch.setattr(binding, name, recorded)
        half = _REAL_KEYS // 2
        keys = KeyBatch(("k",), [pa.array([f"g{i % half}" for i in range(_REAL_KEYS)])])
        sent = _Sent(native=False)
        await read_keyed(sent, "t", keys=keys, columns=["v"], batch=half // 2)
        assert in_process == [], "a real-size batch was made distinct in the serving process"
        assert len(sent.sent) == 2, "its distinct keys, half the batch, in statements of a quarter"
        await read_keyed(sent, "t", keys=KeyBatch(("k",), [pa.array(["a", "b", "a"])]), columns=["v"])
        assert in_process == [3], "a small batch is made distinct here, where the worker's round trip costs more"

    async def test_the_worker_finds_what_the_loop_finds(self) -> None:
        """Flips if the worker's distinct keys differ from the binder's own (a value, a type, their first-named
        order, a chunk's bounds): a text, a UUID and a composite key, each named twice."""
        for keys in _repeating_batches(2 * _INLINE_DISTINCT_KEYS):
            binder = binder_for(object(), keys)
            off_loop = await binding.key_chunks_off_loop(keys, 3_000, binder)
            here = await asyncio.to_thread(key_chunks, keys, 3_000, binder)
            assert len(off_loop) == len(here) == -(-_INLINE_DISTINCT_KEYS // 3_000)
            for theirs, ours in zip(off_loop, here, strict=True):
                for a, b in zip(theirs, ours, strict=True):
                    assert isinstance(a, pa.Array) and isinstance(b, pa.Array) and a.type == b.type and a.equals(b)
