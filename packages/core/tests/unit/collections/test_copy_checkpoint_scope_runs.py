"""a checkpoint write cuts a table's rows by scope into its parts as the cut before 2026-11-26 did, through the write
itself (:meth:`~threetears.core.collections.copy_checkpoint.CopyCheckpoints.write` at a table's cold start, which
reads the copy whole and cuts each batch by its scope column).

The contract this pins:

- each value of the scope column is one part holding exactly the rows with it, the parts in the order the values first
  appear in the copy's rows, each part's rows in the copy's order: for rows grouped by scope and not, nulls grouped and
  scattered (a text column and an integer one), one scope, all nulls, one row and one null row;
- a table with no rows writes no parts.

The cut's cost, which finding the runs on the column itself lowered (12.3 -> ~5.8 ms a 1M-row write of 50 scopes, one
thread, measured 2026-11-26), is held by the whole write's real-size budget:
``packages/core/tests/integration/test_copy_follower_whole_reads_live.py::test_checkpoint_whole_write_speed_at_real_size``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import ExitStack, asynccontextmanager
from typing import Any

import pytest
from sqlalchemy import BigInteger, Column as SaColumn, MetaData, String, Table
from threetears.core.testing.l1_tables import L1_EVICTION_INFO, TEST_L1_TABLE

pa = pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")
ipc = pytest.importorskip("pyarrow.ipc")

from threetears.core.cache.duckdb import STAGE_PREFIX, DuckDBBackend, StagedChange  # noqa: E402
from threetears.core.collections.copy_checkpoint import CopyCheckpoints  # noqa: E402
from threetears.core.collections.copy_follower import FollowedTable  # noqa: E402
from threetears.core.collections.copy_reads import PinnedCopy  # noqa: E402
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column  # noqa: E402

#: a table scoped by a text column, and one scoped by an integer column; each scope column may hold nulls
_BY_STATE = "by_state"
_BY_CODE = "by_code"
_KEY = ("geo_id",)
_SCOPES = {_BY_STATE: ("state", "state:"), _BY_CODE: ("code", "code:")}


def _metadata() -> MetaData:
    metadata = MetaData()
    Table(
        _BY_STATE,
        metadata,
        SaColumn("geo_id", BigInteger, primary_key=True),
        SaColumn("state", String(8), nullable=True),
        info={L1_EVICTION_INFO: TEST_L1_TABLE},
    )
    Table(
        _BY_CODE,
        metadata,
        SaColumn("geo_id", BigInteger, primary_key=True),
        SaColumn("code", BigInteger, nullable=True),
        info={L1_EVICTION_INFO: TEST_L1_TABLE},
    )
    return metadata


def _tables() -> list[FollowedTable]:
    return [
        FollowedTable(
            name=_BY_STATE,
            key=_KEY,
            columns=(Column(name="geo_id", column_type=BIGINT_TYPE), Column(name="state", column_type=STRING_TYPE)),
            scope=_SCOPES[_BY_STATE],
        ),
        FollowedTable(
            name=_BY_CODE,
            key=_KEY,
            columns=(Column(name="geo_id", column_type=BIGINT_TYPE), Column(name="code", column_type=BIGINT_TYPE)),
            scope=_SCOPES[_BY_CODE],
        ),
    ]


class _Store:
    """an Object Store in memory, its room unbounded."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def require_room(self, objects: Sequence[tuple[str, int]]) -> int:
        return sum(size for _, size in objects)

    async def put(self, name: str, data: bytes) -> None:
        self.objects[name] = data

    async def get(self, name: str) -> bytes:
        return self.objects[name]

    async def list_names(self, *, prefix: str = "") -> list[str]:
        return [n for n in self.objects if n.startswith(prefix)]

    async def retire(self, names: list[str]) -> None:
        for name in names:
            self.objects.pop(name, None)


class _Pointers:
    """a pointer bucket in memory: create-if-absent and compare-and-set by revision."""

    def __init__(self) -> None:
        self.entries: dict[str, tuple[bytes, int]] = {}
        self.revision = 0

    async def get_entry(self, *, key: str) -> tuple[bytes, int] | None:
        return self.entries.get(key)

    async def create(self, *, key: str, value: bytes, ttl: Any = None) -> int | None:
        if key in self.entries:
            return None
        self.revision += 1
        self.entries[key] = (value, self.revision)
        return self.revision

    async def update(self, *, key: str, value: bytes, revision: int, ttl: Any = None) -> int | None:
        held = self.entries.get(key)
        if held is None or held[1] != revision:
            return None
        self.revision += 1
        self.entries[key] = (value, self.revision)
        return self.revision

    async def delete(self, *, key: str, revision: int | None = None, marker_ttl: Any = None) -> bool:
        return self.entries.pop(key, None) is not None

    async def list_keys(self, *, prefix: str = "") -> list[str]:
        return [k for k in self.entries if k.startswith(prefix)]


class _Pin:
    """one applied state of the copy, as a checkpoint write pins it (``copy_reads.PinnedCopy``): its jobs run on a
    snapshot cursor of the test's DuckDB, on a thread. A test fixture: a follower's pin is a leased version whose jobs
    run in a query worker."""

    def __init__(self, stack: ExitStack, cursor: Any, mark: int | None) -> None:
        self._stack = stack
        self._cursor = cursor
        self._mark = mark

    @classmethod
    def opened(cls, backend: DuckDBBackend) -> _Pin:
        stack = ExitStack()
        cursor = stack.enter_context(backend.read_snapshot())
        return cls(stack, cursor, backend.staged_mark(cursor))

    @property
    def mark(self) -> int | None:
        return self._mark

    def _run(self, job: Any, args: Sequence[Any], tables: Mapping[str, Any] | None) -> Any:
        named = tuple(() if tables is None else tables.items())
        for name, table in named:
            self._cursor.register(name, table)
        try:
            answer = job(self._cursor, args)
            if isinstance(answer, pa.Table):
                return answer
            return answer.read_all() if isinstance(answer, pa.RecordBatchReader) else answer.fetch_arrow_table()
        finally:
            for name, _ in named:
                self._cursor.unregister(name)

    async def run_job(self, job: Any, args: Sequence[Any] = (), *, tables: Mapping[str, Any] | None = None) -> Any:
        return await asyncio.to_thread(self._run, job, args, tables)

    async def fingerprint(
        self,
        job: Any,
        args: Sequence[Any],
        kinds: tuple[tuple[str, str], ...],
        *,
        tables: Mapping[str, Any] | None = None,
    ) -> str:
        raise AssertionError("a checkpoint write takes no fingerprint")

    async def close(self) -> None:
        await asyncio.to_thread(self._stack.close)


def _filled(table: str, scopes: list[Any]) -> DuckDBBackend:
    """a copy whose ``table`` holds one row a scope value, in that order (row ``i``'s geo_id is ``i``), applied with
    mark 1; the other table empty."""
    column = _SCOPES[table][0]
    backend = DuckDBBackend()
    backend.initialize(_metadata())
    rows = pa.table(
        {
            "geo_id": pa.array(range(len(scopes)), pa.int64()),
            column: pa.array(scopes, pa.string() if table == _BY_STATE else pa.int64()),
        }
    )
    stage = f"{STAGE_PREFIX}{table}__rows"
    backend.stage_rows(stage, table, rows, _KEY)
    backend.apply_staged([StagedChange(table=table, key=_KEY, rows=stage, whole=True, whole_kind="cold_start")], mark=1)
    return backend


async def _written_parts(table: str, scopes: list[Any]) -> list[tuple[str | None, list[Any], list[int]]]:
    """the parts a cold-start checkpoint write gives ``table`` of a copy :func:`_filled` with ``scopes``: each part's
    scope, its scope column's values and its geo_ids, in the order written."""
    backend = await asyncio.to_thread(_filled, table, scopes)
    marks: dict[str, str | None] = {_BY_STATE: "m1", _BY_CODE: "m1"}

    @asynccontextmanager
    async def read() -> AsyncIterator[tuple[PinnedCopy, Any, Mapping[str, str | None]]]:
        state = await asyncio.to_thread(_Pin.opened, backend)
        try:
            yield state, 1, marks
        finally:
            await state.close()

    store = _Store()
    checkpoints = CopyCheckpoints(
        name="scopes",
        tables=_tables(),
        backend=backend,
        store=store,
        pointers=_Pointers(),
        retire=store.retire,
        owner="a",
        part_rows=1_000,
    )
    written = await checkpoints.write(read)
    assert written is not None
    column = _SCOPES[table][0]
    found = []
    for name, scope, held in written.tables[table].parts:
        part = ipc.open_stream(store.objects[name]).read_all()
        assert part.num_rows == held, (name, part.num_rows, held)
        found.append((scope, part.column(column).to_pylist(), part.column("geo_id").to_pylist()))
    return found


def _expected(table: str, scopes: list[Any]) -> list[tuple[str | None, list[Any], list[int]]]:
    """what the cut before 2026-11-26 wrote: one part a value, in the order the values first appear, each holding the
    rows with that value in the copy's order."""
    prefix = _SCOPES[table][1]
    values: list[Any] = []
    for value in scopes:
        if value not in values:
            values.append(value)
    return [
        (f"{prefix}{value}", [value] * scopes.count(value), [at for at, held in enumerate(scopes) if held == value])
        for value in values
    ]


_GROUPED = [
    pytest.param(_BY_STATE, ["S00", "S00", "S00", "S01", "S01", "S02", "S02", "S02", "S02"], id="grouped"),
    pytest.param(_BY_STATE, [None, None, "a", "b", "b"], id="nulls_grouped_first"),
    pytest.param(_BY_STATE, ["a", "a", None, None, None, "b"], id="nulls_grouped_between"),
    pytest.param(_BY_STATE, ["a", "a", None], id="nulls_grouped_last"),
    pytest.param(_BY_CODE, [7, 7, None, None, 3], id="int_nulls_grouped"),
    pytest.param(_BY_STATE, ["only"] * 5, id="single_scope"),
    pytest.param(_BY_STATE, [None] * 4, id="all_null"),
    pytest.param(_BY_STATE, ["one"], id="one_row"),
    pytest.param(_BY_STATE, [None], id="one_null_row"),
]

_NOT_GROUPED = [
    pytest.param(_BY_STATE, ["S01", "S00", "S01", "S02", "S00"], id="ungrouped"),
    pytest.param(_BY_STATE, [None, "a", None, "b"], id="nulls_scattered"),
    pytest.param(_BY_STATE, ["a", None, "a", None, "a"], id="nulls_between_repeats"),
    pytest.param(_BY_CODE, [3, None, 3, 1, None], id="int_nulls_scattered"),
    pytest.param(_BY_STATE, ["a", "b", "a"], id="value_returns"),
]


@pytest.mark.parametrize(("table", "scopes"), _GROUPED)
async def test_rows_grouped_by_scope_are_written_one_part_a_scope_as_the_reference_cut_them(
    table: str, scopes: list[Any]
) -> None:
    """Flips if a write of rows whose scopes run together splits, merges, reorders or loses a scope's rows."""
    found = await _written_parts(table, scopes)
    assert found == _expected(table, scopes)
    assert sum(len(ids) for _, _, ids in found) == len(scopes)


@pytest.mark.parametrize(("table", "scopes"), _NOT_GROUPED)
async def test_rows_not_grouped_by_scope_are_written_one_part_a_scope_in_the_copys_order(
    table: str, scopes: list[Any]
) -> None:
    """Flips if a write of rows whose scopes interleave puts two scopes in one part, a scope in two, or a scope's rows
    out of the copy's order (the sort by scope must be stable)."""
    found = await _written_parts(table, scopes)
    assert found == _expected(table, scopes)
    for _, _, ids in found:
        assert ids == sorted(ids)


async def test_a_table_with_no_rows_writes_no_parts() -> None:
    assert await _written_parts(_BY_STATE, []) == []
