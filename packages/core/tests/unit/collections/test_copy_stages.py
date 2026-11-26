"""a copy follower runs on the copy's worker backend: its stages are a change's pages, applied in the apply worker.

``collections.copy_stages.SharedCopyStages`` is a ``CopyFollower``'s backend over ``shared_copy.SharedCopyBackend``
(``design-copy-worker-backend.md``: the step the 3T-2 changelog left for later). The contract this pins:

- the staging tables' semantics hold as pages: a change's keys leave, then its rows go in; a key named again takes its
  earlier row out of the rows stage; a key taken out of a change is left as the copy holds it; a whole change replaces
  its table; a change whose stage is gone raises ``LookupError`` and one that fails keeps its stages;
- a follower on it applies keyed changes and deletes, a whole-table read, a checkpoint written and a start from it,
  with every read through ``CopyQueries``, and opens no DuckDB connection in the serving process on any thread
  (``test_copy_follower_reads``' spy), L3's own fake aside;
- a reload prepared from reload stages stages in the reload worker, holds no pages in its stages, and publishes only
  when its swap applies it, with the swap's mark and with a change given to its table meanwhile applied over it
  (decision 52);
- with hooks that derive whole, a start that read L3 whole defers the hooks, and the first pass labelled with a
  stamp settles the copy whole (``SharedCopyBackend.settle_whole``) through the stages; a failed whole settle fails
  the pass and the next one settles; changes after it derive as they apply;
- the whole settle is a step of the apply lane: a tier merge it makes due has its turn with no change after it, and a
  change given while it waits its turn applies after it.
"""

from __future__ import annotations

import asyncio
import gc
import os
import sys
import threading
import weakref
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from types import FrameType
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")

from threetears.core.cache.duckdb import STAGE_PREFIX, StagedChange  # noqa: E402
from threetears.core.cache.duckdb_binds import run_bound  # noqa: E402
from threetears.core.collections import CollectionRegistry  # noqa: E402
from threetears.core.collections.complete_copy import Unsettled  # noqa: E402
from threetears.core.collections.copy_checkpoint import CopyCheckpoints  # noqa: E402
from threetears.core.collections.copy_follower import CopyFollower, FollowedTable, is_reload_stage  # noqa: E402
from threetears.core.collections.epoch_change import ChangedRow, EpochChange, EpochWalk, walk_by_count  # noqa: E402
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column  # noqa: E402
from threetears.core.collections.copy_parts import PartRule  # noqa: E402
from threetears.core.collections.shared_copy import MARK_TABLE, ROWS, ChangePage  # noqa: E402
from threetears.core.testing.published_copy import SharedCopy  # noqa: E402
from packages.core.tests.copy_workers_support import COPY_WORKERS  # noqa: E402
from packages.core.tests.unit.collections import copy_hooks, copy_query_jobs  # noqa: E402

_TABLE = "votes"
_KEY = ("state", "geo_id")
_COLUMNS = (
    Column(name="state", column_type=STRING_TYPE),
    Column(name="geo_id", column_type=BIGINT_TYPE),
    Column(name="votes", column_type=BIGINT_TYPE),
)


def reload_stage(table: str) -> str:
    """the stage a running copy's reload of ``table`` stages in, as the follower names it: a backend tells it by
    ``is_reload_stage`` alone, which each test asserts of it."""
    return f"{STAGE_PREFIX}{table}__full"


_FOLLOWED = FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS, scope=("state", "state:"))

_OPENED: list[Callable[[], None]] = []


@pytest.fixture(autouse=True)
def _close_what_the_test_opened() -> Iterator[None]:
    yield
    while _OPENED:
        _OPENED.pop()()


def _rows(*rows: tuple[str, int, int]) -> Any:
    return pa.table(
        {
            "state": pa.array([r[0] for r in rows], pa.string()),
            "geo_id": pa.array([r[1] for r in rows], pa.int64()),
            "votes": pa.array([r[2] for r in rows], pa.int64()),
        }
    )


def _keys(*keys: tuple[str, int]) -> Any:
    """keys as a change record names them: text."""
    return pa.table({"state": [k[0] for k in keys], "geo_id": [str(k[1]) for k in keys]})


async def _copy() -> SharedCopy:
    return await SharedCopy([_FOLLOWED]).started()


async def _staged(copy: SharedCopy, call: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """a staging call as the follower makes it: on a thread."""
    return await asyncio.to_thread(call, *args, **kwargs)


# --- the stages as pages ---------------------------------------------------------------------------------------------


async def test_a_changes_keys_leave_then_its_rows_go_in_and_a_key_named_again_loses_its_staged_row() -> None:
    shared = await _copy()
    copy = shared.copy
    keys, rows = f"{STAGE_PREFIX}k", f"{STAGE_PREFIX}r"
    await _staged(shared, copy.stage_rows, rows, _TABLE, _rows(("VA", 1, 1), ("VA", 2, 2), ("IL", 3, 3)), _KEY)
    assert await copy.apply_staged([StagedChange(table=_TABLE, key=_KEY, rows=rows)], mark=1) == 3
    assert copy.staged_stages() == []
    # a row staged for VA 2 before its key is named again (rows_stage) is taken out; VA 1 is named and not staged
    # again (deleted); VA 2 staged again after (updated); IL 3's row is staged with no key named (upserted)
    await _staged(shared, copy.stage_rows, rows, _TABLE, _rows(("VA", 2, 99)), _KEY)
    await _staged(shared, copy.stage_keys, keys, _TABLE, _KEY, _keys(("VA", 1), ("VA", 2)), rows_stage=rows)
    await _staged(shared, copy.stage_rows, rows, _TABLE, _rows(("VA", 2, 22), ("IL", 3, 33)), _KEY)
    await copy.apply_staged([StagedChange(table=_TABLE, key=_KEY, keys=keys, rows=rows)], mark=2)
    assert shared.rows(_TABLE) == [("IL", 3, 33), ("VA", 2, 22)]


async def test_a_key_taken_out_of_a_change_is_left_as_the_copy_holds_it() -> None:
    """the in-process copy's ``unstage_keys``: each stage's rows and keys of them leave it, the rest kept."""
    shared = await _copy()
    copy = shared.copy
    keys, rows = f"{STAGE_PREFIX}k", f"{STAGE_PREFIX}r"
    await _staged(shared, copy.stage_rows, rows, _TABLE, _rows(("VA", 1, 1), ("VA", 2, 2), ("IL", 3, 3)), _KEY)
    await copy.apply_staged([StagedChange(table=_TABLE, key=_KEY, rows=rows)], mark=1)
    await _staged(shared, copy.stage_keys, keys, _TABLE, _KEY, _keys(("VA", 1), ("VA", 2), ("IL", 3)), rows_stage=rows)
    await _staged(shared, copy.stage_rows, rows, _TABLE, _rows(("VA", 2, 22), ("IL", 3, 33)), _KEY)
    absent = f"{STAGE_PREFIX}absent"
    left = await _staged(shared, copy.unstage_keys, [keys, rows, absent], _TABLE, _KEY, _keys(("VA", 1), ("VA", 2)))
    assert left == 2
    assert absent not in copy.staged_stages()
    await copy.apply_staged([StagedChange(table=_TABLE, key=_KEY, keys=keys, rows=rows)], mark=2)
    # VA 1 (deleted) and VA 2 (updated) were taken out: left as held; IL 3 applied
    assert shared.rows(_TABLE) == [("IL", 3, 33), ("VA", 1, 1), ("VA", 2, 2)]


async def test_a_change_staged_as_many_alternating_pages_crosses_as_one_page_a_kind_and_applies_as_staged() -> None:
    """Flips if each staged page crosses to the apply worker as a file of its own (a commit of 25,000 keys in pages of
    50 ran the process out of open files), or if joining them loses the order they were staged in: a key named again
    after its row was staged, and not staged again, is deleted; one staged again after is upserted."""
    shared = await _copy()
    copy = shared.copy
    keys, rows = f"{STAGE_PREFIX}k", f"{STAGE_PREFIX}r"
    await _staged(shared, copy.stage_rows, rows, _TABLE, _rows(*((("VA", i, i)) for i in range(40))), _KEY)
    await copy.apply_staged([StagedChange(table=_TABLE, key=_KEY, rows=rows)], mark=1)
    handed: list[int] = []
    real = copy.lane.apply

    async def counted(pages: Any, **kwargs: Any) -> Any:
        handed.append(len(pages))
        return await real(pages, **kwargs)

    copy.lane.apply = counted  # type: ignore[method-assign]
    # 20 chunks, each its keys then its rows; then the first 10 keys named again with no row (deleted), and keys 0-4
    # staged again after that (upserted)
    for chunk in range(20):
        at = (chunk * 2, chunk * 2 + 1)
        await _staged(shared, copy.stage_keys, keys, _TABLE, _KEY, _keys(*(("VA", i) for i in at)), rows_stage=rows)
        await _staged(shared, copy.stage_rows, rows, _TABLE, _rows(*(("VA", i, 100 + i) for i in at)), _KEY)
    await _staged(shared, copy.stage_keys, keys, _TABLE, _KEY, _keys(*(("VA", i) for i in range(10))), rows_stage=rows)
    await _staged(shared, copy.stage_keys, keys, _TABLE, _KEY, _keys(*(("VA", i) for i in range(5))), rows_stage=rows)
    await _staged(shared, copy.stage_rows, rows, _TABLE, _rows(*(("VA", i, 500 + i) for i in range(5))), _KEY)
    await copy.apply_staged([StagedChange(table=_TABLE, key=_KEY, keys=keys, rows=rows)], mark=2)
    assert handed == [2], "one keys page and one rows page, whatever the pages staged"
    assert shared.rows(_TABLE) == [("VA", i, 500 + i) for i in range(5)] + [("VA", i, 100 + i) for i in range(10, 40)]


async def test_a_whole_change_replaces_its_table_and_publishes_its_mark() -> None:
    shared = await _copy()
    copy = shared.copy
    rows = f"{STAGE_PREFIX}r"
    await _staged(shared, copy.stage_rows, rows, _TABLE, _rows(("VA", 1, 1), ("VA", 2, 2)), _KEY)
    await copy.apply_staged([StagedChange(table=_TABLE, key=_KEY, rows=rows)], mark=1)
    await _staged(shared, copy.stage_rows, rows, _TABLE, _rows(("OH", 7, 7)), _KEY)
    whole = StagedChange(table=_TABLE, key=_KEY, rows=rows, whole=True, whole_kind="cold_start")
    await copy.apply_staged([whole], mark=5)
    assert shared.rows(_TABLE) == [("OH", 7, 7)]
    assert shared.rows(MARK_TABLE) == [(5,)]


@pytest.mark.parametrize("kind", ["cold_start", "schema_change", "rolling_upgrade"])
async def test_the_apply_marks_a_table_whole_only_for_a_whole_change_never_on_a_keyed_pass(kind: str) -> None:
    """the proving test of the whole marks declared in SharedCopyStages.apply_staged: a keyed change (a normal pass)
    marks nothing whole, so every row it does not name is kept; only a change staged whole, under the declared whole
    read that staged it (a cold start or an operator's reset, a schema change, an older writer's whole record), marks
    its table whole and replaces it. Flips if a keyed pass replaced the table, or a whole change of any kind did not."""
    shared = await _copy()
    copy = shared.copy
    rows = f"{STAGE_PREFIX}r"
    await _staged(shared, copy.stage_rows, rows, _TABLE, _rows(("VA", 1, 1), ("VA", 2, 2)), _KEY)
    await copy.apply_staged([StagedChange(table=_TABLE, key=_KEY, rows=rows)], mark=1)
    # a normal pass: one row named, the other kept
    await _staged(shared, copy.stage_rows, rows, _TABLE, _rows(("IL", 3, 3)), _KEY)
    await copy.apply_staged([StagedChange(table=_TABLE, key=_KEY, rows=rows)], mark=2)
    assert shared.rows(_TABLE) == [("IL", 3, 3), ("VA", 1, 1), ("VA", 2, 2)]
    # a whole change: the table replaced
    await _staged(shared, copy.stage_rows, rows, _TABLE, _rows(("OH", 7, 7)), _KEY)
    whole = StagedChange(table=_TABLE, key=_KEY, rows=rows, whole=True, whole_kind=kind)  # type: ignore[arg-type]
    await copy.apply_staged([whole], mark=3)
    assert shared.rows(_TABLE) == [("OH", 7, 7)]


async def test_a_change_whose_stage_is_gone_raises_lookup_error_and_one_that_fails_keeps_its_stages() -> None:
    shared = await _copy()
    copy = shared.copy
    rows = f"{STAGE_PREFIX}r"
    with pytest.raises(LookupError):
        await copy.apply_staged([StagedChange(table=_TABLE, key=_KEY, rows=rows)], mark=1)
    # a page the apply worker cannot cast to the table's columns: the apply fails, nothing is published
    await _staged(
        shared, copy.stage_rows, rows, _TABLE, pa.table({"state": ["VA"], "geo_id": ["x"], "votes": [1]}), _KEY
    )
    with pytest.raises(pa.ArrowInvalid):
        await copy.apply_staged([StagedChange(table=_TABLE, key=_KEY, rows=rows)], mark=1)
    assert copy.staged_stages() == [rows]
    assert shared.rows(_TABLE) == []
    copy.drop_stage(rows)
    assert copy.staged_stages() == []


async def test_a_prepared_reload_stages_in_the_reload_worker_and_publishes_only_with_its_swap() -> None:
    """``PreparesStaged``: the reload's pages go to the reload lane held, its stages keep none of them, and the copy
    is as it was until the swap's apply, which publishes it with the swap's mark."""
    shared = await _copy()
    copy = shared.copy
    rows = f"{STAGE_PREFIX}r"
    await _staged(shared, copy.stage_rows, rows, _TABLE, _rows(("VA", 1, 1)), _KEY)
    await copy.apply_staged([StagedChange(table=_TABLE, key=_KEY, rows=rows)], mark=1)
    stage = reload_stage(_TABLE)
    assert is_reload_stage(stage)
    await _staged(shared, copy.stage_rows, stage, _TABLE, pa.table({}), _KEY)
    reloaded = _rows(("VA", 1, 10), ("IL", 2, 20))
    # the reload's page, watched: once handed to the reload worker nothing in this process holds it, its stage neither
    page = weakref.ref(reloaded)
    await _staged(shared, copy.stage_rows, stage, _TABLE, reloaded, _KEY)
    del reloaded
    change = StagedChange(table=_TABLE, key=_KEY, rows=stage, whole=True, whole_kind="cold_start")
    await copy.prepare_staged([change])
    # the reload stages beside the copy; nothing of it is published, and its stage holds no pages
    for _ in range(500):
        if copy.reloads.status().staging == 0 and copy.reloads.status().waiting == 0:
            break
        await asyncio.sleep(0.01)
    assert copy.reloads.status().queued == 0
    assert shared.rows(_TABLE) == [("VA", 1, 1)]
    gc.collect()
    assert page() is None, "the stage still holds the page it let go to the reload lane"
    assert copy.staged_stages() == [stage]
    # a change given while it stages applies at once, and again over the reload as it publishes (decision 52)
    await _staged(shared, copy.stage_rows, rows, _TABLE, _rows(("TX", 3, 3)), _KEY)
    await copy.apply_staged([StagedChange(table=_TABLE, key=_KEY, rows=rows)], mark=2)
    assert shared.rows(_TABLE) == [("TX", 3, 3), ("VA", 1, 1)]
    assert await copy.apply_staged([change], mark=3) == 3
    assert shared.rows(_TABLE) == [("IL", 2, 20), ("TX", 3, 3), ("VA", 1, 10)]
    assert shared.rows(MARK_TABLE) == [(3,)]
    assert copy.reloads.status().reloads == 1
    assert copy.staged_stages() == []


async def test_a_prepared_reload_dropped_before_its_swap_publishes_nothing() -> None:
    shared = await _copy()
    copy = shared.copy
    stage = reload_stage(_TABLE)
    await _staged(shared, copy.stage_rows, stage, _TABLE, _rows(("VA", 1, 10)), _KEY)
    change = StagedChange(table=_TABLE, key=_KEY, rows=stage, whole=True, whole_kind="cold_start")
    await copy.prepare_staged([change])
    await asyncio.to_thread(copy.drop_stage, stage)
    for _ in range(500):
        if copy.reloads.status().cancelled:
            break
        await asyncio.sleep(0.01)
    assert copy.reloads.status().cancelled == 1
    with pytest.raises(LookupError):
        await copy.apply_staged([change], mark=1)
    assert shared.rows(_TABLE) == []


# --- a follower on it, end to end ------------------------------------------------------------------------------------


#: set on a thread while the fake L3 answers: its DuckDB is L3's, not the copy's
_IN_L3 = threading.local()


class _DuckDBCalls:
    """every call into DuckDB this process makes outside the fake L3, on any thread (``test_copy_follower_reads``)."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, int]] = []
        self._lock = threading.Lock()

    def __call__(self, frame: FrameType, event: str, arg: Any) -> None:
        if event != "c_call" or getattr(arg, "__module__", None) != "_duckdb" or getattr(_IN_L3, "on", False):
            return
        with self._lock:
            self.seen.append((getattr(arg, "__name__", repr(arg)), threading.get_ident()))

    def __enter__(self) -> _DuckDBCalls:
        sys.setprofile(self)
        threading.setprofile_all_threads(self)
        return self

    def __exit__(self, *exc: object) -> None:
        threading.setprofile_all_threads(None)
        sys.setprofile(None)


def _dict_rows(result: Any) -> list[dict[str, Any]]:
    names = [d[0] for d in result.description]
    return [dict(zip(names, row, strict=True)) for row in result.fetchall()]


@contextmanager
def _in_l3() -> Iterator[None]:
    """the fake L3's own DuckDB use, on this thread: not the copy's."""
    _IN_L3.on = True
    try:
        yield
    finally:
        _IN_L3.on = False


class _L3:
    """L3 as a DuckDB answering ``fetch``; every use of it marked (:data:`_IN_L3`), so the spy sees the copy's alone."""

    def __init__(self) -> None:
        with _in_l3():
            self.db = duckdb.connect(":memory:")
            self.db.execute(
                f"CREATE TABLE {_TABLE} (state VARCHAR, geo_id BIGINT, votes BIGINT, PRIMARY KEY (state, geo_id))"
            )
        _OPENED.append(self.db.close)
        self.statements = 0

    async def answers_from_yugabyte(self) -> bool:
        return False

    def put(self, rows: Sequence[tuple[str, int, int]]) -> None:
        with _in_l3():
            self.db.register("_put", _rows(*rows))
            try:
                self.db.execute(f"INSERT OR REPLACE INTO {_TABLE} SELECT * FROM _put")  # noqa: S608
            finally:
                self.db.unregister("_put")

    def delete(self, keys: Sequence[tuple[str, int]]) -> None:
        with _in_l3():
            for state, geo in keys:
                self.db.execute(f"DELETE FROM {_TABLE} WHERE state = ? AND geo_id = ?", [state, geo])  # noqa: S608

    def held(self) -> list[tuple[Any, ...]]:
        with _in_l3():
            return sorted(self.db.execute(f"SELECT state, geo_id, votes FROM {_TABLE}").fetchall())  # noqa: S608

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        await asyncio.sleep(0)
        with _in_l3():
            rows = run_bound(self.db.cursor(), sql, params, _dict_rows)
        if not sql.startswith("SELECT version()"):
            self.statements += 1
        return rows


class _Log:
    def __init__(self) -> None:
        self.count = 0
        self.records: dict[str, EpochChange] = {}

    async def current(self, table: str) -> str:
        return f"inc:{self.count}"

    def commit(self, keys: Sequence[tuple[str, int]] | None) -> None:
        self.count += 1
        token = f"inc:{self.count}"
        self.records[token] = EpochChange(
            table=_TABLE,
            token=token,
            whole_table=keys is None,
            rows=[ChangedRow(ids=[s, str(g)]) for s, g in keys or ()],
        )

    async def change(self, table_name: str, token: str) -> EpochChange | None:
        return self.records.get(token)

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        return await walk_by_count(self, table_name, after, upto)


class _Seqlock:
    def __init__(self) -> None:
        self.version = 0
        self.writing = False

    async def settled(self) -> Any:
        return Unsettled("a write is in progress") if self.writing else self.version


class _Objects:
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


async def _replica(
    l3: _L3,
    log: _Log,
    lock: _Seqlock,
    objects: _Objects,
    pointers: _Pointers,
    owner: str,
    *,
    shared: SharedCopy | None = None,
) -> tuple[SharedCopy, CopyFollower]:
    if shared is None:
        shared = await SharedCopy([_FOLLOWED]).started()
    tables = [_FOLLOWED]
    checkpoints = CopyCheckpoints(
        name="votes",
        tables=tables,
        backend=shared.copy,
        store=objects,
        pointers=pointers,
        retire=objects.retire,
        owner=owner,
        part_rows=40,
    )
    registry = CollectionRegistry()
    registry.follow_generation(_TABLE)
    follower = CopyFollower(
        registry,
        tables,
        shared.copy,
        l3,
        settled=lock.settled,
        generation=log.current,
        changes=log,
        page_size=50,
        checkpoints=checkpoints,
        stamp_table="scope_epochs",
        drift_workers=COPY_WORKERS,
        queries=shared.queries,
    )
    follower.register()
    return shared, follower


async def _votes(follower: CopyFollower, state: str, geo: int) -> list[Any]:
    """a key's votes in the copy's state now, read in a query worker."""
    async with follower.current() as (pinned, _stamp, _marks):
        rows = await pinned.run_job(
            copy_query_jobs.sql, (f"SELECT votes FROM {_TABLE} WHERE state = ? AND geo_id = ?", state, geo)
        )
    return list(rows.column("votes").to_pylist())


async def test_a_follower_on_the_worker_backend_applies_changes_deletes_a_whole_read_and_a_checkpoint() -> None:
    """Flips if a ``CopyFollower`` cannot run on ``SharedCopyBackend``, or runs DuckDB in the serving process to keep
    its copy: keyed changes and deletes, a whole-table record, a checkpoint written and a second replica started from
    it, each read back through ``CopyQueries`` (``lookup`` and a pinned state's job, both in query workers)."""
    l3, log, lock, objects, pointers = _L3(), _Log(), _Seqlock(), _Objects(), _Pointers()
    l3.put([("VA", i, i) for i in range(120)] + [("IL", i, 100 + i) for i in range(30)])
    log.commit(None)
    with _DuckDBCalls() as calls:
        shared, writer = await _replica(l3, log, lock, objects, pointers, "a")
        assert not await writer.start()  # no checkpoint yet: the first start reads L3 whole
        assert await writer.catch_up()
        assert shared.rows(_TABLE) == l3.held()
        # an update, an insert and two deletes, by their keys
        l3.put([("VA", 3, 333), ("TX", 500, 7)])
        l3.delete([("VA", 4), ("IL", 0)])
        log.commit([("VA", 3), ("TX", 500), ("VA", 4), ("IL", 0)])
        lock.version += 1
        l3.statements = 0
        assert await writer.catch_up()
        assert shared.rows(_TABLE) == l3.held()
        assert l3.statements == 1  # the change's keys, never the table
        assert writer.stamp == lock.version
        found = await shared.queries.lookup(_TABLE, _keys(("VA", 3), ("TX", 500), ("VA", 4)), _KEY)
        try:
            assert sorted(zip(*(found.table.column(c).to_pylist() for c in ("state", "geo_id", "votes")))) == [
                ("TX", 500, 7),
                ("VA", 3, 333),
            ]
            assert found.worker != os.getpid()
        finally:
            found.close()
        assert await _votes(writer, "VA", 3) == [333]
        assert await _votes(writer, "VA", 4) == []
        # a whole-table record: the table read again beside the copy and replaced in one version
        l3.delete([(s, g) for s, g, _ in l3.held() if s == "VA"])
        l3.put([("OH", i, 9) for i in range(5)])
        log.commit(None)
        lock.version += 1
        assert await writer.catch_up()
        assert shared.rows(_TABLE) == l3.held()
        assert await _votes(writer, "VA", 3) == []
        # a checkpoint, and a second replica started from it, taking the next change from the log alone
        await writer.checkpointed()
        assert pointers.entries
        l3.put([("OH", 1, 1111)])
        log.commit([("OH", 1)])
        lock.version += 1
        l3.statements = 0
        second, follower = await _replica(l3, log, lock, objects, pointers, "b")
        assert await follower.start()
        assert await follower.catch_up()
        assert second.rows(_TABLE) == l3.held()
        assert l3.statements == 1
        assert await _votes(follower, "OH", 1) == [1111]
        assert follower.stamp == lock.version
    assert calls.seen == [], f"DuckDB ran in the serving process: {calls.seen[:10]}"
    applied = shared.copy.lane.status()
    assert applied.failed == 0 and applied.applied >= 3


async def test_a_write_in_progress_applies_nothing_on_the_worker_backend() -> None:
    l3, log, lock, objects, pointers = _L3(), _Log(), _Seqlock(), _Objects(), _Pointers()
    l3.put([("VA", i, i) for i in range(10)])
    log.commit(None)
    shared, follower = await _replica(l3, log, lock, objects, pointers, "a")
    await follower.start()
    assert await follower.catch_up()
    before = shared.rows(_TABLE)
    l3.put([("VA", 1, 5)])
    log.commit([("VA", 1)])
    lock.writing = True
    assert not await follower.catch_up()
    assert shared.rows(_TABLE) == before
    lock.writing = False
    lock.version += 1
    assert await follower.catch_up()
    assert shared.rows(_TABLE) == l3.held()


# --- the whole settle after a start that read L3 whole ---------------------------------------------------------------


async def _totaled() -> SharedCopy:
    """a copy whose hooks derive :data:`copy_hooks.VOTE_TOTALS` from the votes, change by change and whole."""
    totals = (copy_hooks.VOTE_TOTALS_SCHEMA, PartRule(copy_hooks.VOTE_TOTALS, ("state",)))
    return await SharedCopy(
        [_FOLLOWED],
        derived={copy_hooks.VOTE_TOTALS: totals},
        hooks=copy_hooks.VoteTotals(),
        preload=(copy_hooks.__name__,),
    ).started()


async def test_a_start_that_read_l3_whole_settles_the_copy_whole_on_its_first_pass_through_the_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flips with no whole settle in the start sequence: the start that read L3 whole leaves the backend's hooks
    deferred, its first pass labelled with a stamp settles every state whole (one ``settle_whole``, after a failed
    one that failed only the pass), and a change after it derives its state as it applies."""
    l3, log, lock, objects, pointers = _L3(), _Log(), _Seqlock(), _Objects(), _Pointers()
    l3.put([("VA", i, 1) for i in range(10)] + [("IL", i, 2) for i in range(5)])
    log.commit(None)
    shared, follower = await _replica(l3, log, lock, objects, pointers, "a", shared=await _totaled())
    backend = shared.copy.backend
    assert not await follower.start()  # no checkpoint: the start reads L3 whole
    assert backend.whole_settle_due  # its changes apply with nothing derived until the whole settle
    settles: list[str] = []
    settle_whole = backend.settle_whole

    async def failing_once() -> Any:
        settles.append("asked")
        if len(settles) == 1:
            raise RuntimeError("the apply worker died mid-settle")
        return await settle_whole()

    monkeypatch.setattr(backend, "settle_whole", failing_once)
    with pytest.raises(RuntimeError, match="mid-settle"):
        await follower.catch_up()
    assert backend.whole_settle_due
    assert shared.rows(copy_hooks.VOTE_TOTALS) == []  # the whole read applied, nothing derived from it
    assert shared.rows(_TABLE) == l3.held()
    assert await follower.catch_up()
    assert settles == ["asked", "asked"]
    assert not backend.whole_settle_due
    assert shared.rows(copy_hooks.VOTE_TOTALS) == [("IL", 10, "whole"), ("VA", 10, "whole")]
    # settled once: a pass after it settles nothing more, and a change derives its own state as it applies
    l3.put([("IL", 0, 50)])
    log.commit([("IL", 0)])
    lock.version += 1
    assert await follower.catch_up()
    assert settles == ["asked", "asked"]
    assert shared.rows(copy_hooks.VOTE_TOTALS) == [("IL", 58, "change"), ("VA", 10, "whole")]


async def test_a_start_from_a_checkpoint_defers_nothing_and_asks_no_whole_settle() -> None:
    """Flips if a start wholly from a checkpoint deferred the hooks: its changes derive as they apply."""
    l3, log, lock, objects, pointers = _L3(), _Log(), _Seqlock(), _Objects(), _Pointers()
    l3.put([("VA", i, 1) for i in range(4)])
    log.commit(None)
    first, writer = await _replica(l3, log, lock, objects, pointers, "a", shared=await _totaled())
    assert not await writer.start()
    assert await writer.catch_up()
    await writer.checkpointed()
    second, follower = await _replica(l3, log, lock, objects, pointers, "b", shared=await _totaled())
    assert await follower.start()
    assert not second.copy.backend.whole_settle_due
    l3.put([("VA", 0, 9)])
    log.commit([("VA", 0)])
    lock.version += 1
    assert await follower.catch_up()
    assert second.rows(copy_hooks.VOTE_TOTALS) == [("VA", 12, "change")]


async def test_the_whole_settle_job_runs_once_at_a_start_not_wholly_from_a_checkpoint_never_on_a_pass() -> None:
    """the proving test of the whole settle declared in shared_copy._settle_whole_job (derive_whole, kind cold_start:
    Pace 2026-11-27, a pod start from the delta-files volume counts as a cold start): it runs once, at the first
    labelled pass after a start not wholly from a checkpoint (here one with no checkpoint: a start from the volume's
    fulls leaves it due the same way, CopyFollower.start), and never on a pass after it; a start wholly from a
    checkpoint never runs it. Its rows say which derived them: ``whole`` by derive_whole, ``change`` by derive."""
    l3, log, lock, objects, pointers = _L3(), _Log(), _Seqlock(), _Objects(), _Pointers()
    l3.put([("VA", i, 1) for i in range(10)] + [("IL", i, 2) for i in range(5)])
    log.commit(None)
    shared, follower = await _replica(l3, log, lock, objects, pointers, "a", shared=await _totaled())
    assert not await follower.start()
    assert shared.copy.backend.whole_settle_due
    assert await follower.catch_up()
    assert shared.rows(copy_hooks.VOTE_TOTALS) == [("IL", 10, "whole"), ("VA", 10, "whole")]
    for votes in (50, 60):
        l3.put([("IL", 0, votes)])
        log.commit([("IL", 0)])
        lock.version += 1
        assert await follower.catch_up()
        assert not shared.copy.backend.whole_settle_due
    assert shared.rows(copy_hooks.VOTE_TOTALS) == [("IL", 68, "change"), ("VA", 10, "whole")]
    await follower.checkpointed()
    second, again = await _replica(l3, log, lock, objects, pointers, "b", shared=await _totaled())
    assert await again.start()  # wholly from the checkpoint
    assert not second.copy.backend.whole_settle_due
    assert await again.catch_up()
    assert all(how != "whole" for _state, _votes, how in second.rows(copy_hooks.VOTE_TOTALS))


def _vote_page(*rows: tuple[str, int, int]) -> ChangePage:
    """a rows page of the followed votes table."""
    states, geos, votes = zip(*rows, strict=True)
    return ChangePage(
        ROWS,
        _TABLE,
        pa.table(
            {
                "state": pa.array(states, pa.string()),
                "geo_id": pa.array(geos, pa.int64()),
                "votes": pa.array(votes, pa.int64()),
            }
        ),
    )


async def test_a_whole_settle_is_a_step_of_the_apply_lane_its_merge_run_and_a_change_behind_it_applied_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flips if the stages' whole settle went past the apply lane: a settle that makes a tier merge due (it writes a
    per-run table's parts) gets that merge's turn with no change after it, and a change given while a settle waits
    its turn applies after it, the settle's merge between them."""
    runs = PartRule(copy_hooks.RUNS, ("state", "run_id", "candidate_id"), scope_column="state", run_column="run_id")
    shared = await SharedCopy(
        [_FOLLOWED],
        derived={copy_hooks.RUNS: (copy_hooks.RUNS_SCHEMA, runs)},
        hooks=copy_hooks.WholeRuns(),
        preload=(copy_hooks.__name__,),
    ).started()
    stages, backend = shared.copy, shared.copy.backend
    order: list[str] = []
    gate = asyncio.Event()
    gate.set()
    settle_whole, apply, merge = backend.settle_whole, backend.apply, backend.merge

    async def gated_settle() -> Any:
        order.append("settle")
        await gate.wait()
        return await settle_whole()

    async def spied_apply(*args: Any, **kwargs: Any) -> Any:
        order.append("apply")
        return await apply(*args, **kwargs)

    async def spied_merge() -> Any:
        order.append("merge")
        return await merge()

    monkeypatch.setattr(backend, "settle_whole", gated_settle)
    monkeypatch.setattr(backend, "apply", spied_apply)
    monkeypatch.setattr(backend, "merge", spied_merge)
    await stages.lane.apply([_vote_page(("VA", 1, 1), ("NC", 1, 2))])
    assert stages.defer_settles_until_whole()
    order.clear()
    settled = await stages.settle_whole()
    assert settled is not None and copy_hooks.RUNS in settled.parts
    await stages.lane.drained()
    # the merge the settle made due had its turn, with no change after it
    assert order == ["settle", "merge"]
    assert stages.lane.status().settles == 1
    # a change given while a settle waits its turn applies after it, the settle's merge between them
    order.clear()
    gate.clear()
    settling = asyncio.create_task(stages.settle_whole())
    for _ in range(1_000):  # 10 s at most for the settle to take its turn
        if order:
            break
        await asyncio.sleep(0.01)
    change = await stages.lane.submit([_vote_page(("VA", 1, 9))])
    await asyncio.sleep(0.05)
    assert order == ["settle"]  # the change waits its turn behind the settle
    gate.set()
    again = await settling
    applied = await change
    assert again is not None and order == ["settle", "merge", "apply"]
    assert int(applied.version.removeprefix("c-")) > int(again.version.removeprefix("c-"))
    assert sorted(shared.rows(_TABLE)) == [("NC", 1, 2), ("VA", 1, 9)]
    assert stages.lane.status().settles == 2
