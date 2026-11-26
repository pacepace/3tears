"""The serving process lets versions of the copy go without holding its event loop (``design-copy-worker-backend.md``,
decision 2b: "the let-go lateness test green on the Linux pool").

What the serving process holds of a version, on the read path of 3T-8 (the engine slots gone): no mapping of the
version itself -- the query workers map it -- but each query's answer (:attr:`QueryAnswer.table`, zero-copy over the
result files the worker wrote) and each lease a reader pins (:meth:`CopyQueries.current`). A page handler closes its
answer (:meth:`QueryAnswer.aclose`, its files' close on a thread) and returns, dropping the table on the loop: the
answer's arrays are freed there, and its mapping goes with the last of them. A version superseded by later changes is
the serving process's no more once the last lease on it is released (an idle query worker moves its kept mapping onto
the current version as it is published, ``shared_arrow._KeptVersions``).

The contract: at a realistic shape -- a split table of 200 columns (reports' parts: candidates 195, geographies 201,
snapshots 197) over many scopes, the small ones packed (decision 2b) and a few big ones in parts of their own -- the
loop runs at most 20 ms later than the host's own idle loop at the same moments while a burst of answers (one scope
each, a big scope each, each let go as its handler returns; and keyed lookups across scopes, one handler's, dropped
together in one turn) and the leases beside them are let go, the versions they held
superseded twice, and the store's thread sweeps.

What a let-go costs the loop is the arrays it frees (an answer's columns times its record batches) and the mappings it
closes, not the rows: the answers here are as wide as the pod's, and as many as its query pool admits and more.
Replaces 3T-8's ``test_shared_arrow_let_go_loop`` (the reader it measured is gone).
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from threetears.core.collections.copy_parts import PartRule, is_pack
from threetears.core.collections.copy_queries import CopyQueries, CopyVersion, QueryAnswer
from threetears.core.collections.shared_copy import ROWS, ChangePage, SharedCopyBackend, held_parts
from threetears.core.shared_arrow import SharedArrowStore
from threetears.core.testing.loop_lateness import HostTicker, Tick, added_lateness, latest, ticks_while, trusted

#: the most the event loop may run late while answers and leases are let go, past what the host's own idle loop,
#: ticking in a process of its own, was late at the same moments (the loop watchdog's bound is 50 ms)
LOOP_MAY_BE_LATE = 0.020

#: what the test loop's ticker asks to sleep
TICK = 0.002

#: the table's width, as reports' widest tables
COLUMNS = 200

#: small scopes (packed together, decision 2b) and each one's rows
SMALL_SCOPES = 600
SMALL_ROWS = 40

#: big scopes (a part each) and each one's rows: an answer of one is several record batches
BIG_SCOPES = 4
BIG_ROWS = 6_000

#: the answers held and let go: one small scope each, one big scope each, and lookups of keys across many scopes
SCOPE_ANSWERS = 48
BIG_ANSWERS = 8
LOOKUPS = 16
LOOKUP_KEYS = 64

#: the leases readers pin beside them (a checkpoint's, a drift check's)
LEASES = 16

#: the query workers, as a reports pod's (``copy_queries._HELD_BACK_CPUS``: 6 CPUs less 3)
WORKERS = 3

_KEY = ("state", "office_key", "candidate_id")
_SCHEMA = pa.schema(
    [
        pa.field("state", pa.string()),
        pa.field("office_key", pa.string()),
        pa.field("candidate_id", pa.int64()),
        *(pa.field(f"c{index:03d}", pa.int64()) for index in range(len(_KEY), COLUMNS)),
    ]
)
_RULE = PartRule("results", _KEY, scope_column="state")


def _small(n: int) -> str:
    return f"S{n:04d}"


def _big(n: int) -> str:
    return f"BIG{n}"


def _rows(seed: int) -> pa.Table:
    """every scope's rows, each value column ``seed`` plus its row.

    :param seed: what the values start from
    :ptype seed: int
    :return: the rows
    :rtype: pa.Table
    """
    states = np.concatenate(
        [
            np.repeat(np.array([_small(n) for n in range(SMALL_SCOPES)], dtype=object), SMALL_ROWS),
            np.repeat(np.array([_big(n) for n in range(BIG_SCOPES)], dtype=object), BIG_ROWS),
        ]
    )
    count = len(states)
    candidates = np.concatenate(
        [np.tile(np.arange(SMALL_ROWS), SMALL_SCOPES), np.tile(np.arange(BIG_ROWS), BIG_SCOPES)]
    ).astype(np.int64)
    values = pa.array(np.arange(seed, seed + count, dtype=np.int64))
    columns: dict[str, pa.Array] = {
        "state": pa.array(states, pa.string()),
        "office_key": pa.array(np.full(count, "G", dtype=object), pa.string()),
        "candidate_id": pa.array(candidates),
    }
    for field in _SCHEMA:
        if field.name not in columns:
            columns[field.name] = values
    return pa.table(columns, schema=_SCHEMA)


def _keys(start: int) -> pa.Table:
    """keys of :data:`LOOKUP_KEYS` small scopes, one row each, from scope ``start``.

    :param start: the first scope
    :ptype start: int
    :return: the keys
    :rtype: pa.Table
    """
    scopes = [(start + step * 7) % SMALL_SCOPES for step in range(LOOKUP_KEYS)]
    return pa.table(
        {
            "state": pa.array([_small(n) for n in scopes], pa.string()),
            "office_key": pa.array(["G"] * LOOKUP_KEYS, pa.string()),
            "candidate_id": pa.array(np.arange(LOOKUP_KEYS) % SMALL_ROWS, pa.int64()),
        }
    )


@pytest.fixture
async def copy(tmp_path: Path) -> AsyncIterator[tuple[SharedArrowStore, SharedCopyBackend, CopyQueries]]:
    """a store, its copy's backend over the one wide split table, and a query pool of the pod's workers.

    :return: the store, the backend and the pool
    :rtype: AsyncIterator[tuple[SharedArrowStore, SharedCopyBackend, CopyQueries]]
    """
    store = SharedArrowStore(Path(tempfile.mkdtemp(dir=tmp_path)) / "shm", max_bytes=8 << 30)
    backend = SharedCopyBackend(store, {"results": _SCHEMA}, {"results": _RULE})
    queries = CopyQueries(store, workers=WORKERS, max_waiting=SCOPE_ANSWERS + BIG_ANSWERS + LOOKUPS)
    try:
        yield store, backend, queries
    finally:
        await queries.close()
        await backend.close()


async def _held(
    backend: SharedCopyBackend, queries: CopyQueries, seed: int
) -> tuple[list[QueryAnswer | None], list[CopyVersion | None]]:
    """a version of every scope, then the answers and leases readers hold over it.

    :return: the answers and the leases
    :rtype: tuple[list[QueryAnswer | None], list[CopyVersion | None]]
    """
    await backend.apply([ChangePage(ROWS, "results", _rows(seed))], whole={"results"})
    leases: list[CopyVersion | None] = [await queries.current() for _ in range(LEASES)]
    sql = "SELECT * FROM results WHERE state = ?"
    asked = [
        *(queries.run(sql, [_small(n * 11 % SMALL_SCOPES)]) for n in range(SCOPE_ANSWERS)),
        *(queries.run(sql, [_big(n % BIG_SCOPES)]) for n in range(BIG_ANSWERS)),
        *(queries.lookup("results", _keys(n * 13), _KEY) for n in range(LOOKUPS)),
    ]
    answers: list[QueryAnswer | None] = list(await asyncio.gather(*asked))
    big = [answer for answer in answers if answer is not None and answer.table.num_rows == BIG_ROWS]
    assert len(big) == BIG_ANSWERS and all(answer.table.num_columns == COLUMNS for answer in big)
    return answers, leases


async def _measured(
    store: SharedArrowStore, backend: SharedCopyBackend, queries: CopyQueries, seed: int
) -> tuple[list[Tick], list[Tick], int]:
    """the test loop's ticks and the host's idle loop's while every answer and lease is let go, as page handlers
    return, after two changes superseded the version they hold; and the arrays let go.

    :return: the loop's ticks, the host's, and the answers' arrays
    :rtype: tuple[list[Tick], list[Tick], int]
    """
    answers, leases = await _held(backend, queries, seed)
    arrays = sum(
        answer.table.num_columns * answer.table.column(0).num_chunks for answer in answers if answer is not None
    )
    held = store.current
    assert held is not None
    # two changes of every scope supersede the version the answers read and the leases pin
    for later in (1, 2):
        await backend.apply([ChangePage(ROWS, "results", _rows(seed + later))], whole={"results"})
    assert held.name in store.versions()

    async def run() -> None:
        alone = SCOPE_ANSWERS + BIG_ANSWERS
        for at in range(alone):
            answer = answers[at]
            assert answer is not None
            await answer.aclose()
            # the handler returns: its answer's table is dropped here, on the loop, its arrays freed and its mapping
            # closed with the last of them
            answers[at] = answer = None
            await asyncio.sleep(0)
        # one handler's lookups (a gather of them), closed together and dropped in one turn as it returns
        await asyncio.gather(*(answer.aclose() for answer in answers[alone:] if answer is not None))
        answers[alone:] = [None] * LOOKUPS
        await asyncio.sleep(0)
        for at, lease in enumerate(leases):
            assert lease is not None
            await lease.aclose()
            leases[at] = lease = None
        # the superseded versions swept, on the store's thread
        await asyncio.to_thread(store.settle)
        await asyncio.to_thread(store.sweep)
        await asyncio.sleep(0.05)

    with HostTicker() as host:
        ticks = await ticks_while(run, every=TICK)
    # every answer's files gone with its last array; the version is the query workers' to keep while idle (a mapping
    # their next query reuses) until the store needs its room
    assert store.status()["segments_claimed"] == 0 and store.segments() == ()
    return ticks, host.ticks, arrays


@pytest.mark.loop_timing
async def test_answers_and_leases_let_go_without_holding_the_loop(
    copy: tuple[SharedArrowStore, SharedCopyBackend, CopyQueries], record_property: Callable[[str, object], None]
) -> None:
    """a burst of answers over a wide, packed version, and the leases beside them, let go as two changes supersede it."""
    store, backend, queries = copy
    seeds = iter(range(0, 1_000_000, 10_000))
    arrays = 0

    async def measure() -> tuple[list[Tick], list[Tick]]:
        nonlocal arrays
        ticks, host, arrays = await _measured(store, backend, queries, next(seeds))
        return ticks, host

    ticks, host = await trusted(measure, lambda measured: measured, host_may_be_late=LOOP_MAY_BE_LATE)
    current = store.current
    assert current is not None
    parts = held_parts(store.root, current.name, "results")
    packs = sum(1 for part in parts if is_pack(part))
    assert packs >= 1 and len(parts) >= BIG_SCOPES + packs, sorted(parts)
    added, late = added_lateness(ticks, host, every=TICK), latest(ticks, every=TICK)
    # the numbers kept with the run (testament's metrics, a JUnit property elsewhere), pass or fail
    record_property("let_go_late_ms", round(late * 1000, 2))
    record_property("let_go_past_host_ms", round(added * 1000, 2))
    record_property("let_go_arrays", arrays)
    print(
        f"let go of {SCOPE_ANSWERS + BIG_ANSWERS + LOOKUPS} answers ({arrays} arrays of {COLUMNS} columns) and "
        f"{LEASES} leases over a version of {len(parts)} parts ({packs} packs): late {late * 1000:.1f} ms, past the "
        f"host {added * 1000:.1f} ms"
    )
    assert added < LOOP_MAY_BE_LATE, (
        f"the loop ran {added * 1000:.1f} ms late past the host's own idle loop ({late * 1000:.1f} ms in all) while "
        f"{SCOPE_ANSWERS + BIG_ANSWERS + LOOKUPS} answers ({arrays} arrays) and {LEASES} leases were let go"
    )
