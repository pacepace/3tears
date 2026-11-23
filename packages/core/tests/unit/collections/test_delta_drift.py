"""the writer's drift sample: random keys of its full read from L3 by key and compared at the same generation.

Decision 32 of ``design-pod-delta-files.md``, check B. The full, the delta files' volume and the keyed read are the real
ones; L3 is a DuckDB answering ``fetch`` that records every statement, and the epoch log is a dict of records. The
contract this pins:

- a row L3 holds otherwise than the full at the same generation is found, named, and counted
  (``delta_fold_drift_keys{table}``, ``delta_fold_drift_sampled_keys_total{outcome="differs"}``);
- the only statements reaching L3 are keyed reads of exactly the sampled keys (nothing to pick them: they are drawn
  from the full on the volume);
- a key a commit changed since its part's version is skipped, not reported;
- a replica that does not hold the write lease samples nothing and reads nothing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from prometheus_client import REGISTRY

pa = pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")

from threetears.core.cache.duckdb_binds import run_bound  # noqa: E402
from threetears.core.collections.copy_follower import FollowedTable  # noqa: E402
from threetears.core.collections.delta_drift import DeltaDriftSampler  # noqa: E402
from threetears.core.collections.epoch_change import ChangedRow, EpochChange, EpochWalk, walk_by_count  # noqa: E402
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column  # noqa: E402
from threetears.core.collections.volume_fulls import FullTable, publish_full_manifest, write_full_part  # noqa: E402

_TABLE = "results"
_KEY = ("state", "geo_id")
_COLUMNS = (
    Column(name="state", column_type=STRING_TYPE),
    Column(name="geo_id", column_type=BIGINT_TYPE),
    Column(name="votes", column_type=BIGINT_TYPE),
)
_NAMES = tuple(column.name for column in _COLUMNS)
_ROWS = [("VA", i, 10 + i) for i in range(4)] + [("IL", i, 20 + i) for i in range(4)]
_FOLLOWED = FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS, scope=("state", "state:"), delta_files=True)

_OPENED: list[Callable[[], None]] = []


@pytest.fixture(autouse=True)
def _close_what_the_test_opened() -> Iterator[None]:
    yield
    while _OPENED:
        _OPENED.pop()()


def _dict_rows(result: Any) -> list[dict[str, Any]]:
    names = [d[0] for d in result.description]
    return [dict(zip(names, row, strict=True)) for row in result.fetchall()]


class _L3:
    """L3 as a DuckDB answering ``fetch``; records every statement and its parameters."""

    def __init__(self) -> None:
        self.db = duckdb.connect(":memory:")
        _OPENED.append(self.db.close)
        self.db.execute(
            f"CREATE TABLE {_TABLE} (state VARCHAR, geo_id BIGINT, votes BIGINT, PRIMARY KEY (state, geo_id))"
        )
        self.statements: list[tuple[str, tuple[Any, ...]]] = []

    def put(self, rows: list[tuple[str, int, int]]) -> None:
        for row in rows:
            self.db.execute(f"INSERT OR REPLACE INTO {_TABLE} VALUES (?, ?, ?)", list(row))  # noqa: S608

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        await asyncio.sleep(0)
        self.statements.append((sql, params))
        return run_bound(self.db.cursor(), sql, params, _dict_rows)


class _Log:
    """the table's write generation and its change records."""

    def __init__(self) -> None:
        self.count = 1
        self.records: dict[tuple[str, str], EpochChange] = {}

    async def current(self, table: str) -> str:
        return f"inc:{self.count}"

    def commit(self, keys: list[tuple[str, int]]) -> str:
        self.count += 1
        token = f"inc:{self.count}"
        self.records[(_TABLE, token)] = EpochChange(
            table=_TABLE, token=token, rows=[ChangedRow(ids=[s, str(g)]) for s, g in keys]
        )
        return token

    async def change(self, table_name: str, token: str) -> EpochChange | None:
        return self.records.get((table_name, token))

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        return await walk_by_count(self, table_name, after, upto)


def _part_rows(rows: list[tuple[str, int, int]]) -> Any:
    return pa.table(
        {
            "state": pa.array([r[0] for r in rows], pa.string()),
            "geo_id": pa.array([r[1] for r in rows], pa.int64()),
            "votes": pa.array([r[2] for r in rows], pa.int64()),
        }
    )


def _world(volume: Path, *, leading: bool = True, sample_keys: int = len(_ROWS)) -> tuple[_L3, _Log, DeltaDriftSampler]:
    """L3 and the writer's full at inc:1, equal; a sampler over them."""
    l3, log = _L3(), _Log()
    l3.put(_ROWS)
    parts = tuple(
        write_full_part(volume, _TABLE, f"state:{state}", _part_rows([r for r in _ROWS if r[0] == state]), "inc:1")
        for state in ("IL", "VA")
    )
    publish_full_manifest(volume, term=1, tables={_TABLE: FullTable(version="inc:1", parts=parts)})
    sampler = DeltaDriftSampler(
        [_FOLLOWED],
        l3,
        generation=log.current,
        changes=log,
        holds_lease=lambda: leading,
        root=volume,
        sample_keys=sample_keys,
        sample_seconds=0,
    )
    return l3, log, sampler


def _sampled(outcome: str) -> float:
    labels = {"table": _TABLE, "outcome": outcome}
    return REGISTRY.get_sample_value("delta_fold_drift_sampled_keys_total", labels) or 0.0


async def test_a_row_l3_holds_otherwise_than_the_full_is_found_and_counted(tmp_path: Path) -> None:
    """Flips if a key whose row L3 holds otherwise than the full at the same generation (votes 999, never committed
    through a delta) is not reported, if an equal key is, or if the gauge and the counter do not say one key
    differs."""
    l3, _log, sampler = _world(tmp_path)
    assert await sampler.sample() == {_TABLE: []}
    assert REGISTRY.get_sample_value("delta_fold_drift_keys", {"table": _TABLE}) == 0
    l3.put([("VA", 1, 999)])
    differs_before = _sampled("differs")
    assert await sampler.sample() == {_TABLE: [("VA", "1")]}
    assert REGISTRY.get_sample_value("delta_fold_drift_keys", {"table": _TABLE}) == 1
    assert _sampled("differs") == differs_before + 1


async def test_only_keyed_reads_of_the_sampled_keys_reach_l3(tmp_path: Path) -> None:
    """Flips if picking the keys reads L3 (a scan, a count, a random order there), or if anything but one keyed read
    naming exactly the sampled keys reaches it: one statement, its bound key arrays the sample's size, its rows
    exactly the sampled keys'."""
    l3, _log, sampler = _world(tmp_path, sample_keys=3)
    assert await sampler.sample() == {_TABLE: []}
    assert len(l3.statements) == 1
    sql, params = l3.statements[0]
    # the key columns bound as one array each, one entry a sampled key (then the planner's hints, deduplicated)
    states, geo_ids = params[0], params[1]
    assert len(set(zip(states, geo_ids, strict=True))) == len(states) == 3
    answered = run_bound(l3.db.cursor(), sql, params, _dict_rows)
    assert {(row["state"], str(row["geo_id"])) for row in answered} == set(zip(states, geo_ids, strict=True))
    assert _sampled("equal") >= 3


async def test_a_key_changed_since_the_full_is_skipped_not_reported(tmp_path: Path) -> None:
    """Flips if a key a commit changed after its part's version (VA 1 at inc:2, the full still at inc:1) is reported
    as drift instead of skipped and counted ``changed_since``; or if the other keys stop being compared."""
    l3, log, sampler = _world(tmp_path)
    l3.put([("VA", 1, 555)])
    log.commit([("VA", 1)])
    changed_before, equal_before = _sampled("changed_since"), _sampled("equal")
    assert await sampler.sample() == {_TABLE: []}
    assert _sampled("changed_since") == changed_before + 1
    assert _sampled("equal") == equal_before + len(_ROWS) - 1


async def test_a_replica_that_is_not_the_writer_samples_nothing(tmp_path: Path) -> None:
    """Flips if a replica that does not hold the write lease samples: it would read L3 from every pod."""
    l3, _log, sampler = _world(tmp_path, leading=False)
    l3.put([("VA", 1, 999)])
    assert await sampler.sample() == {}
    assert l3.statements == []
