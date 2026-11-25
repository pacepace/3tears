"""a copy that starts from the volume checks its own rows of each state against the full's fingerprint, never L3.

Decision 32 of ``design-pod-delta-files.md``, check A: each full part's manifest entry carries a fingerprint of its rows
(``volume_fulls.rows_fingerprint``, one definition both sides use); every pod copy periodically fingerprints its own
rows of each (table, state) it holds at exactly the part's version and compares. The registry, the follower, the delta
files and the fulls are the real ones; L3 is a DuckDB that counts the statements it answers for each table. The
contract this pins:

- a copy equal to the full reads 0 on ``copy_follower_drift_parts`` for every state, whether its states are read from
  the copy by the key's leading state or by the keys each part holds;
- one corrupted copy row sets the gauge to 1 for exactly that table and state, and a later equal check sets it back;
- a state the copy holds at another version than its part's is skipped (``copy_follower_drift_checks_total``
  ``outcome="skipped"``), and once the full holds it at the copy's version it is compared again;
- a row the copy holds of a state and the part does not is found, for a state read by its key's prefix or by its
  scope column (a table keyed without its state);
- a part fingerprinted in an older format counts ``no_fingerprint``, never drift;
- no check issues an L3 statement.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from prometheus_client import REGISTRY
from sqlalchemy import BigInteger, Column as SaColumn, MetaData, String, Table
from threetears.core.testing.l1_tables import L1_EVICTION_INFO, TEST_L1_TABLE

pa = pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")

from threetears.core.cache.duckdb import DuckDBBackend  # noqa: E402
from threetears.core.cache.duckdb_binds import run_bound  # noqa: E402
from threetears.core.collections import CollectionRegistry  # noqa: E402
from threetears.core.collections.copy_follower import CopyFollower, FollowedTable  # noqa: E402
from threetears.core.collections.delta_files import TableDelta, write_delta_file  # noqa: E402
from threetears.core.collections.epoch_change import ChangedRow, EpochChange, EpochWalk, walk_by_count  # noqa: E402
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column, rail_arrow_types  # noqa: E402
from threetears.core.collections.volume_fulls import (  # noqa: E402
    FullTable,
    VolumeFulls,
    current_full_manifest,
    publish_full_manifest,
    write_full_part,
)
from threetears.core.process_workers import ProcessWorkers  # noqa: E402

_RESULTS = "results"
_TALLIES = "tallies"
_OPTED = (_RESULTS, _TALLIES)
#: the key with its state leading (a state is read from the copy by that prefix of the key), with it second (read by
#: the keys its part holds), and without it (an AP results table, its state a row column: read by that column)
_KEYS = {"state_led": ("state", "geo_id"), "state_not_led": ("geo_id", "state"), "state_not_in_key": ("geo_id",)}
_COLUMNS = (
    Column(name="state", column_type=STRING_TYPE),
    Column(name="geo_id", column_type=BIGINT_TYPE),
    Column(name="votes", column_type=BIGINT_TYPE),
)
_NAMES = tuple(column.name for column in _COLUMNS)
_TYPES = rail_arrow_types(_COLUMNS)
#: each geo_id of one state only, so a key without the state names one row
_ROWS = [("VA", i, 10 + i) for i in range(4)] + [("IL", 10 + i, 20 + i) for i in range(4)]

_OPENED: list[Callable[[], None]] = []


@pytest.fixture(autouse=True)
def _close_what_the_test_opened() -> Iterator[None]:
    """close every DuckDB the test opened."""
    yield
    while _OPENED:
        _OPENED.pop()()


@pytest.fixture(autouse=True)
def _volume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """the shared volume the delta files and the fulls are saved on and read off."""
    monkeypatch.setenv("THREETEARS_DELTA_FILES_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
async def workers() -> AsyncIterator[ProcessWorkers]:
    made = ProcessWorkers(workers=1, name="test copy drift")
    try:
        yield made
    finally:
        await made.close()


@pytest.fixture(params=sorted(_KEYS))
def key(request: pytest.FixtureRequest) -> tuple[str, ...]:
    """each test runs over every way a state reaches the copy's rows: its key's prefix, its part's keys, or its
    scope column."""
    return _KEYS[request.param]


def _followed(name: str, key: tuple[str, ...]) -> FollowedTable:
    return FollowedTable(name=name, key=key, columns=_COLUMNS, scope=("state", "state:"), delta_files=True)


def _metadata(key: tuple[str, ...]) -> MetaData:
    metadata = MetaData()
    kinds: dict[str, Any] = {"state": String(8), "geo_id": BigInteger}
    for name in _OPTED:
        Table(
            name,
            metadata,
            *(SaColumn(column, kinds[column], primary_key=True) for column in key),
            *(SaColumn(column, kinds[column]) for column in ("state",) if column not in key),
            SaColumn("votes", BigInteger),
            info={L1_EVICTION_INFO: TEST_L1_TABLE},
        )
    return metadata


def _dict_rows(result: Any) -> list[dict[str, Any]]:
    names = [d[0] for d in result.description]
    return [dict(zip(names, row, strict=True)) for row in result.fetchall()]


def _upsert(db: Any, table: str, rows: Sequence[tuple[str, int, int]], key: Sequence[str]) -> None:
    """``rows`` (state, geo_id, votes) upserted into ``table`` in one statement, the last row of a key winning."""
    if not rows:
        return
    put = pa.table(
        {
            "state": pa.array([r[0] for r in rows], pa.string()),
            "geo_id": pa.array([r[1] for r in rows], pa.int64()),
            "votes": pa.array([r[2] for r in rows], pa.int64()),
            "at__": pa.array(range(len(rows)), pa.int64()),
        }
    )
    db.register("_put", put)
    try:
        db.execute(
            f"INSERT OR REPLACE INTO {table} (state, geo_id, votes) SELECT state, geo_id, votes FROM _put "  # noqa: S608
            f"QUALIFY row_number() OVER (PARTITION BY {', '.join(key)} ORDER BY at__ DESC) = 1"
        )
    finally:
        db.unregister("_put")


class _L3:
    """L3 as a DuckDB answering ``fetch``; counts every statement it answers."""

    def __init__(self, key: tuple[str, ...]) -> None:
        self.key = key
        self.db = duckdb.connect(":memory:")
        _OPENED.append(self.db.close)
        for name in _OPTED:
            self.db.execute(
                f"CREATE TABLE {name} (state VARCHAR, geo_id BIGINT, votes BIGINT, PRIMARY KEY ({', '.join(key)}))"
            )
        self.statements = 0

    def put(self, table: str, rows: list[tuple[str, int, int]]) -> None:
        # one statement over the rows as an Arrow table, the last of a key winning as row-by-row upserts left it: a
        # statement a row binds each value from Python (a failed ``import pandas`` a value, under testament a stat of
        # every path), which held the loop past the loop-stall guard's limit (threetears.core.testing.loop_stall)
        _upsert(self.db, table, rows, self.key)

    def held(self, table: str) -> list[tuple[Any, ...]]:
        return sorted(self.db.execute(f"SELECT {', '.join(_NAMES)} FROM {table}").fetchall())  # noqa: S608

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        await asyncio.sleep(0)
        self.statements += 1
        return run_bound(self.db.cursor(), sql, params, _dict_rows)


class _Log:
    """each table's write generation and its change records."""

    def __init__(self) -> None:
        self.count: dict[str, int] = dict.fromkeys(_OPTED, 1)
        self.records: dict[tuple[str, str], EpochChange] = {}

    async def current(self, table: str) -> str:
        return f"inc:{self.count[table]}"

    def commit(self, table: str, keys: list[tuple[str, ...]], delta_file: str) -> str:
        self.count[table] += 1
        token = f"inc:{self.count[table]}"
        self.records[(table, token)] = EpochChange(
            table=table, token=token, rows=[ChangedRow(ids=list(ids)) for ids in keys], delta_file=delta_file
        )
        return token

    async def change(self, table_name: str, token: str) -> EpochChange | None:
        return self.records.get((table_name, token))

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        return await walk_by_count(self, table_name, after, upto)


class _Scopes:
    """the writer's record, settled: each scope's epoch."""

    def __init__(self) -> None:
        self.version = 1
        self.epochs = {"state:VA": 1, "state:IL": 1}

    async def settled(self) -> Any:
        return SimpleNamespace(version=self.version, epochs=dict(self.epochs), pending=frozenset())

    def commit(self, *scopes: str) -> None:
        self.version += 1
        for scope in scopes:
            self.epochs[scope] = self.version


def _part_rows(rows: list[tuple[str, int, int]]) -> Any:
    return pa.table(
        {
            "state": pa.array([r[0] for r in rows], pa.string()),
            "geo_id": pa.array([r[1] for r in rows], pa.int64()),
            "votes": pa.array([r[2] for r in rows], pa.int64()),
        }
    )


def _entry(volume: Path, table: str, rows: list[tuple[Any, ...]], versions: dict[str, str]) -> FullTable:
    """the table's entry in a full: a part a state, each at its version in ``versions``."""
    parts = tuple(
        write_full_part(volume, table, f"state:{state}", _part_rows([r for r in rows if r[0] == state]), version)
        for state, version in sorted(versions.items())
    )
    return FullTable(version=max(versions.values()), parts=parts)


def _parts(name: str, state: str) -> float | None:
    return REGISTRY.get_sample_value("copy_follower_drift_parts", {"table": name, "state": f"state:{state}"})


def _checks(name: str, outcome: str) -> float:
    return REGISTRY.get_sample_value("copy_follower_drift_checks_total", {"table": name, "outcome": outcome}) or 0.0


def _check_seconds(outcome: str) -> float:
    """how many fingerprint checks' seconds were observed for ``outcome``."""
    return REGISTRY.get_sample_value("volume_fulls_fingerprint_check_seconds_count", {"outcome": outcome}) or 0.0


class _World:
    def __init__(self, volume: Path, workers: ProcessWorkers, key: tuple[str, ...]) -> None:
        self.volume = volume
        self.key = key
        self.l3, self.log, self.lock = _L3(key), _Log(), _Scopes()
        self.backend = DuckDBBackend()
        _OPENED.append(self.backend.reset)
        self.backend.initialize(_metadata(key))
        self.registry = CollectionRegistry()
        for name in _OPTED:
            self.registry.follow_generation(name)
        self.follower = CopyFollower(
            self.registry,
            [_followed(name, key) for name in _OPTED],
            self.backend,
            self.l3,
            settled=self.lock.settled,
            generation=self.log.current,
            changes=self.log,
            page_size=50,
            stamp_table="scope_epochs",
            fulls=VolumeFulls(volume, [_followed(name, key) for name in _OPTED], poll_seconds=0.01),
            drift_check_seconds=0,
            drift_workers=workers,
        )
        self.writes = 0

    async def started(self) -> None:
        """L3 and the writer's first full at inc:1; the copy starts from it."""
        for name in _OPTED:
            self.l3.put(name, _ROWS)
        self.publish(1, {name: {"VA": "inc:1", "IL": "inc:1"} for name in _OPTED})
        await self.follower.start()
        assert await self.follower.catch_up()
        self.l3.statements = 0

    def publish(self, term: int, versions: dict[str, dict[str, str]]) -> None:
        """a full of L3 as it holds each table now, each state's part at its version in ``versions``."""
        publish_full_manifest(
            self.volume,
            term=term,
            tables={name: _entry(self.volume, name, self.l3.held(name), versions[name]) for name in _OPTED},
        )

    def write(self, table: str, rows: list[tuple[str, int, int]]) -> None:
        """a commit: L3 changed and its delta file saved before the record names it."""
        self.l3.put(table, rows)
        self.writes += 1
        delta = TableDelta(name=table, types=_TYPES, key=self.key)
        delta.rows([dict(zip(_NAMES, row, strict=True)) for row in rows], list(_NAMES))
        delta_file = write_delta_file(self.volume, delta, f"w{self.writes}")
        named = [dict(zip(_NAMES, row, strict=True)) for row in rows]
        self.log.commit(table, [tuple(str(row[column]) for column in self.key) for row in named], delta_file)
        self.lock.commit(*{f"state:{r[0]}" for r in rows})


async def test_a_copy_equal_to_the_full_reads_zero_drift_and_reads_no_l3(
    _volume: Path, workers: ProcessWorkers, key: tuple[str, ...]
) -> None:
    """Flips if a copy equal to the full's parts reads as drifted on any state (the fingerprints of one definition
    disagree between the writer's part and the copy's own rows, for the order rows are held in, the copy's DuckDB types
    or the columns' order), if a state at the part's version is not compared, or if the check issues an L3 statement."""
    world = _World(_volume, workers, key)
    await world.started()
    equal_before = _checks(_RESULTS, "equal")
    timed_before = _check_seconds("equal")
    outcomes = await world.follower.check_drift()
    assert outcomes == {(name, f"state:{state}"): "equal" for name in _OPTED for state in ("IL", "VA")}
    for name in _OPTED:
        for state in ("VA", "IL"):
            assert _parts(name, state) == 0
    assert _checks(_RESULTS, "equal") == equal_before + 2
    # each part's check timed by its outcome (volume_fulls_fingerprint_check_seconds), every table's
    assert _check_seconds("equal") == timed_before + 2 * len(_OPTED)
    assert world.l3.statements == 0


async def test_a_corrupted_copy_row_flips_the_gauge_for_exactly_that_table_and_state(
    _volume: Path, workers: ProcessWorkers, key: tuple[str, ...]
) -> None:
    """Flips if one changed row of the copy (votes 999 where the full and L3 hold 11) is not found, if it is reported
    on another table or state than its own, if the gauge stays 1 once the copy is equal again, or if the check reads
    L3."""
    world = _World(_volume, workers, key)
    await world.started()
    drift_before = _checks(_RESULTS, "drift")
    timed_before = _check_seconds("drift")
    world.backend.execute_query("UPDATE results SET votes = 999 WHERE state = 'VA' AND geo_id = 1 RETURNING geo_id")
    outcomes = await world.follower.check_drift()
    assert outcomes[(_RESULTS, "state:VA")] == "drift"
    assert {key: outcome for key, outcome in outcomes.items() if outcome != "equal"} == {
        (_RESULTS, "state:VA"): "drift"
    }
    assert _parts(_RESULTS, "VA") == 1
    assert _parts(_RESULTS, "IL") == 0
    assert _parts(_TALLIES, "VA") == 0
    assert _parts(_TALLIES, "IL") == 0
    assert _checks(_RESULTS, "drift") == drift_before + 1
    assert _check_seconds("drift") == timed_before + 1
    # the row put back: the next check finds the state equal and the gauge drops to 0
    world.backend.execute_query("UPDATE results SET votes = 11 WHERE state = 'VA' AND geo_id = 1 RETURNING geo_id")
    assert (await world.follower.check_drift())[(_RESULTS, "state:VA")] == "equal"
    assert _parts(_RESULTS, "VA") == 0
    assert world.l3.statements == 0


async def test_a_state_held_at_another_version_than_its_part_is_skipped_until_the_full_catches_up(
    _volume: Path, workers: ProcessWorkers, key: tuple[str, ...]
) -> None:
    """Flips if a state the copy applied a later change of (VA at inc:2, its part still at inc:1) is compared with the
    older part (a false drift) instead of skipped and counted ``skipped``; if a state no change touched (IL) stops being
    compared; or if, once the full holds VA at inc:2, the copy does not compare it again and find it equal. No L3."""
    world = _World(_volume, workers, key)
    await world.started()
    world.write(_RESULTS, [("VA", 1, 111)])
    assert await world.follower.catch_up()
    skipped_before = _checks(_RESULTS, "skipped")
    outcomes = await world.follower.check_drift()
    assert outcomes[(_RESULTS, "state:VA")] == "skipped"
    assert outcomes[(_RESULTS, "state:IL")] == "equal"
    assert _checks(_RESULTS, "skipped") == skipped_before + 1
    # the writer folds VA to inc:2: comparable again, and equal
    world.publish(2, {_RESULTS: {"VA": "inc:2", "IL": "inc:1"}, _TALLIES: {"VA": "inc:1", "IL": "inc:1"}})
    outcomes = await world.follower.check_drift()
    assert outcomes[(_RESULTS, "state:VA")] == "equal"
    assert _parts(_RESULTS, "VA") == 0
    assert world.l3.statements == 0


@pytest.mark.parametrize("shape", ["state_led", "state_not_in_key"])
async def test_an_extra_copy_row_of_a_state_flips_the_gauge(_volume: Path, workers: ProcessWorkers, shape: str) -> None:
    """Flips if a row the copy holds of a state and the full's part does not (VA geo 99, never written) is not seen
    for a table whose state is read from the copy by its key's leading state or, keyed without its state (AP results),
    by its scope column: read by the part's keys instead, the copy's extra row is missed and the state reads equal.
    Also flips if the check reads L3."""
    world = _World(_volume, workers, _KEYS[shape])
    await world.started()
    world.backend.execute_query("INSERT INTO results (state, geo_id, votes) VALUES ('VA', 99, 5) RETURNING geo_id")
    outcomes = await world.follower.check_drift()
    assert {key: outcome for key, outcome in outcomes.items() if outcome != "equal"} == {
        (_RESULTS, "state:VA"): "drift"
    }
    assert _parts(_RESULTS, "VA") == 1
    assert _parts(_RESULTS, "IL") == 0
    assert world.l3.statements == 0


async def test_a_part_fingerprinted_in_an_older_format_reads_no_fingerprint_never_drift(
    _volume: Path, workers: ProcessWorkers
) -> None:
    """Flips if a full an older writer fingerprinted (the per-row BLAKE2b format, ``<columns>-<rows>-<sum>``) is
    compared with the copy's new-format fingerprint: every state would read as drifted the moment the copies upgrade
    before the writer. Each part must count ``no_fingerprint``, the drift count not move, and no L3 statement issue, even
    with a copy row changed."""
    world = _World(_volume, workers, _KEYS["state_led"])
    await world.started()
    current = current_full_manifest(_volume)
    assert current is not None
    publish_full_manifest(
        _volume,
        term=2,
        tables={
            name: FullTable(
                version=entry.version,
                parts=tuple(
                    dataclasses.replace(part, fingerprint="b2942aa7-4-136b4254e1a789f18e6ae91cdc46e3dc")
                    for part in entry.parts
                ),
            )
            for name, entry in current.tables.items()
        },
    )
    world.backend.execute_query("UPDATE results SET votes = 999 WHERE state = 'VA' AND geo_id = 1 RETURNING geo_id")
    drift_before = _checks(_RESULTS, "drift")
    outcomes = await world.follower.check_drift()
    assert outcomes == {(name, f"state:{state}"): "no_fingerprint" for name in _OPTED for state in ("IL", "VA")}
    assert _checks(_RESULTS, "drift") == drift_before
    assert world.l3.statements == 0
