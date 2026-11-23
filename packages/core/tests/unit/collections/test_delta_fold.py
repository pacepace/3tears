"""The writer's fold (:mod:`threetears.core.collections.delta_fold`, ``design-pod-delta-files.md`` decisions 8, 9, 10,
13, 15, 16 and 19): after each commit, off the serving process, each state whose deltas since its last fold reach half
its part gets a new part (old part plus deltas, merged as L3 merged them, then the table's keep rule), published in a
manifest at the writer's term; covered deltas and files no manifest in use names are deleted.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import re
import time
from collections import Counter
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from prometheus_client import REGISTRY

pa = pytest.importorskip("pyarrow")
pa_ipc = pytest.importorskip("pyarrow.ipc")
pc = pytest.importorskip("pyarrow.compute")
duckdb = pytest.importorskip("duckdb")

from threetears.core.collections.delta_files import (  # noqa: E402
    OWN_DELTA_SAVES,
    TableDelta,
    read_volume_file,
    apply_delta_row,
    read_delta_file,
    write_delta_file,
)
from threetears.core.collections import delta_files, delta_fold  # noqa: E402
from threetears.core.collections.delta_fold import DeltaFolds, FoldedTable  # noqa: E402
from threetears.core.collections.delta_keep import DeltaKeepRule  # noqa: E402
from threetears.core.collections.epoch_change import (  # noqa: E402
    ChangedRow,
    EpochChange,
    EpochWalk,
    walk_by_count,
)
from threetears.core.collections.noted_keys import NotedKeysCursor, NotedKeysPage  # noqa: E402
from threetears.core.collections.schema_backed import (  # noqa: E402
    BIGINT_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
    rail_arrow_types,
)
from threetears.core.collections.volume_fulls import (  # noqa: E402
    FullManifest,
    FullPart,
    FullTable,
    current_full_manifest,
    full_table_floor,
    publish_full_manifest,
    write_full_part,
)
from threetears.core.exceptions import GenerationUnavailableError  # noqa: E402
from threetears.core.keyed import ThreadedDuckDB, read_keyed_arrow  # noqa: E402
from threetears.core.process_workers import ProcessWorkers  # noqa: E402
from threetears.core.testing.l1_tables import TEST_L1_TABLE  # noqa: E402

_SCOPE = ("state", "state:")


class _NewestTwoRuns(DeltaKeepRule):
    """a thinning-like rule: each race's two newest runs (monotone, as LSF's run thinning is)."""

    @classmethod
    def keep(cls, rows: Any) -> Any:
        by_race: dict[tuple[Any, Any], list[int]] = {}
        for state, race, run in zip(
            rows.column("state").to_pylist(),
            rows.column("race").to_pylist(),
            rows.column("run").to_pylist(),
            strict=True,
        ):
            by_race.setdefault((state, race), []).append(run)
        newest = {race: set(sorted(runs)[-2:]) for race, runs in by_race.items()}
        mask = [
            run in newest[(state, race)]
            for state, race, run in zip(
                rows.column("state").to_pylist(),
                rows.column("race").to_pylist(),
                rows.column("run").to_pylist(),
                strict=True,
            )
        ]
        return rows.filter(pa.array(mask, pa.bool_()))


class _NewestRun(DeltaKeepRule):
    """LSF county projections' rule: each race's newest run only."""

    @classmethod
    def keep(cls, rows: Any) -> Any:
        newest: dict[tuple[Any, Any], int] = {}
        for state, race, run in zip(
            rows.column("state").to_pylist(),
            rows.column("race").to_pylist(),
            rows.column("run").to_pylist(),
            strict=True,
        ):
            newest[(state, race)] = max(run, newest.get((state, race), run))
        mask = [
            newest[(state, race)] == run
            for state, race, run in zip(
                rows.column("state").to_pylist(),
                rows.column("race").to_pylist(),
                rows.column("run").to_pylist(),
                strict=True,
            )
        ]
        return rows.filter(pa.array(mask, pa.bool_()))


class _MarkedByWorker(DeltaKeepRule):
    """keeps every row, its ``votes`` set to the process the rule ran in."""

    @classmethod
    def keep(cls, rows: Any) -> Any:
        index = rows.column_names.index("votes")
        return rows.set_column(index, "votes", pa.array([os.getpid()] * rows.num_rows, pa.int64()))


def _votes_schema(keep: type[DeltaKeepRule] | None = None) -> TableSchema:
    return TableSchema(
        name="votes",
        primary_key=("state", "geo_id"),
        columns=[
            Column("state", STRING_TYPE),
            Column("geo_id", BIGINT_TYPE),
            Column("votes", BIGINT_TYPE, nullable=True),
            Column("note", STRING_TYPE, nullable=True),
        ],
        l1_eviction=TEST_L1_TABLE,
        delta_files=True,
        delta_keep=keep,
    )


def _runs_schema(keep: type[DeltaKeepRule]) -> TableSchema:
    return TableSchema(
        name="runs",
        primary_key=("state", "race", "run"),
        columns=[
            Column("state", STRING_TYPE),
            Column("race", STRING_TYPE),
            Column("run", BIGINT_TYPE),
            Column("votes", BIGINT_TYPE, nullable=True),
        ],
        l1_eviction=TEST_L1_TABLE,
        delta_files=True,
        delta_keep=keep,
    )


class _Writer:
    """a writing load's side of the fold: saves each commit's delta file, then tells the fold, as a commit does."""

    def __init__(self, root: Path, schema: TableSchema, folds: DeltaFolds) -> None:
        self.root = root
        self.schema = schema
        self.folds = folds
        self.count = 0
        self.files: list[str] = []
        self.version = 5

    def save(
        self,
        rows: Sequence[Mapping[str, Any]] = (),
        *,
        written: Sequence[str] | None = None,
        deletes: Sequence[Sequence[Any]] = (),
        delete_scope: Any = None,
    ) -> str:
        """a commit's delta file, saved before the commit; not yet told to the fold."""
        self.count += 1
        delta = TableDelta(
            name=self.schema.name,
            types=rail_arrow_types(self.schema.columns),
            key=tuple(self.schema.pk_columns),
            scope_column=self.schema.delta_scope,
        )
        key = set(self.schema.pk_columns)
        delta.rows(
            list(rows),
            list(written) if written is not None else [c.name for c in self.schema.columns if c.name not in key],
        )
        delta.deletes(list(deletes), scope=delete_scope)
        file = write_delta_file(self.root, delta, f"w{self.count:04d}")
        self.files.append(file)
        return file

    def tell(self, file: str) -> None:
        """the commit landed: the fold hears its file and the generation it wrote."""
        token = f"inc:{self.files.index(file) + 1}"
        self.folds.committed({self.schema.name: file}, {self.schema.name: token}, version=self.version)

    async def commit(self, rows: Sequence[Mapping[str, Any]] = (), **kwargs: Any) -> str:
        file = self.save(rows, **kwargs)
        self.tell(file)
        await self.folds.wait()
        return file


@pytest.fixture
async def workers() -> AsyncIterator[ProcessWorkers]:
    made = ProcessWorkers(workers=1, name="test delta folds")
    try:
        yield made
    finally:
        await made.close()


def _first_full(root: Path, *tables: str) -> FullManifest:
    """the full the writer's rebuild publishes before its first delta: every table, no part yet."""
    return publish_full_manifest(root, term=1, tables={name: FullTable(version="inc:0") for name in tables})


def _writer(root: Path, schema: TableSchema, workers: ProcessWorkers) -> _Writer:
    _first_full(root, schema.name)
    return _Writer(root, schema, DeltaFolds(root, [FoldedTable.of(schema, scope=_SCOPE)], workers=workers))


def _part(root: Path, table: str, scope: str) -> FullPart | None:
    manifest = current_full_manifest(root)
    assert manifest is not None
    return next((p for p in manifest.tables[table].parts if p.scope == scope), None)


def _part_rows(root: Path, part: FullPart) -> list[dict[str, Any]]:
    rows = read_volume_file(root / part.file)
    if "_3tears_deleted" in rows.column_names:
        rows = rows.filter(pc.invert(rows.column("_3tears_deleted")))
        rows = rows.drop_columns(["_3tears_deleted", "_3tears_written"])
    return sorted(rows.to_pylist(), key=lambda row: tuple(str(v) for v in row.values()))


def _votes(state: str, geo_ids: range, votes: int, note: str = "n") -> list[dict[str, Any]]:
    return [{"state": state, "geo_id": geo_id, "votes": votes, "note": note} for geo_id in geo_ids]


async def test_a_state_folds_once_its_deltas_reach_half_its_part_and_not_before(
    tmp_path: Path, workers: ProcessWorkers
) -> None:
    """Flips if the half-size rule (decision 10) moves: a state folding before its deltas since its last fold reach
    half its part's size, or not folding once they do, or a state with no part waiting past its first delta."""
    writer = _writer(tmp_path, _votes_schema(), workers)
    # a part of varied rows, so it is many times the size of one small delta file
    await writer.commit(
        [{"state": "TX", "geo_id": g, "votes": g * 7919 % 100003, "note": f"note-{g * 31 % 997}"} for g in range(5000)]
    )
    part = _part(tmp_path, "votes", "state:TX")
    assert part is not None and part.rows == 5000, "a state with no part folds at its first delta"
    held = (tmp_path / part.file).stat().st_size
    since = 0
    waited = 0
    for step in range(40):
        file = writer.save(_votes("TX", range(1000 + step * 8, 1008 + step * 8), 2))
        since += (tmp_path / file).stat().st_size  # one state's delta: all of its bytes are the state's
        writer.tell(file)
        await writer.folds.wait()
        now = _part(tmp_path, "votes", "state:TX")
        assert now is not None
        if since * 2 >= held:
            assert now.file != part.file and now.version == f"inc:{len(writer.files)}", f"step {step}: due, not folded"
            break
        assert now.file == part.file, f"step {step}: folded at {since} bytes of a {held}-byte part"
        waited += 1
    else:
        pytest.fail("the deltas never reached half the part")
    assert waited >= 2, "the deltas were meant to take several commits to reach half the part"
    await writer.folds.close()


def _unfolded(metric: str, table: str = "votes") -> float | None:
    return REGISTRY.get_sample_value(f"delta_fold_unfolded_{metric}", {"table": table})


async def test_the_unfolded_gauges_rise_with_each_commit_and_fall_to_zero_once_folded(
    tmp_path: Path, workers: ProcessWorkers
) -> None:
    """Flips if the writer stops saying, per table, how many delta files and bytes its fold holds unfolded (gap 3:
    a fold that falls behind, or a state never due, shows here before a copy's start grows slow): the gauges must
    rise with each commit the fold keeps, by exactly the bytes it measured, and read zero once every one is folded."""
    writer = _writer(tmp_path, _votes_schema(), workers)
    await writer.commit(
        [{"state": "TX", "geo_id": g, "votes": g * 7919 % 100003, "note": f"note-{g * 31 % 997}"} for g in range(5000)]
    )
    assert _unfolded("files") == 0 and _unfolded("bytes") == 0, "a state with no part folds at once"
    part = _part(tmp_path, "votes", "state:TX")
    assert part is not None
    held = (tmp_path / part.file).stat().st_size
    since = 0
    for step in range(40):
        file = writer.save(_votes("TX", range(1000 + step * 8, 1008 + step * 8), 2))
        size = (tmp_path / file).stat().st_size
        writer.tell(file)
        await writer.folds.wait()
        if (since + size) * 2 >= held:
            break
        since += size
        assert _unfolded("files") == step + 1
        assert _unfolded("bytes") == since
    else:
        pytest.fail("the deltas never reached half the part")
    assert step >= 2, "the deltas were meant to take several commits to reach half the part"
    assert _unfolded("files") == 0 and _unfolded("bytes") == 0, "every delta folded"
    await writer.folds.close()


async def test_the_folded_part_is_every_delta_applied_to_the_old_part(tmp_path: Path, workers: ProcessWorkers) -> None:
    """Flips if the fold stops merging as L3 merged (decisions 9 and 19): a column a write left out losing its value, a
    delete marker leaving its row, a delta applied out of order, or one state's rows reaching another's part."""
    writer = _writer(tmp_path, _votes_schema(), workers)
    await writer.commit(_votes("TX", range(1, 4), 10, note="kept") + _votes("GA", range(1, 3), 7))
    old = _part(tmp_path, "votes", "state:TX")
    assert old is not None
    # three commits heard at once: one round folds them together, in commit order
    files = [
        writer.save([{"state": "TX", "geo_id": 1, "votes": 11}], written=["votes"]),
        writer.save(deletes=[("TX", 2)]),
        writer.save(
            [{"state": "TX", "geo_id": 9, "votes": 90, "note": "new"}, {"state": "TX", "geo_id": 1, "votes": 12}],
            written=["votes", "note"],
        ),
    ]
    expected = {(row["state"], row["geo_id"]): row for row in _part_rows(tmp_path, old)}
    changes = [change for file in files for change in read_delta_file(tmp_path / file).changes]
    for file in files:
        writer.tell(file)
    await writer.folds.wait()
    new = _part(tmp_path, "votes", "state:TX")
    assert new is not None and new.file != old.file and new.version == "inc:4"
    for change in changes:
        key = (change.row["state"], change.row["geo_id"])
        after = apply_delta_row(expected.get(key), change)
        if after is None:
            expected.pop(key, None)
        else:
            expected[key] = after
    rows = _part_rows(tmp_path, new)
    assert rows == sorted(expected.values(), key=lambda row: tuple(str(v) for v in row.values()))
    by_key = {row["geo_id"]: row for row in rows}
    assert by_key[1] == {"state": "TX", "geo_id": 1, "votes": 12, "note": None}, "the last write set note (to null)"
    assert 2 not in by_key, "the delete marker deleted its row"
    assert by_key[3]["note"] == "kept" and by_key[9]["note"] == "new"
    assert {row["state"] for row in rows} == {"TX"}
    ga = _part(tmp_path, "votes", "state:GA")
    assert ga is not None and [row["geo_id"] for row in _part_rows(tmp_path, ga)] == [1, 2]
    await writer.folds.close()


async def test_a_folded_part_after_a_prefix_replace_holds_what_the_copy_holds(
    tmp_path: Path, workers: ProcessWorkers
) -> None:
    """Flips if the fold keeps a row a cleared prefix took out of L3 (and out of every copy), drops one the load put
    back, takes a row of another state out, or never folds a delete that names a prefix and no row (decision 23): a
    load that replaced TX's rows deletes them first in a file naming the prefix, then loads the export's rows; a gone
    run of GA is a key-prefix delete alone. The part is what L3, and a copy applying the same files, hold."""
    writer = _writer(tmp_path, _votes_schema(), workers)
    await writer.commit(
        _votes("TX", range(1, 6), 10, note="old") + _votes("GA", range(1, 3), 7) + _votes("NC", range(1, 2), 1)
    )
    types = rail_arrow_types(writer.schema.columns)

    def cleared(prefix: dict[str, Any]) -> str:
        writer.count += 1
        delta = TableDelta(name=writer.schema.name, types=types, key=tuple(writer.schema.pk_columns))
        delta.clears([prefix])
        file = write_delta_file(tmp_path, delta, f"w{writer.count:04d}")
        writer.files.append(file)
        return file

    files = [
        cleared({"state": "TX"}),
        # the load's rows: a COPY inserts, setting no column of a row held
        writer.save(_votes("TX", range(2, 3), 20, note="again") + _votes("TX", range(8, 9), 80), written=[]),
        cleared({"state": "GA"}),
    ]
    for file in files:
        writer.tell(file)
    await writer.folds.wait()
    tx = _part(tmp_path, "votes", "state:TX")
    assert tx is not None and tx.version == "inc:3", "TX's last share is the load's"
    assert _part_rows(tmp_path, tx) == [
        {"state": "TX", "geo_id": 2, "votes": 20, "note": "again"},
        {"state": "TX", "geo_id": 8, "votes": 80, "note": "n"},
    ]
    ga = _part(tmp_path, "votes", "state:GA")
    assert ga is not None and _part_rows(tmp_path, ga) == [], "the delete of GA's prefix folded"
    nc = _part(tmp_path, "votes", "state:NC")
    assert nc is not None and [row["geo_id"] for row in _part_rows(tmp_path, nc)] == [1]
    await writer.folds.close()


async def test_a_left_out_column_keeps_its_value_across_a_fold(tmp_path: Path, workers: ProcessWorkers) -> None:
    """Flips if a write that set only some columns overwrites the others with the file's nulls (decision 19)."""
    writer = _writer(tmp_path, _votes_schema(), workers)
    await writer.commit(_votes("TX", range(1, 3), 10, note="kept"))
    await writer.commit(
        [{"state": "TX", "geo_id": 1, "votes": 99}, {"state": "TX", "geo_id": 2, "votes": 98}], written=["votes"]
    )
    part = _part(tmp_path, "votes", "state:TX")
    assert part is not None and part.version == "inc:2"
    assert [(row["votes"], row["note"]) for row in _part_rows(tmp_path, part)] == [(99, "kept"), (98, "kept")]
    await writer.folds.close()


async def test_the_keep_rule_at_each_fold_keeps_what_it_keeps_of_every_row(
    tmp_path: Path, workers: ProcessWorkers
) -> None:
    """Flips if the fold stops applying the table's keep rule (decision 13), or applies it so that the full differs
    from the rule over every row the deltas folded so far wrote: the rows a copy keeping by the same rule holds."""
    schema = _runs_schema(_NewestTwoRuns)
    writer = _writer(tmp_path, schema, workers)
    written: list[list[dict[str, Any]]] = []
    folds_seen = set()
    for run in range(1, 13):
        rows = [{"state": "TX", "race": race, "run": run, "votes": run * 10 + i} for i, race in enumerate("abc")]
        if run % 3:
            rows = rows[: run % 3]
        written.append(rows)
        await writer.commit(rows)
        part = _part(tmp_path, "runs", "state:TX")
        assert part is not None
        folds_seen.add(part.file)
        through = int(part.version.split(":")[1])
        every = pa.Table.from_pylist([row for rows_ in written[:through] for row in rows_])
        kept = _NewestTwoRuns.keep(every).to_pylist()
        assert _part_rows(tmp_path, part) == sorted(kept, key=lambda row: tuple(str(v) for v in row.values()))
    assert len(folds_seen) >= 3, "the test must cross several folds"
    await writer.folds.close()


async def test_a_delta_that_replaces_its_part_whole_is_named_as_the_part(
    tmp_path: Path, workers: ProcessWorkers
) -> None:
    """Flips if decision 16 goes: a delta holding only its state that the keep rule makes the whole part copied to a
    new part file instead of named; or a delta named as a part deleted as covered while a manifest in use names it."""
    writer = _writer(tmp_path, _runs_schema(_NewestRun), workers)
    parts_dir = tmp_path / ".fulls" / "parts" / "runs"
    named: list[str] = []
    for run in (1, 2, 3):
        file = await writer.commit([{"state": "TX", "race": race, "run": run, "votes": run} for race in "abc"])
        part = _part(tmp_path, "runs", "state:TX")
        assert part is not None and part.file == file and part.rows == 3, f"run {run}: the delta is the part"
        assert not parts_dir.exists() or not any(parts_dir.iterdir()), "no row was copied to a part file"
        named.append(file)
    # the current manifest names run 3's, the previous run 2's: both kept; run 1's is named by neither and goes
    assert (tmp_path / named[2]).exists() and (tmp_path / named[1]).exists()
    assert not (tmp_path / named[0]).exists()
    await writer.folds.close()


async def test_a_delta_of_two_states_is_never_named_as_a_part(tmp_path: Path, workers: ProcessWorkers) -> None:
    """Flips if a delta holding another state's rows is named as a state's part: the part would load them."""
    writer = _writer(tmp_path, _runs_schema(_NewestRun), workers)
    await writer.commit([{"state": state, "race": "a", "run": 1, "votes": 1} for state in ("TX", "GA")])
    for state in ("TX", "GA"):
        part = _part(tmp_path, "runs", f"state:{state}")
        assert part is not None and part.file.startswith(".fulls/parts/runs/") and part.rows == 1
    await writer.folds.close()


async def test_covered_deltas_and_unnamed_files_are_deleted_and_the_previous_full_kept(
    tmp_path: Path, workers: ProcessWorkers
) -> None:
    """Flips if the fold deletes a delta a state has not folded past, a part the current or previous manifest names,
    or leaves covered deltas, older parts, older manifests or orphans on the volume."""
    writer = _writer(tmp_path, _votes_schema(), workers)
    await writer.commit(_votes("GA", range(2000), 1))
    ga = _part(tmp_path, "votes", "state:GA")
    assert ga is not None
    # one delta holding both states: TX (no part) folds, GA (a large part) does not, so the delta stays
    shared = await writer.commit(_votes("TX", range(3), 1) + _votes("GA", range(3), 2))
    tx1 = _part(tmp_path, "votes", "state:TX")
    assert tx1 is not None and _part(tmp_path, "votes", "state:GA") == ga
    assert (tmp_path / shared).exists(), "GA has not folded past it"
    # an orphan: saved by this process for a commit that did not land; and a file another process saved, which this
    # one never named (stamped long ago by the volume: no clock compare may take it)
    orphan = writer.save(_votes("TX", range(5), 5))
    writer.files.remove(orphan)
    OWN_DELTA_SAVES.settled([orphan], landed=False)
    older = writer.save(_votes("TX", range(5), 6))
    writer.files.remove(older)
    os.utime(tmp_path / older, (time.time() - 3600, time.time() - 3600))
    # two of TX's three rows: the delta is not the whole part, so a new part file is written
    first = await writer.commit(_votes("TX", range(2), 3))
    tx2 = _part(tmp_path, "votes", "state:TX")
    assert tx2 is not None and tx2.file != tx1.file
    assert not (tmp_path / first).exists(), "a delta the full covers is deleted"
    assert (tmp_path / shared).exists(), "GA still holds it"
    assert (tmp_path / tx1.file).exists(), "the previous manifest's part is kept"
    assert not (tmp_path / orphan).exists(), "this process's orphan is swept"
    assert orphan not in OWN_DELTA_SAVES.orphans, "and forgotten"
    assert (tmp_path / older).exists(), "a file this process never saved is left"
    await writer.commit(_votes("TX", range(3), 4))
    tx3 = _part(tmp_path, "votes", "state:TX")
    assert tx3 is not None
    assert not (tmp_path / tx1.file).exists(), "a part neither the current nor the previous manifest names goes"
    assert (tmp_path / tx2.file).exists() and (tmp_path / tx3.file).exists() and (tmp_path / ga.file).exists()
    manifests = sorted(p.name for p in (tmp_path / ".fulls" / "manifests").iterdir())
    current = current_full_manifest(tmp_path)
    assert current is not None and current.previous is not None
    assert manifests == sorted([current.previous, current.name])
    # GA folds once its deltas reach half its part: the shared delta is then covered and goes
    await writer.commit(_votes("GA", range(3000, 5000), 9))
    assert _part(tmp_path, "votes", "state:GA") != ga
    assert not (tmp_path / shared).exists()
    await writer.folds.close()


async def test_a_stale_writer_s_fold_publishes_nothing(tmp_path: Path, workers: ProcessWorkers) -> None:
    """Flips if the term fence goes (decision 15): a writer whose lease another took (a later write's version is the
    current full's term) publishing its fold, or deleting a file of the current writer's full."""
    writer = _writer(tmp_path, _votes_schema(), workers)
    await writer.commit(_votes("TX", range(10), 1))
    ours = current_full_manifest(tmp_path)
    assert ours is not None and ours.term == writer.version
    # the next writer took the lease: its begin's version is above every one this writer took
    theirs = publish_full_manifest(tmp_path, term=writer.version + 10, tables=dict(ours.tables))
    file = await writer.commit(_votes("TX", range(10), 2))
    assert current_full_manifest(tmp_path) == theirs
    assert (tmp_path / file).exists() and all((tmp_path / name).exists() for name in theirs.files())
    await writer.folds.close()


async def test_the_term_is_the_write_s_version(tmp_path: Path, workers: ProcessWorkers) -> None:
    """Flips if the fold publishes at any term but the version of the write it folds (ScopeEpochs.begin's, taken under
    the write lease): a later writer's full is then not the one readers take."""
    writer = _writer(tmp_path, _votes_schema(), workers)
    writer.version = 41
    await writer.commit(_votes("TX", range(10), 1))
    first = current_full_manifest(tmp_path)
    assert first is not None and (first.term, first.seq) == (41, 1)
    writer.version = 57
    await writer.commit(_votes("GA", range(10), 1))
    second = current_full_manifest(tmp_path)
    assert second is not None and (second.term, second.seq) == (57, 1) and second.previous == first.name
    await writer.folds.close()


async def test_the_fold_runs_off_the_serving_process(tmp_path: Path, workers: ProcessWorkers) -> None:
    """Flips if the fold's merge and keep rule run in the serving process instead of a worker process."""
    writer = _writer(tmp_path, _votes_schema(_MarkedByWorker), workers)
    await writer.commit(_votes("TX", range(3), 1))
    part = _part(tmp_path, "votes", "state:TX")
    assert part is not None
    pids = {row["votes"] for row in _part_rows(tmp_path, part)}
    assert len(pids) == 1 and os.getpid() not in pids
    await writer.folds.close()


async def test_a_fold_that_fails_never_fails_the_write_and_is_retried_at_the_next_commit(
    tmp_path: Path, workers: ProcessWorkers
) -> None:
    """Flips if a fold's failure reaches the writer (the commit call raising or waiting on it), or loses the files of
    the round that failed instead of folding them at the next commit."""
    writer = _writer(tmp_path, _votes_schema(), workers)
    blocked = tmp_path / ".fulls" / "parts" / "votes"
    blocked.parent.mkdir(parents=True)
    blocked.write_bytes(b"not a directory")  # no part can be written
    # two states' rows: no delta can be named as a part, so the fold must write one
    failed = writer.save(_votes("TX", range(3), 1) + _votes("GA", range(3), 1))
    writer.tell(failed)  # returns at once: the fold runs in the background
    await writer.folds.wait()
    assert _part(tmp_path, "votes", "state:TX") is None and (tmp_path / failed).exists()
    blocked.unlink()
    await writer.commit(_votes("TX", range(3, 6), 2))
    part = _part(tmp_path, "votes", "state:TX")
    assert part is not None and sorted(row["geo_id"] for row in _part_rows(tmp_path, part)) == list(range(6))
    assert not (tmp_path / failed).exists()
    await writer.folds.close()


def test_a_folded_table_s_scope_must_be_in_its_key() -> None:
    """Flips if a table whose delete markers cannot name their state is folded per state."""
    with pytest.raises(ValueError, match="must be a key column"):
        FoldedTable.of(_votes_schema(), scope=("note", "note:"))
    with pytest.raises(ValueError, match="opts in to delta files"):
        FoldedTable.of(
            TableSchema(name="plain", primary_key="id", columns=[Column("id", STRING_TYPE)], l1_eviction=TEST_L1_TABLE)
        )


# --- the rebuild from L3 (decisions 12, 14 and 15) ---------------------------------------------------------------


class _L3:
    """L3 holding each table's rows, counting the statements each table's read asks; ``on_read`` runs at each one."""

    def __init__(self, tables: Mapping[str, list[dict[str, Any]]], keys: Mapping[str, Sequence[str]]) -> None:
        self.tables = {name: list(rows) for name, rows in tables.items()}
        self.keys = keys
        self.statements: Counter[str] = Counter()
        self.on_read: Any = None

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        if sql.startswith("SELECT version()"):
            return [{"v": "PostgreSQL 16.4"}]
        found = re.search(r'FROM "?([A-Za-z_]+)"?', sql)
        assert found is not None, sql
        name = found.group(1)
        self.statements[name] += 1
        if self.on_read is not None:
            self.on_read(name)
        key = self.keys[name]
        return sorted(self.tables[name], key=lambda row: tuple(row[column] for column in key))


class _Generations:
    """a writer's generation source: each table's current token; a table never written has none."""

    def __init__(self, tokens: Mapping[str, str]) -> None:
        self.tokens = dict(tokens)

    async def current(self, table_name: str) -> str:
        token = self.tokens.get(table_name)
        if token is None:
            raise GenerationUnavailableError(f"no write generation exists yet for {table_name!r}")
        return token

    async def advance(self, table_name: str) -> str | None:
        raise AssertionError("a rebuild advances nothing")


class _Log:
    """the NATS epoch log as a copy reads it: each advance's record, naming its commit's delta file."""

    def __init__(self) -> None:
        self.records: dict[tuple[str, str], EpochChange] = {}

    def append(self, table: str, token: str, delta_file: str | None) -> None:
        self.records[(table, token)] = EpochChange(
            table=table, token=token, rows=[ChangedRow(ids=["k"])], delta_file=delta_file
        )

    async def change(self, table_name: str, token: str) -> EpochChange | None:
        return self.records.get((table_name, token))

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        return await walk_by_count(self, table_name, after, upto)


def _runs(state: str, race: str, runs: range) -> list[dict[str, Any]]:
    return [{"state": state, "race": race, "run": run, "votes": run * 10} for run in runs]


def _held(root: Path, table: str) -> list[dict[str, Any]]:
    """every row the current full holds of ``table``."""
    manifest = current_full_manifest(root)
    assert manifest is not None
    return sorted(
        (row for part in manifest.tables[table].parts for row in _part_rows(root, part)),
        key=lambda row: tuple(str(v) for v in row.values()),
    )


def _sorted(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted((dict(row) for row in rows), key=lambda row: tuple(str(v) for v in row.values()))


_VOTES_ROWS = _votes("TX", range(5), 3) + _votes("GA", range(2), 4)
_RUNS_ROWS = _runs("TX", "pres", range(1, 4)) + _runs("GA", "sen", range(1, 3))


def _folds(root: Path, workers: ProcessWorkers) -> DeltaFolds:
    return DeltaFolds(
        root,
        [FoldedTable.of(_votes_schema(), scope=_SCOPE), FoldedTable.of(_runs_schema(_NewestRun), scope=_SCOPE)],
        workers=workers,
    )


def _l3() -> _L3:
    return _L3(
        {"votes": _VOTES_ROWS, "runs": _RUNS_ROWS}, {"votes": ("state", "geo_id"), "runs": ("state", "race", "run")}
    )


async def test_no_full_reads_each_table_from_l3_once_and_publishes_its_parts_at_the_term(
    tmp_path: Path, workers: ProcessWorkers
) -> None:
    """Flips if a writer that finds no full on the volume (a lost volume, a new environment) reads a table from L3 more
    or less than once, or publishes other than each table's rows kept by its keep rule, one part a state, current to
    the generation read before the rows, in a manifest at its term."""
    l3, folds = _l3(), _folds(tmp_path, workers)
    rebuilt = await folds.start(l3, _Generations({"votes": "inc:7", "runs": "inc:3"}), changes=_Log(), version=12)
    assert rebuilt == ("runs", "votes")
    assert l3.statements == Counter({"votes": 1, "runs": 1}), "one read of each table, nothing else"
    manifest = current_full_manifest(tmp_path)
    assert manifest is not None and (manifest.term, manifest.seq) == (12, 1)
    votes, runs = manifest.tables["votes"], manifest.tables["runs"]
    assert votes.version == "inc:7" and runs.version == "inc:3"
    assert sorted(part.scope or "" for part in votes.parts) == ["state:GA", "state:TX"]
    assert all(part.version == "inc:7" for part in votes.parts) and all(part.version == "inc:3" for part in runs.parts)
    assert _held(tmp_path, "votes") == _sorted(_VOTES_ROWS)
    assert _held(tmp_path, "runs") == _sorted(_runs("TX", "pres", range(3, 4)) + _runs("GA", "sen", range(2, 3)))
    assert not any((tmp_path / ".fulls" / "rebuild").rglob("*.arrow")), "the read's rows are removed once published"
    # the full now holds every table: the writer's next start reads nothing
    assert (
        await _folds(tmp_path, workers).start(
            l3, _Generations({"votes": "inc:7", "runs": "inc:3"}), changes=_Log(), version=13
        )
        == ()
    )
    assert l3.statements == Counter({"votes": 1, "runs": 1})
    await folds.close()


async def test_the_writer_reads_l3_whole_only_for_a_table_owed_a_rebuild(
    tmp_path: Path, workers: ProcessWorkers, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if the writer's whole read of a table (DeltaFolds._fresh_fulls's, through _cold_start_pages, declared
    cold_start, and _schema_change_pages, declared schema_change, at each level) is reached other than for a table no
    full holds, a reset, or a schema change: a full that holds every table is read nothing, a full lacking one table
    reads that table alone, and a table never written is rebuilt before the round its first commit starts, never
    folded from its deltas alone."""
    declarations: list[str] = []

    def declared(value: Any, *, kind: str, **_: Any) -> Any:
        declarations.append(kind)
        return value

    def kinds() -> list[str]:
        # each read is declared where it is made and again where it is called
        assert declarations[::2] == declarations[1::2], declarations
        return declarations[::2]

    monkeypatch.setattr(delta_fold, "whole_read", declared)
    l3, generations = _l3(), _Generations({"votes": "inc:7"})  # runs never written: no generation yet
    folds = _folds(tmp_path, workers)
    # a full lacking runs (newly opted in): votes is read, runs waits for its generation
    assert await folds.start(l3, generations, changes=_Log(), version=5) == ("votes",)
    assert kinds() == ["cold_start"] and l3.statements == Counter({"votes": 1})
    assert "runs" not in (current_full_manifest(tmp_path) or FullManifest(0, 0, None, {})).tables
    # runs' first commit: rebuilt from L3 before that commit's round, never folded onto nothing
    writer = _Writer(tmp_path, _runs_schema(_NewestRun), folds)
    file = writer.save(_runs("TX", "pres", range(9, 10)))
    l3.tables["runs"] = _RUNS_ROWS + _runs("TX", "pres", range(9, 10))
    generations.tokens["runs"] = "inc:1"
    writer.tell(file)
    await folds.wait()
    assert kinds() == ["cold_start", "cold_start"] and l3.statements == Counter({"votes": 1, "runs": 1})
    assert _held(tmp_path, "runs") == _sorted(_runs("TX", "pres", range(9, 10)) + _runs("GA", "sen", range(2, 3)))
    # a full holding every table: a later commit reads nothing
    await writer.commit(_runs("GA", "sen", range(5, 6)))
    assert len(kinds()) == 2
    # a schema change reads the table under its own declaration; a reset as a cold start
    folds.start_fresh("votes", why="schema_change")
    await folds.wait()
    folds.start_fresh("votes", why="reset")
    await folds.wait()
    assert kinds() == ["cold_start", "cold_start", "schema_change", "cold_start"]
    assert l3.statements == Counter({"votes": 3, "runs": 1})
    await folds.close()


async def test_a_stale_writer_s_rebuild_changes_nothing_readers_see(tmp_path: Path, workers: ProcessWorkers) -> None:
    """Flips if the term fence goes (decision 15): a writer whose lease a later writer took during its read
    publishing its rebuild, deleting any file of the later writer's full, or leaving its own parts or rows behind; or a
    writer whose term is already below the current full's reading L3 at all."""
    l3 = _l3()
    theirs: list[FullManifest] = []

    def later_writer_publishes(name: str) -> None:
        if not theirs:
            part = write_full_part(
                tmp_path,
                "votes",
                "state:VA",
                pa.table({"state": ["VA"], "geo_id": [1], "votes": [1], "note": ["x"]}),
                "inc:20",
            )
            theirs.append(
                publish_full_manifest(
                    tmp_path,
                    term=40,
                    tables={"votes": FullTable("inc:20", (part,)), "runs": FullTable("inc:20")},
                )
            )

    l3.on_read = later_writer_publishes
    folds = _folds(tmp_path, workers)
    assert await folds.start(l3, _Generations({"votes": "inc:7", "runs": "inc:3"}), changes=_Log(), version=12) == ()
    (published,) = theirs
    assert current_full_manifest(tmp_path) == published
    assert all((tmp_path / file).exists() for file in published.files())
    parts = {p.relative_to(tmp_path).as_posix() for p in (tmp_path / ".fulls" / "parts").rglob("*.arrow")}
    assert parts == published.files(), "the stale writer's parts are its own to delete"
    assert not any((tmp_path / ".fulls" / "rebuild").rglob("*.arrow"))
    assert sorted(p.name for p in (tmp_path / ".fulls" / "manifests").iterdir()) == [published.name]
    # a writer already below the current term reads nothing
    reads = sum(l3.statements.values())
    stale = _folds(tmp_path, workers)
    assert await stale.start(l3, _Generations({"votes": "inc:7", "runs": "inc:3"}), changes=_Log(), version=13) == ()
    assert sum(l3.statements.values()) == reads
    await folds.close()
    await stale.close()


async def test_a_reset_drops_the_table_s_files_and_rebuilds_it_at_the_writer_s_term(
    tmp_path: Path, workers: ProcessWorkers
) -> None:
    """Flips if a reset (or a schema change) of an opted-in table leaves its old deltas or parts in use or on the
    volume, folds its old deltas onto the old parts, or publishes other than the table read from L3 again at the
    writer's term; or if a commit after the rebuild's read is lost from the full."""
    l3, generations = _l3(), _Generations({"votes": "inc:0", "runs": "inc:3"})
    folds = _folds(tmp_path, workers)
    await folds.start(l3, generations, changes=_Log(), version=5)
    writer = _Writer(tmp_path, _votes_schema(), folds)
    writer.version = 5
    # a part of varied rows, many times the size of one small delta
    await writer.commit(
        [
            {"state": "TX", "geo_id": g, "votes": g * 7919 % 100003, "note": f"note-{g * 31 % 997}"}
            for g in range(100, 5000)
        ]
    )
    await writer.commit(_votes("GA", range(9), 2))
    # a delta of the old table still pending a fold: too small beside TX's part to be due
    pending = await writer.commit(_votes("TX", range(3), 3))
    old = current_full_manifest(tmp_path)
    assert old is not None
    old_parts = {part.file for part in old.tables["votes"].parts}
    old_deltas = [file for file in writer.files if (tmp_path / file).exists()]
    assert pending in old_deltas and pending not in old.files()
    # an operator resets votes and reloads it
    l3.tables["votes"] = _votes("NC", range(3), 8)
    generations.tokens["votes"] = f"inc:{len(writer.files)}"
    folds.start_fresh("votes", why="reset")
    await folds.wait()
    current = current_full_manifest(tmp_path)
    assert current is not None and current.term == writer.version and current.seq == old.seq + 1
    assert current.tables["votes"].version == f"inc:{len(writer.files)}"
    assert _held(tmp_path, "votes") == _sorted(_votes("NC", range(3), 8))
    assert current.tables["runs"] == old.tables["runs"], "another table's full is kept as it was"
    for file in old_deltas:
        named = file in old.files()
        assert (tmp_path / file).exists() == named, (
            f"{file}: a delta of the old table goes unless a manifest in use names it"
        )
    assert not ({part.file for part in current.tables["votes"].parts} & old_parts)
    # a commit after the reset folds onto the rebuilt parts; the old parts go once no manifest in use names them
    await writer.commit(_votes("NC", range(3, 5), 9))
    await writer.commit(_votes("NC", range(5, 400), 9))
    assert _held(tmp_path, "votes") == _sorted(_votes("NC", range(3), 8) + _votes("NC", range(3, 400), 9))
    assert not any((tmp_path / file).exists() for file in old_parts | set(old_deltas))
    await folds.close()


# --- a restarted writer re-learns its unfolded deltas (decision 24) ----------------------------------------------


async def test_a_restarted_writer_relearns_its_unfolded_deltas_from_the_epoch_log_and_folds_them_at_half(
    tmp_path: Path, workers: ProcessWorkers
) -> None:
    """Flips if a writer that restarts with deltas on the volume not yet folded (decision 24) does not re-learn
    exactly those from the epoch log (none already folded into a part, none it was never told), reads L3 for them, or
    folds them other than at the half-size point counted across the restart (decision 10): a writer that forgot them
    would fold late, and its part would lack their rows."""
    schema = _votes_schema()
    writer = _writer(tmp_path, schema, workers)
    log = _Log()

    async def commit(rows: list[dict[str, Any]]) -> str:
        file = await writer.commit(rows)
        log.append("votes", f"inc:{len(writer.files)}", file)
        return file

    await commit(
        [{"state": "TX", "geo_id": g, "votes": g * 7919 % 100003, "note": f"note-{g * 31 % 997}"} for g in range(5000)]
    )
    await commit(_votes("GA", range(3), 1))
    part = _part(tmp_path, "votes", "state:TX")
    assert part is not None and part.version == "inc:1"
    held = (tmp_path / part.file).stat().st_size
    unfolded = [await commit(_votes("TX", range(1000 + step * 8, 1008 + step * 8), 2)) for step in range(3)]
    since = sum((tmp_path / file).stat().st_size for file in unfolded)
    assert since * 2 < held and _part(tmp_path, "votes", "state:TX") == part, "the deltas were meant to be pending"
    await writer.folds.close()

    # the writer restarts: a new fold, knowing nothing, re-learns from the log alone
    l3 = _L3({"votes": []}, {"votes": ("state", "geo_id")})
    writer.folds = DeltaFolds(tmp_path, [FoldedTable.of(schema, scope=_SCOPE)], workers=workers)
    assert await writer.folds.start(l3, _Generations({"votes": "inc:5"}), changes=log, version=6) == ()
    assert l3.statements == Counter(), "re-learned from the epoch log, never from L3"
    assert writer.folds.unfolded == set(unfolded), "exactly the unfolded deltas"
    assert _part(tmp_path, "votes", "state:TX") == part, "not due yet across the restart"
    for step in range(3, 40):
        file = writer.save(_votes("TX", range(1000 + step * 8, 1008 + step * 8), 2))
        since += (tmp_path / file).stat().st_size
        writer.tell(file)
        await writer.folds.wait()
        now = _part(tmp_path, "votes", "state:TX")
        assert now is not None
        if since * 2 >= held:
            assert now.file != part.file and now.version == f"inc:{len(writer.files)}", f"step {step}: due, not folded"
            rows = {(row["geo_id"], row["votes"]) for row in _part_rows(tmp_path, now)}
            assert {(g, 2) for g in range(1000, 1024)} <= rows, "the re-learned deltas are folded in"
            break
        assert now.file == part.file, f"step {step}: folded at {since} bytes of a {held}-byte part"
    else:
        pytest.fail("the deltas never reached half the part")
    assert l3.statements == Counter()
    await writer.folds.close()


# --- a folded table keyed without its state (decision 27: the state in a row column its markers carry) ----------


def _results_schema() -> TableSchema:
    """AP results as reports keeps them: keyed by the race, whose codes have no single shape a state can be read from;
    the state in a row column outside the key, which every delete marker carries (``delta_scope``)."""
    return TableSchema(
        name="results",
        primary_key=("office_key", "candidate"),
        columns=[
            Column("office_key", STRING_TYPE),
            Column("candidate", STRING_TYPE),
            Column("join_state_code", STRING_TYPE),
            Column("votes", BIGINT_TYPE, nullable=True),
        ],
        l1_eviction=TEST_L1_TABLE,
        delta_files=True,
        delta_scope="join_state_code",
    )


def _results(state: str, race: str, candidates: range, votes: int) -> list[dict[str, Any]]:
    return [{"office_key": race, "candidate": f"c{c}", "join_state_code": state, "votes": votes} for c in candidates]


async def test_a_table_keyed_without_its_state_files_a_delete_marker_under_the_state_it_carries(
    tmp_path: Path, workers: ProcessWorkers
) -> None:
    """Flips if a table keyed without its state (AP results, keyed by office_key, the state in join_state_code) cannot
    be folded per state by that row column: each state one part holding every race of it (from deltas and from the
    rebuild's rows), and a delete marker (its key and the state the writer named) folded into that state's part and
    no other (decision 27); or if a scope column outside the key that the markers do not carry is accepted."""
    schema = _results_schema()
    scope = ("join_state_code", "state:")
    races = {"VA-S": "VA", "X12": "VA", "GA-S": "GA"}  # race codes of no one shape: the state is the row's
    l3 = _L3(
        {"results": [row for race, state in races.items() for row in _results(state, race, range(2), 1)]},
        {"results": ("office_key", "candidate")},
    )
    folds = DeltaFolds(tmp_path, [FoldedTable.of(schema, scope=scope)], workers=workers)
    assert await folds.start(l3, _Generations({"results": "inc:0"}), changes=_Log(), version=5) == ("results",)
    manifest = current_full_manifest(tmp_path)
    assert manifest is not None
    assert sorted(p.scope or "" for p in manifest.tables["results"].parts) == ["state:GA", "state:VA"]
    va = _part(tmp_path, "results", "state:VA")
    assert va is not None and va.rows == 4, "both of Virginia's races in its one part"
    writer = _Writer(tmp_path, schema, folds)
    await writer.commit(_results("VA", "VA-S", range(2, 40), 3) + _results("VA", "X12", range(2, 40), 3))
    ga_before = _part(tmp_path, "results", "state:GA")
    await writer.commit(deletes=[("VA-S", "c0"), ("X12", "c1")], delete_scope="VA")
    await writer.commit(_results("VA", "X12", range(40, 200), 4))
    va = _part(tmp_path, "results", "state:VA")
    assert va is not None
    held = {(row["office_key"], row["candidate"]) for row in _part_rows(tmp_path, va)}
    assert ("VA-S", "c0") not in held and ("X12", "c1") not in held, "each marker folded under the state it carries"
    assert ("VA-S", "c1") in held and ("X12", "c199") in held
    assert all(races[race] == "VA" for race, _ in held)
    assert _part(tmp_path, "results", "state:GA") == ga_before, "no other state's part is touched"
    await folds.close()
    unmarked = dataclasses.replace(schema, delta_scope=None)
    with pytest.raises(ValueError, match="must be a key column"):
        FoldedTable.of(unmarked, scope=scope)


# --- a lost epoch log is recovered by key, never read whole (decision 28) ----------------------------------------


class _KeyedL3(ThreadedDuckDB):
    """L3 as a DuckDB, keeping every statement it answers and the rows each answered: a recovery's reads are keyed by
    the leftover files' keys and prefixes, never the table whole."""

    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        conn = duckdb.connect()
        conn.execute(
            "CREATE TABLE votes (state VARCHAR, geo_id BIGINT, votes BIGINT, note VARCHAR, PRIMARY KEY (state, geo_id))"
        )
        held = pa.Table.from_pylist(
            [{column: row.get(column) for column in ("state", "geo_id", "votes", "note")} for row in rows],
            schema=pa.schema(
                [("state", pa.string()), ("geo_id", pa.int64()), ("votes", pa.int64()), ("note", pa.string())]
            ),
        )
        conn.register("held", held)
        conn.execute("INSERT INTO votes SELECT * FROM held")
        conn.unregister("held")
        super().__init__(conn)
        self.db = conn
        self.statements: list[tuple[str, tuple[Any, ...], int]] = []

    async def fetch(self, query: str, /, *params: Any) -> list[dict[str, Any]]:
        rows = await super().fetch(query, *params)
        self.statements.append((query, params, len(rows)))
        return rows

    async def fetch_arrow(
        self, query: str, /, *params: Any, types: Mapping[str, Any] | None, size: int | None = None
    ) -> Any:
        rows = await super().fetch_arrow(query, *params, types=types, size=size)
        self.statements.append((query, params, rows.num_rows))
        return rows

    def held(self) -> list[dict[str, Any]]:
        columns = ("state", "geo_id", "votes", "note")
        held = self.db.execute("SELECT state, geo_id, votes, note FROM votes").fetchall()
        return _sorted([dict(zip(columns, row, strict=True)) for row in held])


def _recoveries(why: str) -> float:
    labels = {"table": "votes", "why": why, "outcome": "recovered"}
    return REGISTRY.get_sample_value("delta_fold_recoveries_total", labels) or 0.0


def _rebuilds(why: str) -> float:
    labels = {"table": "votes", "why": why, "outcome": "rebuilt"}
    return REGISTRY.get_sample_value("delta_fold_rebuilds_total", labels) or 0.0


def _no_whole_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """fail the test at the writer's whole read of a table (declared cold start or schema change), wherever made."""

    def refused(value: Any, *, kind: str, **_: Any) -> Any:
        raise AssertionError(f"a whole read of L3 ({kind}) where only keyed reads belong")

    monkeypatch.setattr(delta_fold, "whole_read", refused)


async def test_a_lost_log_is_recovered_from_the_leftover_files_keys_read_from_l3_by_key(
    tmp_path: Path, workers: ProcessWorkers, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a writer restarting with its epoch log lost (decision 28) reads L3 whole, drops its full, or reads any
    row from L3 but those the delta files left on the volume name; or if the recovered full is not what L3 holds. The
    leftovers: unfolded updates and a delete of TX, a load that replaced GA's rows (a file clearing the prefix, read by
    a key-led read of it), and a file whose commit never landed (its rows read back as L3 has them, its new key a
    delete marker). The full is kept (NC, which no leftover names, keeps its part file), every key and prefix read in
    one keyed statement each, the result equals L3, the leftovers are deleted, and the recovery is counted."""
    schema = _votes_schema()
    writer = _writer(tmp_path, schema, workers)
    l3: dict[tuple[str, int], dict[str, Any]] = {}

    async def commit(rows: list[dict[str, Any]] = [], **kwargs: Any) -> str:  # noqa: B006 -- never mutated
        file = await writer.commit(rows, **kwargs)
        for row in rows:
            l3[(row["state"], row["geo_id"])] = {**l3.get((row["state"], row["geo_id"]), {}), **row}
        for state, geo_id in kwargs.get("deletes", ()):
            l3.pop((state, geo_id), None)
        return file

    await commit(
        [{"state": "TX", "geo_id": g, "votes": g * 7919 % 100003, "note": f"note-{g * 31 % 997}"} for g in range(5000)]
    )
    await commit(_votes("GA", range(1, 400), 7))
    await commit(_votes("NC", range(1, 400), 1))
    nc = _part(tmp_path, "votes", "state:NC")
    assert nc is not None
    leftovers = [
        await commit(_votes("TX", range(1000, 1008), 2)),
        await commit(deletes=[("TX", 5)]),
    ]
    # a load that replaced GA's rows, its prefix cleared then the export's rows, committed as the writer died: its
    # files are on the volume, never told to the fold
    writer.count += 1
    cleared = TableDelta(name="votes", types=rail_arrow_types(schema.columns), key=("state", "geo_id"))
    cleared.clears([{"state": "GA"}])
    leftovers.append(write_delta_file(tmp_path, cleared, f"w{writer.count:04d}"))
    writer.files.append(leftovers[-1])
    leftovers.append(writer.save(_votes("GA", range(500, 503), 9)))
    for key in [key for key in l3 if key[0] == "GA"]:
        l3.pop(key)
    l3.update({(row["state"], row["geo_id"]): row for row in _votes("GA", range(500, 503), 9)})
    # saved before a commit that never landed: L3 never took its rows
    leftovers.append(
        writer.save([{"state": "TX", "geo_id": 7, "votes": -1, "note": "never"}, *_votes("TX", range(9999, 10000), 5)])
    )
    assert all((tmp_path / file).exists() for file in leftovers), "none folded yet"
    await writer.folds.close()

    named = {
        (change.row["state"], change.row["geo_id"])
        for file in leftovers
        for change in read_delta_file(tmp_path / file).changes
    }
    # the writer restarts with the epoch log gone: nothing names the files left on the volume
    _no_whole_read(monkeypatch)
    before, rebuilt_before = _recoveries("log_lost"), _rebuilds("cold_start")
    reader = _KeyedL3(list(l3.values()))
    folds = DeltaFolds(tmp_path, [FoldedTable.of(schema, scope=_SCOPE)], workers=workers)
    token = f"inc:{len(writer.files)}"
    assert await folds.start(reader, _Generations({"votes": token}), changes=_Log(), version=6) == ("votes",)

    assert _recoveries("log_lost") == before + 1 and _rebuilds("cold_start") == rebuilt_before
    keyed = sorted(named - {key for key in named if key[0] == "GA"})
    (by_key, by_prefix) = reader.statements
    # each key column's values bound as an array (twice: the key's index is read by them)
    assert sorted(zip(by_key[1][0], by_key[1][1], strict=True)) == sorted((s, str(g)) for s, g in keyed), (
        "one statement naming exactly the leftover files' keys outside a cleared prefix"
    )
    assert by_key[2] == sum(1 for key in keyed if key in l3), "it answered the keys L3 holds, no other row"
    assert by_prefix[1][0] == ["GA"] and by_prefix[2] == 3, "the cleared prefix read by a key-led read of it alone"
    assert all('"votes"' in sql and "WHERE" in sql for sql, _, _ in reader.statements)
    assert _held(tmp_path, "votes") == reader.held(), "the recovered full is what L3 holds"
    current = current_full_manifest(tmp_path)
    assert current is not None and current.term == 6
    assert all(part.version == token for part in current.tables["votes"].parts)
    assert _part(tmp_path, "votes", "state:NC") == FullPart("state:NC", nc.file, token, nc.rows), "NC's part is kept"
    assert not any((tmp_path / file).exists() for file in leftovers), "the leftover files are deleted"
    assert not folds.unfolded
    # folding resumes on the recovered full
    folds.committed(
        {"votes": writer.save(_votes("TX", range(1000, 1004), 3))}, {"votes": f"inc:{len(writer.files)}"}, version=6
    )
    await folds.wait()
    assert len(reader.statements) == 2, "a commit after the recovery reads nothing"
    await folds.close()


async def test_a_new_incarnation_is_recovered_by_key_and_never_read_whole(
    tmp_path: Path, workers: ProcessWorkers, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a commit of a new incarnation of a table's generation (its store replaced: the log lost) is refused by
    every round, or recovered other than by key (decision 28): the full kept, the unfolded delta of the old log read
    back by its keys, the parts current to the new incarnation, later commits folded onto them; and the same for a
    writer restarting onto a new incarnation."""
    _no_whole_read(monkeypatch)
    schema = _votes_schema()
    writer = _writer(tmp_path, schema, workers)
    rows = [
        {"state": "TX", "geo_id": g, "votes": g * 7919 % 100003, "note": f"note-{g * 31 % 997}"} for g in range(5000)
    ]
    await writer.commit(rows)
    pending = await writer.commit(_votes("TX", range(3), 3))
    assert writer.folds.unfolded == {pending}
    held = {(row["state"], row["geo_id"]): row for row in rows + _votes("TX", range(3), 3)}
    reader = _KeyedL3(list(held.values()))
    generations = _Generations({"votes": "new:1"})
    await writer.folds.close()
    folds = DeltaFolds(tmp_path, [FoldedTable.of(schema, scope=_SCOPE)], workers=workers)
    log = _Log()
    for index, file in enumerate(writer.files):
        log.append("votes", f"inc:{index + 1}", file)
    before = _recoveries("new_incarnation")
    # a writer restarting onto the new incarnation: the full's version names a log no record orders
    assert await folds.start(reader, generations, changes=log, version=6) == ("votes",)
    assert _recoveries("new_incarnation") == before + 1
    ((_, params, answered),) = reader.statements
    assert sorted(params[1]) == ["0", "1", "2"] and answered == 3, "the leftover delta's three keys, read by key"
    current = current_full_manifest(tmp_path)
    assert current is not None and current.tables["votes"].version == "new:1"
    assert all(part.version == "new:1" for part in current.tables["votes"].parts)
    assert _held(tmp_path, "votes") == reader.held()
    assert not (tmp_path / pending).exists()
    # a commit of yet another incarnation during the run: recovered the same way, before it is folded
    reader.db.execute("UPDATE votes SET votes = 77 WHERE geo_id = 4")
    generations.tokens["votes"] = "again:1"
    later = writer.save([{"state": "TX", "geo_id": 4, "votes": 77}], written=["votes"])
    folds.committed({"votes": later}, {"votes": "again:1"}, version=6)
    await folds.wait()
    assert _recoveries("new_incarnation") == before + 2
    assert reader.statements[-1][1][:2] == (["TX"], ["4"]), "the new incarnation's delta read back by its one key"
    current = current_full_manifest(tmp_path)
    assert current is not None and current.tables["votes"].version == "again:1"
    assert _held(tmp_path, "votes") == reader.held()
    # a later commit of that incarnation folds onto the recovered parts with no read
    reads = len(reader.statements)
    folds.committed({"votes": writer.save(_votes("NC", range(3, 400), 9))}, {"votes": "again:2"}, version=6)
    await folds.wait()
    assert len(reader.statements) == reads
    assert {(row["state"], row["geo_id"]) for row in _held(tmp_path, "votes") if row["state"] == "NC"} == {
        ("NC", g) for g in range(3, 400)
    }
    await folds.close()


async def test_a_lost_file_handed_over_by_decision_29_is_recovered_once_with_every_part_current_past_its_epoch(
    tmp_path: Path, workers: ProcessWorkers, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if the writer's fold does not take decision 29's hand-off (DeltaFolds.recover_table, the
    LostDeltaRecovery the recreator calls for a commit too big to list whose file was lost): if two asks recover the
    table twice, if the recovery reads the damaged file or L3 whole, or if any part of the table (GA's too, which no
    change since its fold touched) ends below the lost file's epoch, which a copy waiting for that file needs before
    it takes the full; or if the damaged file is left on the volume."""
    _no_whole_read(monkeypatch)
    schema = _votes_schema()
    writer = _writer(tmp_path, schema, workers)
    rows = [{"state": "TX", "geo_id": g, "votes": g % 97, "note": "n"} for g in range(5000)]
    await writer.commit(rows)
    await writer.commit(_votes("GA", range(1, 400), 7))
    await writer.commit([{"state": "NC", "geo_id": g, "votes": g % 89, "note": "n"} for g in range(5000)])
    ga = _part(tmp_path, "votes", "state:GA")
    assert ga is not None
    lost = await writer.commit(_votes("TX", range(1000, 1003), 2))
    other = await writer.commit(_votes("NC", range(5000, 5002), 3))
    assert writer.folds.unfolded == {lost, other}
    await writer.folds.close()
    damaged = bytearray((tmp_path / lost).read_bytes())
    damaged[len(damaged) // 2] ^= 0x5A
    (tmp_path / lost).write_bytes(bytes(damaged))
    held = {(row["state"], row["geo_id"]): row for row in [*rows, *_votes("GA", range(1, 400), 7)]}
    held.update(
        {
            (r["state"], r["geo_id"]): r
            for r in [
                *[{"state": "NC", "geo_id": g, "votes": g % 89, "note": "n"} for g in range(5000)],
                *_votes("NC", range(5000, 5002), 3),
            ]
        }
    )
    held.update({(r["state"], r["geo_id"]): r for r in _votes("TX", range(1000, 1003), 2)})
    reader = _KeyedL3(list(held.values()))
    log = _Log()
    for index, file in enumerate(writer.files):
        log.append("votes", f"inc:{index + 1}", file)
    epoch = f"inc:{writer.files.index(lost) + 1}"
    token = f"inc:{len(writer.files)}"  # a commit since the lost one: the generation read is past its epoch
    folds = DeltaFolds(tmp_path, [FoldedTable.of(schema, scope=_SCOPE)], workers=workers)
    assert await folds.start(reader, _Generations({"votes": token}), changes=log, version=6) == ()
    await folds.wait()
    before = (
        REGISTRY.get_sample_value(
            "delta_fold_recoveries_total", {"table": "votes", "why": "file_lost", "outcome": "recovered"}
        )
        or 0.0
    )

    await folds.recover_table("votes", epoch=epoch, file=lost)
    await folds.recover_table("votes", epoch=epoch, file=lost)  # a second copy's ask joins the first
    await folds.wait()

    after = REGISTRY.get_sample_value(
        "delta_fold_recoveries_total", {"table": "votes", "why": "file_lost", "outcome": "recovered"}
    )
    assert after == before + 1, "two asks, one recovery"
    current = current_full_manifest(tmp_path)
    assert current is not None
    entry = current.tables["votes"]
    assert {part.scope for part in entry.parts} == {"state:TX", "state:GA", "state:NC"}
    assert all(part.version == token for part in entry.parts), "every part, GA's untouched one too, is current"
    assert full_table_floor("votes", entry) == token
    gaps = [part for part in entry.parts if part.scope == "state:GA"]
    assert gaps[0].file == ga.file, "GA's rows are kept as they were, only its version moves"
    ((_, params, _),) = reader.statements
    assert sorted(zip(params[0], params[1], strict=True)) == [("NC", "5000"), ("NC", "5001")], (
        "the leftover file's keys alone, never the damaged file's"
    )
    assert not (tmp_path / lost).exists(), "the damaged file is deleted once the recovery is published"
    assert not folds.unfolded
    await folds.close()


async def test_no_full_at_all_with_the_log_lost_is_still_the_one_cold_start_whole_read(
    tmp_path: Path, workers: ProcessWorkers
) -> None:
    """Flips if a writer that finds no full on the volume (decision 14: only L3 is left) recovers by key instead of
    reading each table whole once: with no full, no leftover file can say what the table holds."""
    l3, folds = _l3(), _folds(tmp_path, workers)
    (tmp_path / "votes").mkdir()
    orphan = TableDelta(name="votes", types=rail_arrow_types(_votes_schema().columns), key=("state", "geo_id"))
    orphan.rows(_votes("TX", range(2), 1), ["votes", "note"])
    write_delta_file(tmp_path, orphan, "left")
    before, rebuilt = _recoveries("log_lost"), _rebuilds("cold_start")
    assert await folds.start(l3, _Generations({"votes": "inc:7", "runs": "inc:3"}), changes=_Log(), version=12) == (
        "runs",
        "votes",
    )
    assert l3.statements == Counter({"votes": 1, "runs": 1}), "one whole read of each table"
    assert _recoveries("log_lost") == before and _rebuilds("cold_start") == rebuilt + 1
    assert _held(tmp_path, "votes") == _sorted(_VOTES_ROWS)
    await folds.close()


# --- a lost file of a commit too big to list (decision 31) ------------------------------------------------------------

_BRACKET = (
    datetime(2026, 11, 23, 3, 0, tzinfo=UTC),
    datetime(2026, 11, 23, 3, 0, 2, tzinfo=UTC),
)


class _NotedL3(_KeyedL3):
    """L3 as the rail's proxy is to the writer: it reads the change notes too (``l3.noted_keys``), naming ``noted``
    while the ring covers the bracket, and ``not_covered`` once it does not."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], noted: Sequence[tuple[str, str]], *, covered: bool) -> None:
        super().__init__(rows)
        self.noted = tuple(noted)
        self.covered = covered
        self.asked: list[tuple[str, tuple[datetime, datetime]]] = []

    async def noted_keys(
        self,
        table: str,
        *,
        noted_between: tuple[datetime, datetime],
        cursor: NotedKeysCursor | None,
        limit: int,
        max_bytes: int,
    ) -> NotedKeysPage | str:
        self.asked.append((table, noted_between))
        if not self.covered:
            return "not_covered"
        # two pages, the second past the first's cursor, as the ring answers a bracket's keys
        half = len(self.noted) // 2
        if cursor is None:
            return NotedKeysPage(keys=self.noted[:half], whole_table=False, next=NotedKeysCursor(slot=0, after=_AT))
        return NotedKeysPage(keys=self.noted[half:], whole_table=False, next=None)


_AT = _BRACKET[0]


def _lost_file_recoveries(why: str) -> float:
    labels = {"table": "votes", "why": why}
    return REGISTRY.get_sample_value("delta_fold_lost_file_recoveries_total", labels) or 0.0


async def _big_commit_lost(
    root: Path, workers: ProcessWorkers
) -> tuple[_Writer, str, str, dict[tuple[str, int], dict[str, Any]], _Log]:
    """a writer's history whose one commit too big to list (keys unknown: its record names its notes bracket, its
    states and its file) lost its file: TX, GA and NC folded; the big commit updated four TX rows, inserted one and
    deleted one, then NC took two rows in a commit of its own, unfolded. Returns the writer, the lost file and its
    epoch, what L3 holds, and the epoch log."""
    schema = _votes_schema()
    writer = _writer(root, schema, workers)
    tx = [{"state": "TX", "geo_id": g, "votes": g % 97, "note": "n"} for g in range(5000)]
    nc = [{"state": "NC", "geo_id": g, "votes": g % 89, "note": "n"} for g in range(5000)]
    await writer.commit(tx)
    await writer.commit(_votes("GA", range(1, 400), 7))
    await writer.commit(nc)
    big = [*_votes("TX", range(1000, 1004), 2, "big"), *_votes("TX", range(9000, 9001), 5, "new")]
    lost = await writer.commit(big, deletes=[("TX", 7)])
    other = await writer.commit(_votes("NC", range(5000, 5002), 3))
    assert writer.folds.unfolded == {lost, other}
    await writer.folds.close()
    damaged = bytearray((root / lost).read_bytes())
    damaged[len(damaged) // 2] ^= 0x5A
    (root / lost).write_bytes(bytes(damaged))
    held = {(row["state"], row["geo_id"]): row for row in [*tx, *_votes("GA", range(1, 400), 7), *nc]}
    held.update({(row["state"], row["geo_id"]): row for row in [*big, *_votes("NC", range(5000, 5002), 3)]})
    del held[("TX", 7)]
    log = _Log()
    epoch = f"inc:{writer.files.index(lost) + 1}"
    for index, file in enumerate(writer.files):
        log.append("votes", f"inc:{index + 1}", file)
    log.records[("votes", epoch)] = EpochChange(
        table="votes",
        token=epoch,
        keys_unknown=True,
        noted_between=_BRACKET,
        delta_file=lost,
        scopes=["state:TX"],
    )
    return writer, lost, epoch, held, log


#: the keys the big commit's notes name: its own four updates, its insert and its delete
_NOTED = (*(("TX", str(g)) for g in range(1000, 1004)), ("TX", "9000"), ("TX", "7"))


async def test_a_lost_big_file_within_the_notes_window_is_recovered_from_the_keys_its_notes_name(
    tmp_path: Path, workers: ProcessWorkers, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if the writer, handed a lost file of a commit too big to list whose notes bracket the ring still covers
    (decision 31 (a)), does not read that commit's keys from the change notes, reads any row from L3 but those keys and
    the leftover file's (one keyed statement, never a state or the table), or publishes a full that is not what L3
    holds (the big commit's updates, its insert and its delete included). On the code before decision 31 the
    recovery read only the leftover file's keys, and the full kept TX's old rows and the deleted one."""
    _no_whole_read(monkeypatch)
    writer, lost, epoch, held, log = await _big_commit_lost(tmp_path, workers)
    reader = _NotedL3(list(held.values()), _NOTED, covered=True)
    token = f"inc:{len(writer.files)}"
    folds = DeltaFolds(tmp_path, [FoldedTable.of(_votes_schema(), scope=_SCOPE)], workers=workers)
    assert await folds.start(reader, _Generations({"votes": token}), changes=log, version=6) == ()
    before = (_lost_file_recoveries("notes"), _lost_file_recoveries("states"))

    await folds.recover_table("votes", epoch=epoch, file=lost)
    await folds.wait()

    assert _held(tmp_path, "votes") == reader.held(), "the recovered full is what L3 holds"
    assert reader.asked == [("votes", _BRACKET)] * 2, "the record's bracket, both pages of its notes"
    ((sql, params, answered),) = reader.statements
    expected = sorted([("NC", "5000"), ("NC", "5001"), *_NOTED])
    assert sorted(zip(params[0], params[1], strict=True)) == expected, (
        "one keyed statement naming the leftover file's keys and the noted keys, nothing else"
    )
    assert answered == len(expected) - 1, "only those rows: the deleted key is not in L3"
    assert '"votes"' in sql and "WHERE" in sql
    assert (_lost_file_recoveries("notes"), _lost_file_recoveries("states")) == (before[0] + 1, before[1])
    current = current_full_manifest(tmp_path)
    assert current is not None and all(part.version == token for part in current.tables["votes"].parts)
    assert not (tmp_path / lost).exists()
    await folds.close()


async def test_a_lost_big_file_past_the_notes_window_is_recovered_by_reading_only_its_states(
    tmp_path: Path, workers: ProcessWorkers, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if the writer, handed a lost file of a commit too big to list whose notes the ring no longer covers
    (decision 31 (b)), reads anything from L3 but the leftover file's keys and the rows of the one state the record
    names (a keyset read led by the state, never GA's or NC's, never the table), or publishes a full that is not what
    L3 holds: TX's part is replaced whole by L3's rows (its deleted key gone, its insert there), GA's kept as it was.
    On the code before decision 31 the full kept TX's old rows and the deleted one."""
    _no_whole_read(monkeypatch)
    writer, lost, epoch, held, log = await _big_commit_lost(tmp_path, workers)
    ga = _part(tmp_path, "votes", "state:GA")
    assert ga is not None
    reader = _NotedL3(list(held.values()), _NOTED, covered=False)
    token = f"inc:{len(writer.files)}"
    folds = DeltaFolds(tmp_path, [FoldedTable.of(_votes_schema(), scope=_SCOPE)], workers=workers)
    assert await folds.start(reader, _Generations({"votes": token}), changes=log, version=6) == ()
    before = (_lost_file_recoveries("notes"), _lost_file_recoveries("states"))

    await folds.recover_table("votes", epoch=epoch, file=lost)
    await folds.wait()

    assert _held(tmp_path, "votes") == reader.held(), "the recovered full is what L3 holds"
    # a dialect probe (``SELECT version()``) reads no table
    (by_key, *by_state) = [statement for statement in reader.statements if '"votes"' in statement[0]]
    assert sorted(zip(by_key[1][0], by_key[1][1], strict=True)) == [("NC", "5000"), ("NC", "5001")]
    assert by_state, "the named state is read"
    for sql, params, _ in by_state:
        assert "WHERE" in sql and ["TX"] in [list(p) for p in params if isinstance(p, list | tuple)], (
            f"each page a read of the named state alone, never another: {sql} {params}"
        )
        assert re.search(r'ORDER BY "state"', sql), f"led by the state: {sql}"
    tx_rows = sum(1 for key in held if key[0] == "TX")
    # each page asks one row past itself, to know whether another follows
    assert tx_rows <= sum(answered for _, _, answered in by_state) <= tx_rows + len(by_state), "TX's rows alone, once"
    assert (_lost_file_recoveries("notes"), _lost_file_recoveries("states")) == (before[0], before[1] + 1)
    assert _part(tmp_path, "votes", "state:GA") == FullPart("state:GA", ga.file, token, ga.rows), "GA's part kept"
    assert not (tmp_path / lost).exists()
    await folds.close()


# --- the fold's bounds: a stop within its grace, a new term's start, a queue that never grows without end --------


class _SlowOnce(DeltaKeepRule):
    """keeps every row; the first round it runs in (no mark yet at the path its rows' note names) marks it, then runs
    for 30 s: a round long enough that waiting on it is what a test sees."""

    @classmethod
    def keep(cls, rows: Any) -> Any:
        mark = Path(rows.column("note")[0].as_py())
        if not mark.exists():
            mark.touch()
            time.sleep(30)
        return rows


async def _in_its_worker(mark: Path) -> None:
    """wait until the slow round is inside its worker's keep rule."""
    deadline = time.monotonic() + 30
    while not mark.exists():
        assert time.monotonic() < deadline, "the round never reached its worker"
        await asyncio.sleep(0.05)


async def test_close_within_a_bound_cuts_a_long_round_and_keeps_its_files_for_the_next_writer(
    tmp_path: Path, workers: ProcessWorkers
) -> None:
    """Flips if ``close(within=)`` waits past its bound for a round in progress (a pod's stop held for as long as a
    fold runs), loses the round's files instead of keeping them queued and on the volume, leaves its own worker
    running, or publishes the cut round's part; and if the next writer cannot fold what the cut round held."""
    schema = _votes_schema(_SlowOnce)
    _first_full(tmp_path, "votes")
    folds = DeltaFolds(tmp_path, [FoldedTable.of(schema, scope=_SCOPE)])  # its own worker
    writer = _Writer(tmp_path, schema, folds)
    mark = tmp_path / "slow.mark"
    file = writer.save(_votes("TX", range(3), 1, note=str(mark)))
    writer.tell(file)
    await _in_its_worker(mark)
    cut = REGISTRY.get_sample_value("delta_fold_rounds_cut_total", {"why": "stop"}) or 0.0

    began = time.monotonic()
    await folds.close(within=0.5)
    took = time.monotonic() - began

    assert took < 1.5, f"close(within=0.5) took {took:.2f} s"
    assert REGISTRY.get_sample_value("delta_fold_rounds_cut_total", {"why": "stop"}) == cut + 1
    assert folds.unfolded == {file} and (tmp_path / file).exists(), "the cut round's file is kept, queued"
    assert _part(tmp_path, "votes", "state:TX") is None, "the cut round published nothing"
    # the next writer folds it (the mark is there: its keep rule runs at once)
    after = DeltaFolds(tmp_path, [FoldedTable.of(schema, scope=_SCOPE)], workers=workers)
    after.committed({"votes": file}, {"votes": "inc:1"}, version=6)
    await after.wait()
    part = _part(tmp_path, "votes", "state:TX")
    assert part is not None and part.rows == 3
    await after.close()


async def test_a_newer_term_s_start_never_waits_on_this_process_s_round_of_an_older_term(tmp_path: Path) -> None:
    """Flips if a writer's start at a newer term (another writer wrote since this process's last holding) waits on
    this process's own round of the older term, which is stale (a session's first write held for as long as that fold
    runs); or if the cut round's file is lost instead of folded at the new term."""
    schema = _votes_schema(_SlowOnce)
    _first_full(tmp_path, "votes")
    folds = DeltaFolds(tmp_path, [FoldedTable.of(schema, scope=_SCOPE)])  # its own worker
    writer = _Writer(tmp_path, schema, folds)
    mark = tmp_path / "slow.mark"
    file = writer.save(_votes("TX", range(3), 1, note=str(mark)))
    writer.tell(file)  # a round at term 5
    await _in_its_worker(mark)
    log = _Log()
    log.append("votes", "inc:1", file)

    began = time.monotonic()
    rebuilt = await folds.start(_l3(), _Generations({"votes": "inc:1"}), changes=log, version=9)
    took = time.monotonic() - began

    assert rebuilt == ()
    assert took < 10, f"the start waited {took:.1f} s on the older term's round"
    assert REGISTRY.get_sample_value("delta_fold_rounds_cut_total", {"why": "new_term"}) is not None
    part = _part(tmp_path, "votes", "state:TX")
    assert part is not None and part.rows == 3, "the cut round's file is folded at the new term"
    current = current_full_manifest(tmp_path)
    assert current is not None and current.term == 9
    await folds.close()


async def test_straight_failed_rounds_hand_their_queue_to_a_recovery_by_key(
    tmp_path: Path, workers: ProcessWorkers, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if the fold's queue keeps every file of rounds that all fail (it grows with every commit while the volume
    refuses parts), or loses their changes: past three straight failures the queued table is owed a recovery by key,
    its queue let go (the gauge back to zero), and once the volume takes parts again the recovered full is what L3
    holds, counted as a publish."""
    _no_whole_read(monkeypatch)
    schema = _votes_schema()
    _first_full(tmp_path, "votes")
    # two states a commit: no delta can be named as a part, so each round must write one
    rows = [
        _votes("TX", range(3), 1) + _votes("GA", range(3), 1),
        _votes("GA", range(3), 2) + _votes("NC", range(2), 2),
        _votes("TX", range(2, 5), 3) + _votes("NC", range(1), 3),
        _votes("NC", range(2), 4) + _votes("TX", range(1), 4),
    ]
    held: dict[tuple[str, int], dict[str, Any]] = {}
    for chunk in rows:
        held.update({(row["state"], row["geo_id"]): row for row in chunk})
    l3 = _KeyedL3(list(held.values()))
    generations = _Generations({"votes": "inc:0"})
    folds = DeltaFolds(tmp_path, [FoldedTable.of(schema, scope=_SCOPE)], workers=workers)
    assert await folds.start(l3, generations, changes=_Log(), version=6) == ()
    writer = _Writer(tmp_path, schema, folds)
    writer.version = 6
    blocked = tmp_path / ".fulls" / "parts" / "votes"
    blocked.parent.mkdir(parents=True, exist_ok=True)
    blocked.write_bytes(b"not a directory")  # no part can be written: every round fails
    capped = REGISTRY.get_sample_value("delta_fold_queue_capped_total", {"table": "votes"}) or 0.0
    before = _recoveries("rounds_failed")
    for count, chunk in enumerate(rows[:3], start=1):
        file = writer.save(chunk)
        generations.tokens["votes"] = f"inc:{count}"
        writer.tell(file)
        await folds.wait()
        assert REGISTRY.get_sample_value("delta_fold_queued_files") == (count if count < 3 else 0)
    assert not folds.unfolded, "past three straight failures the queue is let go"
    assert REGISTRY.get_sample_value("delta_fold_queue_capped_total", {"table": "votes"}) == capped + 1
    blocked.unlink()
    published = folds.published
    file = writer.save(rows[3])
    generations.tokens["votes"] = "inc:4"
    writer.tell(file)
    await folds.wait()
    assert _recoveries("rounds_failed") == before + 1
    assert folds.published == published + 1
    assert _held(tmp_path, "votes") == l3.held(), "the recovered full holds every queued change"
    await folds.close()


async def test_a_recovery_builds_its_delta_in_a_worker_and_writes_l3_s_pages_beside_the_fulls(
    tmp_path: Path, workers: ProcessWorkers, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a recovery (decision 28) builds, compresses or writes its recovery delta in the serving process (its
    own file writer refuses here), or holds L3's pages in memory rather than writing each beside the fulls as it is
    read (a pages file is open beside every keyed read), or leaves its scratch files; the recovered full is what L3
    holds."""
    _no_whole_read(monkeypatch)
    schema = _votes_schema()
    writer = _writer(tmp_path, schema, workers)
    held: dict[tuple[str, int], dict[str, Any]] = {}
    for chunk in (_votes("TX", range(400), 1) + _votes("GA", range(300), 1), _votes("NC", range(200), 1)):
        await writer.commit(chunk)
        held.update({(row["state"], row["geo_id"]): row for row in chunk})
    # left on the volume, never told: an update of TX, a delete of GA, and a commit that never landed
    leftovers = [writer.save(_votes("TX", range(100, 150), 7)), writer.save(deletes=[("GA", 5)])]
    held.update({(row["state"], row["geo_id"]): row for row in _votes("TX", range(100, 150), 7)})
    held.pop(("GA", 5))
    leftovers.append(writer.save(_votes("NC", range(900, 905), 9)))
    await writer.folds.close()

    def refused(*_: Any, **__: Any) -> None:
        raise AssertionError("a delta file was written in the serving process")

    monkeypatch.setattr(delta_files, "write_volume_file", refused)
    spilled: list[bool] = []
    async def beside(*args: Any, **kwargs: Any) -> Any:
        spilled.append(any((tmp_path / ".fulls" / "rebuild" / "votes").glob("*.pages.arrow")))
        return await read_keyed_arrow(*args, **kwargs)

    monkeypatch.setattr(delta_fold, "read_keyed_arrow", beside)
    reader = _KeyedL3(list(held.values()))
    folds = DeltaFolds(tmp_path, [FoldedTable.of(schema, scope=_SCOPE)], workers=workers)
    token = f"inc:{len(writer.files)}"
    assert await folds.start(reader, _Generations({"votes": token}), changes=_Log(), version=6) == ("votes",)

    assert spilled and all(spilled), "every keyed read's pages go to a file beside the fulls"
    assert _held(tmp_path, "votes") == reader.held(), "the recovered full is what L3 holds"
    assert not any((tmp_path / ".fulls" / "rebuild").rglob("*.arrow")), "the recovery's own files are removed"
    assert not any((tmp_path / file).exists() for file in leftovers)
    await folds.close()
