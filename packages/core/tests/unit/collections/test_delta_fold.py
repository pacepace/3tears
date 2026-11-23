"""The writer's fold (:mod:`threetears.core.collections.delta_fold`, ``design-pod-delta-files.md`` decisions 8, 9, 10,
13, 15, 16 and 19): after each commit, off the serving process, each state whose deltas since its last fold reach half
its part gets a new part (old part plus deltas, merged as L3 merged them, then the table's keep rule), published in a
manifest at the writer's term; covered deltas and files no manifest in use names are deleted.
"""

from __future__ import annotations

import os
import re
import time
from collections import Counter
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pa_ipc = pytest.importorskip("pyarrow.ipc")
pc = pytest.importorskip("pyarrow.compute")

from threetears.core.collections.delta_files import (  # noqa: E402
    TableDelta,
    apply_delta_row,
    read_delta_file,
    write_delta_file,
)
from threetears.core.collections import delta_fold  # noqa: E402
from threetears.core.collections.delta_fold import DeltaFolds, FoldedTable  # noqa: E402
from threetears.core.collections.delta_keep import DeltaKeepRule  # noqa: E402
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
    publish_full_manifest,
    write_full_part,
)
from threetears.core.exceptions import GenerationUnavailableError  # noqa: E402
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
    ) -> str:
        """a commit's delta file, saved before the commit; not yet told to the fold."""
        self.count += 1
        delta = TableDelta(
            name=self.schema.name, types=rail_arrow_types(self.schema.columns), key=tuple(self.schema.pk_columns)
        )
        key = set(self.schema.pk_columns)
        delta.rows(
            list(rows),
            list(written) if written is not None else [c.name for c in self.schema.columns if c.name not in key],
        )
        delta.deletes(list(deletes))
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
    with pa.memory_map(str(root / part.file)) as source:
        rows = pa_ipc.open_file(source).read_all()
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
    # an orphan: saved after the fold started, never committed; and one older than the fold, which it never heard of
    orphan = writer.save(_votes("TX", range(5), 5))
    writer.files.remove(orphan)
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
    assert not (tmp_path / orphan).exists(), "an orphan saved since the fold started is swept"
    assert (tmp_path / older).exists(), "a file older than the fold is left"
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
    rebuilt = await folds.start(l3, _Generations({"votes": "inc:7", "runs": "inc:3"}), version=12)
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
        await _folds(tmp_path, workers).start(l3, _Generations({"votes": "inc:7", "runs": "inc:3"}), version=13) == ()
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
    assert await folds.start(l3, generations, version=5) == ("votes",)
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
    assert await folds.start(l3, _Generations({"votes": "inc:7", "runs": "inc:3"}), version=12) == ()
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
    assert await stale.start(l3, _Generations({"votes": "inc:7", "runs": "inc:3"}), version=13) == ()
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
    await folds.start(l3, generations, version=5)
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
