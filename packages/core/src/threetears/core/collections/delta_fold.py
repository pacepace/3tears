"""the writer's fold: each state's new part of a full, from its old part and the delta files since.

**Why.** A table that opts in to delta files (:attr:`~threetears.core.collections.schema_backed.TableSchema.delta_files`)
saves every commit's rows as a file a pod copy applies (:mod:`~threetears.core.collections.delta_files`), and a copy
starts from the full on the same volume (:mod:`~threetears.core.collections.volume_fulls`). Nothing expires on a timer,
so without a fold a restarted copy would apply every delta since the full was first written, and the volume would hold
every one. The writing load folds them: a change is kept until a newer full covers it (``design-pod-delta-files.md``,
decisions 2, 8, 9, 10, 13, 15, 16, 19 and "How it works", the writer's step 3).

**When a state folds** (decision 10). :meth:`DeltaFolds.committed` is told each commit's delta files after the commit.
A delta file holds every state its commit wrote (decision 6), so each file's bytes are shared among its states by their
rows. Once a state's bytes since its last fold reach half its part's size on the volume, the state folds; a state with
no part yet folds at its first delta.

**How a state folds** (decisions 9 and 19). The old part's rows, then each delta's rows of the state in commit order,
merged as L3 merged them (:func:`~threetears.core.collections.delta_files.apply_delta_row`: a column the write left out
keeps its value, ``date_created`` keeps the held row's, a delete marker deletes), the newest per key. A history table
(AP snapshots, LSF runs) has a key unique to each run or upload, so the same rule appends its rows. Then the table's
keep rule (decision 13, :class:`~threetears.core.collections.delta_keep.DeltaKeepRule`), the one its copy keeps by: a
table that declares none keeps every row.

**A delta as the part** (decision 16). When the last delta holds only the state and, after the keep rule, the folded
rows are exactly its own (a race's newest run replacing the last), the manifest names that delta file as the part and
no row is copied. Files never change once written, and :func:`~threetears.core.collections.volume_fulls.
deletable_full_files` never lists a delta a manifest in use names; the fold keeps such a delta in its list until none
does, then deletes it.

**Published by term** (decision 15). The new parts, then a manifest naming them and every part kept, at the writer's
term, through :func:`~threetears.core.collections.volume_fulls.publish_full_manifest`. The term is the version of the
writer's own write (:meth:`~threetears.core.collections.scope_epochs.ScopeEpochs.begin`): the fencing version taken
under the write lease, durable in L3 and only ever higher for a later writer, never the NATS KV's revision, which a
memory-only bucket resets. A writing load takes its lease and then begins, so it is the first ``begin`` after the
lease; a writer that lost its lease publishes an older term than the next writer's, which no reader takes, or is
refused (:class:`~threetears.core.collections.volume_fulls.StaleTermError`) and changes nothing.

**Deleted after the publish.** Each delta the new full covers (every state it held is folded past it), every part
and manifest no current or previous manifest names, and every orphan delta file (saved before a commit that never
happened, :func:`~threetears.core.collections.delta_files.sweep_orphan_delta_files`) of the tables folded, saved since
the fold started and before the writer's oldest work in flight: a file older than the fold may be a commit it never
heard of, so it is left.

**Never on the commit path.** :meth:`DeltaFolds.committed` only notes the files and starts a round in the background,
off the commit; each round runs in a worker process (:class:`~threetears.core.process_workers.ProcessWorkers`), since
reading, merging and writing parts is CPU, and hands back only names, versions and counts. A round that fails is
logged and counted (``delta_fold_rounds_total{outcome="failed"}``) and its files are kept for the round the next commit
starts; the write it followed has committed already and never fails for it.

**What it does not do.** A table the current full does not hold yet is not folded: its deltas wait for the writer's
rebuild from L3 to publish its first full (decision 14), since a full made from deltas alone would lack every row
written before them. A reset of a table (a new incarnation of its generation) is refused by the round, logged and
counted, for the same rebuild.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from threetears.observe import get_logger
from threetears.observe.metrics import counter

from threetears.core.collections.delta_files import (
    _DELETED_COLUMN,
    _WRITTEN_COLUMN,
    _WRITTEN_METADATA,
    DeltaChange,
    apply_delta_row,
    sweep_orphan_delta_files,
)
from threetears.core.collections.delta_keep import DeltaKeepRule
from threetears.core.collections.generation import split_generation_token
from threetears.core.collections.schema_backed import TableSchema, rail_arrow_types
from threetears.core.collections.volume_fulls import (
    FullPart,
    FullTable,
    StaleTermError,
    current_full_manifest,
    deletable_full_files,
    publish_full_manifest,
    write_full_part,
)
from threetears.core.process_workers import ProcessWorkers

try:
    import pyarrow as pa
    import pyarrow.ipc as pa_ipc

    _PYARROW = True
except ImportError:  # NOSILENT: the snapshot extra is optional; a fold refuses without it, saying so
    _PYARROW = False

__all__ = ["DeltaFolds", "FoldedTable"]

log = get_logger(__name__)

#: a round's outcome: ``folded`` (a full published), ``nothing_due`` (no state reached half its part), ``no_full`` (no
#: full on the volume holds the table yet: the writer's rebuild publishes one), ``stale`` (a later writer's term is
#: current: this writer's lease was taken), ``failed`` (an error; the round's files are kept for the next), and
#: ``no_generation`` (a commit's file came with no generation to version it by: it is not folded)
_ROUNDS = counter(
    "delta_fold_rounds_total",
    description="writer fold rounds over tables that opt in to delta files, by outcome; failed rounds keep their files "
    "for the round the next commit starts",
    label_names=("outcome",),
)

#: states folded, by table and how: ``written`` (a new part file) or ``reused`` (a delta file named as the part)
_STATES = counter(
    "delta_fold_states_total",
    description="states folded into a new part of a full, by table and how (written: a new part file; reused: the "
    "delta file named as the part, no row copied)",
    label_names=("table", "how"),
)

#: a state folds once the bytes of its deltas since its last fold reach this share of its part's size (decision 10:
#: half), as a numerator over :data:`_DUE_OF`, so the test is whole numbers: ``bytes * _DUE_OF >= part * _DUE_AT``
_DUE_AT: Final = 1
_DUE_OF: Final = 2

#: the start of every file the fulls write themselves (``volume_fulls``): a part named outside it is a delta file
_FULLS_PREFIX: Final = ".fulls/"

#: one (delta file, state) share a round carries: table, scope, file, generation token, the share's bytes
_Share = tuple[str, str | None, str, str, int]


@dataclass(frozen=True, slots=True)
class FoldedTable:
    """one table the writer folds, as its schema declares it.

    :ivar name: the table
    :ivar key: its key columns, in key order
    :ivar columns: its columns, in the order its delta files and parts hold them
    :ivar scope: the scope each row belongs to, ``(column, prefix)`` as the writer's seqlock names it (a row's state is
        ``prefix + str(row[column])``); one part a state. None: one part for the whole table
    :ivar keep: the rows a part keeps (:attr:`~threetears.core.collections.schema_backed.TableSchema.delta_keep`); None
        keeps every row
    """

    name: str
    key: tuple[str, ...]
    columns: tuple[str, ...]
    scope: tuple[str, str] | None = None
    keep: type[DeltaKeepRule] | None = None

    def __post_init__(self) -> None:
        missing = [column for column in self.key if column not in self.columns]
        if missing:
            raise ValueError(f"{self.name}: key columns {missing} are not among its columns")
        if self.scope is not None and self.scope[0] not in self.key:
            # a delete marker carries the key alone: a scope column outside the key leaves a delete's state unknown,
            # so no one state's part could be folded past it
            raise ValueError(
                f"{self.name}: its scope column {self.scope[0]!r} must be a key column, so each delete marker names "
                f"its state"
            )

    @classmethod
    def of(cls, schema: TableSchema, *, scope: tuple[str, str] | None = None) -> FoldedTable:
        """a table as its schema declares it: its key, its columns as its delta files type them, its keep rule.

        :param schema: the table's schema; it must opt in to delta files
        :ptype schema: TableSchema
        :param scope: the scope each row belongs to, ``(column, prefix)``, as its copy follows it
            (:attr:`~threetears.core.collections.copy_follower.FollowedTable.scope`); None folds the table whole
        :ptype scope: tuple[str, str] | None
        :return: the table
        :rtype: FoldedTable
        :raises ValueError: when the table does not opt in to delta files
        """
        if not schema.delta_files:
            raise ValueError(f"{schema.name}: only a table that opts in to delta files (delta_files=True) is folded")
        return cls(
            name=schema.name,
            key=tuple(schema.pk_columns),
            columns=tuple(rail_arrow_types(schema.columns)),
            scope=scope,
            keep=schema.delta_keep,
        )

    def plain(self) -> tuple[Any, ...]:
        """the table as a worker process is handed it: plain data, and its keep rule's class by name.

        :return: ``(name, key, columns, scope column, scope prefix, keep)``
        :rtype: tuple
        """
        column, prefix = self.scope if self.scope is not None else (None, None)
        return (self.name, self.key, self.columns, column, prefix, self.keep)


class DeltaFolds:
    """the writing load's fold of the tables that opt in to delta files: call :meth:`committed` after each commit.

    One a writer, made once and kept: it remembers each state's deltas since its last fold. Only the writer holding
    the write lease makes commits, so only it folds; no other lock.

    :param root: the delta files' root (``delta_files_root()``), the shared volume's mount
    :ptype root: Path
    :param tables: the tables the writer folds
    :ptype tables: Sequence[FoldedTable]
    :param workers: the worker processes the rounds run in; None makes one worker of its own, shut by :meth:`close`
    :ptype workers: ProcessWorkers | None
    """

    def __init__(self, root: Path, tables: Sequence[FoldedTable], *, workers: ProcessWorkers | None = None) -> None:
        if not _PYARROW:
            raise ImportError("a fold needs pyarrow: install 3tears[snapshot]")
        self.root = root
        self.tables: Final = {table.name: table for table in tables}
        self._plain_tables = tuple(table.plain() for table in tables)
        self._owns_workers = workers is None
        self._workers = workers if workers is not None else ProcessWorkers(workers=1, name="delta folds")
        #: delta files older than this the fold never heard named, so its orphan sweep leaves them
        self._since = time.time()
        #: (table, file, token) of each commit not yet handed to a round that finished, in commit order
        self._queued: list[tuple[str, str, str]] = []
        #: the moment the newest queued commit was told, or its writer's oldest work in flight: nothing saved after
        #: it is swept as an orphan
        self._before = self._since
        self._term: int | None = None
        #: each (delta file, state) share not folded yet, the deltas named as parts, the covered deltas not deleted
        self._pending: tuple[_Share, ...] = ()
        self._reused: tuple[str, ...] = ()
        self._doomed: tuple[str, ...] = ()
        self._round: asyncio.Task[None] | None = None
        self._unfolded_warned: set[str] = set()

    def committed(
        self,
        files: Mapping[str, str],
        generations: Mapping[str, str],
        *,
        version: int,
        in_flight_since: float | None = None,
    ) -> None:
        """note a commit's delta files and start a round of folds in the background; never waits, never raises for
        a fold. Call it after the commit, on the event loop.

        :param files: table -> its delta file, as the commit's transaction names them
            (:attr:`~threetears.core.collections.caller_transaction.CallerTransaction.delta_files`)
        :ptype files: Mapping[str, str]
        :param generations: table -> the generation token the commit's advance wrote: the version a part folded past
            the file is current to
        :ptype generations: Mapping[str, str]
        :param version: the write's version (:meth:`~threetears.core.collections.scope_epochs.ScopeEpochs.begin`),
            taken under the write lease: the writer's term (module docstring)
        :ptype version: int
        :param in_flight_since: when the writer's oldest transaction still open began, seconds since the epoch, for a
            writer with transactions open beside this commit; None when none is (the orphan sweep leaves every file
            saved after it)
        :ptype in_flight_since: float | None
        :return: nothing
        :rtype: None
        """
        now = time.time()
        for table, file in files.items():
            if table not in self.tables:
                if table not in self._unfolded_warned:
                    self._unfolded_warned.add(table)
                    log.warning(
                        "a commit named a delta file of a table the writer does not fold; its files are never folded "
                        "or deleted",
                        extra={"extra_data": {"table": table, "file": file}},
                    )
                continue
            token = generations.get(table)
            if token is None or split_generation_token(token) is None:
                _ROUNDS.labels(outcome="no_generation").inc()
                log.error(
                    "a commit's delta file came with no generation to version a part by; it is not folded",
                    extra={"extra_data": {"table": table, "file": file, "generation": token}},
                )
                continue
            self._queued.append((table, file, token))
        self._term = version if self._term is None else max(self._term, version)
        self._before = now if in_flight_since is None else min(now, in_flight_since)
        if self._queued and (self._round is None or self._round.done()):
            self._round = asyncio.get_running_loop().create_task(self._rounds(), name="delta folds")

    async def wait(self) -> None:
        """wait for the round in progress, and any it starts, to end.

        :return: nothing
        :rtype: None
        """
        while self._round is not None and not self._round.done():
            await asyncio.shield(self._round)

    async def close(self) -> None:
        """wait for the round in progress, then shut the worker this fold made.

        :return: nothing
        :rtype: None
        """
        await self.wait()
        if self._owns_workers:
            await self._workers.close()

    async def _rounds(self) -> None:
        """run rounds while commits are queued; a failed round keeps its files and waits for the next commit.

        :return: nothing
        :rtype: None
        """
        while self._queued:
            taken, self._queued = self._queued, []
            term = self._term
            assert term is not None  # noqa: S101 -- set by the commit that queued the files
            try:
                answer = await self._workers.run(
                    fold_round,
                    str(self.root),
                    term,
                    self._plain_tables,
                    tuple(taken),
                    self._pending,
                    self._reused,
                    self._doomed,
                    self._since,
                    self._before,
                )
            except BaseException as error:  # prawduct:allow prawduct/broad-except -- a fold never fails a write: logged, counted, retried at the next commit
                self._queued = taken + self._queued
                if isinstance(error, asyncio.CancelledError):
                    raise
                _ROUNDS.labels(outcome="failed").inc()
                log.error(
                    "a delta fold round failed; its files are kept for the round the next commit starts",
                    extra={"extra_data": {"files": len(taken), "error": repr(error)}},
                    exc_info=error,
                )
                return
            self._took(answer)

    def _took(self, answer: tuple[Any, ...]) -> None:
        """keep what a round answered, and count and log it.

        :param answer: :func:`fold_round`'s answer
        :ptype answer: tuple
        :return: nothing
        :rtype: None
        """
        pending, reused, doomed, outcome, folded, deleted, swept, manifest, pid = answer
        self._pending, self._reused, self._doomed = pending, reused, doomed
        _ROUNDS.labels(outcome=outcome).inc()
        for table, _scope, how, _rows in folded:
            _STATES.labels(table=table, how=how).inc()
        if outcome == "stale":
            log.warning(
                "a later writer's full is current: this writer's lease was taken, so its folds publish nothing",
                extra={"extra_data": {"term": self._term, "manifest": manifest}},
            )
        elif folded:
            log.info(
                "delta files folded into a new full",
                extra={
                    "extra_data": {
                        "manifest": manifest,
                        "states": len(folded),
                        "reused": sum(1 for _, _, how, _ in folded if how == "reused"),
                        "deleted": deleted,
                        "swept": swept,
                        "pending": len(pending),
                        "worker": pid,
                    }
                },
            )


def fold_round(
    root: str,
    term: int,
    tables: tuple[tuple[Any, ...], ...],
    committed: tuple[tuple[str, str, str], ...],
    pending: tuple[_Share, ...],
    reused: tuple[str, ...],
    doomed: tuple[str, ...],
    since: float,
    before: float,
) -> tuple[Any, ...]:
    """one round of folds, in a worker process: note the commits' files, fold each state due, publish, delete.

    Plain data in and out (:mod:`~threetears.core.process_workers`): names, versions and counts, never rows.

    :param root: the delta files' root
    :ptype root: str
    :param term: the writer's term
    :ptype term: int
    :param tables: each folded table, :meth:`FoldedTable.plain`
    :ptype tables: tuple
    :param committed: each new commit's (table, delta file, generation token), in commit order
    :ptype committed: tuple
    :param pending: each (delta file, state) share not folded yet, from the round before
    :ptype pending: tuple
    :param reused: the delta files named as parts, kept until no manifest in use names them
    :ptype reused: tuple[str, ...]
    :param doomed: covered delta files a round before could not delete
    :ptype doomed: tuple[str, ...]
    :param since: when the fold started: older files are left by the orphan sweep
    :ptype since: float
    :param before: the writer's oldest work in flight: newer files are left by the orphan sweep
    :ptype before: float
    :return: ``(pending, reused, doomed, outcome, folded, deleted, swept, manifest, pid)``: the state for the next
        round, the round's outcome (:data:`_ROUNDS`), each state folded as ``(table, scope, how, rows)``, the files
        deleted and swept, the manifest published (or the current one), and this worker's process id
    :rtype: tuple
    """
    base = Path(root)
    config = {plain[0]: _Folding.of(plain) for plain in tables}
    shares = list(pending)
    for table, file, token in committed:
        shares.extend(_shares(base, config[table], file, token))
    current = current_full_manifest(base)

    def answer(
        outcome: str, manifest: str, *, folded: Sequence[Any] = (), deleted: int = 0, swept: int = 0
    ) -> tuple[Any, ...]:
        return (tuple(shares), reused, doomed, outcome, tuple(folded), deleted, swept, manifest, os.getpid())

    if current is None:
        return answer("no_full", "")
    if current.term > term:
        return answer("stale", current.name)
    # a delta a full names as a part is kept until no manifest in use names it, then deleted: a writer that started
    # after the publish that named it learns it here, from the manifest
    reused = tuple(sorted(set(reused) | {file for file in current.files() if not file.startswith(_FULLS_PREFIX)}))
    by_state: dict[tuple[str, str | None], list[_Share]] = {}
    for share in shares:
        if share[0] in current.tables:
            by_state.setdefault((share[0], share[1]), []).append(share)
    tables_now = dict(current.tables)
    folded: list[tuple[str, str | None, str, int]] = []
    folded_shares: set[_Share] = set()
    named_parts: list[str] = []
    for (table, scope), state_shares in by_state.items():
        entry = tables_now[table]
        part = next((p for p in entry.parts if p.scope == scope), None)
        held_bytes = (base / part.file).stat().st_size if part is not None else 0
        if sum(share[4] for share in state_shares) * _DUE_OF < held_bytes * _DUE_AT:
            continue
        new_part, how = _fold_state(base, config[table], scope, part, state_shares)
        if how == "reused":
            named_parts.append(new_part.file)
        tables_now[table] = FullTable(
            version=_newest(table, [entry.version, new_part.version]),
            parts=tuple(p for p in entry.parts if p.scope != scope) + (new_part,),
        )
        folded.append((table, scope, how, new_part.rows))
        folded_shares.update(state_shares)
    if not folded:
        return answer("nothing_due", current.name)
    try:
        published = publish_full_manifest(base, term=term, tables=tables_now)
    except StaleTermError:
        # another writer published a later term since this round read the current full: its parts are orphans
        # the current writer's next round deletes
        return answer("stale", current.name)
    shares = [share for share in shares if share not in folded_shares]
    left = {share[2] for share in shares}
    covered = {share[2] for share in folded_shares} - left
    kept_reused = set(reused) | set(named_parts)
    listed = deletable_full_files(base, covered_deltas=sorted(covered | set(doomed) | kept_reused))
    deleted: set[str] = set()
    for file in listed:
        try:
            (base / file).unlink(missing_ok=True)
        except OSError as error:  # NOSILENT: logged; the file is listed again by the next round
            log.warning(
                "a file the full no longer names could not be deleted",
                extra={"extra_data": {"file": file, "error": repr(error)}},
            )
            continue
        deleted.add(file)
    reused = tuple(sorted(kept_reused - deleted))
    doomed = tuple(sorted((covered | set(doomed)) - deleted - kept_reused))
    named = left | set(reused) | set(doomed) | published.files()
    swept = sweep_orphan_delta_files(base, named=named, before=before, since=since, tables=list(config))
    return answer("folded", published.name, folded=folded, deleted=len(deleted), swept=len(swept))


@dataclass(frozen=True, slots=True)
class _Folding:
    """a folded table as a round reads it from :meth:`FoldedTable.plain`."""

    name: str
    key: tuple[str, ...]
    columns: tuple[str, ...]
    scope_column: str | None
    scope_prefix: str | None
    keep: type[DeltaKeepRule] | None

    @classmethod
    def of(cls, plain: tuple[Any, ...]) -> _Folding:
        name, key, columns, column, prefix, keep = plain
        return cls(name, tuple(key), tuple(columns), column, prefix, keep)

    def scope_of(self, value: Any) -> str:
        """a row's state from its scope column's value, as the writer's seqlock names it."""
        return f"{self.scope_prefix}{value}"


def _read_ipc(path: Path) -> Any:
    """an Arrow IPC file's table, read whole.

    :param path: the file
    :ptype path: Path
    :return: its table
    :rtype: pyarrow.Table
    """
    with pa.memory_map(str(path)) as source:
        return pa_ipc.open_file(source).read_all()


def _shares(root: Path, table: _Folding, file: str, token: str) -> list[_Share]:
    """a delta file's bytes shared among the states its rows belong to, by rows.

    :param root: the delta files' root
    :ptype root: Path
    :param table: the file's table
    :ptype table: _Folding
    :param file: the file, under the root
    :ptype file: str
    :param token: the generation its commit wrote
    :ptype token: str
    :return: one share a state
    :rtype: list
    """
    size = (root / file).stat().st_size
    if table.scope_column is None:
        return [(table.name, None, file, token, max(1, size))]
    rows = _read_ipc(root / file)
    counts = Counter(table.scope_of(value) for value in rows.column(table.scope_column).to_pylist())
    total = max(1, rows.num_rows)
    return [(table.name, scope, file, token, max(1, size * count // total)) for scope, count in counts.items()]


def _held_rows(path: Path, columns: Sequence[str]) -> Any:
    """a part's rows: its table's columns, a delta file named as the part read without its delete markers.

    :param path: the part's file
    :ptype path: Path
    :param columns: the table's columns
    :ptype columns: Sequence[str]
    :return: its rows
    :rtype: pyarrow.Table
    """
    rows = _read_ipc(path)
    if _DELETED_COLUMN in rows.column_names:
        rows = rows.filter(pa.array([not deleted for deleted in rows.column(_DELETED_COLUMN).to_pylist()], pa.bool_()))
    return rows.select(list(columns)).replace_schema_metadata(None)


def _fold_state(
    root: Path, table: _Folding, scope: str | None, part: FullPart | None, shares: Sequence[_Share]
) -> tuple[FullPart, str]:
    """one state's new part: the old part, each delta's rows of the state applied in order, then the keep rule;
    the last delta named as the part when it is exactly what the fold keeps (decision 16), else a new part file.

    :param root: the delta files' root
    :ptype root: Path
    :param table: the table
    :ptype table: _Folding
    :param scope: the state; None for a table folded whole
    :ptype scope: str | None
    :param part: the state's part in the current full; None when it has none
    :ptype part: FullPart | None
    :param shares: the state's shares since its last fold, in commit order
    :ptype shares: Sequence
    :return: the new part, and ``"reused"`` or ``"written"``
    :rtype: tuple[FullPart, str]
    :raises ValueError: when a delta's generation is of another incarnation than the part's (a reset: the rebuild's)
    """
    schema: Any = None
    changes: list[DeltaChange] = []
    last_rows: list[dict[str, Any]] = []
    last_only_state = False
    for index, (_, _, file, _, _) in enumerate(shares):
        delta = _read_ipc(root / file)
        if schema is None:
            schema = pa.schema([delta.schema.field(name).with_nullable(True) for name in table.columns])
        sets = [frozenset(named) for named in json.loads(delta.schema.metadata[_WRITTEN_METADATA])]
        mine = delta
        if scope is not None:
            assert table.scope_column is not None  # noqa: S101 -- a scoped share is a scoped table's
            mask = [table.scope_of(value) == scope for value in delta.column(table.scope_column).to_pylist()]
            mine = delta.filter(pa.array(mask, pa.bool_()))
        is_last = index == len(shares) - 1
        if is_last:
            last_only_state = mine.num_rows == delta.num_rows
            last_rows = []
        for row in mine.to_pylist():
            deleted, written = row.pop(_DELETED_COLUMN), row.pop(_WRITTEN_COLUMN)
            change = DeltaChange(
                row={name: row[name] for name in table.columns}, deleted=deleted, written=sets[written]
            )
            changes.append(change)
            if is_last and not deleted:
                last_rows.append(change.row)
    held = schema.empty_table() if part is None else _held_rows(root / part.file, table.columns).cast(schema)
    version = shares[-1][3]
    if part is not None:
        # refused across a reset: the part and the delta must be of one incarnation
        _newest(table.name, [part.version, version])
    folded = _applied(table, schema, held, changes)
    if table.keep is not None:
        folded = table.keep.keep(folded)
    if last_only_state and _same_rows(table, folded, last_rows):
        last_file = shares[-1][2]
        return FullPart(scope=scope, file=last_file, version=version, rows=len(last_rows)), "reused"
    return write_full_part(root, table.name, scope, folded.select(list(table.columns)), version), "written"


def _applied(table: _Folding, schema: Any, held: Any, changes: Sequence[DeltaChange]) -> Any:
    """the held rows with every change applied in order, merged as L3 merged it (:func:`apply_delta_row`).

    Rows no change names stay as Arrow; only the rows changes name are merged a row at a time.

    :param table: the table
    :ptype table: _Folding
    :param schema: the part's schema
    :ptype schema: pyarrow.Schema
    :param held: the old part's rows
    :ptype held: pyarrow.Table
    :param changes: the deltas' changes of the state, in order
    :ptype changes: Sequence[DeltaChange]
    :return: the folded rows
    :rtype: pyarrow.Table
    """

    def key_of(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(row[name] for name in table.key)

    touched = {key_of(change.row) for change in changes}
    held_keys = list(zip(*(held.column(name).to_pylist() for name in table.key), strict=True))
    hit = [key in touched for key in held_keys]
    merged: dict[tuple[Any, ...], dict[str, Any]] = {
        key_of(row): row for row in held.filter(pa.array(hit, pa.bool_())).to_pylist()
    }
    for change in changes:
        key = key_of(change.row)
        after = apply_delta_row(merged.get(key), change)
        if after is None:
            merged.pop(key, None)
        else:
            merged[key] = after
    untouched = held.filter(pa.array([not h for h in hit], pa.bool_()))
    return pa.concat_tables([untouched, pa.Table.from_pylist(list(merged.values()), schema=schema)])


def _same_rows(table: _Folding, folded: Any, rows: Sequence[Mapping[str, Any]]) -> bool:
    """whether the folded rows are exactly ``rows`` (a delta's rows written), one a key.

    :param table: the table
    :ptype table: _Folding
    :param folded: the rows a part would hold
    :ptype folded: pyarrow.Table
    :param rows: the last delta's rows written
    :ptype rows: Sequence[Mapping[str, Any]]
    :return: whether they are the same rows
    :rtype: bool
    """
    if folded.num_rows != len(rows):
        return False
    by_key = {tuple(row[name] for name in table.key): dict(row) for row in rows}
    if len(by_key) != len(rows):
        # a key written twice in one commit: the file holds both rows, a part may hold only one
        return False
    for row in folded.select(list(table.columns)).to_pylist():
        if by_key.get(tuple(row[name] for name in table.key)) != row:
            return False
    return True


def _newest(table: str, tokens: Sequence[str]) -> str:
    """the newest of a table's generation tokens.

    :param table: the table, for the message
    :ptype table: str
    :param tokens: its tokens
    :ptype tokens: Sequence[str]
    :return: the newest
    :rtype: str
    :raises ValueError: when one is not a token, or two are of different incarnations (a reset: the table starts
        fresh from the writer's rebuild, never from a fold)
    """
    split = []
    for token in tokens:
        parsed = split_generation_token(token)
        if parsed is None:
            raise ValueError(f"{table}: {token!r} is not a generation token")
        split.append((parsed, token))
    if len({incarnation for (incarnation, _), _ in split}) != 1:
        raise ValueError(
            f"{table}: generations of incarnations {sorted({i for (i, _), _ in split})}: a reset starts the table "
            f"fresh from the writer's rebuild, never from a fold"
        )
    return max(split)[1]
