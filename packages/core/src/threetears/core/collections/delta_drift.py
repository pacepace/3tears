"""the writer's drift sample: a few hundred random keys of each opted-in table read from L3 by key and compared with
its full on the volume (``design-pod-delta-files.md``, decision 32, check B).

**Why.** A pod copy checks itself against the full (:meth:`~threetears.core.collections.copy_follower.CopyFollower
.check_drift`), never against L3, so a full that drifted from L3 (a fold's bug, a delta file that lied) would pass every
copy's check. The writer alone, which owns the full, samples L3 against it: every :data:`DRIFT_SAMPLE_SECONDS` by
default (``THREETEARS_DELTA_DRIFT_SAMPLE_SECONDS``, per deployment), ``sample_keys`` keys a table, read by key through
:func:`~threetears.core.keyed.read_keyed_arrow`: a lookup a key, never a scan of the table.

**Random over the full's own keys.** The keys are drawn uniformly from the full's rows (each part weighted by the rows
its manifest entry counts), read off the volume, so picking them asks L3 nothing. Each sampled row keeps its part's
version: the generation the full holds it at.

**At the same generation.** A key changed since its part's version would differ for no fault, so it is skipped: the
epoch log (the NATS log the copies walk, never L3) names every key each commit of the table changed between the oldest
sampled part's version and the generation read after the L3 read; a sampled key one of them names past its part's
version is left out, and a record naming no keys (a whole-table one, a load's or a clear's) leaves out every key it
may reach. With the writer's seqlock given (``settled``), the sample is taken only between writes: a write in progress,
or one that began or ended during the read, skips the round, so a commit whose generation was not yet advanced is
never read as drift. A log that may be missing a record skips the table.

**Compared by the full's own row hash** (:func:`~threetears.core.collections.volume_fulls.row_digests`, the one rule
the fingerprints sum), so a row L3 holds and the full's are equal exactly when their hashes are. A key L3 no longer
holds differs too. ``delta_fold_drift_keys{table}`` is the keys of the last sample that differed (0 when none did),
each sample is counted by outcome (``delta_fold_drift_samples_total``), and a difference is logged at WARNING with a
few of its keys.

**The writer alone.** :class:`DeltaDriftSampler` asks ``holds_lease`` before each sample: a replica that is not the
writer reads nothing, not the volume and not L3.
"""

from __future__ import annotations

import asyncio
import os
import secrets
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from threetears.observe import PeriodicTask, get_logger
from threetears.observe.metrics import counter, gauge

from threetears.core.collections.complete_copy import Unsettled
from threetears.core.collections.copy_follower import FollowedTable
from threetears.core.collections.delta_files import under_prefix
from threetears.core.collections.epoch_change import EpochChangeReader, key_text
from threetears.core.collections.generation import split_generation_token
from threetears.core.collections.schema_backed import rail_arrow_types
from threetears.core.collections.volume_fulls import (
    FullPart,
    current_full_manifest,
    fingerprint_kinds,
    read_full_part,
    row_digests,
)
from threetears.core.keyed import KeyBatch, read_keyed_arrow

__all__ = ["DeltaDriftSampler"]

log = get_logger(__name__)

#: how often the writer samples L3 against its full by default (seconds): a quarter hour
DRIFT_SAMPLE_SECONDS: Final = 900.0

#: the per-deployment setting of the sample's period, in seconds; ``0`` turns it off
_DRIFT_SAMPLE_ENV: Final = "THREETEARS_DELTA_DRIFT_SAMPLE_SECONDS"

#: keys a table each sample reads from L3 by default: a few hundred lookups by key, a quarter hour
_SAMPLE_KEYS: Final = 300

#: the most differing keys a sample's WARNING names
_KEYS_NAMED: Final = 5

_DRIFT_KEYS = gauge(
    "delta_fold_drift_keys",
    description="keys of the writer's last sample of a table whose row in L3 differs from its full's at the same "
    "generation (or is gone from L3); 0 when the last sample found none",
    label_names=("table",),
)

#: samples by table and outcome: ``sampled`` (compared), ``unsettled`` (a write in progress, or one moved during the
#: read: the round is skipped), ``log_incomplete`` (the epoch log may miss a record: changed keys unknown), ``no_full``
#: (no full holds the table yet), ``other_incarnation`` (the full is of another incarnation than the table's log)
_SAMPLES = counter(
    "delta_fold_drift_samples_total",
    description="the writer's samples of a table's keys from L3 against its full, by table and outcome",
    label_names=("table", "outcome"),
)

#: keys sampled, by table and what became of each: ``equal``, ``differs``, ``changed_since`` (a commit since its part's
#: version names it: not comparable, skipped)
_SAMPLED_KEYS = counter(
    "delta_fold_drift_sampled_keys_total",
    description="keys the writer's drift sample drew from a table's full, by table and what became of each (equal, "
    "differs, changed_since: skipped)",
    label_names=("table", "outcome"),
)


def _sample_seconds(given: float | None) -> float:
    """the sample's period: ``given``, else :data:`_DRIFT_SAMPLE_ENV`, else :data:`DRIFT_SAMPLE_SECONDS`.

    :param given: the sampler's argument
    :ptype given: float | None
    :return: seconds; 0 turns the sample off
    :rtype: float
    :raises ValueError: when the period is negative or the setting is not a number
    """
    if given is None:
        named = os.environ.get(_DRIFT_SAMPLE_ENV)
        given = DRIFT_SAMPLE_SECONDS if not named else float(named)
    if given < 0:
        raise ValueError(f"the drift sample's period is not negative, got {given}")
    return given


@dataclass(frozen=True, slots=True)
class _Drawn:
    """the keys drawn from a table's full: each key's text, its typed values, its part's version and its row's hash.

    :ivar keys: each key's values as text, in key order
    :ivar typed: each key's typed values, in key order
    :ivar versions: each key's part's version
    :ivar digests: each key's row hash in the full
    """

    keys: tuple[tuple[str, ...], ...]
    typed: tuple[tuple[Any, ...], ...]
    versions: tuple[str, ...]
    digests: tuple[int, ...]


def _draw(root: Path, table: FollowedTable, parts: Sequence[FullPart], count: int) -> _Drawn:
    """``count`` rows drawn uniformly from a table's parts (each weighted by the rows its manifest entry counts), read
    off the volume: only the parts a drawn row falls in are read. Blocking: run it on a thread.

    :param root: the delta files' root
    :ptype root: Path
    :param table: the table
    :ptype table: FollowedTable
    :param parts: its parts in the current full
    :ptype parts: Sequence[FullPart]
    :param count: the rows to draw (fewer when the full holds fewer)
    :ptype count: int
    :return: the rows drawn
    :rtype: _Drawn
    """
    total = sum(part.rows for part in parts)
    picks = sorted(secrets.SystemRandom().sample(range(total), min(count, total))) if total else []
    by_part: dict[int, list[int]] = {}
    start = 0
    index = 0
    for at, part in enumerate(parts):
        end = start + part.rows
        while index < len(picks) and picks[index] < end:
            by_part.setdefault(at, []).append(picks[index] - start)
            index += 1
        start = end
    types = rail_arrow_types(table.columns)
    kinds = fingerprint_kinds(types)
    keys: list[tuple[str, ...]] = []
    typed: list[tuple[Any, ...]] = []
    versions: list[str] = []
    digests: list[int] = []
    for at, rows_at in by_part.items():
        part = parts[at]
        drawn = read_full_part(root, part).select(list(types)).take(rows_at)
        values = list(zip(*(drawn.column(column).to_pylist() for column in table.key), strict=True))
        keys += [tuple(key_text(value) for value in key) for key in values]
        typed += values
        versions += [part.version] * drawn.num_rows
        digests += row_digests(drawn, kinds)
    return _Drawn(keys=tuple(keys), typed=tuple(typed), versions=tuple(versions), digests=tuple(digests))


def _l3_digests(table: FollowedTable, pages: Sequence[Any]) -> dict[tuple[str, ...], int]:
    """each key's row hash as L3 holds it, from the rows read by key.

    :param table: the table
    :ptype table: FollowedTable
    :param pages: the rows read
    :ptype pages: Sequence[pyarrow.Table]
    :return: key text -> its row's hash
    :rtype: dict[tuple[str, ...], int]
    """
    types = rail_arrow_types(table.columns)
    kinds = fingerprint_kinds(types)
    held: dict[tuple[str, ...], int] = {}
    for page in pages:
        rows = page.select(list(types))
        keys = zip(*(rows.column(column).to_pylist() for column in table.key), strict=True)
        for key, digest in zip(keys, row_digests(rows, kinds), strict=True):
            held[tuple(key_text(value) for value in key)] = digest
    return held


class DeltaDriftSampler:
    """the writer's drift sample of its opted-in tables (module docstring). Made once by the writing load beside its
    :class:`~threetears.core.collections.delta_fold.DeltaFolds`; :meth:`start` it, and it samples only while
    ``holds_lease`` says this replica is the writer.

    :param tables: the opted-in tables to sample, as their copies follow them
    :ptype tables: Sequence[FollowedTable]
    :param l3: the writer's L3 (the rail's proxy, or a pool): read only by key
    :ptype l3: Any
    :param generation: a table's current write generation, or None when it has none
    :ptype generation: Callable[[str], Awaitable[str | None]]
    :param changes: the NATS epoch log: the keys each commit since a part's version changed
    :ptype changes: EpochChangeReader
    :param holds_lease: whether this replica holds the write lease now
    :ptype holds_lease: Callable[[], bool]
    :param root: the delta files' root, the full's volume
    :ptype root: Path
    :param settled: the writer's seqlock (``ScopeEpochs.settled``): a sample is taken only between writes; None takes
        it whenever asked
    :ptype settled: Callable[[], Awaitable[Any]] | None
    :param sample_keys: keys a table each sample reads from L3
    :ptype sample_keys: int
    :param sample_seconds: how often :meth:`start`'s task samples; None reads ``THREETEARS_DELTA_DRIFT_SAMPLE_SECONDS``,
        else :data:`DRIFT_SAMPLE_SECONDS`; 0 turns it off
    :ptype sample_seconds: float | None
    :raises ValueError: when ``sample_keys`` is under one or the period is negative
    """

    def __init__(
        self,
        tables: Sequence[FollowedTable],
        l3: Any,
        *,
        generation: Callable[[str], Awaitable[str | None]],
        changes: EpochChangeReader,
        holds_lease: Callable[[], bool],
        root: Path,
        settled: Callable[[], Awaitable[Any]] | None = None,
        sample_keys: int = _SAMPLE_KEYS,
        sample_seconds: float | None = None,
    ) -> None:
        if sample_keys < 1:
            raise ValueError("sample_keys must be at least one")
        self._tables = {table.name: table for table in tables}
        self._l3 = l3
        self._generation = generation
        self._changes = changes
        self._holds_lease = holds_lease
        self._root = root
        self._settled = settled
        self._sample_keys = sample_keys
        self._seconds = _sample_seconds(sample_seconds)
        self._task: PeriodicTask | None = None

    def start(self) -> None:
        """sample every period, the first at a random point within one; a no-op when off or started.

        :return: nothing
        :rtype: None
        """
        if self._seconds <= 0 or self._task is not None:
            return
        self._task = PeriodicTask(
            self.sample,
            interval=self._seconds,
            name="delta drift sample",
            logger=log,
            first_delay=secrets.SystemRandom().uniform(0, self._seconds),
            failure_message="the writer's drift sample failed; the next period samples again",
        )
        self._task.start()

    async def stop(self) -> None:
        """stop sampling and wait for a sample in flight to end.

        :return: nothing
        :rtype: None
        """
        if self._task is not None:
            await self._task.stop()
            self._task = None

    async def sample(self) -> dict[str, list[tuple[str, ...]]]:
        """one sample of every table, when this replica is the writer: the keys of each whose row in L3 differs from
        its full's at the same generation.

        :return: table -> the differing keys (as text); empty when not the writer, or no table was sampled
        :rtype: dict[str, list[tuple[str, ...]]]
        """
        if not self._holds_lease():
            return {}
        manifest = await asyncio.to_thread(current_full_manifest, self._root)
        differing: dict[str, list[tuple[str, ...]]] = {}
        for name in sorted(self._tables):
            entry = manifest.tables.get(name) if manifest is not None else None
            if entry is None or not entry.parts:
                _SAMPLES.labels(table=name, outcome="no_full").inc()
                continue
            found = await self._sample_table(self._tables[name], entry.parts)
            if found is not None:
                differing[name] = found
        return differing

    async def _sample_table(self, table: FollowedTable, parts: Sequence[FullPart]) -> list[tuple[str, ...]] | None:
        """one table's sample (module docstring).

        :param table: the table
        :ptype table: FollowedTable
        :param parts: its parts in the current full
        :ptype parts: Sequence[FullPart]
        :return: the differing keys; None when the table was not sampled (counted by why)
        :rtype: list[tuple[str, ...]] | None
        """
        drawn = await asyncio.to_thread(_draw, self._root, table, parts, self._sample_keys)
        if not drawn.keys:
            return []
        before = await self._settled() if self._settled is not None else None
        if isinstance(before, Unsettled):
            _SAMPLES.labels(table=table.name, outcome="unsettled").inc()
            return None
        casts = tuple(table.key_cast(column) for column in table.key)
        pages = await read_keyed_arrow(
            self._l3,
            table.name,
            keys=KeyBatch(table.key, list(drawn.keys), casts),
            types=rail_arrow_types(table.columns),
        )
        upto = await self._generation(table.name)
        after = await self._settled() if self._settled is not None else None
        if self._settled is not None and (isinstance(after, Unsettled) or after != before):
            # a write began or ended while L3 was read: a commit it made may not have advanced the generation yet
            _SAMPLES.labels(table=table.name, outcome="unsettled").inc()
            return None
        changed = await self._changed_since(table, drawn, upto)
        if changed is None:
            return None
        held = await asyncio.to_thread(_l3_digests, table, pages)
        differs: list[tuple[str, ...]] = []
        skipped = 0
        for key, digest in zip(drawn.keys, drawn.digests, strict=True):
            if key in changed:
                skipped += 1
            elif held.get(key) != digest:
                differs.append(key)
        _SAMPLES.labels(table=table.name, outcome="sampled").inc()
        _SAMPLED_KEYS.labels(table=table.name, outcome="changed_since").inc(skipped)
        _SAMPLED_KEYS.labels(table=table.name, outcome="differs").inc(len(differs))
        _SAMPLED_KEYS.labels(table=table.name, outcome="equal").inc(len(drawn.keys) - skipped - len(differs))
        _DRIFT_KEYS.labels(table=table.name).set(len(differs))
        if differs:
            log.warning(
                "the writer's full differs from L3 at the same generation for some sampled keys",
                extra={
                    "extra_data": {
                        "table": table.name,
                        "sampled": len(drawn.keys),
                        "changed_since": skipped,
                        "differs": len(differs),
                        "keys": [list(key) for key in differs[:_KEYS_NAMED]],
                        "generation": upto,
                    }
                },
            )
        return differs

    async def _changed_since(
        self, table: FollowedTable, drawn: _Drawn, upto: str | None
    ) -> set[tuple[str, ...]] | None:
        """the drawn keys some commit changed past their part's version, up to ``upto``: from the epoch log, never L3.

        :param table: the table
        :ptype table: FollowedTable
        :param drawn: the keys drawn
        :ptype drawn: _Drawn
        :param upto: the generation read after the L3 read
        :ptype upto: str | None
        :return: the keys; None when they cannot be told (counted by why)
        :rtype: set[tuple[str, ...]] | None
        """
        at = split_generation_token(upto) if upto is not None else None
        held = {version: split_generation_token(version) for version in set(drawn.versions)}
        if at is None or any(parsed is None or parsed[0] != at[0] for parsed in held.values()):
            _SAMPLES.labels(table=table.name, outcome="other_incarnation").inc()
            return None
        assert upto is not None  # noqa: S101 -- parsed above
        counts = {version: parsed[1] for version, parsed in held.items() if parsed is not None}
        low = min(counts.values())
        if at[1] <= low:
            return set()
        walk = await self._changes.changes_after(table.name, f"{at[0]}:{low}", upto)
        if not walk.complete:
            _SAMPLES.labels(table=table.name, outcome="log_incomplete").inc()
            return None
        changed: set[tuple[str, ...]] = set()
        for change in walk.changes:
            parsed = split_generation_token(change.token)
            if parsed is None:
                continue
            past = {key for key, version in zip(drawn.keys, drawn.versions, strict=True) if counts[version] < parsed[1]}
            if change.whole_table or change.keys_unknown or (change.partition and not change.prefixes):
                changed |= past  # a record naming no keys may reach any of them
                continue
            named = {tuple(row.ids) for row in change.rows}
            changed |= past & named
            for key in past - named:
                if change.prefixes and under_prefix(dict(zip(table.key, key, strict=True)), change.prefixes):
                    changed.add(key)
        return changed
