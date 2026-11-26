"""A copy follower's backend over the copy's worker backend: the follower's stages held as the pages of a change, and
each change applied in the apply worker (:class:`~threetears.core.collections.shared_copy.SharedCopyBackend`) through
its apply lane, a reload staged in the reload worker and published through the same lane.

**Why an adapter** (``design-copy-worker-backend.md``: the step the 3T-2 changelog left for later). A
:class:`~threetears.core.collections.copy_follower.CopyFollower` stages a change by name, page by page, beside the copy
(``stage_rows``, ``stage_keys``, ``unstage_keys``, ``drop_stage``) from a dozen sites in the follower, its checkpoint
load and the volume's fulls, then applies what it staged as one change (``apply_staged``). The worker backend takes a
change as its pages, in the order staged (:class:`~threetears.core.collections.shared_copy.ChangePage`), with the same
semantics as the staging tables it was moved from: a key's last page wins. So a stage here is its pages, held as given
(Arrow, the follower's own pages; nothing copied, cast or filtered in this process: the apply worker casts each page to
its table's columns and settles it), and an apply hands them over in order. The follower's staging sites stay as they
are, and so do the backends a test drives the follower over.

**A change's pages, in order.** Each :class:`~threetears.cache.duckdb.StagedChange` gives its keys stage's pages, then
its rows stage's: the in-process copy's apply deleted the keys staged, then inserted the rows staged, so a row staged for
a key named in the keys stage stands. Within a stage, pages keep the order they were staged in:

- ``stage_keys(rows_stage=)`` takes those keys' rows out of the rows stage: a keys page in it, after the rows it
  undoes (the keys are in the keys stage too, so a key not staged again is deleted, as before);
- ``unstage_keys`` takes keys out of the stages named: a leave page in each (``shared_copy.LEAVE``), so the change
  leaves those keys as the copy holds them. The follower names every stage of the change (``_leave_out``).

A table a whole change replaces (``whole``) is its rows stage's pages, the table replaced in the same version.

**Order and bounds** are the apply lane's (:class:`~threetears.core.collections.copy_lane.ApplyLane`): one change at a
time, in the order given, bounded, a dead worker replaced and its change given once more. A change that fails keeps its
stages, so the follower applies it again; one whose stages are gone raises ``LookupError`` (the follower reads it again).

**A reload from the volume** (decision 52; ``CopyFollower._reload_from_volume``) stages outside the pass lock and
swaps under it. :meth:`SharedCopyStages.prepare_staged` gives its pages to the reload lane
(:class:`~threetears.core.collections.copy_reload.ReloadLane`), held: they are cut into parts in the reload worker
while the apply lane applies every other table's changes, and the stage lets them go (the serving process holds none of
a reload's rows while it stages). The swap's :meth:`apply_staged` then lets it publish with the mark the follower took
under the lock: its staged parts linked in place of the table's, with any change given to the table since applied over
them, in one version. A reload the reload worker could not stage, or that was cancelled, has no stages left: the swap
raises ``LookupError`` and the follower looks for the full again. A per-state reload (a keys stage and a rows stage, not
whole) and any change the reload lane cannot take apply as an ordinary change through the apply lane, under the lock.

**Calls and state.** The staging methods are called on threads (the follower's ``asyncio.to_thread``); they replace
the stages, a read-only mapping of read-only stages, whole under a lock, and nothing else: no CPU, no IO. The apply and
the reload's prepare are coroutines on the event loop the lanes serve. The prepared reloads are a tuple replaced whole
under the same lock; a stage dropped on a thread cancels its reload on the loop.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Final

import pyarrow as pa

from threetears.core.cache.duckdb import STAGE_PREFIX, StagedChange
from threetears.core.collections.copy_follower import is_reload_stage
from threetears.core.collections.copy_lane import ApplyLane
from threetears.core.collections.copy_reload import Reload, ReloadLane
from threetears.core.collections.shared_copy import (
    KEYS,
    LEAVE,
    ROWS,
    ApplyHooks,
    ChangePage,
    SharedCopyBackend,
)
from threetears.core.frozen_state import EMPTY, with_entry, without_entries
from threetears.interfaces.keyed.rows import DeclaredKey
from threetears.observe import get_logger

__all__ = ["SharedCopyStages"]

log = get_logger(__name__)

#: the hex digits of a table's schema digest (``CheckpointCopy.schema_digest``): the in-process copy's length
_DIGEST_CHARS: Final = 16


@dataclass(frozen=True, slots=True)
class _Stage:
    """one stage: its table and its pages, in the order staged; ``prepared`` once its pages went to the reload lane."""

    table: str
    pages: tuple[ChangePage, ...] = ()
    prepared: bool = False


@dataclass(frozen=True, slots=True)
class _Prepared:
    """a reload given to the reload lane held, for the stages it was staged from."""

    stages: frozenset[str]
    reload: Reload
    loop: asyncio.AbstractEventLoop


def _named(stage: str) -> str:
    if not stage.startswith(STAGE_PREFIX):
        raise ValueError(f"a stage's name begins with {STAGE_PREFIX!r}, not {stage!r}")
    return stage


class SharedCopyStages:
    """a :class:`~threetears.core.collections.copy_follower.CopyFollower`'s backend (``FollowerBackend``,
    ``PreparesStaged``, and ``copy_checkpoint.CheckpointCopy``) over ``backend``: each change applied in its apply
    worker through an apply lane of its own, each reload from the volume staged in a reload worker of its own (the
    module's docstring). The follower's ``queries`` read the store ``backend`` publishes in.

    Make it on the event loop it serves, as the pod starts or on a thread: the reload lane starts its fork server.

    :param backend: the copy's worker backend; every change and merge reaches it through this object's lane
    :ptype backend: SharedCopyBackend
    :param check: hooks run in the reload worker over each reload's staged tables (``ReloadLane(check=)``)
    :ptype check: ApplyHooks | None
    :param preload: modules the reload worker holds from its start (the check's among them)
    :ptype preload: Sequence[str]
    :param lane: the apply lane; None makes one over ``backend`` with its default bounds
    :ptype lane: ApplyLane | None
    :param reloads: the reload lane, over ``lane``; None makes one
    :ptype reloads: ReloadLane | None
    """

    def __init__(
        self,
        backend: SharedCopyBackend,
        *,
        check: ApplyHooks | None = None,
        preload: Sequence[str] = (),
        lane: ApplyLane | None = None,
        reloads: ReloadLane | None = None,
    ) -> None:
        self._backend = backend
        self._lane = ApplyLane(backend) if lane is None else lane
        self._reloads = ReloadLane(backend, self._lane, check=check, preload=preload) if reloads is None else reloads
        self._lock = threading.Lock()
        #: each stage held: replaced whole, under ``_lock``
        self._stages: Mapping[str, _Stage] = EMPTY
        #: the reloads given held, each for its stages: replaced whole, under ``_lock``
        self._prepared: tuple[_Prepared, ...] = ()
        #: the next place a change's first page takes (``ChangePage.places``): rising, so the changes a reload's watch
        #: keeps stay in order when applied over it together. On the loop alone
        self._places = 0

    @property
    def backend(self) -> SharedCopyBackend:
        """the copy's worker backend.

        :return: the backend
        :rtype: SharedCopyBackend
        """
        return self._backend

    @property
    def lane(self) -> ApplyLane:
        """the apply lane (its ``status()``).

        :return: the lane
        :rtype: ApplyLane
        """
        return self._lane

    @property
    def reloads(self) -> ReloadLane:
        """the reload lane (its ``status()``).

        :return: the lane
        :rtype: ReloadLane
        """
        return self._reloads

    async def start(self) -> None:
        """publish the copy's first version, every table empty and no mark, when the store holds none: the follower's
        reads (``queries``) then find a version before its first change, as they found the in-process copy's empty
        tables. On the event loop.

        :return: nothing
        :rtype: None
        """
        if self._backend.store.current is None:
            await self._lane.apply(())

    # -- staging: on a thread, the stages replaced whole --------------------------------------------------------------

    def _known(self, table: str) -> None:
        if table not in self._backend.schemas:
            raise ValueError(f"unknown table {table!r}")

    def _added(self, stage: str, table: str, pages: Sequence[ChangePage]) -> None:
        """``pages`` added to ``stage`` (made on first use), under the lock."""
        held = self._stages.get(stage)
        if held is None:
            held = _Stage(table)
        elif held.table != table:
            raise ValueError(f"stage {stage!r} is of {held.table!r}, not {table!r}")
        kept = tuple(page for page in pages if page.rows.num_rows)
        self._stages = with_entry(self._stages, stage, _Stage(table, held.pages + kept, held.prepared))

    def stage_rows(
        self,
        stage: str,
        table: str,
        rows: Any,
        key: Annotated[Sequence[str], DeclaredKey],
        *,
        unique_keys: bool = False,
    ) -> int:
        """add rows of ``table`` to ``stage``, made on first use: a rows page of the change, as given (the apply worker
        casts it to the table's columns and keeps each key's last row).

        :param stage: the stage, named with ``STAGE_PREFIX``
        :ptype stage: str
        :param table: the table
        :ptype table: str
        :param rows: the rows
        :ptype rows: pyarrow.Table
        :param key: the table's key columns
        :ptype key: Annotated[Sequence[str], DeclaredKey]
        :param unique_keys: no key repeats among the rows (not read: the worker settles every page alike)
        :ptype unique_keys: bool
        :return: the rows staged
        :rtype: int
        :raises ValueError: for an unknown table, a stage of another table or a name not a stage's
        """
        self._known(table)
        with self._lock:
            self._added(_named(stage), table, [ChangePage(ROWS, table, rows)] if rows.num_rows else [])
        return int(rows.num_rows)

    def stage_keys(
        self,
        stage: str,
        table: str,
        key: Annotated[Sequence[str], DeclaredKey],
        keys: Any,
        *,
        rows_stage: str | None = None,
    ) -> int:
        """add keys of ``table`` to ``stage``, made on first use: a keys page (deleted, unless a row is staged for
        them); with ``rows_stage``, the rows it holds of them leave it (a keys page there too, after them).

        :param stage: the keys stage
        :ptype stage: str
        :param table: the table
        :ptype table: str
        :param key: its key columns
        :ptype key: Annotated[Sequence[str], DeclaredKey]
        :param keys: the keys, a column per key column, as text or typed
        :ptype keys: pyarrow.Table
        :param rows_stage: a rows stage of ``table`` to take these keys' rows out of; None for none
        :ptype rows_stage: str | None
        :return: the keys staged
        :rtype: int
        :raises ValueError: for an unknown table, a stage of another table or a name not a stage's
        """
        self._known(table)
        page = [ChangePage(KEYS, table, keys.select(list(key)))] if keys.num_rows else []
        with self._lock:
            self._added(_named(stage), table, page)
            if rows_stage is not None:
                self._added(_named(rows_stage), table, page)
        return int(keys.num_rows)

    def unstage_keys(
        self, stages: Sequence[str], table: str, key: Annotated[Sequence[str], DeclaredKey], keys: Any
    ) -> int:
        """take keys out of these stages of ``table``: a leave page in each held (the change leaves them as the copy
        holds them, whatever the stage's pages before it said), the rest of each kept. A stage not held is skipped.

        :param stages: the stages
        :ptype stages: Sequence[str]
        :param table: the table
        :ptype table: str
        :param key: its key columns
        :ptype key: Annotated[Sequence[str], DeclaredKey]
        :param keys: the keys, a column per key column, as text or typed
        :ptype keys: pyarrow.Table
        :return: the keys given
        :rtype: int
        :raises ValueError: for an unknown table or a name not a stage's
        """
        self._known(table)
        named = [_named(stage) for stage in stages]
        if not keys.num_rows or not named:
            return 0
        page = ChangePage(LEAVE, table, keys.select(list(key)))
        with self._lock:
            for stage in named:
                if stage in self._stages:
                    self._added(stage, table, [page])
        return int(keys.num_rows)

    def drop_stage(self, *stages: str) -> None:
        """drop these stages (one not held is no error), and cancel a reload prepared from them.

        :param stages: the stages
        :ptype stages: str
        :return: nothing
        :rtype: None
        """
        dropped = frozenset(_named(stage) for stage in stages)
        with self._lock:
            self._stages = without_entries(self._stages, dropped)
            cancelled = tuple(held for held in self._prepared if held.stages & dropped)
            self._prepared = tuple(held for held in self._prepared if not held.stages & dropped)
        for held in cancelled:
            held.loop.call_soon_threadsafe(held.reload.cancel)

    def staged_stages(self) -> list[str]:
        """every stage held.

        :return: their names
        :rtype: list[str]
        """
        with self._lock:
            return sorted(self._stages)

    def schema_digest(self, table: str) -> str:
        """a digest of ``table``'s columns and their Arrow types, in order: a checkpoint part written under another is
        not loaded.

        :param table: the table
        :ptype table: str
        :return: the hex digest
        :rtype: str
        :raises ValueError: for an unknown table
        """
        self._known(table)
        schema = self._backend.schemas[table]
        text = "\n".join(f"{field.name}:{field.type}" for field in schema)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_DIGEST_CHARS]

    # -- the apply and a reload's prepare: on the loop ------------------------------------------------------------------

    def _held(self, changes: Sequence[StagedChange]) -> list[tuple[StagedChange, _Stage | None, _Stage | None]]:
        """each change with its keys and rows stages, under the lock; ``LookupError`` for a stage not held."""
        with self._lock:
            stages = self._stages
        held: list[tuple[StagedChange, _Stage | None, _Stage | None]] = []
        for change in changes:
            self._known(change.table)
            found: list[_Stage | None] = []
            for name in (change.keys, change.rows):
                stage = None if name is None else stages.get(name)
                if name is not None and stage is None:
                    raise LookupError(f"stage {name!r} of {change.table!r} is not held: its change is read again")
                found.append(stage)
            held.append((change, found[0], found[1]))
        return held

    async def prepare_staged(self, changes: Sequence[StagedChange]) -> None:
        """give a reload's changes (each whole, staged under a reload stage, rows pages alone) to the reload lane held:
        cut into parts in the reload worker now, published by the :meth:`apply_staged` that swaps them in. Their stages
        let the pages go. Changes the reload lane cannot take (a per-state reload's keys, a stage not a reload's) are
        left staged, applied as any change.

        :param changes: the reload's changes
        :ptype changes: Sequence[StagedChange]
        :return: nothing
        :rtype: None
        :raises LookupError: when a stage is not held
        """
        held = self._held(changes)
        names = frozenset(stage for change in changes for stage in (change.keys, change.rows) if stage is not None)
        reloadable = bool(held) and all(
            change.whole
            and change.rows is not None
            and is_reload_stage(change.rows)
            and rows is not None
            and not rows.prepared
            and all(page.kind == ROWS for page in rows.pages)
            for change, _keys, rows in held
        )
        if not reloadable:
            return
        pages = tuple(page for _change, _keys, rows in held if rows is not None for page in rows.pages)
        tables = frozenset(change.table for change in changes)
        reload = await self._reloads.reload(pages, tables=tables, held=True)
        with self._lock:
            # the pages are the reload lane's now; the stages say so, and are still the change's
            self._stages = _let_pages_go(self._stages, names)
            self._prepared = (*self._prepared, _Prepared(names, reload, asyncio.get_running_loop()))

    async def apply_staged(self, changes: Sequence[StagedChange], *, mark: int | None = None) -> int:
        """apply every change staged as one change, published as one version with ``mark``: through the apply lane,
        or, for a reload :meth:`prepare_staged` gave the reload lane, its publish. Its stages are dropped once applied.

        :param changes: each table's change
        :ptype changes: Sequence[StagedChange]
        :param mark: the change's mark, published beside it; None for none
        :ptype mark: int | None
        :return: the rows applied
        :rtype: int
        :raises LookupError: when a stage is gone (never staged, dropped, or its reload not staged): nothing applied,
            and the follower reads the change again
        :raises Exception: why the apply failed; its stages are kept, for the change to be applied again
        """
        names = frozenset(stage for change in changes for stage in (change.keys, change.rows) if stage is not None)
        with self._lock:
            prepared = next((held for held in self._prepared if held.stages == names), None)
        if prepared is not None:
            return await self._published(prepared, names, mark)
        held = self._held(changes)
        if any(stage is not None and stage.prepared for _change, *stages in held for stage in stages):
            raise LookupError("a reload's stages went to the reload lane without it: the change is read again")
        pages: list[ChangePage] = []
        whole: set[str] = set()
        for change, keys, rows in held:
            if change.whole:
                whole.add(change.table)
            elif keys is not None:
                pages.extend(keys.pages)
            if rows is not None:
                pages.extend(rows.pages)
        first, self._places = self._places, self._places + len(pages)
        applied = await self._lane.apply(_joined(pages, first), whole=whole, mark=mark)
        self._dropped(names)
        return int(applied.rows)

    async def _published(self, prepared: _Prepared, names: frozenset[str], mark: int | None) -> int:
        """a prepared reload's publish, with ``mark``; its stages dropped however it ends (its pages went with it)."""
        with self._lock:
            self._prepared = tuple(held for held in self._prepared if held is not prepared)
        try:
            try:
                applied = await asyncio.shield(prepared.reload.publish(mark))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise LookupError(
                    f"the reload of {sorted(prepared.reload.tables)} was not published ({type(exc).__name__}: {exc}); "
                    "its full is read again"
                ) from exc
        finally:
            self._dropped(names)
        return int(applied.rows)

    def _dropped(self, names: Collection[str]) -> None:
        with self._lock:
            self._stages = without_entries(self._stages, names)

    async def close(self) -> None:
        """stop the reload lane, the apply lane (applying what is queued) and the backend's worker.

        :return: nothing
        :rtype: None
        """
        with self._lock:
            prepared, self._prepared = self._prepared, ()
        for held in prepared:
            held.reload.cancel()
        await self._reloads.close()
        await self._lane.close()
        await self._backend.close()


def _joined(pages: Sequence[ChangePage], first: int) -> tuple[ChangePage, ...]:
    """``pages`` joined by kind, table and columns, each joined page naming its pages' places (``ChangePage.places``,
    from ``first`` up, in the order given): the same change, crossing to the apply worker as a few pages.

    Why: each page crosses as a file of its own (``SharedArrowPool.run``'s hand-off), and the follower stages a big
    change a page of keys and a page of rows at a time: a commit of 25,000 keys in pages of 50 was some 1,900 files,
    past the process's open files. A joined page is its pages' chunks under one table: zero-copy, no row touched in
    this process; the worker slices it back into its places.

    :param pages: the change's pages, in order
    :ptype pages: Sequence[ChangePage]
    :param first: the first page's place: past every place a change before it was given
    :ptype first: int
    :return: the joined pages
    :rtype: tuple[ChangePage, ...]
    """
    groups: list[tuple[ChangePage, list[tuple[int, ChangePage]]]] = []
    for place, page in enumerate(pages, start=first):
        group = next(
            (
                held
                for lead, held in groups
                if (lead.kind, lead.table) == (page.kind, page.table) and lead.rows.schema.equals(page.rows.schema)
            ),
            None,
        )
        if group is None:
            groups.append((page, [(place, page)]))
        else:
            group.append((place, page))
    return tuple(
        ChangePage(
            lead.kind,
            lead.table,
            pa.concat_tables([page.rows for _place, page in held]),
            places=tuple((place, page.rows.num_rows) for place, page in held),
        )
        for lead, held in groups
    )


def _let_pages_go(stages: Mapping[str, _Stage], names: Collection[str]) -> Mapping[str, _Stage]:
    """``stages`` with each of ``names`` held marked prepared, its pages let go (the reload lane holds them)."""
    for name in names:
        stage = stages.get(name)
        if stage is not None:
            stages = with_entry(stages, name, _Stage(stage.table, (), prepared=True))
    return stages
