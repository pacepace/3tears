"""The copy's apply lane: the one way a copy's changes reach its :class:`~threetears.core.collections.shared_copy.
SharedCopyBackend`, one change or tier merge at a time, in the order given.

**Why a lane** (moved from reports' ``shared_backend.SharedArrowBackend``: ``_applying``, ``_merge_soon``,
``_replace_broken``). The backend applies one change per call; its callers (a follower's passes, a reload's publish)
give changes faster than one worker applies them at election-night peaks. The lane holds them in order, bounded,
gives each to the backend in turn, schedules the tier merges a per-run table's changes make due, and replaces a dead
apply worker, so a caller gives a change and hears its version or its failure.

**Order.** Changes apply in the order :meth:`ApplyLane.submit` took them, each on the version the one before it
published. A change that fails publishes nothing, and every change queued behind it fails with
:class:`EarlierChangeFailed` naming it: none lands on a copy missing an earlier change. The caller gives the failed
change (re-read) and those after it again; the lane takes them as any.

**Back-pressure: bounded, never unbounded.** The lane holds at most ``max_changes`` changes, and at most ``max_bytes``
of their pages (Arrow ``nbytes``), queued behind the one applying. A change past either bound waits in
:meth:`ApplyLane.submit` until the lane has room: the caller is held, never a change dropped or the queue grown.
:meth:`ApplyLane.submit_nowait` refuses at once with :class:`ApplyLaneFull` instead. A change bigger than ``max_bytes``
alone is taken into an empty queue, so it is never refused for good.

**Merges.** A change landing in a per-run table's parts (``PartRule.run_column``) makes a merge due
(:class:`~threetears.core.collections.shared_copy.MergePolicy`). The lane runs one merge, as its own publish, before
the next change, and keeps merging while nothing is queued until none is due: a change waits for one merge at most,
and no merge sits between a change and the version showing it (reports' ``_merge_soon``: one merge a lock turn). A
merge that fails is said and counted (:meth:`ApplyLane.status`); the copy is whole without it, and the next change to
a per-run table makes it due again.

**A dead apply worker** (``BrokenProcessPool``: killed, out of memory) is replaced by the backend's pool, and the
change it was applying given once more to the fresh one; a second death fails the change with
:class:`ApplyWorkerDied`, loudly (logged as an error, counted), and the changes behind it as above.

**A reload's publish** (:mod:`~threetears.core.collections.copy_reload`) is a step of this lane like a change: a
reload stages in a worker of its own while this lane applies on, and its finished staged version is given here
(:meth:`ApplyLane.submit_reload`) to publish in order. The changes given after the reload (:meth:`ApplyLane.watch`
marks the point) to the tables it reloads were applied to the copy meanwhile; they are kept (Arrow, bounded) and
applied again over the reload in its one publish (reports' decision 52: a change heard during a reload is applied over
its full after the swap), so no version shows the reload without them. A reload whose publish fails, or that was
cancelled, publishes nothing and fails no change behind it: the copy is whole without it.

**State.** The queue is a tuple replaced whole, read and written on the event loop alone, under the lane's condition;
counters are ints of this object, written only by its task.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Collection, Sequence
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Final

from threetears.core.collections.shared_copy import Applied, ChangePage, SharedCopyBackend, StagedReload
from threetears.observe import contain, get_logger

__all__ = [
    "ApplyLane",
    "ApplyLaneFull",
    "ApplyWorkerDied",
    "ChangeWatch",
    "EarlierChangeFailed",
    "LaneStatus",
    "ReloadCancelled",
    "ReloadOverrun",
    "ReloadStep",
]

log = get_logger(__name__)

#: the changes a lane holds queued behind the one applying. A follower gives one change a pass, and a pass waits on
#: its change's publish before the next starts (reports' ``CopyFollower``), so a deeper queue only holds more pages in
#: the serving process; 32 lets a burst of reloads and passes queue without holding any of them
_MAX_CHANGES: Final = 32

#: the bytes of queued changes' pages a lane holds (Arrow ``nbytes``): the pages sit in the serving process until their
#: change is handed to the apply worker. Reports' hand-off of one change is bounded at 1.25 GiB
#: (``.devops/chart/values.yaml``, the store's hand-off) and a busy night's change is a few MiB; 512 MiB holds a
#: cold start's large change behind another and keeps the serving process's share bounded
_MAX_BYTES: Final = 512 << 20


class ApplyLaneFull(Exception):
    """the lane holds ``max_changes`` changes or ``max_bytes`` of pages already (:meth:`ApplyLane.submit_nowait`)."""


class ApplyWorkerDied(RuntimeError):
    """the apply worker died applying a change, and again applying it once more in a fresh worker."""


class EarlierChangeFailed(RuntimeError):
    """a change queued behind one that failed: not applied, so it never lands on a copy missing the earlier one."""


class ReloadCancelled(RuntimeError):
    """a reload cancelled before its publish: nothing of it is published."""


class ReloadOverrun(LookupError):
    """the changes given during a reload to the tables it reloads passed the bytes kept to apply over it: nothing is
    published, and the reload is read again (its full holds them now)."""


class ChangeWatch:
    """the changes applied after a point to some tables, kept (their pages, Arrow) to apply over a reload of them
    (:meth:`ApplyLane.watch`). Its attributes are replaced whole, on the event loop, by the lane alone.

    :ivar tables: the tables watched
    :ivar after: changes given after this one are kept (the lane's count of changes given)
    :ivar max_bytes: the most bytes of pages kept; past it the watch is :attr:`overrun` and keeps none
    :ivar pages: the pages kept, in the order applied
    :ivar whole: the tables a kept change replaced whole (their pages before it are not kept)
    :ivar nbytes: the bytes kept
    :ivar newer_mark: whether a change given after the point published a mark (the reload's own is older then)
    :ivar overrun: whether more than ``max_bytes`` was kept
    """

    def __init__(self, tables: frozenset[str], after: int, max_bytes: int) -> None:
        self.tables = tables
        self.after = after
        self.max_bytes = max_bytes
        self.pages: tuple[ChangePage, ...] = ()
        self.whole: frozenset[str] = frozenset()
        self.nbytes = 0
        self.newer_mark = False
        self.overrun = False

    def keep(self, pages: Sequence[ChangePage], whole: frozenset[str], mark: int | None) -> None:
        """keep a change applied after the point: its pages of the watched tables, a whole one's replacing those
        kept of its tables. Called by the lane alone, on the event loop.

        :param pages: the change's pages
        :ptype pages: Sequence[ChangePage]
        :param whole: the tables it replaced whole
        :ptype whole: frozenset[str]
        :param mark: its mark
        :ptype mark: int | None
        :return: nothing
        :rtype: None
        """
        if mark is not None:
            self.newer_mark = True
        replaced = whole & self.tables
        mine = tuple(page for page in pages if page.table in self.tables)
        if self.overrun or not (mine or replaced):
            return
        kept = tuple(page for page in self.pages if page.table not in replaced) + mine
        self.pages, self.whole = kept, self.whole | replaced
        self.nbytes = sum(page.rows.nbytes for page in kept)
        if self.nbytes > self.max_bytes:
            self.pages, self.nbytes, self.overrun = (), 0, True


class ReloadStep:
    """a staged reload's publish, as the lane takes it (:meth:`ApplyLane.submit_reload`).

    :ivar staged: the reload, staged (its owner closes it)
    :ivar watch: the changes kept to apply over it
    :ivar mark: its mark
    :ivar cancelled: set before its turn, it publishes nothing
    :ivar publishing: set as its turn starts: past this point it is not cancelled
    """

    def __init__(self, staged: StagedReload, watch: ChangeWatch, mark: int | None) -> None:
        self.staged = staged
        self.watch = watch
        self.mark = mark
        self.cancelled = False
        self.publishing = False


@dataclass(frozen=True, slots=True)
class LaneStatus:
    """what a lane holds and has done.

    :ivar queued: changes waiting, the one applying aside
    :ivar queued_bytes: their pages' bytes
    :ivar applied: changes published
    :ivar failed: changes that failed (those failed for an earlier one's failure among them)
    :ivar merges: tier merges published
    :ivar restarts: apply workers that died and were replaced
    :ivar waits: submits that waited for room
    :ivar last_error: the last failure, as text; None for none
    :ivar reloads: reloads published (not counted in ``applied``)
    """

    queued: int
    queued_bytes: int
    applied: int
    failed: int
    merges: int
    restarts: int
    waits: int
    last_error: str | None
    reloads: int = 0


@dataclass(frozen=True, slots=True)
class _Change:
    pages: tuple[ChangePage, ...]
    whole: frozenset[str]
    mark: int | None
    nbytes: int
    done: asyncio.Future[Applied]
    #: its place in the order given (:class:`ChangeWatch`)
    seq: int = 0
    #: a reload's publish in place of a change; None for a change
    reload: ReloadStep | None = None


class ApplyLane:
    """a copy's changes, applied one at a time in order through ``backend``, bounded (the module's docstring).

    Make it on the event loop it serves; its task starts with the first change.

    :param backend: the copy's backend: every change and merge goes through this lane alone
    :ptype backend: SharedCopyBackend
    :param max_changes: the most changes queued behind the one applying
    :ptype max_changes: int
    :param max_bytes: the most bytes of queued changes' pages
    :ptype max_bytes: int
    :raises ValueError: for a bound under one
    """

    def __init__(
        self,
        backend: SharedCopyBackend,
        *,
        max_changes: int = _MAX_CHANGES,
        max_bytes: int = _MAX_BYTES,
    ) -> None:
        if max_changes < 1 or max_bytes < 1:
            raise ValueError("a lane holds one change and one byte at least")
        self._backend = backend
        self._max_changes = max_changes
        self._max_bytes = max_bytes
        #: the changes waiting, oldest first: replaced whole, on the loop, under ``_changed``
        self._queue: tuple[_Change, ...] = ()
        #: the submits held for room, in the order given: replaced whole, on the loop, under ``_changed``
        self._held: tuple[object, ...] = ()
        self._queued_bytes = 0
        self._changed = asyncio.Condition()
        self._busy = False
        self._merge_due = False
        #: a merge runs before the next change: set by a change making one due, cleared by the merge
        self._merge_turn = False
        self._closing = False
        self._task: asyncio.Task[None] | None = None
        #: the wake-ups :meth:`submit_nowait` started, held until done (a task nothing holds may be collected)
        self._wakes: tuple[asyncio.Task[None], ...] = ()
        self._applied = 0
        self._failed = 0
        self._merges = 0
        self._restarts = 0
        self._waits = 0
        self._last_error: str | None = None
        #: changes given so far: each change's place in the order
        self._given = 0
        #: the reloads' watches: replaced whole, on the loop
        self._watches: tuple[ChangeWatch, ...] = ()
        self._reloads = 0

    def status(self) -> LaneStatus:
        """what the lane holds and has done.

        :return: the status
        :rtype: LaneStatus
        """
        return LaneStatus(
            queued=len(self._queue),
            queued_bytes=self._queued_bytes,
            applied=self._applied,
            failed=self._failed,
            merges=self._merges,
            restarts=self._restarts,
            waits=self._waits,
            last_error=self._last_error,
            reloads=self._reloads,
        )

    def _room(self, nbytes: int) -> bool:
        if not self._queue:
            return True
        return len(self._queue) < self._max_changes and self._queued_bytes + nbytes <= self._max_bytes

    def _change(
        self, pages: Sequence[ChangePage], whole: Collection[str], mark: int | None, reload: ReloadStep | None = None
    ) -> _Change:
        if self._closing:
            raise RuntimeError("the copy's apply lane is closed")
        held = tuple(pages)
        self._given += 1
        return _Change(
            pages=held,
            whole=frozenset(whole),
            mark=mark,
            nbytes=sum(page.rows.nbytes for page in held),
            done=asyncio.get_running_loop().create_future(),
            seq=self._given,
            reload=reload,
        )

    def watch(self, tables: Collection[str], *, max_bytes: int) -> ChangeWatch:
        """keep, from now, every change given after this call to ``tables`` as it applies: what a reload of them
        applies over itself as it publishes (:meth:`submit_reload`). On the event loop.

        :param tables: the tables
        :ptype tables: Collection[str]
        :param max_bytes: the most bytes of pages it keeps (:class:`ChangeWatch`)
        :ptype max_bytes: int
        :return: the watch
        :rtype: ChangeWatch
        """
        watch = ChangeWatch(frozenset(tables), self._given, max_bytes)
        self._watches = (*self._watches, watch)
        return watch

    def unwatch(self, watch: ChangeWatch) -> None:
        """stop keeping changes for ``watch`` (its reload published, failed or cancelled). Harmless twice.

        :param watch: the watch
        :ptype watch: ChangeWatch
        :return: nothing
        :rtype: None
        """
        self._watches = tuple(held for held in self._watches if held is not watch)

    async def submit_reload(self, step: ReloadStep) -> asyncio.Future[Applied]:
        """queue a staged reload's publish behind the changes given before it, waiting for room as :meth:`submit`.
        Its turn applies the changes its watch kept over it, in one version; past its turn it is not cancelled.

        :param step: the reload's publish
        :ptype step: ReloadStep
        :return: what it published, or why it did not (:class:`ReloadCancelled`, :class:`ReloadOverrun`, the
            apply's own)
        :rtype: asyncio.Future[Applied]
        :raises RuntimeError: when the lane is closed
        """
        return await self._submitted(self._change((), (), step.mark, step))

    def _queued(self, change: _Change) -> asyncio.Future[Applied]:
        """take a change into the queue (under ``_changed``, room checked) and wake the lane."""
        self._queue = (*self._queue, change)
        self._queued_bytes += change.nbytes
        self._started()
        self._changed.notify_all()
        return change.done

    async def submit(
        self, pages: Sequence[ChangePage], *, whole: Collection[str] = (), mark: int | None = None
    ) -> asyncio.Future[Applied]:
        """queue a change, waiting for room while the lane is full (its back-pressure).

        :param pages: the change's pages, in the order staged
        :ptype pages: Sequence[ChangePage]
        :param whole: the tables it replaces whole
        :ptype whole: Collection[str]
        :param mark: its mark (``SharedCopyBackend.apply``)
        :ptype mark: int | None
        :return: the change's outcome, once applied: what it published, or why it failed
        :rtype: asyncio.Future[Applied]
        :raises RuntimeError: when the lane is closed
        """
        return await self._submitted(self._change(pages, whole, mark))

    async def _submitted(self, change: _Change) -> asyncio.Future[Applied]:
        """take a change into the queue, waiting for room while the lane is full."""
        async with self._changed:
            if self._held or not self._room(change.nbytes):
                # held in the order given: a change given later never takes the room before this one
                turn = object()
                self._held = (*self._held, turn)
                self._waits += 1
                log.info(
                    "the copy's apply lane is full; a change waits for room",
                    extra={"extra_data": {"queued": len(self._queue), "queued_bytes": self._queued_bytes}},
                )
                try:
                    await self._changed.wait_for(
                        lambda: self._closing or (self._held[0] is turn and self._room(change.nbytes))
                    )
                finally:
                    self._held = tuple(held for held in self._held if held is not turn)
                    self._changed.notify_all()
                if self._closing:
                    raise RuntimeError("the copy's apply lane is closed")
            return self._queued(change)

    def submit_nowait(
        self, pages: Sequence[ChangePage], *, whole: Collection[str] = (), mark: int | None = None
    ) -> asyncio.Future[Applied]:
        """queue a change now, or refuse it.

        :param pages: as :meth:`submit`
        :ptype pages: Sequence[ChangePage]
        :param whole: as :meth:`submit`
        :ptype whole: Collection[str]
        :param mark: as :meth:`submit`
        :ptype mark: int | None
        :return: as :meth:`submit`
        :rtype: asyncio.Future[Applied]
        :raises ApplyLaneFull: when the lane holds its bound already
        :raises RuntimeError: when the lane is closed
        """
        change = self._change(pages, whole, mark)
        if self._held or not self._room(change.nbytes):
            raise ApplyLaneFull(
                f"the copy's apply lane holds {len(self._queue)} changes ({self._queued_bytes} bytes) of "
                f"{self._max_changes} ({self._max_bytes} bytes), {len(self._held)} more held for room"
            )
        # on the loop, between awaits: no coroutine is inside the condition but at its wait, which gave the lock up,
        # so the queue is replaced here as under it, and the lane woken by a task of its own
        self._queue = (*self._queue, change)
        self._queued_bytes += change.nbytes
        self._started()
        self._wakes = (*self._wakes, asyncio.get_running_loop().create_task(self._wake()))
        return change.done

    def _started(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="copy-apply-lane")

    async def _wake(self) -> None:
        async with self._changed:
            self._changed.notify_all()
        self._wakes = tuple(wake for wake in self._wakes if not wake.done())

    async def apply(
        self, pages: Sequence[ChangePage], *, whole: Collection[str] = (), mark: int | None = None
    ) -> Applied:
        """queue a change (waiting for room) and wait for its publish.

        :param pages: as :meth:`submit`
        :ptype pages: Sequence[ChangePage]
        :param whole: as :meth:`submit`
        :ptype whole: Collection[str]
        :param mark: as :meth:`submit`
        :ptype mark: int | None
        :return: what it published
        :rtype: Applied
        :raises Exception: why it failed (:class:`ApplyWorkerDied`, :class:`EarlierChangeFailed`, the apply's own)
        """
        # shielded: a caller that stops waiting leaves its change in its place, applied in order all the same
        return await asyncio.shield(await self.submit(pages, whole=whole, mark=mark))

    async def drained(self) -> None:
        """wait until nothing is queued or applying and no merge is due.

        :return: nothing
        :rtype: None
        """
        async with self._changed:
            await self._changed.wait_for(lambda: not self._queue and not self._busy and not self._merge_due)

    async def close(self) -> None:
        """take no more changes, apply those queued, then stop (the backend stays open: its owner closes it).

        :return: nothing
        :rtype: None
        """
        async with self._changed:
            self._closing = True
            self._changed.notify_all()
        if self._task is not None:
            await self._task

    # -- the lane's task -------------------------------------------------------------------------------------------

    async def _next(self) -> _Change | bool:
        """the next change to apply, True for a merge's turn, False to stop (closing, nothing queued); waits for one."""
        async with self._changed:
            await self._changed.wait_for(lambda: bool(self._queue) or self._merge_due or self._closing)
            if self._merge_due and (self._merge_turn or not self._queue):
                self._busy = True
                return True
            if not self._queue:
                return False
            change, self._queue = self._queue[0], self._queue[1:]
            self._queued_bytes -= change.nbytes
            self._busy = True
            self._changed.notify_all()
            return change

    async def _run(self) -> None:
        try:
            while (step := await self._next()) is not False:
                try:
                    if isinstance(step, _Change):
                        await self._apply(step)
                    else:
                        await self._merge_once()
                finally:
                    async with self._changed:
                        self._busy = False
                        self._changed.notify_all()
        finally:
            # stopped (closed, cancelled, or a fault of the lane's own): nothing queued is left waiting on it
            async with self._changed:
                self._fail_queued(lambda: RuntimeError("the copy's apply lane stopped before applying this change"))
                self._changed.notify_all()

    async def _apply(self, change: _Change) -> None:
        if change.reload is not None:
            await self._publish_reload(change, change.reload)
            return
        try:
            applied = await self._applied_once_more(change)
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- the caller hears it through its future
            self._failed += 1
            self._last_error = f"{type(exc).__name__}: {exc}"
            contain(
                exc,
                log=log,
                what="a change to the copy failed; nothing published, and the changes queued behind it fail too",
                extra={"behind": len(self._queue)},
                level="error",
            )
            if not change.done.done():
                change.done.set_exception(exc)
            async with self._changed:
                error = self._last_error
                self._fail_queued(lambda: EarlierChangeFailed(f"an earlier change to the copy failed: {error}"))
                self._changed.notify_all()
            return
        except BaseException:
            # the lane stopped mid-apply (cancelled): its caller hears that, never waits on for good
            if not change.done.done():
                change.done.set_exception(
                    RuntimeError("the copy's apply lane stopped while applying this change; it may be published")
                )
            raise
        self._applied += 1
        for watch in self._watches:
            if change.seq > watch.after:
                watch.keep(change.pages, change.whole, change.mark)
        self._merge_after(applied)
        if not change.done.done():
            change.done.set_result(applied)

    def _merge_after(self, applied: Applied) -> None:
        rules = self._backend.rules
        # a per-run table's change may make a tier merge due; the backend says when small parts are due a packing, or
        # the version holds more files than its bound (``shared_copy.MAX_VERSION_FILES``)
        if applied.merge_due or any(
            rules[table].split and rules[table].run_column is not None for table in applied.parts if table in rules
        ):
            self._merge_due = True
            self._merge_turn = True

    async def _publish_reload(self, change: _Change, step: ReloadStep) -> None:
        """a reload's publish in its turn: the changes its watch kept applied over it, in one version; or nothing,
        cancelled or overrun. A failure fails no change behind it."""
        self.unwatch(step.watch)
        watch = step.watch
        try:
            if step.cancelled:
                raise ReloadCancelled("the reload was cancelled before its publish")
            step.publishing = True
            if watch.overrun:
                raise ReloadOverrun(
                    f"the changes given during the reload of {sorted(watch.tables)} passed the {watch.max_bytes} "
                    f"bytes kept to apply over it; the reload is read again"
                )
            # a change given after the reload published a newer mark: the version keeps that one
            mark = None if watch.newer_mark else step.mark
            applied = await self._reload_once_more(step, mark)
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- the reload's caller hears it through its future; the copy is whole without it
            if not isinstance(exc, ReloadCancelled):
                self._failed += 1
                self._last_error = f"a reload: {type(exc).__name__}: {exc}"
                contain(exc, log=log, what="a reload's publish failed; nothing of it published", level="error")
            if not change.done.done():
                change.done.set_exception(exc)
            return
        except BaseException:
            if not change.done.done():
                change.done.set_exception(
                    RuntimeError("the copy's apply lane stopped while publishing this reload; it may be published")
                )
            raise
        self._reloads += 1
        self._merge_after(applied)
        if not change.done.done():
            change.done.set_result(applied)

    async def _reload_once_more(self, step: ReloadStep, mark: int | None) -> Applied:
        """a reload's publish; given once more to a fresh apply worker when its worker died."""
        watch = step.watch
        for attempt in range(2):
            try:
                return await self._backend.publish_reload(step.staged, watch.pages, whole=watch.whole, mark=mark)
            except BrokenProcessPool as died:
                self._restarts += 1
                if attempt:
                    raise ApplyWorkerDied(
                        "the copy's apply worker died publishing a reload, and again in a fresh worker"
                    ) from died
                log.warning("the copy's apply worker died publishing a reload; a fresh one publishes it once more")
        raise AssertionError("unreachable")

    async def _applied_once_more(self, change: _Change) -> Applied:
        """the change applied; given once more to a fresh worker when its worker died (the pool replaces it)."""
        try:
            return await self._backend.apply(change.pages, whole=change.whole, mark=change.mark)
        except BrokenProcessPool as died:
            self._restarts += 1
            log.warning(
                "the copy's apply worker died applying a change; a fresh one applies it once more",
                extra={"extra_data": {"error": f"{died}"}},
            )
        try:
            return await self._backend.apply(change.pages, whole=change.whole, mark=change.mark)
        except BrokenProcessPool as again:
            self._restarts += 1
            raise ApplyWorkerDied(
                "the copy's apply worker died applying a change, and again applying it in a fresh worker"
            ) from again

    async def _merge_once(self) -> None:
        self._merge_turn = False
        try:
            merged = await self._backend.merge()
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- a merge is layout: the copy is whole without it, and the next per-run change makes it due again
            if isinstance(exc, BrokenProcessPool):
                self._restarts += 1
            self._merge_due = False
            self._last_error = f"a tier merge: {type(exc).__name__}: {exc}"
            contain(exc, log=log, what="a tier merge of the copy failed; the copy is as it was", level="error")
            return
        if merged is None:
            self._merge_due = False
        else:
            self._merges += 1

    def _fail_queued(self, error: Callable[[], BaseException]) -> None:
        """fail every queued change with an ``error()`` each (under ``_changed``; the caller wakes the waiters)."""
        queued, self._queue = self._queue, ()
        self._queued_bytes = 0
        for change in queued:
            self._failed += 1
            if not change.done.done():
                change.done.set_exception(error())
