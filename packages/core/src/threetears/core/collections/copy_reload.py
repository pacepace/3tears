"""The copy's reload lane: a whole-table (or per-scope) reload staged in a worker process of its own, beside the
apply lane, and published through it in order.

**Why** (``design-copy-worker-backend.md``, decision 3; reports' ``design-pod-delta-files.md``, decision 52). One
apply worker took reload, merge and change one after another, so a ready table's change waited out a reload's rewrite:
on a local burst replay (2026-11-24) commit to publish was a median 0.5 s with no reload between and a median 9.4 s,
up to 45 s, with one. Here a reload's rewrite runs in the reload worker (one process, named ``copy-reload``, its own
:class:`~threetears.core.shared_arrow.SharedArrowPool`), never the apply worker: the apply lane applies every other
change meanwhile. Only the reload's publish takes an apply-lane turn: its staged parts linked in place of the table's
(never copied), the changes given since applied over them, and the product's hooks, in one version.

**Order.** :meth:`ReloadLane.reload` marks the reload's place in the apply lane's order as it is called
(:meth:`~threetears.core.collections.copy_lane.ApplyLane.watch`): changes given before it are older than its full and
are replaced by it; changes given after it to the tables it reloads apply to the copy as they come (the reload delays
none of them) and are kept, Arrow, to apply again over the reload as it publishes, so the version that shows the
reload shows them too (reports' decision 52: a change heard during a reload is applied over its full after the swap).
The publish is queued behind the changes given before the reload finished staging, and those given after wait for it.
A change since the reload with a mark keeps its mark: the reload's own, older, is not published over it.

**Bounded.** At most ``max_reloads`` reloads, and ``max_bytes`` of their pages, wait or stage; a reload past either
waits in :meth:`ReloadLane.reload` (one bigger alone is taken into an empty lane). The changes kept for a reload are at
most ``max_replay_bytes``: past it the reload publishes nothing and fails with
:class:`~threetears.core.collections.copy_lane.ReloadOverrun` (its caller reads it again; its full holds them then).
The serving process holds a reload's rows only until they are handed to the worker: the staged parts are in the
store, and the pages are let go as the job starts (``SharedArrowPool.run`` holds none after the hand-off).

**Errors surface.** A reload's failure reaches its caller (:attr:`Reload.done`), is said and counted
(:meth:`ReloadLane.status`), and publishes nothing. A dead reload worker (``BrokenProcessPool``: killed, out of
memory) fails its reload with :class:`ReloadWorkerDied` (the reload's rows went with the hand-off; its caller reads it
again) and is replaced: the next reload stages in a fresh worker. A worker past ``private_bytes`` after a reload is
replaced before the next (the pool's own retire).

**Cancellable.** :meth:`Reload.cancel` before the publish's turn publishes nothing: a staging reload stops waiting for
its worker (the job's result is thrown away), a staged one's turn passes with nothing published. Once its turn has
started it is not cancelled.

**State.** The lane's reloads are a tuple replaced whole, on the event loop, under its condition; counters are ints of
this object, written on the loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Final

from threetears.core.collections.copy_lane import (
    ApplyLane,
    ChangeWatch,
    ReloadCancelled,
    ReloadOverrun,
    ReloadStep,
)
from threetears.core.collections.shared_copy import (
    Applied,
    ApplyHooks,
    ChangePage,
    ReloadScope,
    SharedCopyBackend,
    StagedReload,
)
from threetears.core.frozen_state import EMPTY, frozen
from threetears.core.shared_arrow import SharedArrowPool
from threetears.observe import contain, get_logger

__all__ = [
    "Reload",
    "ReloadLane",
    "ReloadStatus",
    "ReloadWorkerDied",
]

log = get_logger(__name__)

#: the name the reload worker gives itself (``/proc/<pid>/comm``, 15 bytes at most)
_RELOAD_PROCESS_NAME: Final = "copy-reload"

#: the reload worker is replaced after a reload leaving it more private memory than this: the apply worker's bound
#: (``shared_copy._APPLY_PRIVATE_BYTES``; reports' ``enr-reload`` was retired past the apply worker's bound too)
_RELOAD_PRIVATE_BYTES: Final = 2 << 30

#: reloads waiting or staging. A copy reloads a table when its full is found newer (one per table waiting, reports'
#: ``CopyFollower._reload_from_volume``); a reports pod copies about a dozen tables, and each reload's pages sit in the
#: serving process until handed over, so a few at once keep that share bounded
_MAX_RELOADS: Final = 4

#: bytes of the pages of reloads waiting or staging (Arrow ``nbytes``): reports' hand-off of one change is bounded at
#: 1.25 GiB (``.devops/chart/values.yaml``); the biggest reloaded table on a replay was 296 MB as Arrow
#: (``design-pod-delta-files.md``, LSF ``feed_why_sources``), so 512 MiB holds one beside a small one
_MAX_BYTES: Final = 512 << 20

#: bytes of the changes kept to apply over one reload as it publishes. A busy night's change is a few MiB and a big
#: reload stages in tens of seconds (reports' replay: a median 12.6 s, up to 25 s), so 256 MiB holds every change of
#: the table given during it; past it the reload is read again, its full holding them
_MAX_REPLAY_BYTES: Final = 256 << 20

#: a reload's states
_WAITING: Final = "waiting"
_STAGING: Final = "staging"
_QUEUED: Final = "queued"
_DONE: Final = "done"


class ReloadWorkerDied(LookupError):
    """the reload worker died staging a reload: nothing is published, the worker is replaced for the next reload, and
    the reload is read again (its rows were handed to the dead worker)."""


@dataclass(frozen=True, slots=True)
class ReloadStatus:
    """what a reload lane holds and has done.

    :ivar waiting: reloads waiting for room or for the worker
    :ivar staging: reloads staging (at most one: the lane has one worker)
    :ivar queued: reloads staged, their publish waiting its apply-lane turn
    :ivar reloads: reloads published
    :ivar cancelled: reloads cancelled
    :ivar failed: reloads that failed
    :ivar restarts: reload workers that died and were replaced
    :ivar waits: reloads that waited for room
    :ivar last_error: the last failure, as text; None for none
    """

    waiting: int
    staging: int
    queued: int
    reloads: int
    cancelled: int
    failed: int
    restarts: int
    waits: int
    last_error: str | None


class Reload:
    """one reload given to a :class:`ReloadLane`: its outcome, and its cancel.

    :ivar tables: the tables it reloads
    :ivar done: what its publish published, or why it did not
    """

    def __init__(
        self,
        on_cancel: Callable[[Reload], None],
        pages: tuple[ChangePage, ...],
        tables: frozenset[str],
        scopes: Mapping[str, ReloadScope],
        mark: int | None,
        watch: ChangeWatch,
    ) -> None:
        self._on_cancel = on_cancel
        #: the reload's pages until they are handed to the worker; empty after
        self._pages = pages
        self.nbytes = sum(page.rows.nbytes for page in pages)
        self.tables = tables
        self.scopes = scopes
        self.mark = mark
        self.watch = watch
        self.done: asyncio.Future[Applied] = asyncio.get_running_loop().create_future()
        self.state = _WAITING
        self.cancelled = False
        self.task: asyncio.Task[None] | None = None
        self.step: ReloadStep | None = None

    def take_pages(self) -> tuple[ChangePage, ...]:
        """the pages, let go here: the staging holds them until the hand-off alone.

        :return: the pages
        :rtype: tuple[ChangePage, ...]
        """
        pages, self._pages = self._pages, ()
        return pages

    def cancel(self) -> bool:
        """publish nothing of this reload, when its publish's turn has not started.

        :return: whether it is cancelled (False once its publish started, or it is done)
        :rtype: bool
        """
        if self.state == _DONE or self.done.done():
            return False
        if self.step is not None and self.step.publishing:
            return False
        self.cancelled = True
        if self.step is not None:
            # queued in the apply lane: its turn publishes nothing
            self.step.cancelled = True
        elif self.task is not None:
            self.task.cancel()
        self._on_cancel(self)
        return True


class ReloadLane:
    """a copy's reloads, staged one at a time in the reload worker beside ``lane`` and published through it in
    order (the module's docstring).

    Make it on the event loop it serves, as the pod starts or on a thread: its pool starts the fork server.

    :param backend: the copy's backend
    :ptype backend: SharedCopyBackend
    :param lane: the copy's apply lane: every reload publishes through it
    :ptype lane: ApplyLane
    :param check: hooks run in the reload worker over each reload's staged tables (a product's check of them); what
        they derive is dropped (derived tables are the publish's, in the apply worker). None for none
    :ptype check: ApplyHooks | None
    :param private_bytes: the reload worker is replaced after a reload leaving it more private memory than this
    :ptype private_bytes: int
    :param preload: modules the worker holds from its start (the check's among them)
    :ptype preload: Sequence[str]
    :param max_reloads: the most reloads waiting or staging
    :ptype max_reloads: int
    :param max_bytes: the most bytes of their pages
    :ptype max_bytes: int
    :param max_replay_bytes: the most bytes of changes kept to apply over one reload
    :ptype max_replay_bytes: int
    :raises ValueError: for a bound under one
    """

    def __init__(
        self,
        backend: SharedCopyBackend,
        lane: ApplyLane,
        *,
        check: ApplyHooks | None = None,
        private_bytes: int = _RELOAD_PRIVATE_BYTES,
        preload: Sequence[str] = (),
        max_reloads: int = _MAX_RELOADS,
        max_bytes: int = _MAX_BYTES,
        max_replay_bytes: int = _MAX_REPLAY_BYTES,
    ) -> None:
        if max_reloads < 1 or max_bytes < 1 or max_replay_bytes < 1:
            raise ValueError("a reload lane holds one reload and one byte at least")
        self._backend = backend
        self._lane = lane
        self._check = check
        self._max_reloads = max_reloads
        self._max_bytes = max_bytes
        self._max_replay_bytes = max_replay_bytes
        self._pool = SharedArrowPool(
            backend.store,
            workers=1,
            private_bytes=private_bytes,
            name="copy reload",
            preload=("threetears.core.collections.shared_copy", *preload),
            process_name=_RELOAD_PROCESS_NAME,
        )
        self._changed = asyncio.Condition()
        #: the reloads holding room (waiting for the worker, or staging): replaced whole, on the loop
        self._room_held: tuple[Reload, ...] = ()
        #: every reload not yet done: replaced whole, on the loop
        self._open: tuple[Reload, ...] = ()
        self._closing = False
        self._reloads = 0
        self._cancelled = 0
        self._failed = 0
        self._restarts = 0
        self._waits = 0
        self._last_error: str | None = None

    def status(self) -> ReloadStatus:
        """what the lane holds and has done.

        :return: the status
        :rtype: ReloadStatus
        """
        return ReloadStatus(
            waiting=sum(1 for reload in self._open if reload.state == _WAITING),
            staging=sum(1 for reload in self._open if reload.state == _STAGING),
            queued=sum(1 for reload in self._open if reload.state == _QUEUED),
            reloads=self._reloads,
            cancelled=self._cancelled,
            failed=self._failed,
            restarts=self._restarts,
            waits=self._waits,
            last_error=self._last_error,
        )

    def _room(self, nbytes: int) -> bool:
        if not self._room_held:
            return True
        held = sum(reload.nbytes for reload in self._room_held)
        return len(self._room_held) < self._max_reloads and held + nbytes <= self._max_bytes

    async def reload(
        self,
        pages: Sequence[ChangePage],
        *,
        tables: Collection[str] | None = None,
        scopes: Mapping[str, ReloadScope] = EMPTY,
        mark: int | None = None,
    ) -> Reload:
        """give a reload: its place in the copy's order is now (changes given after this call to its tables apply over
        it); it stages in the reload worker and publishes through the apply lane. Waits for room while the lane is
        full.

        :param pages: its rows pages (the tables' full, or their scope values' rows)
        :ptype pages: Sequence[ChangePage]
        :param tables: the tables it reloads (a table with no rows is reloaded empty); None for those its pages name
        :ptype tables: Collection[str] | None
        :param scopes: each table reloaded by scope and its values (``ReloadScope``); every other is reloaded whole
        :ptype scopes: Mapping[str, ReloadScope]
        :param mark: its mark, published with it unless a change since published a newer one
        :ptype mark: int | None
        :return: the reload: await :attr:`Reload.done` for what it published
        :rtype: Reload
        :raises RuntimeError: when the lane is closed
        :raises ValueError: for a reload naming no table
        """
        if self._closing:
            raise RuntimeError("the copy's reload lane is closed")
        named = frozenset(tables) if tables is not None else frozenset(page.table for page in pages)
        if not named:
            raise ValueError("a reload names the tables it reloads")
        # its place in the order: taken before any wait, so a change given after this call is kept for it
        watch = self._lane.watch(named, max_bytes=self._max_replay_bytes)
        reload = Reload(self._said_cancelled, tuple(pages), named, frozen(scopes.items()), mark, watch)
        self._open = (*self._open, reload)
        try:
            async with self._changed:
                if not self._room(reload.nbytes):
                    self._waits += 1
                    log.info(
                        "the copy's reload lane is full; a reload waits for room",
                        extra={"extra_data": {"held": len(self._room_held)}},
                    )
                    await self._changed.wait_for(lambda: self._closing or self._room(reload.nbytes))
                if self._closing:
                    raise RuntimeError("the copy's reload lane is closed")
                self._room_held = (*self._room_held, reload)
        except BaseException:
            self._lane.unwatch(watch)
            self._open = tuple(held for held in self._open if held is not reload)
            raise
        reload.task = asyncio.create_task(self._run(reload), name="copy-reload")
        return reload

    async def _let_room_go(self, reload: Reload) -> None:
        async with self._changed:
            self._room_held = tuple(held for held in self._room_held if held is not reload)
            self._changed.notify_all()

    async def _run(self, reload: Reload) -> None:
        staged: StagedReload | None = None
        try:
            if reload.cancelled:
                raise ReloadCancelled("the reload was cancelled before it staged")
            staged = await self._staged(reload)
            await self._let_room_go(reload)
            if reload.cancelled:
                raise ReloadCancelled("the reload was cancelled as it staged")
            if reload.watch.overrun:
                raise ReloadOverrun(
                    f"the changes given during the reload of {sorted(reload.tables)} passed the "
                    f"{reload.watch.max_bytes} bytes kept to apply over it; the reload is read again"
                )
            reload.step = ReloadStep(staged, reload.watch, reload.mark)
            reload.state = _QUEUED
            outcome = await self._lane.submit_reload(reload.step)
            # its turn starting is seen here only once done; the step says it to a cancel meanwhile
            applied = await asyncio.shield(outcome)
        except asyncio.CancelledError:
            self._ended(reload, ReloadCancelled("the reload was cancelled"))
            if not reload.cancelled:
                raise
        except ReloadCancelled as cancelled:
            self._ended(reload, cancelled)
        except (
            Exception
        ) as exc:  # prawduct:allow prawduct/broad-except -- the reload's caller hears it through its future
            self._failed += 1
            self._last_error = f"{type(exc).__name__}: {exc}"
            contain(exc, log=log, what="a reload of the copy failed; nothing of it published", level="error")
            self._ended(reload, exc)
        else:
            self._reloads += 1
            if not reload.done.done():
                reload.done.set_result(applied)
            reload.state = _DONE
        finally:
            self._lane.unwatch(reload.watch)
            self._open = tuple(held for held in self._open if held is not reload)
            await asyncio.shield(self._let_room_go(reload))
            if staged is not None:
                # published: the version holds the files now; else they go
                await asyncio.shield(staged.close())

    async def _staged(self, reload: Reload) -> StagedReload:
        """the reload staged in the reload worker, once it is free; its pages let go as they are handed over."""
        reload.state = _STAGING
        staging = self._backend.reload_run(
            self._pool, reload.take_pages(), tables=reload.tables, scopes=reload.scopes, check=self._check
        )
        try:
            return await staging
        except BrokenProcessPool as died:
            self._restarts += 1
            log.warning("the copy's reload worker died staging a reload; a fresh one stages the next")
            raise ReloadWorkerDied(
                f"the reload worker died staging the reload of {sorted(reload.tables)}; the reload is read again"
            ) from died

    def _ended(self, reload: Reload, exc: Exception) -> None:
        """a reload done with nothing published: its caller told (a cancel not taken through :meth:`Reload.cancel`
        counted here; a failure is said and counted by the caller)."""
        reload.state = _DONE
        if isinstance(exc, ReloadCancelled) and not reload.cancelled:
            self._cancelled += 1
        if not reload.done.done():
            reload.done.set_exception(exc)

    def _said_cancelled(self, reload: Reload) -> None:
        """a cancel taken (:meth:`Reload.cancel`): counted once."""
        self._cancelled += 1
        log.info("a reload of the copy cancelled", extra={"extra_data": {"tables": sorted(reload.tables)}})

    async def close(self) -> None:
        """take no more reloads, cancel those not yet publishing, wait for the rest, and stop the reload worker.

        :return: nothing
        :rtype: None
        """
        async with self._changed:
            self._closing = True
            self._changed.notify_all()
        open_now = self._open
        for reload in open_now:
            reload.cancel()
        tasks = [reload.task for reload in open_now if reload.task is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._pool.close()
