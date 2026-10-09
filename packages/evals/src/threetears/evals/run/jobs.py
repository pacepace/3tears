"""Async job system for eval operations.

Manages background asyncio tasks with progress tracking and concurrency
control. One persistence path: ``start_group`` runs evals (one run per arm), with
state in the database as ``EvalRun`` documents and ``progress`` written to the run's
``progress`` field.

There was a second — ``start_generation_job``, an in-memory path with a
terminal-status TTL, for template generation and bootstrap. Its callers were
an earlier eval surface, since retired, and it outlived them with no
production caller at all. Removed rather than kept as an
extension point: a generation path built against the current template shape
would not resemble that one, since the shape it generated no longer exists.

``start_task`` is the one other kind of job, and it has no persistence path here at all: it runs
work that already writes its OWN durable record (an analysis generation writes an
``EvalAnalysisAttempt`` however it ends), so the manager gives it only what a background job needs
from a process — tracking, a timeout, and cancellation at shutdown — and never a status of its own.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator, Sequence
from concurrent.futures import Executor
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, TypeVar

from threetears.evals.contracts.errors import AdmissionRefusedError, ConflictError, StorageError
from threetears.evals.contracts.models import TERMINAL_RUN_STATUSES, EvalRun
from threetears.evals.contracts.scoring import summarize_completeness
from threetears.evals.contracts.storage import JobStore
from threetears.evals.contracts.offload import run_blocking, wait_through_cancellation
from threetears.evals.run.run_document import update_eval_run
from threetears.observe import get_logger

log = get_logger(__name__)

# The CODE default only. A host with its own configured cap passes it at construction. This
# constant exists so a manager built without an explicit cap (tests, the docstring example
# above) still gets a bounded value rather than an unbounded one; a host that ships its own
# default pins the two equal in its own tests.
MAX_CONCURRENT_JOBS = 2
# The manager's budget for a launch-group member that carries no timeout of its own. A launched run
# always carries one sized to its matrix (``adaptive_job_timeout_s``), so this binds only a caller that
# passes ``None``, and a manager built without an explicit ``job_timeout_s``.
DEFAULT_JOB_TIMEOUT_S = 3600  # 1 hour
# How long :meth:`EvalJobManager.shutdown` waits for cancelled jobs to write their terminal status.
# A host with its own configured value passes it; this is the default for a caller that does not.
SHUTDOWN_SETTLE_TIMEOUT_S = 5.0

# Adaptive eval-job timeout. Eval cells run SEQUENTIALLY — one `await` per
# cell over the run's whole matrix, in the shuffled order ``runner.cell_execution_order``
# derives (it was a nested k×model×test_case loop when this was written; the nesting now
# only BUILDS the cell list, which is then shuffled, so the execution order is no longer
# nested. The arithmetic below is untouched by that: it depends on the cells being serial
# and on N being the full matrix, both of which still hold)
# — each bounded by ``RunnerOptions.cell_timeout_s``
# (600s — already covers candidate turn + delivery drain + judging). So a job's
# worst-case wall-clock is ``N × cell_timeout_s``; a fixed 3600s backstop falls
# below that at N>6 and guillotines cells that are each within budget (a large
# matrix is cut off before its last cells run). We size the job budget to the matrix instead:
# ``clamp(N × per_result_s × margin, floor, cap)``. ``per_result_s`` is the
# caller's real ``cell_timeout_s`` (single source of truth — passed in, never a
# duplicated constant).
JOB_TIMEOUT_SAFETY_MARGIN = 1.1  # per-cell loop overhead (storage save, broadcast) + scheduling jitter
JOB_TIMEOUT_FLOOR_S = 600.0  # a job always gets at least one cell's worth
JOB_TIMEOUT_CAP_S = 28800.0  # 8h (~43 cells @ 600s) — backstop against a fat-fingered/unbounded matrix


def adaptive_job_timeout_s(
    n_results: int,
    per_result_s: float,
    *,
    margin: float = JOB_TIMEOUT_SAFETY_MARGIN,
    floor_s: float = JOB_TIMEOUT_FLOOR_S,
    cap_s: float = JOB_TIMEOUT_CAP_S,
) -> float:
    """Size an eval job's wall-clock budget to its matrix.

    Because cells run sequentially and each is independently capped at
    ``per_result_s``, the job's legitimate worst case is ``n_results ×
    per_result_s`` plus small loop overhead (the ``margin``). Clamped to
    ``[floor_s, cap_s]`` so a single-cell run still gets a floor and a
    pathological matrix can't request an unbounded budget.

    Args:
        n_results: Total cells in the run — ``len(test_cases) × len(models) × k_runs``.
        per_result_s: Per-cell wall-clock ceiling (the runner's ``cell_timeout_s``).
        margin: Multiplier for per-cell loop overhead and scheduling jitter.
        floor_s: Minimum budget, regardless of matrix size.
        cap_s: Absolute maximum budget, regardless of matrix size.

    Returns:
        The job timeout in seconds, within ``[floor_s, cap_s]``.
    """
    raw = n_results * per_result_s * margin
    return max(floor_s, min(cap_s, raw))


# Work function: async fn(progress_callback) -> None
# The progress callback is async fn(dict) -> None
ProgressFn = Callable[[dict[str, Any]], Awaitable[None]]
WorkFn = Callable[[ProgressFn], Awaitable[None]]


class EvalJobTimeout(Exception):
    """A job outlived the wall-clock budget its timeout context was enforcing.

    Engine-owned on purpose. The manager needs one type to branch on, so that a
    breached budget records ``failed`` with a timeout message instead of falling
    through to the boundary that catches everything — and the context manager
    enforcing the budget comes from the host. A type the host owned would make
    the engine name a module it cannot be installed without.

    Attributes:
        budget_s: The budget that was breached, in seconds.
        elapsed_s: Wall-clock seconds spent before it fired.
        attribution: What the host names as innermost when the deadline hit — the
            operation itself when nothing was nested inside it, for a host whose
            timeouts carry names. ``None`` only when the implementation has no
            name to give, which is the case for :func:`default_job_timeout`:
            plain ``asyncio.timeout`` has no vocabulary of operations at all.
    """

    def __init__(self, budget_s: float, elapsed_s: float, attribution: str | None = None) -> None:
        """Record the breached budget, the time spent under it, and the host's attribution."""
        super().__init__(f"job exceeded its {budget_s:.0f}s budget after {elapsed_s:.0f}s")
        self.budget_s = budget_s
        self.elapsed_s = elapsed_s
        self.attribution = attribution


class JobTimeoutFactory(Protocol):
    """Builds the context manager a single job runs inside.

    An async timeout is generic machinery rather than eval domain knowledge, and
    a host that already owns one — a registry of named operation budgets, say,
    with its own telemetry and its own nesting rules — wants the job to run under
    that one rather than beside it. So the engine takes the context manager as a
    value and ships a default (:func:`default_job_timeout`) for a host that has
    none.

    **Called once per job, with that job's own resolved budget**, never once per
    manager: two runs in one process size their matrices differently
    (:func:`adaptive_job_timeout_s`) and must not share a deadline.

    Three obligations, and each is something the manager relies on:

    * Reject a non-positive budget **on entry, before the body runs**. The
      manager tells "this run never got going" from "this run ran" by whether
      its work function started, and a budget rejected after the body had begun
      would make that derivation lie.
    * Raise :class:`EvalJobTimeout` when the budget is breached, so the manager
      records a timeout rather than a generic failure.
    * Let everything else through untouched, cancellation included — an operator
      cancelling a job is not a timeout and must reach its own branch.

    **The manager binds nothing from the yield** — it enters the context with no
    ``as`` — so the type is ``Any`` rather than the budget. An implementation may
    yield whatever its own layer yields (``operation_timeout`` yields the computed
    budget, and its other callers do read it); the point of saying so here is that
    a host owes this port three behaviours and no value.
    """

    def __call__(self, budget_s: float) -> AbstractAsyncContextManager[Any]:
        """Return the context manager that enforces ``budget_s`` for one job."""
        ...


@asynccontextmanager
async def default_job_timeout(budget_s: float) -> AsyncIterator[float]:
    """Enforce a job budget with :func:`asyncio.timeout` and nothing else.

    What the engine ships to a host that supplies no timeout of its own.
    Deliberately thinner than what a host with a timeout registry injects — no
    named operation, no nesting, no attribution — because those are the host's
    concerns, and an engine that assumed them would not install without one.

    Args:
        budget_s: Wall-clock ceiling for the body, in seconds.

    Yields:
        The budget, so a caller can honour it at an inner layer.

    Raises:
        ValueError: If ``budget_s`` is not positive — on entry, before the body
            runs. See :class:`JobTimeoutFactory` for why the ordering is part of
            the contract rather than an implementation detail.
        EvalJobTimeout: If the body outlives the budget.
    """
    if budget_s <= 0:
        raise ValueError(f"job timeout budget must be positive, got {budget_s}")
    started = time.monotonic()
    deadline = asyncio.timeout(budget_s)
    try:
        async with deadline:
            yield budget_s
    except TimeoutError as exc:
        # A body that raises TimeoutError of its own is not this budget firing, and
        # converting it would report a job as having outlived a budget it was inside.
        if not deadline.expired():
            raise
        raise EvalJobTimeout(budget_s=budget_s, elapsed_s=time.monotonic() - started) from exc


_T = TypeVar("_T")


async def _finish_even_if_cancelled(write: Coroutine[Any, Any, _T]) -> _T:
    """Await a storage write to its end, absorbing cancellations aimed at the waiter.

    A terminal status write used to be uninterruptible for free: it was a
    synchronous call on the event loop, so nothing else — ``shutdown``'s
    ``task.cancel()`` included — could run between its first statement and its
    last. Moving the round-trip to a worker thread makes it an await point, and
    an await point inside a job that is already unwinding from one cancel is
    exactly where a second one lands: ``cancel_job`` followed by ``shutdown``
    cancels the same task twice, and the second cancel is delivered to whatever
    the ``except asyncio.CancelledError`` branch is waiting on.

    What that costs, precisely: the worker is usually already running, so the
    write itself normally lands, but its *outcome is never read* — the refusal
    branch that warns about a run left reading ``running`` never runs, and
    neither does the terminal broadcast, so every live reader keeps showing a
    run that has ended. And when the executor has not yet started the call, the
    cancel reaches the pending work item and the write is lost outright,
    reclaimed later as *abandoned*: the mislabel
    :meth:`EvalJobManager._set_status` already warns about for a refused write,
    reached by a different road.

    So a terminal write is shielded and re-awaited until it finishes. Only the
    caller's own cancels are absorbed; a cancellation of the write itself still
    propagates. The caller re-raises its own ``CancelledError`` afterwards, so
    the job still ends cancelled — it just ends with its status recorded.

    Non-terminal writes deliberately do not use this. A cancel that lands on the
    ``running`` flip falls through to the cancel branch, which writes a terminal
    status over it anyway, and a dropped progress tick is superseded by the next
    cell's — both already stated where they are written.

    Args:
        write: The coroutine performing the write.

    Returns:
        Whatever ``write`` returns.

    Raises:
        asyncio.CancelledError: If ``write`` itself is cancelled. A cancel aimed at the
            CALLER is absorbed, including when the write has already finished — the
            distinction is whether the write task is cancelled, not whether it is done.
        Exception: Whatever ``write`` raises. A genuine storage failure propagates to the
            caller rather than being masked as a cancellation.
    """
    task = asyncio.ensure_future(write)
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # ``cancelled()``, never ``done()``: a task that COMPLETED is also done, and
            # re-raising there throws away a result that is sitting right there. That
            # interleaving is reachable — the write finishes and schedules this shield's
            # future, the caller is cancelled before it resumes, and the cancel is
            # delivered at the next step into a wait on an already-done future.
            if task.cancelled():
                raise


class _GroupSlot:
    """One concurrency slot a launch group holds from its first member's start to its last member's end.

    The arms of one campaign launch are one experiment: measured apart in time, a difference between
    them can be the provider's afternoon rather than the setting. Admitting them one semaphore slot
    each would park all but ``max_concurrent`` of them, so the group takes ONE slot and every member
    runs inside it. The slot is released when every member has ended, including a member cancelled
    before it started — which is why release is counted on :meth:`leave`, called when each member's
    task is done, and not on leaving the ``async with``.
    """

    def __init__(self, semaphore: asyncio.Semaphore, members: int) -> None:
        """Bind the group to the manager's semaphore.

        Args:
            semaphore: The manager's concurrency semaphore.
            members: How many jobs the group holds; each must call :meth:`leave` exactly once.
        """
        self._semaphore = semaphore
        self._remaining = members
        self._held = False
        self._acquiring = asyncio.Lock()

    @asynccontextmanager
    async def hold(self) -> AsyncIterator[None]:
        """Enter the group's slot, acquiring it for the group if no member has yet."""
        async with self._acquiring:
            if not self._held:
                await self._semaphore.acquire()
                self._held = True
        yield

    def leave(self) -> None:
        """Record one member's end; the last one out releases the slot."""
        self._remaining -= 1
        if self._remaining == 0 and self._held:
            self._held = False
            self._semaphore.release()


class AdmissionTicket:
    """Room a launch has reserved for runs it has not yet handed to the manager.

    Counted as admitted from the moment it is issued, so a launch preparing its arms — which
    awaits snapshots, generation and storage for seconds or minutes — holds its place against
    every launch that arrives meanwhile. Without that, two launches could each pass the check
    and then both start, which is the unbounded queue the check exists to refuse.

    Released when the launch either starts its runs (they are then counted as live tasks) or
    abandons them. A caller admitted for several launches hands each its share with :meth:`split`.
    """

    def __init__(self, manager: EvalJobManager, runs: int) -> None:
        """Bind the ticket to the manager whose reservation count it holds a share of.

        Args:
            manager: The manager that issued it.
            runs: How many runs it reserves.
        """
        self._manager = manager
        self._held = runs

    @property
    def held(self) -> int:
        """Runs this ticket still reserves."""
        return self._held

    def split(self, runs: int) -> AdmissionTicket:
        """Move ``runs`` of this reservation onto a ticket of their own, admitting nothing new.

        For a caller admitted for several launches at once: each launch takes its share without
        asking the manager for room, so a launch that arrived after the caller cannot take the room
        the caller was already given, and the caller's later launches cannot be refused part-way.

        Args:
            runs: How many runs the new ticket reserves.

        Returns:
            A ticket holding ``runs`` of this one's reservation.

        Raises:
            ValueError: This ticket holds fewer than ``runs`` — the caller admitted less than it
                is now starting, which no caller input can cause.
        """
        if runs > self._held:
            raise ValueError(f"cannot split {runs} run(s) off an admission holding {self._held}")
        self._held -= runs
        return AdmissionTicket(self._manager, runs)

    def release(self) -> None:
        """Hand back whatever this ticket still reserves. Idempotent, so a ``finally`` may always call it."""
        self._manager.release_reservation(self._held)
        self._held = 0


class EvalJobManager:
    """Manages background async eval jobs.

    ``start_group`` runs evals (one run per arm), each saved before its task starts;
    state lives in the database, and one task pool and one concurrency
    semaphore serve them. ``start_task`` runs detached work that records its own
    ending (an analysis generation), outside that semaphore and outside run admission.

    **A launching host does not build one.** :class:`~threetears.evals.run.launch.LaunchHost` builds
    its job manager over its ``EvalHost``'s storage and executor, so the store a run's status is
    written to is the store its results are; the shape below is the manager on its own.

    Usage — the wired shape, because it is the one production uses and the one
    ``test_every_non_test_eval_job_manager_site_wires_a_timeout_factory`` requires. The
    ``job_timeout_factory`` argument does have a default (:func:`default_job_timeout`, plain
    ``asyncio.timeout``), and a host with no timeout layer of its own is meant to take it;
    a host that has one would, by leaving it off, lose the operation budget, the ERROR
    telemetry and the attribution with nothing going red at runtime::

        mgr = EvalJobManager(storage, job_timeout_factory=host_job_timeout, blocking_executor=host_io_pool)
        run = assembled  # an EvalRun, as launch_run assembles it

        async def work(progress):
            for i in range(10):
                await do_some_work(i)
                await progress({"completed": i + 1, "total": 10})

        await mgr.start_group([(run, work, None)])
    """

    def __init__(
        self,
        storage: JobStore,
        max_concurrent: int = MAX_CONCURRENT_JOBS,
        on_progress: Callable[[str, dict[str, Any]], None] | None = None,
        job_timeout_s: float = DEFAULT_JOB_TIMEOUT_S,
        *,
        job_timeout_factory: JobTimeoutFactory | None = None,
        blocking_executor: Executor | None = None,
    ):
        """Wire the job manager to storage and configure concurrency.

        Args:
            storage: The run document store this manager persists ``EvalRun``
                progress and status through — the two conditional-write
                primitives and nothing else.
            max_concurrent: Cap on simultaneously running jobs (semaphore-gated).
            on_progress: Optional ``(job_id, progress_dict)`` callback fired
                on every progress update — typically wired to a WebSocket
                broadcast.
            job_timeout_s: The wall-clock budget, in seconds, for a launch-group
                member that carries none of its own; jobs that exceed it are
                cancelled and recorded as failed.
            job_timeout_factory: Builds the context manager each job runs
                inside, called with that job's resolved budget. Defaults to
                :func:`default_job_timeout`, which is asyncio and nothing else;
                a host with its own timeout machinery injects a factory built
                on it, and :class:`JobTimeoutFactory` states what that factory
                owes the manager. Keyword-only so a construction site names it,
                which is what
                ``tests/test_construction_site_wiring.py`` checks
                for on every non-test site.
            blocking_executor: Where the manager's storage round-trips run — the
                run saves a launch starts with, and every status and progress
                write. ``None`` is the loop's default executor. A host whose
                default executor serves something that must not queue behind
                eval traffic (a liveness probe, say) passes a pool of its
                own; the same construction-site gate requires every non-test
                site to name one.
        """
        self._storage = storage
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._on_progress = on_progress
        self._job_timeout_s = job_timeout_s
        self._job_timeout_factory: JobTimeoutFactory = job_timeout_factory or default_job_timeout
        self._blocking_executor = blocking_executor
        # Operator-supplied cancellation reasons keyed by job id. Written by
        # cancel_job, consumed (popped) by the job's own CancelledError
        # boundary — the only frame that can attach the reason to the
        # terminal status write.
        self._cancel_reasons: dict[str, str] = {}
        # Whether each in-flight run's work function has started, keyed by run id.
        # Set False when the job is admitted and True immediately before the work
        # function is awaited, so a terminal status write can tell "this run never
        # got going" from "this run ran and its own record is the authority" —
        # see _set_status, which derives the empty-matrix stamp from this rather
        # than taking it from each except-branch.
        self._work_ran: dict[str, bool] = {}
        # Runs admitted by a launch that has not yet handed them over — see AdmissionTicket.
        self._reserved = 0
        # Detached tasks (start_task), kept apart from ``_tasks`` because everything that reads
        # ``_tasks`` means RUNS: admission counts it against the run limit and a run with no live
        # task there is repaired as abandoned. ``_task_keys`` holds each live task's exclusivity key.
        self._detached: dict[str, asyncio.Task[None]] = {}
        self._task_keys: dict[str, str] = {}
        # Exclusivity keys held by work running in its CALLER's frame rather than as a task (see
        # hold), keyed by holder id. Read beside ``_task_keys`` wherever a key's holders are asked for.
        self._held_keys: dict[str, str] = {}

    def _safe_broadcast(self, job_id: str, payload: dict[str, Any]) -> None:
        """Invoke ``self._on_progress`` defensively.

        The broadcast callback is best-effort — failures must be logged
        but never propagate, otherwise a raising broadcast would escape
        the status-set path and be re-caught by the top-level job
        boundary as a job failure (corrupting successful job state).
        """
        if self._on_progress is None:
            return
        try:
            self._on_progress(job_id, payload)
        except Exception:  # prawduct:ok-broad-except — best-effort broadcast must not corrupt job state
            log.warning("on_progress broadcast failed for job %s", job_id, exc_info=True)

    @property
    def active_count(self) -> int:
        """Number of currently active (not done) jobs."""
        return sum(1 for t in self._tasks.values() if not t.done())

    @property
    def admitted_count(self) -> int:
        """Runs admitted and unfinished: live tasks, pending or running, plus launches' reservations."""
        return self.active_count + self._reserved

    def admit(self, runs: int, *, limit: int, limit_name: str) -> AdmissionTicket:
        """Reserve room for ``runs`` more runs, or refuse if that would pass ``limit``.

        The semaphore bounds how many launch groups RUN at once, and nothing else bounded how
        many were admitted: every run became a task the moment its launch started it, so a burst
        of launches queued without limit behind the semaphore, each holding its prepared
        snapshot, clients and closures in this process. This is that bound, and it refuses rather
        than queues — a caller told "no" can retry, while a queue is a promise this process cannot
        keep across the restart that an unbounded one invites.

        Synchronous and without an await, so the check and the reservation cannot be separated by
        another launch.

        Args:
            runs: How many runs the launch will admit.
            limit: The most runs that may be admitted and unfinished at once. Passed per call,
                because the host's value is hot-reloadable.
            limit_name: What an operator sets to change ``limit``, for the refusal to name.

        Returns:
            A ticket holding the reservation; release it when the runs are started or abandoned.

        Raises:
            AdmissionRefusedError: Admitting ``runs`` would pass ``limit``.
        """
        admitted = self.admitted_count
        if admitted + runs > limit:
            raise AdmissionRefusedError(requested=runs, admitted=admitted, limit=limit, limit_name=limit_name)
        self._reserved += runs
        return AdmissionTicket(self, runs)

    def release_reservation(self, runs: int) -> None:
        """Return ``runs`` of admitted room — called by :meth:`AdmissionTicket.release`, never directly.

        The reservation count is written only here and in :meth:`admit`, so the number admission
        rests on has one owner; a ticket hands its share back through this rather than reaching
        into the count.

        Args:
            runs: How many reserved runs to hand back; zero for a ticket already released.
        """
        self._reserved -= runs

    def is_active(self, job_id: str) -> bool:
        """Check if a job is currently tracked and not done."""
        task = self._tasks.get(job_id)
        return task is not None and not task.done()

    def get_active_job_ids(self) -> list[str]:
        """Get IDs of all active jobs."""
        return [jid for jid, t in self._tasks.items() if not t.done()]

    async def wait_for(self, job_ids: Sequence[str]) -> None:
        """Wait until every named run's job has ended, however it ended.

        The awaitable a caller that launched runs and must read their results holds — a script, a
        CLI, a test. It returns when each job has written its terminal status, so the run's stored
        ``status`` is the answer to how it went; nothing is re-raised here, because a job records
        its own ending and a failed run is a result, not an error of the wait.

        **Waiting is not owning.** Cancelling the waiter leaves the jobs running, exactly as an
        operator's view that stops polling does; a caller that owns the runs and must not leave
        them behind cancels them itself (:meth:`cancel_job`, or :meth:`shutdown`).

        A job this manager is not running returns at once: one that has ended is popped from the
        manager as it ends, and the two cases cannot be told apart here, so the run's stored status
        is the authority either way.

        Args:
            job_ids: The runs' ids, as :meth:`start_group` returned them.
        """
        tasks = [task for job_id in job_ids if (task := self._tasks.get(job_id)) is not None]
        if tasks:
            await asyncio.wait(tasks)

    def cancel_job(self, job_id: str, reason: str | None = None) -> bool:
        """Cancel a running job.

        Cancellation is request-then-converge: the task's own
        ``CancelledError`` boundary writes the terminal ``cancelled``
        status (with ``reason``, when given) as it unwinds — typically
        at its next await point.

        Args:
            job_id: Run id to cancel.
            reason: Optional operator-facing reason recorded on the terminal
                status write, always to ``EvalRun.cancellation_reason`` and never
                to ``error_details``. Every run :meth:`start_group` starts is saved
                before its task exists, so there is always a document to write it to.

        Returns:
            True if cancellation was requested, False if the job
            is not found or already done.
        """
        task = self._tasks.get(job_id)
        if task is None or task.done():
            return False
        if reason is not None:
            self._cancel_reasons[job_id] = reason
        task.cancel()
        return True

    async def shutdown(self, timeout: float = SHUTDOWN_SETTLE_TIMEOUT_S) -> list[str]:
        """Cancel all active jobs and wait up to ``timeout`` for each to settle.

        A cancelled job settles by writing its terminal status (and, for a run, its completeness
        record) as it unwinds. One whose writes outlast ``timeout`` is given up on, and what that
        leaves depends on its kind, so the two are logged apart by id and returned rather than
        folded into a success: a RUN keeps its non-terminal status in storage until a later process
        relabels it abandoned, while a detached TASK (:meth:`start_task`) has no stored status at all
        — its work had not recorded its ending (a generation writes no attempt), nothing at the next
        boot repairs it, and what it spent is only where its work recorded each call as it went (an
        analysis generation writes every model call to the out-of-run spend ledger).

        Args:
            timeout: Seconds to wait for every cancelled job to finish.

        Returns:
            The ids of the jobs still unfinished when the wait ended; empty when all settled.
        """
        # Detached tasks are cancelled with the runs: each records its own ending as it unwinds
        # (a generation writes a `cancelled` attempt), which is the settling this waits for.
        every = {**self._tasks, **self._detached}
        for task in list(every.values()):
            if not task.done():
                task.cancel()
        if every:
            await asyncio.wait(list(every.values()), timeout=timeout)
        unsettled_runs = [job_id for job_id, task in self._tasks.items() if not task.done()]
        unsettled_tasks = [job_id for job_id, task in self._detached.items() if not task.done()]
        if unsettled_runs:
            log.warning(
                "eval.shutdown gave up on %d job(s) after %.1fs — their runs keep a non-terminal status "
                "until the next boot relabels them abandoned: %s",
                len(unsettled_runs),
                timeout,
                ", ".join(unsettled_runs),
            )
        if unsettled_tasks:
            log.warning(
                "eval.shutdown gave up on %d background task(s) after %.1fs — none is a run: their work "
                "recorded no ending (a memo generation stored no attempt), nothing at the next boot "
                "repairs that, and what they spent is not recorded: %s",
                len(unsettled_tasks),
                timeout,
                ", ".join(unsettled_tasks),
            )
        unsettled = unsettled_runs + unsettled_tasks
        self._tasks.clear()
        self._detached.clear()
        self._task_keys.clear()
        return unsettled

    def start_task(
        self, job_id: str, work: Callable[[], Awaitable[None]], *, budget_s: float, key: str | None = None
    ) -> str:
        """Run ``work`` in the background under a timeout, tracked until it ends.

        For work that writes its own durable record of how it ended — the manager writes nothing,
        so a caller asking "how did it go?" reads that record, and asks the manager only "is it
        still going?" (:meth:`is_task_active`). A task is never persisted here, so one alive at a
        restart leaves no trace in the manager, and whatever its work had not yet recorded is lost
        with it.

        A task does NOT take a slot from the run semaphore: a run holds its slot for as long as its
        whole matrix takes, and short work queued behind two of them would wait hours for no reason
        of its own. Nor does it count toward run admission.

        Args:
            job_id: The task's id; the caller's record of the work should carry the same id, so
                the two can be joined once the task is gone.
            work: The coroutine function to run. Whatever it raises ends the task and is logged,
                never re-raised — the work's own record is where its ending is stated.
            budget_s: The wall-clock ceiling, entered through this manager's timeout factory. The
                caller derives it from the work's own inner ceilings.
            key: An exclusivity key: while a task with this key is live, starting another with
                the same key is refused. ``None`` for no exclusivity.

        Returns:
            ``job_id``.

        Raises:
            ConflictError: A live task already holds ``key``.
        """
        if key is not None:
            self._refuse_held(key)
        task = asyncio.create_task(self._run_task(job_id, work, budget_s), name=f"eval-task-{job_id[:8]}")
        self._detached[job_id] = task
        if key is not None:
            self._task_keys[job_id] = key
        return job_id

    def is_task_active(self, job_id: str) -> bool:
        """Whether a :meth:`start_task` task is tracked and not yet done."""
        task = self._detached.get(job_id)
        return task is not None and not task.done()

    def cancel_task(self, job_id: str) -> bool:
        """Request cancellation of a :meth:`start_task` task — request-then-converge, as :meth:`cancel_job` is.

        The task's work records its own ending as it unwinds (a generation writes a ``cancelled``
        attempt), so nothing is written here; a caller confirms the ending by reading that record.

        Args:
            job_id: The task's id, as :meth:`start_task` returned it.

        Returns:
            True when cancellation was requested; False when no such task is tracked or it has
            already ended.
        """
        task = self._detached.get(job_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    def active_task_ids(self, key: str) -> list[str]:
        """The ids of everything holding ``key`` now: live :meth:`start_task` tasks, then live :meth:`hold` holders."""
        tasks = [job_id for job_id, held in self._task_keys.items() if held == key and self.is_task_active(job_id)]
        return tasks + [holder_id for holder_id, held in self._held_keys.items() if held == key]

    @contextmanager
    def hold(self, key: str, holder_id: str) -> Iterator[None]:
        """Hold an exclusivity key for work its caller runs inline, refused exactly as :meth:`start_task` is.

        The same key space as a task's ``key``: while held, :meth:`start_task` and another
        :meth:`hold` of that key are refused, and :meth:`active_task_ids` reports ``holder_id``.
        Taken and released on the loop with no await between the check and the take, so two
        callers cannot both hold it.

        Args:
            key: The exclusivity key.
            holder_id: The id the holder is reported under, as a task's ``job_id`` is.

        Yields:
            Nothing; the key is held for the ``with`` block and released however it ends.

        Raises:
            ConflictError: A live task or another holder already holds ``key``.
        """
        self._refuse_held(key)
        self._held_keys[holder_id] = key
        try:
            yield
        finally:
            self._held_keys.pop(holder_id, None)

    def _refuse_held(self, key: str) -> None:
        """Refuse a new holder of ``key`` while anything holds it — the one check :meth:`start_task` and :meth:`hold` share."""
        holders = self.active_task_ids(key)
        if holders:
            raise ConflictError(f"task '{holders[0]}' is already running for {key}")

    async def _run_task(self, job_id: str, work: Callable[[], Awaitable[None]], budget_s: float) -> None:
        """Execute one detached task under its budget; log how it ended if it did not end cleanly."""
        try:
            async with self._job_timeout_factory(budget_s):
                await work()
        except asyncio.CancelledError:
            raise
        except EvalJobTimeout as timed_out:
            log.error(
                "Task %s timed out after %.0fs (attribution=%s)", job_id, timed_out.elapsed_s, timed_out.attribution
            )
        # prawduct:ok-broad-except — top-level task boundary: the work records its own ending, so this only logs
        except Exception:
            log.exception("Task %s failed", job_id)
        finally:
            # Only after the work has returned or raised: a caller reading "not active" must find
            # the work's own record already written.
            self._detached.pop(job_id, None)
            self._task_keys.pop(job_id, None)

    async def start_group(self, members: Sequence[tuple[EvalRun, WorkFn, float | None]]) -> list[str]:
        """Start several runs as one group: they share one concurrency slot and start together.

        Every run is saved before any task starts, so a storage refusal leaves nothing running.

        Args:
            members: ``(run, work, job_timeout_s)`` per run; a ``None`` timeout falls back to the manager's default.

        Returns:
            The run ids, in the order given.

        Raises:
            ValueError: ``members`` is empty.
            asyncio.CancelledError: The caller was cancelled while the runs were being saved; they
                were settled ``cancelled`` and none started.
        """
        if not members:
            raise ValueError("a launch group needs at least one run")
        runs = [run for run, _, _ in members]
        # Every save, and the settling of a partial set, in ONE hop off the loop, waited for to its end
        # even through a cancel: a run saved ``pending`` with no task behind it waits for a slot
        # nothing will ever give it, and a cancel landing between two saves used to be impossible only
        # because the saves were synchronous on the loop. A launch cancelled while its runs were being
        # saved gets them settled rather than started, and then its cancellation back.
        saving = asyncio.ensure_future(run_blocking(self._blocking_executor, self._save_group, runs))
        if await wait_through_cancellation(saving):
            # A failed save has already settled what it saved; only a complete one needs settling here.
            if saving.exception() is None:
                await _finish_even_if_cancelled(
                    run_blocking(
                        self._blocking_executor,
                        self._settle_unstarted,
                        runs,
                        "its launch was cancelled before it started",
                    )
                )
            raise asyncio.CancelledError
        saving.result()
        slot = _GroupSlot(self._semaphore, len(members))
        for run, work, job_timeout_s in members:
            task = asyncio.create_task(
                self._run_job(run.id, run.scope_id, work, job_timeout_s, slot=slot),
                name=f"eval-{run.id[:8]}",
            )
            # Released on the TASK's end, not in the coroutine's ``finally``: a task cancelled before
            # its first step never runs its body, and its share of the slot would never come back.
            task.add_done_callback(lambda _task: slot.leave())
            self._tasks[run.id] = task
        return [run.id for run, _, _ in members]

    def _save_group(self, runs: list[EvalRun]) -> None:
        """Save every run of a group, or settle the ones already saved and re-raise.

        The blocking half of :meth:`start_group`, whole so it takes one worker hop.

        Args:
            runs: The group's runs, in the order they are started.
        """
        saved: list[EvalRun] = []
        try:
            for run in runs:
                self._storage.save_eval_run(run)
                saved.append(run)
        # prawduct:allow prawduct/broad-except -- cleanup-then-reraise: whatever stopped a sibling's save, the runs already saved must not be left pending
        except Exception:
            self._settle_unstarted(saved, "its launch group did not start: a sibling run could not be saved")
            raise

    def _settle_unstarted(self, runs: list[EvalRun], reason: str) -> None:
        """Save runs that were saved ``pending`` and will never start as ``cancelled``, with ``reason``.

        A run saved as ``pending`` with no task would wait for a slot nothing will ever give it.
        Each is settled on its own: one that cannot be saved is logged, and must neither stop the
        rest being settled nor replace the error that abandoned the group.

        Args:
            runs: The runs to settle.
            reason: The cancellation reason each records.
        """
        for run in runs:
            run.status = "cancelled"
            run.cancellation_reason = reason
            try:
                self._storage.save_eval_run(run)
            except ConflictError, StorageError:
                log.exception("eval.start_group run=%s left pending: its cancellation could not be saved", run.id)

    async def _run_job(
        self,
        run_id: str,
        scope_id: str,
        work: WorkFn,
        job_timeout_s: float | None = None,
        *,
        slot: _GroupSlot | None = None,
    ) -> None:
        """Execute a job with lifecycle management and timeout.

        ``job_timeout_s`` is the resolved per-job budget; ``None`` falls back to
        the manager-level default. ``slot`` is the launch group's shared slot, taken in
        place of one of the manager's own.
        """
        from threetears.evals.run.budget import AccountExhaustedError, BudgetStoppedError

        budget_s = job_timeout_s if job_timeout_s is not None else self._job_timeout_s
        # A job waits here for a slot, and jobs are cancellable while they wait: a
        # sweep of ten arms parks eight at the semaphore. Whether the work function
        # ever ran decides who owes the completeness record — it records its own in
        # a ``finally``, but a task that dies before that first await leaves no one
        # to, and the run would go terminal carrying nothing. This lives on the
        # manager rather than in a local so every terminal write below derives it,
        # including branches added after this was written.
        self._work_ran[run_id] = False
        try:
            async with slot.hold() if slot is not None else self._semaphore:
                await self._set_status(run_id, scope_id, "running")

                async def progress_fn(progress: dict[str, Any]) -> None:
                    await self._update_progress(run_id, scope_id, progress)

                async with self._job_timeout_factory(budget_s):
                    self._work_ran[run_id] = True
                    await work(progress_fn)
                await self._set_status(run_id, scope_id, "completed")

        except asyncio.CancelledError:
            # The reason goes to its own channel, not to ``error_details``: a human
            # decided to stop this, which is neither a failure nor a success, and
            # filing it beside genuine harness errors made every later triage read
            # a controlled stop as breakage.
            await self._set_status(
                run_id,
                scope_id,
                "cancelled",
                cancellation_reason=self._cancel_reasons.get(run_id),
            )
            raise
        except BudgetStoppedError as stop:
            # Graceful mid-run cost-cap stop: already-delivered results
            # are persisted. This is an honest terminal outcome, NOT an infra
            # failure — record it as its own ``budget_stopped`` status.
            log.info("Job %s stopped by per-run cost cap: %s", run_id, stop)
            # The reason goes to its own channel, not to ``error_details``, on the
            # cancel's precedent: a cap the operator configured doing what it was
            # configured to do is a designed outcome, and every run carries a cap
            # (an omitted override inherits the configured default), so filing it
            # as an error gave every capped run that bound a phantom in the count
            # operators scan for real faults.
            await self._set_status(run_id, scope_id, "budget_stopped", budget_stop_reason=str(stop))
        except AccountExhaustedError as exhausted:
            # The provider account paying for the run refused a call — out of credit, or
            # the key refused. Delivered results are persisted, the refused cell's
            # included, and the loop launched nothing after it. Its own status, never ``failed``:
            # the harness did not break and the candidate did not fail, so neither the harness
            # triage that reads ``failed`` nor any quality view should see this run as theirs.
            #
            # The reason goes to ``error_details``, NOT to a channel of its own as the two designed
            # stops do: nobody configured the account to run dry, and an operator has to act on it
            # (top up, raise the key's limit) before another launch can measure anything. The
            # message carries the spend the run had already incurred.
            log.warning("Job %s stopped: %s", run_id, exhausted)
            await self._set_status(run_id, scope_id, "exhausted", error=str(exhausted))
        except EvalJobTimeout as timed_out:
            # Keyed `attribution=` and not `inner_op=`, which is what every other timeout line in
            # this app emits. The value here is "the innermost operation the host's timeout
            # layer could name, falling back to the operation that fired" — those emitters'
            # `inner_op` is empty in exactly the case this one reads `eval_job`, so borrowing the
            # key would answer their grep with a value none of them can produce. Nothing is lost:
            # a host whose layer logs its own breach line has already emitted one.
            log.error(
                "Job %s timed out after %.0fs (attribution=%s)", run_id, timed_out.elapsed_s, timed_out.attribution
            )
            await self._set_status(run_id, scope_id, "failed", error=f"Job timed out after {timed_out.budget_s:.0f}s")
        except (
            Exception
        ) as exc:  # prawduct:ok-broad-except — top-level job boundary; must catch all to set failed status
            log.exception("Job %s failed", run_id)
            await self._set_status(run_id, scope_id, "failed", error=str(exc))
        finally:
            self._tasks.pop(run_id, None)
            self._cancel_reasons.pop(run_id, None)
            self._work_ran.pop(run_id, None)

    async def _set_status(
        self,
        run_id: str,
        scope_id: str,
        status: str,
        error: str | None = None,
        cancellation_reason: str | None = None,
        budget_stop_reason: str | None = None,
    ) -> None:
        """Update an EvalRun's status in storage.

        ``error``, ``cancellation_reason`` and ``budget_stop_reason`` are separate
        channels and the caller picks one, at the boundary where it knows which
        outcome it is handling: a harness failure is an error, an operator's cancel
        is not, a cost cap doing its configured job is not either, and a single
        parameter routed by status would put that decision here — away from the
        `except` clause that actually knows. The two designed stops therefore leave
        ``error_details`` empty, which is what makes its length a count worth
        triaging on.

        **A terminal write always settles the completeness record**, and does so by
        deriving rather than being told. When a run goes terminal without its work
        function having started, nothing else will ever record how much of the
        matrix it delivered and the honest answer is *none of it*; when the work
        function did start, it records its own tally in a ``finally`` and that
        tally is the authority. The fact separating those two cases —
        ``self._work_ran`` — belongs to the job, so it is read here rather than
        passed by each ``except`` branch. That is deliberate and is the whole
        mechanism: the invariant "every terminal run carries a record" used to rest
        on one keyword argument at one of four terminal branches, and on the
        assumption that the other three could only be reached once the work
        function had started. Nothing local made that true. It held on invariants
        owned by other modules — the job's budget being resolvable at all, that
        budget being clamped above zero, and ``update_eval_run`` reporting failure
        rather than raising — because a job's timeout context validates its budget
        on *entry*, before the body runs (:class:`JobTimeoutFactory` states that as
        an obligation), so a non-positive budget reaches the broad handler with
        nothing having run. No production caller produces one; the manager's own
        API accepts one. A branch added here now inherits the behaviour instead of
        having to remember it, and it no longer matters which of those modules
        changes — nor which host supplied the timeout.

        The derivation is from the job's own state and never from a missing record:
        a run whose loop DID run and whose record write was refused also arrives
        here with no record, and stamping zero on that one would replace a lost
        disclosure with a false one. A run this manager is not tracking — no
        ``_work_ran`` entry — gets no stamp for the same reason: absent state is
        not evidence that nothing ran.

        Optimistic concurrency (ETag), retried on refusal: the write the run's
        own work function makes just before this one — its completeness record —
        touches the same document, so losing a race here is the ordinary case
        rather than the exotic one. An unretried refusal used to pass unnoticed
        (the conditional write's return was not read at all), leaving a job that
        had finished with a document that still says ``running``.

        **A terminal write runs to completion** (:func:`_finish_even_if_cancelled`)
        because the storage round-trip is now awaited rather than called inline,
        and an awaited write is one a second cancel can truncate. That property
        used to come free from being synchronous; it is now stated and tested
        rather than inherited.
        """
        # ``is False`` rather than ``not ...``: an untracked run reads as None, and
        # only a tracked run that never started its work function earns the stamp.
        record_empty_matrix = status in TERMINAL_RUN_STATUSES and self._work_ran.get(run_id) is False

        def _stamp(run: EvalRun) -> dict[str, Any] | None:
            # A terminal status is final, and this is the only place that can say so:
            # `run` here is the document the LAST writer committed, re-read on every
            # attempt. Checking before the call would check a document another writer
            # may already have replaced.
            #
            # The race this closes is not exotic, it is the ordinary cancel. A cancel
            # arriving at the unshielded `running` write raises CancelledError at the
            # await but cannot stop the worker thread already inside the round-trip, so
            # the cancel branch's terminal write and the in-flight `running` write are
            # two concurrent writers against one document. Whichever loses re-reads and
            # re-applies — and an unguarded re-apply would put `status="running"` back
            # onto a document already carrying `completed_at` and `cancellation_reason`,
            # with no live task behind it, after the socket had broadcast `cancelled`.
            # Shielding the non-terminal write would narrow that window rather than
            # close it; refusing the transition closes it.
            if run.status in TERMINAL_RUN_STATUSES and status != run.status:
                return None
            data = run.to_dict()
            data["status"] = status
            if error is not None:
                data.setdefault("error_details", [])
                if isinstance(data["error_details"], list):
                    data["error_details"].append(error)
            if cancellation_reason is not None:
                data["cancellation_reason"] = cancellation_reason
            if budget_stop_reason is not None:
                data["budget_stop_reason"] = budget_stop_reason
            if record_empty_matrix and run.completeness is None:
                data["completeness"] = summarize_completeness(run, []).to_dict()
            if status in TERMINAL_RUN_STATUSES:
                data["completed_at"] = datetime.now(UTC).isoformat()
            return data

        # Off the event loop: this is a conditional read-modify-write retried up to
        # ``RUN_DOCUMENT_WRITE_ATTEMPTS`` times, so a contended flip is up to three
        # SEQUENTIAL database round-trips. Run here, as it was, each of them stalled
        # every other coroutine in the process — the other jobs, the WebSocket
        # fan-out, the API. Only the round-trip moves: the logging and the broadcast
        # below stay on the loop, so ``on_progress`` is still invoked from the thread
        # its callers were written against.
        write = run_blocking(self._blocking_executor, update_eval_run, self._storage, run_id, scope_id, _stamp)
        if status in TERMINAL_RUN_STATUSES:
            outcome = await _finish_even_if_cancelled(write)
        else:
            outcome = await write
        if outcome == "declined":
            # Not a failure: the run reached a terminal status first and that document
            # stands. Deliberately NOT broadcast — a socket message naming this status
            # would tell every live reader the opposite of what is stored, which is the
            # half of the defect that outlives the process.
            log.info(
                "Job %s: status=%s not recorded — the run is already terminal and a terminal status is final",
                run_id,
                status,
            )
            return
        if outcome == "missing":
            # Every run is saved before its task starts, so this is a document deleted from under a
            # live job. Nothing to write to; broadcast only.
            log.warning("Job %s: EvalRun not found (deleted while running?), broadcasting status=%s", run_id, status)
            self._safe_broadcast(run_id, {"status": status, **({"error": error} if error else {})})
            return
        if outcome == "refused":
            # The document keeps whatever non-terminal status it had, with no live
            # job behind it. Nothing in this process retries later, so the repair
            # is the next process's abandoned-run sweep — which will record it as
            # cancelled, a mislabel this log line is the only warning of.
            log.error(
                "Job %s: status=%s NOT recorded (no attempt completed — a refused write or a failing read, "
                "whichever the eval.update_run lines above show) — the run document "
                "still reads non-terminal and will be reclaimed as abandoned by the next process",
                run_id,
                status,
            )

        self._safe_broadcast(run_id, {"status": status})

    async def _update_progress(self, run_id: str, scope_id: str, progress: dict[str, Any]) -> None:
        """Update an EvalRun's progress in storage.

        Deliberately NOT retried, unlike the status write above: a progress
        figure is superseded by the next cell's within the minute, so a refused
        write costs a stale count on a live run rather than a permanent
        misstatement about a finished one. The refusal is still *read* and
        logged, with the figure that was lost — an unread write result is how both
        of this path's siblings came to lose data silently, and a log line naming
        neither the run nor the value it failed to write is barely better. At
        WARNING rather than debug: this fires on a *refused* write, not on every
        cell, so it is as rare as its siblings and equally worth seeing.

        Off the event loop, and in ONE hop rather than two awaits: this fires per
        cell, so on the loop it stalled every other coroutine twice per tick, and
        splitting the load from the save would open an interleaving point inside a
        read-modify-write that did not have one while it was synchronous. The
        logging and the broadcast stay on the loop, as in the status write.

        Not run to completion under cancellation, unlike a terminal status write:
        a cancelled job's next stop is that write, which supersedes whatever tick
        was in flight.
        """
        outcome = await run_blocking(self._blocking_executor, self._write_progress, run_id, scope_id, progress)
        if outcome == "declined":
            # The store already refused to reopen the run; broadcasting `running` here
            # would do to every live reader what the write was stopped from doing to
            # the document.
            log.info("Job %s: progress tick dropped — the run is already terminal", run_id)
            return
        if outcome == "refused":
            log.warning(
                "Job %s: progress write refused (%s) — not retried; the next cell's write supersedes it",
                run_id,
                progress,
            )

        self._safe_broadcast(run_id, {"status": "running", "progress": progress})

    def _write_progress(
        self, run_id: str, scope_id: str, progress: dict[str, Any]
    ) -> Literal["saved", "refused", "declined"] | None:
        """Stamp ``progress`` onto the run document under its own ETag.

        The blocking half of :meth:`_update_progress`, kept whole so the load and
        the conditional save share a single worker-thread hop.

        Args:
            run_id: The run to stamp.
            scope_id: Partition key the store reads the run under.
            progress: The progress payload to record.

        Returns:
            ``"saved"``, or ``"refused"`` when the conditional write lost its race or
            failed — a tick is not retried, since the next one supersedes it; ``None`` when there is no
            document to write to — one deleted from under the live job; or ``"declined"`` when the run has
            already reached a terminal status, which no progress tick may reopen. Neither ``None`` nor
            ``"declined"`` is a refusal and neither is logged as one.
        """
        run, etag = self._storage.load_eval_run_with_etag(run_id, scope_id)
        if run is None:
            return None
        if run.status in TERMINAL_RUN_STATUSES:
            # Same rule as the status write, and the same reason: this line used to set
            # `status = "running"` unconditionally, so a tick still in flight when the
            # job went terminal resurrected a finished run. A tick is never worth
            # reopening a run that has ended.
            return "declined"
        data = run.to_dict()
        data["progress"] = progress
        data["status"] = "running"
        try:
            self._storage.save_eval_run(EvalRun.from_dict(data), if_match=etag)
        except ConflictError, StorageError:
            return "refused"
        return "saved"


__all__ = [
    "DEFAULT_JOB_TIMEOUT_S",
    "JOB_TIMEOUT_CAP_S",
    "JOB_TIMEOUT_FLOOR_S",
    "JOB_TIMEOUT_SAFETY_MARGIN",
    "MAX_CONCURRENT_JOBS",
    "SHUTDOWN_SETTLE_TIMEOUT_S",
    "AdmissionTicket",
    "EvalJobManager",
    "EvalJobTimeout",
    "JobTimeoutFactory",
    "ProgressFn",
    "WorkFn",
    "adaptive_job_timeout_s",
    "default_job_timeout",
]
