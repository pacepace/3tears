"""Tests for ``EvalJobManager`` cancellation.

The cancel machinery predates these tests; what they pin is the
operator-facing *reason* plumbing — ``cancel_job(job_id, reason=...)`` stores
the reason and the job's own ``CancelledError`` boundary attaches it to the
terminal status write. These tests pin that contract for EvalRun-backed jobs
plus the semaphore-queued edge.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest

from threetears.evals.contracts.errors import ConflictError
from threetears.evals.contracts.models import EvalRun
from threetears.evals.run.jobs import (
    JOB_TIMEOUT_CAP_S,
    JOB_TIMEOUT_FLOOR_S,
    EvalJobManager,
    EvalJobTimeout,
    adaptive_job_timeout_s,
    default_job_timeout,
)
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.job_support import InMemoryRunStore, blocked_work, cancelled, settled, start_tracked


async def test_cancel_job_records_the_reason_in_its_own_channel_not_among_the_errors() -> None:
    """A cancel is a terminal OUTCOME, not a failure, and reads as one.

    The reason was previously appended to ``error_details``, so a run an
    operator stopped on purpose rendered as ``Errors: 1`` beside runs that
    actually broke, and later triage could not tell the two apart. Whether the
    run's partial data is usable is a separate question with a separate answer
    (``completeness``); the two were tangled together until they were split.
    """
    storage = InMemoryRunStore()
    manager = EvalJobManager(storage)
    run = make_eval_run()
    work, started = blocked_work()

    [task] = await start_tracked(manager, [(run, work, None)])
    await asyncio.wait_for(started.wait(), timeout=2)

    assert manager.cancel_job(run.id, reason="harness defect invalidated the sweep") is True
    await cancelled(task)

    persisted = storage.runs[run.id]
    assert persisted.status == "cancelled"
    assert persisted.cancellation_reason == "harness defect invalidated the sweep"
    assert persisted.error_details == [], "an operator's decision is not an error the run recorded"
    assert persisted.completed_at is not None

    # Consumed, not leaked: the same id run again and cancelled without a reason records none.
    rerun_work, rerun_started = blocked_work()
    [rerun] = await start_tracked(manager, [(make_eval_run(id=run.id), rerun_work, None)])
    await asyncio.wait_for(rerun_started.wait(), timeout=2)
    assert manager.cancel_job(run.id) is True
    await cancelled(rerun)
    assert storage.runs[run.id].cancellation_reason is None, "the first cancel's reason leaked onto the next job"


async def test_cancel_job_without_reason_records_neither_channel() -> None:
    storage = InMemoryRunStore()
    manager = EvalJobManager(storage)
    run = make_eval_run()
    work, started = blocked_work()

    [task] = await start_tracked(manager, [(run, work, None)])
    await asyncio.wait_for(started.wait(), timeout=2)

    assert manager.cancel_job(run.id) is True
    await cancelled(task)

    persisted = storage.runs[run.id]
    assert persisted.status == "cancelled"
    assert persisted.error_details == []
    assert persisted.cancellation_reason is None, "no reason given is not the same as a reason of empty"


async def test_cancel_job_while_semaphore_queued() -> None:
    """A job cancelled before it acquires a slot still terminates as cancelled."""
    storage = InMemoryRunStore()
    manager = EvalJobManager(storage, max_concurrent=1)
    blocker, blocker_started = blocked_work()
    queued_run = make_eval_run()
    queued_work, queued_started = blocked_work()

    await manager.start_group([(make_eval_run(), blocker, None)])
    await asyncio.wait_for(blocker_started.wait(), timeout=2)
    [task] = await start_tracked(manager, [(queued_run, queued_work, None)])
    await asyncio.sleep(0)  # let the queued task block on the semaphore

    assert not queued_started.is_set()
    assert manager.cancel_job(queued_run.id, reason="operator cancel") is True
    await cancelled(task)

    persisted = storage.runs[queued_run.id]
    assert persisted.status == "cancelled"
    assert persisted.cancellation_reason == "operator cancel"

    await manager.shutdown()


async def test_a_job_cancelled_while_queued_records_that_it_delivered_nothing() -> None:
    """The stop that reaches no ``finally``, because no work function ever ran.

    A job waits for a semaphore slot before its work function is called, and a
    sweep of ten arms parks eight there in ``pending``. Cancelling one — which
    ``cancel_run`` explicitly permits for a pending run — raises at the
    semaphore, so the work function's own record never happens and the reclaim
    path never sees the run (it has a live task). Without this the run went
    terminal carrying no completeness at all, which every read surface is
    entitled to interpret as "its record's write was refused".

    The honest record is **none of the matrix**: expected counts the run's
    promise, produced counts what ran, and nothing ran.
    """
    storage = InMemoryRunStore()
    manager = EvalJobManager(storage, max_concurrent=1)
    blocker, blocker_started = blocked_work()
    queued_run = make_eval_run(candidate_model="m1", k_runs=1, test_case_ids=["tc-1", "tc-2", "tc-3"])
    queued_work, queued_started = blocked_work()

    await manager.start_group([(make_eval_run(), blocker, None)])
    await asyncio.wait_for(blocker_started.wait(), timeout=2)
    [task] = await start_tracked(manager, [(queued_run, queued_work, None)])
    await asyncio.sleep(0)

    assert not queued_started.is_set(), "the arm under test needs the job still waiting for a slot"
    assert manager.cancel_job(queued_run.id, reason="sweep abandoned before it started") is True
    await cancelled(task)

    persisted = storage.runs[queued_run.id]
    assert persisted.status == "cancelled"
    assert persisted.completeness is not None, (
        "a terminal run with no record reads as one whose record write was refused"
    )
    assert persisted.completeness.expected_cells == 3
    assert persisted.completeness.produced_cells == 0
    assert persisted.completeness.degraded is True

    await manager.shutdown()


async def test_a_cancel_after_the_work_ran_leaves_the_loops_own_record_alone() -> None:
    """The empty stamp must never overwrite a tally that saw the cells run.

    A run whose work function got going records its own completeness in a
    ``finally``; stamping zero over that would replace a real observation with a
    false one. The empty stamp is derived from whether the work function started —
    the job's own state — and never from a missing record, because a loop that DID
    run and whose record write was refused also arrives with none.
    """
    storage = InMemoryRunStore()
    manager = EvalJobManager(storage)
    run = make_eval_run(candidate_model="m1", k_runs=1, test_case_ids=["tc-1", "tc-2"])
    started = asyncio.Event()

    async def work_that_records_then_parks(progress: Any) -> None:
        # Stands in for the service's work function, which records completeness
        # from its own ``finally`` before this boundary ever sees the cancel.
        stored = storage.runs[run.id].to_dict()
        stored["completeness"] = {
            "expected_cells": 2,
            "produced_cells": 1,
            "persisted_cells": 1,
            "infra_excluded_cells": 0,
            "counted_from": "run_loop",
        }
        storage.runs[run.id] = EvalRun.from_dict(stored)
        started.set()
        await asyncio.Event().wait()

    [task] = await start_tracked(manager, [(run, work_that_records_then_parks, None)])
    await asyncio.wait_for(started.wait(), timeout=2)

    assert manager.cancel_job(run.id, reason="stopped after the first cell") is True
    await cancelled(task)

    persisted = storage.runs[run.id]
    assert persisted.completeness is not None
    assert persisted.completeness.produced_cells == 1, "the loop's own count was overwritten with a zero"


async def test_a_terminal_branch_other_than_cancel_records_the_empty_matrix() -> None:
    """The invariant must not depend on which ``except`` clause caught the stop.

    Cancellation used to be the only terminal branch that recorded an empty
    matrix; the other three took the default and wrote nothing. That was safe
    only because they cannot be raised before the work function starts — a fact
    written down at none of the four call sites, so the next branch added would
    have dropped the record in silence.

    This drives the ``EvalJobTimeout`` arm with the timeout raised on entry to the
    context, so the run goes terminal down a **non-cancel** branch having never
    started its work function. Nothing else will ever record how much of the matrix
    it delivered, so the terminal write owes it.

    Injected rather than stubbed: the timeout context is a constructor argument
    (:class:`~threetears.evals.run.jobs.JobTimeoutFactory`), so the arm is reachable by
    handing the manager a factory that raises — no patching of anything the manager
    imports, and nothing here that any host could not also write.
    """

    class _RaisingTimeout:
        async def __aenter__(self):
            raise EvalJobTimeout(budget_s=1.0, elapsed_s=1.0)

        async def __aexit__(self, *exc):
            return False

    storage = InMemoryRunStore()
    manager = EvalJobManager(storage, job_timeout_factory=lambda budget_s: _RaisingTimeout())
    run = make_eval_run(candidate_model="m1", k_runs=1, test_case_ids=["tc-1", "tc-2", "tc-3"])

    await manager.start_group([(run, blocked_work()[0], None)])
    await settled(manager, run.id, timeout=2)

    persisted = storage.runs[run.id]
    assert persisted.status == "failed"
    assert persisted.completeness is not None, (
        "a terminal run with no record reads as one whose record write was refused"
    )
    assert persisted.completeness.expected_cells == 3
    assert persisted.completeness.produced_cells == 0
    assert persisted.completeness.degraded is True

    await manager.shutdown()


async def test_a_budget_rejected_before_the_work_starts_still_records_the_empty_matrix() -> None:
    """The reachable case, with nothing stubbed: the broad handler, before anything ran.

    A job's timeout context validates its budget on entry — a non-positive one raises
    ``ValueError`` before the body, which :class:`~threetears.evals.run.jobs.JobTimeoutFactory` states
    as an obligation and this manager's default meets — so the broad ``except Exception`` arm is
    reachable with the work function never started. That arm recorded no completeness, which makes this a
    live gap rather than a latent one: the run went terminal carrying nothing, and a run
    carrying nothing is indistinguishable downstream from one whose record write was refused, which
    every compare surface reads as *not short*.
    """
    storage = InMemoryRunStore()
    manager = EvalJobManager(storage)
    run = make_eval_run(candidate_model="m1", k_runs=1, test_case_ids=["tc-1", "tc-2", "tc-3"])
    work, started = blocked_work()

    await manager.start_group([(run, work, 0)])
    await settled(manager, run.id, timeout=2)

    assert not started.is_set(), "the arm under test needs the work function never to have run"
    persisted = storage.runs[run.id]
    assert persisted.status == "failed"
    assert persisted.completeness is not None, (
        "a terminal run with no record reads as one whose record write was refused"
    )
    assert persisted.completeness.expected_cells == 3
    assert persisted.completeness.produced_cells == 0

    await manager.shutdown()


async def test_cancel_job_unknown_or_done_returns_false() -> None:
    storage = InMemoryRunStore()
    manager = EvalJobManager(storage)
    assert manager.cancel_job("nope", reason="x") is False

    # The reason is not stored for an unknown job: a job later started under that id and
    # cancelled without one records none.
    work, started = blocked_work()
    [task] = await start_tracked(manager, [(make_eval_run(id="nope"), work, None)])
    await asyncio.wait_for(started.wait(), timeout=2)
    assert manager.cancel_job("nope") is True
    await cancelled(task)
    assert storage.runs["nope"].cancellation_reason is None, "a refused cancel stored its reason anyway"


# =============================================================================
# Adaptive job timeout
# =============================================================================


def test_adaptive_job_timeout_scales_linearly_with_matrix() -> None:
    """A 12-cell matrix sizes well above the old fixed 3600s cap.

    12 cells × 600s per-cell ceiling × 1.1 margin = 7920s — the exact shape
    (k=3×4) that was guillotined at 3600 now gets room to finish.
    """
    assert adaptive_job_timeout_s(12, 600.0) == pytest.approx(7920.0)
    assert adaptive_job_timeout_s(12, 600.0) > 3600


def test_adaptive_job_timeout_floors_small_matrices() -> None:
    """A tiny matrix (or tiny per-result budget) never drops below the floor."""
    assert adaptive_job_timeout_s(1, 100.0) == JOB_TIMEOUT_FLOOR_S
    assert adaptive_job_timeout_s(0, 600.0) == JOB_TIMEOUT_FLOOR_S


def test_adaptive_job_timeout_caps_pathological_matrices() -> None:
    """A fat-fingered matrix can't request an unbounded budget."""
    assert adaptive_job_timeout_s(10_000, 600.0) == JOB_TIMEOUT_CAP_S


async def test_start_job_enforces_passed_timeout() -> None:
    """A per-job ``job_timeout_s`` reaches real enforcement, not just the manager's field.

    Work that outlives the budget is cancelled and the run recorded ``failed``
    with a timeout message — the same path a too-small fixed cap took, now
    driven by the passed value, through whichever timeout context the manager
    was built with (here the engine default).
    """
    storage = InMemoryRunStore()
    manager = EvalJobManager(storage)
    run = make_eval_run()

    async def slow_work(progress: Any) -> None:
        await asyncio.sleep(5)

    await manager.start_group([(run, slow_work, 0.05)])
    await settled(manager, run.id, timeout=2)

    persisted = storage.runs[run.id]
    assert persisted.status == "failed"
    assert any("timed out" in e.lower() for e in persisted.error_details)


async def test_start_job_completes_within_generous_timeout() -> None:
    """Fast work under a generous per-job budget completes normally."""
    storage = InMemoryRunStore()
    manager = EvalJobManager(storage)
    run = make_eval_run()
    ran = asyncio.Event()

    async def fast_work(progress: Any) -> None:
        await progress({"completed": 1, "total": 1})
        ran.set()

    await manager.start_group([(run, fast_work, 30.0)])
    await settled(manager, run.id, timeout=2)

    assert ran.is_set()
    assert storage.runs[run.id].status == "completed"


async def test_a_terminal_status_write_that_loses_one_race_still_lands() -> None:
    """A finished job must not leave a document that says ``running``.

    The status write is conditional, and the run's own completeness record is
    written to the same document immediately before it, so losing the race is the
    ordinary case here rather than an exotic one. This return value was not read
    at all: a lost write left a job that had ended behind a document still
    reading non-terminal, reclaimable only by the next process's abandoned-run
    sweep — which records it as *cancelled*, mislabelling a run that completed.
    """
    storage = InMemoryRunStore()
    manager = EvalJobManager(storage)
    run = make_eval_run()

    refused: list[str] = []
    real_save = storage.save_eval_run

    def _refuse_the_first_terminal_write(saving: EvalRun, *, if_match: str | None = None) -> None:
        if saving.status == "completed" and not refused:
            refused.append(saving.status)
            raise ConflictError("lost the race")
        real_save(saving, if_match=if_match)

    storage.save_eval_run = _refuse_the_first_terminal_write  # type: ignore[method-assign]

    async def fast_work(progress: Any) -> None:
        return None

    await manager.start_group([(run, fast_work, None)])
    await settled(manager, run.id, timeout=2)

    assert refused == ["completed"], "the arm under test never fired — the write was not refused"
    assert storage.runs[run.id].status == "completed"


async def test_budget_stopped_work_records_the_stop_in_its_own_channel_not_among_the_errors() -> None:
    """A ``BudgetStoppedError`` from the work fn becomes the honest ``budget_stopped``
    terminal status — not ``failed`` — with the reason on its own channel.

    The reason was previously appended to ``error_details``, exactly as a cancel's
    once was, so a run its own configured cap stopped rendered as ``Errors: 1``
    beside runs that actually broke. That was not a rare miscount: an omitted
    ``max_cost_usd`` inherits ``config.budget.eval_run_max_cost_usd``, so EVERY run
    carries a cap and every capped run that bound contributed a phantom to the count
    an operator scans for real faults. ``completed_at`` is still stamped — the run
    did end.
    """
    from threetears.evals.run.budget import BudgetStoppedError, CapBreach

    storage = InMemoryRunStore()
    manager = EvalJobManager(storage)
    run = make_eval_run()

    async def work(progress: Any) -> None:
        raise BudgetStoppedError(2, 5, CapBreach(max_cost_usd=0.02, accumulated_usd=0.0299, unpriced_results=0))

    [task] = await start_tracked(manager, [(run, work, None)])
    await task  # BudgetStoppedError is handled inside _run_job, not re-raised

    persisted = storage.runs[run.id]
    assert persisted.status == "budget_stopped"
    assert persisted.completed_at is not None
    assert "2/5" in (persisted.budget_stop_reason or "")
    assert persisted.error_details == [], "a cap doing its configured job is not an error the run recorded"


async def test_an_exhausted_account_is_its_own_terminal_status_with_its_reason_among_the_errors() -> None:
    """An ``AccountExhaustedError`` from the work fn becomes ``exhausted`` — never ``failed``.

    Unlike the two designed stops its reason IS filed under ``error_details``: nobody configured
    the account to run dry, and an operator has to act on it before the next launch. ``completed_at``
    is stamped because ``exhausted`` is terminal.
    """
    from threetears.evals.run.budget import AccountExhaustedError

    storage = InMemoryRunStore()
    manager = EvalJobManager(storage)
    run = make_eval_run()

    async def work(progress: Any) -> None:
        raise AccountExhaustedError(
            2, 5, accumulated_usd=0.0321, unpriced_results=0, detail="apparatus: refused (payment)"
        )

    await manager.start_group([(run, work, None)])
    await settled(manager, run.id, timeout=2)

    persisted = storage.runs[run.id]
    assert persisted.status == "exhausted"
    assert persisted.completed_at is not None
    [reason] = persisted.error_details
    assert "2/5" in reason and "$0.0321 spent" in reason and "refused (payment)" in reason
    assert persisted.budget_stop_reason is None


async def test_a_genuine_failure_still_lands_among_the_errors() -> None:
    """The counterpart to the two designed stops: a real fault keeps the error channel.

    Both graceful terminations now write elsewhere, so this pins that the channel
    they left is still the one a harness exception uses — the emptiness asserted
    above has to mean "nothing broke", not "nothing is ever recorded here".
    """
    storage = InMemoryRunStore()
    manager = EvalJobManager(storage)
    run = make_eval_run()

    async def broken_work(progress: Any) -> None:
        raise RuntimeError("judge service refused the connection")

    await manager.start_group([(run, broken_work, None)])
    await settled(manager, run.id, timeout=2)

    persisted = storage.runs[run.id]
    assert persisted.status == "failed"
    assert any("judge service refused the connection" in detail for detail in persisted.error_details)
    assert persisted.budget_stop_reason is None
    assert persisted.cancellation_reason is None


# =============================================================================
# The injected timeout seam
#
# The manager bounds each job with a context manager it is handed rather than
# one it imports, so the engine installs in a host that has no timeout layer of
# its own. What the tests below hold is the contract between the two halves:
# what the shipped default does, and that the manager actually delegates.
# =============================================================================


async def test_the_default_timeout_rejects_a_non_positive_budget_before_the_body_runs() -> None:
    """Ordering, not just rejection — the manager derives an invariant from it.

    ``_set_status`` tells "this run never got going" from "this run ran and owns its own
    tally" by whether the work function started. A budget rejected *after* the body had begun
    would make that derivation stamp an empty matrix on a run that had produced cells. So the
    default validates on entry, and this pins the entry part rather than only the raise.
    """
    entered = False

    with pytest.raises(ValueError, match="must be positive"):
        async with default_job_timeout(0):
            entered = True

    assert not entered, "the budget must be refused before the body is allowed to run"


async def test_the_default_timeout_raises_the_engine_type_on_breach() -> None:
    """A breached budget reaches the manager as ``EvalJobTimeout``, attributing nothing.

    The engine's default nests nothing, so it has no innermost frame to name — ``attribution``
    is ``None`` here and is populated only by a host whose timeout layer tracks nesting.
    """
    with pytest.raises(EvalJobTimeout) as caught:
        async with default_job_timeout(0.01):
            await asyncio.sleep(5)

    assert caught.value.budget_s == 0.01
    assert caught.value.elapsed_s > 0
    assert caught.value.attribution is None


async def test_the_default_timeout_passes_a_body_raised_timeout_error_through() -> None:
    """A ``TimeoutError`` from inside is not this budget firing, and must not be relabelled.

    ``asyncio.timeout`` converts its own expiry into ``TimeoutError``, so a bare ``except
    TimeoutError`` cannot tell the two apart — and a body that raises one (an HTTP client, an
    inner ``wait_for``) would be reported as a job that outlived a budget it was comfortably
    inside. The deadline's own ``expired()`` is what separates them.
    """
    with pytest.raises(TimeoutError) as caught:
        async with default_job_timeout(30):
            raise TimeoutError("the judge's HTTP client gave up")

    assert not isinstance(caught.value, EvalJobTimeout)
    assert "judge's HTTP client" in str(caught.value)


async def test_the_manager_hands_each_job_its_own_resolved_budget() -> None:
    """Per job, not per manager — and resolved at start, not captured at construction.

    Two runs in one process size their matrices differently, so a factory called once and
    reused would give the second run the first one's deadline. This drives two jobs through one
    manager with different budgets and reads back what the factory was actually called with.
    """
    seen: list[float] = []

    @asynccontextmanager
    async def recording_timeout(budget_s: float):
        seen.append(budget_s)
        yield budget_s

    storage = InMemoryRunStore()
    manager = EvalJobManager(storage, job_timeout_factory=recording_timeout)

    async def quick(progress: Any) -> None:
        return None

    first = make_eval_run()
    second = make_eval_run()
    await manager.start_group([(first, quick, 111.0)])
    await settled(manager, first.id, timeout=2)
    await manager.start_group([(second, quick, 222.0)])
    await settled(manager, second.id, timeout=2)

    assert seen == [111.0, 222.0]
    await manager.shutdown()


async def test_a_manager_built_with_no_factory_uses_the_engine_default() -> None:
    """The engine installs and runs without a host supplying anything.

    The whole point of the default: a second consumer constructs the manager with a store and
    nothing else, and its jobs are still bounded. An unwired construction is legitimate HERE —
    what is not legitimate is an unwired construction in a host that has a timeout layer of its
    own, which that host's suite gates separately.
    """
    storage = InMemoryRunStore()
    manager = EvalJobManager(storage)
    run = make_eval_run()

    # Bounded with nothing wired: the default factory times the job out on its budget.
    await manager.start_group([(run, blocked_work()[0], 0.05)])
    await settled(manager, run.id, timeout=2)

    persisted = storage.runs[run.id]
    assert persisted.status == "failed"
    assert persisted.error_details == ["Job timed out after 0s"]
    await manager.shutdown()


# =============================================================================
# The lifecycle writes and the event loop
#
# ``_set_status`` and ``_update_progress`` are storage round-trips — the status
# one a read-modify-write retried up to RUN_DOCUMENT_WRITE_ATTEMPTS times — and
# they were called inline from the async ``_run_job`` body, so every status flip
# and every progress tick stalled every other coroutine in the process for one
# to three sequential database calls. They now hop to a worker thread.
#
# The hop is not free of consequence, which is what the rest of this section is
# about: it turns two synchronous calls into await points, and an await point
# can be interleaved with, reordered around, and cancelled. What was previously
# true for free has to be pinned.
# =============================================================================


class _ParkingStorage(InMemoryRunStore):
    """Storage that parks the first write matching a predicate until released.

    It parks in whatever thread the write runs on, which is the whole point: a
    write left on the event loop parks the loop, and ``thread_name`` then reads
    back as the loop's own thread.

    The selector is a predicate over the document being saved rather than a
    status string, because the status alone does not separate the writes these
    tests are about: a progress tick and the lifecycle's own ``running`` flip
    both save status ``running``, and a status-keyed version of this double
    parked the flip in the tick's test — which then passed with the tick's hop
    reverted, because it was never asserting about the tick.
    """

    def __init__(self, park_when: Callable[[EvalRun], bool]) -> None:
        super().__init__()
        self._park_when = park_when
        self.entered = threading.Event()
        self.release = threading.Event()
        self.thread_name: str | None = None

    def save_eval_run(self, run: EvalRun, *, if_match: str | None = None) -> None:
        if self.thread_name is None and self._park_when(run):
            self.thread_name = threading.current_thread().name
            self.entered.set()
            # Bounded on purpose: a regression that runs this on the loop would
            # otherwise deadlock the whole test session rather than failing the
            # assertion that names the defect.
            self.release.wait(timeout=2.0)
        super().save_eval_run(run, if_match=if_match)


def _a_lifecycle_flip_to(status: str) -> Callable[[EvalRun], bool]:
    """Select a status flip — a write that carries no progress figure."""
    return lambda run: run.status == status and not run.progress


def _a_progress_tick(run: EvalRun) -> bool:
    """Select a progress write — the only writes that carry a figure."""
    return bool(run.progress)


async def test_a_status_flip_does_not_stall_the_event_loop() -> None:
    """The flip's database round-trip runs off the loop.

    ``_set_status`` routes through ``update_eval_run``, a conditional
    read-modify-write that retries up to three times, so a contended flip was up
    to three *sequential* round-trips with the loop held throughout — the other
    jobs, the WebSocket fan-out and the API all waiting on it.

    Pinned as "not the loop's thread" rather than as a latency figure, because a
    timing assertion on a shared box measures the box.
    """
    loop_thread_name = threading.current_thread().name
    storage = _ParkingStorage(_a_lifecycle_flip_to("running"))
    manager = EvalJobManager(storage)
    run = make_eval_run()
    work, _started = blocked_work()

    await manager.start_group([(run, work, None)])
    # Reachable only because the loop is free while the write is parked.
    await asyncio.wait_for(asyncio.to_thread(storage.entered.wait, 5), timeout=5)
    storage.release.set()

    assert storage.thread_name is not None, "the status write never reached storage"
    assert storage.thread_name != loop_thread_name, "the status write ran on the event loop thread"

    await manager.shutdown()


async def test_a_progress_tick_does_not_stall_the_event_loop() -> None:
    """The same for progress, which fires per cell rather than per run.

    The status flips are bounded by the lifecycle; progress ticks are bounded by
    the matrix, so this is the hotter of the two paths and the one where a
    stalled loop is felt as a stuttering UI.
    """
    loop_thread_name = threading.current_thread().name
    # The tick, specifically — the lifecycle's own flip to "running" writes the
    # same status and must not be the write this parks.
    storage = _ParkingStorage(_a_progress_tick)
    manager = EvalJobManager(storage)
    run = make_eval_run()

    async def work(progress: Any) -> None:
        await progress({"completed": 1, "total": 2})
        await asyncio.Event().wait()

    await manager.start_group([(run, work, None)])
    # Reachable only because the loop is free while the tick's write is parked.
    await asyncio.wait_for(asyncio.to_thread(storage.entered.wait, 5), timeout=5)
    storage.release.set()
    await asyncio.sleep(0.05)

    assert storage.thread_name is not None, "the progress write never reached storage"
    assert storage.thread_name != loop_thread_name, "a progress write ran on the event loop thread"
    assert storage.runs[run.id].progress == {"completed": 1, "total": 2}

    await manager.shutdown()


async def test_the_thread_hop_does_not_reorder_the_lifecycle_or_move_the_broadcast() -> None:
    """Two properties the hop could have taken, pinned together.

    **Order.** Every write is still awaited in place from one task, so a reader
    of the broadcast stream sees ``running``, then each tick in the order the
    work function produced it, then the terminal status. A hop that fired and
    forgot would let a terminal status overtake a tick.

    **Thread.** Only the storage round-trip moves. ``on_progress`` is a plain
    synchronous callback the manager's callers hand in — the production one
    reaches the WebSocket manager — and it is still invoked from the event loop
    thread, as every caller was written against. This is the assertion that goes
    red if someone later "simplifies" by moving the whole method into the thread.
    """
    loop_thread_name = threading.current_thread().name
    storage = InMemoryRunStore()
    seen: list[tuple[str, str, Any]] = []

    def on_progress(job_id: str, payload: dict[str, Any]) -> None:
        seen.append((threading.current_thread().name, payload["status"], payload.get("progress")))

    manager = EvalJobManager(storage, on_progress=on_progress)
    run = make_eval_run()

    async def work(progress: Any) -> None:
        for completed in (1, 2, 3):
            await progress({"completed": completed, "total": 3})

    await manager.start_group([(run, work, None)])
    await settled(manager, run.id, timeout=5)

    assert [(status, prog) for _thread, status, prog in seen] == [
        ("running", None),
        ("running", {"completed": 1, "total": 3}),
        ("running", {"completed": 2, "total": 3}),
        ("running", {"completed": 3, "total": 3}),
        ("completed", None),
    ]
    assert {thread for thread, _status, _prog in seen} == {loop_thread_name}, (
        "on_progress was invoked off the event loop thread"
    )


async def test_a_second_cancel_cannot_truncate_the_terminal_status_write() -> None:
    """``cancel_job`` then ``shutdown`` cancels one task twice, and the run must still settle.

    While the write was synchronous this was free — nothing could run between its
    first statement and its last, so ``shutdown`` could not land inside it. As an
    await it is interruptible, and the branch it sits in is the one already
    unwinding from the *first* cancel, so the second is delivered straight into
    it. What that costs is the outcome never being read: the refusal branch that
    warns about a run left reading ``running`` never runs, and neither does the
    terminal broadcast, so every live UI keeps showing a run that has ended. The
    write itself can be lost too, when the executor has not yet started it.

    ``_finish_even_if_cancelled`` is what makes it settle; the job still ends
    cancelled, because the branch re-raises its own ``CancelledError`` after.
    """
    storage = _ParkingStorage(_a_lifecycle_flip_to("cancelled"))
    seen: list[str] = []
    manager = EvalJobManager(storage, on_progress=lambda job_id, payload: seen.append(payload["status"]))
    run = make_eval_run()
    work, started = blocked_work()

    [task] = await start_tracked(manager, [(run, work, None)])
    await asyncio.wait_for(started.wait(), timeout=2)

    assert manager.cancel_job(run.id, reason="operator stopped it") is True
    await asyncio.wait_for(asyncio.to_thread(storage.entered.wait, 5), timeout=5)

    # What shutdown() does to a task that has not finished unwinding yet.
    task.cancel()
    await asyncio.sleep(0)
    storage.release.set()
    await cancelled(task)

    persisted = storage.runs[run.id]
    assert persisted.status == "cancelled"
    assert persisted.cancellation_reason == "operator stopped it"
    assert seen[-1] == "cancelled", "the terminal broadcast was lost to the second cancel"


# =============================================================================
# Progress writes
#
# This path had no test of its own at all. It is reached once per cell, so it is
# the manager's hottest storage caller, and its refusal branch is the one place
# a lost figure is disclosed.
# =============================================================================


async def test_progress_reaches_both_storage_and_the_broadcast() -> None:
    """A tick is persisted on the run document AND announced, not one or the other."""
    storage = InMemoryRunStore()
    seen: list[dict[str, Any]] = []
    manager = EvalJobManager(storage, on_progress=lambda job_id, payload: seen.append(payload))
    run = make_eval_run()

    async def work(progress: Any) -> None:
        await progress({"completed": 4, "total": 9})

    await manager.start_group([(run, work, None)])
    await settled(manager, run.id, timeout=5)

    assert storage.runs[run.id].progress == {"completed": 4, "total": 9}
    assert {"status": "running", "progress": {"completed": 4, "total": 9}} in seen


async def test_a_refused_progress_write_is_logged_with_the_figure_it_lost(caplog: Any) -> None:
    """Deliberately not retried, so the disclosure is the whole repair.

    A status write re-reads and re-applies; a progress figure is superseded by
    the next cell's within the minute, so retrying it buys a stale number. What
    it must not do is drop the refusal silently — the log line carries the run
    and the figure, because a line naming neither is barely better than none.
    """
    storage = InMemoryRunStore()
    manager = EvalJobManager(storage)
    run = make_eval_run()

    real_save = storage.save_eval_run

    def _refuse_progress_writes(saving: EvalRun, *, if_match: str | None = None) -> None:
        if saving.progress:
            raise ConflictError("lost the race")
        real_save(saving, if_match=if_match)

    storage.save_eval_run = _refuse_progress_writes  # type: ignore[method-assign]

    async def work(progress: Any) -> None:
        await progress({"completed": 7, "total": 11})

    with caplog.at_level(logging.WARNING, logger="threetears.evals.run.jobs"):
        await manager.start_group([(run, work, None)])
        await settled(manager, run.id, timeout=5)

    refusals = [r for r in caplog.records if "progress write refused" in r.getMessage()]
    assert len(refusals) == 1, "a refused progress write went unreported"
    assert run.id in refusals[0].getMessage()
    assert "'completed': 7" in refusals[0].getMessage(), "the log line did not carry the figure it lost"


class TestATerminalStatusIsFinal:
    """The transition model the thread hop made necessary.

    While both status writes were synchronous, a run could not be written twice at once:
    nothing ran between a write's first and last statement. ``asyncio.to_thread`` removed
    that, and ``update_eval_run`` re-applies its mutator against the WINNER's document on a
    refused write — so an unguarded re-apply can put a non-terminal status back onto a
    document that has already gone terminal.

    The rule is therefore enforced where the winner's document is in hand, inside the
    mutator, rather than by shielding the writes. Shields narrow the window; the rule
    closes it.
    """

    async def test_a_late_running_write_cannot_reopen_a_cancelled_run(self) -> None:
        """The exact ordering a cancel produces: `cancelled` lands, `running` arrives after."""
        storage = _RacedToTerminal(status="cancelled", cancellation_reason="operator stopped it")
        manager = EvalJobManager(storage)
        run = make_eval_run()

        await manager.start_group([(run, _no_work, None)])
        await settled(manager, run.id)

        stored = storage.runs[run.id]
        assert stored.status == "cancelled", "a late running write reopened a finished run"
        assert stored.cancellation_reason == "operator stopped it", "the decline must not strip the winner's fields"

    async def test_a_declined_status_is_not_broadcast(self) -> None:
        """The half that outlives the process is the socket message, not the document.

        A reader told `running` after being told `cancelled` sees a run reopen. Persisting
        correctly while announcing the opposite is not half a fix.
        """
        storage = _RacedToTerminal(status="completed")
        seen: list[dict[str, Any]] = []
        manager = EvalJobManager(storage, on_progress=lambda job_id, payload: seen.append(payload))
        run = make_eval_run()

        await manager.start_group([(run, _no_work, None)])
        await settled(manager, run.id)

        assert [payload for payload in seen if payload["status"] == "running"] == [], (
            f"a declined write was announced anyway: {seen}"
        )

    async def test_one_terminal_status_does_not_overwrite_another(self) -> None:
        """A completed run is not also cancelled. First terminal wins, and it is the true one."""
        storage = _RacedToTerminal(status="completed")
        manager = EvalJobManager(storage)
        run = make_eval_run()
        work, started = blocked_work()

        [task] = await start_tracked(manager, [(run, work, None)])
        await asyncio.wait_for(started.wait(), timeout=2)
        assert manager.cancel_job(run.id, reason="too late") is True
        await cancelled(task)

        stored = storage.runs[run.id]
        assert stored.status == "completed"
        assert stored.cancellation_reason is None, "a losing terminal write still stamped its own channel"

    async def test_re_stamping_the_same_terminal_status_is_still_allowed(self) -> None:
        """The rule is about CHANGING a terminal status, not about writing one twice.

        A retry that re-applies the same terminal write must still land, or the fix for a
        reopened run becomes a lost completeness record.
        """
        storage = _RacedToTerminal(status="failed", error_details=["first"])
        manager = EvalJobManager(storage)
        run = make_eval_run()

        async def broken(progress: Any) -> None:
            raise RuntimeError("second")

        await manager.start_group([(run, broken, None)])
        await settled(manager, run.id)

        stored = storage.runs[run.id]
        assert stored.status == "failed"
        assert stored.error_details == ["first", "second"], "the second write of the same status was declined"

    async def test_a_progress_tick_cannot_reopen_a_terminal_run(self) -> None:
        """The second member of the class: a progress write set status='running' unconditionally."""
        storage = _RacedToTerminal(status="completed")
        seen: list[dict[str, Any]] = []
        manager = EvalJobManager(storage, on_progress=lambda job_id, payload: seen.append(payload))
        run = make_eval_run()

        async def ticks(progress: Any) -> None:
            await progress({"completed": 3, "total": 9})

        await manager.start_group([(run, ticks, None)])
        await settled(manager, run.id)

        assert storage.runs[run.id].status == "completed", "a progress tick resurrected a finished run"
        assert storage.runs[run.id].progress != {"completed": 3, "total": 9}
        assert [payload for payload in seen if payload.get("progress")] == [], "a dropped tick was announced anyway"


async def _no_work(progress: Any) -> None:
    """A work function that does nothing, so the job goes straight to its terminal write."""


class _RacedToTerminal(InMemoryRunStore):
    """A store in which another writer settled the run terminal the moment its launch saved it.

    Every status the job then writes arrives AFTER a terminal one, which is the ordering the
    transition model exists for, reached through the manager's own lifecycle rather than by
    calling its status writer directly.
    """

    def __init__(self, **terminal: Any) -> None:
        super().__init__()
        self._terminal = terminal
        self._raced = False

    def save_eval_run(self, run: EvalRun, *, if_match: str | None = None) -> None:
        super().save_eval_run(run, if_match=if_match)
        if not self._raced:
            self._raced = True
            self.runs[run.id] = run.model_copy(update=self._terminal)


class TestShutdownSaysWhatItCouldNotSettle:
    """``shutdown`` returns and names the jobs still unfinished when its wait ends, rather than dropping them."""

    @staticmethod
    async def _started(manager: EvalJobManager, storage: InMemoryRunStore, run_id: str, work: Any) -> asyncio.Task[Any]:
        [task] = await start_tracked(manager, [(make_eval_run(id=run_id), work, None)])
        for _ in range(200):
            if storage.runs[run_id].status == "running":
                return task
            await asyncio.sleep(0.01)
        raise AssertionError(f"{run_id} never started")

    async def test_every_job_settled_reports_none(self, caplog):
        storage = InMemoryRunStore()
        manager = EvalJobManager(storage)

        async def work(_progress: Any) -> None:
            await asyncio.Event().wait()

        await self._started(manager, storage, "run-quick", work)
        with caplog.at_level(logging.WARNING, logger="threetears.evals.run.jobs"):
            unsettled = await manager.shutdown(timeout=5.0)

        assert unsettled == []
        assert storage.runs["run-quick"].status == "cancelled"
        assert not [r for r in caplog.records if "eval.shutdown gave up" in r.getMessage()]

    async def test_a_job_outlasting_the_wait_is_returned_and_named(self, caplog):
        storage = InMemoryRunStore()
        manager = EvalJobManager(storage)
        release = asyncio.Event()

        async def work(_progress: Any) -> None:
            try:
                await asyncio.Event().wait()
            finally:
                # A cancelled job whose teardown outlasts the settle window — the shape of a
                # terminal write stuck behind a slow store.
                await asyncio.shield(release.wait())

        task = await self._started(manager, storage, "run-slow", work)
        try:
            with caplog.at_level(logging.WARNING, logger="threetears.evals.run.jobs"):
                unsettled = await manager.shutdown(timeout=0.05)
        finally:
            release.set()
            await asyncio.wait({task}, timeout=5)

        assert unsettled == ["run-slow"]
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("eval.shutdown gave up on 1 job(s)" in m and "run-slow" in m for m in warnings), warnings


async def test_wait_for_returns_once_every_named_job_has_written_its_terminal_status() -> None:
    """``wait_for`` is the awaitable a launcher holds: it returns after the jobs end, not before."""
    store = InMemoryRunStore()
    manager = EvalJobManager(store, job_timeout_factory=default_job_timeout)
    release = asyncio.Event()
    first, second = make_eval_run(), make_eval_run()

    async def work(progress: Any) -> None:
        await release.wait()

    await manager.start_group([(first, work, None), (second, work, None)])
    waiting = asyncio.create_task(manager.wait_for([first.id, second.id]))
    await asyncio.sleep(0.01)
    assert not waiting.done(), "wait_for returned while both jobs were still parked"
    release.set()
    await asyncio.wait_for(waiting, timeout=5.0)
    assert store.runs[first.id].status == "completed"
    assert store.runs[second.id].status == "completed"


async def test_wait_for_a_job_it_is_not_running_returns_at_once() -> None:
    """An ended job is popped from the manager, so a wait on it — or on an id it never ran — returns."""
    manager = EvalJobManager(InMemoryRunStore(), job_timeout_factory=default_job_timeout)
    await asyncio.wait_for(manager.wait_for(["never-started"]), timeout=1.0)


async def test_cancelling_the_waiter_leaves_the_job_running() -> None:
    """Waiting is not owning: the waiter's cancel does not reach the job it was waiting on."""
    manager = EvalJobManager(InMemoryRunStore(), job_timeout_factory=default_job_timeout)
    work, started = blocked_work()
    run = make_eval_run()
    await manager.start_group([(run, work, None)])
    await started.wait()
    waiting = asyncio.create_task(manager.wait_for([run.id]))
    await asyncio.sleep(0)
    waiting.cancel()
    await cancelled(waiting)
    assert manager.is_active(run.id), "cancelling the waiter cancelled the job"
    await manager.shutdown()
