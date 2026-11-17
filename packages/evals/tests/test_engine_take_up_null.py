"""The engine's interrupt rule, held cell by cell, and its take-up under the null (``engine_take_up.py`` holds the
simulation: an engine comparison whose launches an operator interrupts and the take-up finishes, over two arms of equal
accuracy whose latency follows their answers).

A first ``job_cancel`` drains: the run starts no further cell, and the cells in flight finish and record. A second cuts
what is still in flight, and the take-up runs each cut cell again, written over its row; the family says how many ran
again. A cancel the launch's lease finds on the record, or delivers before its own process counts it, is the same cancel.

The reading's own rate on coin pairs, uninterrupted, is exact (:func:`~packages.evals.tests.engine_take_up.exact_rate`),
pinned here to the digits. The unit suite keeps a smoke check of every simulation: a few comparisons each, every drained
one read exactly as uninterrupted on its first draws. The full simulations, and what each shows against the exact rate,
are the statistics certificate's claims (``stats_certificate.py``; ``docs/testing.md`` says how to re-certify).
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

import pytest

from threetears.evals.analysis.bundle.comparisons import redrawn_sentence
from threetears.evals.analysis.bundle.trials import RedrawnCells, redrawn_cells
from threetears.evals.kernel.result_condition import runs_again
from threetears.evals.ops import job_cancel, job_poll
from threetears.evals.ops import jobs as ops_jobs

from packages.evals.tests.engine_take_up import (
    EXACT_LOOKS,
    Grader,
    Reading,
    SCOPE,
    SETTINGS,
    SMOKE,
    at_a_fixed_count,
    comparison_tag,
    drained_as_uninterrupted,
    exact_rate,
    held_launch,
    make_engine,
    rows_of,
    said,
    simulate,
    taken_up_and_read,
    watched_over_planned_looks,
)
from packages.evals.tests.off_loop import off_loop


async def test_an_operator_s_first_cancel_of_a_launch_lets_its_cells_in_flight_finish_and_the_take_up_runs_the_rest() -> (
    None
):
    """One ``job_cancel`` of an engine launch starts no further cell: the cells in flight, held, keep it running, and
    once they finish each records its answer; the launch ends cancelled with no cell cut, its take-up runs only the cells
    never started, and the family says nothing ran again."""
    engine = make_engine(random.Random("drained"), "alike-20x", "grade")
    job_id, launch_id = await held_launch(engine, 8)
    await job_cancel(engine.ops, job_id, SCOPE, reason="shipping a fix")
    await asyncio.sleep(0.05)
    assert (await job_poll(engine.ops, job_id, SCOPE)).state == "running", "the cells in flight keep it running"
    assert engine.landings.holding == SETTINGS.max_concurrent_cells
    assert engine.landings.held is not None
    engine.landings.held.set()
    await engine.ops.launch.job_manager.wait_for([launch_id])
    assert (await job_poll(engine.ops, job_id, SCOPE)).state == "cancelled"
    rows = off_loop(rows_of, engine, launch_id)
    assert len(rows) == SETTINGS.max_concurrent_cells, "each cell in flight recorded, none started after"
    assert {row.termination for row in rows} == {"completed"} and not any(runs_again(row) for row in rows)

    family = await taken_up_and_read(engine, job_id, launch_id)
    rows = off_loop(rows_of, engine, launch_id)
    assert len(rows) == 16 and not any(row.written_over for row in rows)
    assert redrawn_cells(rows) == RedrawnCells()
    assert "recorded no answer" not in family.disclosure


async def test_a_second_cancel_cuts_the_cells_in_flight_and_the_family_says_how_many_ran_again() -> None:
    """A second ``job_cancel`` while the first lets the cells finish cuts them: each is recorded cut, the take-up runs it
    again written over its row (``written_over``: ``cut``), and the engine family says how many cells ran again, the cut
    ones among them, in the one sentence that says it (``redrawn_sentence``)."""
    engine = make_engine(random.Random("cut"), "alike-20x", "grade")
    job_id, launch_id = await held_launch(engine, 8)
    await job_cancel(engine.ops, job_id, SCOPE, reason="shipping a fix")
    await asyncio.sleep(0.05)
    await job_cancel(engine.ops, job_id, SCOPE, reason="shipping a fix")
    await engine.ops.launch.job_manager.wait_for([launch_id])
    rows = off_loop(rows_of, engine, launch_id)
    cut = [row for row in rows if row.termination == "cancelled"]
    assert len(cut) == SETTINGS.max_concurrent_cells, "every cell in flight was cut"

    family = await taken_up_and_read(engine, job_id, launch_id)
    rows = off_loop(rows_of, engine, launch_id)
    assert len(rows) == 16 and {row.id for row in rows if row.written_over == "cut"} == {row.id for row in cut}
    assert redrawn_cells(rows) == RedrawnCells(taken_up=len(cut), taken_up_cut=len(cut))
    assert redrawn_sentence(*redrawn_cells(rows)) in family.disclosure
    assert "ran again when its run was taken up" in family.disclosure


async def test_a_cancel_the_lease_then_finds_on_the_record_is_the_same_cancel_and_does_not_cut() -> None:
    """An operator's ``job_cancel`` cancels the launch here at once and writes the request on its record; the launch's
    lease, renewing, finds that request and does not deliver it again, so the launch drains rather than being cut by
    its own first cancel."""
    engine = make_engine(random.Random("leased"), "alike-20x", "grade", lease_s=0.3)
    job_id, launch_id = await held_launch(engine, 8)
    await job_cancel(engine.ops, job_id, SCOPE, reason="shipping a fix")
    await asyncio.sleep(0.5)  # several renewals, each reading the request
    assert (await job_poll(engine.ops, job_id, SCOPE)).state == "running", "the lease delivered no second cancel"
    assert engine.landings.held is not None
    engine.landings.held.set()
    await engine.ops.launch.job_manager.wait_for([launch_id])
    rows = off_loop(rows_of, engine, launch_id)
    assert rows and {row.termination for row in rows} == {"completed"}, "nothing was cut"


async def test_a_cancel_the_lease_delivers_before_its_own_process_counts_it_is_still_one_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``job_cancel`` writes its request on the launch's record, then cancels the launch here. Where the lease, renewing,
    finds the request and delivers it in between, the cancel here is that same request, read by what it wrote
    (``cancel_requested_at``), so it is no second cancel: the launch drains and nothing is cut."""
    engine = make_engine(random.Random("lease-first"), "alike-20x", "grade", lease_s=0.3)
    job_id, launch_id = await held_launch(engine, 8)
    written = ops_jobs.request_job_cancel

    def written_then_delivered(*args: Any, **kwargs: Any) -> Any:
        # Off the loop: the request is on the record, and the lease renews several times (every 0.1 s) before the
        # cancel here is made.
        stamp = written(*args, **kwargs)
        time.sleep(0.5)
        return stamp

    monkeypatch.setattr(ops_jobs, "request_job_cancel", written_then_delivered)
    await job_cancel(engine.ops, job_id, SCOPE, reason="shipping a fix")
    await asyncio.sleep(0.2)
    assert (await job_poll(engine.ops, job_id, SCOPE)).state == "running", "one cancel drains: the held cells run on"
    assert engine.landings.held is not None
    engine.landings.held.set()
    await engine.ops.launch.job_manager.wait_for([launch_id])
    rows = off_loop(rows_of, engine, launch_id)
    assert rows and {row.termination for row in rows} == {"completed"}, "nothing was cut"


async def test_the_reading_s_own_rate_is_exact_and_above_alpha_at_a_fixed_count() -> None:
    """The engine's paired t test on coin pairs, at a fixed count of 40, separates at 5.28% with no interrupt at all:
    the test is discrete on such pairs, so an at-most-α claim on it fails before any cut. Under the plan of three looks
    at α/3 it separates at the rate below. Held to the digits, so a change to the reading moves this first."""
    assert exact_rate("fixed") == pytest.approx(0.05276, abs=5e-5)
    assert exact_rate("looks") == pytest.approx(EXACT_LOOKS, abs=5e-5)


async def test_a_few_engine_comparisons_interrupted_and_taken_up_finish_whole_and_read_by_the_t_test() -> None:
    """The unit suite's smoke check of every simulation: a few comparisons each, every launch held through its
    interrupts and take-ups to an end where every cell has one row and none runs again (inside ``engine_take_up._launched``; a
    drained launch's take-ups run no cell again), each read by the paired t test, at α at a fixed count and at α/L under
    the plan; and, drained, the reading separating at most one of each few, which at a rate of α happens with
    probability above 0.99 (cut at every interrupt, it leans, as the cut rows show)."""
    for reading in ("fixed", "looks"):
        for stop in ("drain", "second", "cut"):
            separated, cuts = await simulate(reading, "b-slower-20x", "check", stop, SMOKE, "smoke")
            if stop != "cut":  # every interrupt cutting leans the reading: that is what the cut rows show
                assert separated <= 1, said(reading, "b-slower-20x", "check", stop, separated, SMOKE, cuts)
            assert cuts.holdings >= SMOKE
            if stop == "drain":
                assert cuts.again == {"a": 0, "b": 0} and not cuts.lost


@pytest.mark.parametrize(("reading", "grader"), [("fixed", "grade"), ("looks", "check")])
async def test_a_drained_engine_comparison_reads_as_the_uninterrupted_one_on_its_first_draws(
    reading: Reading, grader: Grader
) -> None:
    """Drained, every cell of a comparison draws once, and each look's p is the paired t test on those first draws: the
    comparison separates exactly where the uninterrupted reading of the same answers does, so its rate is the exact
    rate. A few comparisons, at the law that leans most where a cut is allowed."""
    run = at_a_fixed_count if reading == "fixed" else watched_over_planned_looks
    for index in range(SMOKE):
        tag = comparison_tag("smoke-exact", reading, "b-slower-20x", grader, "drain", index)
        outcome = await run(tag, "b-slower-20x", grader, "drain")
        assert outcome.reads and drained_as_uninterrupted(outcome) is None, drained_as_uninterrupted(outcome)
