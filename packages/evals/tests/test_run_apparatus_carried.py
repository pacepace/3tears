"""The rig a run was measured with, carried from the run to the bundle, the report input and the analyst's overview.

A run records its apparatus at launch (:class:`~threetears.evals.schema.models.EvalRun`): which model each judged
dimension was asked of, what every judge call was asked for and at what temperature, the simulated user's model,
whether latency was under test, how its arm shared time with the others, how many cells ran at once, and each rubric
dimension's scale; each result records the execution mode it ran in. Contracts held here:

- **the bundle**: each run summary carries the run's apparatus, copied (``RunSummary.apparatus``), with its results
  counted by execution mode; a run that recorded nothing past a serial run pinning and declaring nothing carries none,
  and its summary dumps with no ``apparatus`` key, as it did before the field existed;
- **None is kept through every hop**: a field the run left unset is None in the bundle, the report input and the
  overview, never a default standing in for it; ``measure_latency`` False and None stay two answers;
- **the report input**: ``apparatus`` lists each run that recorded one, by run, arm and aspect, its judge's asked
  settings copied; an input over runs that recorded none carries no ``apparatus`` key;
- **the overview**: ``campaign_overview`` shows each rig once, naming the runs measured with it, and leaves the field
  out where the input holds none (``test_overview_text_and_rigs.py`` holds its text and its page budget).
"""

from __future__ import annotations

from packages.evals.tests.run_facts_support import ASPECT, CASES, aspect_bundle, campaign_over, result_of, run_of
from threetears.evals.analysis.analyst.facts import campaign_overview
from threetears.evals.analysis.bundle.schema import RunApparatus
from threetears.evals.analysis.report_input import RequestSettings
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.schema.models import OUTCOME_DIM_ID, TRANSCRIPT_DIM_ID, EvalResult, EvalRun

BASE = "base"
FAST = "fast"
JUDGE = "judge/one"


def _rigged() -> list[tuple[EvalRun, list[EvalResult]]]:
    """The control judged, its latency under test, its arms interleaved, one pass/fail rubric dim; the candidate four
    cells at once, latency not under test, a simulated user, every result marked concurrent but one serial."""
    base = run_of(
        BASE,
        judge_model=JUDGE,
        measure_latency=True,
        arm_schedule="interleaved",
        rubric_scales={"answer.correct": "pass_fail"},
    )
    fast = run_of(FAST, measure_latency=False, cell_concurrency=4, simulator_model="sim/one")
    return [
        (base, [result_of(base, index) for index in range(CASES)]),
        (
            fast,
            [
                result_of(fast, index, covariates={"execution_mode": "serial" if index == 0 else "concurrent"})
                for index in range(CASES)
            ],
        ),
    ]


def _plain() -> list[tuple[EvalRun, list[EvalResult]]]:
    """Two serial runs that pinned and declared nothing."""
    base, fast = run_of(BASE), run_of(FAST)
    return [(run, [result_of(run, index) for index in range(CASES)]) for run in (base, fast)]


# --- the bundle -----------------------------------------------------------------------------------------------------


def test_each_run_summary_carries_the_rig_its_run_recorded() -> None:
    trials = _rigged()
    _, bundle, _ = aspect_bundle(trials)
    (base_run, _), (fast_run, _) = trials
    summaries = {summary.run_id: summary for summary in bundle.run_summaries}
    assert summaries[base_run.id].apparatus == RunApparatus(
        effective_judges={OUTCOME_DIM_ID: JUDGE, TRANSCRIPT_DIM_ID: JUDGE},
        judge_asked_settings=base_run.judge_asked_settings,
        judge_temperature=base_run.judge_temperature,
        measure_latency=True,
        arm_schedule="interleaved",
        cell_concurrency=1,
        rubric_scales={"answer.correct": "pass_fail"},
    )
    assert summaries[fast_run.id].apparatus == RunApparatus(
        simulator_model="sim/one",
        measure_latency=False,
        cell_concurrency=4,
        execution_modes={"concurrent": CASES - 1, "serial": 1},
    )


def test_a_field_the_run_left_unset_is_none_in_the_bundle_never_a_default() -> None:
    _, bundle, _ = aspect_bundle(_rigged())
    fast = next(summary.apparatus for summary in bundle.run_summaries if summary.candidate_model == FAST)
    assert fast is not None
    assert (fast.effective_judges, fast.judge_asked_settings, fast.judge_temperature, fast.arm_schedule) == (
        None,
        None,
        None,
        None,
    )
    assert fast.measure_latency is False


def test_a_run_that_recorded_nothing_carries_no_apparatus_and_dumps_as_before() -> None:
    _, bundle, _ = aspect_bundle(_plain())
    assert [summary.apparatus for summary in bundle.run_summaries] == [None, None]
    assert all("apparatus" not in summary.to_dict() for summary in bundle.run_summaries)
    assert all("apparatus" not in summary for summary in bundle.to_dict()["run_summaries"])


# --- the report input -----------------------------------------------------------------------------------------------


def test_the_input_lists_each_rigged_run_by_run_arm_and_aspect() -> None:
    trials = _rigged()
    held = report_input(campaign_over(trials))
    (base_run, _), (fast_run, _) = trials
    assert held.apparatus is not None
    rigs = {rig.run: rig for rig in held.apparatus}
    assert sorted(rigs) == sorted([base_run.id, fast_run.id])
    base, fast = rigs[base_run.id], rigs[fast_run.id]
    assert (base.arm, base.aspect, fast.arm, fast.aspect) == (BASE, ASPECT, FAST, ASPECT)
    asked = base_run.judge_asked_settings
    assert asked is not None
    assert base.judge_asked == RequestSettings(
        max_tokens=asked.max_tokens,
        reasoning_max_tokens=asked.reasoning_max_tokens,
        reasoning_effort=asked.reasoning_effort,
        strict_output=asked.strict_output,
    )
    assert (base.judges, base.judge_temperature, base.measure_latency, base.arm_schedule) == (
        {OUTCOME_DIM_ID: JUDGE, TRANSCRIPT_DIM_ID: JUDGE},
        base_run.judge_temperature,
        True,
        "interleaved",
    )
    assert (base.simulator, base.cell_concurrency, base.rubric_scales) == (None, 1, {"answer.correct": "pass_fail"})
    assert (fast.simulator, fast.cell_concurrency, fast.execution_modes) == (
        "sim/one",
        4,
        {"concurrent": CASES - 1, "serial": 1},
    )
    assert (fast.judges, fast.judge_asked, fast.judge_temperature, fast.arm_schedule, fast.measure_latency) == (
        None,
        None,
        None,
        None,
        False,
    )


def test_an_input_over_runs_that_recorded_none_carries_no_apparatus_key() -> None:
    held = report_input(campaign_over(_plain()))
    assert held.apparatus is None
    assert "apparatus" not in held.model_dump(mode="json", by_alias=True)


def test_the_input_round_trips_with_its_apparatus() -> None:
    held = report_input(campaign_over(_rigged()))
    assert type(held).model_validate(held.model_dump(mode="json", by_alias=True)) == held


# --- the overview ---------------------------------------------------------------------------------------------------


def test_the_overview_shows_each_rig_once_naming_its_runs() -> None:
    held = report_input(campaign_over(_rigged()))
    assert held.apparatus is not None
    rows = campaign_overview(held, "snap").apparatus
    assert sorted(run for row in rows for run in row.runs) == sorted(rig.run for rig in held.apparatus)
    assert {run: arm for row in rows for run, arm in row.runs.items()} == {rig.run: rig.arm for rig in held.apparatus}
    by_run = {rig.run: rig for rig in held.apparatus}
    for row in rows:
        (first, *_) = (by_run[run] for run in row.runs)
        assert (row.judges, row.judge_asked, row.simulator, row.measure_latency, row.cell_concurrency) == (
            first.judges,
            first.judge_asked,
            first.simulator,
            first.measure_latency,
            first.cell_concurrency,
        )


def test_the_overview_of_an_input_with_no_apparatus_leaves_the_field_out() -> None:
    overview = campaign_overview(report_input(campaign_over(_plain())), "snap")
    assert overview.apparatus == []
    assert "apparatus" not in overview.model_dump()
