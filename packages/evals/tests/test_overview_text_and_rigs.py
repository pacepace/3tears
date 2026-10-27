"""The overview's text leaves out what its model leaves out, and its rigs fit a page however many runs share them.

``campaign_overview`` carries two fields that are left out of its document when unset: each arm row's ``unanswered``
(a list row's field) and the overview's ``apparatus`` (a field of the result itself). Contracts held here:

- **the text as the analyst reads it** (:func:`~threetears.evals.actions.render.render_cited`): an overview over an
  input with no unanswered calls and no apparatus writes neither an ``unanswered`` nor an ``apparatus`` line, as the
  overview was written before the fields existed; over an input carrying both it writes both, the arm row's under the
  row and the rigs under the overview;
- **one row per rig**: runs measured alike share one row naming every run, their execution modes summed; runs measured
  differently are a row each;
- **the page budget**: eight runs shaped as a real host's (nineteen pass/fail rubric dimensions and the two reserved
  axes, each named in the judges and the scales) fit one page when they share their rig; measured eight different
  ways, they are paged as rows, every page within the 12,000 characters one result may take, and following the pages
  reads every rig once.
"""

from __future__ import annotations

from typing import Any

from packages.evals.tests.run_facts_support import CASES, campaign_over, result_of, run_of
from threetears.evals.actions.analyst_facts import SnapshotParams
from threetears.evals.actions.render import page_sizing, page_unit, render_cited, render_page
from threetears.evals.analysis.analyst.facts import OVERVIEW_LIMIT, CampaignOverview, campaign_overview
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.ops.analyst import sized_page
from threetears.evals.schema.models import OUTCOME_DIM_ID, TRANSCRIPT_DIM_ID, EvalResult, EvalRun, RoleUsage

SNAPSHOT = "report-snapshot:" + "0" * 32

#: The most characters one analyst action's text may run to (``analyst_support.TEXT_BUDGET``'s figure).
BUDGET = 12_000

#: A host's judged campaign: nineteen pass/fail rubric dimensions with long names, and the judge it pins.
DIMS = [f"persona.holds_character_under_pressure_{index:02d}" for index in range(19)]
JUDGE = "anthropic/claude-sonnet-4-5-20250929"


def _trials(**run: Any) -> list[tuple[EvalRun, list[EvalResult]]]:
    base, fast = run_of("base", **run), run_of("fast", **run)
    counted = [
        RoleUsage(role="candidate", model="fast", cost_usd=0.002, call_count=3, unanswered_retried=1, unanswered_lost=0)
    ]
    return [
        (base, [result_of(base, index) for index in range(CASES)]),
        (fast, [result_of(fast, index, usage=counted) for index in range(CASES)]),
    ]


# --- the text --------------------------------------------------------------------------------------------------------


def test_an_overview_with_neither_writes_no_unanswered_and_no_apparatus_line() -> None:
    plain = report_input(
        campaign_over(
            [(run, [result_of(run, index) for index in range(CASES)]) for run in (run_of("base"), run_of("fast"))]
        )
    )
    text = render_cited(campaign_overview(plain, SNAPSHOT))
    # The arm rows are a list's rows (each a bullet of its plain fields); the rigs a field of the overview itself.
    assert [line for line in text.splitlines() if line.startswith("arms (")]
    assert "unanswered" not in text
    assert not [line for line in text.splitlines() if line.startswith("apparatus")]


def test_an_overview_with_both_writes_the_arms_unanswered_calls_and_the_rigs() -> None:
    held = report_input(campaign_over(_trials(judge_model=JUDGE)))
    text = render_cited(campaign_overview(held, SNAPSHOT))
    assert "calls.unanswered.fast = 6" in text
    assert "calls.unanswered.base" not in text
    assert "\napparatus (1):\n" in text
    assert "runs: [run-base, run-fast]" in text


# --- one row per rig -------------------------------------------------------------------------------------------------


def test_runs_measured_alike_share_one_row_and_runs_measured_apart_are_a_row_each() -> None:
    alike = campaign_overview(report_input(campaign_over(_trials(judge_model=JUDGE))), SNAPSHOT).apparatus
    assert [row.runs for row in alike] == [["run-base", "run-fast"]]
    assert alike[0].arms == ["base", "fast"]
    base, fast = run_of("base", judge_model=JUDGE), run_of("fast", judge_model=JUDGE, cell_concurrency=4)
    apart = report_input(
        campaign_over([(run, [result_of(run, index) for index in range(CASES)]) for run in (base, fast)])
    )
    assert [row.runs for row in campaign_overview(apart, SNAPSHOT).apparatus] == [["run-base"], ["run-fast"]]


def test_a_rows_execution_modes_are_its_runs_summed() -> None:
    runs = [run_of(arm, judge_model=JUDGE) for arm in ("base", "fast")]
    trials = [
        (run, [result_of(run, index, covariates={"execution_mode": "concurrent"}) for index in range(CASES)])
        for run in runs
    ]
    (row,) = campaign_overview(report_input(campaign_over(trials)), SNAPSHOT).apparatus
    assert row.execution_modes == {"concurrent": 2 * CASES}


# --- the page budget -------------------------------------------------------------------------------------------------


def _host_shaped(*, apart: bool) -> ReportInput:
    """Eight arms, one run each, judged on nineteen pass/fail dims and the two reserved axes; ``apart`` asks each run's
    judge for its own temperature, so no two share a rig."""
    judges = {**dict.fromkeys(DIMS, JUDGE), OUTCOME_DIM_ID: JUDGE, TRANSCRIPT_DIM_ID: JUDGE}
    runs = [
        run_of(
            f"arm-{index}",
            judge_model=JUDGE,
            effective_judges=judges,
            rubric_scales=dict.fromkeys(DIMS, "pass_fail"),
            measure_latency=False,
            arm_schedule="interleaved",
            judge_temperature=index / 10 if apart else 0.0,
        )
        for index in range(8)
    ]
    return report_input(campaign_over([(run, [result_of(run, index) for index in range(CASES)]) for run in runs]))


def _page(held: ReportInput, offset: int) -> tuple[CampaignOverview, str]:
    params = SnapshotParams(snapshot_id=SNAPSHOT, offset=offset)
    limit, fits = page_sizing("campaign_overview", params, OVERVIEW_LIMIT)
    page = sized_page(lambda rows: campaign_overview(held, SNAPSHOT, limit=rows, offset=offset), limit, fits)
    text, _ = render_page(
        "campaign_overview", params, render_cited(page), page.model_dump(mode="json"), page_unit(page)
    )
    return page, text


def test_eight_host_shaped_runs_sharing_a_rig_fit_one_page_as_one_row() -> None:
    page, text = _page(_host_shaped(apart=False), 0)
    assert len(text) <= BUDGET
    assert page.next_offset is None
    assert [len(row.runs) for row in page.apparatus] == [8]


def test_eight_host_shaped_rigs_are_paged_every_page_within_the_budget_and_each_read_once() -> None:
    held = _host_shaped(apart=True)
    read: list[list[str]] = []
    offset: int | None = 0
    while offset is not None:
        page, text = _page(held, offset)
        assert len(text) <= BUDGET, (offset, len(text))
        read += [row.runs for row in page.apparatus]
        offset = page.next_offset
    assert sorted(read) == sorted([[f"run-arm-{index}"] for index in range(8)])
