"""A floor warns, read on one fixture shape in every direction: broken, held, and unclear on either side of the margin.

The shape: twenty cases × 3 repeats, production failing its last four cases outright and passing the rest, the candidate
failing its last ``failed`` cases outright; one
10-point floor on every pass rate and one cost floor, both on the campaign's one aspect, declared before the run. A
floor is broken exactly where its clause, read as a rule, would settle keep, at the nominal 95% with no correction: its
interval is computed again from the family comparison's per-case values, never read at the family's level. A broken
floor is the page's first line, above the title; a held one reads quietly in the Decision section, and decides nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Mapping
from typing import Any, get_args

import pytest

from threetears.evals import Answer, Eval
from threetears.evals.analysis import (
    AnalysisContextBundle,
    FloorState,
    FloorStatus,
    Report,
    assemble_context_bundle,
    build_code_only_report,
    get_campaign,
    report_html,
    report_markdown,
)
from threetears.evals.analysis.report import DisclosureBlock, TableBlock
from threetears.evals.analysis.report.decision import (
    FloorOutcome,
    aspect_pass_rate,
    evaluate_floor,
    floor_readings,
    project_rate,
)
from threetears.evals.analysis.stats import INTERVAL_LEVEL, bounded_difference_interval
from threetears.evals.analysis.style import lint_page, lint_prose
from threetears.evals.kernel import RefusedInput
from threetears.evals.kernel.declaration import CostLower, DecisionClause, Floor, RateDrop

from packages.evals.tests.aspect_support import redeclared
from packages.evals.tests.off_loop import off_loop

#: The aspect's floors: no pass rate drops by more than 10 points, and the candidate costs less per reply.
FLOORS: tuple[DecisionClause, ...] = (RateDrop(max_points=10), CostLower())

#: The question the campaign asks before it runs.
QUESTION = "Does the candidate hold the queue within 10 points of production, at lower cost?"

#: Twenty cases; production passes every trial of the first sixteen.
CASES = 20
PRODUCTION_PASSES = 16


def _first_cases(right: int, cost: float) -> Callable[[Mapping[str, Any]], Coroutine[Any, Any, Answer]]:
    """A runner answering right on every trial of the first ``right`` cases and wrong on the rest, at ``cost``."""

    async def run(case: Mapping[str, Any]) -> Answer:
        passes = int(case["text"].rsplit(" ", 1)[1]) < right
        return Answer(
            case["label"] if passes else "skipped", model="m", input_tokens=100, output_tokens=10, cost_usd=cost
        )

    return run


async def _shape(failed: int, *, candidate_cost: float = 0.0021) -> tuple[AnalysisContextBundle, str, str]:
    """The fixture shape with the candidate passing all but its last ``failed`` cases, and the floors declared."""
    cases = [{"text": f"queue request {index}", "label": "queued"} for index in range(CASES)]
    arms = {
        "production": _first_cases(PRODUCTION_PASSES, 0.0031),
        "candidate": _first_cases(CASES - failed, candidate_cost),
    }
    comparison = await Eval(cases, arms, ["label"], k=3, name="floors").run(store=":memory:")
    storage = comparison.host.storage
    campaign = off_loop(get_campaign, storage, comparison.campaign_id, comparison.scope_id)
    campaign = redeclared(
        campaign,
        question=QUESTION,
        asked_at="2000-01-01T00:00:00+00:00",
        design={"merit_axes": []},
        margin=10.0,
        floors=FLOORS,
    )
    bundle = off_loop(assemble_context_bundle, campaign, storage=storage, profile=comparison.host.profile)
    assert bundle.design.control_arm is not None
    control = bundle.design.control_arm.variant_key
    (candidate,) = [board.variant_key for board in bundle.scoreboard if board.variant_key != control]
    return bundle, control, candidate


def _statuses(bundle: AnalysisContextBundle, control: str, candidate: str) -> list[FloorStatus]:
    """Each declared floor's status on the candidate."""
    assert bundle.declared is not None
    readings = floor_readings(bundle, control, candidate)
    return [
        evaluate_floor(floor, readings, arm=candidate, arm_label="candidate", control_label="production")
        for floor in bundle.declared.floors
    ]


def _page(bundle: AnalysisContextBundle) -> Report:
    return build_code_only_report(bundle, assembled_at="2026-10-18T00:00:00+00:00", campaign_name="floors")


def _warnings(report: Report) -> list[str]:
    return [
        block.text
        for block in report.blocks
        if isinstance(block, DisclosureBlock) and block.section == "header" and block.source == "floor"
    ]


def test_a_floor_s_states_are_the_campaign_readings_states() -> None:
    """The decision module spells the three states itself, since the readings module reads it: they cannot drift."""
    assert get_args(FloorOutcome) == get_args(FloorState) == ("broken", "held", "unclear")


# =============================================================================
# One shape, every direction
# =============================================================================


async def test_fourteen_cases_failed_breaks_the_rate_floor_on_its_whole_interval() -> None:
    bundle, control, candidate = await _shape(14)
    rate, cost = _statuses(bundle, control, candidate)
    aspect = bundle.aspect
    assert (rate.floor, rate.aspect, rate.arm, rate.arm_label) == (
        f"{aspect}/rate_drop/*",
        aspect,
        candidate,
        "candidate",
    )
    assert rate.status == "broken"
    assert rate.interval is not None and rate.interval[1] < -10.0, "the whole interval past the margin, in points"
    assert rate.reading is None, "the trials' own pass rate broke it, which no clause names"
    assert rate.reason.startswith(
        "candidate passed 18 of 60 to production's 48, a drop that exceeds the margin of 10 points on its whole interval"
    )
    # The candidate costs less per reply, but a cost with no spread cannot show it: unclear, never held.
    assert (cost.floor, cost.status, cost.reading) == (f"{aspect}/cost_lower/*", "unclear", None)


async def test_a_candidate_failing_no_case_holds_the_rate_floor() -> None:
    bundle, control, candidate = await _shape(0)
    rate, _ = _statuses(bundle, control, candidate)
    assert rate.status == "held"
    assert rate.interval is not None and rate.interval[0] >= -10.0
    assert rate.reason.startswith("candidate's drop is shown to stay within 10 points")


async def test_a_drop_past_the_margin_on_its_point_estimate_alone_is_unclear_on_the_broken_side() -> None:
    bundle, control, candidate = await _shape(7)
    rate, _ = _statuses(bundle, control, candidate)
    assert rate.status == "unclear"
    assert rate.interval is not None and rate.interval[0] < -10.0 < rate.interval[1]
    assert rate.reason.startswith(
        "candidate's drop is past 10 points on the point estimate alone, not on its interval (from "
    ), rate.reason


async def test_a_drop_within_the_margin_whose_interval_reaches_past_it_is_unclear_on_the_held_side() -> None:
    bundle, control, candidate = await _shape(5)
    rate, _ = _statuses(bundle, control, candidate)
    assert rate.status == "unclear"
    assert rate.interval is not None and rate.interval[0] < -10.0
    assert rate.reason.startswith("candidate's drop is not shown to stay within 10 points (it could be as much as ")


async def test_a_candidate_costing_more_breaks_the_cost_floor_on_any_number_of_cases() -> None:
    bundle, control, candidate = await _shape(0, candidate_cost=0.0041)
    _, cost = _statuses(bundle, control, candidate)
    assert (cost.status, cost.reason) == ("broken", "candidate costs more per reply than production")


# =============================================================================
# No correction: the nominal level, whatever the family's size
# =============================================================================


async def test_the_floor_reads_its_interval_at_the_nominal_level_with_no_correction() -> None:
    bundle, control, candidate = await _shape(7)
    (family,) = bundle.multiple_comparisons.families
    assert family.family_size > 1 and family.interval_level is not None and family.interval_level > INTERVAL_LEVEL
    (passed,) = [comparison for comparison in family.comparisons if comparison.reading == "trial"]
    assert passed.case_values is not None and passed.bounded_interval is not None
    nominal = bounded_difference_interval(
        *passed.case_values, paired=passed.test == "paired", value_range=(0.0, 1.0), confidence=INTERVAL_LEVEL
    )
    assert nominal is not None
    rate, _ = _statuses(bundle, control, candidate)
    assert rate.interval == pytest.approx((100 * nominal[0], 100 * nominal[1]))
    assert rate.interval != pytest.approx(tuple(100 * end for end in passed.bounded_interval)), "not the family's"


async def test_the_aspect_pass_rate_is_the_trials_own_with_what_a_projection_reads() -> None:
    bundle, control, candidate = await _shape(5)
    rate = aspect_pass_rate(bundle, control, candidate)
    assert rate is not None and rate.measure is None and rate.reading() is None
    assert (rate.control_passed, rate.control_n, rate.candidate_passed, rate.candidate_n) == (48, 60, 45, 60)
    assert rate.samples is not None and rate.confidence == INTERVAL_LEVEL
    # Projected against the aspect's margin, it says how many cases would show it within.
    assert bundle.declared is not None and bundle.declared.margin == 10.0
    projection = project_rate(rate, cases=CASES, margin=bundle.declared.margin)
    assert projection is not None and projection.if_none.kind == "more" and projection.if_none.cases > CASES
    assert aspect_pass_rate(bundle, control, "no-such-arm") is None


# =============================================================================
# The page: a broken floor first, loud; a held one quiet
# =============================================================================


async def test_a_broken_floor_s_warning_is_the_first_line_of_the_page() -> None:
    bundle, _, _ = await _shape(14)
    report = _page(bundle)
    (warning,) = _warnings(report)
    assert warning.startswith(
        f"Floor broken on {bundle.aspect}: candidate's pass rate falls past the 10-point margin on its whole interval, "
        "a drop of "
    ), warning
    assert lint_prose(warning) == []
    first = report.blocks[0]
    assert isinstance(first, DisclosureBlock) and first.text == warning
    markdown = report_markdown(report)
    assert markdown.splitlines()[0] == f"**{warning}**", "above the title"
    assert markdown.splitlines()[2] == f"# {report.title}"
    page = report_html(report)
    assert page.index('<p class="floor" role="alert">') < page.index("<h1>")
    assert lint_page(markdown, question=QUESTION) == []


async def test_a_held_floor_reads_quietly_and_opens_nothing() -> None:
    bundle, _, _ = await _shape(0)
    report = _page(bundle)
    assert _warnings(report) == []
    assert report_markdown(report).startswith(f"# {report.title}\n")
    (table,) = [block for block in report.blocks if isinstance(block, TableBlock) and block.name == "decision"]
    statuses = [str(row["outcome"]) for row in table.values()]
    assert statuses[0].startswith("held: candidate's drop is shown to stay within 10 points")


async def test_no_floor_decides_the_verdict_line() -> None:
    bundle, _, _ = await _shape(14)
    (verdict,) = [
        block.text for block in _page(bundle).blocks if isinstance(block, DisclosureBlock) and block.source == "verdict"
    ]
    assert not verdict.startswith(("Keep", "Switch", "Unsettled")), verdict


# =============================================================================
# The bundle reads one aspect, named or implied
# =============================================================================


async def test_a_bundle_names_the_aspect_it_reads_and_refuses_one_the_campaign_lacks() -> None:
    cases = [{"text": f"queue request {index}", "label": "queued"} for index in range(5)]
    arms = {"production": _first_cases(5, 0.0031), "candidate": _first_cases(4, 0.0021)}
    comparison = await Eval(cases, arms, ["label"], k=1, name="aspects").run(store=":memory:")
    storage, profile = comparison.host.storage, comparison.host.profile
    campaign = off_loop(get_campaign, storage, comparison.campaign_id, comparison.scope_id)
    implied = off_loop(assemble_context_bundle, campaign, storage=storage, profile=profile)
    named = off_loop(assemble_context_bundle, campaign, aspect=implied.aspect, storage=storage, profile=profile)
    assert named.aspect == implied.aspect == implied.behavior
    assert named.fingerprint() == implied.fingerprint()
    with pytest.raises(RefusedInput, match="has no aspect 'elsewhere'"):
        off_loop(assemble_context_bundle, campaign, aspect="elsewhere", storage=storage, profile=profile)


def test_a_floor_key_names_its_aspect_kind_and_reading() -> None:
    assert Floor(aspect="queue", clause=RateDrop(max_points=5, reading="tone")).key == "queue/rate_drop/tone"
