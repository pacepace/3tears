"""An aspect's floors, each read on a candidate, on the G2 campaign's own numbers.

G2 (2026-10) declared: switch only if no pass rate drops by more than 10 points and the cheaper model costs less per
reply, otherwise keep production. That rule reads back as two floors on its one aspect (``RateDrop(max_points=10)`` and
``CostLower()``), which warn and never decide. It ran 4 cases × 3 repeats: production passed 10 of 12, the cheaper
candidate 2 of 12, at a cost per reply of 0.003092 against 0.003301. A cost observed higher breaks the cost floor on
any number of cases: the arm's price is not a sample. A rate floor is read either way only on an interval over enough
cases: broken where the drop's whole interval is past the margin, held where it is within it. A drop past the margin
on its point estimate alone is unclear, and says so (#845). Each rate's interval is the bounded one, computed again at
the nominal 95% from the family comparison's per-case values, uncorrected.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal

from threetears.evals import Answer, Eval, Result
from threetears.evals.analysis import (
    AnalysisContextBundle,
    Report,
    assemble_context_bundle,
    build_code_only_report,
    get_campaign,
)
from threetears.evals.analysis.bundle.schema import JudgedArm, JudgedMeasure
from threetears.evals.analysis.report import DisclosureBlock, TableBlock
from threetears.evals.analysis import report_html, report_markdown
from threetears.evals.analysis.report.build import FLOORS_WARN_ONLY
from threetears.evals.analysis.report.decision import (
    FloorReadings,
    FloorVerdict,
    RateReading,
    floor_readings,
    floor_statement,
    read_floor,
)
from threetears.evals.analysis.stats import INTERVAL_LEVEL, bounded_difference_interval
from threetears.evals.kernel.declaration import (
    CostLower,
    DecisionClause,
    DesignRefused,
    Floor,
    RateDrop,
    decision_readings,
    refuse_an_undeclarable_design,
)
from threetears.evals.kernel.metrics import ACCURACY_MEASURE, MetricDescriptor, goal_check_measure
import pytest
from pydantic import ValidationError

from packages.evals.tests.aspect_support import redeclared
from packages.evals.tests.factories import minimal_aspect_design
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.run import toyhost_template
from packages.evals.tests.g2_support import G2_FLOORS, G2_QUESTION, g2_ruled_report
from packages.evals.tests.off_loop import off_loop


#: The arms by the names a page gives them, which every reason says.
_ARMS = {"control": "production", "candidate": "cheaper candidate"}


def _read(
    clauses: Sequence[DecisionClause],
    readings: FloorReadings,
    *,
    control: str = "production",
    candidate: str = "cheaper candidate",
) -> list[FloorVerdict]:
    """Each clause read as a floor on one aspect, on ``readings``."""
    return [
        read_floor(Floor(aspect="queue", clause=clause), readings, control=control, candidate=candidate)
        for clause in clauses
    ]


@pytest.fixture
async def g2_ruled() -> tuple[Result, Report]:
    """The G2 numbers through the quick path, with the G2 floors declared before the run."""
    return await g2_ruled_report()


def test_the_g2_floors_are_both_unclear_under_the_case_floor() -> None:
    """#845 changed this contract: a 66.7-point drop on four cases once settled the rate clause ``keep``. And until the
    cost floor read its own interval, a cost observed higher broke it on any number of cases.

    Four cases are under the floor for any interval, so no drop is shown past the margin: the rate floor is unclear and
    says the point estimate is past, and why that is not enough. The cost floor, read on its own interval as a rate is,
    is unclear on four cases too, and says so.
    """
    readings = FloorReadings(
        rates=(RateReading("the pass rate", 10, 12, 2, 12),),
        control_cost=0.003092,
        candidate_cost=0.003301,
        control_cases=4,
        candidate_cases=4,
        cost_interval=(0.0001, 0.0003),
        cost_compared=True,
    )
    rate, cost = _read(G2_FLOORS, readings)
    assert (rate.outcome, cost.outcome) == ("unclear", "unclear")
    assert cost.reason == (
        "4 cases are too few for the interval that would show the cost either way, which is estimated only from 5 "
        "independent cases"
    )
    assert rate.reason == (
        "cheaper candidate's drop is past 10 points on the point estimate alone (4 cases are too few for the interval "
        "that would show it, which is estimated only from 5 independent cases)"
    )
    assert rate.rate is not None and rate.rate.measure is None, "the rate that left it unclear is named"


def test_a_rate_floor_the_remaining_trials_could_still_decide_is_unclear() -> None:
    close = FloorReadings(
        rates=(RateReading("the pass rate", 10, 12, 8, 10, 0, 2, interval=(-0.05, 0.02)),),
        control_cost=2.0,
        candidate_cost=1.0,
        control_cases=6,
        candidate_cases=6,
        cost_interval=(-1.5, -0.5),
        cost_compared=True,
    )
    rate, cost = _read(G2_FLOORS, close)
    assert (rate.outcome, cost.outcome) == ("unclear", "held")
    assert rate.reason == (
        "cheaper candidate's drop is not shown to stay within 10 points (the lost trials could still take it past)"
    )
    # Equal rates hold only on an interval over enough cases. The bare counts (no case count, no interval) once
    # settled switch here: that was the defect.
    settled = FloorReadings(
        rates=(RateReading("the pass rate", 12, 12, 12, 12, interval=(-0.04, 0.04)),),
        control_cost=2.0,
        candidate_cost=1.0,
        control_cases=6,
        candidate_cases=6,
        cost_interval=(-1.5, -0.5),
        cost_compared=True,
    )
    assert [verdict.outcome for verdict in _read(G2_FLOORS, settled)] == ["held", "held"]
    bare = FloorReadings(rates=(RateReading("the pass rate", 12, 12, 12, 12),), control_cost=2.0, candidate_cost=1.0)
    assert [verdict.outcome for verdict in _read(G2_FLOORS, bare)] == ["unclear", "unclear"]


def test_equal_rates_on_too_few_cases_are_unclear_not_held() -> None:
    equal = FloorReadings(
        rates=(RateReading("the pass rate", 10, 12, 10, 12, interval=(-0.02, 0.02)),),
        control_cost=2.0,
        candidate_cost=1.0,
        control_cases=4,
        candidate_cases=4,
        cost_interval=(-1.5, -0.5),
        cost_compared=True,
    )
    rate, cost = _read(G2_FLOORS, equal)
    assert (rate.outcome, cost.outcome) == ("unclear", "unclear")
    # What is withheld and why, in the reader's terms: never a bare "needs 5".
    assert rate.reason == (
        "cheaper candidate's drop is not shown to stay within 10 points (4 cases are too few for the interval that "
        "would show it, which is estimated only from 5 independent cases)"
    )
    assert cost.reason == (
        "4 cases are too few for the interval that would show the cost either way, which is estimated only from 5 "
        "independent cases"
    )


def test_a_floor_met_run_with_a_tight_interval_holds() -> None:
    tight = FloorReadings(
        rates=(RateReading("the pass rate", 50, 60, 49, 60, interval=(-0.08, 0.05)),),
        control_cost=0.0031,
        candidate_cost=0.0021,
        control_cases=20,
        candidate_cases=20,
        cost_interval=(-0.0014, -0.0006),
        cost_compared=True,
    )
    rate, cost = _read(G2_FLOORS, tight)
    assert (rate.outcome, cost.outcome) == ("held", "held")
    assert rate.reason == "cheaper candidate's drop is shown to stay within 10 points (it is at most 8)"
    assert rate.interval() == pytest.approx((-8.0, 5.0)), "in points, the candidate minus the control"
    # The same run with an interval reaching past the margin is unclear, and says how far it reaches.
    wide = replace(tight, rates=(RateReading("the pass rate", 50, 60, 49, 60, interval=(-0.23, 0.2)),))
    (unclear, _) = _read(G2_FLOORS, wide)
    assert unclear.outcome == "unclear"
    assert (
        unclear.reason == "cheaper candidate's drop is not shown to stay within 10 points (it could be as much as 23)"
    )


def test_a_cost_lower_by_a_point_estimate_only_is_unclear() -> None:
    readings = FloorReadings(
        rates=(RateReading("the pass rate", 50, 60, 50, 60, interval=(-0.05, 0.05)),),
        control_cost=0.0031,
        candidate_cost=0.0029,
        control_cases=20,
        candidate_cases=20,
        cost_interval=(-0.0006, 0.0002),
        cost_compared=True,
    )
    rate, cost = _read(G2_FLOORS, readings)
    assert (rate.outcome, cost.outcome) == ("held", "unclear")
    assert cost.reason == "the cost difference did not separate"
    assert cost.interval() == (-0.0006, 0.0002), "a cost floor's interval is in dollars per reply"
    # Until the cost floor read its own interval, a cost observed higher broke it on any number of cases. Now it is
    # broken only on an interval wholly above zero, over enough cases, as a rate floor is.
    dearer = replace(readings, control_cost=0.0029, candidate_cost=0.0031, cost_interval=(0.0001, 0.0003))
    assert _read(G2_FLOORS, dearer)[1].outcome == "broken"
    straddling = replace(dearer, cost_interval=(-0.0001, 0.0003))
    assert _read(G2_FLOORS, straddling)[1].outcome == "unclear"
    thin = replace(dearer, control_cases=4, candidate_cases=4)
    assert _read(G2_FLOORS, thin)[1].outcome == "unclear"


def test_a_cost_floor_says_why_it_reads_no_interval() -> None:
    """Cost compared on no case says so apart from cost alike on every case, and from cost not measured at all."""
    base = FloorReadings(
        rates=(RateReading("the pass rate", 50, 60, 50, 60, interval=(-0.05, 0.05)),),
        control_cost=0.0031,
        candidate_cost=0.0041,
        control_cases=20,
        candidate_cases=20,
    )
    assert _read(G2_FLOORS, base)[1].reason == "cost per reply was not compared: no case carries it on both arms"
    alike = replace(base, cost_compared=True)
    assert _read(G2_FLOORS, alike)[1].reason == (
        "every case cost alike on both arms, so no interval on the cost difference exists"
    )
    unmeasured = replace(base, candidate_cost=None)
    assert _read(G2_FLOORS, unmeasured)[1].reason == "cost per reply was not measured on both arms"
    assert all(_read(G2_FLOORS, readings)[1].outcome == "unclear" for readings in (base, alike, unmeasured))


def _verdict_line(report: Report) -> str:
    """The page's verdict line."""
    (verdict,) = [b for b in report.blocks if isinstance(b, DisclosureBlock) and b.source == "verdict"]
    return verdict.text


def _floor_rows(report: Report) -> list[tuple[str, str]]:
    """The Decision table's rows, as each row's reading and its status."""
    (table,) = [b for b in report.blocks if isinstance(b, TableBlock) and b.name == "decision"]
    return [(str(row["reading"]), str(row["outcome"])) for row in table.values()]


def _warnings(report: Report) -> list[str]:
    """The broken floors' warnings, each a line under the verdict."""
    return [
        b.text
        for b in report.blocks
        if isinstance(b, DisclosureBlock) and b.section == "header" and b.source == "floor"
    ]


async def test_on_four_cases_no_floor_is_broken_and_the_verdict_decides_nothing(
    g2_ruled: tuple[Result, Report],
) -> None:
    _, report = g2_ruled
    # On four cases no interval reads either floor, so none is broken and the page opens on its title.
    assert _warnings(report) == []
    assert report_markdown(report).startswith(f"# {report.title}\n")
    # No floor decides the verdict: on four cases it is the too-few line, the question declared.
    assert _verdict_line(report) == (
        "cheaper candidate passed less often than production; too few cases to call a difference."
    )
    (table,) = [b for b in report.blocks if isinstance(b, TableBlock) and b.name == "decision"]
    # One candidate: each arm heads its own column by name.
    assert [column.header for column in table.columns] == [
        "Floor",
        "Reading",
        "production",
        "cheaper candidate",
        "Status",
    ]
    # Each reading its own row and its own status: a status is never beside a reading it is not about.
    floor = "4 cases are too few for the interval that would show it, which is estimated only from 5 independent cases"
    assert _floor_rows(report) == [
        (
            "the pass rate",
            f"unclear: cheaper candidate's drop is past 10 points on the point estimate alone ({floor})",
        ),
        (
            "Label matched",
            f"unclear: cheaper candidate's drop on Label matched is past 10 points on the point estimate alone "
            f"({floor})",
        ),
        (
            "cost per 1,000 replies",
            "unclear: 4 cases are too few for the interval that would show the cost either way, which is estimated "
            "only from 5 independent cases",
        ),
    ]
    # Under a cent per reply on every arm, the cost row reads per 1,000 replies, as the scoreboard does.
    assert {"reading": "cost per 1,000 replies", "control": "$3.092", "candidate": "$3.301"}.items() <= next(
        row for row in table.values() if row["reading"] == "cost per 1,000 replies"
    ).items()


async def test_a_floor_beside_a_contrast_not_separated_says_why_once_under_the_floors(
    g2_ruled: tuple[Result, Report],
) -> None:
    """A floor can read unclear beside a contrast the contrasts table reads as not separated, and says why once.

    A floor warns and never decides, and reads its own uncorrected bounded interval, which can differ from the t
    interval the contrasts table prints. The page says so once, under the floors, and each reading's row carries its
    own status, its reason saying the counts and the margin in words, naming the arms. A rate floor is broken only
    where that interval is wholly past the margin (#845).
    """
    _, report = g2_ruled
    said = [b for b in report.blocks if isinstance(b, DisclosureBlock) and b.section == "decision"]
    assert [(block.source, block.text) for block in said][0] == ("floor", FLOORS_WARN_ONLY)
    markdown = report_markdown(report)
    assert markdown.count(FLOORS_WARN_ONLY) == 1
    decision = markdown.split("\n## Floors\n", 1)[1].split("\n## ", 1)[0]
    assert decision.index(f"**Floors on {report.bundle.aspect}**") < decision.index(FLOORS_WARN_ONLY)
    assert "unclear: 4 cases are too few for the interval that would show the cost either way" in decision
    # The prod numbers: v2 passed 14 of 60 trials to v1's 29. Shown past the margin on the whole interval, it is
    # broken, and says the drop exceeds the margin; past it on the point estimate alone, it is unclear.
    on = {"control": "v1", "candidate": "v2"}
    rule = (RateDrop(max_points=10),)
    dropped = FloorReadings(
        rates=(RateReading("the pass rate", 29, 60, 14, 60, interval=(-0.4, -0.123)),),
        control_cases=20,
        candidate_cases=20,
    )
    assert _read(rule, dropped, **on)[0].reason == (
        "v2 passed 14 of 60 to v1's 29, a drop that exceeds the margin of 10 points on its whole interval (at least "
        "12.3)"
    )
    unshown_drop = replace(dropped, rates=(replace(dropped.rates[0], interval=(-0.536, 0.036)),))
    assert _read(rule, unshown_drop, **on)[0].reason == (
        "v2's drop is past 10 points on the point estimate alone, not on its interval (from a gain of 3.6 to a drop "
        "of 53.6)"
    )
    wide = FloorReadings(
        rates=(RateReading("the pass rate", 29, 60, 28, 60, interval=(-0.536, 0.036)),),
        control_cases=20,
        candidate_cases=20,
    )
    (unshown,) = _read(rule, wide, **on)
    assert (unshown.outcome, unshown.reason) == (
        "unclear",
        "v2's drop is not shown to stay within 10 points (it could be as much as 53.6)",
    )


# --- a floor reads the bounded interval, at the nominal level, on every rate it reads ----------------------------


def _passing(passes: Sequence[int], cost: float) -> Callable[[Mapping[str, Any]], Coroutine[Any, Any, Answer]]:
    """A runner that answers case ``i`` right on its first ``passes[i]`` repeats and wrong after, each at ``cost``."""
    calls: Counter[str] = Counter()

    async def run(case: Mapping[str, Any]) -> Answer:
        calls[case["text"]] += 1
        right = calls[case["text"]] <= passes[int(case["text"].rsplit(" ", 1)[1])]
        return Answer(
            case["label"] if right else "skipped", model="m", input_tokens=100, output_tokens=10, cost_usd=cost
        )

    return run


#: When G2's question was asked: before any run, so it confirms.
_ASKED_AT = "2000-01-01T00:00:00+00:00"


def _margin(floors: Sequence[DecisionClause]) -> float | None:
    """The aspect's margin, as an upgraded aspect's is: the floor reading every pass rate's."""
    return next((c.max_points for c in floors if isinstance(c, RateDrop) and c.reading is None), None)


async def _assembled(comparison: Result, floors: Sequence[DecisionClause]) -> AnalysisContextBundle:
    """``comparison``'s campaign with G2's question asked before its runs and ``floors`` on its aspect, assembled."""
    storage = comparison.host.storage
    campaign = off_loop(get_campaign, storage, comparison.campaign_id, comparison.scope_id)
    campaign = redeclared(
        campaign,
        question=G2_QUESTION,
        asked_at=_ASKED_AT,
        design={"merit_axes": []},
        margin=_margin(floors),
        floors=floors,
    )
    return off_loop(assemble_context_bundle, campaign, storage=storage, profile=comparison.host.profile)


async def _ruled(
    control_passes: Sequence[int], candidate_passes: Sequence[int], floors: Sequence[DecisionClause] = G2_FLOORS
) -> tuple[AnalysisContextBundle, str, str]:
    """Two arms over ``len(control_passes)`` cases × 3 repeats, each passing its cases' counts, with G2's floors.

    The floors are on the campaign before its bundle is assembled, so its family is one a floor can read: the label
    check's pass rate (compared as ``accuracy``) and, with a margin declared, the overall pass rate.
    """
    cases = [{"text": f"queue request {index}", "label": "queued"} for index in range(len(control_passes))]
    # Each arm at one cost per reply: a gap with no spread on a measure with no declared range, which no test reads, so
    # the family is the two rates alone.
    arms = {"production": _passing(control_passes, 0.0031), "candidate": _passing(candidate_passes, 0.0021)}
    comparison = await Eval(cases, arms, ["label"], k=3, name="ruled").run(store=":memory:")
    bundle = await _assembled(comparison, floors)
    assert bundle.design.control_arm is not None
    control = bundle.design.control_arm.variant_key
    (candidate,) = [board.variant_key for board in bundle.scoreboard if board.variant_key != control]
    assert [board.passed for board in bundle.scoreboard] == [sum(control_passes), sum(candidate_passes)]
    return bundle, control, candidate


def _rate_floor(bundle: AnalysisContextBundle, control: str, candidate: str) -> tuple[str, str]:
    """G2's rate floor on the bundle: its status and its reason."""
    (verdict, _) = _read(G2_FLOORS, floor_readings(bundle, control, candidate))
    assert isinstance(verdict.floor.clause, RateDrop)
    return verdict.outcome, verdict.reason


async def test_the_overall_pass_rate_joins_the_family_when_the_aspect_declares_a_margin() -> None:
    bundle, control, candidate = await _ruled([0] * 7, [3] * 7)
    (family,) = bundle.multiple_comparisons.families
    assert family.aspect == bundle.aspect, "the question was asked before the run: the family is the aspect's"
    (passed,) = [comparison for comparison in family.comparisons if comparison.reading == "trial"]
    assert passed.name == "passed"
    # The case is the unit: seven cases a side, each the share of its three trials that passed.
    assert (passed.control.n_cases, passed.contrast.n_cases) == (7, 7)
    assert (passed.control.mean, passed.contrast.mean) == (0.0, 1.0)
    # It is one of the family's corrected comparisons, beside the label check's (as accuracy).
    tested = {
        (comparison.reading, comparison.name) for comparison in family.comparisons if comparison.p_raw is not None
    }
    assert tested == {("trial", "passed"), ("measure", "accuracy")}
    assert family.family_size == 2
    assert passed.bounded_interval is not None and passed.case_values is not None


async def test_an_aspect_with_no_margin_does_not_compare_the_overall_pass_rate() -> None:
    bundle, _, _ = await _ruled([0] * 7, [3] * 7, floors=(RateDrop(max_points=10, reading="match"),))
    assert bundle.declared is not None and bundle.declared.margin is None
    (family,) = bundle.multiple_comparisons.families
    assert not [comparison for comparison in family.comparisons if comparison.reading == "trial"]


async def test_seven_cases_with_the_bounded_interval_inside_the_margin_hold_the_rate_floor() -> None:
    # Only a sweep holds a 10-point floor on seven cases: the candidate passes every trial, the control none.
    bundle, control, candidate = await _ruled([0] * 7, [3] * 7)
    readings = floor_readings(bundle, control, candidate)
    assert [rate.name for rate in readings.rates] == ["the pass rate", "Label matched"]
    assert all(rate.interval is not None and rate.interval[0] > -0.10 for rate in readings.rates)
    assert _rate_floor(bundle, control, candidate) == (
        "held",
        "cheaper candidate's drop is shown to stay within 10 points",
    )


async def test_seven_cases_with_the_bounded_interval_too_wide_are_unclear_and_say_how_far_it_reaches() -> None:
    # One trial in 21 lost on the control: the arms all but agree, and seven cases still bound nothing near 10 points.
    bundle, control, candidate = await _ruled([3, 3, 3, 3, 3, 3, 2], [3] * 7)
    assert _rate_floor(bundle, control, candidate) == (
        "unclear",
        "cheaper candidate's drop is not shown to stay within 10 points (it could be as much as 42.9)",
    )


async def test_four_cases_are_unclear_on_the_case_floor() -> None:
    # One trial in twelve lost on the candidate: an 8.3-point drop, inside the margin on its point estimate.
    bundle, control, candidate = await _ruled([3] * 4, [3, 3, 3, 2])
    assert _rate_floor(bundle, control, candidate) == (
        "unclear",
        "cheaper candidate's drop is not shown to stay within 10 points (4 cases are too few for the interval that "
        "would show it, which is estimated only from 5 independent cases)",
    )


async def test_a_rate_both_arms_pass_on_every_trial_says_it_cannot_separate_them_under_the_floor_too() -> None:
    bundle, control, candidate = await _ruled([3] * 4, [3] * 4)
    assert _rate_floor(bundle, control, candidate) == (
        "unclear",
        "the pass rate cannot separate the arms (both 12 of 12)",
    )


async def test_a_floor_reads_its_interval_at_the_nominal_level_never_the_family_s() -> None:
    """A floor is read intersection-union, each at its own nominal level: the family's Bonferroni level would make a
    warning fire less often for every reading the family adds. Its interval is computed again from the comparison's
    per-case values at 95%, so it is narrower than the family's bounded interval, never the same."""
    bundle, control, candidate = await _ruled([3, 3, 3, 3, 3, 3, 2], [3] * 7)
    (family,) = bundle.multiple_comparisons.families
    assert family.interval_level is not None and family.interval_level > INTERVAL_LEVEL
    (check,) = [comparison for comparison in family.comparisons if comparison.name == "accuracy"]
    assert check.case_values is not None and check.bounded_interval is not None
    (label,) = [rate for rate in floor_readings(bundle, control, candidate).rates if rate.name == "Label matched"]
    nominal = bounded_difference_interval(
        *check.case_values, paired=check.test == "paired", value_range=(0.0, 1.0), confidence=INTERVAL_LEVEL
    )
    assert label.interval == nominal and label.confidence == INTERVAL_LEVEL
    assert label.interval is not None and label.interval[0] > check.bounded_interval[0], "narrower: uncorrected"


async def test_a_check_floor_never_holds_on_a_t_interval_the_bounded_one_does_not_back() -> None:
    """The .29 defect: a check's rate read switch off the family's t interval, which overstates its confidence.

    Simulated at the margin, that t interval let switch through up to 12% of the time against the 2.5% it states
    (``docs/measuring-soundly.md``). Here it clears the margin while the bounded interval reaches far past it.
    """
    bundle, control, candidate = await _ruled([3, 3, 3, 3, 3, 3, 2], [3] * 7)
    (check,) = [
        comparison
        for family in bundle.multiple_comparisons.families
        for comparison in family.comparisons
        if comparison.name == "accuracy"
    ]
    assert check.basis == "t" and check.interval is not None
    assert check.interval[0] > -0.10, "the t interval clears the 10-point margin"
    readings = floor_readings(bundle, control, candidate)
    (label,) = [rate for rate in readings.rates if rate.name == "Label matched"]
    assert label.interval is not None and label.interval[0] < -0.10, "the bounded interval does not"
    alone = replace(readings, rates=(label,))
    assert _read(G2_FLOORS, alone)[0].outcome == "unclear"
    # Read on the t interval, as .29 read it, the same floor held.
    on_t = replace(readings, rates=(replace(label, interval=check.interval),))
    assert _read(G2_FLOORS, on_t)[0].outcome == "held"


async def test_a_goal_check_carried_as_a_mean_is_read_as_its_count_with_its_bounded_interval() -> None:
    """A goal check reaches the bundle as a mean of 0s and 1s, not a rate: a floor reads it as the count it is.

    The ruled campaign's label check stands in for one: its per-cell summary and its contrast, renamed to a goal
    check's measure and carried as a mean, as a goal check's are.
    """
    bundle, control, candidate = await _ruled([3, 3, 3, 3, 3, 3, 2], [3] * 7)
    check = goal_check_measure('call_count("music.queue_track") >= 2')
    cells = [
        cell.model_copy(
            update={
                "measures": cell.measures.model_copy(
                    update={
                        "measures": [
                            *cell.measures.measures,
                            *(
                                summary.model_copy(update={"name": check, "rate": None, "n_true": None})
                                for summary in cell.measures.measures
                                if summary.name == "accuracy"
                            ),
                        ]
                    }
                )
            }
        )
        for cell in bundle.cell_measures
    ]
    families = [
        family.model_copy(
            update={
                "comparisons": [
                    *family.comparisons,
                    *(c.model_copy(update={"name": check}) for c in family.comparisons if c.name == "accuracy"),
                ]
            }
        )
        for family in bundle.multiple_comparisons.families
    ]
    as_goal_check = bundle.model_copy(
        update={
            "cell_measures": cells,
            "multiple_comparisons": bundle.multiple_comparisons.model_copy(update={"families": families}),
        }
    )
    readings = floor_readings(as_goal_check, control, candidate)
    (rate,) = [rate for rate in readings.rates if rate.measure == check]
    (label,) = [rate for rate in readings.rates if rate.measure == "match"]
    assert (rate.control_passed, rate.control_n, rate.candidate_passed, rate.candidate_n) == (20, 21, 21, 21)
    assert rate.interval == label.interval, "the same per-case values, read at the same nominal level"
    # A floor names a check by its verbatim text, and so does the status it reads.
    assert rate.reading() == 'call_count("music.queue_track") >= 2'


# --- #845: only a drop shown past the margin on its whole interval reads as a regression ---------------------------

#: One rate floor, the shape Discodon's campaigns declared.
_RATE_ONLY: tuple[DecisionClause, ...] = (RateDrop(max_points=10),)


async def _rate_only_page(control_passes: Sequence[int], candidate_passes: Sequence[int]) -> Report:
    """The page of two arms passing those counts per case, with one 10-point rate floor declared before the run."""
    bundle, _, _ = await _ruled(control_passes, candidate_passes, floors=_RATE_ONLY)
    return build_code_only_report(bundle, assembled_at="2026-10-17T12:00:00+00:00", campaign_name="ruled")


def test_a_drop_shown_past_the_margin_breaks_only_if_the_lost_trials_could_not_bring_it_within() -> None:
    """Broken needs the drop past the margin with every lost trial given its best outcome, as well as on the interval.

    Control 60 of 60; the candidate passed 35 of the 40 trials it kept and lost 20. The interval puts the drop at 20
    points at least, but had every lost trial passed, the candidate would be at 55 of 60, an 8.3-point drop.
    """
    on = {"control": "v1", "candidate": "v2"}
    lost = FloorReadings(
        rates=(RateReading("the pass rate", 60, 60, 35, 40, 0, 20, interval=(-0.5, -0.2)),),
        control_cases=20,
        candidate_cases=20,
    )
    (floor,) = _read(_RATE_ONLY, lost, **on)
    assert (floor.outcome, floor.reason) == (
        "unclear",
        "v2's drop is past 10 points on the point estimate alone (the lost trials could still bring it within)",
    )
    # With fewer lost, the drop at best (33.3 points) is past the margin, and nearer zero than the interval's (40):
    # the broken floor states the smaller.
    fewer = replace(lost, rates=(RateReading("the pass rate", 60, 60, 30, 50, 0, 10, interval=(-0.6, -0.4)),))
    (broken,) = _read(_RATE_ONLY, fewer, **on)
    assert (broken.outcome, broken.reason) == (
        "broken",
        "v2 passed 30 of 50 to v1's 60 of 60, a drop that exceeds the margin of 10 points on its whole interval (at "
        "least 33.3)",
    )


def test_an_arrival_shaped_drop_past_the_margin_on_its_point_estimate_alone_reads_unclear() -> None:
    """Discodon's arrival campaign (2026-10-17): 6 of 21 trials to 12, a 28.6-point drop, its interval -92.7 to 35.5
    points. It headlined Keep, which reads as a measured regression its own contrasts called not separated."""
    readings = FloorReadings(
        rates=(RateReading("the pass rate", 12, 21, 6, 21, interval=(-0.927, 0.355)),),
        control_cases=7,
        candidate_cases=7,
    )
    (floor,) = _read(_RATE_ONLY, readings, control="September build", candidate="1017 build")
    assert floor.outcome == "unclear"
    assert floor.reason == (
        "1017 build's drop is past 10 points on the point estimate alone, not on its interval (from a gain of 35.5 to "
        "a drop of 92.7)"
    )


async def test_an_arrival_shaped_page_warns_of_nothing_and_says_why_on_the_floor_s_row() -> None:
    # Seven cases a side, 12 of 21 trials to 6: a 28.6-point drop, its bounded interval reaching well into a gain.
    report = await _rate_only_page([3, 3, 2, 2, 1, 1, 0], [1, 1, 1, 1, 1, 1, 0])
    assert _warnings(report) == [], "unclear is not broken: the page opens on no warning"
    reading, status = _floor_rows(report)[0]
    assert reading == "the pass rate"
    assert status.startswith(
        "unclear: candidate's drop is past 10 points on the point estimate alone, not on its interval (from a gain of "
    ), status
    # The page's own term of art, defined in its glossary.
    assert "point estimate" in report.terms
    assert "- **point estimate**: " in report_markdown(report)


async def test_a_drop_whose_whole_interval_is_past_the_margin_breaks_and_one_reaching_back_within_it_does_not() -> None:
    """One fixture shape, both directions: twenty cases, production passing every trial, the candidate failing some
    cases outright. Ten failed, the drop's whole interval is past the margin: broken, a line under the verdict. Eight
    failed, the drop is still 40 points but its interval reaches back within the margin: unclear, and no warning."""
    shown = await _rate_only_page([3] * 20, [3] * 10 + [0] * 10)
    aspect = shown.bundle.aspect
    (warning,) = _warnings(shown)
    assert warning == f"Adopting candidate overrides the 10-point floor it breaks on {aspect}."
    markdown = report_markdown(shown)
    assert markdown.startswith("# ") and f"\n**{warning}**\n" in markdown, "under the title, never above it"
    reading, status = _floor_rows(shown)[0]
    assert reading == "the pass rate" and status.startswith(
        "broken: candidate passed 30 of 60 to production's 60, a drop that exceeds the margin of 10 points on its "
        "whole interval (at least "
    ), status
    unshown = await _rate_only_page([3] * 20, [3] * 12 + [0] * 8)
    assert _warnings(unshown) == []
    reading, status = _floor_rows(unshown)[0]
    assert status.startswith(
        "unclear: candidate's drop is past 10 points on the point estimate alone, not on its interval (from a drop of "
    ), status
    assert "exceeds" not in report_markdown(unshown)
    assert "point estimate" in unshown.terms and "point estimate" not in shown.terms


def on_time(case: Mapping[str, Any], answer: Any) -> bool:
    """Whether the reply came in on time: never, on either arm."""
    return False


async def test_a_readiness_shaped_reading_both_arms_never_pass_says_it_cannot_separate_them() -> None:
    """Discodon's readiness campaign (2026-10-17): its rule's one clause read every pass/fail rate (``RateDrop`` with
    no reading). The pass rate was 0 of 18 on both arms, and the clause's one outcome, a check's 12 of 18 to 14, sat
    on the pass rate's row. Each reading now has its own status, about that reading alone."""
    cases = [{"text": f"queue request {index}", "label": "queued"} for index in range(6)]
    arms = {"production": _passing([3, 3, 3, 3, 2, 0], 0.0031), "candidate": _passing([3, 3, 2, 2, 2, 0], 0.0021)}
    comparison = await Eval(cases, arms, ["label", on_time], k=3, name="readiness").run(store=":memory:")
    bundle = await _assembled(comparison, _RATE_ONLY)
    report = build_code_only_report(bundle, assembled_at="2026-10-17T12:00:00+00:00", campaign_name="readiness")
    rows = dict(_floor_rows(report))
    assert list(rows) == ["the pass rate", "Label matched", "On time score"]
    assert rows["the pass rate"] == "unclear: the pass rate cannot separate the arms (both 0 of 18)"
    assert rows["On time score"] == "unclear: On time score cannot separate the arms (both 0 of 18)"
    # 12 of 18 to 14 is an 11.1-point drop: past the margin on its point estimate, on its own row.
    assert rows["Label matched"].startswith(
        "unclear: candidate's drop on Label matched is past 10 points on the point estimate alone, not on its "
        "interval (from a gain of "
    )
    for reading, outcome in rows.items():
        others = [other for other in rows if other != reading]
        assert not [other for other in others if other in outcome], (reading, outcome)
    assert _warnings(report) == []
    # A floor naming the degenerate reading reads it alone, and says why it is unclear.
    control = bundle.design.control_arm.variant_key if bundle.design.control_arm is not None else ""
    (candidate,) = [board.variant_key for board in bundle.scoreboard if board.variant_key != control]
    readings = floor_readings(bundle, control, candidate)
    (floor,) = _read((RateDrop(max_points=10, reading="on_time"),), readings)
    assert (floor.outcome, floor.reason) == ("unclear", "On time score cannot separate the arms (both 0 of 18)")
    assert [read.rate.measure for read in floor.rates] == ["on_time"]


def test_every_reason_names_the_arms_by_their_names() -> None:
    cheaper = FloorReadings(
        rates=(RateReading("the pass rate", 50, 60, 50, 60, interval=(-0.05, 0.05)),),
        control_cost=0.0031,
        candidate_cost=0.0021,
        control_cases=20,
        candidate_cases=20,
        cost_interval=(-0.0014, -0.0006),
        cost_compared=True,
    )
    dearer = replace(cheaper, candidate_cost=0.0041, cost_interval=(0.0006, 0.0014))
    level = replace(cheaper, candidate_cost=0.0031, cost_interval=(-0.0004, 0.0004))
    on = {"control": "v1", "candidate": "v2"}
    assert _read(G2_FLOORS, cheaper, **on)[1].reason == "v2 costs less per reply than v1"
    assert _read(G2_FLOORS, dearer, **on)[1].reason == "v2 costs more per reply than v1 on the whole interval"
    assert _read(G2_FLOORS, level, **on)[1].reason == "the cost difference did not separate"
    statements = [floor.statement for floor in _read(G2_FLOORS, cheaper, **on)]
    assert statements == ["no pass rate drops by more than 10 points", "v2 costs less per reply"]
    assert floor_statement(RateDrop(max_points=2.5, reading="no_invented_total"), candidate="v2") == (
        "no no invented total rate drops by more than 2.5 points"
    )


async def test_a_floor_naming_one_reading_reads_that_rate_alone() -> None:
    # The overall pass rate drops 50 points; the label check, which the floor names, does too, on an interval wholly
    # past the margin over twenty cases: broken on it alone.
    bundle, control, candidate = await _ruled([3] * 20, [3] * 10 + [0] * 10)
    scoped = (RateDrop(max_points=10, reading="match"),)
    readings = floor_readings(bundle, control, candidate)
    (floor,) = _read(scoped, readings)
    assert floor.outcome == "broken"
    assert floor.rate is not None and floor.rate.measure == "match"
    assert [read.rate.measure for read in floor.rates] == ["match"]
    assert floor.statement == "no Label matched rate drops by more than 10 points"
    # A name the arms do not carry is said, and left unclear, never read as every rate.
    (absent,) = _read((RateDrop(max_points=10, reading="tone"),), readings)
    assert (absent.outcome, absent.reason) == ("unclear", "tone was not measured on both arms")


async def test_a_question_that_declares_no_floor_says_so_in_one_line() -> None:
    bundle, _, _ = await _ruled([3] * 7, [2] * 7, floors=())
    report = build_code_only_report(bundle, assembled_at="2026-10-14T00:00:00+00:00", campaign_name="no floors")
    decision = [block for block in report.blocks if block.section == "decision"]
    assert [(type(block).__name__, getattr(block, "text", None)) for block in decision] == [
        ("DisclosureBlock", f"No floor was declared on {bundle.aspect}.")
    ]
    assert _warnings(report) == []
    assert f"## Floors\n\nNo floor was declared on {bundle.aspect}.\n" in report_markdown(report)


async def test_an_exploratory_comparison_has_no_floor_block_at_all() -> None:
    """A quick comparison asks no question, so it can declare no floor: the page says nothing about floors, as it did
    before floors existed."""
    cases = [{"text": f"queue request {index}", "label": "queued"} for index in range(7)]
    arms = {"production": _passing([3] * 7, 0.0031), "candidate": _passing([2] * 7, 0.0021)}
    comparison = await Eval(cases, arms, ["label"], k=3, name="exploratory").run(store=":memory:")
    report = comparison.report
    assert report.bundle.declared is not None and not report.bundle.declared.question
    assert not [block for block in report.blocks if block.section == "decision"]
    assert "## Floors" not in report_markdown(report) and "No floor was declared" not in report_markdown(report)


# --- the Decision section speaks the reader's words, never the engine's ---------------------------------------------

#: The engine's names for a floor's parts, which the Decision section never prints.
_SYSTEM_NAMES = re.compile(r"rate_drop|cost_lower|max_points|\bclauses?\b|\bkind\b")

#: The roles of a comparison's arms, which the Decision section never names an arm by.
_ROLE_WORDS = re.compile(r"\b(candidate|control)\b", re.I)


def _decision_section(page: str, *, html_page: bool) -> str:
    """The Decision section's visible text, from its heading to the next section's."""
    if html_page:
        body = page[page.index('<section id="decision">') :]
        return html.unescape(re.sub(r"<[^>]+>", " ", body[: body.index("</section>")]))
    body = page.split("\n## Floors\n", 1)[1]
    return body.split("\n## ", 1)[0]


async def test_the_decision_section_names_no_system_word_and_no_arm_by_its_role() -> None:
    _, one = await g2_ruled_report()
    bundle, _, _ = await _ruled([3] * 7, [2] * 7)
    two = build_code_only_report(bundle, assembled_at="2026-10-14T12:00:00+00:00", campaign_name="ruled")
    for report, labels in ((one, ("cheaper candidate", "production")), (two, ("candidate", "production"))):
        for door, page in (("markdown", report_markdown(report)), ("html", report_html(report))):
            section = _decision_section(page, html_page=door == "html")
            # No rule is stated anywhere: a floor warns and decides nothing.
            assert "Switch to " not in page, door
            assert not _SYSTEM_NAMES.search(section), (door, _SYSTEM_NAMES.findall(section))
            # An arm's own name may hold a role word (an arm a host called "cheaper candidate"); nothing else may.
            for label in labels:
                section = re.sub(rf"(?<![\w-]){re.escape(label)}(?![\w-])", "ARM", section)
            assert not _ROLE_WORDS.search(section), (door, _ROLE_WORDS.findall(section), section[:400])


# --- a floor's clause is typed: built, stored and refused as data, never parsed from words ---


def test_a_clause_needs_a_margin_inside_the_scale() -> None:
    for margin in (0.0, -1.0, 100.0, float("nan")):
        with pytest.raises(ValidationError, match="max_points"):
            RateDrop(max_points=margin)
    with pytest.raises(ValidationError, match="reading"):
        RateDrop(max_points=2, reading="")


def test_a_floor_reads_back_from_its_stored_form_clause_by_clause() -> None:
    floors = [
        Floor(aspect="tone", clause=clause)
        for clause in (RateDrop(max_points=2, reading="tone"), CostLower(), RateDrop(max_points=5))
    ]
    stored = [floor.model_dump(mode="json") for floor in floors]
    assert [Floor.model_validate(floor) for floor in stored] == floors
    assert [type(Floor.model_validate(floor).clause) for floor in stored] == [RateDrop, CostLower, RateDrop]


#: The toy host's template's one goal-state check, verbatim.
_TOY_CHECK = 'call_count("extractor.emit_field") == 4'


def _declare(*readings: str | None) -> None:
    """Declare one aspect with a rate floor per reading, through the declaration gate."""
    floors = [Floor(aspect="scoped", clause=RateDrop(max_points=10, reading=reading)) for reading in readings]
    refuse_an_undeclarable_design(
        minimal_aspect_design(), floors=floors, behavior="b", template=toyhost_template(), profile=toyhost_profile()
    )


def test_a_reading_the_campaign_does_not_measure_is_refused_naming_what_it_does() -> None:
    with pytest.raises(DesignRefused, match="floor scoped/rate_drop/tone names 'tone'") as refused:
        _declare("tone")
    # The remedy lists what the template and the host offer: the check verbatim and the host's pass/fail measures.
    assert repr(_TOY_CHECK) in refused.value.remedy and "'match'" in refused.value.remedy
    assert decision_readings(toyhost_template(), toyhost_profile()) == [_TOY_CHECK, "goal_state", "match"]


def test_a_reading_the_campaign_measures_or_none_is_declared() -> None:
    _declare(_TOY_CHECK, "match", None)


async def test_a_pass_fail_judged_dimension_is_read_as_its_count() -> None:
    """A judged dimension on the pass/fail scale is a pass rate a floor names: its mean times its scores, per arm."""
    bundle, control, candidate = await _ruled([3] * 7, [3] * 7)

    def arm(variant_key: str, passed: int) -> JudgedArm:
        return JudgedArm(
            variant_key=variant_key,
            apparatus_class_id="rig",
            run_ids=["run"],
            n=21,
            n_independent=7,
            mean=passed / 21,
            evidence_tier="undetermined",
        )

    def judged(name: str, axis: Literal["capability", "boundary"]) -> JudgedMeasure:
        arms = [arm(control, 20), arm(candidate, 12)]
        return JudgedMeasure(name=name, family="rubric", value_range=(0.0, 1.0), scale="ratio", axis=axis, arms=arms)

    scored = bundle.model_copy(update={"judged_measures": [judged("tone", "capability"), judged("safe", "boundary")]})
    readings = floor_readings(scored, control, candidate)
    (tone,) = [rate for rate in readings.rates if rate.measure == "tone"]
    assert (tone.control_passed, tone.control_n, tone.candidate_passed, tone.candidate_n) == (20, 21, 12, 21)
    assert not [rate for rate in readings.rates if rate.measure == "safe"], "a guardrail is never a floor's reading"
    (floor,) = _read((RateDrop(max_points=10, reading="tone"),), readings)
    assert floor.statement == "no tone rate drops by more than 10 points"
    # A 38-point drop with no contrast on the dimension is past the margin on its point estimate alone (#845: it read
    # keep until then), and the floor reads that rate and no other.
    assert (floor.outcome, floor.reason) == (
        "unclear",
        "cheaper candidate's drop is past 10 points on the point estimate alone (no interval was computed on tone)",
    )
    assert [read.rate for read in floor.rates] == [tone]


# --- a floor reads every check, each on its better end, as the declaration offers it ------------------------------


def _hallucinated(bundle: AnalysisContextBundle) -> AnalysisContextBundle:
    """``bundle`` with a lower-is-better boolean ``hallucinated``, true on every trial whose label missed.

    Its per-cell summaries are the label check's, counted the other way up, and its contrast is ``accuracy``'s on
    the measure's own scale (the arm minus the control, so a fall in hallucinations is a negative difference), its
    per-case values each ``1 - v``.
    """
    name = "hallucinated"
    descriptor = MetricDescriptor(
        name=name,
        data_type="boolean",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description="The answer asserted something the case does not hold.",
        higher_is_better=False,
        merit_axis="quality",
    )

    def flipped(summary: Any) -> Any:
        n_true = summary.n - round(summary.mean * summary.n)
        return summary.model_copy(update={"name": name, "rate": n_true / summary.n, "n_true": n_true, "mean": None})

    cells = [
        cell.model_copy(
            update={
                "measures": cell.measures.model_copy(
                    update={
                        "measures": [
                            *cell.measures.measures,
                            *(flipped(s) for s in cell.measures.measures if s.name == ACCURACY_MEASURE),
                        ]
                    }
                )
            }
        )
        for cell in bundle.cell_measures
    ]

    def negated(interval: tuple[float, float] | None) -> tuple[float, float] | None:
        return None if interval is None else (-interval[1], -interval[0])

    def flipped_values(values: Any) -> Any:
        return None if values is None else tuple([1.0 - v for v in side] for side in values)

    families = [
        family.model_copy(
            update={
                "comparisons": [
                    *family.comparisons,
                    *(
                        c.model_copy(
                            update={
                                "name": name,
                                "higher_is_better": False,
                                "interval": negated(c.interval),
                                "bounded_interval": negated(c.bounded_interval),
                                "case_values": flipped_values(c.case_values),
                            }
                        )
                        for c in family.comparisons
                        if c.name == ACCURACY_MEASURE
                    ),
                ]
            }
        )
        for family in bundle.multiple_comparisons.families
    ]
    return bundle.model_copy(
        update={
            "cell_measures": cells,
            "measure_catalog": {**bundle.measure_catalog, name: descriptor},
            "multiple_comparisons": bundle.multiple_comparisons.model_copy(update={"families": families}),
        }
    )


async def test_a_lower_is_better_check_reads_its_fall_as_a_gain_and_holds() -> None:
    """``hallucinated`` true on 20 of 60 trials for the control and 2 of 60 for v2 is a 30-point gain, not a drop."""
    bundle, control, candidate = await _ruled([2] * 20, [3] * 18 + [2] * 2)
    readings = floor_readings(_hallucinated(bundle), control, candidate)
    (fewer,) = [rate for rate in readings.rates if rate.measure == "hallucinated"]
    assert (fewer.control_passed, fewer.control_n, fewer.candidate_passed, fewer.candidate_n) == (40, 60, 58, 60)
    (floor,) = _read((RateDrop(max_points=10),), readings)
    assert floor.outcome == "held", floor.reason


def exact_match(case: Mapping[str, Any], answer: Any) -> bool:
    """The answer is the case's label."""
    return bool(answer == case["label"])


async def test_a_pass_fail_scorer_is_a_reading_a_floor_declares_and_reads() -> None:
    cases = [{"text": f"queue request {index}", "label": "queued"} for index in range(7)]
    arms = {"production": _passing([3] * 7, 0.0031), "candidate": _passing([2] * 7, 0.0021)}
    comparison = await Eval(cases, arms, [exact_match], k=3, name="scored").run(store=":memory:")
    profile = comparison.host.profile
    named = (RateDrop(max_points=10, reading="exact_match"),)
    refuse_an_undeclarable_design(
        minimal_aspect_design(),
        floors=[Floor(aspect="scored", clause=clause) for clause in named],
        behavior="b",
        template=None,
        profile=profile,
    )

    bundle = await _assembled(comparison, G2_FLOORS)
    assert bundle.design.control_arm is not None
    control = bundle.design.control_arm.variant_key
    (candidate,) = [board.variant_key for board in bundle.scoreboard if board.variant_key != control]
    readings = floor_readings(bundle, control, candidate)
    (scored,) = [rate for rate in readings.rates if rate.measure == "exact_match"]
    assert (scored.control_passed, scored.control_n, scored.candidate_passed, scored.candidate_n) == (21, 21, 14, 21)
    (alone,) = _read(named, readings)
    # A 33-point drop on seven cases is past the margin on its point estimate, not on its interval: unclear (#845;
    # it read keep until then), on the scorer's rate alone.
    assert alone.outcome == "unclear" and [read.rate for read in alone.rates] == [scored]
    (every,) = _read((RateDrop(max_points=10),), readings)
    assert isinstance(every.floor.clause, RateDrop) and scored in readings.read_by(every.floor.clause)
    # Every pass/fail rate, the overall pass rate and the scorer's, is past the margin on its point estimate alone.
    assert every.outcome == "unclear"
    assert [(read.rate, read.outcome) for read in every.rates] == [
        (readings.rates[0], "unclear"),
        (scored, "unclear"),
    ]
    assert all("on the point estimate alone, not on its interval" in read.reason for read in every.rates)
    # The declaration offers exactly the checks the page reads, wherever the run measured them.
    offered = [name for name in decision_readings(None, profile) if name in bundle.measure_catalog]
    assert offered == [rate.measure for rate in readings.rates if rate.measure is not None] == ["exact_match"]
