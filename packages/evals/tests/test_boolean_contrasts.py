"""A directed boolean measure on a merit axis is contrasted between arms on its rate per case.

A host's boolean measure (``top_up_never``: lower is better, on reliability) summarises as a rate in each cell, the
share of its observations held True. The family of comparisons tests it on that rate read per case, the share of a
case's repeats held True, paired by case, as the overall pass rate is tested on a case's share of trials passed: the
value tested and the cell's ``rate`` are one computation, the delta is in points of a share, and the correction
counts it. An undirected boolean, and a categorical or text measure, have no better end or no order, so no family
tests them.

Mutations that turn this file red (each run against a saved copy and restored from it):

- ``_per_case_values``: dropping the boolean rates (the old behaviour), so the boolean has no per-case value and no
  contrast, and the family is one smaller.
- ``_measure_range``: reading a boolean's declared range (none) instead of a share's, so the comparison carries no
  bounded interval or case values.
- ``_guardrail_readings``: leaving booleans out of the guardrails, so a boolean declared one is never checked.
- ``_is_rate``: dropping the check branch (``is_check``), so the directed boolean is drawn on its t interval.

A directed boolean declared a guardrail is held the same way, on its rate per case: its check reads each case's share
held True, on the share's 0 to 1 range, and is held or breached against its declared margin.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from threetears.evals.analysis import AnalysisContextBundle, ComparisonFamily, assemble_context_bundle
from threetears.evals.analysis.bundle.comparisons import planning_readings
from threetears.evals.analysis.stats import guardrail_decision
from threetears.evals.kernel import CampaignArm
from threetears.evals.kernel.host import HostProfile, MeasureRegistry
from threetears.evals.kernel.metrics import MetricDescriptor
from threetears.evals.schema import EvalResult
from packages.evals.tests.factories import (
    fixture_variant_key,
    make_aspect_campaign,
    make_eval_result,
    make_eval_run,
    minimal_aspect_design,
)
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES, toyhost_profile

CONTROL = "control-model"
CONTRAST = "contrast-model"
K = 3

#: Each case's repeats that never topped up, of ``K``: the control rarely, the contrast almost always.
_CONTROL_TRUE = (0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0)
_CONTRAST_TRUE = (3, 2, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3)


def _boolean(name: str, **declared: object) -> MetricDescriptor:
    return MetricDescriptor(
        name=name,
        reader_name=name.replace("_", " ").capitalize(),
        data_type="boolean",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description=f"Whether {name} held on one trial.",
        **declared,  # type: ignore[arg-type]
    )


#: The directed boolean under test, and three measures no family may test.
_MEASURES = (
    _boolean("top_up_never", higher_is_better=False, merit_axis="reliability"),
    _boolean("top_up_seen"),
    MetricDescriptor(
        name="queue_choice",
        reader_name="Queue choice",
        data_type="categorical",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description="Which queue the reply chose.",
        higher_is_better=True,
        merit_axis="reliability",
    ),
    MetricDescriptor(
        name="queue_note",
        reader_name="Queue note",
        data_type="text",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description="What the reply said of the queue.",
        higher_is_better=True,
        merit_axis="reliability",
    ),
)


def _profile() -> HostProfile:
    profile = toyhost_profile()
    return replace(
        profile, measures=MeasureRegistry((*TOYHOST_MEASURES, *_MEASURES), families=profile.measures.families)
    )


def _evidence() -> tuple[list, dict[str, list[EvalResult]]]:
    """A control and a contrast over twelve cases at ``K`` repeats each, every result carrying the four measures."""
    runs = []
    results: dict[str, list[EvalResult]] = {}
    for model, trues in ((CONTROL, _CONTROL_TRUE), (CONTRAST, _CONTRAST_TRUE)):
        run = make_eval_run(status="completed", candidate_model=model)
        runs.append(run)
        results[run.id] = [
            make_eval_result(
                id=f"{model}-{case}-{k}",
                eval_run_id=run.id,
                scope_id=run.scope_id,
                model=model,
                test_case_id=f"tc-{case:02d}",
                k_iteration=k,
                goal_state_outcomes=[],
                rubric_scores=[],
                cost_usd=None,
                host_measures={
                    "top_up_never": k <= trues[case],
                    "top_up_seen": k % 2 == 0,
                    "queue_choice": "left" if (model == CONTRAST) == (k <= trues[case]) else "right",
                    "queue_note": f"case {case} repeat {k}",
                },
            )
            for case in range(len(trues))
            for k in range(1, K + 1)
        ]
    return runs, results


def _bundle() -> AnalysisContextBundle:
    runs, results = _evidence()
    asked = min(run.created_at for run in runs)
    campaign = make_aspect_campaign(
        label="arrival",
        run_ids=[run.id for run in runs],
        design=minimal_aspect_design(merit_axes=["reliability"]),
        question="does the contrast top up as often?",
        asked_at=asked,
        control=CampaignArm(label="control", role="control", variant_key=fixture_variant_key(CONTROL), named_at=asked),
        scope_id=runs[0].scope_id,
        name="boolean contrasts",
        subject_id=runs[0].subject_snapshot.subject_id,
        subject_kind="s",
    )
    return assemble_context_bundle(campaign, storage=ToyhostStorage(runs, results), profile=_profile())


def _family(bundle: AnalysisContextBundle) -> ComparisonFamily:
    (family,) = bundle.multiple_comparisons.families
    return family


@pytest.fixture(scope="module")
def bundle() -> AnalysisContextBundle:
    return _bundle()


def test_a_directed_boolean_is_contrasted_on_its_rate_per_case_paired_by_case(bundle: AnalysisContextBundle) -> None:
    family = _family(bundle)
    (comparison,) = [c for c in family.comparisons if c.name == "top_up_never"]

    assert comparison.reading == "measure" and comparison.test == "paired"
    assert comparison.case_values == (
        [trues / K for trues in _CONTROL_TRUE],
        [trues / K for trues in _CONTRAST_TRUE],
    ), "each case's value is the share of its repeats held True, aligned case by case"
    control_rate = sum(_CONTROL_TRUE) / (K * len(_CONTROL_TRUE))
    contrast_rate = sum(_CONTRAST_TRUE) / (K * len(_CONTRAST_TRUE))
    assert comparison.control.n_cases == comparison.contrast.n_cases == len(_CONTROL_TRUE)
    # The delta is in points of a share, on 0 to 1, and the bounded interval reads the share's range.
    assert comparison.delta == pytest.approx(contrast_rate - control_rate)
    assert comparison.bounded_interval is not None
    assert -1.0 <= comparison.bounded_interval[0] <= comparison.bounded_interval[1] <= 1.0
    # Lower is better: the contrast never tops up far more often, so it regressed.
    assert comparison.higher_is_better is False
    assert comparison.verdict == "regressed"


def test_a_directed_boolean_is_drawn_on_its_bounded_interval(bundle: AnalysisContextBundle) -> None:
    (comparison,) = [c for c in _family(bundle).comparisons if c.name == "top_up_never"]

    assert comparison.basis == "t", "its p is the t test's"
    assert comparison.interval is not None and comparison.interval == comparison.bounded_interval
    assert comparison.interval_basis == "bounded"


def test_the_cell_s_rate_and_the_contrast_s_arm_values_agree(bundle: AnalysisContextBundle) -> None:
    (comparison,) = [c for c in _family(bundle).comparisons if c.name == "top_up_never"]
    rates = {
        cell.variant_key: summary.rate
        for cell in bundle.cell_measures
        for summary in cell.measures.measures
        if summary.name == "top_up_never"
    }
    # Every case ran K times, so the mean of the per-case rates is the cell's rate over every observation.
    assert comparison.control.mean == pytest.approx(rates[fixture_variant_key(CONTROL)])
    assert comparison.contrast.mean == pytest.approx(rates[fixture_variant_key(CONTRAST)])


def test_an_undirected_boolean_and_a_categorical_or_text_measure_get_no_contrast(
    bundle: AnalysisContextBundle,
) -> None:
    family = _family(bundle)
    names = {comparison.name for comparison in family.comparisons}

    assert names == {"top_up_never"}, "no better end, or no order: nothing to improve or regress on"
    assert family.family_size == 1, "the boolean's contrast is the family's one tested comparison"


def test_the_report_states_the_boolean_s_contrast_in_points_of_a_share(bundle: AnalysisContextBundle) -> None:
    from threetears.evals.analysis.report.build import build_code_only_report

    report = build_code_only_report(bundle, assembled_at="2026-10-31T00:00:00Z")
    (table,) = [block for block in report.blocks if getattr(block, "name", None) == "comparisons"]
    (row,) = [row for row in table.values() if row["reading"] == "Top up never"]  # type: ignore[attr-defined]
    (folded,) = [block for block in report.blocks if getattr(block, "name", None) == "comparison_statistics"]
    (stats,) = [row for row in folded.values() if row["reading"] == "Top up never"]  # type: ignore[attr-defined]

    assert isinstance(row["interval"], str) and row["interval"].endswith(" pts")
    assert stats["control_mean"] == pytest.approx(100 * sum(_CONTROL_TRUE) / (K * len(_CONTROL_TRUE)))


def test_the_planning_read_plans_the_boolean_on_a_share_s_range() -> None:
    runs, results = _evidence()
    readings = {
        reading.name: reading
        for reading in planning_readings(runs, [r for rows in results.values() for r in rows], profile=_profile())
    }

    assert set(readings) == {"top_up_never"}
    planned = readings["top_up_never"]
    assert (planned.higher_is_better, planned.value_range) == (False, (0.0, 1.0))
    control_run = runs[0].id
    assert planned.repeats[control_run]["tc-01"] == [1.0, 0.0, 0.0], "one repeat's rate is its own 1 or 0"


# --- a directed boolean declared a guardrail ------------------------------------------------------------------------

#: The guardrail's margin: how far its rate may fall, in points of a share, before it is breached.
_GUARDRAIL_MARGIN = 0.25

#: Each case's repeats that kept quiet, of ``K``: the control and an arm that matches it, then an arm far below it.
_QUIET_CONTROL = (3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3) * 2
_QUIET_HELD = (3, 3, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3) * 2
_QUIET_BREACHED = (1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1) * 2


def _guardrail_profile() -> HostProfile:
    profile = toyhost_profile()
    quiet = _boolean("kept_quiet", higher_is_better=True, guardrail=True, materiality_threshold=_GUARDRAIL_MARGIN)
    return replace(profile, measures=MeasureRegistry((*TOYHOST_MEASURES, quiet), families=profile.measures.families))


def _guardrail_bundle(contrast_quiet: tuple[int, ...]) -> AnalysisContextBundle:
    """The control and one arm over ``len(_QUIET_CONTROL)`` cases at ``K`` repeats, every result carrying the
    guardrail, the arm keeping quiet on each case as often as ``contrast_quiet`` says."""
    runs = []
    results: dict[str, list[EvalResult]] = {}
    for model, quiet in ((CONTROL, _QUIET_CONTROL), (CONTRAST, contrast_quiet)):
        run = make_eval_run(status="completed", candidate_model=model)
        runs.append(run)
        results[run.id] = [
            make_eval_result(
                id=f"{model}-{case}-{k}",
                eval_run_id=run.id,
                scope_id=run.scope_id,
                model=model,
                test_case_id=f"tc-{case:02d}",
                k_iteration=k,
                goal_state_outcomes=[],
                rubric_scores=[],
                cost_usd=None,
                host_measures={"kept_quiet": k <= quiet[case]},
            )
            for case in range(len(quiet))
            for k in range(1, K + 1)
        ]
    asked = min(run.created_at for run in runs)
    campaign = make_aspect_campaign(
        label="arrival",
        run_ids=[run.id for run in runs],
        design=minimal_aspect_design(merit_axes=["reliability"]),
        question="does the contrast keep quiet as often?",
        asked_at=asked,
        control=CampaignArm(label="control", role="control", variant_key=fixture_variant_key(CONTROL), named_at=asked),
        scope_id=runs[0].scope_id,
        name="boolean guardrail",
        subject_id=runs[0].subject_snapshot.subject_id,
        subject_kind="s",
    )
    return assemble_context_bundle(campaign, storage=ToyhostStorage(runs, results), profile=_guardrail_profile())


@pytest.fixture(scope="module")
def guardrail_held() -> AnalysisContextBundle:
    return _guardrail_bundle(_QUIET_HELD)


@pytest.fixture(scope="module")
def guardrail_breached() -> AnalysisContextBundle:
    return _guardrail_bundle(_QUIET_BREACHED)


def test_a_boolean_guardrail_is_checked_on_its_rate_per_case_on_a_share_s_range(
    guardrail_held: AnalysisContextBundle,
) -> None:
    assert guardrail_held.guardrails.measures == ["kept_quiet"]
    (check,) = guardrail_held.guardrails.checks

    assert (check.reading, check.name, check.higher_is_better) == ("measure", "kept_quiet", True)
    assert check.test == "paired"
    # Its per-case samples are the shares of each case's repeats that kept quiet: their means are the shares' means.
    assert check.control.n_cases == check.contrast.n_cases == len(_QUIET_CONTROL)
    assert check.control.mean == pytest.approx(sum(q / K for q in _QUIET_CONTROL) / len(_QUIET_CONTROL))
    assert check.contrast.mean == pytest.approx(sum(q / K for q in _QUIET_HELD) / len(_QUIET_HELD))
    expected = guardrail_decision(
        [q / K for q in _QUIET_CONTROL],
        [q / K for q in _QUIET_HELD],
        paired=True,
        margin=_GUARDRAIL_MARGIN,
        higher_is_better=True,
        value_range=(0.0, 1.0),
    )
    # Its interval is the bounded test's on the shares' 0 to 1 range, the boolean declaring no range of its own.
    assert check.interval_basis == "bounded"
    assert check.interval == expected.interval
    assert check.interval is not None and -1.0 <= check.interval[0] <= check.interval[1] <= 1.0
    assert (check.margin, check.margin_declared) == (_GUARDRAIL_MARGIN, True)


def test_a_boolean_guardrail_within_its_margin_is_held(guardrail_held: AnalysisContextBundle) -> None:
    (check,) = guardrail_held.guardrails.checks

    assert check.interval is not None and check.interval[0] > -_GUARDRAIL_MARGIN
    assert check.decision == "held"


def test_a_boolean_guardrail_past_its_margin_is_breached(guardrail_breached: AnalysisContextBundle) -> None:
    (check,) = guardrail_breached.guardrails.checks

    assert check.interval is not None and check.interval[1] < -_GUARDRAIL_MARGIN
    assert check.decision == "breached"


def test_a_boolean_guardrail_joins_no_family(guardrail_held: AnalysisContextBundle) -> None:
    assert all(
        comparison.name != "kept_quiet"
        for family in guardrail_held.multiple_comparisons.families
        for comparison in family.comparisons
    )
