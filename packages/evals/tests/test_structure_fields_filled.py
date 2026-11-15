"""The structure fields a report input carries for the written page, each copied off the bundle, never recomputed.

What the code page states and the written page could not, because the input did not carry it (#1126):

- **A contrast's paired baseline** (``Contrast.baseline``, ``baseline_n_cases``): the control's mean over the cases
  the test read, which its delta is a change from (#1125); the family it was corrected in (``family``, an id
  ``contrast_families`` resolves); and, where the delta is not the difference of the two arms' stated figures, why
  (``delta_note``, #1142 U2).
- **A figure's spread** (``Stat.spread``) and, where every case read one value, that value (``no_variation``,
  #1129), computed where the bundle summarises the cases; and a figure with no interval says why, as a state
  (``Stat.state``: not available, ``too_few`` or ``no_variation``).
- **A reading's flip margin** and whether its aspect's margin is informative (``Reading.flip_margin``,
  ``Aspect.margin_informative``, #1142 U6), from the one arithmetic in ``report_margins``.
- **An arm's spelling** (``Arm.spelling``): the levers code spelled its name from, so a page reads a cut level by its
  lever with no text read back (#1141). ``test_host_surfaces_listing.py`` holds what a page prints from it.
- **The figures the code page reads off the bundle itself**: each stratum's measure figures beside its judged ones
  and a measure figure's standard error (``Stratum.per_arm``, ``Stat.sem``), a judged mean's interval
  (``Judged.per_arm[...].ci``, by the one rule every page reads it with), each arm's pass^k and the cells kept out of
  it (``PassK.per_arm``, ``at_common_depth``, ``cells``), each arm's figures across the aspects (``across_aspects``),
  the arm's side of a contrast and the margin its delta was read against (``Contrast.arm_mean``, ``margin``, decided
  on the numbers side by ``contrast_margin``), and the campaign's declared volume (``Meta.volume_per_month``).

Every one is optional and left out where unset, so a stored input digests as it did
(``test_shape_move_moves_no_digest.py``), and each frozen input still verifies its digest (the last test here).
"""

from __future__ import annotations

import gzip
import json
import math
import statistics
from collections.abc import Iterator, Mapping
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.factories import fixture_variant_key, make_campaign, make_eval_result, make_eval_run
from packages.evals.tests.test_hot_path_oracle import WITH_BUNDLES, Goldens, subject
from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle
from threetears.evals.analysis.bundle.comparisons import contrast_margin
from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.references import resolve_reading
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.report_input import Reading, ReportInput, report_input_document_digest
from threetears.evals.analysis.report_input_build import arm_spellings, family_id, report_input
from threetears.evals.analysis.report_margins import flip_margin, margin_informative
from threetears.evals.analysis.service import assemble_campaign_bundle
from threetears.evals.kernel import EvalStorage
from threetears.evals.kernel.host import SHARED_CORE, HostProfile, MeasureRegistry
from threetears.evals.kernel.metrics import MetricDescriptor
from threetears.evals.schema import EvalResult, GoalStateOutcome, LatencyMetrics
from threetears.evals.schema.hashing import identity_json, round_figures
from threetears.evals.storage import InMemoryDocumentStore

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@cache
def _built(name: str) -> tuple[CampaignBundle, ReportInput]:
    read = subject(name)
    bundle = assemble_campaign_bundle(read.host, read.campaign_id, read.scope_id)
    return bundle, report_input(bundle)


# --- a contrast's paired baseline, its family and its note ---------------------------------------------------------


#: The hot-path campaigns that hold contrasts against a control: the toy host's and the courier's are choices.
CONTRASTED = ("scaled", "scaled_two_aspects")


@pytest.mark.parametrize("name", CONTRASTED)
def test_every_contrast_carries_its_paired_baseline_and_the_family_it_was_corrected_in(name: str) -> None:
    bundle, built = _built(name)
    families = {
        family_id(aspect.aspect, position): (aspect.aspect, family)
        for aspect in bundle.aspects
        for position, family in enumerate(aspect.multiple_comparisons.families)
    }
    listed = [(family.aspect, family.family_size) for family in built.contrast_families]
    assert [(aspect, family.family_size) for aspect, family in families.values()] == listed, (
        "a family's id is its place among its aspect's families, as contrast_families lists them"
    )
    assert built.contrasts, f"{name} holds no contrast, so this checks nothing"
    for contrast in built.contrasts:
        assert contrast.family in families, contrast.family
        aspect, family = families[contrast.family]
        (comparison,) = [
            held
            for held in family.comparisons
            if (held.name, held.reading) == (contrast.measure, contrast.kind)
            and bundle.label_of(held.contrast.variant_key) == contrast.arm
            and bundle.label_of(held.control.variant_key) == contrast.control
        ]
        assert contrast.aspect == aspect
        assert contrast.margin == contrast_margin(bundle.aspect(aspect), comparison)
        assert contrast.arm_mean == comparison.contrast.mean
        assert contrast.baseline == comparison.control.mean
        assert contrast.baseline_n_cases == (
            comparison.control.n_cases if comparison.control.mean is not None else None
        )
        if contrast.baseline is not None and contrast.delta is not None:
            assert math.isclose(contrast.baseline + contrast.delta, comparison.contrast.mean or 0.0, rel_tol=1e-12)


def _stated(built: ReportInput, kind: str | None, measure: str, aspect: str | None, arm: str) -> float | None:
    """An arm's figure as the input states it beside a contrast: its measure's or dimension's mean on the aspect, or
    its pass rate over its usable trials."""
    if kind == "trial" and measure == TRIAL_PASS_NAME:
        # The lead aspect's scoreboard is the input's own; every other aspect carries its own.
        (held,) = [held for held in built.aspects if held.id == aspect]
        row = (held.scoreboard if held.scoreboard is not None else built.scoreboard).get(arm)
        return row.passed / row.usable if row is not None and row.usable else None
    figures = built.judged.get(measure) if kind == "judged" else built.measures.get(measure)
    if figures is None:
        return None
    per_arm = figures.per_arm if aspect not in figures.by_aspect else figures.by_aspect[aspect]
    stat = per_arm.get(arm)
    return None if stat is None or stat.mean is None else float(stat.mean)


@pytest.mark.parametrize("name", CONTRASTED)
def test_a_delta_carries_a_note_exactly_where_both_arms_state_a_figure_and_it_is_not_their_difference(
    name: str,
) -> None:
    _, built = _built(name)
    unstated = 0
    for contrast in built.contrasts:
        if contrast.delta is None:
            assert contrast.delta_note is None
            continue
        sides = [
            (_stated(built, contrast.kind, contrast.measure, contrast.aspect, arm), mean)
            for arm, mean in ((contrast.control, contrast.baseline), (contrast.arm, contrast.arm_mean))
        ]
        if any(stated is None or mean is None for stated, mean in sides):
            # No figure beside the delta to explain: an arm with no cell states none.
            unstated += any(stated is None for stated, _ in sides)
            assert contrast.delta_note is None, (contrast.arm, contrast.measure, sides)
            continue
        alike = all(math.isclose(stated, mean, rel_tol=1e-9, abs_tol=1e-12) for stated, mean in sides)  # type: ignore[arg-type]
        assert (contrast.delta_note is None) == alike, (contrast.arm, contrast.measure, sides, contrast.delta_note)
        assert contrast.delta_note in (None, "paired_cases")


# --- a figure's spread, its common value, and the state of an interval it lacks ------------------------------------

_SCOPE = "uni-1"
_MODEL = "alpha"
_WAIT = MetricDescriptor(
    name="wait_seconds",
    reader_name="Wait seconds",
    data_type="numeric",
    family="mechanical",
    transferability_class="mechanical",
    attribution_scope="end_to_end",
    description="The wait seconds of one turn.",
    higher_is_better=False,
    unit="s",
    nonnegative=True,
)
_PROFILE = HostProfile(host_id="spread", host_sweepables=SHARED_CORE, measures=MeasureRegistry([_WAIT]))


def _summary(values: list[list[float]]) -> Any:
    """The one arm's summary of ``wait_seconds`` over cases holding ``values``, each case's observations in turn."""
    (cell,) = _aspect(values).cell_measures
    return next(summary for summary in cell.measures.measures if summary.name == "wait_seconds")


def _aspect(values: list[list[float]]) -> AnalysisContextBundle:
    """The one arm's aspect over cases holding ``values`` of ``wait_seconds``, each case's observations in turn."""
    storage = EvalStorage(InMemoryDocumentStore())
    cases = [f"case-{index:02d}" for index in range(len(values))]
    run = make_eval_run(
        id="run-alpha", scope_id=_SCOPE, candidate_model=_MODEL, k_runs=1, test_case_ids=cases, status="completed"
    )
    storage.save_eval_run(run)
    for case, observed in zip(cases, values, strict=True):
        for value in observed:
            result: EvalResult = make_eval_result(
                scope_id=_SCOPE,
                eval_run_id=run.id,
                test_case_id=case,
                model=_MODEL,
                variant_key=fixture_variant_key(_MODEL),
                latency=LatencyMetrics(total_ms=1000.0),
                cost_usd=0.01,
                host_measures={"wait_seconds": value},
                goal_state_outcomes=[GoalStateOutcome(expression="state.answer.delivered == true", passed=True)],
                rubric_scores=[],
            )
            storage.save_eval_result(result)
    campaign = make_campaign(scope_id=_SCOPE, run_ids=[run.id])
    storage.save_campaign(campaign)
    bundle: AnalysisContextBundle = assemble_context_bundle(campaign, storage=storage, profile=_PROFILE)
    return bundle


def test_a_spread_is_over_the_cases_each_case_s_repeats_one_draw() -> None:
    values = [[1.0, 3.0], [4.0], [9.0, 10.0, 11.0], [2.5]]
    summary = _summary(values)

    assert summary.spread == pytest.approx(statistics.stdev([2.0, 4.0, 10.0, 2.5]), rel=1e-12)
    assert summary.no_variation is None


def test_cases_that_all_read_one_value_state_it_and_no_spread() -> None:
    summary = _summary([[2.0, 2.0], [2.0], [2.0], [2.0], [2.0], [2.0]])

    assert (summary.spread, summary.no_variation) == (0.0, 2.0)


def test_an_interval_with_no_width_is_no_interval_its_figure_states_why() -> None:
    """Cases that all read one value give an interval of no width where any is given: the figure reads not available,
    no_variation, never counted (W-2)."""
    aspect = _aspect([[2.0, 2.0], [2.0], [2.0], [2.0], [2.0], [2.0]])
    built = report_input(campaign_bundle(aspect))
    (stat,) = built.measures["wait_seconds"].per_arm.values()

    assert stat.ci is None or stat.ci[0] == stat.ci[1], "the cases did not vary, so no interval has width"
    assert stat.no_variation == 2.0
    assert stat.state is not None and (stat.state.state, stat.state.reason) == ("not_available", "no_variation")


def test_one_case_states_no_spread() -> None:
    summary = _summary([[1.0, 5.0]])

    assert (summary.spread, summary.no_variation) == (None, None)


def _stats(built: ReportInput) -> Iterator[tuple[str, str, Any]]:
    for name, measure in built.measures.items():
        for per_arm in (measure.per_arm, *measure.by_aspect.values()):
            for arm, stat in per_arm.items():
                yield name, arm, stat


@pytest.mark.parametrize("name", WITH_BUNDLES)
def test_a_figure_with_no_interval_says_why_and_one_with_an_interval_carries_no_state(name: str) -> None:
    _, built = _built(name)
    stats = list(_stats(built))
    assert any(stat.spread is not None for _, _, stat in stats), f"{name} states no spread, so this checks little"
    for measure, arm, stat in stats:
        unvaried = stat.no_variation is not None or (stat.ci is not None and stat.ci[0] == stat.ci[1])
        if stat.mean is None or (stat.ci is not None and not unvaried):
            assert stat.state is None, (measure, arm)
            continue
        assert stat.state is not None and stat.state.state == "not_available", (measure, arm)
        assert stat.state.reason == ("no_variation" if unvaried else "too_few"), (measure, arm)
        if stat.no_variation is not None:
            assert stat.spread == 0.0 and round_figures(stat.no_variation) == round_figures(stat.mean)


# --- the figures the code page reads off the bundle ---------------------------------------------------------------


@pytest.mark.parametrize("name", WITH_BUNDLES)
def test_every_measure_figure_carries_its_standard_error_and_every_stratum_its_measures(name: str) -> None:
    bundle, built = _built(name)
    labels = {arm.variant_key: arm.id for arm in built.arms}
    checked = 0
    for aspect in bundle.aspects:
        held = built.measures
        for cell in aspect.cell_measures:
            label = labels.get(cell.variant_key)
            for summary in cell.measures.measures:
                if summary.texts or summary.categories or label is None:
                    continue
                figures = held[summary.name].by_aspect.get(aspect.aspect) or held[summary.name].per_arm
                if label in figures:
                    assert figures[label].sem == summary.sem, (summary.name, label)
                    checked += 1
            strata = {stratum.id: stratum for stratum in built.strata if stratum.aspect == aspect.aspect}
            for facts in cell.strata:
                stratum = strata[facts.stratum if facts.stratum is not None else "unstratified"]
                for summary in facts.measures.measures:
                    if summary.texts or summary.categories or label not in stratum.per_arm:
                        continue
                    stat = stratum.per_arm[label][summary.name]
                    point = summary.rate if summary.rate is not None else summary.mean
                    assert (stat.mean, stat.sem, stat.n) == (point, summary.sem, summary.n), (summary.name, label)
                    checked += 1
    assert checked, f"{name} holds no measure figure, so this checks nothing"


@pytest.mark.parametrize("name", WITH_BUNDLES)
def test_a_judged_mean_carries_the_interval_every_page_reads_it_with(name: str) -> None:
    bundle, built = _built(name)
    labels = {arm.variant_key: arm.id for arm in built.arms}
    checked = 0
    for aspect in bundle.aspects:
        surface = bundle_decision_surface(aspect)
        for cell in aspect.cell_measures:
            label = labels.get(cell.variant_key)
            for reading in cell.judged:
                judged = built.judged[reading.dimension]
                figures = judged.by_aspect.get(aspect.aspect) or judged.per_arm
                if label not in figures or reading.mean is None:
                    continue
                resolved = resolve_reading(
                    surface, cell_ref(cell.variant_key, cell.apparatus_class_id), reading.dimension, "judged"
                )
                expected = None if resolved.ci_low is None else (resolved.ci_low, resolved.ci_high)
                assert figures[label].ci == expected, (reading.dimension, label)
                assert judged.level == ("95%" if expected is not None else judged.level)
                checked += 1
    if any(aspect.judged_measures for aspect in bundle.aspects):
        assert checked, f"{name} judges, and no judged figure was checked"


@pytest.mark.parametrize("name", WITH_BUNDLES)
def test_each_arm_s_pass_k_and_its_figures_across_the_aspects_are_the_bundle_s(name: str) -> None:
    bundle, built = _built(name)
    for aspect, held in zip(bundle.aspects, built.aspects, strict=True):
        subjects = [subject for subject in aspect.frontier.subjects if subject.points]
        for pass_k in held.pass_k or []:
            (subject,) = [subject for subject in subjects if subject.k == pass_k.k and pass_k.per_arm is not None]
            assert pass_k.per_arm is not None
            for point in subject.points:
                stat = pass_k.per_arm[bundle.label_of(point.variant_key)]
                assert (stat.mean, stat.n_cases) == (point.pass_hat_k, point.n_pass_cases)
                assert (stat.state is None) == (point.pass_hat_k is not None)
            kept = [cell for point in subject.points for cell in point.pass_hat_k_cells if cell.reason is not None]
            assert len(pass_k.cells or []) == len(kept)
            for cell, carried in zip(kept, pass_k.cells or [], strict=True):
                assert (carried.runs, carried.headline, carried.pass_k.mean) == (
                    cell.run_ids,
                    cell.headline,
                    cell.pass_hat_k,
                )
                assert carried.state.state == "withheld"
    aggregate = bundle.readings.aggregate
    if aggregate is None:
        assert built.across_aspects is None
        return
    assert built.across_aspects is not None
    assert list(built.across_aspects.per_arm) == [bundle.label_of(arm.arm) for arm in aggregate.arms]
    for arm in aggregate.arms:
        carried = built.across_aspects.per_arm[bundle.label_of(arm.arm)]
        assert (carried.pass_rate.value, carried.pass_rate.interval) == (arm.pass_rate.value, arm.pass_rate.interval)
        assert (carried.reply_cost_usd.value, carried.reply_cost_usd.missing) == (arm.cost.value, arm.cost.missing)
        assert (carried.change is None) == (arm.change is None)
    assert built.meta.volume_per_month == bundle.declaration.volume_per_month


# --- a reading's flip margin and an aspect's margin ----------------------------------------------------------------


def _reading(low: float | None, *, margin: float | None = None) -> Reading:
    return Reading(
        arm="b",
        reading="not_separated",
        change=1.0,
        within_interval=None if low is None else (low, 4.0),
        margin=margin,
    )


def test_a_reading_flips_at_its_interval_s_worst_drop_and_one_stored_with_its_flip_reads_it() -> None:
    assert flip_margin(_reading(-3.5)) == 3.5
    assert flip_margin(_reading(0.5)) == 0.0
    assert flip_margin(_reading(None)) is None
    assert flip_margin(_reading(-3.5).model_copy(update={"flip_margin": 2.0})) == 2.0, "a stored flip is read"


def test_a_margin_no_plausible_result_could_fail_is_not_informative() -> None:
    readings = [_reading(-3.0), _reading(-1.0)]

    assert margin_informative(6.0, readings) is True, "twice the worst drop is still informative"
    assert margin_informative(6.5, readings) is False
    assert margin_informative(1.0, [_reading(0.5)]) is False, "no interval allows a drop"
    assert margin_informative(None, readings) is None
    assert margin_informative(2.0, [_reading(None)]) is None


@pytest.mark.parametrize("name", WITH_BUNDLES)
def test_every_built_reading_carries_its_flip_and_every_margined_aspect_whether_it_informs(name: str) -> None:
    _, built = _built(name)
    for aspect in built.aspects:
        for reading in aspect.readings:
            expected = None if reading.within_interval is None else max(0.0, -float(reading.within_interval[0]))
            assert reading.flip_margin == expected
        assert aspect.margin_informative == margin_informative(aspect.margin, aspect.readings)


# --- an arm's spelling ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WITH_BUNDLES)
def test_an_arm_code_named_carries_its_spelling_and_one_a_person_named_none(name: str) -> None:
    bundle, built = _built(name)
    spellings = arm_spellings(bundle)
    given = {key for arm in bundle.declaration.arms for key in bundle.arm_keys(arm)}
    for arm in built.arms:
        assert arm.variant_key is not None
        assert arm.spelling == spellings.get(arm.variant_key)
        if arm.variant_key in given:
            assert arm.spelling is None
    for key, spelling in spellings.items():
        frozen = bundle.arm_spellings.get(key)
        assert list(spelling) == (frozen.levers if frozen is not None else list(spelling)), "in the name's order"
        assert frozen is not None or len(spelling) == 1


# --- what moves and what does not -----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WITH_BUNDLES)
def test_no_fingerprint_and_no_frame_moves(name: str) -> None:
    """The fields filled here are no evidence: each aspect's fingerprint, the campaign's frame and its fingerprint are
    their goldens', and every page is the hot-path oracle's (``test_every_report_read_body_is_unchanged``). An aspect's
    bundle document gains its numeric summaries' spread, which the oracle's ``bundle:<aspect>`` goldens hold whole."""
    bundle, _ = _built(name)
    goldens = Goldens(name)
    for aspect in bundle.aspects:
        goldens.check(f"fingerprint:{aspect.aspect}", aspect.fingerprint())
    goldens.check("frame", identity_json(bundle.frame()))
    goldens.check("campaign_fingerprint", bundle.fingerprint())
    goldens.assert_unchanged()


# --- every frozen input still verifies -----------------------------------------------------------------------------


def _frozen(node: Any) -> Iterator[Mapping[str, Any]]:
    """Every object of a JSON document that holds a report input beside the digest it was frozen under."""
    if isinstance(node, dict):
        if isinstance(node.get("report_input"), dict) and isinstance(node.get("report_input_digest"), str):
            yield node
        for value in node.values():
            yield from _frozen(value)
    elif isinstance(node, list):
        for value in node:
            yield from _frozen(value)


def test_every_frozen_input_among_the_fixtures_still_verifies_its_digest() -> None:
    held = 0
    for path in sorted(FIXTURES.rglob("*.json*")):
        if path.suffix not in (".json", ".gz"):
            continue
        raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
        try:
            document = json.loads(raw)
        except ValueError:
            continue
        for frozen in _frozen(document):
            held += 1
            assert report_input_document_digest(frozen["report_input"]) == frozen["report_input_digest"], path
            ReportInput.model_validate(frozen["report_input"])
    assert held, "no fixture holds a frozen input, so this checks nothing"
