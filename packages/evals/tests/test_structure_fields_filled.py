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

from packages.evals.tests.factories import fixture_variant_key, make_campaign, make_eval_result, make_eval_run
from packages.evals.tests.test_hot_path_oracle import WITH_BUNDLES, Goldens, subject
from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle
from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME
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


@pytest.mark.parametrize("name", WITH_BUNDLES)
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


@pytest.mark.parametrize("name", WITH_BUNDLES)
def test_a_delta_carries_a_note_exactly_where_it_is_not_the_difference_of_the_arms_stated_figures(name: str) -> None:
    _, built = _built(name)
    for contrast in built.contrasts:
        if contrast.delta is None:
            assert contrast.delta_note is None
            continue
        sides = [
            (_stated(built, contrast.kind, contrast.measure, contrast.aspect, arm), mean)
            for arm, mean in (
                (contrast.control, contrast.baseline),
                (contrast.arm, None if contrast.baseline is None else contrast.baseline + contrast.delta),
            )
        ]
        alike = all(
            stated is not None and mean is not None and math.isclose(stated, mean, rel_tol=1e-9, abs_tol=1e-12)
            for stated, mean in sides
        )
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
    (cell,) = bundle.cell_measures
    return next(summary for summary in cell.measures.measures if summary.name == "wait_seconds")


def test_a_spread_is_over_the_cases_each_case_s_repeats_one_draw() -> None:
    values = [[1.0, 3.0], [4.0], [9.0, 10.0, 11.0], [2.5]]
    summary = _summary(values)

    assert summary.spread == pytest.approx(statistics.stdev([2.0, 4.0, 10.0, 2.5]), rel=1e-12)
    assert summary.no_variation is None


def test_cases_that_all_read_one_value_state_it_and_no_spread() -> None:
    summary = _summary([[2.0, 2.0], [2.0], [2.0], [2.0], [2.0], [2.0]])

    assert (summary.spread, summary.no_variation) == (0.0, 2.0)


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
        if stat.mean is None or stat.ci is not None:
            assert stat.state is None, (measure, arm)
            continue
        assert stat.state is not None and stat.state.state == "not_available", (measure, arm)
        assert stat.state.reason == ("no_variation" if stat.no_variation is not None else "too_few"), (measure, arm)
        if stat.no_variation is not None:
            assert stat.spread == 0.0 and round_figures(stat.no_variation) == round_figures(stat.mean)


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
