"""A check's measure is a pass rate in the report input: its grade says so, and every page and tool reads it as one.

A goal-state check's measure (``goal_state:<expression>``) pools observations that are each a 1 (passed) or a 0, as
does every measure of the goal-state family, a host's own check (a boolean with a better end, or a scorer graded pass
or fail) and a measurement's pass rate; ``accuracy`` and ``match`` pool classifications each right or wrong. The input
states that grade (``Measure.grade``), so the figure keys read the measure on the share scale: it prints as a percent,
takes the ``pct`` filter, and a price per point on it is per percentage point. A measure that is no check (a cost, a
latency) carries no grade.

Held on every shape campaign and on Kairo round 2, each built from its bundle, and on a change whose trials also carry
a host's checks and a classification. Every expectation is read off the bundle and the input.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache

import pytest

from packages.evals.tests.fixtures.report_plans.dataset import kairo_bundle
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_EXTRACTION_FAMILY, TOYHOST_MEASURES, toyhost_profile
from packages.evals.tests.report_shapes_support import CHECKS, SHAPES
from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.analysis.analyst.cited import cite
from threetears.evals.analysis.analyst.price import price_tradeoff
from threetears.evals.analysis.analyst.snapshot import analyst_input
from threetears.evals.analysis.campaign_bundle import CampaignBundle, build_campaign_bundle
from threetears.evals.analysis.campaign_readings import campaign_readings
from threetears.evals.analysis.report.figure_keys import format_figure, resolve
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel import ACCURACY_MEASURE, MATCH_MEASURE, MetricDescriptor
from threetears.evals.kernel.metrics import goal_check_measure
from threetears.evals.kernel.host import MeasureRegistry

#: A host's own checks: a condition with a better end, and a scorer graded pass or fail.
_ON_TARGET = MetricDescriptor(
    name="on_target",
    reader_name="On target",
    data_type="boolean",
    family="mechanical",
    transferability_class="mechanical",
    attribution_scope="end_to_end",
    higher_is_better=True,
    description="Whether the reply landed on the case's target.",
)
_FIELDS_MATCH = MetricDescriptor(
    name="fields_match",
    reader_name="Fields match",
    data_type="numeric",
    family="mechanical",
    transferability_class="mechanical",
    attribution_scope="end_to_end",
    higher_is_better=True,
    value_range=(0.0, 1.0),
    pass_fail=True,
    description="Whether every field of the reply matched the case's: 1 or 0.",
)


def _hosted() -> CampaignBundle:
    """The one-aspect change (``s4``) with each trial also landing the host's two checks and a classification:
    on target and matching on every case but every third, its fields matching on even cases."""
    shape = SHAPES["s4"]()
    profile = toyhost_profile()
    profile = replace(
        profile,
        measures=MeasureRegistry((*TOYHOST_MEASURES, _ON_TARGET, _FIELDS_MATCH), families=(TOYHOST_EXTRACTION_FAMILY,)),
    )
    results = {
        run_id: [
            result.model_copy(
                update={
                    "host_measures": {
                        "on_target": index % 3 != 0,
                        "fields_match": 1.0 if index % 2 == 0 else 0.0,
                        MATCH_MEASURE: index % 3 != 0,
                    }
                }
            )
            for index, result in enumerate(held)
        ]
        for run_id, held in shape.results.items()
    }
    ((label, _),) = shape.bundles.items()
    bundle = assemble_context_bundle(shape.campaign, storage=ToyhostStorage(shape.runs, results), profile=profile)
    readings = campaign_readings(shape.campaign, {label: bundle}, arms=shape.arms)
    return build_campaign_bundle(shape.campaign, [bundle], arms=shape.arms, readings=readings)


#: The inputs' names: every shape, Kairo round 2, and the change carrying a host's checks.
_NAMES = [*SHAPES, "kairo round 2", "s4 with a host's checks"]


@cache
def _bundles() -> dict[str, CampaignBundle]:
    bundles = {
        name: build_campaign_bundle(
            shape.campaign, list(shape.bundles.values()), arms=shape.arms, readings=shape.readings
        )
        for name, shape in ((name, make()) for name, make in SHAPES.items())
    }
    bundles["kairo round 2"] = kairo_bundle()
    bundles["s4 with a host's checks"] = _hosted()
    return bundles


@pytest.fixture(scope="module")
def inputs() -> dict[str, ReportInput]:
    return {name: analyst_input(bundle) for name, bundle in _bundles().items()}


def _what_each_is(bundle: CampaignBundle) -> dict[str, str]:
    """Every measure whose observations are graded, by the grade, as the bundle describes it: a goal-state measure or
    a check by its catalog descriptor, ``accuracy`` and ``match`` by class label, a measurement's pass rate by the
    trials it counts passed."""
    graded: dict[str, str] = {}
    for aspect in bundle.aspects:
        for name, descriptor in aspect.measure_catalog.items():
            if name in (ACCURACY_MEASURE, MATCH_MEASURE):
                graded[name] = "class_label"
            elif descriptor.family == "goal_state" or descriptor.is_check:
                graded[name] = "pass_fail"
    for measured in bundle.readings.measures:
        if measured.passed is not None:
            graded.setdefault(measured.measure, "pass_fail")
    return graded


@pytest.mark.parametrize("name", _NAMES)
def test_every_check_measure_states_its_grade_and_no_other_measure_does(
    name: str, inputs: dict[str, ReportInput]
) -> None:
    held = inputs[name]
    graded = _what_each_is(_bundles()[name])
    figured = {key for key, measure in held.measures.items() if measure.per_arm or measure.by_aspect}

    assert figured & graded.keys(), f"{name} holds no graded measure with a figure, so it tests nothing"
    for key, measure in held.measures.items():
        assert measure.grade == graded.get(key), f"{name}: {key} is graded {measure.grade}, not {graded.get(key)}"


@pytest.mark.parametrize("name", _NAMES)
def test_a_graded_measure_reads_as_a_percent(name: str, inputs: dict[str, ReportInput]) -> None:
    held = inputs[name]
    graded = [key for key, measure in held.measures.items() if measure.grade is not None and measure.per_arm]
    assert graded, f"{name} holds no graded measure on its lead aspect"
    for key in graded:
        for arm, stat in held.measures[key].per_arm.items():
            if stat.mean is None:
                continue
            path = f"measure.{key}.{arm}.mean"
            figure = cite(path, held)
            assert figure.scale == "share", f"{name}: {path} reads on the {figure.scale} scale"
            assert figure.reads.endswith("%"), f"{name}: {path} prints {figure.reads}, not a percent"
            assert float(figure.reads.removesuffix("%")) == pytest.approx(stat.mean * 100, abs=0.05)
            assert format_figure(resolve(path, held), ("pct",)) == figure.reads


def test_a_price_per_point_on_a_check_is_per_percentage_point(inputs: dict[str, ReportInput]) -> None:
    """On the choice, the price of a point of the check's pass rate is the cost gap over the gain in points: the
    difference of the two arms' rates, times 100."""
    held = inputs["s2"]
    key = goal_check_measure(CHECKS[0])
    rates = {arm: stat.mean for arm, stat in held.measures[key].per_arm.items()}
    traded = price_tradeoff(held, key)

    assert traded.steps, "the choice's frontier takes no step, so no price per point is tested"
    for price in traded.steps:
        cheaper, dearer = price.from_arm.removeprefix("arm:"), price.to_arm.removeprefix("arm:")
        low, high = rates[cheaper], rates[dearer]
        assert low is not None and high is not None
        gap = cite(f"cost_per_1k.{dearer}", held).value
        base = cite(f"cost_per_1k.{cheaper}", held).value
        assert isinstance(gap, float) and isinstance(base, float)
        assert price.gain.value == pytest.approx(high - low)
        assert price.cost_gap.value == pytest.approx(gap - base)
        assert price.per_point is not None
        assert price.per_point.value == pytest.approx((gap - base) / ((high - low) * 100))
