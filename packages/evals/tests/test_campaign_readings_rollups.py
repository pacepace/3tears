"""What code reads across a campaign's aspects: each arm's figures, floors and sizings, and its control's bounds.

- **A sizing carries its kind.** An unclear reading says what would settle it on each side (``if_none`` and
  ``if_observed``); an observed drop already past the margin reads ``never``, on the reading and on the page.
- **Rollups.** Each arm's figures across aspects with their weighting (``CampaignReadings.aggregate``), a cost an
  aspect lacks withheld naming it; each arm's floors, broken, unclear and held, each on the rate it read; each aspect
  whose control cannot let a candidate show a drop or a gain, as the scorecard marks it.
- **Named as the page.** A floor on a measure no check reads, or on a judged dimension, goes by its page name.
- **A campaign bundle frozen at shape 2 reads back the same readings** as the one it was frozen from.
"""

from __future__ import annotations

import pytest

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.campaign_bundle import CampaignBundle, build_campaign_bundle
from threetears.evals.analysis.campaign_readings import campaign_readings
from threetears.evals.analysis.report.campaign.names import measure_name_of
from threetears.evals.analysis.floors import control_cannot_show
from threetears.evals.analysis.report.model import CampaignReport, TableBlock
from threetears.evals.kernel.declaration import Floor, RateDrop
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.kernel.metrics import goal_check_measure
from packages.evals.tests.report_shapes_support import (
    CHECKS,
    ArmTrials,
    AspectTrials,
    Shape,
    s1_change,
    s2_choice,
    shape_of,
)


def _bundle(shape: Shape) -> CampaignBundle:
    """The campaign bundle of ``shape``."""
    bundles = [shape.bundles[aspect.label] for aspect in shape.campaign.aspects]
    return build_campaign_bundle(shape.campaign, bundles, arms=shape.arms, readings=shape.readings)


def _a_drop_past_the_margin() -> Shape:
    """A change whose candidate drops fifteen points on ``billing``, past its ten-point margin and floor, on an
    interval too wide to show it: unclear, and no count of cases would show it within."""
    return shape_of(
        "drop",
        [
            AspectTrials(
                "billing",
                [ArmTrials("production", passing=15), ArmTrials("v2", passing=12)],
                floors=(RateDrop(max_points=10),),
            ),
            AspectTrials("tone", [ArmTrials("production", passing=10), ArmTrials("v2", passing=10)]),
        ],
        control=True,
    )


def _table(page: CampaignReport, name: str) -> TableBlock:
    return next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == name)


# --- sizings with their kind --------------------------------------------------------------------------------------


def test_an_observed_drop_past_the_margin_reads_never_on_the_reading_and_on_the_page() -> None:
    shape = _a_drop_past_the_margin()
    (reading,) = shape.readings.on_aspect("billing")
    assert (reading.reading, round(reading.change, 1)) == ("unclear", -15.0)
    assert reading.projection is not None
    assert reading.projection.if_observed.kind == "never"
    assert reading.projection.if_none.kind == "more"
    (floor,) = shape.readings.floors
    assert (floor.aspect, floor.status) == ("billing", "unclear")
    # The page sizes the same reading alike.
    settle = _table(shape.page(), "to_settle")
    assert [row["observed"].value for row in settle.rows if row["aspect"].value == "billing"] == ["none would"]


# --- rollups ------------------------------------------------------------------------------------------------------


def test_each_arm_s_figures_across_aspects_carry_their_weighting() -> None:
    shape = s1_change()
    read = shape.readings.aggregate
    assert read is not None
    labels = [aspect.label for aspect in shape.campaign.aspects]
    assert read.weighting.basis == "equal" and list(read.weighting.weights) == labels
    assert read.arms
    for figures in read.arms:
        assert figures.pass_rate.value is not None


def test_an_arm_unpriced_on_one_aspect_has_its_cost_across_aspects_withheld_naming_that_aspect() -> None:
    shape = shape_of(
        "unpriced",
        [
            AspectTrials("billing", [ArmTrials("production", passing=15), ArmTrials("v2", passing=14)]),
            AspectTrials("tone", [ArmTrials("production", passing=10), ArmTrials("v2", passing=11, cost=None)]),
        ],
        control=True,
    )
    read = shape.readings.aggregate
    assert read is not None
    (figures,) = [arm for arm in read.arms if arm.arm_label == "v2"]
    assert figures.cost.missing == ["tone"]
    assert figures.pass_rate.value is not None, "only the figure the aspect lacks is withheld"


def test_each_arm_s_floors_are_read_broken_unclear_or_held_each_on_the_rate_it_read() -> None:
    readings = s1_change().readings
    broken = {
        arm: [(floor.aspect, floor.reading) for floor in readings.broken_floors() if floor.arm_label == arm]
        for arm in ("v2", "v4")
    }
    # The pass rate is the floor's own reading (none named); a check's floor reads that check.
    assert broken == {"v2": [("billing", None)], "v4": [("tone", CHECKS[1])]}
    assert {floor.status for floor in readings.floors} <= {"broken", "unclear", "held"}
    assert s2_choice().readings.floors == []


def _cannot_show(shape: Shape) -> dict[str, tuple[str, ...]]:
    """What the control's pass rate on each aspect leaves no candidate able to show, as the scorecard marks it."""
    readings = shape.readings
    if readings.control is None:
        return {}
    flagged: dict[str, tuple[str, ...]] = {}
    for aspect in shape.campaign.aspects:
        rows = readings.on_aspect(aspect.label)
        if not rows:
            continue
        first = rows[0]
        flagged[aspect.label] = control_cannot_show(
            shape.bundles[aspect.label],
            readings.control,
            [row.arm for row in rows],
            margin=first.margin,
            passed=first.control_passed,
            trials=first.control_trials,
        )
    return flagged


def test_a_control_that_cannot_show_a_drop_is_flagged_on_its_aspect_alone() -> None:
    shape = s1_change()
    assert shape.readings.control_label == "production"
    assert _cannot_show(shape) == {"billing": (), "tone": (), "refunds": ("drop",)}
    assert _cannot_show(s2_choice()) == {}


#: A rate code reads off a cell as a floor's reading that is no goal check: a measure with a reader name.
_ON_TIME = "on_time"


def _measured_on_time(bundle: AnalysisContextBundle) -> AnalysisContextBundle:
    """``bundle`` with every cell also measuring :data:`_ON_TIME`, a pass/fail measure no check reads, passing where
    the first check passes, and its catalog naming it for a reader."""
    first = goal_check_measure(CHECKS[0])
    described = bundle.measure_catalog[first].model_copy(
        update={"name": _ON_TIME, "reader_name": "Replies on time", "family": "outcome", "pass_fail": True}
    )
    cells = [
        cell.model_copy(
            update={
                "measures": cell.measures.model_copy(
                    update={
                        "measures": [
                            *cell.measures.measures,
                            *(
                                summary.model_copy(update={"name": _ON_TIME})
                                for summary in cell.measures.measures
                                if summary.name == first
                            ),
                        ]
                    }
                )
            }
        )
        for cell in bundle.cell_measures
    ]
    return bundle.model_copy(
        update={"cell_measures": cells, "measure_catalog": {**bundle.measure_catalog, _ON_TIME: described}}
    )


def test_a_floor_on_a_measure_no_check_reads_goes_by_the_measure_s_page_name() -> None:
    shape = s1_change()
    bundles = [_measured_on_time(shape.bundles[aspect.label]) for aspect in shape.campaign.aspects]
    on_time = Floor(aspect="billing", clause=RateDrop(max_points=10, reading=_ON_TIME))
    campaign = shape.campaign.model_copy(update={"floors": [*shape.campaign.floors, on_time]})
    readings = campaign_readings(campaign, {bundle.aspect: bundle for bundle in bundles}, arms=shape.arms)
    reads = [floor.reading for floor in readings.floors if floor.floor == on_time.key]
    assert reads and set(reads) == {_ON_TIME}, "the floor reads the measure it names"
    assert measure_name_of(bundles[0], _ON_TIME) == "Replies on time", "by its page name, never its key"


def test_a_floor_on_a_judged_dimension_goes_by_the_name_the_page_s_to_settle_table_gives_it() -> None:
    scored = {"answer.correct": [1 if index < 14 else 0 for index in range(20)]}
    shape = shape_of(
        "judged",
        [
            AspectTrials(
                "billing",
                [
                    ArmTrials("production", passing=15, scores={"answer.correct": [1] * 15 + [0] * 5}),
                    ArmTrials("v2", passing=15, scores=scored),
                ],
                floors=(RateDrop(max_points=10, reading="answer.correct"),),
            ),
            AspectTrials("tone", [ArmTrials("production", passing=10), ArmTrials("v2", passing=10)]),
        ],
        control=True,
    )
    (floor,) = shape.readings.floors
    assert (floor.status, floor.reading) == ("unclear", "answer.correct")
    (row,) = [row for row in _table(shape.page(), "to_settle").rows if row["shows"].value == "the 10-point floor holds"]
    assert row["decided_by"].value != "answer.correct", "the page names the dimension, never by its key"


# --- a bundle frozen at an earlier shape --------------------------------------------------------------------------


def test_a_campaign_bundle_frozen_at_shape_2_is_refused_and_regenerated_not_read_as_this_shape() -> None:
    # A deliberate change (owner ruling, 2026-11-11, no back compat): shape 2 was read back as 3, its figures across
    # aspects read off its readings; names leaving the readings (shape 4) ended reading any earlier shape.
    bundle = _bundle(s1_change())
    document = bundle.to_dict()
    readings = {key: value for key, value in document["readings"].items() if key != "aggregate"}
    with pytest.raises(RefusedInput, match="frozen before names left the readings"):
        CampaignBundle.from_dict({**document, "schema_version": 2, "readings": readings})
