"""The writer's view hands facts at the altitude the writer makes claims, each said once, spelled as the page spells it.

- **Report-wide disclosures are the campaign's, said once.** The criteria set after their readings were visible, the
  latency and request-settings disclosures, and any disclosure two or more aspects carry alike sit in the campaign's
  own ``disclosures``, each naming the aspects carrying it; no aspect's part repeats one, nor a confound's
  explanation another part carries.
- **A sizing carries its kind.** Each unclear reading, and each unclear floor, says what would settle it on each side
  (``to_settle``: ``if_none`` and ``if_observed``, each a kind and, where it counts one, the cases in all), never the
  cases the interval was read over under a sizing's name. An observed drop already past the margin reads ``never``.
- **Rollups.** Each arm's figures across aspects with their weighting (``aggregate``); each arm's floors, broken,
  unclear and held, with the rate each read; each aspect whose control cannot let a candidate show a drop or a gain
  (``control_floor``).
- **Spelled as the page.** A change and every other figure in points reads to the one decimal the page prints, and a
  measure, a check and an aspect go by the names the page gives them.
- **A campaign bundle frozen at shape 2 hands the writer the same view** as the one it was frozen from.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, get_args

import pytest

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, LateCriterion
from threetears.evals.analysis.campaign_bundle import CampaignBundle, build_campaign_bundle, writer_view
from threetears.evals.analysis.campaign_readings import campaign_readings
from threetears.evals.analysis.depth import REPEATS_FLOOR
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.report.campaign.names import measure_name_of
from threetears.evals.analysis.report.decision import NeedKind
from threetears.evals.analysis.report.model import CampaignReport, TableBlock, TableCell
from threetears.evals.kernel.declaration import Floor, RateDrop
from threetears.evals.kernel.metrics import goal_check_measure, goal_check_of
from packages.evals.tests.report_shapes_support import (
    CHECKS,
    ArmTrials,
    AspectTrials,
    Shape,
    s1_change,
    s2_choice,
    shape_of,
)

#: The shapes every behaviour here is read on.
_SHAPES: dict[str, Callable[[], Shape]] = {"s1": s1_change, "s2": s2_choice}

#: A disclosure every aspect of a test campaign carries alike, which the page states once for the whole report.
_SETTINGS = "The simulated user's request settings were not recorded as sent."

#: A latency disclosure only one aspect carries.
_CONTENDED = "Latency is read under contention on this aspect."

#: When a criterion was set, after the readings it judges.
_SET_LATE = "2099-01-01T00:00:00+00:00"


def _bundle(
    shape: Shape, change: Callable[[AnalysisContextBundle], AnalysisContextBundle] | None = None
) -> CampaignBundle:
    """The campaign bundle of ``shape``, each aspect's evidence changed by ``change`` where given."""
    bundles = [shape.bundles[aspect.label] for aspect in shape.campaign.aspects]
    if change is not None:
        bundles = [change(bundle) for bundle in bundles]
    return build_campaign_bundle(shape.campaign, bundles, arms=shape.arms, readings=shape.readings)


def _view(shape: Shape) -> dict[str, Any]:
    return writer_view(_bundle(shape))


def _disclosed(first: str) -> Callable[[AnalysisContextBundle], AnalysisContextBundle]:
    """Each aspect's evidence carrying the report-wide disclosures: the request settings on every aspect, and on the
    aspect ``first`` alone a latency disclosure and its margin set after its readings were visible."""

    def change(bundle: AnalysisContextBundle) -> AnalysisContextBundle:
        update: dict[str, Any] = {"request_settings_disclosure": _SETTINGS}
        if bundle.aspect == first:
            update["latency_contended"] = _CONTENDED
            late = [LateCriterion(criterion="margin", set_at=_SET_LATE)]
            update["reading_scope"] = bundle.reading_scope.model_copy(update={"late_criteria": late})
        return bundle.model_copy(update=update)

    return change


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


def _tables(page: CampaignReport, name: str) -> list[TableBlock]:
    return [block for block in page.blocks if isinstance(block, TableBlock) and block.name == name]


def _as_the_page_prints(side: dict[str, Any]) -> str:
    """One side of a view's ``to_settle``, spelled as the page's To settle cell spells it, from the view's own
    ``cases`` and ``usd``: a count of cases it hands is the one the cell prints, and its price the cell's price."""
    kind = side["kind"]
    if kind == "never":
        return "none would"
    if kind == "lost":
        return "re-run the lost trials"
    if kind == "enough":
        return f"this run's {side['cases']} cases"
    if kind == "beyond":
        return f"more than {format_number(side['cases'])} cases"
    priced = "" if "usd" not in side else f", {format_number(side['usd'], prefix='$')}"
    return f"{side['cases']} cases{priced}"


# --- disclosures, once --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(_SHAPES))
def test_no_disclosure_repeats_across_aspect_parts(name: str) -> None:
    shape = _SHAPES[name]()
    first = shape.campaign.aspects[0].label
    view = writer_view(_bundle(shape, _disclosed(first)))
    parts = view["aspects"]
    said = [(key, repr(value)) for part in parts for key, value in part["disclosures"].items()]
    assert len(said) == len(set(said)), "a disclosure two aspects carry alike is the campaign's"
    explained = [(dimension, why) for part in parts for dimension, why in part.get("confound_catalog", {}).items()]
    assert len(explained) == len(set(explained))
    for part in parts:
        assert not {"latency_contended", "request_settings_disclosure"} & set(part["disclosures"])
        assert "late_criteria" not in part
    disclosures = view["disclosures"]
    labels = [part["aspect"] for part in parts]
    assert disclosures["request_settings_disclosure"] == [{"aspects": labels, "said": _SETTINGS}]
    assert disclosures["latency_contended"] == [{"aspects": [first], "said": _CONTENDED}]
    assert disclosures["late_criteria"] == [
        {"criterion": "margin", "name": None, "set_at": "1 Jan 2099", "aspects": [first]}
    ]


def test_a_disclosure_aspects_carry_alike_is_said_once_naming_them_and_one_aspect_s_own_stays_its_own() -> None:
    view = _view(s1_change())
    labels = [part["aspect"] for part in view["aspects"]]
    for key, groups in view["disclosures"].items():
        for group in groups:
            assert len(group["aspects"]) > 1 and set(group["aspects"]) <= set(labels), key
            carriers = [part for part in view["aspects"] if part["aspect"] in group["aspects"]]
            assert all(key not in part["disclosures"] for part in carriers)
    assert view["confound_catalog"], "every S1 aspect explains its confounds alike"
    assert all("confound_catalog" not in part for part in view["aspects"])


def test_a_campaign_of_one_aspect_keeps_that_aspect_s_own_disclosures_in_its_part() -> None:
    view = _view(s2_choice())
    (part,) = view["aspects"]
    assert part["disclosures"] and part["confound_catalog"]
    assert "disclosures" not in view and "confound_catalog" not in view


@pytest.mark.parametrize("name", list(_SHAPES))
def test_what_a_between_case_share_means_is_defined_once(name: str) -> None:
    view = _view(_SHAPES[name]())
    assert json.dumps(view, ensure_ascii=False).count(json.dumps(REPEATS_FLOOR, ensure_ascii=False)[1:-1]) == 1
    assert view["precision_means"] == REPEATS_FLOOR
    precisions = [reading["precision"] for part in view["aspects"] for reading in part["readings"]]
    precisions += [pair["precision"] for pair in view.get("pairs", [])]
    assert precisions and all("means" not in precision for precision in precisions if precision is not None)


# --- sizings with their kind --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(_SHAPES))
def test_to_settle_carries_kinds_and_no_reading_hands_a_count_under_a_sizing_s_name(name: str) -> None:
    shape = _SHAPES[name]()
    view = _view(shape)
    kinds = set(get_args(NeedKind))
    sizings = [
        reading["to_settle"] for part in view["aspects"] for reading in part["readings"] if reading.get("to_settle")
    ]
    sizings += [floor["to_settle"] for arm in view["floors"] for floor in arm.get("unclear", [])]
    for sizing in sizings:
        assert set(sizing) == {"if_none", "if_observed"}
        for side in sizing.values():
            assert side["kind"] in kinds
            assert ("cases" in side) == (side["kind"] in ("more", "enough", "beyond"))
    assert "cases_to_settle" not in repr(view)
    rows = [row for row in shape.readings.readings if row.projection is not None]
    assert len(sizings) >= len(rows)
    for row in rows:
        (reading,) = [
            reading
            for part in view["aspects"]
            if part["aspect"] == row.aspect
            for reading in part["readings"]
            if reading["arm"] == row.arm_label
        ]
        assert row.projection is not None
        assert reading["to_settle"]["if_none"]["kind"] == row.projection.if_none.kind
        assert reading["to_settle"]["if_observed"]["kind"] == row.projection.if_observed.kind
    assert (name == "s1") == bool(rows), "S1 has readings to settle; a choice has none"


def test_an_observed_drop_past_the_margin_reads_never_on_the_reading_and_its_floor_as_on_the_page() -> None:
    shape = _a_drop_past_the_margin()
    view = _view(shape)
    (billing, _) = view["aspects"]
    (reading,) = billing["readings"]
    assert (reading["reading"], reading["change"]) == ("unclear", -15.0)
    assert reading["to_settle"]["if_observed"] == {"kind": "never"}
    assert reading["to_settle"]["if_none"]["kind"] == "more"
    (arm,) = view["floors"]
    (floor,) = arm["unclear"]
    assert floor["to_settle"]["if_observed"] == {"kind": "never"}
    # The page sizes the same reading alike.
    settle = _table(shape.page(), "to_settle")
    assert [row["observed"].value for row in settle.rows if row["aspect"].value == "billing"] == ["none would"]


@pytest.mark.parametrize("name", list(_SHAPES))
def test_every_count_and_price_to_settle_is_the_one_the_page_s_to_settle_table_prints(name: str) -> None:
    shape = _SHAPES[name]()
    view = _view(shape)
    rows: list[dict[str, TableCell]] = [row for table in _tables(shape.page(), "to_settle") for row in table.rows]
    # Each sizing the view hands: its aspect, its arm, what the page's row says it shows, and its two sides.
    sizings = [
        (part["aspect"], reading["arm"], f"no worse than the {format_number(reading['margin'])}-point margin", settle)
        for part in view["aspects"]
        for reading in part["readings"]
        if (settle := reading.get("to_settle")) is not None
    ]
    sizings += [
        (floor["aspect"], arm["arm"], f"the {format_number(floor['margin'])}-point floor holds", floor["to_settle"])
        for arm in view["floors"]
        for floor in arm.get("unclear", [])
        if floor.get("margin") is not None
    ]
    claimed: set[int] = set()
    for aspect, arm, shows, settle in sizings:
        (at,) = [
            at
            for at, row in enumerate(rows)
            if (row["aspect"].value, row["arm"].value) == (aspect, arm) and shows in str(row["shows"].value)
        ]
        claimed.add(at)
        printed = (rows[at]["none"].value, rows[at]["observed"].value)
        assert (_as_the_page_prints(settle["if_none"]), _as_the_page_prints(settle["if_observed"])) == printed, (
            aspect,
            arm,
            shows,
        )
    assert claimed == set(range(len(rows))), "every row of the page's To settle table is one the view hands"
    sides = [side for *_, settle in sizings for side in settle.values()]
    assert (name == "s1") == any(side["kind"] == "more" and "usd" in side for side in sides)


# --- rollups ------------------------------------------------------------------------------------------------------


def test_each_arm_s_figures_across_aspects_carry_their_weighting_and_one_aspect_hands_none() -> None:
    shape = s1_change()
    view = _view(shape)
    aggregate = view["aggregate"]
    labels = [aspect.label for aspect in shape.campaign.aspects]
    assert aggregate["weighting"]["basis"] == "equal" and list(aggregate["weighting"]["weights"]) == labels
    read = shape.readings.aggregate
    assert read is not None
    assert [arm["arm"] for arm in aggregate["arms"]] == [arm.arm_label for arm in read.arms]
    for arm, figures in zip(aggregate["arms"], read.arms, strict=True):
        assert figures.pass_rate.value is not None
        assert arm["pass_rate_pct"]["value"] == round(figures.pass_rate.value, 1)
        assert "cost_per_reply_usd" in arm
        assert ("change_pts" in arm) == (figures.change is not None)
    assert "aggregate" not in _view(s2_choice()), "one aspect's figures are its own part's"


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
    (arm,) = [arm for arm in _view(shape)["aggregate"]["arms"] if arm["arm"] == "v2"]
    assert arm["cost_per_reply_usd"] == {"withheld_for": ["tone"]}
    assert "value" in arm["pass_rate_pct"], "only the figure the aspect lacks is withheld"


def test_each_arm_s_floors_roll_up_broken_unclear_and_held_each_with_the_rate_it_read() -> None:
    view = _view(s1_change())
    by_arm = {arm["arm"]: arm for arm in view["floors"]}
    assert list(by_arm)[0] in {arm for arm, held in by_arm.items() if "broken" in held}, "a broken floor first"
    assert [(floor["aspect"], floor["reads"]) for floor in by_arm["v2"]["broken"]] == [("billing", "the pass rate")]
    assert [(floor["aspect"], floor["reads"]) for floor in by_arm["v4"]["broken"]] == [("tone", CHECKS[1])]
    for arm in by_arm.values():
        assert set(arm) - {"arm"} <= {"broken", "unclear", "held"}
        for floor in arm.get("broken", []) + arm.get("held", []):
            assert "to_settle" not in floor
    assert _view(s2_choice())["floors"] == []


def test_a_control_that_cannot_show_a_drop_is_flagged_on_its_aspect_alone() -> None:
    view = _view(s1_change())
    flagged = {part["aspect"]: part.get("control_floor") for part in view["aspects"]}
    assert flagged == {"billing": None, "tone": None, "refunds": {"control": "production", "cannot_show": ["drop"]}}
    assert all("control_floor" not in part for part in _view(s2_choice())["aspects"])


# --- spelled as the page ------------------------------------------------------------------------------------------


def test_a_change_reads_as_the_page_prints_it() -> None:
    shape = s1_change()
    view = _view(shape)
    printed = {
        (row["aspect"].value, row["arm"].value): row["change"].value for row in _table(shape.page(), "aspects").rows
    }
    handed = {
        (part["aspect"], reading["arm"]): reading["change"] for part in view["aspects"] for reading in part["readings"]
    }
    assert handed == printed
    assert {(row["aspect"], row["arm"]): row["change"] for row in view["pass_rate_verdicts"]} == printed
    for part in view["aspects"]:
        for reading in part["readings"]:
            for end in reading["within_interval"] or []:
                assert end == round(end, 1)


@pytest.mark.parametrize("name", list(_SHAPES))
def test_every_measure_goes_by_its_page_name(name: str) -> None:
    shape = _SHAPES[name]()
    view = _view(shape)
    for part in view["aspects"]:
        bundle = shape.bundles[part["aspect"]]
        for key, meaning in part["measures"].items():
            if (check := goal_check_of(key)) is not None:
                assert meaning["name"] == check, "an unnamed check by its whole expression"
            else:
                assert meaning["name"] == measure_name_of(bundle, key)
    costs = [part["measures"].get("production_replicating_cost") for part in view["aspects"]]
    assert all(cost is not None and cost["name"] != "production_replicating_cost" for cost in costs)


def test_declared_names_reach_the_writer_and_move_no_reading() -> None:
    shape = s1_change()
    names = {CHECKS[1]: "cites policy"}

    def named(bundle: AnalysisContextBundle) -> AnalysisContextBundle:
        return bundle.model_copy(update={"short_name": bundle.aspect.title(), "goal_check_names": names})

    plain, view = _view(shape), writer_view(_bundle(shape, named))
    assert [part["short_name"] for part in view["aspects"]] == ["Billing", "Tone", "Refunds"]
    assert all("short_name" not in part for part in plain["aspects"])
    for part in view["aspects"]:
        assert part["measures"][goal_check_measure(CHECKS[1])]["name"] == "cites policy"
        assert part["measures"][goal_check_measure(CHECKS[0])]["name"] == CHECKS[0]
    (v4,) = [arm for arm in view["floors"] if arm["arm"] == "v4"]
    assert [floor["reads"] for floor in v4["broken"]] == ["cites policy"]
    assert [part["readings"] for part in view["aspects"]] == [part["readings"] for part in plain["aspects"]]


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
    view = writer_view(build_campaign_bundle(campaign, bundles, arms=shape.arms, readings=readings))
    reads = [
        floor["reads"]
        for arm in view["floors"]
        for state in ("broken", "unclear", "held")
        for floor in arm.get(state, [])
        if floor["floor"] == on_time.key
    ]
    (billing,) = [part for part in view["aspects"] if part["aspect"] == "billing"]
    named = measure_name_of(bundles[0], _ON_TIME)
    assert named == billing["measures"][_ON_TIME]["name"] == "Replies on time"
    assert reads and set(reads) == {named}, "never the measure's key"


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
    (arm,) = _view(shape)["floors"]
    (floor,) = arm["unclear"]
    (row,) = [row for row in _table(shape.page(), "to_settle").rows if row["shows"].value == "the 10-point floor holds"]
    assert floor["reads"] == row["decided_by"].value != "answer.correct"


# --- a bundle frozen at an earlier shape --------------------------------------------------------------------------


def test_a_campaign_bundle_frozen_at_shape_2_hands_the_writer_the_same_view() -> None:
    bundle = _bundle(s1_change())
    document = bundle.to_dict()
    readings = {key: value for key, value in document["readings"].items() if key != "aggregate"}
    frozen = CampaignBundle.from_dict({**document, "schema_version": 2, "readings": readings})
    assert writer_view(frozen) == writer_view(bundle)
    assert writer_view(frozen)["aggregate"]["arms"]
