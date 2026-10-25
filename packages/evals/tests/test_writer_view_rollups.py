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

from collections.abc import Callable
from typing import Any, get_args

import pytest

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, LateCriterion
from threetears.evals.analysis.campaign_bundle import CampaignBundle, build_campaign_bundle, writer_view
from threetears.evals.analysis.report.campaign.names import measure_name_of
from threetears.evals.analysis.report.decision import NeedKind
from threetears.evals.analysis.report.model import CampaignReport, TableBlock
from threetears.evals.kernel.declaration import RateDrop
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


# --- a bundle frozen at an earlier shape --------------------------------------------------------------------------


def test_a_campaign_bundle_frozen_at_shape_2_hands_the_writer_the_same_view() -> None:
    bundle = _bundle(s1_change())
    document = bundle.to_dict()
    readings = {key: value for key, value in document["readings"].items() if key != "aggregate"}
    frozen = CampaignBundle.from_dict({**document, "schema_version": 2, "readings": readings})
    assert writer_view(frozen) == writer_view(bundle)
    assert writer_view(frozen)["aggregate"]["arms"]
