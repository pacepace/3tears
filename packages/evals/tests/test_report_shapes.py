"""The four shared campaign shapes are what every later test of the page relies on them being.

A test of a page rule passes on at least two of them, so each must stay the shape its docstring names: a change of
several candidates and aspects with an uneven depth, a choice, a measurement, and a change of one aspect. These
tests hold the shapes, not the page: what the page says over them is each page test's own.
"""

from __future__ import annotations

import pytest

from threetears.evals.analysis import campaign_report_html, campaign_report_markdown

from packages.evals.tests.report_shapes_support import CHECKS, SHAPES, s1_change, s2_choice, s3_measurement, s4_change


def test_the_four_shapes_are_four_kinds_of_campaign() -> None:
    shapes = {name: build() for name, build in SHAPES.items()}
    assert {name: shape.readings.kind for name, shape in shapes.items()} == {
        "s1": "change",
        "s2": "choice",
        "s3": "measurement",
        "s4": "change",
    }
    assert {name: len(shape.campaign.aspects) for name, shape in shapes.items()} == {"s1": 3, "s2": 1, "s3": 2, "s4": 1}
    assert {name: len(shape.arms) for name, shape in shapes.items()} == {"s1": 4, "s2": 3, "s3": 1, "s4": 2}


def test_s1_runs_one_arm_deeper_on_one_aspect_floors_a_check_and_has_a_control_passing_nothing() -> None:
    shape = s1_change()
    assert [arm.label for arm in shape.arms if arm.role == "control"] == ["production"]
    trials = {(row.aspect, row.arm_label): row.arm_trials for row in shape.readings.readings}
    assert trials[("tone", "v3")] == 40
    assert {count for key, count in trials.items() if key != ("tone", "v3")} == {20}
    assert [(floor.aspect, floor.clause.model_dump().get("reading")) for floor in shape.campaign.floors] == [
        ("billing", None),
        ("tone", CHECKS[1]),
    ]
    refunds = shape.readings.on_aspect("refunds")
    assert {row.control_passed for row in refunds} == {0}


def test_s2_is_three_arms_with_no_control_priced_and_timed_against_two_bars() -> None:
    shape = s2_choice()
    assert shape.readings.control is None and not shape.campaign.floors
    assert {bar.measure for bar in shape.readings.bars} == {"cost_usd", "total_ms"}
    assert len(shape.readings.pairs) == 3


def test_s3_measures_one_arm_on_judged_dimensions_against_a_bar() -> None:
    shape = s3_measurement()
    assert {bar.measure for bar in shape.readings.bars} == {"answer.correct"}
    assert {reading.aspect for reading in shape.readings.measures} == {"accuracy", "clarity"}


def test_s4_is_one_candidate_on_one_aspect_with_no_floor_and_no_margin() -> None:
    shape = s4_change()
    (row,) = shape.readings.readings
    assert row.margin is None and not shape.campaign.floors


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_each_shape_renders_the_same_page_every_time_it_is_built(name: str) -> None:
    first, second = SHAPES[name]().page(), SHAPES[name]().page()
    assert campaign_report_markdown(first) == campaign_report_markdown(second)
    assert campaign_report_html(first) == campaign_report_html(second)
