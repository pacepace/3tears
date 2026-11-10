"""Prose keys print the number policy: a figure said in words keeps its number, and a gap is a change read as a distance.

Contracts held here (``threetears.evals.analysis.report.figure_keys``):

- **A number said in words is a number.** A settle figure's level (``95% two-sided``), the interval a weakest margin is
  read on (``the 95% interval``, ``the 95% interval, adjusted for 16 comparisons``) and an arm's
  unanswered calls by reason (``timeout 1,204, refused 3``) resolve to their numbers; ``format_figure`` prints each
  through the page's forms, as it prints every other figure, so no number reads one way in a sentence and another
  beside it.
- **A gap is a change read as a distance** (#1127 item 4). A criterion's gap and the weakest margin carry
  ``Figure.change``, so the page's forms read them at the measure's changes' precision, and ``Figure.distance``, so
  they print no ``+``: the prose's own words say which side of the line they are on ("2.2 points inside the margin").
  The ``signed`` filter signs one in any unit (``+99 ms``, ``+$0.20 per 1,000 replies``), and ``pts`` in points.
- **A declared line is no change.** The weakest margin's threshold and a settle's ``line`` print as declared.
- **A figure in points names its rate.** One whose path names no measure carries the rate its family reads: the
  aspect's pass rate for a reading or a margin, a floor's own rate for a floor (Kairo's ``persona.holds_character``),
  which the page's forms read its precision from.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.evals.tests.fixtures.report_inputs import dataset
from packages.evals.tests.report_doc_support import snapshot_of
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.figure_keys import format_figure, parse_key, pass_rate_measure, render, resolve
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report_input import ReportInput, Unanswered
from threetears.evals.analysis.report_margins import weakest_margin


#: Kairo's stored plan, whose prose cites the weakest margin.
_KAIRO_PLAN = Path(__file__).resolve().parent / "fixtures" / "plan_charts" / "r5-reference.json"


def _input(name: str) -> ReportInput:
    return dataset.report_input_of(name)


def _said(key: str, held: ReportInput) -> str:
    return format_figure(resolve(key, held), forms=PageForms.of(held))


def test_a_settle_level_resolves_to_its_number_and_reads_with_its_sidedness() -> None:
    held = _input("r1")
    key = "settle.atlas-l.atlas-m.overall.task.billing.level"
    figure = resolve(key, held)
    assert (figure.kind, figure.value, figure.level) == ("number", pytest.approx(0.95), True)
    assert _said(key, held) == "95% two-sided"


def test_the_plain_interval_a_weakest_margin_is_read_on_holds_its_level_as_a_number() -> None:
    held = _input("r5")
    bare = held.model_copy(update={"contrasts": [c for c in held.contrasts if c.kind != "trial"]})
    figure = resolve("weakest.interval", bare)
    assert (figure.kind, figure.value, figure.level, figure.joint) == ("number", pytest.approx(0.95), True, None)
    assert _said("weakest.interval", bare) == "the 95% interval"


def test_the_corrected_interval_a_weakest_margin_is_read_on_states_its_joint_guarantee() -> None:
    """The package's own words say what the corrected interval guarantees together, never its 99.69% (owner, Q3): the
    figure holds the joint level as its value and, beside it, the family's size, for the page's forms to print."""
    held = _input("r5")
    figure = resolve("weakest.interval", held)
    assert (figure.kind, figure.value, figure.level) == ("number", pytest.approx(0.95), True)
    assert figure.joint == (pytest.approx(0.95), 16)
    assert _said("weakest.interval", held) == "the 95% interval, adjusted for 16 comparisons"


def test_a_family_s_level_keeps_its_meaning_the_per_interval_level_as_stated() -> None:
    held = _input("r5")
    assert _said("family.level", held) == "99.69%"


def test_an_arms_unanswered_reasons_read_each_count_grouped_as_a_count_reads() -> None:
    held = _input("r5")
    arm = next(iter(held.scoreboard))
    counted = Unanswered(retried=1000, lost=207, reasons={"timeout": 1204, "refused": 3}, calls=5000)
    row = held.scoreboard[arm].model_copy(update={"unanswered": {"candidate": counted}})
    widened = held.model_copy(update={"scoreboard": {**held.scoreboard, arm: row}})
    figure = resolve(f"calls.unanswered.{arm}.reasons", widened)
    assert figure.counts == (("timeout", 1204), ("refused", 3))
    assert figure.value == "timeout 1,204, refused 3"
    assert _said(f"calls.unanswered.{arm}.reasons", widened) == "timeout 1,204, refused 3"


@pytest.mark.parametrize(
    ("name", "reads", "signed"),
    # r3's balanced accuracy states no interval, so it reads whole (two figures of its least level, 78.2), and r5's
    # pass rate reads whole points from its compared intervals (owner ruling, set digits): 0.7 and 2.15 read 1 and 2.
    # r2's 0.1 keeps its tenth, as a figure that would print as zero keeps the digit that holds it off zero.
    [("r1", "0", "0"), ("r2", "0.1", "+0.1"), ("r3", "1", "+1"), ("r4", "-0.6", "-0.6"), ("r5", "2", "+2")],
)
def test_the_weakest_margin_is_a_change_that_prints_no_plus_unless_a_filter_signs_it(
    name: str, reads: str, signed: str
) -> None:
    held = _input(name)
    figure = resolve("weakest.margin", held)
    assert figure.change and figure.distance
    assert _said("weakest.margin", held) == reads
    assert format_figure(figure, ("pts",), forms=PageForms.of(held)) == signed
    assert format_figure(figure, ("signed",), forms=PageForms.of(held)) == signed


def test_kairos_stored_prose_reads_its_weakest_margin_as_a_distance_inside_the_margin() -> None:
    plan = json.loads(_KAIRO_PLAN.read_text(encoding="utf-8"))
    watch = next(slot["text"] for slot in plan["decision"]["slots"] if slot["key"] == "watch")
    held = _input("r5")
    names = {arm.id: arm.name or arm.id for arm in held.arms}
    # In whole points, the pass rate's precision (owner ruling, set digits); the margin's 2.15 reads 2.
    assert "sits 2 points inside the margin" in render(watch, names, forms=PageForms.of(held))
    assert "+2 points inside" not in render(watch, names, forms=PageForms.of(held))
    assert render("{weakest.margin|pts}", names, forms=PageForms.of(held)) == "+2"
    assert render("{weakest.margin|signed}", names, forms=PageForms.of(held)) == "+2"


@pytest.mark.parametrize(("name", "reads"), [("r2", "clears it by 0.1"), ("r4", "clears it by -0.6")])
def test_a_criterion_s_gap_is_a_distance_that_prints_no_plus(name: str, reads: str) -> None:
    """r2's and r4's weakest margins are criteria's, so their gap is the criterion's: a clearance and a miss."""
    held = _input(name)
    found = weakest_margin(held)
    assert found is not None and found.kind == "criterion"
    assert render("{weakest.arm} clears it by {weakest.margin}", {}, forms=PageForms.of(held)).endswith(reads)


@pytest.mark.parametrize(
    ("criterion", "reads", "signed"),
    [
        ("answers/bar/total_ms", "99 ms", "+99 ms"),
        ("answers/bar/cost_usd", "$0.20 per 1,000 replies", "+$0.20 per 1,000 replies"),
    ],
)
def test_a_gap_in_ms_or_in_dollars_signs_with_the_signed_filter(criterion: str, reads: str, signed: str) -> None:
    held = ReportInput.model_validate(snapshot_of("prototype2/s2").report_input)
    figure = resolve(f"criteria.{criterion}.beta.gap", held)
    assert format_figure(figure, forms=PageForms.of(held)) == reads
    assert format_figure(figure, ("signed",), forms=PageForms.of(held)) == signed


def test_signed_signs_only_a_change_and_never_beside_abs() -> None:
    held = _input("r5")
    with pytest.raises(UnresolvableReference, match="no change"):
        format_figure(resolve("weakest.threshold", held), ("signed",), forms=PageForms.of(held))
    with pytest.raises(UnresolvableReference, match="`abs` and `signed`"):
        parse_key("weakest.margin|abs|signed")


@pytest.mark.parametrize(
    "key",
    ["weakest.threshold", "settle.margin.character adherence (v2).qwen3.8-flash.line"],
)
def test_a_declared_line_is_no_change(key: str) -> None:
    held = _input("r5")
    figure = resolve(key, held)
    assert not figure.change and _said(key, held) == "-15"


def test_a_figure_in_points_whose_path_names_no_measure_is_its_family_s_rate() -> None:
    held = _input("r5")
    for key in ("weakest.margin", "weakest.value", "weakest.threshold", "reading.1017.change"):
        figure = resolve(key, held)
        assert figure.scale == "points" and figure.set_of == pass_rate_measure(held), key


#: Kairo's floor on character adherence, which candidate 1017 reads on the judged rate it names.
_FLOOR = "character adherence (v2)/rate_drop/*"
_FLOOR_ASPECT = "character adherence (v2)"


@pytest.mark.parametrize(
    "key",
    [
        f"aspects.{_FLOOR_ASPECT}.floor.{_FLOOR}.1017.low",
        f"aspects.{_FLOOR_ASPECT}.floor.{_FLOOR}.1017.interval",
        f"settle.floor.{_FLOOR_ASPECT}.{_FLOOR}.1017.line",
    ],
)
def test_a_floor_read_on_its_own_rate_is_held_under_that_rate_not_the_pass_rate(key: str) -> None:
    held = _input("r5")
    figure = resolve(key, held)
    assert figure.set_of == ("persona.holds_character", "judged")
    forms = PageForms.of(held)
    start = len(forms.ledger)
    format_figure(figure, forms=forms)
    assert {entry.measure for entry in forms.ledger[start:]} == {"persona.holds_character"}


def test_a_floor_that_names_no_rate_reads_the_pass_rate() -> None:
    held = _input("r5")
    figure = resolve(f"aspects.{_FLOOR_ASPECT}.floor.{_FLOOR}.qwen3.8-flash.low", held)
    assert figure.set_of == pass_rate_measure(held)


def test_a_weakest_margin_on_a_floor_reads_the_floor_s_own_rate() -> None:
    """With no reading's margin and only 1017's floor left, the weakest margin is that floor, on its judged rate."""
    held = _input("r5")
    aspects = [
        aspect.model_copy(
            update={
                "margin": None,
                "readings": [reading.model_copy(update={"margin": None}) for reading in aspect.readings],
            }
        )
        for aspect in held.aspects
    ]
    floors = [read for read in held.floors if read.arm == "1017"]
    only_floor = held.model_copy(update={"aspects": aspects, "floors": floors})
    found = weakest_margin(only_floor)
    assert found is not None and (found.kind, found.arm) == ("floor", "1017")
    for key in ("weakest.margin", "weakest.value", "weakest.threshold"):
        assert resolve(key, only_floor).set_of == ("persona.holds_character", "judged"), key
