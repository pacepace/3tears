"""The weakest margin and reader names: two facts code computes about a report input.

Contracts held here (``threetears.evals.analysis.report_margins``, ``report.figure_keys``, ``report_doc_prose``):

- **The weakest margin** is the figure nearest the line it is read against, relative to the line: on r1 to r5 it is the
  margin a reviewer names (the pick at its troubleshooting floor; the pick a fraction over the pass-rate gate; the pick's
  balanced accuracy over the overall gate, not a rejected arm's narrower pass of a gate it does not need; a rejected
  arm one guardrail short; the candidate whose worst drop sits nearest the margin). An option that does not survive
  counts only by the one gate figure keeping it out; a reading wins a tie with the floor that copies it. A figure or
  interval end that is not a finite number is no figure: it gives no margin, and an option lacking a gate's figure
  does not survive, as the gate check reads it.
- ``weakest.*`` figure keys render it, in its line's units, and refuse by name where the input reads no figure against
  a line.
- **Reader names**: each id's one name (declared, else the package's reader name for it), which the ``.name`` and
  ``.label`` keys print. Prose's typed ids are matched longest first, so a shown name that is a bounded prefix of an
  id never hides it (``typed_ids``; a report document's invariants refuse what it finds, ``test_report_doc_checks.py``).
"""

from __future__ import annotations

import math

import pytest

from packages.evals.tests.fixtures.report_inputs import dataset
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.figure_keys import reader_name, reader_names, render, resolve
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_decision import surviving_options
from threetears.evals.analysis.report_margins import criterion_figures, margins, weakest_margin
from threetears.evals.analysis.report_doc_prose import typed_ids


def _input(name: str) -> ReportInput:
    return dataset.report_input_of(name)


# --- the weakest margin --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "arm", "source", "where", "gap"),
    [
        ("r1", "atlas-m", "trouble.resolves", "trouble", 0.0),
        ("r2", "brisk-s.low.r1", "pass_rate", None, 0.14),
        ("r3", "brisk-s.few", "overall", None, 0.7),
        ("r4", "f16", "facts", None, -0.6),
        # Re-pinned from 4.21, then from 2.41: the reading's worst case is read on the multiplicity-corrected interval of
        # the arm's pass-rate contrast, not the plain 95% one (-10.79); since the bounded rate interval (shape 65) that
        # bound is -12.85, not -12.59, so it sits 2.15 points from the -15 margin.
        ("r5", "qwen3.8-flash", "character adherence (v2)", None, 2.15),
    ],
)
def test_the_weakest_margin_on_each_fixture(name: str, arm: str, source: str, where: str | None, gap: float) -> None:
    found = weakest_margin(_input(name))
    assert found is not None
    assert (found.arm, found.source, found.task) == (arm, source, where)
    assert found.gap == pytest.approx(gap)


def test_a_rejected_option_counts_only_by_the_one_gate_figure_keeping_it_out() -> None:
    held = margins(_input("r3"))
    nano = [margin for margin in held if margin.arm == "nano-xs.few"]
    assert [(margin.source, margin.cell, margin.gap < 0) for margin in nano] == [("cell_floor", "complex|news", True)]


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_a_gate_figure_that_is_not_finite_is_no_figure_and_gives_no_margin(value: float) -> None:
    """r2's pick at a pass rate that is not a number: no figure for the gate, so it neither survives nor holds a
    margin, as the gate check and the survivors read it."""
    report_input = _input("r2")
    weakest = weakest_margin(report_input)
    assert weakest is not None
    data = report_input.model_dump(by_alias=True, exclude_none=True)
    key = next(c["measure"] for c in data["criteria"] if c["id"] == weakest.source)
    data["measures"][key]["per_arm"][weakest.arm]["mean"] = value
    broken = ReportInput.model_validate(data)
    criterion = next(c for c in broken.criteria if c.id == weakest.source)
    assert criterion_figures(broken, criterion, weakest.arm) == []
    assert weakest.arm not in surviving_options(broken)
    held = margins(broken)
    assert weakest.arm not in {margin.arm for margin in held}
    assert all(math.isfinite(margin.gap) for margin in held)


def test_a_reading_interval_end_that_is_not_finite_gives_no_margin() -> None:
    """r5's weakest margin is a reading's worst end; read as NaN, it gives no margin, and the weakest is finite."""
    report_input = _input("r5")
    weakest = weakest_margin(report_input)
    assert weakest is not None and weakest.kind == "margin"
    data = report_input.model_dump(by_alias=True, exclude_none=True)
    aspect = next(a for a in data["aspects"] if a["id"] == weakest.source)
    reading = next(r for r in aspect["readings"] if r["arm"] == weakest.arm)
    reading["within_interval"] = [math.nan, reading["within_interval"][1]]
    for floor in data["floors"]:
        if floor["arm"] == weakest.arm:
            floor["interval"] = [math.nan, floor["interval"][1]]
    held = margins(ReportInput.model_validate(data))
    assert not [m for m in held if m.arm == weakest.arm and m.kind in ("margin", "floor")]
    assert all(math.isfinite(margin.gap) for margin in held)


def test_a_reading_read_on_its_corrected_interval_is_weaker_than_the_floor_that_copies_its_plain_one() -> None:
    # Re-pinned from a tie: the floor and the reading used to share the plain interval's worst end (-10.79), so the
    # reading won the tie. The reading's worst case is now the corrected interval's (-12.85), so it is the weaker of
    # the two outright, and the floor still reads the plain one at the nominal level, as a floor does.
    found = weakest_margin(_input("r5"))
    assert found is not None and found.kind == "margin" and found.corrected
    floors = [margin for margin in margins(_input("r5")) if margin.kind == "floor" and margin.arm == found.arm]
    assert floors and all(not margin.corrected and margin.relative > found.relative for margin in floors)


def test_the_weakest_keys_render_in_their_lines_units() -> None:
    said = "{weakest.arm}: {weakest.value} against {weakest.criterion} at {weakest.threshold}, gap {weakest.margin}"
    # The worst case is the corrected interval's lower bound (-12.85), 2.15 inside the -15 margin.
    assert render(said, {}, forms=PageForms.of(_input("r5"))) == (
        "qwen3.8-flash (A): -12.8 against the margin on character adherence (v2) at -15, gap 2.2"
    )
    assert render(said, {}, forms=PageForms.of(_input("r1"))) == (
        "Atlas-M (B): 90% against Resolves the issue in Troubleshooting (multi-turn) at 90%, gap 0"
    )
    assert render("{weakest.margin|pts}", {}, forms=PageForms.of(_input("r4"))) == "-0.6"


def test_a_weakest_key_is_refused_by_name_where_nothing_is_read_against_a_line() -> None:
    bare = _input("r4").model_copy(update={"criteria": []})
    assert weakest_margin(bare) is None
    with pytest.raises(UnresolvableReference, match="reads no figure against a criterion, margin or floor"):
        resolve("weakest.margin", bare)


# --- reader names --------------------------------------------------------------------------------------------------


def test_each_id_has_one_reader_name_its_keys_print() -> None:
    r5 = _input("r5")
    names = reader_names(r5)
    assert names["stratum"]["off_lane"] == "off lane"
    assert names["judged"]["persona.holds_character"] == "persona holds character"
    assert names["aspect"] == {"character adherence (v2)": "character adherence (v2)"}
    assert resolve("judged.persona.holds_character.label", r5).value == "persona holds character"
    r1 = _input("r1")
    assert reader_name("criterion", "trouble.resolves", r1) == "Resolves the issue"
    assert reader_name("task", "trouble", r1) == "Troubleshooting (multi-turn)"
    assert reader_name("measure", "cost_per_1k", r1) == "Cost per 1k conversations"


def test_a_shown_name_that_is_a_bounded_prefix_of_an_id_does_not_hide_it() -> None:
    """Names and ids match in one pass, longest first: the shown name ``Tone`` never masks the ``tone`` of the id
    ``tone.kept``, so both ids the text types are found, in the order it types them."""
    ids = {"tone.kept": "stratum", "off_lane": "stratum"}
    assert typed_ids("The tone.kept loss and off_lane.", ids, ["Tone", "tone kept", "off lane"]) == [
        "tone.kept",
        "off_lane",
    ]
