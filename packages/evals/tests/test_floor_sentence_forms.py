"""A floor's sentences are read as numbers and spelled in the forms of the page that prints them.

A floor's reasons are read before any page exists, so each figure in them is held as a number
(:class:`~threetears.evals.analysis.report.said.Said`) and the page states it in its own forms
(:meth:`~threetears.evals.analysis.report.forms.PageForms.said`): how far a pass rate's drop reaches reads at the places
that page reads the pass rate's changes at, a count as the page states any count, a margin as declared. What the
campaign's readings freeze keeps its one decimal and its ungrouped counts, since the campaign's fingerprint reads it.
"""

from __future__ import annotations

import functools
import operator
from collections.abc import Callable
from typing import Literal

import pytest

from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME
from threetears.evals.analysis.report.decision import FloorReadings, RateReading, floor_readings, read_floor
from threetears.evals.analysis.report.forms import MeasureForm, PageForms
from threetears.evals.analysis.report.said import Said, SaidFigure
from threetears.evals.kernel.declaration import Floor, RateDrop

from packages.evals.tests.report_shapes_support import Shape, s1_change


@functools.cache
def _s1() -> Shape:
    return s1_change()


def _billing_floor(shape: Shape) -> Floor:
    return next(floor for floor in shape.campaign.floors if floor.aspect == "billing")


def _v2_on_billing(shape: Shape) -> tuple[Said, str]:
    """v2's broken floor on billing, as read, and as the campaign's readings froze it."""
    control = shape.readings.control
    assert control is not None
    bundle = shape.bundles["billing"]
    verdict = read_floor(
        _billing_floor(shape),
        floor_readings(bundle, control, shape.key("v2"), check_names=bundle.named_checks()),
        control=shape.readings.control_label,
        candidate="v2",
    )
    (status,) = [status for status in shape.readings.floors if status.aspect == "billing" and status.arm_label == "v2"]
    assert verdict.outcome == "broken"
    return verdict.reason, status.reason


def _code_page_at(places: int) -> PageForms:
    """A code page whose pass rate reads at ``places``, its changes in points."""
    form = MeasureForm(
        key=TRIAL_PASS_NAME,
        kind="trial",
        reads_as="rate",
        unit="%",
        change_unit="pts",
        factor=100.0,
        change_factor=1.0,
        span=100.0,
        places=places,
        half_width=9.0,
    )
    return PageForms("code", {(TRIAL_PASS_NAME, "trial"): form})


def test_the_code_page_states_a_floor_s_drop_at_its_pass_rate_s_places() -> None:
    """s1's code page reads its pass rate in whole points, so v2's broken floor says how far its drop reaches in whole
    points there, rounded down since the words state a lower bound (``at least 23`` of 23.94: never ``24``, which
    would overstate it), while the readings keep the one decimal they froze (``at least 23.9``)."""
    shape = _s1()
    forms = PageForms.of_campaign(shape.readings, shape.reports)
    assert forms.measure(TRIAL_PASS_NAME, kind="trial").places == 0
    reason, stored = _v2_on_billing(shape)
    assert stored == (
        "v2 passed 6 of 20 to production's 18, a drop that exceeds the margin of 10 points on its whole interval (at "
        "least 23.9)"
    )
    assert forms.said(reason) == stored.replace("23.9", "23")
    assert reason.stored() == stored
    assert "23" in {entry.text for entry in forms.issued(TRIAL_PASS_NAME, kind="trial", role="change")}


def test_an_aspect_page_states_a_floor_s_drop_at_its_pass_rate_s_set() -> None:
    """An aspect page holds no form of its own for the pass rate: it reads the pass rate's set, whole points on s1's
    billing, as its contrasts do."""
    shape = _s1()
    forms = PageForms.of_aspects(list(shape.bundles.values()), shape.readings)["billing"]
    places = forms.contrast_places(TRIAL_PASS_NAME, "trial")
    assert places is not None and places.change == 0
    reason, stored = _v2_on_billing(shape)
    assert forms.said(reason) == stored.replace("23.9", "23")


def test_a_page_reading_the_pass_rate_to_a_decimal_states_the_drop_to_one() -> None:
    reason, stored = _v2_on_billing(_s1())
    assert _code_page_at(1).said(reason) == stored
    assert _code_page_at(2).said(reason).endswith("(at least 23.94)")


def test_a_page_that_states_no_change_of_the_rate_reads_it_to_a_decimal_as_frozen() -> None:
    reason, stored = _v2_on_billing(_s1())
    assert PageForms.off_page().said(reason) == stored


def test_a_bound_never_reads_past_itself() -> None:
    """A lower bound (``at least``) rounds down at the page's places and an upper one (``at most``) up; one at the
    places already reads as itself; a figure stated as itself rounds to the nearest."""

    def said(value: float, bound: Literal["lower", "upper"] | None) -> Said:
        return Said.of(SaidFigure(value, "points", bound=bound))

    whole = _code_page_at(0)
    assert [whole.said(said(23.9, bound)) for bound in ("lower", "upper", None)] == ["23", "24", "24"]
    assert [whole.said(said(23.1, bound)) for bound in ("lower", "upper", None)] == ["23", "24", "23"]
    assert [whole.said(said(24.0, bound)) for bound in ("lower", "upper", None)] == ["24", "24", "24"]
    assert [_code_page_at(1).said(said(28.44, bound)) for bound in ("lower", "upper")] == ["28.4", "28.5"]
    # A page that states no change of the rate reads it to a decimal, its bounds directed as any page's.
    assert [PageForms.off_page().said(said(28.44, bound)) for bound in ("lower", "upper")] == ["28.4", "28.5"]


def test_a_bound_never_rounds_across_its_margin_or_zero() -> None:
    """Where a bound's directed rounding at the page's places would land past the margin its sentence compares it with,
    or on zero, it takes a decimal more until it does not: a held floor of 2.5 points on a page of whole points reads
    ``at most 2.4``, never ``at most 3``; a broken one never reads as on its margin."""
    held = RateReading("the pass rate", 90, 100, 90, 100, interval=(-0.024, 0.01))
    floor = Floor(aspect="billing", clause=RateDrop(max_points=2.5))
    verdict = read_floor(
        floor, FloorReadings(rates=(held,), control_cases=100, candidate_cases=100), control="v1", candidate="v2"
    )
    assert verdict.outcome == "held"
    assert _code_page_at(0).said(verdict.reason) == "v2's drop is shown to stay within 2.5 points (it is at most 2.4)"
    whole = _code_page_at(0)

    def bound(value: float, side: Literal["lower", "upper"], margin: float | None) -> str:
        return whole.said(Said.of(SaidFigure(value, "points", bound=side, margin=margin)))

    assert bound(2.5, "upper", 2.5) == "2.5"
    assert bound(10.3, "lower", 10.0) == "10.3"
    assert bound(10.04, "lower", 10.0) == "10.04"
    # Never "at least 0" for a drop that is one.
    assert bound(0.6, "lower", None) == "0.6"
    # A value on its margin but for float residue, either way, prints on it.
    assert bound(100 * (0.30 - 0.20), "upper", 10.0) == "10"
    assert bound(100 * (0.30 - 0.20), "lower", 10.0) == "10"
    assert bound(10.000000000000002, "upper", 10.0) == "10"
    assert bound(10.000000000000002, "lower", 10.0) == "10"
    # An ordinary bound, clear of its margin, is unaffected.
    assert bound(23.9, "lower", 10.0) == "23"
    assert bound(28.4, "upper", 15.0) == "29"


def test_a_drop_is_kept_off_zero_at_the_page_s_places() -> None:
    """A drop of 0.04 points on a page of whole points reads ``0.04``, and is a drop, never ``no change``; frozen to a
    decimal it reads as no change, as it always did."""
    said = Said.of("from ", SaidFigure(0.04, "change"), " to at most ", SaidFigure(0.04, "points"))
    assert _code_page_at(0).said(said) == "from a drop of 0.04 to at most 0.04"
    assert said.stored() == "from no change to at most 0"
    assert _code_page_at(0).said(Said.of(SaidFigure(-35.5, "change"))) == "a gain of 36"


def test_a_floor_s_counts_read_as_the_page_states_a_count() -> None:
    """The counts a reason rests on are grouped on the page (``1,200``); the readings freeze them as read (``1200``)."""
    floor = Floor(aspect="billing", clause=RateDrop(max_points=10))
    dropped = RateReading("the pass rate", 1180, 1200, 600, 1200, interval=(-0.6, -0.4))
    broken = read_floor(
        floor, FloorReadings(rates=(dropped,), control_cases=1200, candidate_cases=1200), control="v1", candidate="v2"
    )
    assert broken.outcome == "broken"
    assert PageForms.off_page().said(broken.reason).startswith("v2 passed 600 of 1,200 to v1's 1,180, ")
    assert broken.reason.stored().startswith("v2 passed 600 of 1200 to v1's 1180, ")
    alike = RateReading("the pass rate", 1200, 1200, 1300, 1300)
    unclear = read_floor(
        floor, FloorReadings(rates=(alike,), control_cases=1200, candidate_cases=1300), control="v1", candidate="v2"
    )
    assert PageForms.off_page().said(unclear.reason) == (
        "the pass rate cannot separate the arms (v2 1,300 of 1,300, v1 1,200 of 1,200)"
    )


def test_a_margin_reads_as_declared() -> None:
    said = Said.of("within ", SaidFigure(12.345, "margin"), " points")
    assert PageForms.off_page().said(said) == "within 12.345 points"


@pytest.mark.parametrize(
    "turned",
    [str, format, lambda said: f"{said}", lambda said: f"{said!s}", lambda said: operator.mod("%s", said)],
    ids=["str", "format", "f-string", "f-string !s", "percent"],
)
def test_a_sentence_held_as_numbers_is_never_turned_into_text_as_held(turned: Callable[[Said], str]) -> None:
    """Its figures have no text until a page's forms give them one, so turning it into text another way refuses
    rather than printing ``Said(parts=...)``; its repr stays, for a debugger."""
    said = Said.of("at least ", SaidFigure(23.9, "points"))
    with pytest.raises(TypeError, match="spelled by a page's forms"):
        turned(said)
    assert repr(said).startswith("Said(parts=")
