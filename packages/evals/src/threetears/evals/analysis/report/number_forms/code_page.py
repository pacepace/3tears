"""The code page's decisions: the scorecard's form for each measure, and the cells, changes and intervals it prints
at that form.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from threetears.evals.analysis.campaign_readings import MeasureReading
from threetears.evals.analysis.quantities import MONEY, PERCENT, POINTS
from threetears.evals.analysis.report.cost_unit import MoneyRuler
from threetears.evals.analysis.report.number_forms.policy import (
    _POLICY_MOST_PLACES,
    _RATE_BOUNDS,
    _ZERO_BOUND,
    policy_bounded_places,
    policy_figure,
    policy_interval,
    policy_places,
    policy_rate,
)
from threetears.evals.analysis.report.number_forms.printed import MeasureForm
from threetears.evals.analysis.said import SaidFigure
from threetears.evals.analysis.report.unit_once import interval_alone, interval_beside, unit_spaced
from threetears.evals.kernel.numbers import round_half_even


# --- the code page: the scorecard's decisions --------------------------------------------------------------------------


def _rate_form(key: str, kind: str, widths: Sequence[float], *, change_factor: float) -> MeasureForm:
    """A rate's form on a code page: in percent, its changes in points, every figure at the places its narrowest
    interval sets (:func:`policy_places`, capped at one decimal)."""
    places = policy_places([100.0 * width for width in widths], capped=True)
    return MeasureForm(
        key=key,
        kind=kind,
        reads_as="rate",
        unit=PERCENT,
        change_unit=POINTS,
        factor=100.0,
        change_factor=change_factor,
        span=100.0,
        places=places,
        half_width=100.0 * min(widths) if widths else None,
    )


def _scorecard_form(
    key: str,
    shown: Sequence[MeasureReading],
    rates: frozenset[str],
    rulers: Mapping[str, MoneyRuler],
    widths: Sequence[float],
) -> MeasureForm:
    """One measure's form on a measurement's scorecard: a rate in percent, money on its ruler, any other measure in its
    own unit, a rate and a measured figure at the places the campaign's set of it sets (:func:`policy_places`); a
    measure with no interval at two significant figures of its least figure, over every cell that holds one figure of
    it (a cell read under several setups holds none)."""
    # A rate is decided before money, so a measure both a rate and money would print its money through the rate's form;
    # no measure the package reads is both, and the scorecard's cells check money first.
    if key in rates:
        return _rate_form(key, "measure", widths, change_factor=100.0)
    ruler = rulers.get(key)
    if ruler is not None:
        return MeasureForm(
            key=key,
            kind="measure",
            reads_as="money",
            unit=MONEY,
            change_unit=MONEY,
            factor=ruler.scale,
            change_factor=ruler.scale,
            money=ruler.form,
            ruler=ruler,
        )
    cells: dict[tuple[str, str], list[MeasureReading]] = {}
    for reading in shown:
        if reading.measure == key:
            cells.setdefault((reading.aspect, reading.arm), []).append(reading)
    figures = [found[0].value for found in cells.values() if len(found) == 1 and found[0].value is not None]
    places = policy_places(widths, figures)
    return MeasureForm(
        key=key,
        kind="measure",
        reads_as="measured",
        unit="",
        change_unit="",
        factor=1.0,
        change_factor=1.0,
        places=places,
        half_width=min(widths) if widths else None,
    )


def _code_places(form: MeasureForm) -> int:
    """The places every figure of a code page's measure reads at (:attr:`MeasureForm.places`)."""
    return 0 if form.places is None else form.places


def _code_bounds(form: MeasureForm, *, change: bool) -> tuple[float, ...]:
    """The boundaries a code page's figure may not print onto: a rate's floor and ceiling for its level, zero for a
    change in points; none for any other measure's figure (rule 2 sets its places alone)."""
    if form.reads_as != "rate":
        return ()
    return _ZERO_BOUND if change else _RATE_BOUNDS


def _code_scaled(form: MeasureForm, value: float) -> float:
    """A level as given, scaled into the unit the code page prints it in: money on its ruler, a share in percent."""
    return form.ruler.drawn(value) if form.ruler is not None else value * form.factor


def _code_level_places(form: MeasureForm, scaled: float) -> int:
    """The places one level reads at: its measure's, or more where a rate would print onto its floor or ceiling."""
    return policy_bounded_places(scaled, _code_places(form), bounds=_code_bounds(form, change=False))


def _code_held(form: MeasureForm, scaled: float) -> float:
    """The number a scorecard's cell holds for a level, scaled: rounded half to even at the places it reads at, so a
    cell that writes it at those places prints its decided text."""
    return float(round_half_even(scaled, _code_level_places(form, scaled)))


def _code_cell_text(form: MeasureForm, scaled: float) -> str:
    """A level's text as the code page states it: a rate in percent (``0%`` for none), any other figure at its
    measure's places in its unit, an exact zero whole."""
    if form.reads_as == "rate":
        return policy_rate(scaled, _code_places(form))
    return unit_spaced(policy_figure(scaled, _code_places(form), bounds=()), form.unit)


def _code_change_places(form: MeasureForm, value: float) -> int:
    """The places one change reads at: its measure's, or more where a change in points would print onto zero."""
    return policy_bounded_places(value * form.change_factor, _code_places(form), bounds=_code_bounds(form, change=True))


def _code_change_held(form: MeasureForm, value: float) -> float:
    """The number a scorecard's cell holds for a change (or how far one reaches), scaled into its unit and rounded half
    to even at the places it reads at."""
    return float(round_half_even(value * form.change_factor, _code_change_places(form, value)))


def _code_change(form: MeasureForm, value: float) -> str:
    """A change as the code page states it: signed, in its unit (points for a rate), at its measure's places."""
    digits = policy_figure(
        value * form.change_factor, _code_places(form), bounds=_code_bounds(form, change=True), signed=True
    )
    return unit_spaced(digits, form.change_unit)


def _said_held(figure: SaidFigure, places: int) -> tuple[float, int]:
    """A floor's figure as printed and the places it prints at: ``places``, or as many more as keep a bound on its
    side of zero and of the margin its sentence compares it with (owner's boundary rule), at most the policy's cap."""
    held = _said_bound(figure, places)
    while places < _POLICY_MOST_PLACES and _said_crosses(figure, held):
        places += 1
        held = _said_bound(figure, places)
    return held, places


def _said_crosses(figure: SaidFigure, held: float) -> bool:
    """Whether a bound printed as ``held`` lands on the far side of zero or its margin from the value it bounds: a
    value past a line printed on it or short of it, one within a line printed past it, or one on a line printed off
    it."""
    if figure.bound is None:
        return False
    # Each is compared at the policy's cap, so a value on its margin but for float residue (``0.30 - 0.20`` in points)
    # reads as on it.
    value = round(figure.value, _POLICY_MOST_PLACES)
    lines = (0.0,) if figure.margin is None else (0.0, round(figure.margin, _POLICY_MOST_PLACES))
    return any(
        (value > line and held <= line) or (value < line and held > line) or (value == line and held != line)
        for line in lines
    )


def _said_bound(figure: SaidFigure, places: int) -> float:
    """A floor's figure at ``places``: rounded to the nearest, or, where its sentence states it as a bound, toward
    what the bound claims (down for a lower bound, up for an upper one), so it never reads past itself. A figure at
    the places already (``24.0``) stays as it is, float residue aside."""
    if figure.bound is None:
        return round(figure.value, places)
    factor = 10.0**places
    scaled = round(figure.value * factor, 9)
    return (math.floor(scaled) if figure.bound == "lower" else math.ceil(scaled)) / factor


def _code_interval(
    form: MeasureForm, low: float, high: float, *, change: bool, drawn: bool = False
) -> tuple[str, str, str]:
    """An interval as the code page states it, in its one form, ``low to high`` (:func:`policy_interval`): a change's
    ends signed, in points for a rate, the unit once after the high end; money's ends each in its ruler's form. Ends
    given as drawn (``drawn``: a rate already in percent) are not scaled again."""
    if form.ruler is not None and not change:
        low_text, high_text = form.ruler.spell_ends(low, high)
        return low_text, high_text, interval_beside(low_text, high_text)
    factor = 1.0 if drawn else form.change_factor if change else form.factor
    ends = [low * factor, high * factor]
    if form.span is not None:
        floor = -form.span if change else 0.0
        ends = [min(max(end, floor), form.span) for end in ends]
    unit = form.change_unit if change else form.unit
    bounds = _RATE_BOUNDS if form.reads_as == "rate" and not change else ()
    low_text, high_text, _ = policy_interval(ends[0], ends[1], _code_places(form), change=change, bounds=bounds)
    return unit_spaced(low_text, unit), unit_spaced(high_text, unit), interval_alone(low_text, high_text, unit)


__all__: list[str] = []
