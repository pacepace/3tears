"""How a page looks, by one owner: each arm's colour, tag and mark, and how a quantity is stated on a chart.

**An arm's look.** The control is ink-gray and marked by a square; each candidate takes the kit's series colours in
fixed order and a mark shape in fixed order, so an arm reads apart without its colour (GENERATOR §5.2). Two keys
pick it: an arm's letter tag (:func:`arm_token`, :func:`arm_shape`), which the report's own charts carry, and a
plan's colour slot (:func:`slot_token`, :func:`slot_shape`: ``-1`` for the control, then ``0``, ``1``… in the
input's order), which a planned page reads. A planned page holds each arm's look in a :class:`PageLook`, with the
reader names the page writes in place of ids.

**A quantity on a chart** (:func:`quantity`): the factor from the input's figure to the drawn one, the kit's format,
and the unit the package's number rule writes beside it (:func:`~threetears.evals.analysis.quantities.with_unit`).

==============  ===========================================================  ==================
unit            drawn as                                                     kit format
==============  ===========================================================  ==================
share           a rate on 0 to 1, as a percent (×100); a change, in points    ``pct``; ``+num:0``
percent, pct    a percent as it is; a change, in points                      ``pct``; ``+num:0``
points          points                                                       ``+num:0``
usd             money per reply, always per 1,000 replies (×1000)            ``usd:2``, tick ``usd:1``
usd_per_1k      dollars per 1,000 replies                                     ``usd:2`` under $10, else ``usd``
usd_per_1m      dollars per 1,000,000 replies                                 ``usd``
dollars         dollars as they are (a run's total)                           ``usd:2``
ms, s           milliseconds, seconds                                        ``ms``, ``s:1``
score           a judged score                                               ``num:2``
(other)         the number, its unit beside it                               ``num``
==============  ===========================================================  ==================

**Notes are code's, by the package's number rule** (:func:`figure_text`), so a chart never prints its own rounding of
a figure: the kit's would read a change of 0.04 points as "+0", and the rule reads it "+0.04".

**Axes** (:func:`magnitude_domain`, :func:`centred_domain`, :func:`band_domain`): bars from zero to a round value past
the largest; a change about its centre, spanning every end, the centre and any reference line, a point past each
end; a rate's band clipped to 0 to 100.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from threetears.evals.analysis.quantities import MONEY, PERCENT, POINTS, signed_with_unit, with_unit
from threetears.evals.analysis.report.charts import ChartShape
from threetears.evals.analysis.style import humanize

if TYPE_CHECKING:
    # The page builder's facts sit above the report's blocks, which read this module: named for the checker alone.
    from threetears.evals.analysis.report.figure_keys import NamedKind
    from threetears.evals.analysis.report_plan_facts import FactArm

__all__ = [
    "ArmLook",
    "PageLook",
    "Quantity",
    "arm_shape",
    "arm_token",
    "band_domain",
    "centred_domain",
    "cost_axis_format",
    "figure_text",
    "magnitude_domain",
    "page_look",
    "quantity",
]

#: Each candidate's identity colour, by its place after the control: the kit's series in fixed order (GENERATOR
#: §5.2), repeating past the fifth.
_SERIES = ("--s1", "--s2", "--s3", "--s4", "--s5")

#: The control's identity colour: the kit's ink-gray.
_CONTROL_TOKEN = "--ink-2"

#: The control's tag (GENERATOR §6.1).
_CONTROL_TAG = "P"

#: The control's mark on a scatter.
_CONTROL_SHAPE: ChartShape = "square"

#: Each candidate's mark on a scatter, by its place, so an arm reads apart without its colour.
_SHAPES: tuple[ChartShape, ...] = ("circle", "triangle", "diamond")

#: A plan's colour slot for the control.
_CONTROL_SLOT = -1


def _tag_place(tag: str) -> int:
    """A candidate's place after the control by its tag: A is 0, Z 25, AA 26."""
    place = 0
    for letter in tag:
        place = place * 26 + (ord(letter) - ord("A") + 1)
    return place - 1


def arm_token(tag: str | None) -> str | None:
    """An arm's identity colour by its letter tag: the control's ink-gray, a candidate's series colour.

    :param tag: The arm's tag (``P`` for the control), or None for an arm the page has not tagged.
    :ptype tag: str | None
    :return: The kit colour token, or None untagged.
    :rtype: str | None
    """
    if tag is None:
        return None
    return _CONTROL_TOKEN if tag == _CONTROL_TAG else _SERIES[_tag_place(tag) % len(_SERIES)]


def arm_shape(tag: str | None) -> ChartShape | None:
    """An arm's scatter mark by its letter tag: the control's square, a candidate's shape in order.

    :param tag: The arm's tag (``P`` for the control), or None for an arm the page has not tagged.
    :ptype tag: str | None
    :return: The mark, or None untagged.
    :rtype: ChartShape | None
    """
    if tag is None:
        return None
    return _CONTROL_SHAPE if tag == _CONTROL_TAG else _SHAPES[_tag_place(tag) % len(_SHAPES)]


def _slot_token(slot: int) -> str:
    """An arm's identity colour by its plan colour slot."""
    return _CONTROL_TOKEN if slot == _CONTROL_SLOT else _SERIES[slot % len(_SERIES)]


def _slot_shape(slot: int) -> ChartShape:
    """An arm's scatter mark by its plan colour slot."""
    return _CONTROL_SHAPE if slot == _CONTROL_SLOT else _SHAPES[slot % len(_SHAPES)]


def cost_axis_format(top: float, *, money: bool) -> str:
    """The kit format of a cost axis running from zero to ``top``: as many decimals as a tick a fifth of the way up
    needs to print a digit of its own, so no two ticks read alike and none reads as zero; money keeps at least the
    cents and drops them from $100.

    :param top: The axis's top, in the cost's unit, above zero.
    :ptype top: float
    :param money: Whether the cost is money (``usd``) rather than a plain number (``num``).
    :ptype money: bool
    :return: The format, e.g. ``usd:2`` for an axis to $0.48, ``usd:4`` for one to $0.0048.
    :rtype: str
    """
    digits = max(0, math.ceil(-math.log10(top / 5)))
    if money:
        digits = 0 if top >= 100 else max(2, digits)
    return f"{'usd' if money else 'num'}:{min(digits, 9)}"


@dataclass(frozen=True)
class ArmLook:
    """One arm as every part of a planned page shows it.

    :ivar id: The arm's id.
    :ivar name: Its short name (its declared short name, else its alias, else its id).
    :ivar tag: Its letter tag: ``P`` for the control, then ``A``, ``B``…
    :ivar token: Its identity colour, a kit token.
    :ivar shape: Its scatter mark.
    :ivar control: Whether it is the control.
    """

    id: str
    name: str
    tag: str
    token: str
    shape: ChartShape
    control: bool

    @property
    def label(self) -> str:
        """The arm as a chart row names it: its tag, then its name (``A · qwen3.8-flash``)."""
        return f"{self.tag} · {self.name}"


@dataclass(frozen=True)
class PageLook:
    """Every arm's look on one planned page, the order the page lists arms in, and the names it writes for ids.

    :ivar arms: Each arm's look, by id, in the input's order.
    :ivar order: The options table's order, then every other arm in the input's order: the order a summary chart's
        rows take (GENERATOR §5.3).
    :ivar survivors: The arms that pass every gate (:func:`~threetears.evals.analysis.report_decision.surviving_options`):
        a summary chart fades the rest.
    :ivar recommended: The arm the plan recommends, which a summary chart bolds; None for none.
    :ivar names: Each id's one reader name, by kind (aspect, measure, judged, stratum, task, criterion), then id.
    """

    arms: Mapping[str, ArmLook]
    order: tuple[str, ...]
    survivors: frozenset[str]
    recommended: str | None
    names: Mapping[str, Mapping[str, str]]

    def arm(self, arm_id: str) -> ArmLook:
        """One arm's look.

        :param arm_id: The arm's id.
        :ptype arm_id: str
        :return: Its look.
        :rtype: ArmLook
        :raises KeyError: The input holds no such arm.
        """
        return self.arms[arm_id]

    @property
    def control(self) -> ArmLook | None:
        """The control's look; None where the input has no control."""
        return next((look for look in self.arms.values() if look.control), None)

    def reader(self, kind: NamedKind, identifier: str) -> str:
        """The one name a reader is shown for an id of a kind, else the id in words (its underscores and dots as
        spaces), never the id itself.

        :param kind: What the id names.
        :ptype kind: NamedKind
        :param identifier: The id.
        :ptype identifier: str
        :return: The reader name.
        :rtype: str
        """
        return self.names.get(kind, {}).get(identifier) or humanize("dimension", identifier)

    def ordered(self, arm_ids: Iterable[str]) -> list[str]:
        """``arm_ids`` in the page's order (:attr:`order`), any the order does not hold after them as given.

        :param arm_ids: Arm ids.
        :ptype arm_ids: Iterable[str]
        :return: The same ids, reordered.
        :rtype: list[str]
        """
        given = list(dict.fromkeys(arm_ids))
        place = {arm: index for index, arm in enumerate(self.order)}
        return sorted(given, key=lambda arm: (place.get(arm, len(place)), given.index(arm)))


def page_look(
    arms: Sequence[FactArm],
    slots: Mapping[str, int],
    *,
    order: Sequence[str],
    survivors: Iterable[str],
    recommended: str | None,
    names: Mapping[str, Mapping[str, str]],
) -> PageLook:
    """A planned page's look: each arm's tag and name as code's facts give them, its colour and mark from the plan's
    colour slot (an arm the plan holds no slot for takes the facts' own).

    :param arms: Each arm as code's facts tell of it, in the input's order.
    :ptype arms: Sequence[FactArm]
    :param slots: Each arm's colour slot, by id, as the plan stores it.
    :ptype slots: Mapping[str, int]
    :param order: The options table's arms, in its order.
    :ptype order: Sequence[str]
    :param survivors: The arms that pass every gate.
    :ptype survivors: Iterable[str]
    :param recommended: The arm the plan recommends; None for none.
    :ptype recommended: str | None
    :param names: Each id's reader name, by kind, then id.
    :ptype names: Mapping[str, Mapping[str, str]]
    :return: The look.
    :rtype: PageLook
    """
    looks: dict[str, ArmLook] = {}
    for arm in arms:
        slot = slots.get(arm.id, arm.color_slot)
        looks[arm.id] = ArmLook(
            id=arm.id,
            name=arm.name,
            tag=arm.tag,
            token=_slot_token(slot),
            shape=_slot_shape(slot),
            control=arm.role == "control",
        )
    listed = [arm for arm in order if arm in looks]
    return PageLook(
        arms=looks,
        order=tuple(dict.fromkeys([*listed, *looks])),
        survivors=frozenset(survivors),
        recommended=recommended,
        names=names,
    )


@dataclass(frozen=True)
class Quantity:
    """How one quantity is drawn on a chart (this module's table).

    :ivar factor: What the input's figure is multiplied by to reach the drawn one.
    :ivar fmt: The kit's format for the drawn values.
    :ivar unit: The unit the package's number rule writes beside a drawn value (``%``, ``pts``, ``usd``, ``ms``…);
        empty for a bare number.
    :ivar tick_fmt: The axis labels' format, where it differs from the values'.
    :ivar change: Whether the values are changes, read signed.
    :ivar rate: Whether the values are a rate in percent (or a change in one), whose axis never runs past 0 to 100.
    """

    factor: float
    fmt: str
    unit: str
    tick_fmt: str | None = None
    change: bool = False
    rate: bool = False


#: The money units a cost is stated in, per reply and per a number of replies, and dollars as they are (a run's total).
_PER_REPLY, _PER_THOUSAND, _PER_MILLION, _DOLLARS = "usd", "usd_per_1k", "usd_per_1m", "dollars"

#: The magnitude from which dollars per 1,000 replies drop their cents on a chart.
_WHOLE_DOLLARS_FROM = 10.0


def quantity(unit: str, values: Sequence[float] = (), *, change: bool = False) -> Quantity:
    """How a quantity in ``unit`` is drawn (this module's table).

    :param unit: The figure's unit, as the page reads it: ``share``, ``percent`` (or ``pct``), ``points``, ``usd``,
        ``usd_per_1k``, ``usd_per_1m``, ``ms``, ``s``, ``score``, empty for a bare number, or any other unit, written
        beside the number.
    :ptype unit: str
    :param values: The drawn values, in display units, where the format follows their magnitude.
    :ptype values: Sequence[float]
    :param change: Whether the values are changes (an arm less the control), read signed.
    :ptype change: bool
    :return: The quantity.
    :rtype: Quantity
    """
    sign = "+" if change else ""
    if unit in ("share", "percent", "pct"):
        factor = 100.0 if unit == "share" else 1.0
        if change:
            return Quantity(factor, "+num:0", POINTS, change=True, rate=False)
        return Quantity(factor, "pct", PERCENT, rate=True)
    if unit == "points":
        return Quantity(1.0, "+num:0", POINTS, change=True)
    if unit == _PER_REPLY:
        return Quantity(1000.0, f"{sign}usd:2", MONEY, tick_fmt=f"{sign}usd:1", change=change)
    if unit == _PER_THOUSAND:
        largest = max((abs(value) for value in values), default=0.0)
        return Quantity(1.0, f"{sign}usd" if largest >= _WHOLE_DOLLARS_FROM else f"{sign}usd:2", MONEY, change=change)
    if unit == _PER_MILLION:
        return Quantity(1.0, f"{sign}usd", MONEY, change=change)
    if unit == _DOLLARS:
        return Quantity(1.0, f"{sign}usd:2", MONEY, change=change)
    if unit == "ms":
        return Quantity(1.0, f"{sign}ms", "ms", change=change)
    if unit == "s":
        return Quantity(1.0, f"{sign}s:1", "s", change=change)
    if unit == "score":
        return Quantity(1.0, f"{sign}num:2", "", change=change)
    return Quantity(1.0, f"{sign}num", unit, change=change)


def figure_text(value: float, drawn: Quantity) -> str:
    """A drawn value as a chart's note writes it: by the package's number rule, in its unit, signed where the
    quantity is a change (never "+0" for a change that rounds there: the rule keeps its digits).

    :param value: The value, in display units.
    :ptype value: float
    :param drawn: Its quantity.
    :ptype drawn: Quantity
    :return: The text.
    :rtype: str
    """
    return signed_with_unit(value, drawn.unit) if drawn.change else with_unit(value, drawn.unit)


#: The round steps an axis ends on, within each power of ten.
_ROUND_STEPS = (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)


def _round_up(value: float) -> float:
    """The least round value (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6 or 8 times a power of ten) at or above ``value``, which is above zero."""
    power = 10.0 ** math.floor(math.log10(value))
    return next(step * power for step in _ROUND_STEPS if step * power >= value)


def magnitude_domain(values: Iterable[float], *, rate: bool = False) -> tuple[float, float]:
    """A bar chart's axis: from zero (or a round value under the least, where one is negative) to a round value past
    the largest; a rate's never past 100.

    :param values: Every value and interval end the chart draws, in display units.
    :ptype values: Iterable[float]
    :param rate: Whether the values are a rate in percent.
    :ptype rate: bool
    :return: ``(low, high)``.
    :rtype: tuple[float, float]
    """
    drawn = [value for value in values if math.isfinite(value)]
    top = max(drawn, default=0.0)
    bottom = min(drawn, default=0.0)
    high = _round_up(top * 1.05) if top > 0 else 1.0
    low = -_round_up(-bottom * 1.05) if bottom < 0 else 0.0
    return low, min(high, 100.0) if rate and top <= 100 else high


def _padded(low: float, high: float, drawn: Quantity) -> tuple[float, float]:
    """``low`` to ``high`` with room past each end: a point (whole points) for a rate or a change in points, else a
    twentieth of the span."""
    if drawn.unit in (POINTS, PERCENT):
        return float(math.floor(low) - 1), float(math.ceil(high) + 1)
    pad = (high - low) * 0.05 or abs(high) * 0.05 or 1.0
    return low - pad, high + pad


def centred_domain(values: Iterable[float], drawn: Quantity, *, centre: float = 0.0) -> tuple[float, float]:
    """A change's axis: every value, interval end and reference line it draws, and its centre (zero, or a
    no-preference point), with room past each end.

    :param values: Every value, interval end and reference line drawn, in display units.
    :ptype values: Iterable[float]
    :param drawn: The quantity drawn.
    :ptype drawn: Quantity
    :param centre: The value the chart is centred on.
    :ptype centre: float
    :return: ``(low, high)``.
    :rtype: tuple[float, float]
    """
    ends = [value for value in values if math.isfinite(value)] + [centre]
    return _padded(min(ends), max(ends), drawn)


def band_domain(values: Iterable[float], drawn: Quantity) -> tuple[float, float]:
    """A level's axis (lines, dot strips): every value and band end drawn, with room past each end, a rate's held to
    0 to 100.

    :param values: Every value and band end drawn, in display units.
    :ptype values: Iterable[float]
    :param drawn: The quantity drawn.
    :ptype drawn: Quantity
    :return: ``(low, high)``.
    :rtype: tuple[float, float]
    """
    ends = [value for value in values if math.isfinite(value)] or [0.0]
    low, high = _padded(min(ends), max(ends), drawn)
    if drawn.rate:
        return max(low, 0.0), min(high, 100.0)
    return low, high
