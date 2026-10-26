"""The price of a trade: one axis priced in the other, by code, every figure a key.

``price_tradeoff`` does the arithmetic a tradeoff page owes (``docs/eval-report-kit/GENERATOR.md`` §4: price one axis
in the other, at real volume, and say where the best option flips), which the planner otherwise did in prose. Every
number it returns is a figure key (:mod:`~threetears.evals.analysis.analyst.cited`): the ``price.<from>.<to>.<gain>.*``
and ``cost_per_month.<arm>`` families of :mod:`~threetears.evals.analysis.report.figure_keys`.

Design note. What the spec left open and this module decided is marked *(inferred)*.

**The options priced.** Only options that survive the gates
(:func:`~threetears.evals.analysis.report_decision.surviving_options`): an option a gate or a guardrail removes is
never priced beside the others, since a trade that includes it would weigh a broken policy as a cost. Of the
survivors, an option is priced when the cost rule states its cost per 1,000 replies (``cost_per_1k.<arm>``) and the
gain axis places it.

**The two axes.** Cost is cost per 1,000 replies, read as exact: the input states no interval on it *(inferred: the
pipeline's cost is tokens at list price, with no sampling error stated)*. The gain axis is the one measure or judged
dimension the caller names (``gain_key``, as ``campaign_overview`` lists it), read toward its better side: a judged
dimension is higher is better, and a measure that states no direction is refused.
:func:`~threetears.evals.analysis.report.figure_keys.gain_axis` places each arm on it: on its contrast against the
control, when every arm holding a figure there holds exactly one contrast stating a difference (the paired estimate,
which the contrast's interval belongs to); otherwise on its overall mean, with no interval. One basis per axis, so
every price on an axis is a difference of the same positions *(inferred)*.

**The frontier.** An option is dominated when another costs no more and reads no worse, and is ahead on one of the
two. Of the undominated options, ordered by cost, the frontier is the upper hull: the options that are best at some
price per point, where an option's net value at a price ``p`` is ``p`` times its position less its cost. Each step
from one frontier option to the next is priced by ``price.<cheaper>.<dearer>.<gain>.per_point``, the price per point
at which the best option moves from the one to the other: GENERATOR §4's flip points, read on every surviving
option. An undominated option under the hull is best at no single price, and is listed off the frontier with the two
step prices that pass it by.

**A price per point, and its interval.** ``per_point`` is the cost gap over the gain, the gain in points (a share's
difference times 100; per unit on an axis not in points). Cost being exact, the interval comes from the gain's: over
the part of the gain's interval on the gain's side of zero, the price runs from the cost gap over the end farthest
from zero (``per_point_low``) to the cost gap over the end nearest it (``per_point_high``). When the gain's interval
reaches or crosses zero, the evidence allows a gain as small as nothing, so a point may cost any amount: the high end
is **unbounded**. The key refuses it as unbounded and the result says so in words; no number stands in for it. A gain
with no interval (a move between two candidates, each holding its contrast against the control alone; an axis read
on means) has no interval on its price, and the result says that too. A move where one option is at least as good on
both axes has no price per point: the key refuses it, and the result names which option is ahead on both.

**Against one option.** Beside the steps, every priced option is priced against one: the control, or ``arm_filter``,
each move written from the cheaper option to the dearer, so ``per_point`` reads as the saving per point lost, or the
cost per point gained (GENERATOR §4's formula). These are paged with ``limit`` and ``offset`` *(inferred: the design's
result lists the frontier, the steps, the monthly costs and the options off the frontier; GENERATOR states its price
against the control, so that comparison is here too)*.

**A month's cost.** Only at the declared ``meta.volume_per_month``: ``cost_per_month.<arm>``, and the ``monthly_``
price fields, each the figure per 1,000 replies times the volume over 1,000 *(inferred: the monthly fields are added
to the price family, so that the monthly price per point GENERATOR asks for is a key)*. With no declared volume no
monthly figure is stated, and the result says so. A volume the caller supplies is never used: no key could state a
figure on it.

**Off the frontier, and why.** Every other option, with its reason: removed by a gate or guardrail (each failing
figure, as the gate check words it), no cost stated, no figure on the gain axis, dominated (by which option, with
that move's gain and cost gap), or under the hull (best at no price).

**Hybrids: deferred.** A modeled hybrid (GENERATOR §4: hand conversations back after turn five) is an option nobody
ran. Its cost and gain would be a model resting on an assumption (that the control's quality returns after the hand
back), and no key of the frozen input states them, so code could only print numbers no key names. It stays the
analyst's to describe from ``measure_breakdown`` and the strata keys, labelled modeled, not measured. A routed mix
over routable tasks is code's (:func:`~threetears.evals.analysis.report_gates.routing_picks`) and shows on
``block_render``'s gate matrix, in its pick column; it is not priced here, since the blended cost of the picks is a
number no key states. Pricing either needs a ruling on which assumptions a key may rest on.

**Code's arithmetic, not a call.** The result prices the trade and chooses nothing: where nothing ranked the axes in
advance, which price band holds is the decider's call. No condition here reads what kind of eval a campaign is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.analysis.analyst.cited import CitedFigure, cite, cite_if
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.figure_keys import GainAxis, gain_axis
from threetears.evals.analysis.report_decision import control_arm, surviving_options
from threetears.evals.analysis.report_gates import gate_check
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.errors import RefusedInput

__all__ = ["PRICE_PAGE", "PriceTradeoff", "price_tradeoff"]

#: The options one page covers unless the caller asks for more: each is a move priced against one option (up to
#: ten figures) or a reason it is off the frontier.
PRICE_PAGE = 6

#: How many valid choices a refusal lists before it stops.
_SHOWN = 24

#: What every result says it is.
_READS_AS = (
    "code's arithmetic on the frozen input: each option's cost priced in its gain, and where the best option moves "
    "as the price of a point moves. It chooses nothing; where nothing declared an exchange rate, the band that holds "
    "is the decider's call"
)

#: Why hybrids are not priced, as every result says it.
_HYBRIDS = (
    "not priced: a modeled hybrid is an option nobody ran, and no key states its cost or gain; describe one from "
    "measure_breakdown and the strata keys, labelled modeled, not measured. A routed mix over routable tasks shows "
    "in a gate_matrix block's pick column"
)

#: How the interval on a move's price reads: both ends stated, the high end unbounded, no interval stated, or no price.
Bounds = Literal["bounded", "unbounded_above", "unstated", "no_price"]

#: Why an option is off the frontier.
OffReason = Literal["gated", "no_cost", "no_gain_figure", "dominated", "never_best"]

_MODEL = ConfigDict(frozen=True, extra="forbid")


def _arm(arm_id: str) -> str:
    """An arm as a result names it: ``arm:<id>``, which renders as the arm's name and tag."""
    return f"arm:{arm_id}"


def _listed(names: Sequence[str]) -> str:
    shown = ", ".join(repr(name) for name in names[:_SHOWN])
    return shown + (f", and {len(names) - _SHOWN} more" if len(names) > _SHOWN else "") if names else "none"


def _number(figure: CitedFigure) -> float:
    if not isinstance(figure.value, int | float):
        raise TypeError(f"{figure.key} is a {figure.kind}, not a number")
    return float(figure.value)


class FrontierOption(BaseModel):
    """One option on the frontier: best at some price per point."""

    model_config = _MODEL

    arm: str = Field(description="The option, as arm:<id>.")
    cost: CitedFigure = Field(description="Its cost per 1,000 replies.")
    monthly_cost: CitedFigure | None = Field(description="Its cost a month at the declared volume; None without one.")
    level: CitedFigure | None = Field(description="Its own mean on the gain axis, where the input states one.")


class Price(BaseModel):
    """One move priced: from the cheaper option to the dearer, its gain on the axis, and what a point of it costs."""

    model_config = _MODEL

    from_arm: str = Field(description="The cheaper option, as arm:<id>.")
    to_arm: str = Field(description="The dearer option, as arm:<id>.")
    gain: CitedFigure = Field(description="How much better the dearer reads on the gain axis, toward better.")
    gain_interval: CitedFigure | None = Field(description="The interval on the gain; None where none is stated.")
    cost_gap: CitedFigure = Field(description="What the dearer costs more per 1,000 replies.")
    per_point: CitedFigure | None = Field(description="The cost gap per point gained; None where the move has none.")
    per_point_low: CitedFigure | None = Field(description="The least a point costs, as the gain's interval allows.")
    per_point_high: CitedFigure | None = Field(
        description="The most a point costs, as the gain's interval allows; None where unbounded or unstated."
    )
    bounds: Bounds = Field(description="How the price's interval reads: bounded, unbounded_above, unstated, no_price.")
    says: str = Field(description="The price's interval, or the absence of one, in words.")
    monthly_cost_gap: CitedFigure | None = Field(description="The cost gap a month at the declared volume.")
    monthly_per_point: CitedFigure | None = Field(description="The price per point a month at the declared volume.")
    monthly_per_point_low: CitedFigure | None = Field(description="Its low end a month, where stated.")
    monthly_per_point_high: CitedFigure | None = Field(description="Its high end a month, where stated and bounded.")


class OffFrontier(BaseModel):
    """One option not on the frontier, and why."""

    model_config = _MODEL

    arm: str = Field(description="The option, as arm:<id>.")
    reason: OffReason = Field(
        description="gated, no_cost, no_gain_figure, dominated (another is at least as good on both axes) or "
        "never_best (under the frontier: best at no single price per point)."
    )
    by: list[str] = Field(description="The options that put it off: the one dominating it, or the two passing it.")
    says: list[str] = Field(description="Why, in words: each failing gate figure, or what the figures below show.")
    figures: list[CitedFigure] = Field(description="The figures that show it, where a figure does.")


class PriceTradeoff(BaseModel):
    """A trade priced: the options best at some price, the prices the best option moves at, and every move priced
    against one option."""

    model_config = _MODEL

    reads_as: str = Field(description="What this is: code's arithmetic, which chooses nothing.")
    notes: list[str] = Field(description="What could not be read, and why; first, so a cut reading keeps it.")
    gain_key: str = Field(description="The gain axis: a measure or judged dimension, by key.")
    better: Literal["higher_is_better", "lower_is_better"] = Field(description="Which way the gain axis is better.")
    basis: str = Field(description="What the gain axis places each option by, in words.")
    volume: CitedFigure | None = Field(description="The declared monthly volume; None where the input states none.")
    frontier: list[FrontierOption] = Field(description="The options best at some price per point, cheapest first.")
    steps: list[Price] = Field(
        description="Each move along the frontier, cheapest first: its per_point is the price per point at which the "
        "best option moves from from_arm to to_arm."
    )
    against_arm: str | None = Field(description="The option the others are priced against, as arm:<id>.")
    against: list[Price] = Field(description="Each priced option on this page, priced against against_arm.")
    off_frontier: list[OffFrontier] = Field(description="Each option on this page off the frontier, and why.")
    total: int = Field(description="The options paged, every arm in the input's order.")
    offset: int = Field(description="The first option this page covers.")
    next_offset: int | None = Field(description="The offset of the next page; None when this is the last.")
    hybrids: str = Field(description="Why a modeled hybrid or a routed mix is not priced here.")


@dataclass(frozen=True)
class _Option:
    """One priced option: its cost figure, and its place on the gain axis (used to order, never printed)."""

    arm: str
    cost: CitedFigure
    position: float

    @property
    def spend(self) -> float:
        return _number(self.cost)


def _cost_or_none(report_input: ReportInput, arm: str) -> CitedFigure | None:
    try:
        return cite(f"cost_per_1k.{arm}", report_input)
    except UnresolvableReference:
        # NOSILENT: an arm the cost rule states no cost for has no `cost_per_1k.<arm>` key at all, so its refusal is
        # how "no cost" reads; the option is listed off the frontier as no_cost, with that reason in the result.
        return None


def _key(first: str, second: str, gain_key: str, field: str) -> str:
    return f"price.{first}.{second}.{gain_key}.{field}"


def _price(report_input: ReportInput, first: str, second: str, gain_key: str, monthly: bool) -> Price:
    """The move from ``first`` (the cheaper) to ``second``, every figure its key's."""
    gain = cite(_key(first, second, gain_key, "gain"), report_input)
    interval = cite_if(_key(first, second, gain_key, "gain_interval"), report_input)
    gap = cite(_key(first, second, gain_key, "cost_gap"), report_input)
    gained, spent = _number(gain), _number(gap)

    def at(field: str) -> CitedFigure:
        return cite(_key(first, second, gain_key, field), report_input)

    per_point = low = high = None
    monthly_per_point = monthly_low = monthly_high = None
    if gained == 0 or gained * spent <= 0:
        ahead = second if gained >= 0 and spent <= 0 else first
        bounds: Bounds = "no_price"
        says = f"no price per point: {_arm(ahead)} is at least as good on both axes"
    else:
        per_point = at("per_point")
        monthly_per_point = at("monthly_per_point") if monthly else None
        if interval is None:
            bounds, says = "unstated", "no interval: the input states none on this gain"
        else:
            low = at("per_point_low")
            monthly_low = at("monthly_per_point_low") if monthly else None
            ends = interval.value
            assert isinstance(ends, tuple)
            near = ends[0] if gained > 0 else ends[1]
            if near * gained <= 0:
                bounds = "unbounded_above"
                says = "unbounded above: the gain's interval reaches zero, so any price above the low end fits"
            else:
                high = at("per_point_high")
                monthly_high = at("monthly_per_point_high") if monthly else None
                bounds, says = "bounded", "both ends read off the gain's interval"
    return Price(
        from_arm=_arm(first),
        to_arm=_arm(second),
        gain=gain,
        gain_interval=interval,
        cost_gap=gap,
        per_point=per_point,
        per_point_low=low,
        per_point_high=high,
        bounds=bounds,
        says=says,
        monthly_cost_gap=at("monthly_cost_gap") if monthly else None,
        monthly_per_point=monthly_per_point,
        monthly_per_point_low=monthly_low,
        monthly_per_point_high=monthly_high,
    )


def _gated(report_input: ReportInput, arm: str) -> OffFrontier:
    """An option the gates or a guardrail removed, with each failing figure as the gate check words it."""
    says = list(gate_check(report_input, arm).fails)
    for guardrail in report_input.guardrails:
        if str(guardrail.get("arm")) == arm and guardrail.get("decision") == "breached":
            named = guardrail.get("dimension") or guardrail.get("measure") or guardrail.get("id")
            says.append(f"guardrail {named} read breached" if named else "a guardrail read breached")
    return OffFrontier(arm=_arm(arm), reason="gated", by=[], says=says or ["removed by the gates"], figures=[])


def _at_least(first: _Option, second: _Option) -> bool:
    """Whether ``first`` costs no more than ``second`` and reads no worse."""
    return first.spend <= second.spend and first.position >= second.position


def _undominated(options: Sequence[_Option]) -> tuple[list[_Option], dict[str, _Option]]:
    """The options no other is at least as good as on both axes while ahead on one; and, for each other, the first
    option (in the input's order) that puts it off. Of two alike on both axes, the first in the input's order stays."""
    kept: list[_Option] = []
    beaten: dict[str, _Option] = {}
    for index, option in enumerate(options):
        for rank, other in enumerate(options):
            if other is option or not _at_least(other, option):
                continue
            alike = other.spend == option.spend and other.position == option.position
            if not alike or rank < index:
                beaten[option.arm] = other
                break
        else:
            kept.append(option)
    return kept, beaten


def _hull(options: Sequence[_Option]) -> tuple[list[_Option], dict[str, tuple[_Option, _Option]]]:
    """The undominated options best at some price per point (the upper hull, cheapest first), and, for each option
    under it, the two frontier options either side of it, whose step passes it by."""
    ordered = sorted(options, key=lambda option: (option.spend, option.position))
    hull: list[_Option] = []
    for option in ordered:
        while len(hull) >= 2 and _passed_by(hull[-2], hull[-1], option):
            hull.pop()
        hull.append(option)
    on = {option.arm for option in hull}
    under: dict[str, tuple[_Option, _Option]] = {}
    for option in ordered:
        if option.arm in on:
            continue
        below = max((held for held in hull if held.spend < option.spend), key=lambda held: held.spend)
        above = min((held for held in hull if held.spend > option.spend), key=lambda held: held.spend)
        under[option.arm] = (below, above)
    return hull, under


def _passed_by(cheaper: _Option, middle: _Option, dearer: _Option) -> bool:
    """Whether a point bought on the way to ``middle`` costs at least what one bought on from it does: then the best
    option moves from ``cheaper`` to ``dearer`` without stopping at ``middle``."""
    into = (middle.spend - cheaper.spend) * (dearer.position - middle.position)
    onward = (dearer.spend - middle.spend) * (middle.position - cheaper.position)
    return into >= onward


def _basis(axis: GainAxis) -> str:
    if axis.basis == "contrast":
        return (
            f"each option's contrast against the control on {axis.key}, the control at zero: a move between the "
            "control and a candidate carries the contrast's interval, a move between two candidates none"
        )
    return f"each option's overall mean on {axis.key}, which carries no paired interval"


def _refuse_unknown(report_input: ReportInput, gain_key: str) -> GainAxis:
    known = list(dict.fromkeys([*report_input.measures, *report_input.judged]))
    if gain_key not in known:
        raise RefusedInput(
            f"gain_key {gain_key!r} names no measure or judged dimension this snapshot holds",
            remedy=f"name one by the key campaign_overview lists: {_listed(known)}",
            parameters=("gain_key",),
            valid_values=known,
        )
    try:
        return gain_axis(report_input, gain_key)
    except UnresolvableReference as refused:
        directed = [
            key
            for key in known
            if key in report_input.judged or report_input.measures[key].higher_is_better is not None
        ]
        raise RefusedInput(
            f"gain_key {gain_key!r} states no direction (higher or lower is better), so no gain is read on it",
            remedy=f"name a measure with a direction, or a judged dimension: {_listed(directed)}",
            parameters=("gain_key",),
            valid_values=directed,
        ) from refused


def price_tradeoff(
    report_input: ReportInput,
    gain_key: str,
    *,
    arm: str | None = None,
    limit: int = PRICE_PAGE,
    offset: int = 0,
) -> PriceTradeoff:
    """The trade between the surviving options priced on one gain axis: the frontier, the price per point at each
    step along it, every option priced against one, and why each other option is off the frontier.

    :param report_input: The snapshot's report input.
    :ptype report_input: ReportInput
    :param gain_key: The gain axis: a measure with a direction, or a judged dimension, by key.
    :ptype gain_key: str
    :param arm: The option every other is priced against; None for the control, where it is priced.
    :ptype arm: str | None
    :param limit: The most options one page covers, in the input's arm order: each priced against that option, or
        listed off the frontier with why.
    :ptype limit: int
    :param offset: The first option the page covers.
    :ptype offset: int
    :return: The trade, priced.
    :rtype: PriceTradeoff
    :raises RefusedInput: The gain key names no measure or judged dimension, or one with no direction; no arm has a
        cost; or ``arm`` names no priced option.
    """
    axis = _refuse_unknown(report_input, gain_key)
    arms = [held.id for held in report_input.arms]
    costs = {held: _cost_or_none(report_input, held) for held in arms}
    if not any(costs.values()):
        raise RefusedInput(
            "price_tradeoff prices each option's cost per 1,000 replies, and this snapshot states no arm's: no "
            "measure in usd_per_1k or usd_per_1m, no dollar measure on the cost axis, and no scoreboard reply_cost_usd",
            remedy="compare the options without a price with options_dominance",
        )
    notes: list[str] = []
    volume = cite_if("meta.volume_per_month", report_input)
    if volume is None:
        notes.append(
            "no monthly volume is declared (meta.volume_per_month): prices are per 1,000 replies, and no monthly "
            "figure is stated"
        )
    survivors = set(surviving_options(report_input))
    off: list[OffFrontier] = []
    options: list[_Option] = []
    for held in arms:
        cost = costs[held]
        if held not in survivors:
            off.append(_gated(report_input, held))
        elif cost is None:
            says = "the cost rule states no cost per 1,000 replies for it"
            off.append(OffFrontier(arm=_arm(held), reason="no_cost", by=[], says=[says], figures=[]))
        elif held not in axis.positions:
            says = f"it holds no figure on {gain_key} that the gain axis reads"
            off.append(OffFrontier(arm=_arm(held), reason="no_gain_figure", by=[], says=[says], figures=[]))
        else:
            options.append(_Option(held, cost, axis.positions[held]))
    priced = [option.arm for option in options]
    if arm is not None and arm not in priced:
        raise RefusedInput(
            f"arm_filter {arm!r} names no option this tool prices: it must survive the gates, hold a cost and hold a "
            f"figure on {gain_key}",
            remedy=f"name a priced option: {_listed(priced)}, or omit arm_filter to price against the control",
            parameters=("arm_filter",),
            valid_values=priced,
        )
    if not survivors:
        notes.append("no option survives the gates: there is nothing to price")
    elif len(options) == 1:
        notes.append("one option is priced: there is no trade to price")
    monthly = volume is not None
    kept, beaten = _undominated(options)
    hull, under = _hull(kept)
    for option in options:
        if option.arm in beaten:
            by = beaten[option.arm]
            first, second = (option, by) if option.spend < by.spend else (by, option)
            off.append(
                OffFrontier(
                    arm=_arm(option.arm),
                    reason="dominated",
                    by=[_arm(by.arm)],
                    says=[f"{_arm(by.arm)} costs no more and reads no worse on {gain_key}"],
                    figures=[
                        cite(_key(first.arm, second.arm, gain_key, "gain"), report_input),
                        cite(_key(first.arm, second.arm, gain_key, "cost_gap"), report_input),
                    ],
                )
            )
        elif option.arm in under:
            below, above = under[option.arm]
            off.append(
                OffFrontier(
                    arm=_arm(option.arm),
                    reason="never_best",
                    by=[_arm(below.arm), _arm(above.arm)],
                    says=[
                        f"a point bought from {_arm(below.arm)} to it costs at least what one bought on to "
                        f"{_arm(above.arm)} does, so the best option moves past it at every price"
                    ],
                    figures=[
                        cite(_key(below.arm, option.arm, gain_key, "per_point"), report_input),
                        cite(_key(option.arm, above.arm, gain_key, "per_point"), report_input),
                    ],
                )
            )
    baseline = arm if arm is not None else control_arm(report_input)
    if baseline is not None and baseline not in priced:
        notes.append(f"{_arm(baseline)} is not priced, so no option is priced against it; name arm_filter")
        baseline = None
    elif baseline is None and options:
        notes.append("no control: name arm_filter to price every option against one")
    window = set(arms[offset : offset + limit])
    end = min(offset + limit, len(arms))
    base = next((option for option in options if option.arm == baseline), None)
    shown = [option for option in options if option.arm in window]
    against = _against(report_input, shown, base, gain_key, monthly)
    off = [row for row in off if row.arm.removeprefix("arm:") in window]
    return PriceTradeoff(
        reads_as=_READS_AS,
        notes=notes,
        gain_key=gain_key,
        better="higher_is_better" if axis.higher_is_better else "lower_is_better",
        basis=_basis(axis),
        volume=volume,
        frontier=[
            FrontierOption(
                arm=_arm(option.arm),
                cost=option.cost,
                monthly_cost=cite(f"cost_per_month.{option.arm}", report_input) if monthly else None,
                level=_level(report_input, gain_key, option.arm),
            )
            for option in hull
        ],
        steps=[
            _price(report_input, cheaper.arm, dearer.arm, gain_key, monthly)
            for cheaper, dearer in zip(hull, hull[1:], strict=False)
        ],
        against_arm=None if baseline is None else _arm(baseline),
        against=against,
        off_frontier=off,
        total=len(arms),
        offset=offset,
        next_offset=end if end < len(arms) else None,
        hybrids=_HYBRIDS,
    )


def _against(
    report_input: ReportInput,
    options: Sequence[_Option],
    baseline: _Option | None,
    gain_key: str,
    monthly: bool,
) -> list[Price]:
    """Every other priced option against the baseline, in the input's order, each move from the cheaper to the
    dearer (the baseline first where the two cost the same)."""
    if baseline is None:
        return []
    moves: list[Price] = []
    for option in options:
        if option.arm == baseline.arm:
            continue
        first, second = (option, baseline) if option.spend < baseline.spend else (baseline, option)
        moves.append(_price(report_input, first.arm, second.arm, gain_key, monthly))
    return moves


def _level(report_input: ReportInput, gain_key: str, arm: str) -> CitedFigure | None:
    """An option's own mean on the gain axis, where the input states one."""
    if gain_key in report_input.judged:
        held = arm in report_input.judged[gain_key].per_arm
        return cite_if(f"judged.{gain_key}.{arm}.mean", report_input) if held else None
    if arm not in report_input.measures[gain_key].per_arm:
        return None
    return cite_if(f"measure.{gain_key}.{arm}.mean", report_input)
