"""The price of a trade: one axis priced in the other, by code, every figure a key.

``price_tradeoff`` does the arithmetic a tradeoff page owes (``docs/eval-report-kit/GENERATOR.md`` §4: price one axis
in the other, at real volume, and say where the best option flips), which the planner otherwise did in prose. Every
number it returns is a figure key (:mod:`~threetears.evals.analysis.analyst.cited`): the ``price.<from>.<to>.<gain>.*``
and ``cost_per_month.<arm>`` families of :mod:`~threetears.evals.analysis.report.figure_keys`, the same within one
slice (``task.<task>.price.…``, ``strata.<stratum>.price.…``, ``cells.<cell>.price.…``), and the frontier's counts.

Design note. What the spec left open and this module decided is marked *(inferred)*.

**The options priced, and the frontier**, are :func:`~threetears.evals.analysis.report.figure_keys.price_frontier`'s,
the one owner of that rule, so the count keys, a ``prices`` block and this tool read one frontier. Only options that
survive the gates are priced: an option a gate or a guardrail removes is never priced beside the others, since a trade
that includes it would weigh a broken policy as a cost. An option a gate has no figure for somewhere, while none it has
fails, is *undecided* (:func:`~threetears.evals.analysis.report_margins.undecided_arms`): not shown to pass, so not
priced, and not removed either. It is listed off the frontier as ``undecided`` with each place the input states no
figure for it, never as removed by the gates, as ``options_dominance`` reads it. Of the survivors, an option is priced
when the cost rule states its cost per 1,000 of what the cost counts and the gain axis places it. An option is dominated when another costs no
more and reads no worse, and is ahead on one of the two. Of the undominated options, ordered by cost, the frontier is
the upper hull: the options that are best at some price per point. Each step from one frontier option to the next is
priced by ``price.<cheaper>.<dearer>.<gain>.per_point``, the price per point at which the best option moves from the one
to the other: GENERATOR §4's flip points. An undominated option under the hull is best at no single price, and is listed
off the frontier with the two step prices that pass it by.

**The two axes.** Cost is cost per 1,000 of what the snapshot's cost counts
(:func:`~threetears.evals.analysis.report.figure_keys.cost_basis`: ``per 1,000 conversations``, ``per 1,000 calls``;
every text that names it says the snapshot's own count), read as exact: the input states no interval on it *(inferred:
the pipeline's cost is tokens at list price, with no sampling error stated)*. The gain axis is the one measure or judged
dimension the caller names (``gain_key``), read toward its better side; a measure that states no direction is refused.
:func:`~threetears.evals.analysis.report.figure_keys.gain_axis` places each arm on it: on its contrast against the
control where every arm holding a figure there holds exactly one; otherwise on its overall mean; and where no arm states
an overall figure, on its figures **pooled** over the slices holding them (``pooled.<gain>.<arm>.*``): weighted by the
traffic share the input states for every slice, else equally. An equal weighting is no traffic mix: the result says so
as its basis and as a caveat in its notes, since a mix would move every price *(inferred: the owner asked for equal
weights stated as such, with a caveat)*.

**Within one slice** (``slice_filter``: a task, a stratum or a cell), each option is placed by its mean there, at the
slice's own cost where the slice states one for every option it places, else at each option's overall cost; the
result's ``cost_basis`` says which. There is no monthly figure within a slice: the declared volume is the whole
campaign's *(inferred)*.

**A price per point, and its interval.** ``per_point`` is the cost gap over the gain, the gain in points (a share's
difference times 100; per unit on an axis not in points). Cost being exact, the interval comes from the gain's. Two
intervals carry through, each labelled:

- **paired** (``per_point_low``, ``per_point_high``): the contrast's interval, on a move between the control and a
  candidate on the contrast basis;
- **unpaired** (``unpaired``: the ``_unpaired`` fields): the gain plus or minus the two options' own half-widths on the
  axis combined as independent, wherever both state one: descriptive and uncorrected, and wider than a paired interval
  where the two share cases. It is what prices a move between two candidates, a move on means, pooled figures or
  within a slice.

Over the part of an interval on the gain's side of zero, the price runs from the cost gap over the end farthest from
zero (low) to the cost gap over the end nearest it (high). When the interval reaches or crosses zero, a point may cost
any amount: the high end is **unbounded**, refused so by its key and said in words. A move where one option is at least
as good on both axes has no price per point: the key refuses it, and the result names which option is ahead on both.

**Against one option.** Beside the steps, every priced option is priced against one: the control, or ``arm_filter``,
each move written from the cheaper option to the dearer.

**A what-if** (``what_if``): an option a gate or guardrail removes, or an undecided one, priced as if it survived, and
labelled so: never a recommendation, its label naming each gate figure it fails (by keys) and each guardrail read
breached, or, for an undecided option, saying it is undecided and naming each place with no figure (``unknown``). It shows the
frontier with that option admitted, each step touching it, and its move against the option the others are priced
against. Its figures are the same price keys, which resolve for any costed option.

**A month's cost.** Only at the declared ``meta.volume_per_month``: ``cost_per_month.<arm>``, and the ``monthly_``
price fields. With no declared volume no monthly figure is stated, and the result says so. A volume the caller
supplies is never used: no key could state a figure on it.

**Paging.** A page covers a window of the options, the frontier's first, cheapest first, then every other in the
input's order: the frontier options in it, the steps starting from one of them, the moves against the baseline and
the options off the frontier among them. Every page states how many options the whole frontier holds and how many
steps it takes (``count.frontier.<gain>``, ``count.steps.<gain>``; overall only), so a page is never read as the
whole frontier.

**Hybrids: not priced.** A modeled hybrid, or a mix nobody ran, is an option nobody measured; no key states its cost
or gain (owner: backlog #657). A routed mix over routable tasks shows in a gate_matrix block's pick column.

**Code's arithmetic, not a call.** The result prices the trade and chooses nothing: where nothing ranked the axes in
advance, which price band holds is the decider's call. No condition here reads what kind of eval a campaign is.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.analysis.analyst.analyses import GateRead, breached_guardrails, gate_reads
from threetears.evals.analysis.analyst.cited import CitedFigure, cite, cite_if
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.cost_unit import per_count
from threetears.evals.analysis.report.figure_keys import (
    POOL_WEIGHT_WORDS,
    SLICE_COST_WORDS,
    PriceFrontier,
    SliceKind,
    cost_basis,
    gain_axis,
    pooled_figures,
    price_frontier,
)
from threetears.evals.analysis.report_decision import control_arm, surviving_options
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_margins import undecided_arms
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
    "not priced: a modeled hybrid or a mix nobody ran is an option nobody measured, and no key states its cost or "
    "gain; name it as a next step. A routed mix over routable tasks shows in a gate_matrix block's pick column"
)

#: How an unpaired interval is labelled.
_UNPAIRED = (
    "unpaired: the gain plus or minus the two options' own half-widths combined as independent; descriptive and "
    "uncorrected, wider than a paired interval where the two share cases"
)

#: How a what-if is labelled, before the gates it fails.
_WHAT_IF = (
    "a what-if, never a recommendation: {arm} is removed by the gates, and is priced here as if it survived; each "
    "figure it fails is under gates"
)

#: How a what-if on an undecided option is labelled: it is not removed, so the label never says it is.
_WHAT_IF_UNDECIDED = (
    "a what-if, never a recommendation: {arm} is undecided (no gate figure fails it, and a gate has no figure for it "
    "somewhere), and is priced here as if it passed; each place with no figure is under unknown"
)

#: Why an undecided option is off the frontier, in words.
_UNDECIDED_SAYS = (
    "undecided: no gate figure fails it, and a gate has no figure for it somewhere, so it is shown neither to pass nor "
    "to fail, and is not priced; each place with no figure is under unknown"
)

#: How an option is named in a refusal's list of what a parameter takes.
_SLICE_KINDS: tuple[SliceKind, ...] = ("task", "strata", "cells")

#: How a move's price interval reads: both ends stated, the high end unbounded, no interval stated, or no price.
Bounds = Literal["bounded", "unbounded_above", "unstated", "no_price"]

#: Why an option is off the frontier.
OffReason = Literal["gated", "undecided", "no_cost", "no_gain_figure", "dominated", "never_best"]

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
    cost: CitedFigure = Field(
        description="Its cost: per 1,000 of what the cost counts, or within a slice its figure there on the cost "
        "measure, as the key states it."
    )
    monthly_cost: CitedFigure | None = Field(description="Its cost a month at the declared volume; None without one.")
    level: CitedFigure | None = Field(description="Its own figure on the gain axis, where the input states one.")


class UnpairedInterval(BaseModel):
    """A move's price read off an unpaired interval on its gain: the two options' own half-widths combined."""

    model_config = _MODEL

    gain_interval: CitedFigure = Field(description="The unpaired interval on the gain.")
    per_point_low: CitedFigure | None = Field(description="The least a point costs, as the interval allows.")
    per_point_high: CitedFigure | None = Field(description="The most a point costs; None where unbounded.")
    monthly_per_point_low: CitedFigure | None = Field(description="Its low end a month, where stated.")
    monthly_per_point_high: CitedFigure | None = Field(description="Its high end a month, where stated and bounded.")
    bounds: Bounds = Field(description="How the price's interval reads: bounded, unbounded_above or no_price.")
    says: str = Field(description="What the interval is, and how the price's ends read, in words.")


class Price(BaseModel):
    """One move priced: from the cheaper option to the dearer, its gain on the axis, and what a point of it costs."""

    model_config = _MODEL

    from_arm: str = Field(description="The cheaper option, as arm:<id>.")
    to_arm: str = Field(description="The dearer option, as arm:<id>.")
    gain: CitedFigure = Field(description="How much better the dearer reads on the gain axis, toward better.")
    gain_interval: CitedFigure | None = Field(description="The paired interval on the gain; None where none is stated.")
    cost_gap: CitedFigure = Field(description="What the dearer costs more per 1,000 of what the cost counts.")
    per_point: CitedFigure | None = Field(description="The cost gap per point gained; None where the move has none.")
    per_point_low: CitedFigure | None = Field(description="The least a point costs, as the paired interval allows.")
    per_point_high: CitedFigure | None = Field(
        description="The most a point costs, as the paired interval allows; None where unbounded or unstated."
    )
    bounds: Bounds = Field(description="How the paired interval reads: bounded, unbounded_above, unstated, no_price.")
    says: str = Field(description="The price's paired interval, or the absence of one, in words.")
    unpaired: UnpairedInterval | None = Field(
        description="The price read off the unpaired interval, where both options state their own half-width."
    )
    monthly_cost_gap: CitedFigure | None = Field(description="The cost gap a month at the declared volume.")
    monthly_per_point: CitedFigure | None = Field(description="The price per point a month at the declared volume.")
    monthly_per_point_low: CitedFigure | None = Field(description="Its paired low end a month, where stated.")
    monthly_per_point_high: CitedFigure | None = Field(description="Its paired high end a month, where bounded.")


class OffFrontier(BaseModel):
    """One option not on the frontier, and why."""

    model_config = _MODEL

    arm: str = Field(description="The option, as arm:<id>.")
    reason: OffReason = Field(
        description="gated (a gate figure fails it, or a guardrail read breached), undecided (no gate figure fails "
        "it and a gate has no figure for it somewhere: neither passed nor removed), no_cost, no_gain_figure, dominated "
        "(another is at least as good on both axes) or never_best (under the frontier: best at no single price per "
        "point)."
    )
    by: list[str] = Field(description="The options that put it off: the one dominating it, or the two passing it.")
    says: list[str] = Field(description="Why, in words: what the gates, or the figures below, show.")
    gates: list[GateRead] = Field(
        description="For a gated option, each gate or guardrail figure it fails on, by the keys that state it."
    )
    figures: list[CitedFigure] = Field(description="The figures that show it, where a figure does.")
    unknown: list[GateRead] = Field(
        default_factory=list,
        description="For a gated or undecided option, each place a gate or guardrail has no figure for it: unknown "
        "there, never among what removes it.",
    )


class WhatIf(BaseModel):
    """An option the gates remove, or an undecided one, priced as if it survived: a labelled what-if, never a
    recommendation."""

    model_config = _MODEL

    arm: str = Field(description="The option, as arm:<id>.")
    label: str = Field(description="What this is: a what-if, never a recommendation, and why the option is out.")
    gates: list[GateRead] = Field(description="Each gate or guardrail figure it fails, by the keys that state it.")
    breached: list[str] = Field(description="Each guardrail the pipeline read breached on it, by name.")
    unknown: list[GateRead] = Field(
        default_factory=list,
        description="Each place a gate or guardrail has no figure for it: what leaves an undecided option out.",
    )
    frontier: list[str] = Field(description="The frontier with it admitted, cheapest first, as arm:<id>.")
    steps: list[Price] = Field(description="Each step of that frontier that starts or ends at it.")
    against: Price | None = Field(description="It priced against the option the others are priced against.")
    off: OffFrontier | None = Field(description="Why it is off the admitted frontier, where it is.")


class PriceTradeoff(BaseModel):
    """A trade priced: the options best at some price, the prices the best option moves at, and every move priced
    against one option."""

    model_config = _MODEL

    reads_as: str = Field(description="What this is: code's arithmetic, which chooses nothing.")
    notes: list[str] = Field(description="What could not be read, and why, and the caveats; first, so a cut keeps it.")
    gain_key: str = Field(description="The gain axis: a measure or judged dimension, by key.")
    better: Literal["higher_is_better", "lower_is_better"] = Field(description="Which way the gain axis is better.")
    basis: str = Field(description="What the gain axis places each option by, and whose cost it reads, in words.")
    within: str | None = Field(description="The slice the trade is priced within, as its keys begin; None overall.")
    volume: CitedFigure | None = Field(description="The declared monthly volume; None where unstated or in a slice.")
    frontier_options: CitedFigure | None = Field(description="How many options the whole frontier holds; overall only.")
    frontier_steps: CitedFigure | None = Field(description="How many steps the whole frontier takes; overall only.")
    frontier: list[FrontierOption] = Field(description="The frontier options in this page's window, cheapest first.")
    steps: list[Price] = Field(
        description="Each move along the frontier starting at an option in this page's window, cheapest first: its "
        "per_point is the price per point at which the best option moves from from_arm to to_arm."
    )
    against_arm: str | None = Field(description="The option the others are priced against, as arm:<id>.")
    against: list[Price] = Field(description="Each priced option on this page, priced against against_arm.")
    off_frontier: list[OffFrontier] = Field(description="Each option on this page off the frontier, and why.")
    what_if: WhatIf | None = Field(
        description="The what-if asked for: an option the gates remove or leave undecided, priced as if it passed."
    )
    total: int = Field(description="The options paged: the frontier's, cheapest first, then every other arm.")
    offset: int = Field(description="The first option this page covers.")
    next_offset: int | None = Field(description="The offset of the next page; None when this is the last.")
    hybrids: str = Field(description="Why a modeled hybrid or a mix nobody ran is not priced here.")


def _say_bounds(gained: float, interval: tuple[float, float]) -> tuple[Bounds, str]:
    """How a price's interval reads off a gain's interval: bounded, or unbounded above where it reaches zero."""
    near = interval[0] if gained > 0 else interval[1]
    if near * gained <= 0:
        return (
            "unbounded_above",
            "unbounded above: the gain's interval reaches zero, so any price above the low end fits",
        )
    return "bounded", "both ends read off the gain's interval"


def _unpaired(
    report_input: ReportInput, frontier: PriceFrontier, first: str, second: str, priced: bool, monthly: bool
) -> UnpairedInterval | None:
    interval = cite_if(frontier.key(first, second, "gain_interval_unpaired"), report_input)
    if interval is None:
        return None
    if not priced:
        return UnpairedInterval(
            gain_interval=interval,
            per_point_low=None,
            per_point_high=None,
            monthly_per_point_low=None,
            monthly_per_point_high=None,
            bounds="no_price",
            says=f"{_UNPAIRED}; the move has no price per point",
        )

    def at(field: str) -> CitedFigure:
        return cite(frontier.key(first, second, field), report_input)

    ends = interval.value
    if not isinstance(ends, tuple):
        raise TypeError(f"{interval.key} reads {ends!r}, not an interval")
    gained = _number(cite(frontier.key(first, second, "gain"), report_input))
    if (ends[1] if gained > 0 else ends[0]) * gained <= 0:
        return UnpairedInterval(
            gain_interval=interval,
            per_point_low=None,
            per_point_high=None,
            monthly_per_point_low=None,
            monthly_per_point_high=None,
            bounds="no_price",
            says=f"{_UNPAIRED}; it does not hold the gain on its own side of zero, so no price is read off it",
        )
    bounds, says = _say_bounds(gained, ends)
    bounded = bounds == "bounded"
    return UnpairedInterval(
        gain_interval=interval,
        per_point_low=at("per_point_low_unpaired"),
        per_point_high=at("per_point_high_unpaired") if bounded else None,
        monthly_per_point_low=at("monthly_per_point_low_unpaired") if monthly else None,
        monthly_per_point_high=at("monthly_per_point_high_unpaired") if monthly and bounded else None,
        bounds=bounds,
        says=f"{_UNPAIRED}; {says}",
    )


def _price(report_input: ReportInput, frontier: PriceFrontier, first: str, second: str, monthly: bool) -> Price:
    """The move from ``first`` (the cheaper) to ``second``, every figure its key's."""
    key = frontier.key
    gain = cite(key(first, second, "gain"), report_input)
    interval = cite_if(key(first, second, "gain_interval"), report_input) if not frontier.within else None
    gap = cite(key(first, second, "cost_gap"), report_input)
    gained, spent = _number(gain), _number(gap)

    def at(field: str) -> CitedFigure:
        return cite(key(first, second, field), report_input)

    per_point = low = high = None
    monthly_per_point = monthly_low = monthly_high = None
    priced = not (gained == 0 or gained * spent <= 0)
    if not priced:
        ahead = second if gained >= 0 and spent <= 0 else first
        bounds: Bounds = "no_price"
        says = f"no price per point: {_arm(ahead)} is at least as good on both axes"
    else:
        per_point = at("per_point")
        monthly_per_point = at("monthly_per_point") if monthly else None
        if interval is None:
            bounds, says = "unstated", "no interval paired by a contrast: the input states none on this gain"
        else:
            ends = interval.value
            if not isinstance(ends, tuple):
                raise TypeError(f"{interval.key} reads {ends!r}, not an interval")
            low = at("per_point_low")
            monthly_low = at("monthly_per_point_low") if monthly else None
            bounds, says = _say_bounds(gained, ends)
            if bounds == "bounded":
                high = at("per_point_high")
                monthly_high = at("monthly_per_point_high") if monthly else None
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
        unpaired=_unpaired(report_input, frontier, first, second, priced, monthly),
        monthly_cost_gap=at("monthly_cost_gap") if monthly else None,
        monthly_per_point=monthly_per_point,
        monthly_per_point_low=monthly_low,
        monthly_per_point_high=monthly_high,
    )


def _gated(report_input: ReportInput, arm: str) -> OffFrontier:
    """An option the gates or a guardrail removed, with each failing figure by the keys that state it, and each place
    with no figure apart."""
    says = ["removed by the gates: each figure it fails on is under gates"]
    says += [f"guardrail {named} read breached" for named in breached_guardrails(report_input, arm)]
    gates = gate_reads(report_input, arm, "fail")
    unknown = gate_reads(report_input, arm, "unknown")
    return OffFrontier(arm=_arm(arm), reason="gated", by=[], says=says, gates=gates, figures=[], unknown=unknown)


def _undecided(report_input: ReportInput, arm: str) -> OffFrontier:
    """An option neither shown to pass nor removed, with each place a gate has no figure for it."""
    unknown = gate_reads(report_input, arm, "unknown")
    return OffFrontier(
        arm=_arm(arm), reason="undecided", by=[], says=[_UNDECIDED_SAYS], gates=[], figures=[], unknown=unknown
    )


def _off(report_input: ReportInput, frontier: PriceFrontier, arm: str) -> OffFrontier | None:
    """Why one option is off the frontier, with the figures that show it; None where it is on it."""
    gain_key = frontier.gain_key
    reason = frontier.unpriced.get(arm)
    if reason == "gated":
        return _gated(report_input, arm)
    if reason == "undecided":
        return _undecided(report_input, arm)
    if reason == "no_cost":
        says = f"the cost rule states no cost {per_count(cost_basis(report_input).noun)} for it"
        return OffFrontier(arm=_arm(arm), reason="no_cost", by=[], says=[says], gates=[], figures=[])
    if reason == "no_gain_figure":
        says = f"it holds no figure on {gain_key} that the gain axis reads"
        return OffFrontier(arm=_arm(arm), reason="no_gain_figure", by=[], says=[says], gates=[], figures=[])
    if arm in frontier.dominated:
        by = frontier.dominated[arm]
        first, second = (arm, by) if frontier.costs[arm] < frontier.costs[by] else (by, arm)
        return OffFrontier(
            arm=_arm(arm),
            reason="dominated",
            by=[_arm(by)],
            says=[f"{_arm(by)} costs no more and reads no worse on {gain_key}"],
            gates=[],
            figures=[
                cite(frontier.key(first, second, "gain"), report_input),
                cite(frontier.key(first, second, "cost_gap"), report_input),
            ],
        )
    if arm in frontier.under:
        below, above = frontier.under[arm]
        return OffFrontier(
            arm=_arm(arm),
            reason="never_best",
            by=[_arm(below), _arm(above)],
            says=[
                f"a point bought from {_arm(below)} to it costs at least what one bought on to {_arm(above)} does, so "
                "the best option moves past it at every price"
            ],
            gates=[],
            figures=[
                cite(frontier.key(below, arm, "per_point"), report_input),
                cite(frontier.key(arm, above, "per_point"), report_input),
            ],
        )
    return None


def _basis(report_input: ReportInput, frontier: PriceFrontier) -> str:
    axis = frontier.axis
    if frontier.within:
        whose = f"; cost is {SLICE_COST_WORDS['slice' if frontier.cost_key else 'overall']}"
        return (
            f"each option's mean on {axis.key} within {frontier.within}, with no paired interval and no monthly "
            f"figure{whose}"
        )
    if axis.basis == "contrast":
        return (
            f"each option's contrast against the control on {axis.key}, the control at zero: a move between the "
            "control and a candidate carries the contrast's interval (paired), a move between two candidates the "
            "unpaired one where both state their own"
        )
    if axis.basis == "pooled":
        weights = {figure.weights for figure in pooled_figures(report_input, axis.key).values()}
        words = "; ".join(POOL_WEIGHT_WORDS[weight] for weight in sorted(weights))
        return (
            f"each option's figures on {axis.key} pooled over the slices holding them ({words}): no option states an "
            f"overall figure; pooled.{axis.key}.<arm> keys read each"
        )
    return f"each option's overall mean on {axis.key}, which carries no paired interval"


def _refuse_unknown(report_input: ReportInput, gain_key: str) -> None:
    known = list(dict.fromkeys([*report_input.measures, *report_input.judged]))
    if gain_key not in known:
        raise RefusedInput(
            f"gain_key {gain_key!r} names no measure or judged dimension this snapshot holds",
            remedy=f"name one by the key campaign_overview lists: {_listed(known)}",
            parameters=("gain_key",),
            valid_values=known,
        )
    try:
        gain_axis(report_input, gain_key)
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


def _slices(report_input: ReportInput) -> dict[str, list[tuple[SliceKind, str]]]:
    """Every slice a trade may be priced within, by the id a caller names it by: each task, each stratum, each cell
    (as ``<row>.<column>``); an id two slices share lists both."""
    named: dict[str, list[tuple[SliceKind, str]]] = {}
    places: list[tuple[SliceKind, str]] = [("task", task.id) for task in report_input.tasks]
    places += [("strata", stratum.id) for stratum in report_input.strata]
    places += [("cells", f"{cell.row}.{cell.column}") for cell in report_input.cells.values()]
    for kind, slice_id in places:
        if (kind, slice_id) not in named.get(slice_id, []):
            named.setdefault(slice_id, []).append((kind, slice_id))
    return named


def _within(report_input: ReportInput, slice_filter: str | None) -> tuple[SliceKind, str] | None:
    if slice_filter is None:
        return None
    named = _slices(report_input)
    found = named.get(slice_filter, [])
    if len(found) != 1:
        unique = [slice_id for slice_id, held in named.items() if len(held) == 1]
        why = "names two slices" if found else "names no task, stratum or cell this snapshot holds"
        raise RefusedInput(
            f"slice_filter {slice_filter!r} {why}",
            remedy=f"name one slice by the id campaign_overview lists: {_listed(unique)}",
            parameters=("slice_filter",),
            valid_values=unique,
        )
    return found[0]


def _refuse_what_if(report_input: ReportInput, what_if: str) -> None:
    gated = [arm.id for arm in report_input.arms if arm.id not in set(surviving_options(report_input))]
    if what_if not in gated:
        survives = what_if in {arm.id for arm in report_input.arms}
        why = "survives the gates, so it is priced already" if survives else "names no option this snapshot holds"
        raise RefusedInput(
            f"what_if {what_if!r} {why}: a what-if prices an option the gates remove or leave undecided",
            remedy=f"name an option the gates remove or leave undecided: {_listed(gated)}, or omit what_if",
            parameters=("what_if",),
            valid_values=gated,
        )


def price_tradeoff(
    report_input: ReportInput,
    gain_key: str,
    *,
    arm: str | None = None,
    slice_filter: str | None = None,
    what_if: str | None = None,
    limit: int = PRICE_PAGE,
    offset: int = 0,
) -> PriceTradeoff:
    """The trade between the surviving options priced on one gain axis: the frontier, the price per point at each
    step along it, every option priced against one, why each other option is off the frontier, and a what-if.

    :param report_input: The snapshot's report input.
    :ptype report_input: ReportInput
    :param gain_key: The gain axis: a measure with a direction, or a judged dimension, by key.
    :ptype gain_key: str
    :param arm: The option every other is priced against; None for the control, where it is priced.
    :ptype arm: str | None
    :param slice_filter: The one task, stratum or cell to price within, by id (a cell as ``<row>.<column>``); None
        prices overall.
    :ptype slice_filter: str | None
    :param what_if: An option the gates remove or leave undecided, to price as a labelled what-if; None for none.
    :ptype what_if: str | None
    :param limit: The most options one page covers: the frontier's first, cheapest first, then the others.
    :ptype limit: int
    :param offset: The first option the page covers.
    :ptype offset: int
    :return: The trade, priced.
    :rtype: PriceTradeoff
    :raises RefusedInput: The gain key names no measure or judged dimension, or one with no direction; no arm has a
        cost; ``arm`` names no priced option; ``slice_filter`` names no one slice; ``what_if`` names no option the
        gates remove or leave undecided.
    """
    _refuse_unknown(report_input, gain_key)
    within = _within(report_input, slice_filter)
    frontier = price_frontier(report_input, gain_key, within=within)
    if not frontier.costs:
        raise RefusedInput(
            "price_tradeoff prices each option's cost per 1,000 of what the cost counts, and this snapshot states no "
            "arm's: no "
            "measure in usd_per_1k or usd_per_1m, no dollar measure on the cost axis, and no scoreboard reply_cost_usd",
            remedy="compare the options without a price with options_dominance",
        )
    if what_if is not None:
        _refuse_what_if(report_input, what_if)
    arms = [held.id for held in report_input.arms]
    priced = list(frontier.priced)
    if arm is not None and arm not in priced:
        raise RefusedInput(
            f"arm_filter {arm!r} names no option this tool prices: it must survive the gates, hold a cost and hold a "
            f"figure on {gain_key}",
            remedy=f"name a priced option: {_listed(priced)}, or omit arm_filter to price against the control",
            parameters=("arm_filter",),
            valid_values=priced,
        )
    notes: list[str] = []
    volume = None if frontier.within else cite_if("meta.volume_per_month", report_input)
    if frontier.within:
        notes.append(f"priced within {frontier.within}: no monthly figure is stated, the volume being the campaign's")
    elif volume is None:
        notes.append(
            "no monthly volume is declared (meta.volume_per_month): prices are "
            f"{per_count(cost_basis(report_input).noun)}, and no monthly figure is stated"
        )
    if frontier.axis.basis == "pooled" and any(
        figure.weights == "equal" for figure in pooled_figures(report_input, gain_key).values()
    ):
        notes.append(
            f"caveat: the gain axis pools each option's figures on {gain_key} over its slices weighted equally, "
            "because the input states no traffic share for every slice; a traffic mix would move every price"
        )
    survivors = set(surviving_options(report_input))
    undecided = undecided_arms(report_input)
    if undecided:
        notes.append(
            f"undecided, so not priced and not removed: {_listed(undecided)}; a gate has no figure for each somewhere "
            "and none it has fails (each off the frontier with its unknown places)"
        )
    if not survivors:
        notes.append(
            "no option is shown to pass the gates: there is nothing to price"
            if undecided
            else "no option survives the gates: there is nothing to price"
        )
    elif len(priced) == 1:
        notes.append("one option is priced: there is no trade to price")
    monthly = volume is not None
    baseline = arm if arm is not None else control_arm(report_input)
    if baseline is not None and baseline not in priced:
        notes.append(f"{_arm(baseline)} is not priced, so no option is priced against it; name arm_filter")
        baseline = None
    elif baseline is None and priced:
        notes.append("no control: name arm_filter to price every option against one")
    order = [*frontier.frontier, *(held for held in arms if held not in frontier.frontier)]
    paged = order[offset : offset + limit]
    window = set(paged)
    end = min(offset + limit, len(order))
    off = [found for held in paged if (found := _off(report_input, frontier, held)) is not None]
    return PriceTradeoff(
        reads_as=_READS_AS,
        notes=notes,
        gain_key=gain_key,
        better="higher_is_better" if frontier.axis.higher_is_better else "lower_is_better",
        basis=_basis(report_input, frontier),
        within=frontier.within or None,
        volume=volume,
        frontier_options=None if frontier.within else cite(f"count.frontier.{gain_key}", report_input),
        frontier_steps=None if frontier.within else cite(f"count.steps.{gain_key}", report_input),
        frontier=[
            _frontier_option(report_input, frontier, held, monthly) for held in frontier.frontier if held in window
        ],
        steps=[
            _price(report_input, frontier, cheaper, dearer, monthly)
            for cheaper, dearer in frontier.steps
            if cheaper in window
        ],
        against_arm=None if baseline is None else _arm(baseline),
        against=_against(report_input, frontier, [held for held in paged if held in priced], baseline, monthly),
        off_frontier=off,
        what_if=None if what_if is None else _what_if(report_input, frontier, what_if, within, baseline, monthly),
        total=len(order),
        offset=offset,
        next_offset=end if end < len(order) else None,
        hybrids=_HYBRIDS,
    )


def _frontier_option(report_input: ReportInput, frontier: PriceFrontier, arm: str, monthly: bool) -> FrontierOption:
    cost = cite(f"cost_per_1k.{arm}", report_input) if not frontier.within else _slice_cost(report_input, frontier, arm)
    return FrontierOption(
        arm=_arm(arm),
        cost=cost,
        monthly_cost=cite(f"cost_per_month.{arm}", report_input) if monthly else None,
        level=_level(report_input, frontier, arm),
    )


def _slice_cost(report_input: ReportInput, frontier: PriceFrontier, arm: str) -> CitedFigure:
    """An option's cost within a slice, as the key the basis reads it by states it: its figure there on the cost
    measure where the slice states every option's own, else its overall cost per 1,000 of what the cost counts."""
    if frontier.cost_key:
        return cite(f"{frontier.within}.{arm}.{frontier.cost_key}.mean", report_input)
    return cite(f"cost_per_1k.{arm}", report_input)


def _against(
    report_input: ReportInput,
    frontier: PriceFrontier,
    options: Sequence[str],
    baseline: str | None,
    monthly: bool,
) -> list[Price]:
    """Every other priced option against the baseline, in the input's order, each move from the cheaper to the
    dearer (the baseline first where the two cost the same)."""
    if baseline is None:
        return []
    moves: list[Price] = []
    for option in options:
        if option == baseline:
            continue
        cheaper = frontier.costs[option] < frontier.costs[baseline]
        first, second = (option, baseline) if cheaper else (baseline, option)
        moves.append(_price(report_input, frontier, first, second, monthly))
    return moves


def _what_if(
    report_input: ReportInput,
    frontier: PriceFrontier,
    what_if: str,
    within: tuple[SliceKind, str] | None,
    baseline: str | None,
    monthly: bool,
) -> WhatIf:
    """The option the gates remove, or an undecided one, priced as if it survived, labelled so."""
    admitted = price_frontier(report_input, frontier.gain_key, within=within, admit=(what_if,))
    off = _off(report_input, admitted, what_if)
    against = None
    if baseline is not None and what_if in admitted.priced and baseline != what_if:
        cheaper = admitted.costs[what_if] < admitted.costs[baseline]
        first, second = (what_if, baseline) if cheaper else (baseline, what_if)
        against = _price(report_input, admitted, first, second, monthly)
    label = _WHAT_IF_UNDECIDED if what_if in undecided_arms(report_input) else _WHAT_IF
    return WhatIf(
        arm=_arm(what_if),
        label=label.format(arm=_arm(what_if)),
        gates=gate_reads(report_input, what_if, "fail"),
        breached=breached_guardrails(report_input, what_if),
        unknown=gate_reads(report_input, what_if, "unknown"),
        frontier=[_arm(held) for held in admitted.frontier],
        steps=[
            _price(report_input, admitted, cheaper, dearer, monthly)
            for cheaper, dearer in admitted.steps
            if what_if in (cheaper, dearer)
        ],
        against=against,
        off=off,
    )


def _level(report_input: ReportInput, frontier: PriceFrontier, arm: str) -> CitedFigure | None:
    """An option's own figure on the gain axis, where the input states one: its mean overall, its pooled figure, or
    its mean within the slice."""
    gain_key = frontier.gain_key
    if frontier.within:
        kind, slice_id = frontier.within.split(".", 1)
        return cite_if(f"{kind}.{slice_id}.{arm}.{gain_key}.mean", report_input)
    if frontier.axis.basis == "pooled":
        return cite_if(f"pooled.{gain_key}.{arm}.mean", report_input)
    if gain_key in report_input.judged:
        held = arm in report_input.judged[gain_key].per_arm
        return cite_if(f"judged.{gain_key}.{arm}.mean", report_input) if held else None
    if arm not in report_input.measures[gain_key].per_arm:
        return None
    return cite_if(f"measure.{gain_key}.{arm}.mean", report_input)
