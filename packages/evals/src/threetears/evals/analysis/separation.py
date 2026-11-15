"""How far apart two figures read given their spread, and the cases that would settle them: one arithmetic for the
figure keys that state a difference between two arms (``compare.*``), an arm's spread (``spread.*``) and the cases a
reading needs (``settle.*``), :mod:`~threetears.evals.analysis.report.figure_keys`.

Design note. What a figure is read from is the figure-key module's (which statistic, which count, which level); this
module is the arithmetic on those numbers alone, so it imports nothing of the report input. Choices the W2 fix list
left open are marked *(inferred)*.

**An arm's spread** (:func:`stat_spread`). The standard error of one arm's mean, from what its statistic states, in
this order:

1. its interval (``ci``), half its width over the two-sided normal quantile at the level the interval was stated at;
2. its half-width, over the same quantile;
3. its standard error (``sem``), as stated;
4. for a pass/fail rate with a count, the binomial spread ``sqrt(p(1 - p) / n)``, ``p`` read as a share; where the
   count is trials over fewer stated cases, ``n`` is the cases (see *What a case is*).

An overall figure that states no spread of its own and is the plain mean of its target-matrix cells, each stating
one, is read by the standard error derived from theirs (basis ``cells``, :mod:`~threetears.evals.analysis.cell_mean`),
before any binomial spread, as a gate on it reads that interval; the figure-key module reads it, at the cells' stated
level. Its spread of one case is that standard error times the root of the overall's count, so a size read from it
counts cases at today's mix of cells.

An interval or a half-width is read only at a stated level: a level nobody stated would make the standard error a
guess, so such a statistic is read by its ``sem`` or as a rate, or not at all *(inferred)*. A standard error or a
binomial spread carries no level of its own, so the result is read at the snapshot's stated level where it has one,
else at :data:`DEFAULT_LEVEL` (95%) *(inferred: the convention the pipeline's own intervals use)*. The spread of one
case (``per_case``) is the standard error times the square root of the number of cases it rests on, which needs the
count: the count itself where it counts cases, and the distinct cases where it counts trials and the figure states its
cases (see *What a case is*). An interval that is not symmetric about the mean (a Wilson interval on a rate near an
end) is read by its half-width, a normal approximation.

**An arm's own interval** (:func:`spread_interval`): its mean plus and minus its standard error times the two-sided
normal quantile at the spread's level, a normal approximation. Read off a stated interval it restates that interval
symmetrically; read off a pass/fail rate's count it is the interval the input never stated, which a chart of the rate
draws where the input states none. A rate's ends are held to its scale (none to all): the normal approximation runs
past them near an end, and a rate never does. A rate of none or all reads a binomial spread of zero, which gives no
interval: :class:`NoSpread`, since a zero-width interval would read as certain.

**The difference between two arms** (:func:`independent_difference`). Without a paired estimate the two arms are read
as independent samples: the standard error of the difference is the root of the sum of the two squared standard
errors, and its interval is that times the two-sided quantile at the level, about the difference. Two arms that ran
the same cases are correlated case by case, which this ignores; for arms that agree case by case (the usual way) the
independent interval is wider than the paired one, so a separation it shows holds, and one it misses may still be
there. Where the snapshot holds a paired estimate (a contrast of one arm against the other) the figure-key module
reads that instead, and says so.

**The cases that would settle a reading** (:func:`cases_to_separate`). The cases one design needs for an interval at
the level, centred on today's estimate, to sit wholly on today's side of the line (an arm against a gate, floor, bar
or a hypothetical line) or of zero (two arms, each running that many): ``ceil((z * per_case / distance) ** 2)``, with
the two arms' per-case spreads combined as the root of their sum of squares. The size is read on the distance's size
alone, so it settles whichever side today's estimate is on: for an estimate on the passing side it is the size that
would show the line cleared, for one on the failing side the size that would confirm the miss; a caller says which.
If the truth equals today's estimate, half of all designs of that size clear and half do not: it is the size at which
clearing becomes a coin toss, not a promise. :data:`POWER` sizes the design four in five of which clear (``z`` plus
the one-sided quantile of the power) *(inferred: the conventional 80% power, offered beside the coin-toss size so a
reader sees both)*.

*Sidedness.* Against a line the question is one-sided (only one side of the line is asked), so ``z`` is the one-sided
quantile (1.645 at 95%); between two arms it is two-sided (1.96 at 95%). An interval a report prints is two-sided, and
a one-sided bound at a level is a two-sided interval at a lower one (95% one-sided is 90% two-sided), so the one-sided
size is smaller than the size at which the interval the page prints clears. A caller therefore states the sidedness
with every level it reports, and against a line gives the two-sided size beside the one-sided one (``sided=2``).

*What a case is.* A fresh run adds cases, each run as often as today's design runs one, so a size counts cases. The
count a spread rests on may be trials, where a design repeats a case; where the figure also states its distinct cases,
one case's spread is that of its mean over today's repeats, ``se * sqrt(cases)``, and the size counts cases at today's
repeats (the trials it takes are that times the repeats). A stated standard error carries the pipeline's own reading
of repeats. A binomial spread is read on the distinct cases, each counted once whatever its repeats: a case's mean
over its repeats spreads between ``sqrt(p(1 - p) / repeats)`` (repeats independent) and ``sqrt(p(1 - p))`` (repeats
agreeing), and the snapshot holds no per-case outcomes to say where, so the spread is read at the wide end
*(inferred: the conservative bound, so neither an interval nor a size is ever narrower than the cases support; read
as independent, the interval would be the root of the repeats narrower and the size that many times fewer cases)*.
Where the figure states no distinct cases the size counts what its count counts, and a caller says which.

**The cases that lift a reading** (:func:`cases_to_lift`, :func:`figure_cases_to_lift`). A step that adds cases to a
reading keeps the cases it holds, so the chance it lifts is the chance the new cases carry: the size above, less the
cases held, lifts about ``Phi(z(power) * sqrt(N / m))`` of the time (nine in ten where half the design is new). The
plan instead holds the cases fixed, takes the truth as today's estimate and draws the new cases like the held ones,
and finds the fewest that clear with chance :data:`POWER`: a contrast on its per-case values, read by the bounded
interval it is read by (through :func:`~threetears.evals.analysis.stats.bounded_cases_to_clear`); a figure on its
stated spread, read by its t interval. Where none can be read the answer is a state, ``not_available`` with its
reason, never a number.

**A stated difference** (:func:`difference_spread`): a candidate's reading against the control states its change and
an interval at its own level, from paired cases; its standard error is the interval's half-width over the two-sided
quantile at that level, and one case's spread that times the root of the harmonic mean of the arms' cases, so a size
read from it is the cases each arm runs. Against a margin it is settled as any figure against a line (the line minus
the margin, in points).

**How often today's design lands on the wrong side** (:func:`wrong_side_share`): if the truth equals today's estimate,
a fresh run of today's size reads its estimate on the other side of the line (or the other arm ahead) with probability
``Phi(-|distance| / se)``. An estimate exactly on its line reads one half: centred on the line, a fresh run lands on
either side alike, which is the answer to how sure today's reading is, not a refusal.

**What it refuses**, each by raising :class:`NoSpread` with the reason in words: a statistic with no spread to read
(no interval at a stated level, no standard error, and no count for a rate); a design size for an estimate exactly on
its line, since no finite number of cases settles a reading whose truth is today's estimate (a caller says what would:
a truth away from the line); a per-case spread of zero (every case read alike), which no design size changes. Nothing
here reads what kind of eval a figure came from: every condition is on the numbers given.
"""

from __future__ import annotations

import math
import random
import re
from bisect import bisect_right
from collections.abc import Callable, Iterable, Sequence
from itertools import accumulate
from dataclasses import dataclass
from statistics import NormalDist
from typing import Final, Literal

from threetears.evals.analysis.stats import (
    INTERVAL_LEVEL,
    PROJECTION_CASE_CAP,
    bounded_cases_to_clear,
    bounded_difference_interval,
    ci_half_width,
)
from threetears.evals.schema.reading_states import StateOf

__all__ = [
    "POWER",
    "NoSpread",
    "Spread",
    "cases_to_lift",
    "cases_to_separate",
    "figure_cases_to_lift",
    "difference_spread",
    "independent_difference",
    "level_share",
    "spread_interval",
    "stat_spread",
    "wrong_side_share",
]

#: The level a spread is read at when the snapshot states none: the pipeline's own intervals' level.
DEFAULT_LEVEL = 0.95

#: The share of designs that clear, for the second design size a settle result gives beside the coin-toss one.
POWER = 0.8

#: What an arm's standard error was read from: its own interval, half-width or standard error, a rate's binomial
#: spread, or (``cells``) the standard error derived for an overall that is the plain mean of its target-matrix cells,
#: which the figure-key module reads off the cells (:mod:`~threetears.evals.analysis.cell_mean`).
SpreadBasis = Literal["interval", "half_width", "sem", "binomial", "cells"]

#: A level as the input writes one: a share (``0.95``) or a percent (``95%``, ``99.69%``).
_LEVEL = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(%?)\s*$")

_NORMAL = NormalDist()


class NoSpread(ValueError):
    """A figure, or a pair of them, has no spread to read a difference or a design size from; the message says why."""


@dataclass(frozen=True)
class Spread:
    """One arm's mean and its spread, as :func:`stat_spread` reads them.

    :ivar mean: The arm's mean, in its figure's units.
    :ivar se: The standard error of the mean, in the same units.
    :ivar n: The count the spread rests on; None where none is stated.
    :ivar per_case: The spread of one case, ``se * sqrt(cases)`` where the count is trials over stated cases, else
        ``se * sqrt(n)`` (for a rate on cases, ``sqrt(p(1 - p))``); None without a count.
    :ivar level: The level a result read from this spread is stated at, as a share.
    :ivar basis: What the standard error was read from.
    """

    mean: float
    se: float
    n: float | None
    per_case: float | None
    level: float
    basis: SpreadBasis
    cases: float | None = None

    @property
    def repeats(self) -> float | None:
        """The trials per case today, where the count is trials over stated cases; None otherwise."""
        return None if self.cases is None or self.n is None else self.n / self.cases


def level_share(text: str | float | None) -> float | None:
    """A stated level as a share: ``95%`` and ``0.95`` are both 0.95; None where none is stated or it is no level.

    :param text: The level as the input states it.
    :ptype text: str | float | None
    :return: The level, strictly between 0 and 1; None otherwise.
    :rtype: float | None
    """
    if text is None:
        return None
    if isinstance(text, int | float):
        value = float(text)
    else:
        match = _LEVEL.match(text)
        if match is None:
            return None
        value = float(match.group(1)) / (100.0 if match.group(2) else 1.0)
    return value if 0.0 < value < 1.0 else None


def _z_two_sided(level: float) -> float:
    """The normal quantile a two-sided interval at ``level`` spans on each side of its centre (1.96 at 95%).

    :param level: The level, as a share.
    :ptype level: float
    :return: The quantile.
    :rtype: float
    """
    return _NORMAL.inv_cdf(0.5 + level / 2.0)


def _z_one_sided(level: float) -> float:
    """The normal quantile a one-sided bound at ``level`` sits at (1.645 at 95%).

    :param level: The level, as a share.
    :ptype level: float
    :return: The quantile.
    :rtype: float
    """
    return _NORMAL.inv_cdf(level)


def stat_spread(
    mean: float | None,
    *,
    ci: tuple[float, float] | None = None,
    half_width: float | None = None,
    sem: float | None = None,
    n: float | None = None,
    stated_level: float | None = None,
    rate: Literal["share", "percent"] | None = None,
    cases: float | None = None,
) -> Spread:
    """One arm's mean and the standard error it carries, read from the first of its interval, half-width, standard
    error, or (for a pass/fail rate with a count) its binomial spread.

    :param mean: The arm's mean.
    :ptype mean: float | None
    :param ci: Its interval, low then high, where stated.
    :ptype ci: tuple[float, float] | None
    :param half_width: Its interval's half-width, where only that is stated.
    :ptype half_width: float | None
    :param sem: Its standard error, where stated.
    :ptype sem: float | None
    :param n: The count it rests on, where one is known.
    :ptype n: float | None
    :param stated_level: The level its interval was stated at, as a share; None where the input states none.
    :ptype stated_level: float | None
    :param rate: For a pass/fail rate, the scale its mean is on (a share, or a percent); None for any other figure.
    :ptype rate: Literal["share", "percent"] | None
    :param cases: The distinct cases behind a count of trials, where the figure states them and they are fewer: one
        case's spread is then its mean's over today's repeats. None where the count counts cases, or says neither.
    :ptype cases: float | None
    :return: The spread.
    :rtype: Spread
    :raises NoSpread: The mean is absent, or nothing it states gives a spread.
    """
    if mean is None or not math.isfinite(mean):
        raise NoSpread("it states no mean")
    count = float(n) if n is not None and n > 0 else None
    distinct = float(cases) if cases is not None and count is not None and 0 < cases < count else None
    level = stated_level if stated_level is not None else DEFAULT_LEVEL
    se: float | None = None
    basis: SpreadBasis
    if ci is not None and stated_level is not None:
        se, basis = (float(ci[1]) - float(ci[0])) / 2.0 / _z_two_sided(stated_level), "interval"
    elif half_width is not None and stated_level is not None:
        se, basis = float(half_width) / _z_two_sided(stated_level), "half_width"
    elif sem is not None:
        se, basis = float(sem), "sem"
    elif rate is not None and count is not None:
        scale = 100.0 if rate == "percent" else 1.0
        share = mean / scale
        if not 0.0 <= share <= 1.0:
            raise NoSpread(f"its mean is no rate between none and all ({rate} scale), so no binomial spread is read")
        per_case = math.sqrt(share * (1.0 - share)) * scale
        return Spread(mean, per_case / math.sqrt(distinct or count), count, per_case, level, "binomial", distinct)
    else:
        if ci is not None or half_width is not None:
            what = "it states an interval at no stated level, and no standard error"
        else:
            what = "it states no interval, half-width or standard error"
        tail = (
            ", and no count to read a rate's spread from"
            if rate is not None
            else ", and is no share of counted trials (a pass/fail rate) to read a binomial spread on"
        )
        raise NoSpread(what + tail)
    if not math.isfinite(se) or se < 0:
        raise NoSpread("its interval reads no spread")
    spread = se * math.sqrt(distinct or count) if count is not None else None
    return Spread(mean, se, count, spread, level, basis, distinct)


def difference_spread(
    change: float, interval: tuple[float, float], level: float | None, counts: tuple[float, ...]
) -> Spread:
    """The spread of a stated difference between an arm and the control (a reading's change), read from its own
    interval at its stated level, with the cases each arm ran.

    One case's spread is the standard error times the root of the harmonic mean of the arms' cases: each arm running
    that many reads the same standard error where the two arms' spreads are alike, so a size read from it counts the
    cases each arm runs *(inferred: the reading states no per-arm spread to weigh the two by)*.

    :param change: The difference, in its own units (points for a rate).
    :ptype change: float
    :param interval: Its interval, low then high, in the same units.
    :ptype interval: tuple[float, float]
    :param level: The interval's stated level, as a share; None where none is stated.
    :ptype level: float | None
    :param counts: The cases each arm ran; empty where none is stated.
    :ptype counts: tuple[float, ...]
    :return: The spread, its count the harmonic mean of the arms' cases.
    :rtype: Spread
    :raises NoSpread: No level is stated, or the interval reads no spread.
    """
    if level is None:
        raise NoSpread("its interval is at no stated level")
    se = (float(interval[1]) - float(interval[0])) / 2.0 / _z_two_sided(level)
    if not math.isfinite(se) or se < 0:
        raise NoSpread("its interval reads no spread")
    usable = [float(count) for count in counts if count > 0]
    cases = len(usable) / sum(1.0 / count for count in usable) if usable else None
    per_case = se * math.sqrt(cases) if cases is not None else None
    return Spread(float(change), se, cases, per_case, level, "interval")


def spread_interval(spread: Spread, *, rate: Literal["share", "percent"] | None = None) -> tuple[float, float]:
    """An arm's interval at its spread's level: its mean plus and minus the two-sided normal quantile times its standard
    error, a rate's ends held to its scale (this module's An arm's own interval).

    :param spread: The arm's spread (:func:`stat_spread`).
    :ptype spread: Spread
    :param rate: For a pass/fail rate, the scale its mean is on (a share, or a percent); None for any other figure.
    :ptype rate: Literal["share", "percent"] | None
    :return: The interval, low then high, in the mean's units.
    :rtype: tuple[float, float]
    :raises NoSpread: A rate's binomial spread is zero (a rate of none or all).
    """
    if spread.basis == "binomial" and spread.se == 0:
        raise NoSpread("its rate is none or all, so its binomial spread is zero and reads no interval")
    half = _z_two_sided(spread.level) * spread.se
    low, high = spread.mean - half, spread.mean + half
    if rate is None:
        return low, high
    top = 100.0 if rate == "percent" else 1.0
    return max(low, 0.0), min(high, top)


def independent_difference(first: Spread, second: Spread, level: float) -> tuple[float, float]:
    """The difference of two independent arms' means, first minus second, and its interval's half-width at ``level``.

    :param first: The first arm's spread.
    :ptype first: Spread
    :param second: The second arm's.
    :ptype second: Spread
    :param level: The level, as a share.
    :ptype level: float
    :return: The difference, and the half-width of its interval.
    :rtype: tuple[float, float]
    """
    return first.mean - second.mean, _z_two_sided(level) * math.hypot(first.se, second.se)


def cases_to_separate(per_case: float, distance: float, level: float, *, sided: Literal[1, 2], power: float) -> int:
    """The cases a design needs for an interval at ``level``, centred on today's estimate, to clear its line, with the
    chance ``power`` that one does if the truth equals today's estimate (0.5: the coin-toss size).

    :param per_case: The spread of one case (two arms: the root of the sum of their squares), in the figure's units.
    :ptype per_case: float
    :param distance: Today's estimate's distance from the line (two arms: their difference), in the same units.
    :ptype distance: float
    :param level: The level, as a share.
    :ptype level: float
    :param sided: 1 for a one-sided bound at ``level``; 2 for an interval stated two-sided at ``level``: two arms
        against each other, or an estimate whose two-sided interval must clear its line (what a page prints and what
        a gate's borderline or an unsettled reading is lifted by).
    :ptype sided: Literal[1, 2]
    :param power: The share of such designs that clear, if the truth equals today's estimate.
    :ptype power: float
    :return: The cases (per arm, for two arms).
    :rtype: int
    :raises NoSpread: The estimate sits on its line, or one case carries no spread.
    """
    if distance == 0 or not math.isfinite(distance):
        raise NoSpread(
            "the estimate sits exactly on its line, so no finite number of cases settles it at today's estimate: a "
            "run of any size centred on the line lands on either side alike"
        )
    if per_case <= 0 or not math.isfinite(per_case):
        raise NoSpread("every case reads alike (no spread per case), so no design size changes what it reads")
    z = (_z_one_sided(level) if sided == 1 else _z_two_sided(level)) + (_z_one_sided(power) if power != 0.5 else 0.0)
    return math.ceil((z * per_case / abs(distance)) ** 2)


def wrong_side_share(distance: float, se: float) -> float:
    """How often a fresh design of today's size reads its estimate on the other side of the line (or the other arm
    ahead), if the truth equals today's estimate: ``Phi(-|distance| / se)``.

    :param distance: Today's estimate's distance from the line, or the two arms' difference.
    :ptype distance: float
    :param se: Its standard error at today's size.
    :ptype se: float
    :return: The share of such designs, 0 to one half; one half for an estimate exactly on its line.
    :rtype: float
    :raises NoSpread: The distance is not a number, or the estimate carries no spread.
    """
    if not math.isfinite(distance):
        raise NoSpread("the estimate's distance from its line is not a number")
    if se <= 0 or not math.isfinite(se):
        raise NoSpread("the estimate carries no spread, so a fresh run reads it exactly again")
    return _NORMAL.cdf(-abs(distance) / se)


#: How near its line, as a share of the scale its values are on, an estimate reads as on it: a distance of rounding,
#: at which no finite count of cases lifts it.
_ON_THE_LINE: Final = 1e-9

#: How close the search for the cases that lift a reading comes to the fewest, as a share of the count: a plan needs
#: no exact count in the thousands, and each count it tries reads intervals over that many cases.
_LIFT_RESOLUTION: Final = 0.01

#: How far the tilt that moves a draw's mean is searched, in the pairs' own range of differences, and in how many
#: halvings: past the bound the tilted draw is every pair at one end.
_TILT_BOUND: Final = 60.0
_TILT_STEPS: Final = 60

#: The seeded extends the reach's own spread is read on (:func:`_reach_noise`).
_DRAWS: Final = 16

#: Too few cases to read: fewer than two a side, no count, or no count up to the search's cap (an estimate on its
#: line, or so near it) at which the reading clears.
_TOO_FEW: Final = StateOf(state="not_available", reason="too_few")

#: Every case read alike, on a reading whose interval narrows only with its spread: no added case changes it.
_NO_VARIATION: Final = StateOf(state="not_available", reason="no_variation")

#: No count says how many cases the figure rests on.
_NO_COUNT: Final = StateOf(state="not_available", reason="not_recorded")


def _spread_of(values: Sequence[float]) -> float:
    """The standard deviation of ``values`` as a distribution (over ``n``): the spread of one value drawn from them,
    which is how new cases are drawn like the held ones."""
    centre = math.fsum(values) / len(values)
    return math.sqrt(math.fsum((value - centre) ** 2 for value in values) / len(values))


def _fewest(lifts: Callable[[int], bool], fixed: int, cap: int) -> int | None:
    """The fewest cases in all, above ``fixed``, at which ``lifts`` holds: doubling from the cases held, then bisecting
    to within :data:`_LIFT_RESOLUTION` of the count. None where even ``cap`` does not.

    Between the cases held and twice them the cushion a projection keeps for the new cases grows with the count while
    the interval's reach falls, so the test can fail, then hold, then fail again on the way up; past twice the cases
    held both fall. Doubling from twice the cases held and bisecting below the first count that holds finds the one
    place it starts to hold for good: below it, the cases held already read past the line unsettled and the cushion
    only adds to that.
    """
    low = fixed
    high = min(max(2 * fixed, fixed + 1, 2), cap)
    while not lifts(high):
        if high >= cap:
            return None
        low, high = high, min(2 * high, cap)
    while high - low > max(1, int(high * _LIFT_RESOLUTION)):
        middle = (low + high) // 2
        if lifts(middle):
            high = middle
        else:
            low = middle
    return high


def _tilted(differences: Sequence[float], base: Sequence[float], target: float) -> list[float]:
    """Each difference's chance under the distribution closest to ``base`` (the chances the draws are made with) whose
    mean is ``target``: its exponential tilt, held to the differences' own range. A count drawn from it is a count
    drawn like the held ones whose mean sits at ``target``, with the spread such a draw has.

    :return: The chances, aligned with ``differences``, summing to one.
    """
    low, high = min(differences), max(differences)
    if high - low <= 0:
        return list(base)
    centre = math.fsum(chance * difference for chance, difference in zip(base, differences, strict=True))

    def weights(theta: float) -> list[float]:
        exponents = [theta * (difference - centre) / (high - low) for difference in differences]
        top = max(exponents)
        raw = [chance * math.exp(exponent - top) for chance, exponent in zip(base, exponents, strict=True)]
        total = math.fsum(raw)
        return [weight / total for weight in raw]

    def mean_at(theta: float) -> float:
        return math.fsum(weight * difference for weight, difference in zip(weights(theta), differences, strict=True))

    # The tilted mean rises with theta from the least difference to the greatest, so it is bisected on theta; a
    # target past either end is held at that end, since no draw like the held ones averages past it.
    lower, upper = -_TILT_BOUND, _TILT_BOUND
    for _ in range(_TILT_STEPS):
        middle = (lower + upper) / 2.0
        if mean_at(middle) < target:
            lower = middle
        else:
            upper = middle
    return weights((lower + upper) / 2.0)


def _like(pairs: Iterable[tuple[float, float]]) -> tuple[list[tuple[float, float]], list[float]]:
    """The distinct pairs among ``pairs``, in the order of their differences, and each one's share of them: the
    distribution new pairs are drawn from."""
    counts: dict[tuple[float, float], int] = {}
    for pair in pairs:
        counts[pair] = counts.get(pair, 0) + 1
    distinct = sorted(counts, key=lambda pair: (pair[1] - pair[0], pair))
    total = sum(counts.values())
    return distinct, [counts[pair] / total for pair in distinct]


def _drawn(pairs: Sequence[tuple[float, float]], chances: Sequence[float], count: int) -> list[tuple[float, float]]:
    """``count`` pairs read off ``chances`` at evenly spaced quantiles, ``pairs`` in the order of their differences:
    the draw of that many with no luck in it."""
    drawn: list[tuple[float, float]] = []
    cumulative, at = chances[0], 0
    for step in range(count):
        quantile = (step + 0.5) / count
        while quantile > cumulative and at < len(pairs) - 1:
            at += 1
            cumulative += chances[at]
        drawn.append(pairs[at])
    return drawn


def _reach_noise(
    held: Sequence[tuple[float, float]],
    like: Sequence[tuple[float, float]],
    base: Sequence[float],
    new: int,
    value_range: tuple[float, float],
    confidence: float,
) -> float:
    """How far the bounded interval's reach below its estimate varies between readings after an extend, beyond what
    goes with the estimate itself: over :data:`_DRAWS` seeded extends (the held pairs and ``new`` drawn like them, in a
    seeded order), the spread of the reach about its straight-line fit on the estimate.

    Two things move the reach that the estimate does not carry. The order the reading bets in is fixed by its cases'
    ids, and new cases' ids fall among the held ones at random, so it is a draw; and on a few distinct values (a pass
    rate's pairs, which mostly agree) how many of the rare ones are drawn moves the interval's width more than its
    centre. What does go with the estimate (a draw nearer the line reading narrower on bounded values) is read on the
    critical draw instead (:func:`cases_to_lift`), so the fit keeps it from being counted twice."""
    total = len(held) + new
    cumulative = list(accumulate(base))
    estimates, reaches = [], []
    for seed in range(_DRAWS):
        drawn = random.Random(f"lift-draw-{seed}-{total}")
        design = [*held, *(like[min(bisect_right(cumulative, drawn.random()), len(like) - 1)] for _ in range(new))]
        drawn.shuffle(design)
        first, second = [a for a, _ in design], [b for _, b in design]
        interval = bounded_difference_interval(
            first, second, paired=True, value_range=value_range, confidence=confidence
        )
        if interval is None:
            return 0.0
        estimate = (math.fsum(second) - math.fsum(first)) / total
        estimates.append(estimate)
        reaches.append(estimate - interval[0])
    centre, middle = math.fsum(estimates) / _DRAWS, math.fsum(reaches) / _DRAWS
    across = math.fsum((estimate - centre) ** 2 for estimate in estimates)
    slope = (
        math.fsum((e - centre) * (r - middle) for e, r in zip(estimates, reaches, strict=True)) / across
        if across > 0
        else 0.0
    )
    residuals = [r - middle - slope * (e - centre) for e, r in zip(estimates, reaches, strict=True)]
    return math.sqrt(math.fsum(value**2 for value in residuals) / (_DRAWS - 2))


def _crossed_spread(control: Sequence[float], contrast: Sequence[float], total: int) -> float:
    """The spread of an unpaired contrast's estimate once read paired over ``total`` pairs, the held cases cross-filled
    first (:func:`cases_to_lift`).

    Of the first ``c = min(total, held)`` pairs, each side's share of the held cases is cross-filled (in expectation,
    ``c`` in proportion to each side's held count): a cross-filled pair keeps its held value and draws the other
    side's, so it carries the other side's spread; and where fewer than every held case of a side is run, which ones
    are is a draw without replacement from them, which carries that side's spread times the share left out. The
    ``total - c`` pairs past them draw both halves.

    :return: The standard deviation of the estimate at ``total`` pairs, in the values' units.
    """
    count_a, count_b = len(control), len(contrast)
    var_a, var_b = _spread_of(control) ** 2, _spread_of(contrast) ** 2
    crossed = min(total, count_a + count_b)
    of_a, of_b = crossed * count_a / (count_a + count_b), crossed * count_b / (count_a + count_b)
    variance = (
        of_a * var_b
        + of_b * var_a
        + of_a * var_a * (count_a - of_a) / (count_a - 1)
        + of_b * var_b * (count_b - of_b) / (count_b - 1)
        + (total - crossed) * (var_a + var_b)
    )
    return math.sqrt(max(variance, 0.0)) / total


def cases_to_lift(
    control: Sequence[float],
    contrast: Sequence[float],
    *,
    paired: bool,
    value_range: tuple[float, float],
    confidence: float,
    line: float,
    power: float = POWER,
    cap: int = PROJECTION_CASE_CAP,
) -> int | StateOf:
    """The pairs to add to a contrast so that, read again by its bounded interval at ``confidence``, the interval
    clears ``line`` on today's side with chance ``power``, if the truth is today's estimate and the cases held stay as
    they are.

    **What is fixed.** The pairs held are kept, and new pairs are drawn like them, so their truth is today's estimate.
    At ``N = n0 + m`` pairs (``n0`` held, ``m`` new) two things vary: the estimate, with spread ``s * sqrt(m) / N``
    (``s`` one drawn pair's), and the interval's reach beyond what goes with the estimate (:func:`_reach_noise`: the
    order the reading bets in, and on a few distinct values how many rare ones are drawn). The reading lifts with
    chance ``power`` where it lifts at the **critical outcome**: the shortfall ``z(power)`` times their joint spread,
    split between them by their spreads. The estimate's part is a **critical draw**: ``m`` pairs read evenly off the
    distribution of the held pairs tilted to that mean (:func:`_tilted`, :func:`_drawn`), so the spread a draw with
    that mean has is in it (on bounded values a draw nearer the line reads a narrower interval, which a reach read at
    today's estimate misses); the reach's part is added to it. The held pairs and the
    critical draw are read by the bounded interval itself, at ``N``
    (:func:`~threetears.evals.analysis.stats.bounded_cases_to_clear`, the room the critical draw's estimate less the
    line, its cap ``N``). The fewest such ``N`` is found as :func:`_fewest` says. Sizing the whole design to clear at
    ``power`` and subtracting the held cases over-asks: the held half carries no chance, so such a plan lifts about
    ``Phi(z(power) * sqrt(N / m))`` of the time, 88% at ``m = N / 2``.

    **An unpaired contrast** (each side's own cases, sharing fewer than two) is read paired once each side runs the
    cases only the other ran: no pair is held whole, so ``n0`` is 0. A cross-filled pair keeps its held half and draws
    the other like that side's cases; past the cases held, a pair draws both halves. So the estimate's spread at ``N``
    counts the cross-filled pairs' new halves, which held cases among them are run where the plan runs fewer than
    every one, and the fully new pairs' two halves (:func:`_crossed_spread`), at no correlation between the arms, which
    is the most a cross-filled pair can vary; arms that agree case by case lift more often.

    **Today's side.** An estimate below its line lifts by its interval's high end clearing below it: the sides are read
    swapped, which mirrors the interval.

    :param control: The control's per-case values, aligned with ``contrast`` when ``paired``.
    :ptype control: Sequence[float]
    :param contrast: The arm's per-case values, oriented so higher is better.
    :ptype contrast: Sequence[float]
    :param paired: Whether the two are one-to-one on the same cases.
    :ptype paired: bool
    :param value_range: The reading's declared inclusive bounds.
    :ptype value_range: tuple[float, float]
    :param confidence: The two-sided level the reading is read at.
    :ptype confidence: float
    :param line: The line the difference (contrast minus control) must clear, in the values' units.
    :ptype line: float
    :param power: The chance of clearing, if the truth is today's estimate.
    :ptype power: float
    :param cap: The most pairs, in all, searched.
    :ptype cap: int
    :return: The pairs to add, at least one; or, ``not_available``: ``too_few`` with fewer than two values a side, an
        estimate on its line, or no count up to ``cap`` that lifts it.
    :rtype: int | StateOf
    :raises ValueError: A value lies outside ``value_range``.
    """
    if len(control) < 2 or len(contrast) < 2:
        return _TOO_FEW
    a, b = [float(value) for value in control], [float(value) for value in contrast]
    if any(not value_range[0] <= value <= value_range[1] for value in (*a, *b)):
        raise ValueError("a per-case value lies outside the reading's declared range")
    estimate = math.fsum(b) / len(b) - math.fsum(a) / len(a)
    distance = estimate - line
    if abs(distance) <= _ON_THE_LINE * max(1.0, value_range[1] - value_range[0]):
        return _TOO_FEW
    if distance < 0:
        a, b, estimate, distance = b, a, -estimate, -distance
    held = list(zip(a, b, strict=True)) if paired else []
    like, base = _like(held if paired else ((x, y) for x in a for y in b))
    differences = [y - x for x, y in like]
    spread = _spread_of([y - x for x, y in held]) if paired else 0.0
    z = _z_one_sided(power)
    fixed = len(held)

    def lifts(total: int) -> bool:
        new = total - fixed
        if total < 2 or new < 1:
            return False
        # The estimate at ``total`` and the interval's reach both vary, the one with the new pairs drawn, the other
        # with the order the reading bets in; the critical outcome splits the shortfall between them by their spreads.
        estimate_spread = spread * math.sqrt(new) / total if paired else _crossed_spread(a, b, total)
        order_spread = _reach_noise(held, like, base, new, value_range, confidence)
        both = math.hypot(estimate_spread, order_spread)
        shift = z * estimate_spread**2 / both if both > 0 else 0.0
        allowance = z * order_spread**2 / both if both > 0 else 0.0
        drawn = _drawn(like, _tilted(differences, base, estimate - shift * total / new), new)
        design = [*held, *drawn]
        first, second = [x for x, _ in design], [y for _, y in design]
        room = (math.fsum(second) - math.fsum(first)) / total - (estimate - distance) - allowance
        if room <= 0:
            return False
        found = bounded_cases_to_clear(
            first, second, paired=True, value_range=value_range, confidence=confidence, room=room, cap=total
        )
        return found is not None and found <= total

    total = _fewest(lifts, fixed, cap)
    return _TOO_FEW if total is None else total - fixed


def figure_cases_to_lift(
    spread: Spread, distance: float, *, power: float = POWER, cap: int = PROJECTION_CASE_CAP
) -> int | StateOf:
    """The cases to add to one arm's figure so that its interval, two-sided at the spread's level, clears its line on
    today's side with chance ``power``, if the truth is today's estimate and the cases held stay as they are: the rule
    of :func:`cases_to_lift` in one-sample form.

    A figure states its interval, not its cases' values, so its reach at ``N`` cases is its own rule's at the spread
    one case carries: the t interval a mean is stated on (:func:`~threetears.evals.analysis.stats.ci_half_width`), at
    the spread's level by the normal quantiles' ratio where that is not the nominal one.

    :param spread: The figure's spread (:func:`stat_spread`), with the cases it rests on.
    :ptype spread: Spread
    :param distance: Today's estimate's distance from its line, in the figure's units.
    :ptype distance: float
    :param power: The chance of clearing, if the truth is today's estimate.
    :ptype power: float
    :param cap: The most cases, in all, searched.
    :ptype cap: int
    :return: The cases to add, at least one; or, ``not_available``: ``not_recorded`` where no count says how many
        cases the figure rests on, ``too_few`` below two of them, on its line, or past ``cap``; ``no_variation``
        where every case read alike, so no added case narrows its interval.
    :rtype: int | StateOf
    """
    held_count = spread.cases if spread.cases is not None else spread.n
    if held_count is None or spread.per_case is None:
        return _NO_COUNT
    held = math.ceil(held_count)
    if held < 2 or not math.isfinite(distance):
        return _TOO_FEW
    if abs(distance) <= _ON_THE_LINE * max(1.0, abs(spread.mean)):
        return _TOO_FEW
    if spread.per_case <= 0:
        return _NO_VARIATION
    # New cases are drawn like the held ones: one case's spread among them, not the estimate of the spread of the
    # cases they were drawn from (which divides by one fewer).
    per_case, gap = spread.per_case * math.sqrt((held - 1) / held), abs(distance)
    scale = _z_two_sided(spread.level) / _z_two_sided(INTERVAL_LEVEL)
    z = _z_one_sided(power)

    def lifts(total: int) -> bool:
        reach = ci_half_width(per_case / math.sqrt(total), total)
        return reach is not None and reach * scale + z * per_case * math.sqrt(total - held) / total <= gap

    total = _fewest(lifts, held, cap)
    return _TOO_FEW if total is None else total - held
