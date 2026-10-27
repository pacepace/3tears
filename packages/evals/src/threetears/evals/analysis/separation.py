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
4. for a pass/fail rate with a count, the binomial spread ``sqrt(p(1 - p) / n)``, ``p`` read as a share.

An interval or a half-width is read only at a stated level: a level nobody stated would make the standard error a
guess, so such a statistic is read by its ``sem`` or as a rate, or not at all *(inferred)*. A standard error or a
binomial spread carries no level of its own, so the result is read at the snapshot's stated level where it has one,
else at :data:`DEFAULT_LEVEL` (95%) *(inferred: the convention the pipeline's own intervals use)*. The spread of one
case (``per_case``) is the standard error times the square root of the count, which needs the count; for a rate it
is ``sqrt(p(1 - p))`` directly. An interval that is not symmetric about the mean (a Wilson interval on a rate near an
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

*What a case is.* The count a spread rests on is whatever count the figure states: distinct cases, or trials where a
design repeats a case. A design size read from it counts the same thing, so on a repeated design it is a count of
trials, not of cases; and it reads each trial as independent, which understates the spread where a case's repeats
agree. A caller says which the count counts.

**How often today's design lands on the wrong side** (:func:`wrong_side_share`): if the truth equals today's estimate,
a fresh run of today's size reads its estimate on the other side of the line (or the other arm ahead) with probability
``Phi(-|distance| / se)``.

**What it refuses**, each by raising :class:`NoSpread` with the reason in words: a statistic with no spread to read
(no interval at a stated level, no standard error, and no count for a rate); an estimate exactly on its line, which no
number of cases separates; a per-case spread of zero (every case read alike), which no design size changes. Nothing
here reads what kind of eval a figure came from: every condition is on the numbers given.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

__all__ = [
    "POWER",
    "NoSpread",
    "Spread",
    "cases_to_separate",
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

#: What an arm's standard error was read from.
SpreadBasis = Literal["interval", "half_width", "sem", "binomial"]

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
    :ivar per_case: The spread of one case, ``se * sqrt(n)`` (for a rate, ``sqrt(p(1 - p))``); None without a count.
    :ivar level: The level a result read from this spread is stated at, as a share.
    :ivar basis: What the standard error was read from.
    """

    mean: float
    se: float
    n: float | None
    per_case: float | None
    level: float
    basis: SpreadBasis


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
    :return: The spread.
    :rtype: Spread
    :raises NoSpread: The mean is absent, or nothing it states gives a spread.
    """
    if mean is None or not math.isfinite(mean):
        raise NoSpread("it states no mean")
    count = float(n) if n is not None and n > 0 else None
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
        return Spread(mean, per_case / math.sqrt(count), count, per_case, level, "binomial")
    else:
        if ci is not None or half_width is not None:
            what = "it states an interval at no stated level, and no standard error"
        else:
            what = "it states no interval, half-width or standard error"
        tail = ", and no count to read a rate's spread from" if rate is not None else ", and is no pass/fail rate"
        raise NoSpread(what + tail)
    if not math.isfinite(se) or se < 0:
        raise NoSpread("its interval reads no spread")
    spread = se * math.sqrt(count) if count is not None else None
    return Spread(mean, se, count, spread, level, basis)


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
    :param sided: 1 for an estimate against a line, 2 for two arms against each other.
    :ptype sided: Literal[1, 2]
    :param power: The share of such designs that clear, if the truth equals today's estimate.
    :ptype power: float
    :return: The cases (per arm, for two arms).
    :rtype: int
    :raises NoSpread: The estimate sits on its line, or one case carries no spread.
    """
    if distance == 0 or not math.isfinite(distance):
        raise NoSpread("the estimate sits exactly on its line, so no number of cases separates it")
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
    :return: The share of such designs, 0 to one half.
    :rtype: float
    :raises NoSpread: The estimate sits on its line, or carries no spread.
    """
    if distance == 0 or not math.isfinite(distance):
        raise NoSpread("the estimate sits exactly on its line: a fresh run lands on either side alike")
    if se <= 0 or not math.isfinite(se):
        raise NoSpread("the estimate carries no spread, so a fresh run reads it exactly again")
    return _NORMAL.cdf(-abs(distance) / se)
