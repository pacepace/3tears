"""Comparisons whose candidate is not better, read by the advice (:func:`~threetears.evals.analysis.stats.advice_path`)
after every case: what the statistics certificate counts its advice claims on (``stats_certificate.py``).

The advice keeps one promise: if the candidate is not really better, it says switch at most one time in three (the
``easy`` bar of :data:`~threetears.evals.analysis.stats.ADVICE_BARS`), or one time in ten (``hard``), however often
anyone looks. It is read here on the package's own function over paired per-case values, the very path a comparison's
family reads it on (``comparisons._advice_of``), in the order the cases arrive, at every count from two to
:data:`HORIZON`. The worst case for the promise is a candidate exactly as good as the control, and that is every
comparison here, on four families of data:

- **pass/fail**: one trial a case; both arms pass at one share drawn from 0.2 to 0.7 per case;
- **on-ramp**: apps as cases (the on-ramp harness's own model, ``onramp/reading.py``): a base share drawn from 0.1 to
  0.6, each arm's share on the app moved by a normal draw of sd 0.1 and held to 0 to 1, three trials an arm;
- **full swing**: every case a full win or a full loss for the candidate, each half the time: the most spread a
  difference on the range can have;
- **skewed**: the candidate gains a third of the range on three cases in four and loses all of it on the fourth, a
  mean of zero that is not symmetric, where a symmetric test reads a gain.

Each comparison is seeded by its family and index alone (``advice-<family>-<index>``), so a count is reproducible.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Literal

from threetears.evals.analysis.stats import (
    ADVICE_BARS,
    DECLARED_DIRECTION_SHARE,
    SIGNIFICANCE_ALPHA,
    advice_path,
    bet_decided_at,
    bounded_separation_p,
    holm_adjust,
)

#: The families of not-better candidates (the module docstring).
Family = Literal["passfail", "onramp", "swing", "skewed"]

FAMILIES: tuple[Family, ...] = ("passfail", "onramp", "swing", "skewed")

#: The most cases a comparison is read over, after every one from the second.
HORIZON = 100

#: The counts a single read is taken at: one look at each, as a reader who looks once there reads it.
ONCE_AT: tuple[int, ...] = (8, 24, 100)


def pair(family: Family, draw: random.Random) -> tuple[Fraction, Fraction]:
    """One case's (control, candidate) values, each a share of 0 to 1, from ``family``, the two arms alike."""
    if family == "passfail":
        share = draw.uniform(0.2, 0.7)
        return Fraction(int(draw.random() < share)), Fraction(int(draw.random() < share))
    if family == "onramp":
        base = draw.uniform(0.1, 0.6)
        shares = [min(1.0, max(0.0, base + draw.gauss(0.0, 0.1))) for _ in range(2)]
        a, b = (sum(draw.random() < share for _ in range(3)) for share in shares)
        return Fraction(a, 3), Fraction(b, 3)
    if family == "swing":
        return (Fraction(0), Fraction(1)) if draw.random() < 0.5 else (Fraction(1), Fraction(0))
    # Skewed: +1/3 on three cases in four, -1 on the fourth.
    return (Fraction(0), Fraction(1, 3)) if draw.random() < 0.75 else (Fraction(1), Fraction(0))


def comparison(family: Family, index: int) -> tuple[list[Fraction], list[Fraction]]:
    """The control's and the candidate's values over :data:`HORIZON` cases, in the order they arrive."""
    draw = random.Random(f"advice-{family}-{index}")
    pairs = [pair(family, draw) for _ in range(HORIZON)]
    return [a for a, _ in pairs], [b for _, b in pairs]


def advised(family: Family, index: int) -> dict[str, bool]:
    """What the advice said on one comparison at each bar: ``ever<bar>`` whether it said switch at any count from two to
    :data:`HORIZON` (a reader who looks after every case, and acts the first time it says switch), and
    ``once<n>_<bar>`` whether it said switch at ``n`` cases (a reader who looks once there)."""
    control, candidate = comparison(family, index)
    said: dict[str, bool] = {}
    for bar in ADVICE_BARS.values():
        path = advice_path(control, candidate, value_range=(0.0, 1.0), bar=bar)
        assert path is not None
        tag = f"{bar:g}"
        said[f"ever{tag}"] = any(path)
        for n in ONCE_AT:
            said[f"once{n}_{tag}"] = path[n - 1]
    return said


def count(family: Family, indices: Sequence[int]) -> dict[str, int]:
    """How many of the comparisons ``indices`` the advice said switch on, by field (:func:`advised`), and ``n``."""
    totals: dict[str, int] = {"n": len(indices)}
    for index in indices:
        for field, said in advised(family, index).items():
            totals[field] = totals.get(field, 0) + int(said)
    return totals


#: The margin a question declares in the margin claims: five points of a pass rate.
MARGIN = 0.05

#: The families a candidate exactly the margin worse is drawn from: each holds the mean difference at exactly
#: ``-MARGIN`` (the on-ramp's clipped shares would not).
MarginFamily = Literal["passfail", "swing", "skewed"]

MARGIN_FAMILIES: tuple[MarginFamily, ...] = ("passfail", "swing", "skewed")

#: How many arms the several-arms claims read against one control, and the families they are drawn from.
ARMS = 3

ArmsFamily = Literal["passfail", "onramp"]

ARMS_FAMILIES: tuple[ArmsFamily, ...] = ("passfail", "onramp")


def _worse_by_margin(family: MarginFamily, draw: random.Random) -> tuple[Fraction, Fraction]:
    """One case's (control, candidate) values from ``family``, the candidate worse by exactly :data:`MARGIN`."""
    if family == "passfail":
        share = draw.uniform(0.25, 0.7)
        return Fraction(int(draw.random() < share)), Fraction(int(draw.random() < share - MARGIN))
    if family == "swing":
        # A full win half of (1 - margin) of the time: a mean difference of -margin.
        return (Fraction(0), Fraction(1)) if draw.random() < (1 - MARGIN) / 2 else (Fraction(1), Fraction(0))
    # Skewed: +1/3 on a share q of cases and -1 on the rest, q / 3 - (1 - q) = -margin.
    return (Fraction(0), Fraction(1, 3)) if draw.random() < (3 - 3 * MARGIN) / 4 else (Fraction(1), Fraction(0))


def margin_advised(family: MarginFamily, index: int) -> dict[str, bool]:
    """Whether a question with a margin of :data:`MARGIN` advised switch, at any count from two to :data:`HORIZON`, on
    a candidate exactly the margin worse (``ever<bar>``), seeded ``advice-margin-<family>-<index>``."""
    draw = random.Random(f"advice-margin-{family}-{index}")
    pairs = [_worse_by_margin(family, draw) for _ in range(HORIZON)]
    control, candidate = [a for a, _ in pairs], [b for _, b in pairs]
    said: dict[str, bool] = {}
    for bar in ADVICE_BARS.values():
        path = advice_path(control, candidate, value_range=(0.0, 1.0), bar=bar, margin=MARGIN)
        assert path is not None
        said[f"ever{bar:g}"] = any(path)
    return said


def _arms_case(family: ArmsFamily, draw: random.Random) -> list[Fraction]:
    """One case's values for the control and :data:`ARMS` arms, every arm exactly as good as the control."""
    if family == "passfail":
        share = draw.uniform(0.2, 0.7)
        return [Fraction(int(draw.random() < share)) for _ in range(ARMS + 1)]
    base = draw.uniform(0.1, 0.6)
    shares = [min(1.0, max(0.0, base + draw.gauss(0.0, 0.1))) for _ in range(ARMS + 1)]
    return [Fraction(sum(draw.random() < share for _ in range(3)), 3) for share in shares]


#: How many arms the decided-arms claims read against one control.
DECIDED_ARMS: tuple[int, ...] = (3, 5)


def _cases_of_arms(family: ArmsFamily, draw: random.Random, arms: int) -> list[Fraction]:
    """One case's values for the control and ``arms`` arms, every arm exactly as good as the control."""
    if family == "passfail":
        share = draw.uniform(0.2, 0.7)
        return [Fraction(int(draw.random() < share)) for _ in range(arms + 1)]
    base = draw.uniform(0.1, 0.6)
    shares = [min(1.0, max(0.0, base + draw.gauss(0.0, 0.1))) for _ in range(arms + 1)]
    return [Fraction(sum(draw.random() < share for _ in range(3)), 3) for share in shares]


def decided_arms_advised(family: ArmsFamily, index: int, arms: int) -> dict[str, bool]:
    """Whether any of ``arms`` arms, none better than the control, was advised switch at any count from two to
    :data:`HORIZON`, where a standing decision leads the advice as a family reads it: an arm the proof decided better is
    advised switch whatever its evidence. The proof is the question's, each arm's bet staked
    :data:`~threetears.evals.analysis.stats.DECLARED_DIRECTION_SHARE` toward better and Holm across the arms at α, a
    decision standing once made (so one by the last count is one the advice followed from where it was made, and the
    count is of every arm decided better at the last count, or advised switch by its evidence at any count). Seeded
    ``advice-decided-<arms>-<family>-<index>``; ``ever<bar>`` at each bar times the arms."""
    draw = random.Random(f"advice-decided-{arms}-{family}-{index}")
    cases = [_cases_of_arms(family, draw, arms) for _ in range(HORIZON)]
    control = [case[0] for case in cases]
    each = [[case[arm] for case in cases] for arm in range(1, arms + 1)]
    raws = []
    for values in each:
        p = bounded_separation_p(
            control, values, paired=True, value_range=(0.0, 1.0), mixture=True, toward=DECLARED_DIRECTION_SHARE
        )
        raws.append(1.0 if p is None else p)
    adjusted = holm_adjust(raws, max_true=arms)
    improved = []
    for raw, adjusted_p, values in zip(raws, adjusted, each, strict=True):
        if adjusted_p >= SIGNIFICANCE_ALPHA:
            improved.append(False)
            continue
        level = SIGNIFICANCE_ALPHA * raw / adjusted_p if adjusted_p > 0 else SIGNIFICANCE_ALPHA
        held = bet_decided_at(
            control, values, value_range=(0.0, 1.0), toward=DECLARED_DIRECTION_SHARE, alpha=min(1.0, level)
        )
        improved.append(held is not None and held[1])
    said: dict[str, bool] = {}
    for bar in ADVICE_BARS.values():
        paths = [advice_path(control, values, value_range=(0.0, 1.0), bar=bar * arms) for values in each]
        said[f"ever{bar:g}"] = any(decided or any(path or ()) for decided, path in zip(improved, paths, strict=True))
    said["decided"] = any(improved)
    return said


def arms_advised(family: ArmsFamily, index: int) -> dict[str, bool]:
    """Whether any of :data:`ARMS` arms, none better than the control, was advised switch at any count from two to
    :data:`HORIZON`, at each bar times the arms read (``ever<bar>``), seeded ``advice-arms-<family>-<index>``."""
    draw = random.Random(f"advice-arms-{family}-{index}")
    cases = [_arms_case(family, draw) for _ in range(HORIZON)]
    control = [case[0] for case in cases]
    said: dict[str, bool] = {}
    for bar in ADVICE_BARS.values():
        paths = [
            advice_path(control, [case[arm] for case in cases], value_range=(0.0, 1.0), bar=bar * ARMS)
            for arm in range(1, ARMS + 1)
        ]
        said[f"ever{bar:g}"] = any(any(path or ()) for path in paths)
    return said


def count_of(advised_of: Callable[[str, int], dict[str, bool]], family: str, indices: Sequence[int]) -> dict[str, int]:
    """How many of the comparisons ``indices`` ``advised_of`` said switch on, by field, and ``n``."""
    totals: dict[str, int] = {"n": len(indices)}
    for index in indices:
        for field, said in advised_of(family, index).items():
            totals[field] = totals.get(field, 0) + int(said)
    return totals


__all__ = [
    "ARMS",
    "ARMS_FAMILIES",
    "DECIDED_ARMS",
    "FAMILIES",
    "HORIZON",
    "MARGIN",
    "MARGIN_FAMILIES",
    "ONCE_AT",
    "ArmsFamily",
    "Family",
    "MarginFamily",
    "advised",
    "arms_advised",
    "comparison",
    "decided_arms_advised",
    "count",
    "count_of",
    "margin_advised",
    "pair",
]
