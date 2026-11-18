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
from collections.abc import Sequence
from fractions import Fraction
from typing import Literal

from threetears.evals.analysis.stats import ADVICE_BARS, advice_path

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


__all__ = ["FAMILIES", "HORIZON", "ONCE_AT", "Family", "advised", "comparison", "count", "pair"]
