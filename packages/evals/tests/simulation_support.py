"""Seeded simulation support: data with a known truth, the Monte-Carlo arithmetic, and reference answers.

The statistics tests that pin an output on a fixed input say the arithmetic is what it was. They cannot
say the number is RIGHT: that an interval labelled 95% covers the truth 95% of the time, that a test at
α=0.05 calls a difference that is not there 5% of the time, or that a family correction holds the
family's error at α. Those are properties of a procedure over many draws, so they are checked by
drawing, from a generator whose truth is known, at the sample sizes the engine sees: 2–15 cases and
1–5 repeats (the engine's default ``k`` is 3; a classifier bank runs 8–20 cases).

Three parts, each usable on its own:

- **Generators** (:func:`draw_clustered`, :func:`draw_paired_arms`, :func:`draw_binary_paired_arms`,
  :func:`draw_confusion`, :func:`draw_rater_pairs`). Every one takes a :class:`random.Random`, so a
  test owns its seed, and every one documents the truth it was drawn around.
- **Monte-Carlo bounds** (:func:`at_most`, :func:`at_least`, :func:`within`). A simulated rate is an
  estimate; its tolerance band is derived from its own standard error, never chosen.
- **Reference answers** (:func:`student_t_two_sided_mass`, :func:`student_t_critical`,
  :func:`paired_t_power`, :func:`hedges_correction`). Computed here from closed forms that share no code
  with the engine's own (its Student-t runs through the incomplete beta function; these through the
  integer-df trigonometric series and a direct integral), so a reference cannot agree with the engine by
  inheriting its mistake.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

from threetears.evals.analysis.stats import (
    SIGNIFICANCE_ALPHA,
    difference_interval,
    exact_decimal,
    holm_adjust,
    paired_equivalence,
    separation_p,
)
from threetears.evals.contracts.metrics import confusion_cell

__all__ = [
    "TOLERANCE_Z",
    "ClusteredDesign",
    "FamilyVerdict",
    "at_least",
    "at_most",
    "case_means",
    "family_verdicts",
    "draw_binary_paired_arms",
    "draw_clustered",
    "draw_confusion",
    "draw_paired_arms",
    "draw_rater_pairs",
    "hedges_correction",
    "monte_carlo_se",
    "paired_t_power",
    "student_t_critical",
    "student_t_two_sided_mass",
    "within",
]

# ---------------------------------------------------------------------------------------------------
# Monte-Carlo bounds
# ---------------------------------------------------------------------------------------------------

#: How many Monte-Carlo standard errors a simulated rate may sit from the value it is checked against.
#: Four: a one-sided normal tail of about 3e-5 per check, so a suite of ~60 checks raises a false alarm
#: under a changed seed well under 1% of the time, while a rate one percentage point off at a few
#: thousand replicates still lands outside the band.
TOLERANCE_Z = 4.0


def monte_carlo_se(rate: float, replicates: int) -> float:
    """The standard error of a simulated rate whose true value is ``rate``: ``sqrt(rate (1 - rate) / replicates)``.

    Args:
        rate: The rate the simulation would converge to — the nominal, when checking against it.
        replicates: Independent replicates behind the estimate.
    """
    return math.sqrt(rate * (1.0 - rate) / replicates)


def at_most(nominal: float, replicates: int) -> float:
    """The largest simulated rate still consistent with a true rate of ``nominal``: ``nominal + 4 SE``."""
    return nominal + TOLERANCE_Z * monte_carlo_se(nominal, replicates)


def at_least(nominal: float, replicates: int) -> float:
    """The smallest simulated rate still consistent with a true rate of ``nominal``: ``nominal - 4 SE``."""
    return nominal - TOLERANCE_Z * monte_carlo_se(nominal, replicates)


def within(observed: float, nominal: float, replicates: int) -> bool:
    """Whether ``observed`` is within :data:`TOLERANCE_Z` standard errors of ``nominal`` on either side."""
    return at_least(nominal, replicates) <= observed <= at_most(nominal, replicates)


# ---------------------------------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusteredDesign:
    """Cases × repeats, the shape every arm's observations take.

    A case is the independent draw; its repeats share the case's own level and differ by repeat noise.
    ``between_case_sd`` is the spread of case levels around the arm's mean, ``repeat_sd`` the spread of
    one observation around its case's level, so the intra-case correlation of two repeats is
    ``between² / (between² + repeat²)``.

    Attributes:
        n_cases: Cases, at least one.
        repeats: Repeats per case (the engine's ``k``), at least one.
        between_case_sd: Standard deviation of case levels.
        repeat_sd: Standard deviation of one observation around its case's level.
    """

    n_cases: int
    repeats: int
    between_case_sd: float = 1.0
    repeat_sd: float = 0.5


def draw_clustered(rng: random.Random, design: ClusteredDesign, *, mean: float = 0.0) -> list[list[float]]:
    """One arm's normal observations, grouped by case. Truth: every observation's expectation is ``mean``.

    Returns:
        ``n_cases`` lists of ``repeats`` observations each.
    """
    observations = []
    for _ in range(design.n_cases):
        level = mean + rng.gauss(0.0, design.between_case_sd)
        observations.append([level + rng.gauss(0.0, design.repeat_sd) for _ in range(design.repeats)])
    return observations


def draw_paired_arms(
    rng: random.Random, design: ClusteredDesign, *, effect: float = 0.0, correlation: float = 0.5
) -> tuple[list[list[float]], list[list[float]]]:
    """A control and a contrast over the SAME cases, normal, grouped by case.

    Each case has a level on each arm; the two levels have standard deviation ``between_case_sd`` and
    correlation ``correlation`` (how alike the arms find the same case, which is what pairing buys).
    The contrast's expectation is the control's plus ``effect``, at every case. Truth: the per-case
    difference of case means is normal with mean ``effect`` and variance
    ``2 between² (1 - correlation) + 2 repeat² / repeats``.

    Returns:
        ``(control, contrast)``, each ``n_cases`` lists of ``repeats`` observations, case-aligned.
    """
    shared_weight = math.sqrt(correlation)
    own_weight = math.sqrt(1.0 - correlation)
    control, contrast = [], []
    for _ in range(design.n_cases):
        shared = rng.gauss(0.0, 1.0)
        levels = (
            design.between_case_sd * (shared_weight * shared + own_weight * rng.gauss(0.0, 1.0)),
            effect + design.between_case_sd * (shared_weight * shared + own_weight * rng.gauss(0.0, 1.0)),
        )
        control.append([levels[0] + rng.gauss(0.0, design.repeat_sd) for _ in range(design.repeats)])
        contrast.append([levels[1] + rng.gauss(0.0, design.repeat_sd) for _ in range(design.repeats)])
    return control, contrast


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def draw_binary_paired_arms(
    rng: random.Random,
    n_cases: int,
    repeats: int,
    *,
    base_rate: float,
    between_case_sd: float = 1.0,
    effect: float = 0.0,
) -> tuple[list[list[float]], list[list[float]]]:
    """A control and a contrast over the SAME cases, each observation a pass (1.0) or a fail (0.0).

    A case's pass probability is ``logistic(logit(base_rate) + between_case_sd * z_case)`` on the control
    and the same plus ``effect`` on the logit scale on the contrast, ``z_case`` shared by both arms: some
    cases are hard for every arm. Repeats are independent given the case. Truth: with ``effect=0`` every
    case has the same pass probability on both arms, so no difference exists to find.

    Returns:
        ``(control, contrast)``, each ``n_cases`` lists of ``repeats`` 0.0/1.0 observations.
    """
    logit = math.log(base_rate / (1.0 - base_rate))
    control, contrast = [], []
    for _ in range(n_cases):
        level = logit + between_case_sd * rng.gauss(0.0, 1.0)
        p_control, p_contrast = _logistic(level), _logistic(level + effect)
        control.append([1.0 if rng.random() < p_control else 0.0 for _ in range(repeats)])
        contrast.append([1.0 if rng.random() < p_contrast else 0.0 for _ in range(repeats)])
    return control, contrast


def case_means(observations: Sequence[Sequence[float]]) -> list[float]:
    """Each case's mean over its repeats — the unit every engine comparison tests."""
    return [sum(case) / len(case) for case in observations]


def draw_confusion(
    rng: random.Random,
    labels: Sequence[str],
    *,
    cases_per_label: int,
    repeats: int,
    accuracy: float,
    intra_case_correlation: float,
) -> list[tuple[str, str]]:
    """A classifier's observations over a balanced bank, every repeat of every case, each tagged with its case.

    Each case expects one label (``cases_per_label`` cases per label). A case has its own probability of
    being classified correctly, drawn from a Beta with mean ``accuracy`` and intra-case correlation
    ``intra_case_correlation`` (0 = every repeat independent; near 1 = a case is right on every repeat
    or wrong on every repeat, the near-deterministic classifier). A wrong answer is one of the other labels,
    uniformly. Truth: every label's recall AND precision is ``accuracy`` (balanced bank, symmetric errors:
    each label receives ``(1 - accuracy) / (L - 1)`` of each other label's cases, which sums to its own
    missed share).

    Returns:
        One ``(confusion_cell value, test case id)`` per observation, the shape
        :func:`~threetears.evals.analysis.confusion.label_statistics` reads: the case is what lets its
        intervals count cases rather than repeats.
    """
    if intra_case_correlation > 0.0:
        concentration = 1.0 / intra_case_correlation - 1.0
        alpha, beta = accuracy * concentration, (1.0 - accuracy) * concentration
    observations: list[tuple[str, str]] = []
    for expected in labels:
        others = [label for label in labels if label != expected]
        for index in range(cases_per_label):
            case = f"{expected}-{index}"
            right = rng.betavariate(alpha, beta) if intra_case_correlation > 0.0 else accuracy
            for _ in range(repeats):
                predicted = expected if rng.random() < right else rng.choice(others)
                observations.append((confusion_cell(expected, predicted), case))
    return observations


def draw_rater_pairs(
    rng: random.Random,
    n: int,
    categories: Sequence[int],
    *,
    marginal: Sequence[float],
    agreement: float,
) -> list[tuple[int, int]]:
    """Two raters' answers over ``n`` items, at a known population kappa.

    The first rater draws from ``marginal``; the second copies the first with probability ``agreement``
    and otherwise draws independently from the same ``marginal``. Truth: the population kappa is
    ``agreement`` under ANY disagreement cost with zero on the diagonal — unweighted, quadratic, or with a
    category off the scale — because the observed disagreement is exactly ``1 - agreement`` times the
    disagreement independent raters with these marginals would show, and kappa is one minus that ratio.

    Returns:
        ``n`` ``(first, second)`` pairs.
    """
    pairs = []
    for _ in range(n):
        first = rng.choices(categories, weights=marginal)[0]
        second = first if rng.random() < agreement else rng.choices(categories, weights=marginal)[0]
        pairs.append((first, second))
    return pairs


# ---------------------------------------------------------------------------------------------------
# Reference answers, independent of the engine's arithmetic
# ---------------------------------------------------------------------------------------------------


def student_t_two_sided_mass(t: float, df: int) -> float:
    """``P(|T| <= t)`` for Student's t on an integer ``df``, by the closed-form trigonometric series.

    Abramowitz & Stegun 26.7.3 (odd df) and 26.7.4 (even df), with ``θ = atan(t / sqrt(df))``. Exact for
    integer df and sharing nothing with the engine's incomplete-beta route.

    Args:
        t: A non-negative statistic.
        df: Degrees of freedom, at least 1.
    """
    theta = math.atan(abs(t) / math.sqrt(df))
    cosine, sine = math.cos(theta), math.sin(theta)
    if df % 2 == 1:
        series, term = 0.0, cosine
        for j in range(1, (df - 1) // 2 + 1):
            if j > 1:
                term *= cosine * cosine * (2 * j - 2) / (2 * j - 1)
            series += term
        return 2.0 / math.pi * (theta + sine * series) if df > 1 else 2.0 / math.pi * theta
    series, term = 1.0, 1.0
    for j in range(1, df // 2):
        term *= cosine * cosine * (2 * j - 1) / (2 * j)
        series += term
    return sine * series


def student_t_critical(confidence: float, df: int) -> float:
    """The ``t`` with ``P(|T| <= t) = confidence`` on an integer ``df``, by bisection on the closed form."""
    low, high = 0.0, 1.0e4
    for _ in range(200):
        middle = 0.5 * (low + high)
        if student_t_two_sided_mass(middle, df) < confidence:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def paired_t_power(n: int, effect_size: float, alpha: float) -> float:
    """The power of a two-sided one-sample (paired) t-test on ``n`` differences at standardized effect ``effect_size``.

    ``P(|T'| > c)`` for the noncentral t with ``n - 1`` df and noncentrality ``effect_size * sqrt(n)``,
    ``c`` the two-sided critical value at ``alpha``. Computed as an integral over ``S = sqrt(V)``,
    ``V ~ χ²(df)``: ``power = E[Φ(δ - c S/√df) + Φ(-δ - c S/√df)]``, by Simpson's rule on 4,000 panels
    of ``[0, √df + 12]`` — far past where the χ density has any mass.
    """
    df = n - 1
    critical = student_t_critical(1.0 - alpha, df)
    noncentrality = effect_size * math.sqrt(n)
    normal = NormalDist()
    log_norm = (df / 2.0 - 1.0) * math.log(2.0) + math.lgamma(df / 2.0)

    def integrand(s: float) -> float:
        if s == 0.0:
            density = 0.0 if df > 1 else math.exp(-log_norm)
        else:
            density = math.exp((df - 1) * math.log(s) - s * s / 2.0 - log_norm)
        scaled = critical * s / math.sqrt(df)
        return density * (normal.cdf(noncentrality - scaled) + normal.cdf(-noncentrality - scaled))

    panels, upper = 4000, math.sqrt(df) + 12.0
    width = upper / panels
    total = integrand(0.0) + integrand(upper)
    for index in range(1, panels):
        total += (4.0 if index % 2 else 2.0) * integrand(index * width)
    return total * width / 3.0


def hedges_correction(df: int) -> float:
    """``E[d̂] / δ`` for a standardized mean difference whose SD has ``df`` degrees of freedom.

    ``J(df)^-1`` with ``J(df) = Γ(df/2) / (sqrt(df/2) Γ((df-1)/2))`` (Hedges 1981): the factor by which
    the sample effect size overstates the population one on average.
    """
    return math.exp(math.lgamma((df - 1) / 2.0) - math.lgamma(df / 2.0)) * math.sqrt(df / 2.0)


# ---------------------------------------------------------------------------------------------------
# The engine's family rule, composed from its public statistics
# ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyVerdict:
    """One contrast-against-control comparison as the family rule decides it.

    Attributes:
        p_raw: The separation test's two-sided p, or None when it ran no test.
        p_adjusted: The Holm-adjusted p over the family, or None when there was no raw p.
        verdict: ``untested``, ``not_separated``, ``equivalent``, ``improved`` or ``regressed``.
        interval: The interval on the difference at the family's level, or None when no test ran.
        equivalence_p_adjusted: The adjusted TOST p, or None when no equivalence test ran.
    """

    p_raw: float | None
    p_adjusted: float | None
    verdict: str
    interval: tuple[float, float] | None = None
    equivalence_p_adjusted: float | None = None


def family_verdicts(
    comparisons: Sequence[tuple[dict[str, float], dict[str, float], bool]],
    *,
    margins: Sequence[float | None] | None = None,
    value_ranges: Sequence[tuple[float, float] | None] | None = None,
) -> list[FamilyVerdict]:
    """Decide one family of comparisons by the rule the analysis bundle's ``multiple_comparisons`` states.

    Each comparison is ``(control per-case values, contrast per-case values, higher_is_better)``. It is
    tested paired over the cases both sides ran when they share at least two, else unpaired over each
    side's values, its p :func:`~threetears.evals.analysis.stats.separation_p`'s — the t-test's where the
    values have spread, the exact permutation p where they have none, None where no test can decide — as the
    bundle's is. A paired comparison with a declared margin (``margins``, aligned with ``comparisons``) also
    runs the paired TOST against it, on the reading's declared range where ``value_ranges`` gives one
    (:func:`~threetears.evals.analysis.stats.paired_equivalence`). Every separation p and TOST p is
    Holm-adjusted together, the multiplier capped at the separation count
    (:func:`~threetears.evals.analysis.stats.holm_adjust`); a comparison separates when its adjusted p is
    below α and its delta is nonzero, in the direction its sign and ``higher_is_better`` give, and is
    otherwise equivalent when its adjusted TOST p is below α. Its interval is at ``1 − α/m`` over the ``m``
    separations (:func:`~threetears.evals.analysis.stats.difference_interval`).

    This is composed here so a simulation can run it thousands of times without assembling a bundle each
    time. ``test_simulated_multiple_comparisons`` pins the bundle to it: on seeded campaigns, every
    comparison the bundle publishes carries exactly this p, adjusted p, interval and verdict.

    Returns:
        One verdict per comparison, in order.
    """
    tested: list[tuple[float | None, float | None, float | None, list[float], list[float], bool]] = []
    for index, (control, contrast, _) in enumerate(comparisons):
        shared = sorted(set(control) & set(contrast))
        paired = len(shared) >= 2
        a = [control[case] for case in shared] if paired else list(control.values())
        b = [contrast[case] for case in shared] if paired else list(contrast.values())
        delta = sum(b) / len(b) - sum(a) / len(a) if a and b else None
        p_raw = separation_p(a, b, paired=paired)
        margin = margins[index] if margins is not None else None
        value_range = value_ranges[index] if value_ranges is not None else None
        equivalence_p = None
        if paired and margin and p_raw is not None:
            diffs = [exact_decimal(y) - exact_decimal(x) for x, y in zip(a, b)]
            equivalence_p = paired_equivalence(diffs, margin, value_range=value_range)[1]
        tested.append((p_raw, equivalence_p, delta, a, b, paired))
    m = sum(1 for p_raw, *_ in tested if p_raw is not None)
    raw = [p for p_raw, equivalence_p, *_ in tested for p in (p_raw, equivalence_p) if p is not None]
    adjusted = iter(holm_adjust(raw, max_true=m) if raw else [])
    verdicts = []
    for (p_raw, equivalence_p, delta, a, b, paired), (_, _, higher_is_better) in zip(tested, comparisons, strict=True):
        if p_raw is None:
            verdicts.append(FamilyVerdict(None, None, "untested"))
            continue
        p_adjusted = next(adjusted)
        equivalence_adjusted = next(adjusted) if equivalence_p is not None else None
        verdict = "not_separated"
        if p_adjusted < SIGNIFICANCE_ALPHA and delta:
            verdict = "improved" if (delta > 0) == higher_is_better else "regressed"
        elif equivalence_adjusted is not None and equivalence_adjusted < SIGNIFICANCE_ALPHA:
            verdict = "equivalent"
        interval = difference_interval(a, b, paired=paired, confidence=1.0 - SIGNIFICANCE_ALPHA / m)
        verdicts.append(FamilyVerdict(p_raw, p_adjusted, verdict, interval, equivalence_adjusted))
    return verdicts
