"""Descriptive and two-sample statistics for eval reporting.

Dependency-free: the project carries no scientific stack (no numpy/scipy), and
pulling scipy's compiled binaries in for one t-test — feeding a single cell of a
comparison table — is not worth the deploy weight. So the Student-t
two-sided p-value is computed from the regularized incomplete beta function
(the standard closed form), implemented here in stdlib ``math`` and pinned by
tests against scipy-computed reference values.

:func:`standard_error_of_mean` serves the reporting layer's rule that no
aggregate is ever displayed as a bare point estimate: we already pay for k
repeats, and reporting only the mean discards what they bought.

Consumed by :func:`threetears.evals.analysis.reads.compare_two_runs` to attach
an effect size (Hedges' g), the p-value, and a significance flag to each
per-model composite delta. The p travels with the flag rather than being
consumed and dropped: a verdict a reader cannot check against the number it was
thresholded on is indistinguishable from one no test produced.
Composite scores are continuous per-case quality values in ``[0, 1]``
(see :func:`threetears.evals.contracts.scoring.compute_per_case_composites`). Samples are
*paired* by case when the two runs actually scored the same frozen
``test_case_id`` s (a paired t-test, far more powerful); when they scored
different cases — including two runs of one template whose case sets do not
intersect — they are two independent samples (Welch's statistic, read on Hsu's
conservative degrees of freedom; see :func:`composite_significance`).
The caller decides which, and the two effect sizes are not interchangeable:
paired yields Hedges' g_z over the difference SD, unpaired Hedges' g over the
pooled SD — each the textbook Cohen's d times Hedges' small-sample factor J,
because d itself overstates the effect by 77% at three cases.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Hashable, Mapping, Sequence
from fractions import Fraction
from functools import lru_cache
from statistics import NormalDist
from typing import Final, Literal, NamedTuple

from threetears.evals.contracts.surface import GuardrailDecision

# Two-sided p-value below which a composite delta is called significant.
SIGNIFICANCE_ALPHA = 0.05


def _sign_flip_p(n: int) -> float:
    """The exact two-sided sign-flip p of ``n`` paired differences all one nonzero amount: ``2 ** (1 - n)``.

    Under the null of exchangeable signs each of the ``2 ** n`` assignments is equally likely, and only the
    two all-one-sign ones are as extreme as the observed.
    """
    return min(1.0, 2.0 ** (1 - n))


def _min_pairs_for_sign_flip(alpha: float) -> int:
    """Smallest ``n`` whose exact paired sign-flip test can reach ``alpha``.

    Under the null of exchangeable signs there are ``2 ** n`` equally likely sign
    assignments, so the smallest attainable two-sided p is ``2 ** (1 - n)``. This
    returns the first ``n`` for which that lands at or below ``alpha`` — computed
    rather than written down, so it tracks the threshold instead of drifting from
    it.
    """
    n = 2
    while _sign_flip_p(n) > alpha:
        n += 1
    return n


#: Pair-count floor for claiming significance from a zero-variance difference.
#: Derived from the alpha above, not chosen: below it, no exact test of a perfectly
#: consistent move could reject at that alpha, so the claim would outrun the data.
MIN_PAIRS_FOR_DETERMINISTIC_GAP = _min_pairs_for_sign_flip(SIGNIFICANCE_ALPHA)

#: The paired test the change classifier discloses, so a regression flag names the
#: statistics it rests on rather than presenting a bare verdict.
PAIRED_TEST_NAME = (
    "paired two-sided t-test on shared per-case values (the exact sign-flip test where every difference is one "
    f"amount), α={SIGNIFICANCE_ALPHA}"
)

# The test that runs when the two samples cannot be paired — no shared frozen
# case set, so the cases on each side are different questions. Named beside the
# paired one because a surface that discloses only "t-test" leaves a reader
# unable to tell a powerful within-case comparison from a weak between-case one,
# and that difference is most of what a small eval arm's verdict rests on.
UNPAIRED_TEST_NAME = (
    "Welch's unequal-variance two-sided t statistic on unpaired per-case values, read on Hsu's conservative "
    f"min(n_a, n_b) − 1 degrees of freedom, α={SIGNIFICANCE_ALPHA}"
)

#: The equivalence test the change classifier runs beside the paired test, named for
#: the same reason: an `equivalent` label names the statistics it rests on.
EQUIVALENCE_TEST_NAME = (
    "two one-sided paired tests (TOST) against ± the measure's declared margin, each a bounded test by betting on "
    "the measure's declared range, which holds α for any distribution on it at every n; a measure declaring no "
    f"range is not tested for equivalence, α={SIGNIFICANCE_ALPHA}"
)

#: Why a measure that declares a margin and no range is never tested for equivalence — the reason every surface
#: carrying the refusal states, naming the remedy first.
EQUIVALENCE_NEEDS_RANGE = (
    "declare value_range on this measure to test equivalence: it declares a margin and no range, and with no range "
    "no test of a mean holds its error rate (an unbounded value can hide a rare large move), so equivalence is "
    "untested and nothing here says the two are alike"
)


def equivalence_untested_reason(margin: float | None, value_range: tuple[float, float] | None) -> str | None:
    """Why :func:`paired_equivalence` refuses a measure's margin outright, or None when it does not.

    One derivation for every surface that carries the refusal beside its verdict. Only the missing range is named
    here: the per-sample refusals (too few pairs, a difference outside the range) are the test's own outcome.

    Args:
        margin: The measure's declared margin, or None.
        value_range: The measure's declared inclusive bounds, or None.

    Returns:
        :data:`EQUIVALENCE_NEEDS_RANGE` when a positive margin is declared with no range, else None.
    """
    return EQUIVALENCE_NEEDS_RANGE if margin is not None and margin > 0.0 and value_range is None else None


def _sample_std(values: list[float]) -> float:
    """Sample standard deviation (ddof=1); 0.0 for a constant/short sample."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def standard_error_of_mean(values: list[float]) -> float | None:
    """Standard error of the mean — how precisely this sample locates the mean.

    ``None`` for fewer than two observations, which is the honest answer rather
    than ``0.0``: a single measurement has *unknown* precision, and rendering it
    as zero spread would present the least certain cell in a pivot as the most
    certain one. A genuinely constant sample of two or more does return ``0.0``,
    because that is a measurement, not an absence.

    Args:
        values: The sample. Must already be the quantity being averaged — pass
            per-case means when the aggregate is a mean over cases, so the
            dispersion describes the same estimate the value reports.

    Returns:
        ``sd / sqrt(n)`` with ``ddof=1``, or ``None`` when ``n < 2``.
    """
    n = len(values)
    if n < 2:
        return None
    return _sample_std([float(v) for v in values]) / math.sqrt(n)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction expansion for the incomplete beta (Lentz's method)."""
    max_iter = 300
    eps = 3.0e-16
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        step = d * c
        h *= step
        if abs(step - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function ``I_x(a, b)`` for ``0 <= x <= 1``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _student_t_two_sided_p(t: float, df: float) -> float:
    """Two-tailed p-value for t-statistic ``t`` on ``df`` degrees of freedom.

    Uses the closed form ``P(|T| > |t|) = I_{df/(df+t^2)}(df/2, 1/2)``.
    """
    if df <= 0.0:
        return float("nan")
    return _betai(0.5 * df, 0.5, df / (df + t * t))


@lru_cache(maxsize=1024)
def t_critical_two_sided(confidence: float, df: float) -> float:
    """The two-sided t multiplier for a confidence level on ``df`` degrees of freedom.

    The inverse of :func:`_student_t_two_sided_p`, found by bisection rather than by a
    closed form — the forward function is already exact here, and a table of critical
    values would be a second source of truth that could drift from it.

    Cached: the bisection costs a few hundred incomplete-beta evaluations, and every interval an
    analysis states asks it again for one of a handful of ``(confidence, df)`` pairs.

    Why this exists rather than a fixed 1.96: the normal multiplier is the large-sample
    limit, and the samples an eval arm produces are routinely small. At df=2 (three
    observations) the true 95% multiplier is ~4.30, so quoting 1.96 would publish an
    interval less than half its honest width and label it "95%" — an overconfident interval
    is worse than a wide one, because it reads as precision that was measured.

    Args:
        confidence: Central mass the interval should cover, e.g. ``0.95``.
        df: Degrees of freedom (``n - 1``); must be positive.

    Returns:
        The multiplier ``t`` such that ``P(|T| <= t) == confidence``.

    Raises:
        ValueError: If ``df`` is not positive or ``confidence`` is not in ``(0, 1)``.
    """
    if df <= 0.0:
        raise ValueError(f"df must be positive, got {df}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    target_p = 1.0 - confidence
    low, high = 0.0, 1000.0
    # 200 halvings drive the bracket far below any precision a report renders; the
    # function is monotone decreasing in t, so the bracket is always valid.
    for _ in range(200):
        mid = 0.5 * (low + high)
        if _student_t_two_sided_p(mid, df) > target_p:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


#: The coverage of every interval the analysis engine reports on a mean — a measure's or a judged
#: dimension's. One level, owned here beside the one function that computes the width at it, because
#: the width and the label are decided together or not at all: an interval computed at one level and
#: captioned with another is a wrong statement that reads as a precise one. Two intervals drawn side by
#: side are always comparable because they were computed here.
INTERVAL_LEVEL = 0.95


@functools.cache
def _cached_t_critical(confidence: float, df: int) -> float:
    return t_critical_two_sided(confidence, df)


def _t_multiplier(df: int) -> float:
    """The two-sided t multiplier at :data:`INTERVAL_LEVEL` on ``df`` degrees of freedom.

    Cached on the level and df: the bisection behind :func:`t_critical_two_sided` costs a few hundred
    incomplete-beta evaluations, and every reading in a bundle asks again at one of a handful of case
    counts. The level is read at call time, so the width always follows the declared level.
    """
    return _cached_t_critical(INTERVAL_LEVEL, df)


def ci_half_width(sem: float, n_cases: int) -> float | None:
    """Half-width of the reported interval on a mean at :data:`INTERVAL_LEVEL`: the t multiplier times the SEM.

    Returns ``None`` rather than 0.0 below two cases, taking the same position
    :func:`standard_error_of_mean` already takes: the spread is *unestimable* there, not zero. A 0.0
    would render as a zero-width 95% interval — a point estimate wearing a confidence label, which is
    the strongest possible claim made from the least possible evidence.

    The multiplier is t, not a fixed 1.96: an arm can be three cases, where the honest 95%
    multiplier is ~4.30, and quoting the large-sample constant would publish an interval at under half
    its true width while labelling it "95%".

    Args:
        sem: The standard error of the mean — over cases (:func:`clustered_standard_error`) when an
            observation can repeat a case.
        n_cases: Independent cases behind it (``n_cases - 1`` degrees of freedom). Where every case was
            observed once that is the observation count; where cases repeat, it is the case count, never
            the observations — counting k repeats as k draws is what narrows an interval below its truth.

    Returns:
        The half-width, or ``None`` where no interval is estimable.
    """
    if n_cases < 2:
        return None
    return _t_multiplier(n_cases - 1) * sem


def _wilson_bounds(rate: float, n: float, critical: float) -> tuple[float, float]:
    """The Wilson score bounds on ``rate`` over ``n`` (possibly an effective n) at multiplier ``critical``.

    The bounds are pinned to contain the rate; see :func:`wilson_interval` for why.
    """
    denominator = 1 + critical * critical / n
    centre = (rate + critical * critical / (2 * n)) / denominator
    half = critical * math.sqrt(rate * (1 - rate) / n + critical * critical / (4 * n * n)) / denominator
    return min(rate, max(0.0, centre - half)), max(rate, min(1.0, centre + half))


def wilson_interval(n_true: int, n: int) -> tuple[float, float] | None:
    """The Wilson score interval on a proportion at :data:`INTERVAL_LEVEL` over ``n`` independent trials.

    Wilson rather than the normal approximation because a boolean measure's rate sits at 0 or 1
    exactly when it is most interesting (every encounter on target, none), where the normal interval
    collapses to a zero-width point and reads as certainty from three observations. Wilson stays
    inside [0, 1] and keeps a width at the ends.

    The trials must be independent: one per case. A reading whose observations can repeat a case takes
    :func:`proportion_interval`, which is this interval wherever every case was observed once.

    Args:
        n_true: Observations that held.
        n: Observations.

    The interval always contains the rate, exactly. Algebraically the Wilson bound at a rate of 1 is
    1 and at a rate of 0 is 0, but ``centre ± half`` reaches them through cancelling floating-point
    terms and lands a hair to either side — 4 of 4 gave an upper bound of ``0.9999999999999999``. An
    interval that misses its own estimate by one ulp is refused by every surface that draws it
    (:class:`~threetears.evals.analysis.viz.payloads.ConfidenceInterval`), so the bounds are pinned
    to contain the rate rather than every consumer tolerating the noise.

    Returns:
        ``(low, high)``, or ``None`` with no observations — there is no rate to bound.

    Raises:
        ValueError: ``n_true`` is negative or above ``n``.
    """
    if not 0 <= n_true <= n:
        raise ValueError(f"a proportion needs 0 <= n_true <= n; got n_true={n_true}, n={n}")
    if n == 0:
        return None
    z = NormalDist().inv_cdf(1 - (1 - INTERVAL_LEVEL) / 2)
    return _wilson_bounds(n_true / n, n, z)


def _require_aligned(values: Sequence[object], cases: Sequence[Hashable]) -> None:
    if len(values) != len(cases):
        raise ValueError(f"every observation needs its case; got {len(values)} observations and {len(cases)} cases")


def clustered_standard_error(values: Sequence[float], cases: Sequence[Hashable]) -> float | None:
    """Standard error of the mean of observations that come in cases — the cluster-robust form.

    Repeats of one case share everything about the case, so they are not independent draws, and the
    plain SEM over them is too small by about the square root of the repeats when cases differ more
    than repeats do. This is Miller's cluster-robust standard error ("Adding Error Bars to Evals",
    2024) with the ``G / (G - 1)`` small-sample factor over ``G`` cases:
    ``sqrt(G / (G - 1) * Σ_g (Σ_i (y_gi - ȳ))²) / N``. It is the SE of the mean every reading reports — the
    mean over the observations — and it is read on ``G - 1`` degrees of freedom.

    It reduces to the forms a reader already knows. With every case observed once it is
    :func:`standard_error_of_mean` (and returns exactly that). With every case observed the same number
    of times it equals the SEM of the case means, the statistic a comparison between arms runs on.

    Args:
        values: The observations.
        cases: Each observation's case, aligned with ``values``.

    Returns:
        The standard error, or ``None`` below two cases: one case has no between-case spread to
        estimate, however often it was repeated, and its repeats say nothing about another case.

    Raises:
        ValueError: ``values`` and ``cases`` differ in length.
    """
    _require_aligned(values, cases)
    n = len(values)
    distinct = set(cases)
    if len(distinct) == n:
        return standard_error_of_mean([float(value) for value in values])
    groups = len(distinct)
    if groups < 2:
        return None
    mean = sum(values) / n
    residual_by_case: dict[Hashable, float] = {}
    for value, case in zip(values, cases):
        residual_by_case[case] = residual_by_case.get(case, 0.0) + (value - mean)
    return math.sqrt(groups / (groups - 1) * sum(r * r for r in residual_by_case.values())) / n


def proportion_interval(outcomes: Sequence[bool], cases: Sequence[Hashable]) -> tuple[float, float] | None:
    """The interval on a rate at :data:`INTERVAL_LEVEL` over observations that come in cases.

    With every case observed once it is :func:`wilson_interval`, unchanged. Where cases repeat, it is
    the Wilson interval on the observed rate with the sample size replaced by the effective one — the
    observations divided by the design effect the clustering measured — and the normal multiplier by
    t on ``cases - 1`` degrees of freedom, because the spread is now estimated from the cases.
    Wilson's shape is kept for the reason it was chosen: a perfect rate keeps a width.

    **The design effect is estimated with two pseudo-cases added**, one that held on every repeat and
    one that held on none, each of the mean repeat depth — the case-level analogue of Agresti and
    Coull's two added successes and failures. A design effect read off a handful of minority outcomes
    is mostly noise: one miss among forty observations says nothing about whether misses cluster, and
    the raw estimate calls it unclustered and returns a narrow interval exactly when the cases seen
    were the easy ones. The pseudo-cases pull a thin estimate toward full clustering, give a perfect
    rate an effective size of about its case count without a special case, and matter less as cases
    are added. The effective size never exceeds the observations: clustering never adds information.

    In simulation over 2 to 15 cases at 2 to 5 repeats, with case rates spread from mostly-shared to
    all-or-nothing, it covered the true rate at least 95% of the time in every configuration, where
    the Wilson interval over observations covered as little as 67%
    (``tests/test_sim_reading_intervals.py``).

    Args:
        outcomes: Each observation's outcome.
        cases: Each observation's case, aligned with ``outcomes``.

    Returns:
        ``(low, high)``; ``None`` with no observations, or when the observations repeat a single case —
        one case has no between-case spread to estimate.

    Raises:
        ValueError: ``outcomes`` and ``cases`` differ in length.
    """
    _require_aligned(outcomes, cases)
    n = len(outcomes)
    n_true = sum(1 for outcome in outcomes if outcome)
    size_by_case: dict[Hashable, int] = {}
    true_by_case: dict[Hashable, int] = {}
    for outcome, case in zip(outcomes, cases):
        size_by_case[case] = size_by_case.get(case, 0) + 1
        true_by_case[case] = true_by_case.get(case, 0) + (1 if outcome else 0)
    groups = len(size_by_case)
    if groups == n:
        return wilson_interval(n_true, n)
    if groups < 2:
        return None
    depth = n / groups
    held = [float(true_by_case[case]) for case in size_by_case] + [depth, 0.0]
    sizes = [float(size) for size in size_by_case.values()] + [depth, depth]
    total = n + 2 * depth
    smoothed = (n_true + depth) / total
    pooled_groups = groups + 2
    variance = (
        pooled_groups
        / (pooled_groups - 1)
        * sum((h - size * smoothed) ** 2 for h, size in zip(held, sizes))
        / (total * total)
    )
    effective_n = min(float(n), smoothed * (1 - smoothed) / variance)
    return _wilson_bounds(n_true / n, effective_n, _t_multiplier(groups - 1))


def _beta_quantile(p: float, a: float, b: float) -> float:
    """The ``p`` quantile of a Beta(a, b), by bisection on :func:`_betai` (monotone in ``x``)."""
    low, high = 0.0, 1.0
    for _ in range(64):
        middle = 0.5 * (low + high)
        if _betai(a, b, middle) < p:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def case_rate_interval(case_rates: Sequence[float], *, max_effective_n: float) -> tuple[float, float] | None:
    """The interval at :data:`INTERVAL_LEVEL` on the mean of per-case rates, each in ``[0, 1]``.

    For a reading that averages one estimate per case — pass^k's ``C(c, k) / C(n, k)`` is the one this
    serves — where the cases are the independent draws and each case's own estimate is noisy. It takes
    :func:`proportion_interval`'s route to an effective sample size, and with it that function's guard:
    the spread of the case rates is read with two pseudo-cases added, one at 1 and one at 0, so a handful
    of identical cases (every one passing) is not read as no spread at all, and the effective size is
    ``s̃ (1 − s̃) / Var̃``, never more than ``max_effective_n``.

    **The bounds are Clopper–Pearson's on that effective size**, not Wilson's, at the tail mass the
    t multiplier on ``cases − 1`` degrees of freedom leaves on each side. Wilson's shape covered as little as
    87% of the time in simulation where the truth sat near 0 or 1 at 8–15 cases; Clopper–Pearson's held at
    least 96% over 2–15 cases, depth 1–5 and case rates from all-alike to all-or-nothing, for about a tenth
    more width (``tests/test_simulated_frontier.py`` checks it on the engine).

    Args:
        case_rates: One estimate per case, each in ``[0, 1]``.
        max_effective_n: The most information the cases can carry — for pass^k, the scored attempts at the
            qualifying cases over ``k``: the number of separate ``k``-attempt runs they contain.

    Returns:
        ``(low, high)`` containing the mean of ``case_rates``, or ``None`` below two cases, where there is
        no between-case spread to estimate.

    Raises:
        ValueError: A rate is outside ``[0, 1]``.
    """
    stray = [rate for rate in case_rates if not 0.0 <= rate <= 1.0]
    if stray:
        raise ValueError(f"case rates must lie in [0, 1]; got {stray}")
    cases = len(case_rates)
    if cases < 2:
        return None
    rate = math.fsum(case_rates) / cases
    padded = [float(value) for value in case_rates] + [1.0, 0.0]
    smoothed = math.fsum(padded) / len(padded)
    variance = math.fsum((value - smoothed) ** 2 for value in padded) / ((len(padded) - 1) * len(padded))
    effective_n = max(1.0, min(max_effective_n, smoothed * (1.0 - smoothed) / variance))
    tail = 1.0 - NormalDist().cdf(_t_multiplier(cases - 1))
    held = rate * effective_n
    low = 0.0 if held <= 0.0 else _beta_quantile(tail, held, effective_n - held + 1.0)
    high = 1.0 if held >= effective_n else _beta_quantile(1.0 - tail, held + 1.0, effective_n - held)
    return min(rate, low), max(rate, high)


def mean_interval(
    mean: float,
    sem: float,
    n_cases: int,
    *,
    value_range: tuple[float, float] | None = None,
    floor: float | None = None,
) -> tuple[float, float] | None:
    """The interval on a mean at :data:`INTERVAL_LEVEL`, kept inside the scale the measure is declared on.

    ``mean ± t·sem`` (:func:`ci_half_width`), clipped to ``value_range`` when one is declared. The
    symmetric t interval knows nothing of a bound, so a mean near the top of a bounded scale got an
    upper bound past it — a 0.8 accuracy over ten observations read ``[0.498, 1.102]``, a share above
    all of them. The scale is a fact about every value the mean could take, so no part of the
    interval beyond it is a value the mean could have. A measure bounded below only — a time, a spend, a
    count (``MetricDescriptor.nonnegative``) — is clipped at ``floor`` the same way: a cost interval read
    ``[-0.0002991, 0.000801]`` dollars before it, a spend no arm can have. Only an interval on a mean is
    clipped; one on a difference of two means never is, since a difference can fall either way.

    Args:
        mean: The point estimate.
        sem: Its standard error — :func:`clustered_standard_error` where observations can repeat a case.
        n_cases: Independent cases behind it; see :func:`ci_half_width`.
        value_range: The measure's declared inclusive bounds, or None when it declares none.
        floor: The lowest value the measure can take where it declares no range
            (``MetricDescriptor.interval_floor``), or None when nothing bounds it below.

    Returns:
        ``(low, high)``, or ``None`` below two cases, where no interval is estimable.
    """
    half = ci_half_width(sem, n_cases)
    if half is None:
        return None
    low, high = mean - half, mean + half
    if value_range is not None:
        bottom, top = value_range
        low, high = max(bottom, low), min(top, high)
    elif floor is not None:
        low = max(floor, low)
    return low, high


def observed_mean_interval(
    values: Sequence[float],
    *,
    cases: Sequence[Hashable],
    value_range: tuple[float, float] | None = None,
    floor: float | None = None,
) -> tuple[float, float] | None:
    """The interval on the mean of a numeric measure's observations — the ONE rule every numeric summary takes.

    Observations that are each 0 or 1 on a measure declared on ``[0, 1]`` are trials, and their mean
    is a proportion: ``accuracy``, derived from each observation's ``match``, is exactly that. A
    proportion is bounded by :func:`proportion_interval`, the rule its boolean twin takes, so ``accuracy``
    and ``match`` over the same observations state one interval rather than two different ones — and
    a perfect score keeps a width instead of the t interval's zero-width point. Every other numeric
    measure takes :func:`mean_interval` on the :func:`clustered_standard_error` over its cases, clipped to
    its declared scale, or at its floor where it is bounded below only.

    Args:
        values: The observations.
        cases: Each observation's case, aligned with ``values`` — required, because whether fifteen
            observations are fifteen cases or five cases three times is the difference between two
            interval widths, and nothing in the values says which.
        value_range: The measure's declared inclusive bounds, or None when it declares none.
        floor: The measure's lower bound where it declares no range (``MetricDescriptor.interval_floor``),
            or None; see :func:`mean_interval`.

    Returns:
        ``(low, high)``, or ``None`` below two observations, where no interval is estimable — the
        position :func:`standard_error_of_mean` takes, held for a proportion too, so a numeric
        measure's interval appears and disappears at one n whatever its values — and ``None`` when the
        observations all repeat one case, which has no between-case spread to estimate.

    Raises:
        ValueError: ``values`` and ``cases`` differ in length.
    """
    _require_aligned(values, cases)
    n = len(values)
    if n < 2:
        return None
    if value_range == (0.0, 1.0) and all(value in (0.0, 1.0) for value in values):
        return proportion_interval([value == 1.0 for value in values], cases)
    sem = clustered_standard_error(values, cases)
    if sem is None:
        return None
    return mean_interval(sum(values) / n, sem, len(set(cases)), value_range=value_range, floor=floor)


#: How far below its own mean (above, where lower is better) an incumbent's bar is seeded, as a fraction of
#: the permissive half of its interval. ``√2 − 1``, derived rather than chosen: a candidate measured as the
#: incumbent was (same cases, same spread) misses a bar at ``T`` when its interval's near end falls past it,
#: so an unchanged one misses when its mean trails the incumbent's by more than ``h + c·h`` (``h`` the
#: half-width). The difference of two such means has standard error ``√2·SE``, and a one-sided test at the
#: interval's own ``(1 − level) / 2`` rejects past ``t·√2·SE = √2·h``; ``1 + c = √2`` makes the two one rule.
BAR_SEED_HALF_WIDTH_FRACTION: Final = math.sqrt(2.0) - 1.0


def bar_seed(mean: float, interval: tuple[float, float], *, higher_is_better: bool) -> float:
    """The threshold a measured incumbent proposes as its bar: its mean, less the share of its own noise it carries.

    "Never ship worse than what runs today" anchors the bar at the incumbent's mean. But that mean was
    measured, and a bar is then held fixed: read by :func:`interval_clears`, an unchanged incumbent re-measured
    misses a bar at its old mean whenever the new mean trails the old by more than the new interval's
    half-width — about 8% of the time at large n, past the 2.5% the interval promises, because the old mean's
    own error was never counted. The seed moves the bar
    :data:`BAR_SEED_HALF_WIDTH_FRACTION` of the incumbent's permissive half-width toward the permissive
    end, which is exactly what makes a miss of an unchanged incumbent measured on as many cases a one-sided
    test at the nominal 2.5%.

    The neighbouring choices fail it. At the mean, the unchanged incumbent misses up to 8% of the time. At
    the interval's permissive end the bar sits a whole half-width under today's mean, so at three cases a
    candidate 1.6σ worse than the incumbent is still shown to clear it about a fifth of the time.

    The calibration assumes the candidate is measured on about as many cases as the incumbent was, the
    usual shape of a bar read on campaigns over one case bank. Fewer cases make a miss rarer than nominal.
    Many more cases make it commoner, since the incumbent's own error, frozen into the bar, is then the
    larger share of the difference.

    Args:
        mean: The incumbent's mean on the measure.
        interval: Its interval on that mean, as the summary states it (:func:`observed_mean_interval`).
        higher_is_better: The measure's declared direction.

    Returns:
        The seed threshold.
    """
    low, high = interval
    if higher_is_better:
        return mean - BAR_SEED_HALF_WIDTH_FRACTION * (mean - low)
    return mean + BAR_SEED_HALF_WIDTH_FRACTION * (high - mean)


def interval_clears(
    interval: tuple[float, float], threshold: float, *, margin: float | None, higher_is_better: bool
) -> bool | None:
    """Whether a cell's interval clears a bar, misses it, or cannot tell — against the measure's declared margin.

    Three states, with the discipline of ``not_separated``: an absence of evidence is never a claim. The
    line is the threshold less the margin on the worse side — ``threshold − margin`` where higher is
    better, ``threshold + margin`` where lower is.

    - **True — cleared**: the whole interval lies on the good side of the line (at it counts). The cell
      is shown to be no worse than the bar by more than a shortfall too small to act on.
    - **False — missed**: the whole interval lies on the bad side. The cell is shown to fall short by
      more than that.
    - **None — undecided**: the interval straddles the line. The data cannot say which, and a wide
      interval on a handful of cases usually lands here. It is neither a pass nor a failure.

    Args:
        interval: The cell's interval on the mean. A cell with none (fewer than two observations) is
            not passed here: it has no interval to read, which its caller states as such.
        threshold: The bar's threshold, in the measure's units.
        margin: The measure's declared margin (:attr:`MetricDescriptor.materiality_threshold`), in its
            units, or None when it declares none — the bar is then held at the threshold itself.
        higher_is_better: Which way clearing runs.

    Returns:
        True when cleared, False when missed, None when undecided.
    """
    slack = margin or 0.0
    low, high = interval
    if higher_is_better:
        line = threshold - slack
        good, bad = low >= line, high < line
    else:
        line = threshold + slack
        good, bad = high <= line, low > line
    if good:
        return True
    if bad:
        return False
    return None


class KappaMoments(NamedTuple):
    """What Cohen's kappa is computed from, over one rater pair's items, under one disagreement cost.

    ``kappa = 1 - observed / expected``. The squared costs are what an interval on kappa reads: how
    spread out a disagreement is, as well as how often one happens (:mod:`threetears.evals.analysis.agreement`).
    """

    #: The items.
    n: int
    #: The mean disagreement cost over the items.
    observed: float
    #: The mean squared disagreement cost over the items.
    observed_square: float
    #: The mean cost two raters with these marginals would show by chance alone.
    expected: float
    #: The mean squared cost two raters with these marginals would show by chance alone.
    expected_square: float
    #: Each item's disagreement cost, in the order the pairs were given.
    costs: tuple[float, ...] = ()


def kappa_moments(
    pairs: Sequence[tuple[int, int]],
    categories: Sequence[int],
    *,
    weights: Literal["none", "quadratic"] = "none",
    unordered: Sequence[int] = (),
) -> KappaMoments | None:
    """The disagreement moments Cohen's kappa is computed from — see :func:`cohen_kappa` for the costs.

    Args:
        pairs: ``(first, second)`` per item.
        categories: Every category either rater could give, in order.
        weights: ``"none"`` or ``"quadratic"``.
        unordered: Categories outside the scale, each maximally far from every other.

    Returns:
        The moments; ``None`` with no pairs.

    Raises:
        ValueError: As :func:`cohen_kappa`.
    """
    if len(categories) < 2:
        raise ValueError(f"kappa needs at least two categories; got {list(categories)}")
    if overlap := sorted(set(categories) & set(unordered)):
        raise ValueError(f"categories {overlap} cannot be both on the scale and off it")
    ordered = len(categories)
    index = {category: position for position, category in enumerate([*categories, *unordered])}
    stray = sorted({value for pair in pairs for value in pair if value not in index})
    if stray:
        raise ValueError(f"values {stray} are not among the categories {list(categories)} or {list(unordered)}")
    n = len(pairs)
    if n == 0:
        return None
    k = len(index)

    def cost(i: int, j: int) -> float:
        if i == j:
            return 0.0
        if i >= ordered or j >= ordered:
            return 1.0
        if weights == "quadratic":
            return (i - j) ** 2 / (ordered - 1) ** 2
        return 1.0

    first = [0] * k
    second = [0] * k
    costs = []
    for a, b in pairs:
        i, j = index[a], index[b]
        first[i] += 1
        second[j] += 1
        costs.append(cost(i, j))
    observed = sum(costs)
    observed_square = sum(c * c for c in costs)
    crossed = [(first[i] * second[j], cost(i, j)) for i in range(k) for j in range(k)]
    return KappaMoments(
        n=n,
        observed=observed / n,
        observed_square=observed_square / n,
        expected=sum(count * c for count, c in crossed) / (n * n),
        expected_square=sum(count * c * c for count, c in crossed) / (n * n),
        costs=tuple(costs),
    )


def cohen_kappa(
    pairs: Sequence[tuple[int, int]],
    categories: Sequence[int],
    *,
    weights: Literal["none", "quadratic"] = "none",
    unordered: Sequence[int] = (),
) -> float | None:
    """Cohen's kappa between two raters over the same items: agreement beyond what chance would give.

    ``1 - observed disagreement / expected disagreement``, where the expectation crosses the two
    raters' own marginal distributions. Unweighted, every disagreement costs the same; quadratic,
    a disagreement costs ``(i - j)² / (k - 1)²`` across ``k`` ordered categories, so a 4 against a
    5 is a near miss and a 1 against a 5 is not — the reading an ordinal judge's calibration wants.
    Over two categories the two weightings are the same number.

    An ``unordered`` category sits on no scale — an answer that is not a score at all, such as a
    judge declining to score — so it is at the greatest distance from every other category under
    either weighting (cost 1), and agrees only with itself (cost 0).

    Args:
        pairs: ``(first, second)`` per item.
        categories: Every category either rater could give, in order — the scale, not merely the
            values seen, because the quadratic cost is a distance along it.
        weights: ``"none"`` or ``"quadratic"``.
        unordered: Categories outside the scale, each maximally far from every other.

    Returns:
        Kappa, in ``[-1, 1]``; ``None`` with no pairs, or when chance alone predicts no
        disagreement (both raters gave one and the same category to every item), where the ratio
        is undefined rather than perfect.

    Raises:
        ValueError: A pair holds a value outside ``categories`` and ``unordered``, an ``unordered``
            category is also on the scale, or fewer than two categories.
    """
    moments = kappa_moments(pairs, categories, weights=weights, unordered=unordered)
    if moments is None or moments.expected == 0:
        return None
    return 1 - moments.observed / moments.expected


#: The family-wise correction every family of comparisons is adjusted by. Named so a surface can
#: state the method beside the adjusted figure rather than leaving a reader to guess which one ran.
MULTIPLE_COMPARISON_CORRECTION: Final = "holm"


def holm_adjust(p_values: Sequence[float], *, max_true: int | None = None) -> list[float]:
    """Holm-Bonferroni adjusted p-values for one family of comparisons, in the order given.

    Testing ten comparisons at α=0.05 each finds a "significant" one by chance in most families,
    so a verdict drawn from a family is read off the ADJUSTED p, which controls the probability
    that any verdict in the family is false at α. Holm's step-down is uniformly more powerful than
    plain Bonferroni and assumes nothing about how the comparisons depend on each other, which is
    the honest assumption for several measures read off the same cells.

    The ``i``-th smallest p (1-based) is multiplied by ``m - i + 1``, capped at 1, and made
    monotone by a running maximum, so an adjusted p is never smaller than one ranked below it.
    Comparing each adjusted p with α gives exactly Holm's rejection set.

    ``max_true`` caps the multiplier where the hypotheses are logically restricted so that no more than
    that many can be true at once — Shaffer's (1986) refinement. A family holding a difference test and an
    equivalence test of the same comparison is the case: the difference is either zero or at least the
    margin, never both, so ``m`` comparisons tested both ways hold at most ``m`` true nulls among their
    ``m + e`` hypotheses. The proof is Holm's: the first true null rejected comes after only false ones,
    so at most ``min(m + e − i + 1, max_true)`` nulls are true at that step, and that is the multiplier.

    Args:
        p_values: The raw p of every hypothesis in the family. The family is exactly
            these: a comparison that ran no test has no p and is not passed, since it is not a
            hypothesis this family tested.
        max_true: The most hypotheses that can be true together, or None for no restriction.

    Returns:
        One adjusted p per input, in input order. Empty for an empty family.

    Raises:
        ValueError: A p is not a probability, or ``max_true`` is below one.
    """
    stray = [p for p in p_values if not 0.0 <= p <= 1.0]
    if stray:
        raise ValueError(f"p-values must lie in [0, 1]; got {stray}")
    if max_true is not None and max_true < 1:
        raise ValueError(f"max_true must be at least 1; got {max_true}")
    m = len(p_values)
    order = sorted(range(m), key=lambda index: p_values[index])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        multiplier = m - rank if max_true is None else min(m - rank, max_true)
        running = max(running, min(1.0, multiplier * p_values[index]))
        adjusted[index] = running
    return adjusted


def hedges_j(df: int) -> float | None:
    """Hedges' small-sample factor ``J(df) = Γ(df/2) / (√(df/2) · Γ((df − 1)/2))`` (Hedges 1981).

    A standardized mean difference divides by a sample SD, and ``1/s`` overshoots ``1/σ`` on average, so
    the textbook estimator (Cohen's d) overstates the effect by the factor ``1/J``: 77% at two degrees
    of freedom, 25% at four, 9% at nine. Multiplying by ``J`` removes the bias exactly for normal data
    (``tests/test_simulated_separation.py``).

    Args:
        df: The degrees of freedom of the SD the effect is standardized by.

    Returns:
        ``J``, or ``None`` below two degrees of freedom: at one, ``E[1/s]`` diverges, so no factor makes
        the estimate unbiased and none is reported.
    """
    if df < 2:
        return None
    return math.exp(math.lgamma(df / 2.0) - math.lgamma((df - 1) / 2.0)) / math.sqrt(df / 2.0)


class SignificanceResult(NamedTuple):
    """A composite comparison's effect size, its verdict, and the p behind it.

    ``hedges_g`` is the bias-corrected standardized difference (:func:`hedges_j`): paired, the mean
    per-case difference over the SD of the differences (``g_z``); unpaired, the difference of means over
    the pooled SD (``g``). Named for what it is: Cohen's d, the uncorrected estimator, reads 0.88 for a
    true 0.5 at three paired cases.

    ``p_value`` is the number ``significant`` was thresholded against, carried
    rather than discarded so a surface can show the verdict *and* the statistic
    it rests on. A bare boolean cannot be checked by the reader, and a boolean
    reported without its statistic is indistinguishable from one no test
    produced — which is the distinction every reporting surface here is required
    to preserve.

    ``p_value`` is ``None`` exactly when no t-test was evaluated. It is never
    ``1.0`` as a stand-in for "no test": 1.0 is a real result (t=0, the two
    samples coincide) and using it for absence would make a run nobody tested
    read as one tested and found identical.
    """

    hedges_g: float | None
    significant: bool | None
    p_value: float | None


# Nothing was testable — fewer than two usable observations, a length mismatch,
# or a degenerate variance. Named so the branches below cannot drift apart on
# what "undefined" returns.
_UNTESTED = SignificanceResult(None, None, None)


class _TStatistic(NamedTuple):
    """The pieces of one two-sample t test: the difference it tests, its standard error, df, and effect size."""

    delta: float
    se: float
    df: float
    hedges_g: float | None


def _t_statistic(a: list[float], b: list[float], *, paired: bool) -> _TStatistic | SignificanceResult:
    """The statistic :func:`composite_significance` tests and :func:`difference_interval` bounds — one computation.

    Returns:
        The statistic, or the :class:`SignificanceResult` that stands for no test (too few observations,
        a length mismatch, or zero spread).
    """
    if paired:
        if len(a) != len(b) or len(a) < 2:
            return _UNTESTED
        diffs = [y - x for x, y in zip(a, b)]
        n = len(diffs)
        mean_diff = sum(diffs) / n
        sd = _sample_std(diffs)
        if sd == 0.0:
            return SignificanceResult(0.0, False, None) if mean_diff == 0.0 else _UNTESTED
        j = hedges_j(n - 1)
        return _TStatistic(mean_diff, sd / math.sqrt(n), float(n - 1), None if j is None else j * mean_diff / sd)
    if len(a) < 2 or len(b) < 2:
        return _UNTESTED
    na, nb = len(a), len(b)
    mean_a = sum(a) / na
    mean_b = sum(b) / nb
    sd_a = _sample_std(a)
    sd_b = _sample_std(b)
    pooled = math.sqrt(((na - 1) * sd_a**2 + (nb - 1) * sd_b**2) / (na + nb - 2))
    if pooled == 0.0:
        return SignificanceResult(0.0, False, None) if mean_a == mean_b else _UNTESTED
    j = hedges_j(na + nb - 2)
    se = math.sqrt(sd_a**2 / na + sd_b**2 / nb)
    if se == 0.0:
        return _UNTESTED
    # Hsu's degrees of freedom, not Welch–Satterthwaite's: see composite_significance.
    return _TStatistic(
        mean_b - mean_a, se, float(min(na, nb) - 1), None if j is None else j * (mean_b - mean_a) / pooled
    )


def composite_significance(
    sample_a: list[float],
    sample_b: list[float],
    *,
    paired: bool,
) -> SignificanceResult:
    """Effect size + significance for two composite-score samples.

    **Paired**: the one-sample t test on the per-case differences ``b − a`` on ``n − 1`` degrees of
    freedom — exact for normal differences.

    **Unpaired**: Welch's statistic (the difference of means over ``√(s_a²/n_a + s_b²/n_b)``) read on
    Hsu's conservative ``min(n_a, n_b) − 1`` degrees of freedom rather than the Welch–Satterthwaite
    estimate. Satterthwaite's df overstates what a two- or three-case side knows: by simulation, Welch's
    test called a difference that was not there 10% of the time at 2 vs 10 cases (11% when the small
    side's SD was 3×), 7% at 3 vs 10 (3×) and 5.6% at 5 vs 10 (3×), against a nominal 5%. On
    ``min(n) − 1`` df the test is conservative for normal data at every sample size and variance ratio
    (Mickey & Brown 1966); simulated, it held at most 4.3% across those designs. An exact permutation
    test was weighed and rejected: it is exact only when the two sides share one distribution, and the
    3×-SD design above is exactly where it is not. The price is power on the unpaired path — 0.53 → 0.48 at
    10 vs 10 cases and d = 1 — which falls on the fallback, not the paired design a fixed case set exists
    for.

    Args:
        sample_a: Run A's per-case composite scores. When ``paired``, aligned
            one-to-one with ``sample_b`` (same case order).
        sample_b: Run B's per-case composite scores.
        paired: True for the paired test on ``b - a`` differences; False for the unpaired test.

    Returns:
        A :class:`SignificanceResult`. Every field is ``None`` when the test is
        undefined — fewer than two usable observations, or zero variance with a
        non-zero mean difference (a deterministic constant gap has no finite
        effect size). A zero mean difference with zero variance returns
        ``(0.0, False, None)``: the samples are identical, which is definitively
        not a significant difference, but no t-statistic exists to quote — its
        denominator is zero — so the p stays absent rather than being invented.
        ``hedges_g`` is ``None`` beside a p where its SD has one degree of freedom (two pairs), where no
        unbiased effect size exists (:func:`hedges_j`).
    """
    statistic = _t_statistic([float(x) for x in sample_a], [float(x) for x in sample_b], paired=paired)
    if isinstance(statistic, SignificanceResult):
        return statistic
    p_value = _student_t_two_sided_p(statistic.delta / statistic.se, statistic.df)
    if math.isnan(p_value):
        return _UNTESTED
    return SignificanceResult(statistic.hedges_g, p_value < SIGNIFICANCE_ALPHA, p_value)


def difference_interval(
    sample_a: list[float],
    sample_b: list[float],
    *,
    paired: bool,
    confidence: float = INTERVAL_LEVEL,
    value_range: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    """The interval on ``mean(b) − mean(a)`` that :func:`composite_significance`'s test inverts.

    ``delta ± t · se`` on the test's own standard error and degrees of freedom — paired over the per-case
    differences, unpaired on Welch's SE and Hsu's df — so at ``confidence = 1 − α`` the interval excludes
    zero exactly when the test's p is below α.

    **With a declared range it is clipped to the differences the range allows**, ``± (high − low)``: two
    means on 0-1 cannot differ by more than 1, and a t interval on a few coarse cases runs past that (a
    pass/fail delta of +0.5 over six cases read ``[-0.07, 1.07]``). The true difference always lies inside
    the clip, so the clipped interval covers it whenever the unclipped one does, and zero, inside it too,
    is excluded exactly when it was. It is not clipped to ``[low, high]`` itself: a difference can run
    either way.

    Args:
        sample_a: The baseline's per-case values.
        sample_b: The compared side's, aligned with ``sample_a`` when ``paired``.
        paired: Which test the interval belongs to.
        confidence: The coverage, ``INTERVAL_LEVEL`` unless a family's correction asks for more.
        value_range: The measure's declared inclusive bounds, or None when it declares none, and then the
            interval is not clipped: no bound is known to clip it to.

    Returns:
        ``(low, high)``, or ``None`` wherever the test runs no t — too few observations, or no spread,
        where a zero-width interval would state certainty no data measured.
    """
    statistic = _t_statistic([float(x) for x in sample_a], [float(x) for x in sample_b], paired=paired)
    if isinstance(statistic, SignificanceResult):
        return None
    half = t_critical_two_sided(confidence, statistic.df) * statistic.se
    low, high = statistic.delta - half, statistic.delta + half
    if value_range is not None:
        span = value_range[1] - value_range[0]
        low, high = max(low, -span), min(high, span)
    return low, high


def _constant_split_p(n_a: int, n_b: int) -> float:
    """The exact two-sided permutation p of two unpaired samples, each constant and the two different.

    Of the ``C(n_a + n_b, n_a)`` equally likely ways to split the pooled values between the sides, only the
    two that put one side's values all above the other's are as extreme as the observed: ``2 / C(n_a + n_b,
    n_a)``. The unpaired counterpart of :func:`_sign_flip_p`, and the one p both :func:`separation_p` and
    :func:`level_difference` state for that pattern.
    """
    return min(1.0, 2.0 / math.comb(n_a + n_b, n_a))


def exact_decimal(value: float | Fraction) -> Fraction:
    """A value as the decimal it is written as, exactly — the one way the engine reads a value exactly.

    Over the float's own binary value, two cases written 0.1 and 0.6 differ by a hair over 0.5, and a
    constant per-case shift picks up a residue a t-test reads as a tiny, perfectly consistent spread: its p
    collapses to ~1e-113 where the exact sign-flip p is ``2 ** (1 - n)``. Over the shortest decimal that
    round-trips the float, they differ by exactly 0.5. A :class:`~fractions.Fraction` passes through, so a
    caller that already averaged exactly keeps its exact mean.

    Args:
        value: A float, or an exact value already.

    Returns:
        The exact rational.
    """
    return value if isinstance(value, Fraction) else Fraction(repr(float(value)))


def no_spread_p(a: Sequence[Fraction], b: Sequence[Fraction], *, paired: bool) -> float | None:
    """The exact permutation p where two exact samples have no spread to test, else ``None``.

    Paired, every difference one amount: ``2 ** (1 - n)`` (:func:`_sign_flip_p`), or 1 when that amount is
    zero. Unpaired, each side constant: ``2 / C(n_a + n_b, n_a)`` (:func:`_constant_split_p`), or 1 when the
    two constants agree. Decided on exact values (:func:`exact_decimal`), so a float residue cannot pass for a
    spread. The one reading of that pattern, shared by :func:`separation_p` and :func:`level_difference`.

    Args:
        a: One side's exact values, at least two.
        b: The other's, aligned with ``a`` when ``paired``.
        paired: Whether the two are one-to-one.

    Returns:
        The exact p, or ``None`` when the values have spread and a t-test reads them.
    """
    if paired:
        diffs = [y - x for x, y in zip(a, b)]
        if not _all_equal(diffs):
            return None
        return 1.0 if diffs[0] == 0 else _sign_flip_p(len(diffs))
    if not (_all_equal(a) and _all_equal(b)):
        return None
    return 1.0 if a[0] == b[0] else _constant_split_p(len(a), len(b))


class GuardrailVerdict(NamedTuple):
    """What :func:`guardrail_decision` came to: the decision, the interval it read, and how that interval was formed."""

    decision: GuardrailDecision
    interval: tuple[float, float] | None
    basis: Literal["t", "bounded"] | None


def guardrail_decision(
    control: Sequence[float],
    contrast: Sequence[float],
    *,
    paired: bool,
    margin: float | None,
    higher_is_better: bool,
    value_range: tuple[float, float] | None = None,
) -> GuardrailVerdict:
    """Decide one guardrail for an arm against the control: held, breached or undecided — non-inferiority.

    The bar rule (:func:`interval_clears`) read on the difference instead of a level: the interval on
    ``mean(contrast) − mean(control)`` at :data:`INTERVAL_LEVEL`, from the comparison test's own inversion
    (:func:`difference_interval`), against a line at zero change less the margin on the worse side. ``held``
    when the whole interval is on the good side of it — the arm is shown no worse than the control by more
    than the margin; ``breached`` when the whole interval is beyond it; ``undecided`` when it straddles the
    line or no interval exists. With no margin the line is zero change itself, so ``held`` needs the arm
    shown no worse at all. Each one-sided claim errs at most 2.5% of the time at the interval's coverage.

    **When every paired case moved by the same amount** the t interval has no width to give — the state of a
    guardrail at its ceiling, where both arms pass every case, and of a blatant regression, where every case
    flipped. For a reading with a declared range, the interval is then the one boundedness allows: no case
    moved otherwise in ``n``, so the share that could is at most ``1 − 0.025^(1/n)`` (Clopper–Pearson with no
    events, one-sided 2.5%), and each such case moves at most the full span of the scale either way. It is
    wide by design — a perfect record over fifteen cases does not show a rare failure absent — and with no
    range there is no such bound, so the guardrail is undecided.

    Args:
        control: The control's per-case values.
        contrast: The arm's, aligned with ``control`` when ``paired``.
        paired: Whether the two are one-to-one on the same cases.
        margin: The reading's declared margin, in its units, or None when it declares none.
        higher_is_better: Which way is better on the reading.
        value_range: The reading's declared inclusive bounds, or None when it declares none.

    Returns:
        The decision, the interval it read (None when none exists) and the interval's basis.
    """
    interval = difference_interval(list(control), list(contrast), paired=paired, value_range=value_range)
    basis: Literal["t", "bounded"] | None = "t" if interval is not None else None
    if interval is None and paired and value_range is not None and len(control) == len(contrast) >= 2:
        diffs = [float(b) - float(a) for a, b in zip(control, contrast)]
        if _sample_std(diffs) == 0.0:
            span = value_range[1] - value_range[0]
            moved = diffs[0]
            share = 1.0 - ((1.0 - INTERVAL_LEVEL) / 2.0) ** (1.0 / len(diffs))
            interval = (moved - (moved + span) * share, moved + (span - moved) * share)
            basis = "bounded"
    if interval is None:
        return GuardrailVerdict("undecided", None, None)
    cleared = interval_clears(interval, 0.0, margin=margin, higher_is_better=higher_is_better)
    decision: GuardrailDecision = "undecided" if cleared is None else ("held" if cleared else "breached")
    return GuardrailVerdict(decision, interval, basis)


def separation_p(
    sample_a: Sequence[float | Fraction], sample_b: Sequence[float | Fraction], *, paired: bool
) -> float | None:
    """The two-sided p of the separation test between two samples, where a test can decide.

    :func:`composite_significance`'s p — paired t on shared per-case values, Welch otherwise — and, where
    the values have no spread, the exact permutation p :func:`level_difference` reads for the same pattern,
    so one concept has one answer: every paired difference the same nonzero amount reads ``2^(1 − n)``
    (:func:`_sign_flip_p`), two unpaired sides each constant and different read ``2 / C(n_a + n_b, n_a)``
    (:func:`_constant_split_p`), and identical values (no gap, no spread) read 1.

    **The spread is decided on exact values** (:func:`exact_decimal`), the arithmetic
    :func:`level_difference` uses. Over floats, a constant shift such as ``i/10`` against ``i/10 + 0.5``
    carries a residue in its differences, and the t-test reads that residue as a tiny, perfectly consistent
    spread, with a p near 1e-113 where the exact one is ``2^(1 − n)``.

    **Where the exact p cannot reach α the answer is ``None`` — untested, never a p.** At three pairs the
    sign-flip p is 0.25 whatever the data: no test can decide, and :func:`level_difference` calls the same
    pattern untested. Stating the p would let a caller read the pattern as tested and not separated, which
    claims the data was asked and could not tell — a different statement from "no test could ask".

    Args:
        sample_a: One side's per-case values.
        sample_b: The other's, aligned with ``sample_a`` when ``paired``.
        paired: Whether the two are one-to-one on the same cases.

    Returns:
        The p, or ``None`` where no test can decide: fewer than two values a side, paired samples of
        different lengths, a spread that vanishes in floating point, or no spread over too few cases for the
        exact test to reach α.
    """
    a = [exact_decimal(x) for x in sample_a]
    b = [exact_decimal(y) for y in sample_b]
    if len(a) < 2 or len(b) < 2 or (paired and len(a) != len(b)):
        return None
    exact = no_spread_p(a, b, paired=paired)
    if exact is not None:
        return exact if exact == 1.0 or exact <= SIGNIFICANCE_ALPHA else None
    # The spread is exactly nonzero, so the t statistic exists unless its float residue vanishes.
    return composite_significance([float(x) for x in a], [float(y) for y in b], paired=paired).p_value


#: What a change between two paired samples reads as — see :class:`ChangeVerdict`.
ChangeLabel = Literal["improved", "regressed", "equivalent", "below_threshold", "not_separated", "untested"]


class ChangeVerdict(NamedTuple):
    """Whether the change between two paired samples is a real regression/improvement.

    A change earns a directional label only when it is BOTH statistically
    significant (a significant paired decline is the regression definition)
    AND large enough to clear a magnitude threshold — the joint gate that keeps a
    tiny-but-significant wobble in a large sample from reading as a regression. The
    numbers are carried so a surface can disclose the test and thresholds beside
    the label rather than presenting a bare verdict (descriptive, not an alert).

    **A move that misses significance is not "no change".** At two to six cases almost
    nothing is significant, so a label claiming stability there would be the default
    claim and a false one. "No meaningful change" is a claim of its own, made only by an
    equivalence test (TOST, Lakens 2017) against the measure's declared margin, and never
    without one.

    ``label`` is one of:

    - ``"regressed"`` — significant, over threshold, in the worse direction.
    - ``"improved"`` — significant, over threshold, in the better direction.
    - ``"equivalent"`` — no directional label, and the move is shown to lie inside ± the
      measure's declared margin: both one-sided tests reject at α, which is the 90% interval
      on the paired difference sitting inside the margin. The one label that claims no
      meaningful change.
    - ``"below_threshold"`` — significant, but under the caller's magnitude gate, and not
      shown equivalent: a real move too small to flag, never a claim of no change.
    - ``"not_separated"`` — not significant and not shown equivalent: the data cannot tell
      this move from noise, in either direction. Says nothing about whether it changed.
    - ``"untested"`` — no test could decide: fewer than two pairs, every case moved by the same
      nonzero amount over too few pairs for the exact sign-flip p to reach α, or a spread that
      vanishes in floating point. The engine-wide word for an undecidable reading, the one
      :func:`level_difference` and the pivot use; it says nothing about whether the measure changed.
    """

    label: ChangeLabel
    delta: float | None
    relative_delta: float | None
    significant: bool | None
    exceeds_threshold: bool | None
    hedges_g: float | None
    n_pairs: int
    #: The p the verdict was thresholded against, carried for the same reason the
    #: effect size is: a label a reader cannot check is an assertion. The t-test's p,
    #: or where the paired differences have no spread the exact sign-flip p
    #: (:func:`separation_p`'s reading of the same pattern). ``None`` when the verdict
    #: is ``untested``.
    p_value: float | None = None
    #: The margin the equivalence test runs against, in the measure's units — the
    #: measure's declared materiality threshold. ``None`` when none was declared, and
    #: then no equivalence test ran and no label claims one. Declared with no range, no
    #: test ran either, and ``equivalence_untested_reason`` says so.
    equivalence_margin: float | None = None
    #: The TOST p: the larger of the two one-sided p's, thresholded at α — each the bounded
    #: test's on the measure's declared range (:func:`paired_equivalence`). ``None`` wherever no equivalence test could decide — no
    #: margin, no declared range, fewer than two pairs, or a difference outside the declared range.
    equivalence_p: float | None = None
    #: Why a measure with a margin was not tested for equivalence at all: it declares no range
    #: (:data:`EQUIVALENCE_NEEDS_RANGE`, which names the remedy). ``None`` otherwise.
    equivalence_untested_reason: str | None = None


#: The largest fraction of its capital :func:`bounded_mean_p` stakes on one case: the betting fraction is capped
#: at this share of the most the bet could stake without risking ruin on a case at the bottom of the range.
#: Waudby-Smith and Ramdas suggest 1/2 or 3/4; 0.9 was chosen by simulation over the coarse supports the engine
#: produces (pass/fail and 1-5 differences, 5 to 50 pairs), where it reached equivalence at the fewest pairs
#: on agreeing and on noisy samples alike. The cap moves power only: any cap below 1 is valid.
_BET_CAP = 0.9


def bounded_mean_p(
    values: Sequence[float | Fraction],
    null_mean: float | Fraction,
    value_range: tuple[float, float],
    *,
    alpha: float = SIGNIFICANCE_ALPHA,
) -> float:
    """The one-sided p for H0 ``E[X] ≤ null_mean`` against ``E[X] > null_mean``, for any ``X`` inside ``value_range``.

    A test by betting (Waudby-Smith and Ramdas, "Estimating means of bounded random variables by betting",
    JRSS-B 2024, the predictable plug-in bet): rescaled to ``[0, 1]``, each value in turn multiplies a capital
    of 1 by ``1 + λ (x − m)``, where ``m`` is the rescaled null mean and the stake ``λ`` is set from the values
    before it, ``min(sqrt(2 ln(1/α) / (n σ̂²)), cap / m)``, with ``σ̂²`` their regularised variance. Under H0
    the capital is a nonnegative supermartingale started at 1, so by Ville's inequality it ever reaches ``1/α``
    with probability at most α, and the p is ``1 / max capital``. **That holds at every n and for every
    distribution on the range** — a two-point lattice, a pass/fail difference, a 1-5 difference with a rare
    four-point drop — which is what a t-test on coarse values cannot promise: it reads a sample of agreeing
    cases as having no spread, when a regression that broke one case in ten leaves ten agreeing cases one time
    in three.

    The stakes are set in the order the values come, so pass them in an order fixed before the values were
    seen (the engine's is the cases' sorted ids); a different fixed order is an equally valid test.

    Args:
        values: The observations, each inside ``value_range``.
        null_mean: The largest mean H0 allows.
        value_range: The declared inclusive bounds every observation lies in.
        alpha: The level the stake is tuned for.

    Returns:
        The p, in ``[0, 1]``.

    Raises:
        ValueError: The range is empty, or a value lies outside it.
    """
    low, high = exact_decimal(value_range[0]), exact_decimal(value_range[1])
    if high <= low:
        raise ValueError(f"value_range {value_range} is empty")
    exact = [exact_decimal(v) for v in values]
    if any(not low <= v <= high for v in exact):
        raise ValueError(f"a value lies outside value_range {value_range}")
    # Checked exactly above; the betting itself is float arithmetic, clamped so a rounding cannot leave the range.
    low_f, width = float(low), float(high - low)
    scaled = [min(1.0, max(0.0, (float(v) - low_f) / width)) for v in exact]
    m = float((exact_decimal(null_mean) - low) / (high - low))
    if m <= 0.0:
        # H0 puts every value at the bottom of the range, so one value above it refutes H0 outright.
        return 0.0 if any(z > 0.0 for z in scaled) else 1.0
    if m >= 1.0:
        return 1.0
    n = len(scaled)
    log_capital = best = 0.0
    total = squares = 0.0
    for t, z in enumerate(scaled):
        variance = (0.25 + squares) / (t + 1)
        stake = min(math.sqrt(2.0 * math.log(1.0 / alpha) / (n * variance)), _BET_CAP / m)
        log_capital += math.log1p(stake * (z - m))
        best = max(best, log_capital)
        total += z
        squares += (z - (0.5 + total) / (t + 2)) ** 2
    return min(1.0, math.exp(-best))


def paired_equivalence(
    diffs: Sequence[float | Fraction], margin: float | None, *, value_range: tuple[float, float] | None = None
) -> tuple[bool | None, float | None]:
    """The paired TOST against ``± margin``: whether the mean difference is shown inside it, and its p.

    Two one-sided tests on the paired differences, each at :data:`SIGNIFICANCE_ALPHA`: H0 ``δ ≤ −margin``
    and H0 ``δ ≥ margin``. Equivalence is claimed only when both reject — the larger of the two p's
    below α.

    **With the measure's declared range, each one-sided test is the bounded test by betting**
    (:func:`bounded_mean_p`) on differences that lie within ± the range's width. It holds α for every
    distribution of differences on that range, at every n — the guarantee coarse scores need. A one-sided
    t-test does not hold it there: at the margin, on a two-point lattice over 12 pairs it claimed
    ``equivalent`` 7% of the time, and where a regression broke a few cases outright (pass/fail differences
    of 0 save a rare −1) the t-test and the old exact sign-flip reading of a sample with no spread claimed it
    up to 30-60% of the time, because a sample of agreeing cases is what such a regression usually leaves.
    The price is the truth about coarse data: no valid test can show a mean inside a margin that is small
    against the range from a few cases. Even n cases that all agree give p ``(1 − margin / width) ** n`` at
    best (one case in ``width / margin`` could have dropped the full width unseen), so 1-5 rubric
    differences (width 4) within 0.5 need 23 agreeing pairs under any valid test, and 26 under this one. Below that the test
    still runs and its p says the data could not show equivalence, which claims nothing either way.

    **Without a declared range nothing is tested** (:data:`EQUIVALENCE_NEEDS_RANGE`). There is no
    finite-sample-valid test of a mean then (Bahadur and Savage 1956: an unbounded value can hide a rare large
    move), and the paired t TOST this used to fall back on claimed ``equivalent`` 11-13% of the time against a
    nominal 5% on skewed coarse values (#695). An ``equivalent`` reading is the one claim that two arms are
    alike, so it is never made at an error rate above α: a measure with a margin declares its range to be tested.

    Args:
        diffs: The paired differences, current minus baseline, in a fixed case order — exact
            (:class:`~fractions.Fraction`) where the caller read its samples exactly.
        margin: The declared margin, or None.
        value_range: The measure's declared inclusive bounds, or None when it declares none.

    Returns:
        ``(equivalent, p)``. Both None when no test could decide: no positive margin, no declared range
        (:func:`equivalence_untested_reason` says so), fewer than two pairs, or a difference outside ± the
        range's width (the declared range is contradicted, so its bound is not a bound).
    """
    n = len(diffs)
    if margin is None or margin <= 0.0 or value_range is None or n < 2:
        return None, None
    exact = [exact_decimal(d) for d in diffs]
    width = exact_decimal(value_range[1]) - exact_decimal(value_range[0])
    if any(abs(d) > width for d in exact):
        return None, None
    p = _bounded_tost_p(exact, margin, (float(-width), float(width)))
    return p < SIGNIFICANCE_ALPHA, p


def _bounded_tost_p(diffs: Sequence[Fraction], margin: float, bounds: tuple[float, float]) -> float:
    """The TOST p of :func:`paired_equivalence` on a declared range: the larger bounded one-sided p."""
    above_lower = bounded_mean_p(diffs, -exact_decimal(margin), bounds)
    below_upper = bounded_mean_p([-d for d in diffs], -exact_decimal(margin), bounds)
    return max(above_lower, below_upper)


def paired_change(
    baseline: list[float],
    current: list[float],
    *,
    min_absolute_change: float,
    min_relative_change: float,
    higher_is_better: bool,
    equivalence_margin: float | None = None,
    value_range: tuple[float, float] | None = None,
) -> ChangeVerdict:
    """Classify the change from ``baseline`` to ``current``: a direction, equivalence, or not separated.

    The two samples are paired one-to-one (same case order), so pass the per-case
    values aligned on the cases the two runs share. A regression is a significant
    paired move in the *worse* direction that also clears a magnitude threshold;
    an improvement is the same in the better direction. A move that earns neither
    reads ``"equivalent"`` only when the equivalence test shows it inside ± the
    declared margin; otherwise ``"below_threshold"`` when it was significant but
    under the gate, and ``"not_separated"`` when it was not — which claims nothing
    about whether the measure changed. A move no test can decide — fewer than two
    pairs, or a uniform move over too few pairs for the exact p to reach α — is
    ``"untested"``. Values are read exactly (:func:`exact_decimal`), so a float
    residue never passes for a spread.

    The magnitude gate passes when any *active* threshold is cleared: the absolute
    change clearing ``min_absolute_change`` OR the relative change (against the
    baseline mean) clearing ``min_relative_change``. A threshold of ``0.0`` turns
    that gate OFF — it stops being a criterion rather than passing everything, so
    setting only the absolute threshold is not silently nullified by an off
    relative one. With both off the gate imposes no magnitude floor and
    significance alone flags — honest because the thresholds ride on every answer,
    so a reader sees whether a magnitude floor was applied.

    The gate and the margin are different things on purpose. The gate is the
    caller's, per request: how large a move must be before this read flags it.
    The margin is the host's declaration about the measure: the difference too
    small to act on (:attr:`~threetears.evals.contracts.metrics.MetricDescriptor.materiality_threshold`),
    the same margin a bar's verdict is read against. Only the margin licenses a
    claim of no meaningful change.

    Args:
        baseline: Earlier run's per-case values.
        current: Later run's per-case values, aligned one-to-one with ``baseline``.
        min_absolute_change: Smallest absolute change that counts, in the measure's
            own unit. ``0.0`` turns the absolute gate off.
        min_relative_change: Smallest change relative to the baseline mean that
            counts, as a fraction. ``0.0`` turns the relative gate off. On a zero
            baseline a nonzero move reads as an unbounded change and clears it.
        higher_is_better: The measure's direction — ``True`` for quality (a decline
            is the regression), ``False`` for cost/latency (an increase is).
        equivalence_margin: The measure's declared margin, in its own unit, or None
            when it declares none — then no equivalence test runs and no label
            claims one.
        value_range: The measure's declared inclusive bounds, or None. The equivalence test reads it
            (:func:`paired_equivalence`): on a declared range its error rate holds at every n, and with
            none it does not run, so the move never reads ``"equivalent"`` and the verdict carries
            :data:`EQUIVALENCE_NEEDS_RANGE` as its ``equivalence_untested_reason``.

    Returns:
        A :class:`ChangeVerdict`. ``delta``/``relative_delta`` are ``None`` only
        when there are no pairs; ``significant`` is ``None`` below two pairs.
        ``p_value`` is the p the label was thresholded against, and is ``None``
        wherever the t-test was not evaluated — so a reader can tell a label that
        came from a test from one reasoned around it. ``equivalence_p`` is the same
        for the equivalence test.

    Raises:
        ValueError: The samples are not the same length — they must be pre-aligned
            by the caller, and a length mismatch is a pairing bug, not missing data.
    """
    if len(baseline) != len(current):
        raise ValueError(f"paired_change requires samples aligned one-to-one; got {len(baseline)} vs {len(current)}")

    # Read exactly (:func:`exact_decimal`), as :func:`separation_p` and :func:`level_difference` read: a
    # constant per-case shift such as ``i/10`` against ``i/10 + 0.5`` carries a float residue in its
    # differences that a t-test reads as a tiny, perfectly consistent spread (p near 1e-113), where the
    # exact sign-flip p is ``2 ** (1 - n)`` — and that residue also decided whether the deterministic-gap
    # branch ran at all.
    a = [exact_decimal(x) for x in baseline]
    b = [exact_decimal(x) for x in current]
    n_pairs = len(a)
    if n_pairs == 0:
        return ChangeVerdict("untested", None, None, None, None, None, 0, None, equivalence_margin)

    mean_a = sum(a, Fraction(0)) / n_pairs
    exact_delta = sum(b, Fraction(0)) / n_pairs - mean_a
    delta = float(exact_delta)
    relative = float(exact_delta / abs(mean_a)) if mean_a != 0 else None

    # Each gate at 0.0 is OFF (not a criterion), never a gate that passes
    # everything — otherwise OR-ing an active gate with an off one at 0.0 would
    # let the off gate nullify the active one. A gate contributes True/False only
    # when active; with no active gate there is no magnitude requirement, so
    # significance alone flags. A relative gate on a zero baseline reads a nonzero
    # move as an unbounded change (it clears) and a zero move as no change.
    absolute_gate = (abs(delta) >= min_absolute_change) if min_absolute_change > 0.0 else None
    if min_relative_change <= 0.0:
        relative_gate: bool | None = None
    elif relative is None:
        relative_gate = exact_delta != 0
    else:
        relative_gate = abs(relative) >= min_relative_change
    active_gates = [gate for gate in (absolute_gate, relative_gate) if gate is not None]
    exceeds = any(active_gates) if active_gates else True

    hedges_g: float | None
    significant: bool | None
    p_value: float | None
    exact = no_spread_p(a, b, paired=True) if n_pairs >= 2 else None
    if exact is None:
        hedges_g, significant, p_value = composite_significance(
            [float(x) for x in a], [float(y) for y in b], paired=True
        )
    elif exact == 1.0:
        # Every case moved by exactly nothing: definitively not separated, with the exact p of 1.
        hedges_g, significant, p_value = 0.0, False, 1.0
    elif exact <= SIGNIFICANCE_ALPHA:
        # A deterministic gap: every case moved by the same nonzero amount. No t exists (the
        # difference SD is zero), but the exact paired sign-flip test does: its p is ``2 ** (1 - n)``,
        # one of the ``2 ** n`` equally likely sign assignments in each tail, and here it clears α. The
        # p is carried, so the label is checkable; the effect size stays None (unbounded).
        hedges_g, significant, p_value = None, True, exact
    else:
        # The same pattern over too few pairs for the exact p to reach α — at three pairs it is 0.25
        # whatever the data. No test can decide, so the move is untested, never "not separated" (which
        # would say the data was asked and could not tell). Composite values live on a coarse lattice,
        # so "every case moved by exactly the same amount" is an ordinary coincidence at small n.
        hedges_g, significant, p_value = None, None, None
    equivalent, equivalence_p = paired_equivalence(
        [y - x for x, y in zip(a, b)], equivalence_margin, value_range=value_range
    )

    def verdict(label: ChangeLabel) -> ChangeVerdict:
        return ChangeVerdict(
            label,
            delta,
            relative,
            significant,
            exceeds,
            hedges_g,
            n_pairs,
            p_value,
            equivalence_margin,
            equivalence_p,
            equivalence_untested_reason(equivalence_margin, value_range),
        )

    if significant is None:
        # Genuinely untestable — fewer than two pairs, a uniform move below the
        # pair floor, or a spread that vanishes in floating point. Report the measured
        # delta if there is one, but never a directional label from a test that did not
        # run, and never equivalence.
        return ChangeVerdict("untested", delta, relative, None, exceeds, hedges_g, n_pairs, None, equivalence_margin)
    if significant and exceeds:
        return verdict("improved" if (delta > 0) == higher_is_better else "regressed")
    if equivalent:
        return verdict("equivalent")
    return verdict("below_threshold" if significant else "not_separated")


class LevelDifference(NamedTuple):
    """How one quantity differs between two levels, read over per-case values, and whether that separates.

    ``separated`` is three-valued, like every verdict here: True when the test rejects at α, False when it
    ran and did not (the data cannot tell the difference from noise, which says nothing about whether there
    is one), None when no test could run (``untested_reason`` says why). ``equivalent`` is the only field
    that claims the difference is small, and only against a declared margin (:func:`paired_equivalence`).
    """

    #: ``paired`` over the cases both levels carry when they share two or more; ``unpaired`` (Welch's
    #: statistic on Hsu's ``min(n_a, n_b) − 1`` df) over each level's own cases otherwise. None when no test ran.
    test: Literal["paired", "unpaired"] | None
    #: The cases read on each side: the shared cases when paired, each level's own otherwise.
    n_a: int
    n_b: int
    #: Mean of the per-case values read on each side; None when a side read none.
    mean_a: float | None
    mean_b: float | None
    #: ``mean_b - mean_a``; None when either side is empty.
    delta: float | None
    #: The standard error the test read ``delta`` against; None when no test ran.
    se: float | None
    #: The two-sided p, uncorrected. A t-test's, or an exact permutation p where the values have no
    #: spread (see :func:`level_difference`). None when no test ran.
    p_value: float | None
    separated: bool | None
    untested_reason: str | None
    #: The paired TOST against ``± equivalence_margin``; None when no margin, or the test is unpaired.
    equivalent: bool | None
    equivalence_p: float | None


def _all_equal(values: Sequence[float | Fraction]) -> bool:
    return all(value == values[0] for value in values)


def level_difference[Case: Hashable](
    values_a: Mapping[Case, float | Fraction],
    values_b: Mapping[Case, float | Fraction],
    *,
    equivalence_margin: float | None = None,
    value_range: tuple[float, float] | None = None,
) -> LevelDifference:
    """Test the difference between two levels of a quantity, over one value per case at each level.

    Pass per-case values — each case's mean over its repeats — so the test's unit is the case and repeats
    of one case are not counted as independent draws (with balanced repeats the SEM of case means is the
    cluster-robust SEM, :func:`clustered_standard_error`). The test is the one every between-level verdict
    here uses, and the same computation :func:`composite_significance` runs: a paired t-test over the cases
    both levels carry when they share at least two, which cancels the between-case spread both levels share;
    Welch's unequal-variance statistic over each level's cases otherwise. Each is read against Student's t on
    its own degrees of freedom (``n - 1`` paired; Hsu's conservative ``min(n_a, n_b) − 1`` unpaired, which
    holds α where Welch–Satterthwaite's does not — see :func:`composite_significance`), never a fixed
    multiple of the standard error: at three cases a level a fixed two standard errors calls a difference
    nearly 11% of the time when there is none.

    **A difference with no spread is read by an exact test, not by reasoning.** Every shared case moving by
    one nonzero amount (or two different constants, unpaired) leaves a t-test undefined. The exact
    permutation test is not: the observed arrangement is the most extreme of ``2 ** n`` equally likely sign
    flips (paired) or of ``C(n_a + n_b, n_a)`` splits (unpaired), so its two-sided p is ``2 ** (1 - n)``,
    or ``2 / C(n_a + n_b, n_a)``. Where that p cannot reach α the pattern is a coincidence the data cannot
    rule out — a 0/1 value moving the same way on two cases happens one time in eight by chance — so it is
    untested, never separated. Identical values on both sides (no gap, no spread) have the exact p of 1.

    Every value is read exactly (:func:`exact_decimal`) so a zero spread is decided exactly: two cases whose
    difference is the same decimal must not acquire a float residue a t-test would read as a tiny spread. A
    :class:`~fractions.Fraction` passes through, so a caller that averaged repeats exactly keeps that mean.

    Args:
        values_a: Case -> its value at the first level.
        values_b: Case -> its value at the second level.
        equivalence_margin: The measure's declared margin, or None. With one, a paired difference is also
            tested for equivalence (:func:`paired_equivalence`); an unpaired one never is.
        value_range: The quantity's declared inclusive bounds, or None. The equivalence test reads it
            (:func:`paired_equivalence`): on a declared range its error rate holds at every n; with none
            no equivalence test runs.

    Returns:
        A :class:`LevelDifference`.
    """
    shared = [case for case in values_a if case in values_b]
    paired = len(shared) >= 2
    a = [exact_decimal(values_a[case]) for case in shared] if paired else [exact_decimal(v) for v in values_a.values()]
    b = [exact_decimal(values_b[case]) for case in shared] if paired else [exact_decimal(v) for v in values_b.values()]
    n_a, n_b = len(a), len(b)
    mean_a = float(sum(a, Fraction(0)) / n_a) if n_a else None
    mean_b = float(sum(b, Fraction(0)) / n_b) if n_b else None
    delta = None if mean_a is None or mean_b is None else mean_b - mean_a
    test: Literal["paired", "unpaired"] = "paired" if paired else "unpaired"

    def untested(reason: str) -> LevelDifference:
        return LevelDifference(None, n_a, n_b, mean_a, mean_b, delta, None, None, None, reason, None, None)

    if n_a < 2 or n_b < 2:
        return untested("fewer than two cases on a side")
    equivalent: bool | None = None
    equivalence_p: float | None = None
    diffs = [y - x for x, y in zip(a, b)] if paired else []
    exact = no_spread_p(a, b, paired=paired)
    if exact is not None:
        if paired:
            # Decided on the exact differences, so the equivalence test reads no residue either.
            equivalent, equivalence_p = paired_equivalence(diffs, equivalence_margin, value_range=value_range)
        else:
            equivalent, equivalence_p = None, None
        if exact == 1.0:
            return LevelDifference(
                test, n_a, n_b, mean_a, mean_b, delta, 0.0, 1.0, False, None, equivalent, equivalence_p
            )
        if exact > SIGNIFICANCE_ALPHA:
            return untested(
                f"every shared case moved by the same amount, and over {len(diffs)} cases no exact test can "
                f"call that at α={SIGNIFICANCE_ALPHA}"
                if paired
                else f"each side's values are constant, and over {n_a} and {n_b} cases no exact test can call two "
                f"constants apart at α={SIGNIFICANCE_ALPHA}"
            )
        return LevelDifference(test, n_a, n_b, mean_a, mean_b, delta, 0.0, exact, True, None, None, None)
    if paired:
        equivalent, equivalence_p = paired_equivalence(diffs, equivalence_margin, value_range=value_range)
    # The spread is exactly nonzero, so the shared statistic exists; its float residue is all that could
    # still vanish, and then no t is quoted.
    statistic = _t_statistic([float(x) for x in a], [float(y) for y in b], paired=paired)
    if isinstance(statistic, SignificanceResult):
        return untested("the values' spread vanishes in floating point, so no t statistic exists")
    p = _student_t_two_sided_p(statistic.delta / statistic.se, statistic.df)
    return LevelDifference(
        test, n_a, n_b, mean_a, mean_b, delta, statistic.se, p, p < SIGNIFICANCE_ALPHA, None, equivalent, equivalence_p
    )


def _lower_incomplete_gamma(a: float, x: float) -> float:
    """The regularized lower incomplete gamma ``P(a, x)``: series below ``a + 1``, continued fraction above."""
    if x <= 0.0:
        return 0.0
    log_front = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1.0:
        term = total = 1.0 / a
        shape = a
        for _ in range(1000):
            shape += 1.0
            term *= x / shape
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return total * math.exp(log_front)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        step = d * c
        h *= step
        if abs(step - 1.0) < 1e-15:
            break
    return 1.0 - math.exp(log_front) * h


#: How many equal-probability points stand for the chi-square law of a spread estimate when a prediction
#: band integrates over it. 128 places the outermost at the 0.4% quantile; raised to 1,024, the band's ends
#: moved under 1%, and under 2% at three observations and a log-SD of 1.5, where the band spans eight orders.
_SPREAD_NODES = 128


@functools.cache
def _chi_square_nodes(df: int) -> tuple[float, ...]:
    """The chi-square law on ``df`` degrees of freedom as :data:`_SPREAD_NODES` equal-probability points."""
    nodes = []
    for index in range(_SPREAD_NODES):
        target = (index + 0.5) / _SPREAD_NODES
        low, high = 0.0, df + 100.0 * math.sqrt(df) + 100.0
        for _ in range(100):
            middle = 0.5 * (low + high)
            if _lower_incomplete_gamma(0.5 * df, 0.5 * middle) < target:
                low = middle
            else:
                high = middle
        nodes.append(0.5 * (low + high))
    return tuple(nodes)


def _log_sum_variance(spread: float, count: int) -> float:
    """``ln(1 + (e^σ² − 1) / m)``, the log-variance of a sum of ``m`` lognormals matched on two moments."""
    if spread < 30.0:
        return math.log1p(math.expm1(spread) / count)
    return spread - math.log(count) + math.log1p((count - 1) * math.exp(-spread))


def lognormal_sum_prediction_band(history: Sequence[float], n_future: int) -> tuple[float, float] | None:
    """A prediction band at :data:`INTERVAL_LEVEL` for the TOTAL of ``n_future`` new draws like ``history``.

    For positive, right-skewed quantities — what a turn costs — read on the log scale, where they are
    near normal. The logs of the history give a mean and a spread; the band is the predictive law of the
    future total under a lognormal with that mean and spread unknown (the reference prior, ``1/σ``): the
    spread's uncertainty is its chi-square law on ``n − 1`` degrees of freedom, the mean's is normal given
    the spread, and a sum of ``m`` lognormals is matched on its first two moments (Fenton–Wilkinson).
    For one future draw this is exactly the log-scale t prediction interval.

    Why not the normal-theory band ``t · s · sqrt(m + m²/n)``: costs are positive and their tail is long.
    With five historical observations of a lognormal at log-SD 1.0, that band covered a 15-observation
    sweep's total 82% of the time; this one 96%. It holds within a point or two of 95% on normal costs
    too (the normal band's own case), and runs wide on lighter tails (gamma, exponential) rather than narrow.

    **It treats every historical observation as an independent draw**, and so does the sweep it predicts.
    Repeats of one case are not: where cases cost differently and each is repeated, the history's spread
    and the sweep's both cluster, and the band is narrower than it should be.

    Args:
        history: Past observations, each positive.
        n_future: How many new observations the total sums, at least one.

    Returns:
        ``(low, high)``, or ``None`` with fewer than two observations or any that is not positive — the
        log scale cannot read a zero, and a caller that has some falls back to a band that can.

    Raises:
        ValueError: ``n_future`` is below one.
    """
    if n_future < 1:
        raise ValueError(f"a prediction needs at least one future observation, got {n_future}")
    n = len(history)
    if n < 2 or any(value <= 0.0 for value in history):
        return None
    logs = [math.log(value) for value in history]
    centre = math.fsum(logs) / n
    spread = math.fsum((value - centre) ** 2 for value in logs) / (n - 1)
    if spread == 0.0:
        total = n_future * math.exp(centre)
        return total, total
    df = n - 1
    components = []
    for node in _chi_square_nodes(df):
        variance = df * spread / node
        sum_variance = _log_sum_variance(variance, n_future)
        components.append(
            (
                math.log(n_future) + centre + 0.5 * variance - 0.5 * sum_variance,
                math.sqrt(sum_variance + variance / n),
            )
        )
    tail = 0.5 * (1.0 - INTERVAL_LEVEL)

    root_two = math.sqrt(2.0)
    root_two_pi = math.sqrt(2.0 * math.pi)

    def cdf_and_density(log_total: float) -> tuple[float, float]:
        mass = density = 0.0
        for mean, scale in components:
            z = (log_total - mean) / scale
            mass += 0.5 * math.erfc(-z / root_two)
            density += math.exp(-0.5 * z * z) / (scale * root_two_pi)
        return mass / len(components), density / len(components)

    def quantile(p: float) -> float:
        # Newton's method on the mixture's CDF, kept inside a bracket that every component's own quantile
        # bounds, and bisecting whenever a step would leave it.
        z = NormalDist().inv_cdf(p)
        low = min(mean + scale * z for mean, scale in components)
        high = max(mean + scale * z for mean, scale in components)
        guess = 0.5 * (low + high)
        for _ in range(100):
            mass, density = cdf_and_density(guess)
            if mass < p:
                low = guess
            else:
                high = guess
            step = guess - (mass - p) / density if density > 0.0 else 0.5 * (low + high)
            if not low < step < high:
                step = 0.5 * (low + high)
            if abs(step - guess) < 1e-12 * max(1.0, abs(guess)):
                return math.exp(step)
            guess = step
        return math.exp(guess)

    return quantile(tail), quantile(1.0 - tail)


#: The power a detectable difference is stated at: the smallest difference the planned comparison finds four
#: times in five. The conventional planning figure (Cohen 1988); the estimate names it beside every number.
DETECTABLE_POWER: Final = 0.8


def paired_t_power(n_pairs: int, effect: float, difference_sd: float, *, alpha: float) -> float:
    """The power of the two-sided paired t-test on ``n_pairs`` differences against a true mean difference ``effect``.

    The noncentral t: ``P(|T'| > c)`` with ``n − 1`` df, noncentrality ``effect · √n / difference_sd`` and ``c``
    the two-sided critical value at ``alpha`` (:func:`t_critical_two_sided`). Integrated over the chi-square
    law of the differences' variance, ``E[Φ(λ − c·√(V/df)) + Φ(−λ − c·√(V/df))]``, on the
    :data:`_SPREAD_NODES` equal-probability points :func:`_chi_square_nodes` places — the law the lognormal
    cost band integrates over. Exact for normal differences up to that quadrature (under 0.2 points of power
    against a 4,000-panel Simpson integral, ``tests/test_power_preflight.py``).

    Args:
        n_pairs: The pairs the test reads, at least two.
        effect: The true mean difference.
        difference_sd: The true standard deviation of one pair's difference, positive.
        alpha: The two-sided level the test rejects at.

    Returns:
        The probability the test rejects.

    Raises:
        ValueError: Fewer than two pairs, or a spread that is not positive.
    """
    if n_pairs < 2:
        raise ValueError(f"a paired t-test needs at least two pairs, got {n_pairs}")
    if difference_sd <= 0.0:
        raise ValueError(f"the differences' standard deviation must be positive, got {difference_sd}")
    df = n_pairs - 1
    critical = t_critical_two_sided(1.0 - alpha, df)
    noncentrality = abs(effect) * math.sqrt(n_pairs) / difference_sd
    normal = NormalDist()
    total = 0.0
    for node in _chi_square_nodes(df):
        scaled = critical * math.sqrt(node / df)
        total += normal.cdf(noncentrality - scaled) + normal.cdf(-noncentrality - scaled)
    return total / _SPREAD_NODES


def paired_detectable_difference(
    n_pairs: int, difference_sd: float, *, alpha: float, power: float = DETECTABLE_POWER
) -> float:
    """The smallest true mean difference the two-sided paired t-test on ``n_pairs`` pairs finds with ``power``.

    The inverse of :func:`paired_t_power` in the effect, by bisection: the power rises monotonically in
    ``|effect|``. It is in the unit of ``difference_sd``.

    Args:
        n_pairs: The pairs the test reads, at least two.
        difference_sd: The standard deviation of one pair's difference, positive.
        alpha: The two-sided level each comparison is rejected at — ``α/m`` for the first step of Holm's
            correction over ``m`` comparisons.
        power: The power the difference is found with.

    Returns:
        The difference.

    Raises:
        ValueError: As :func:`paired_t_power`, or a ``power`` outside ``(alpha, 1)``.
    """
    if not alpha < power < 1.0:
        raise ValueError(f"power must lie between alpha and 1, got {power}")
    low, high = 0.0, difference_sd
    while paired_t_power(n_pairs, high, difference_sd, alpha=alpha) < power:
        low, high = high, 2.0 * high
    for _ in range(60):
        middle = 0.5 * (low + high)
        if paired_t_power(n_pairs, middle, difference_sd, alpha=alpha) < power:
            low = middle
        else:
            high = middle
    return high


class VarianceComponents(NamedTuple):
    """How one reading varies across cases and across repeats of one case, estimated from earlier runs.

    ``within_case`` is the pooled variance of repeats around their case's mean; ``between_case`` the variance
    of case levels around the run's mean, with the repeat noise each case mean carries taken out (the
    one-way random-effects method of moments, floored at zero). Both are variances, in the reading's unit
    squared.
    """

    within_case: float
    #: The degrees of freedom behind ``within_case``: every repeat beyond a case's first, over every run.
    within_df: int
    between_case: float
    #: The cases behind ``between_case``, summed over the runs that had two or more.
    n_cases: int
    #: How many cases (over every run) were repeated, and so carried a within-case spread.
    n_repeated_cases: int


def variance_components(runs: Sequence[Sequence[Sequence[float]]]) -> VarianceComponents | None:
    """Estimate one reading's within-case and between-case variance from earlier runs.

    Each run is its cases, each case its repeats' values. The within-case variance pools every repeated case's
    sample variance by its degrees of freedom. The between-case variance is estimated per run as the variance
    of its case means less the share of it the repeat noise explains (``within_case`` times the mean of
    ``1/k`` over its cases), floored at zero, and pooled over runs by ``cases − 1``. A run is one arm, so its
    case means vary only by case and by repeat noise; two arms are never pooled into one case's level.

    Args:
        runs: Per run, per case, the values of that case's repeats.

    Returns:
        The components, or ``None`` when fewer than two cases were repeated (no within-case spread can be
        told from case-to-case spread then) or no run carries two cases.
    """
    within_ss = 0.0
    within_df = 0
    repeated = 0
    for run in runs:
        for case in run:
            if len(case) >= 2:
                mean = math.fsum(case) / len(case)
                within_ss += math.fsum((value - mean) ** 2 for value in case)
                within_df += len(case) - 1
                repeated += 1
    if repeated < 2:
        return None
    within = within_ss / within_df
    between_weighted = 0.0
    between_df = 0
    n_cases = 0
    for run in runs:
        cases = [case for case in run if case]
        if len(cases) < 2:
            continue
        means = [math.fsum(case) / len(case) for case in cases]
        spread = _sample_std(means) ** 2
        noise = within * math.fsum(1.0 / len(case) for case in cases) / len(cases)
        between_weighted += (len(cases) - 1) * max(0.0, spread - noise)
        between_df += len(cases) - 1
        n_cases += len(cases)
    if between_df == 0:
        return None
    return VarianceComponents(within, within_df, between_weighted / between_df, n_cases, repeated)


def paired_case_variance(
    pairs: Sequence[tuple[Sequence[Sequence[float]], Sequence[Sequence[float]]]], within_case: float
) -> tuple[float, int] | None:
    """The between-case variance of a paired difference: how much two arms disagree about the same case.

    Each pair is two earlier runs' repeats over the cases both ran, aligned by case. A case's difference of
    means varies by how differently the two arms find that case (what pairing does not cancel) and by both
    sides' repeat noise; the noise's share, ``within_case · (1/k_a + 1/k_b)`` averaged over the cases, is taken
    out of the differences' sample variance, floored at zero, and the pairs pooled by ``cases − 1``.

    Args:
        pairs: Per pair of runs, ``(one run's cases, the other's)``, case-aligned, each case its repeats' values.
        within_case: The pooled within-case variance (:func:`variance_components`).

    Returns:
        ``(variance, cases behind it)``, or ``None`` when no pair shares two cases.
    """
    weighted = 0.0
    df = 0
    n_cases = 0
    for left, right in pairs:
        aligned = [(a, b) for a, b in zip(left, right, strict=True) if a and b]
        if len(aligned) < 2:
            continue
        diffs = [math.fsum(b) / len(b) - math.fsum(a) / len(a) for a, b in aligned]
        noise = within_case * math.fsum(1.0 / len(a) + 1.0 / len(b) for a, b in aligned) / len(aligned)
        weighted += (len(aligned) - 1) * max(0.0, _sample_std(diffs) ** 2 - noise)
        df += len(aligned) - 1
        n_cases += len(aligned)
    return None if df == 0 else (weighted / df, n_cases)


__all__ = [
    "BAR_SEED_HALF_WIDTH_FRACTION",
    "DETECTABLE_POWER",
    "EQUIVALENCE_NEEDS_RANGE",
    "EQUIVALENCE_TEST_NAME",
    "GuardrailVerdict",
    "INTERVAL_LEVEL",
    "MIN_PAIRS_FOR_DETERMINISTIC_GAP",
    "MULTIPLE_COMPARISON_CORRECTION",
    "PAIRED_TEST_NAME",
    "SIGNIFICANCE_ALPHA",
    "UNPAIRED_TEST_NAME",
    "ChangeLabel",
    "ChangeVerdict",
    "KappaMoments",
    "LevelDifference",
    "SignificanceResult",
    "VarianceComponents",
    "bar_seed",
    "bounded_mean_p",
    "case_rate_interval",
    "ci_half_width",
    "clustered_standard_error",
    "cohen_kappa",
    "composite_significance",
    "difference_interval",
    "equivalence_untested_reason",
    "exact_decimal",
    "guardrail_decision",
    "hedges_j",
    "holm_adjust",
    "interval_clears",
    "kappa_moments",
    "level_difference",
    "lognormal_sum_prediction_band",
    "mean_interval",
    "no_spread_p",
    "observed_mean_interval",
    "paired_case_variance",
    "paired_change",
    "paired_detectable_difference",
    "paired_equivalence",
    "paired_t_power",
    "proportion_interval",
    "separation_p",
    "standard_error_of_mean",
    "t_critical_two_sided",
    "variance_components",
    "wilson_interval",
]
