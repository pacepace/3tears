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
an effect size (Cohen's d), the p-value, and a significance flag to each
per-model composite delta. The p travels with the flag rather than being
consumed and dropped: a verdict a reader cannot check against the number it was
thresholded on is indistinguishable from one no test produced.
Composite scores are continuous per-case quality values in ``[0, 1]``
(see :func:`threetears.evals.contracts.scoring.compute_per_case_composites`). Samples are
*paired* by case when the two runs actually scored the same frozen
``test_case_id`` s (a paired t-test, far more powerful); when they scored
different cases — including two runs of one template whose case sets do not
intersect — they are two independent samples (Welch's unequal-variance t-test).
The caller decides which, and the two effect sizes are not interchangeable:
paired yields Cohen's d_z over the difference SD, unpaired Cohen's d over the
pooled SD.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from collections.abc import Sequence
from typing import Final, Literal, NamedTuple

# Two-sided p-value below which a composite delta is called significant.
SIGNIFICANCE_ALPHA = 0.05


def _min_pairs_for_sign_flip(alpha: float) -> int:
    """Smallest ``n`` whose exact paired sign-flip test can reach ``alpha``.

    Under the null of exchangeable signs there are ``2 ** n`` equally likely sign
    assignments, so the smallest attainable two-sided p is ``2 ** (1 - n)``. This
    returns the first ``n`` for which that lands at or below ``alpha`` — computed
    rather than written down, so it tracks the threshold instead of drifting from
    it.
    """
    n = 2
    while 2.0 ** (1 - n) > alpha:
        n += 1
    return n


#: Pair-count floor for claiming significance from a zero-variance difference.
#: Derived from the alpha above, not chosen: below it, no exact test of a perfectly
#: consistent move could reject at that alpha, so the claim would outrun the data.
_MIN_PAIRS_FOR_DETERMINISTIC_GAP = _min_pairs_for_sign_flip(SIGNIFICANCE_ALPHA)

# The paired test the change classifier discloses, so a regression flag names the
# statistics it rests on rather than presenting a bare verdict.
PAIRED_TEST_NAME = f"paired two-sided t-test on shared per-case values, α={SIGNIFICANCE_ALPHA}"

# The test that runs when the two samples cannot be paired — no shared frozen
# case set, so the cases on each side are different questions. Named beside the
# paired one because a surface that discloses only "t-test" leaves a reader
# unable to tell a powerful within-case comparison from a weak between-case one,
# and that difference is most of what a small eval arm's verdict rests on.
UNPAIRED_TEST_NAME = f"Welch's unequal-variance two-sided t-test on unpaired per-case values, α={SIGNIFICANCE_ALPHA}"


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


def t_critical_two_sided(confidence: float, df: float) -> float:
    """The two-sided t multiplier for a confidence level on ``df`` degrees of freedom.

    The inverse of :func:`_student_t_two_sided_p`, found by bisection rather than by a
    closed form — the forward function is already exact here, and a table of critical
    values would be a second source of truth that could drift from it.

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


def ci_half_width(sem: float, n: int) -> float | None:
    """Half-width of the reported interval on a mean at :data:`INTERVAL_LEVEL`: the t multiplier times the SEM.

    Returns ``None`` rather than 0.0 below two observations, taking the same position
    :func:`standard_error_of_mean` already takes: the spread is *unestimable* there, not zero. A 0.0
    would render as a zero-width 95% interval — a point estimate wearing a confidence label, which is
    the strongest possible claim made from the least possible evidence.

    The multiplier is t, not a fixed 1.96: an arm can be three observations, where the honest 95%
    multiplier is ~4.30, and quoting the large-sample constant would publish an interval at under half
    its true width while labelling it "95%".

    Args:
        sem: The standard error of the mean.
        n: Observations behind it (``n - 1`` degrees of freedom).

    Returns:
        The half-width, or ``None`` where no interval is estimable.
    """
    if n < 2:
        return None
    return t_critical_two_sided(INTERVAL_LEVEL, n - 1) * sem


def wilson_interval(n_true: int, n: int) -> tuple[float, float] | None:
    """The Wilson score interval on a proportion at :data:`INTERVAL_LEVEL` — how a boolean measure's rate is bounded.

    Wilson rather than the normal approximation because a boolean measure's rate sits at 0 or 1
    exactly when it is most interesting (every encounter on target, none), where the normal interval
    collapses to a zero-width point and reads as certainty from three observations. Wilson stays
    inside [0, 1] and keeps a width at the ends.

    Args:
        n_true: Observations that held.
        n: Observations.

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
    rate = n_true / n
    denominator = 1 + z * z / n
    centre = (rate + z * z / (2 * n)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


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
    observed = 0.0
    for a, b in pairs:
        i, j = index[a], index[b]
        first[i] += 1
        second[j] += 1
        observed += cost(i, j)
    observed /= n
    expected = sum(first[i] * second[j] * cost(i, j) for i in range(k) for j in range(k)) / (n * n)
    if expected == 0:
        return None
    return 1 - observed / expected


#: The family-wise correction every family of comparisons is adjusted by. Named so a surface can
#: state the method beside the adjusted figure rather than leaving a reader to guess which one ran.
MULTIPLE_COMPARISON_CORRECTION: Final = "holm"


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values for one family of comparisons, in the order given.

    Testing ten comparisons at α=0.05 each finds a "significant" one by chance in most families,
    so a verdict drawn from a family is read off the ADJUSTED p, which controls the probability
    that any verdict in the family is false at α. Holm's step-down is uniformly more powerful than
    plain Bonferroni and assumes nothing about how the comparisons depend on each other, which is
    the honest assumption for several measures read off the same cells.

    The ``i``-th smallest p (1-based) is multiplied by ``m - i + 1``, capped at 1, and made
    monotone by a running maximum, so an adjusted p is never smaller than one ranked below it.
    Comparing each adjusted p with α gives exactly Holm's rejection set.

    Args:
        p_values: The raw two-sided p of every comparison in the family. The family is exactly
            these: a comparison that ran no test has no p and is not passed, since it is not a
            hypothesis this family tested.

    Returns:
        One adjusted p per input, in input order. Empty for an empty family.

    Raises:
        ValueError: A p is not a probability.
    """
    stray = [p for p in p_values if not 0.0 <= p <= 1.0]
    if stray:
        raise ValueError(f"p-values must lie in [0, 1]; got {stray}")
    m = len(p_values)
    order = sorted(range(m), key=lambda index: p_values[index])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


class SignificanceResult(NamedTuple):
    """A composite comparison's effect size, its verdict, and the p behind it.

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

    cohens_d: float | None
    significant: bool | None
    p_value: float | None


# Nothing was testable — fewer than two usable observations, a length mismatch,
# or a degenerate variance. Named so the branches below cannot drift apart on
# what "undefined" returns.
_UNTESTED = SignificanceResult(None, None, None)


def composite_significance(
    sample_a: list[float],
    sample_b: list[float],
    *,
    paired: bool,
) -> SignificanceResult:
    """Effect size + significance for two composite-score samples.

    Args:
        sample_a: Run A's per-case composite scores. When ``paired``, aligned
            one-to-one with ``sample_b`` (same case order).
        sample_b: Run B's per-case composite scores.
        paired: True to run a paired t-test on ``b - a`` differences (Cohen's
            ``d_z`` = mean(diff) / sd(diff)); False for Welch's unequal-variance
            t-test (Cohen's d over the pooled SD).

    Returns:
        A :class:`SignificanceResult`. Every field is ``None`` when the test is
        undefined — fewer than two usable observations, or zero variance with a
        non-zero mean difference (a deterministic constant gap has no finite
        effect size). A zero mean difference with zero variance returns
        ``(0.0, False, None)``: the samples are identical, which is definitively
        not a significant difference, but no t-statistic exists to quote — its
        denominator is zero — so the p stays absent rather than being invented.
    """
    a = [float(x) for x in sample_a]
    b = [float(x) for x in sample_b]

    if paired:
        if len(a) != len(b) or len(a) < 2:
            return _UNTESTED
        diffs = [y - x for x, y in zip(a, b)]
        n = len(diffs)
        mean_diff = sum(diffs) / n
        sd = _sample_std(diffs)
        if sd == 0.0:
            return SignificanceResult(0.0, False, None) if mean_diff == 0.0 else _UNTESTED
        cohens_d = mean_diff / sd
        t_stat = mean_diff / (sd / math.sqrt(n))
        df = float(n - 1)
    else:
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
        cohens_d = (mean_b - mean_a) / pooled
        var_a, var_b = sd_a**2 / na, sd_b**2 / nb
        se = math.sqrt(var_a + var_b)
        t_stat = (mean_b - mean_a) / se
        # Welch–Satterthwaite degrees of freedom.
        df = (var_a + var_b) ** 2 / (var_a**2 / (na - 1) + var_b**2 / (nb - 1))

    p_value = _student_t_two_sided_p(t_stat, df)
    if math.isnan(p_value) or not math.isfinite(cohens_d):
        return _UNTESTED
    return SignificanceResult(cohens_d, p_value < SIGNIFICANCE_ALPHA, p_value)


class ChangeVerdict(NamedTuple):
    """Whether the change between two paired samples is a real regression/improvement.

    A change earns a directional label only when it is BOTH statistically
    significant (a significant paired decline is the regression definition)
    AND large enough to clear a magnitude threshold — the joint gate that keeps a
    tiny-but-significant wobble in a large sample from reading as a regression. The
    numbers are carried so a surface can disclose the test and thresholds beside
    the label rather than presenting a bare verdict (descriptive, not an alert).

    ``label`` is one of:

    - ``"regressed"`` — significant, over threshold, in the worse direction.
    - ``"improved"`` — significant, over threshold, in the better direction.
    - ``"flat"`` — measured, but not significant or not over threshold.
    - ``"inconclusive"`` — too few paired observations to run the test (< 2 pairs).
    """

    label: str
    delta: float | None
    relative_delta: float | None
    significant: bool | None
    exceeds_threshold: bool | None
    cohens_d: float | None
    n_pairs: int
    #: The p the verdict was thresholded against, carried for the same reason the
    #: effect size is: a label a reader cannot check is an assertion. ``None``
    #: when no t-test was evaluated — including the deterministic-gap case below,
    #: where the label is reasoned from the zero variance rather than from a t.
    p_value: float | None = None


def paired_change(
    baseline: list[float],
    current: list[float],
    *,
    min_absolute_change: float,
    min_relative_change: float,
    higher_is_better: bool,
) -> ChangeVerdict:
    """Classify the change from ``baseline`` to ``current`` as regression / improvement / flat.

    The two samples are paired one-to-one (same case order), so pass the per-case
    values aligned on the cases the two runs share. A regression is a significant
    paired move in the *worse* direction that also clears a magnitude threshold;
    an improvement is the same in the better direction. Below either gate the
    change is ``"flat"`` — real noise, not a finding — and fewer than two pairs is
    ``"inconclusive"`` because the test is undefined, never silently ``"flat"``.

    The magnitude gate passes when any *active* threshold is cleared: the absolute
    change clearing ``min_absolute_change`` OR the relative change (against the
    baseline mean) clearing ``min_relative_change``. A threshold of ``0.0`` turns
    that gate OFF — it stops being a criterion rather than passing everything, so
    setting only the absolute threshold is not silently nullified by an off
    relative one. With both off the gate imposes no magnitude floor and
    significance alone flags — honest because the thresholds ride on every answer,
    so a reader sees whether a magnitude floor was applied.

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

    Returns:
        A :class:`ChangeVerdict`. ``delta``/``relative_delta`` are ``None`` only
        when there are no pairs; ``significant`` is ``None`` below two pairs.
        ``p_value`` is the p the label was thresholded against, and is ``None``
        wherever the t-test was not evaluated — so a reader can tell a label that
        came from a test from one reasoned around it.

    Raises:
        ValueError: The samples are not the same length — they must be pre-aligned
            by the caller, and a length mismatch is a pairing bug, not missing data.
    """
    if len(baseline) != len(current):
        raise ValueError(f"paired_change requires samples aligned one-to-one; got {len(baseline)} vs {len(current)}")

    a = [float(x) for x in baseline]
    b = [float(x) for x in current]
    n_pairs = len(a)
    if n_pairs == 0:
        return ChangeVerdict("inconclusive", None, None, None, None, None, 0)

    mean_a = sum(a) / n_pairs
    mean_b = sum(b) / n_pairs
    delta = mean_b - mean_a
    relative = (delta / abs(mean_a)) if mean_a != 0 else None

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
        relative_gate = delta != 0.0
    else:
        relative_gate = abs(relative) >= min_relative_change
    active_gates = [gate for gate in (absolute_gate, relative_gate) if gate is not None]
    exceeds = any(active_gates) if active_gates else True

    tested = composite_significance(a, b, paired=True)
    cohens_d, significant, p_value = tested
    if (
        significant is None
        and n_pairs >= _MIN_PAIRS_FOR_DETERMINISTIC_GAP
        and _sample_std([y - x for x, y in zip(a, b)]) == 0.0
        and delta != 0.0
    ):
        # A deterministic gap: every case moved by the same amount. The paired
        # t-test is undefined (the difference SD is zero, so its t-statistic
        # divides by zero) and `composite_significance` reports no finite effect
        # size — but a perfectly consistent change is strong evidence of a real
        # move, so a regression flag calls it significant rather than
        # inconclusive. Effect size stays None (unbounded).
        #
        # The pair-count floor is what keeps that reasoning honest. With every
        # case moving the same way, the sharpest claim the data can support is the
        # exact paired sign-flip test, whose smallest attainable two-sided p is
        # ``2 ** (1 - n)`` — one of the 2**n equally likely sign assignments in
        # each tail. Below the floor even a perfect gap cannot clear alpha, so
        # calling it significant would assert something no test could establish.
        # Composite values also live on a coarse lattice (a mean over a 5-point
        # rubric), which makes "every case moved by exactly the same amount" an
        # ordinary coincidence at small n rather than a finding.
        significant = True
    if significant is None:
        # Genuinely untestable — fewer than two pairs. Report the measured delta if
        # there is one, but never a directional label from a test that did not run.
        return ChangeVerdict("inconclusive", delta, relative, None, exceeds, cohens_d, n_pairs, p_value)

    if not (significant and exceeds):
        return ChangeVerdict("flat", delta, relative, significant, exceeds, cohens_d, n_pairs, p_value)

    improved = (delta > 0) == higher_is_better
    return ChangeVerdict(
        "improved" if improved else "regressed", delta, relative, significant, exceeds, cohens_d, n_pairs, p_value
    )


__all__ = [
    "INTERVAL_LEVEL",
    "MULTIPLE_COMPARISON_CORRECTION",
    "PAIRED_TEST_NAME",
    "SIGNIFICANCE_ALPHA",
    "UNPAIRED_TEST_NAME",
    "ChangeVerdict",
    "SignificanceResult",
    "ci_half_width",
    "cohen_kappa",
    "composite_significance",
    "holm_adjust",
    "paired_change",
    "standard_error_of_mean",
    "t_critical_two_sided",
    "wilson_interval",
]
