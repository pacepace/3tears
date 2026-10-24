"""How deep each arm was measured, the depth a comparison of arms is read at, and what limits its precision.

**Depth** is how many times an arm ran each case: its cases, its trials, and its repeats per case (fewest, median,
most). Arms pooled from runs of different ``k``, or a run stopped part-way, differ in depth across arms and across the
cases of one arm, so every reading states each arm's (:class:`Depth`).

**The common depth.** pass^k at depth ``k`` asks that ``k`` attempts at a case all pass, so two arms are compared at
one ``k``: the fewest times any arm ran any case they all share (owner, 2026-10-24: compare at the minimum depth),
over every shared case, so no shared case is left out for depth and a case some arm never ran is left out and counted
(:func:`common_depth`). A deeper arm is read at that ``k`` over all its runs,
``C(c, k) / C(n, k)`` per case, the exact mean over every ``k``-subset of its attempts, so its extra runs tighten its
own estimate with no model. A depth some arm never reached is unmeasured for that arm, and said so. At depth 1 the
comparison is of pass rates, not of reliability.

**One opt-in model.** Above an arm's depth nothing is measured. :func:`model_pass_hat_k` fits a beta-binomial over
the per-case pass chance and reads pass^k off it at any depth; it is a model, labelled as one with what it assumes
(:data:`MODEL_ASSUMES`), off unless a host asks for it (``HostProfile.model_pass_hat_k``), and never ranked or gated on.

**Precision.** A comparison's interval is no tighter than its noisier side allows, and extra repeats of a case shrink
only the noise within cases: the spread between cases is a floor that only more cases lower. :func:`precision` states
both: each side's own half-width, which side limits the comparison, and the between-case share of the difference's
variance (:func:`~threetears.evals.analysis.stats.variance_components`,
:func:`~threetears.evals.analysis.stats.paired_case_variance`).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, NamedTuple

from pydantic import Field

from threetears.evals.analysis.stats import (
    INTERVAL_LEVEL,
    bounded_mean_interval,
    paired_case_variance,
    variance_components,
)
from threetears.evals.kernel.scoring import case_pass_hat_k
from threetears.evals.schema.base import EvalDocumentModel

__all__ = [
    "MODEL_ASSUMES",
    "ModelPassHatK",
    "model_pass_hat_k",
    "CommonDepth",
    "Depth",
    "Precision",
    "REPEATS_FLOOR",
    "common_depth",
    "common_depth_sentence",
    "depth_of",
    "depth_sentence",
    "depths_differ",
    "pass_hat_k_over",
    "precision",
    "times",
    "unmeasured_sentence",
]

#: Why more repeats of the same cases tighten a comparison little: said once wherever precision is shown.
REPEATS_FLOOR = (
    "Extra repeats of a case shrink only the noise within cases; the spread between cases is a floor that only "
    "more cases lower."
)

#: What the opt-in model-based pass^k assumes, carried with every value it reads.
MODEL_ASSUMES = (
    "model-based, assumes one beta distribution of pass chance across cases, each case's attempts independent"
)

#: The share range a pass rate's per-case values lie in.
_SHARE_RANGE = (0.0, 1.0)


class Depth(EvalDocumentModel):
    """How deep one arm was measured on one reading: its cases, its trials, and its repeats per case."""

    cases: int = Field(ge=0, description="The cases with at least one usable trial.")
    trials: int = Field(ge=0, description="The usable trials over those cases.")
    repeats_min: int = Field(ge=0, description="The fewest usable trials any one case had.")
    repeats_median: float = Field(ge=0, description="The median usable trials per case.")
    repeats_max: int = Field(ge=0, description="The most usable trials any one case had.")

    def uniform(self) -> bool:
        """Whether every case ran the same number of times."""
        return self.repeats_min == self.repeats_max

    def spread(self) -> tuple[int, float, int]:
        """The repeats per case, fewest, median and most: what two depths are told apart by."""
        return self.repeats_min, self.repeats_median, self.repeats_max


def depth_of(repeats: Iterable[int]) -> Depth | None:
    """An arm's depth from its usable trials per case.

    :param repeats: Each case's usable trials; a case with none is not counted.
    :ptype repeats: Iterable[int]
    :return: The depth, or None with no case.
    :rtype: Depth | None
    """
    counts = sorted(count for count in repeats if count > 0)
    if not counts:
        return None
    return Depth(
        cases=len(counts),
        trials=sum(counts),
        repeats_min=counts[0],
        repeats_median=float(statistics.median(counts)),
        repeats_max=counts[-1],
    )


def times(depth: Depth) -> str:
    """How many times each case ran, in words: ``once``, ``twice``, ``3 times``, ``1 to 3 times``."""
    if depth.uniform():
        return _times(depth.repeats_min)
    return f"{depth.repeats_min} to {depth.repeats_max} times"


def _times(count: int) -> str:
    """A count of runs in words."""
    return {1: "once", 2: "twice"}.get(count, f"{count} times")


def depths_differ(depths: Iterable[Depth | None]) -> bool:
    """Whether the arms' repeats per case differ, fewest, median or most.

    :param depths: Each arm's depth; an arm with none is not compared.
    :ptype depths: Iterable[Depth | None]
    :return: True where two arms differ.
    :rtype: bool
    """
    return len({depth.spread() for depth in depths if depth is not None}) > 1


def depth_sentence(aspect: str, depths: Mapping[str, Depth | None]) -> str | None:
    """One plain sentence saying the arms ran to different depths on an aspect, and what that does; None where none
    differ: ``v2 was run twice on character, the others once; that tightens v2's own estimate, not the comparison
    much.``

    :param aspect: The aspect's label.
    :ptype aspect: str
    :param depths: Each arm's depth, by the label a reader calls it.
    :ptype depths: Mapping[str, Depth | None]
    :return: The sentence, or None.
    :rtype: str | None
    """
    known = {label: depth for label, depth in depths.items() if depth is not None}
    if not depths_differ(known.values()):
        return None
    deepest = max(depth.spread() for depth in known.values())
    deep = [label for label, depth in known.items() if depth.spread() == deepest]
    rest = [label for label in known if label not in deep]
    rest_spreads = {known[label].spread() for label in rest}
    if len(deep) == 1 and len(rest_spreads) == 1:
        others = "the others" if len(rest) > 1 else rest[0]
        return (
            f"{deep[0]} was run {times(known[deep[0]])} on {aspect}, {others} {times(known[rest[0]])}; that "
            f"tightens {deep[0]}'s own estimate, not the comparison much."
        )
    said = ", ".join(f"{label} {times(depth)}" for label, depth in known.items())
    return f"On {aspect} the arms ran each case {said}; extra repeats tighten an arm's own estimate, not the comparison much."


class CommonDepth(NamedTuple):
    """The depth a comparison of arms reads pass^k at, and the cases it counts there."""

    #: The deepest ``k`` every arm reached: the shallowest of the arms' deepest cases. None with no arm measured.
    k: int | None
    #: The cases every arm measured at least ``k`` times, which the comparison counts.
    cases: frozenset[str]
    #: Each arm's measured cases the comparison leaves out at ``k``: some arm measured them fewer times, or not at all.
    left_out: dict[str, int]


def common_depth(attempts: Mapping[str, Mapping[str, Sequence[bool]]]) -> CommonDepth:
    """The depth arms are compared at: the fewest times any arm ran any case every arm shares, over those cases.

    :param attempts: Each arm's scored attempts per case, by arm then case; an arm with none is not compared.
    :ptype attempts: Mapping[str, Mapping[str, Sequence[bool]]]
    :return: The depth, the cases it counts and what each arm leaves out.
    :rtype: CommonDepth
    """
    measured = {arm: {case: len(tries) for case, tries in cases.items() if tries} for arm, cases in attempts.items()}
    measured = {arm: cases for arm, cases in measured.items() if cases}
    if not measured:
        return CommonDepth(k=None, cases=frozenset(), left_out={})
    counted = frozenset(set.intersection(*(set(cases) for cases in measured.values())))
    return CommonDepth(
        k=min((cases[case] for cases in measured.values() for case in counted), default=None),
        cases=counted,
        left_out={arm: sum(1 for case in cases if case not in counted) for arm, cases in measured.items()},
    )


def pass_hat_k_over(attempts: Mapping[str, Sequence[bool]], cases: Iterable[str], k: int) -> float | None:
    """One arm's pass^k at ``k`` over the named cases: the mean of each case's ``C(c, k) / C(n, k)`` over all its
    attempts, the estimator every pass^k headline reads.

    :param attempts: The arm's scored attempts per case.
    :ptype attempts: Mapping[str, Sequence[bool]]
    :param cases: The cases to average over; one it measured fewer than ``k`` times is left out.
    :ptype cases: Iterable[str]
    :param k: The depth, at least 1.
    :ptype k: int
    :return: The mean, or None where no named case was measured ``k`` times.
    :rtype: float | None
    """
    estimates = [
        estimate for case in sorted(cases) if (estimate := case_pass_hat_k(attempts.get(case, ()), k)) is not None
    ]
    return math.fsum(estimates) / len(estimates) if estimates else None


def common_depth_sentence(k: int | None, *, left_out: int) -> str | None:
    """What depth a comparison of pass^k reads, in one sentence; None with no depth.

    :param k: The common depth (:func:`common_depth`).
    :ptype k: int | None
    :param left_out: The cases some arm measured that the comparison leaves out at ``k``.
    :ptype left_out: int
    :return: The sentence, or None.
    :rtype: str | None
    """
    if k is None:
        return None
    if k == 1:
        said = "pass^k is compared at depth 1, the fewest runs any arm gave a case, so it compares pass rates, not reliability"
    else:
        said = f"pass^k is compared at depth {k}, the fewest runs any arm gave a case, over each arm's runs"
    if left_out:
        cases = "1 case" if left_out == 1 else f"{left_out} cases"
        said += f"; {cases} some arm never ran {'is' if left_out == 1 else 'are'} left out"
    return said + "."


def unmeasured_sentence(k: int, arms: Sequence[str]) -> str | None:
    """That pass^k at ``k`` is unmeasured for the arms that never reached it, in one sentence; None with none.

    :param k: The depth read.
    :ptype k: int
    :param arms: The arms, by label, that measured no case ``k`` times.
    :ptype arms: Sequence[str]
    :return: The sentence, or None.
    :rtype: str | None
    """
    if not arms:
        return None
    return f"pass^{k} is unmeasured for {', '.join(arms)}, which ran no case {_times(k)}."


# =============================================================================
# What limits a comparison's precision
# =============================================================================


class Precision(EvalDocumentModel):
    """What limits one comparison's interval: each side's own half-width, the side that limits it, and how much of the
    difference's variance lies between cases, which extra repeats cannot lower.

    Half-widths are in points of a 0-100 pass rate, at the nominal level (:data:`~threetears.evals.analysis.stats.
    INTERVAL_LEVEL`). Variances are of a 0-1 share, from the per-case repeats.
    """

    half_width: float = Field(ge=0, description="The comparison's interval half-width, in points.")
    half_widths: dict[str, float] = Field(
        description="Each side's own interval half-width on its per-case shares, in points, by variant key."
    )
    limited_by: str | None = Field(
        description="The variant key of the side with the wider own interval, which limits the comparison; None when "
        "the two are equal."
    )
    within_case: float | None = Field(
        default=None,
        description="The pooled variance of repeats around their case's share (`stats.variance_components`); None "
        "where fewer than two cases were repeated.",
    )
    between_case: float | None = Field(
        default=None,
        description="The between-case variance: of the paired difference (`stats.paired_case_variance`) where the "
        "sides pair, else of the case levels pooled over both; None with `within_case`.",
    )
    between_share: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="The share of the difference's variance that lies between cases: what extra repeats cannot lower. "
        "None with `within_case`, or with no variance.",
    )


def _repeats(share: float, trials: int) -> list[float]:
    """A case's repeats as passes and fails, from its share passed and its usable trials."""
    passed = round(share * trials)
    return [1.0] * passed + [0.0] * (trials - passed)


def _half_width(interval: tuple[float, float] | None) -> float | None:
    """An interval's half-width in points; None with none."""
    return None if interval is None else 50 * (interval[1] - interval[0])


def precision(
    samples: tuple[Sequence[float], Sequence[float]],
    trials: tuple[Sequence[int], Sequence[int]] | None,
    *,
    keys: tuple[str, str],
    interval: tuple[float, float] | None,
    paired: bool,
) -> Precision | None:
    """What limits one pass-rate comparison: each side's own half-width, the side limiting it, and the variance split.

    :param samples: Each side's per-case shares, aligned by case when ``paired``.
    :ptype samples: tuple[Sequence[float], Sequence[float]]
    :param trials: Each side's usable trials per case, aligned with ``samples``; None where not carried, and then no
        variance split is read.
    :ptype trials: tuple[Sequence[int], Sequence[int]] | None
    :param keys: Each side's variant key, in ``samples``' order.
    :ptype keys: tuple[str, str]
    :param interval: The comparison's interval as a share, at the nominal level.
    :ptype interval: tuple[float, float] | None
    :param paired: Whether the sides pair by case.
    :ptype paired: bool
    :return: The attribution, or None where the comparison or a side has no interval.
    :rtype: Precision | None
    """
    whole = _half_width(interval)
    sides = [_half_width(bounded_mean_interval(side, _SHARE_RANGE, confidence=INTERVAL_LEVEL)) for side in samples]
    if whole is None or sides[0] is None or sides[1] is None:
        return None
    first, second = sides[0], sides[1]
    limited = None if first == second else keys[0] if first > second else keys[1]
    attribution = Precision(half_width=whole, half_widths={keys[0]: first, keys[1]: second}, limited_by=limited)
    if trials is None:
        return attribution
    cases = [
        [_repeats(share, count) for share, count in zip(side, counts, strict=True)]
        for side, counts in zip(samples, trials, strict=True)
    ]
    components = variance_components(cases)
    if components is None:
        return attribution
    within = components.within_case
    if paired:
        read = paired_case_variance([(cases[0], cases[1])], within)
        between = read[0] if read is not None else components.between_case
        noise = within * math.fsum(1 / len(a) + 1 / len(b) for a, b in zip(*cases, strict=True)) / len(cases[0])
        signal = between
    else:
        between = components.between_case
        # Each side's mean varies by the case levels over its cases and by its repeats' noise.
        signal = between * sum(1 / len(side) for side in cases)
        noise = within * sum(math.fsum(1 / len(case) for case in side) / len(side) ** 2 for side in cases)
    total = signal + noise
    return attribution.model_copy(
        update={
            "within_case": within,
            "between_case": between,
            "between_share": signal / total if total > 0 else None,
        }
    )


# =============================================================================
# The opt-in model: pass^k above an arm's depth
# =============================================================================


class ModelPassHatK(EvalDocumentModel):
    """pass^k read off a beta-binomial fit over the per-case pass chance: a model, never a measurement."""

    basis: Literal["model"] = Field(default="model", description="Always `model`: never an unbiased reading.")
    k: int = Field(ge=1, description="The depth read.")
    pass_hat_k: float = Field(ge=0, le=1, description="The model's pass^k at `k`.")
    alpha: float | None = Field(
        description="The fitted beta's first shape; None where the cases show no spread beyond their repeats' noise, "
        "and every case is read at the mean chance."
    )
    beta: float | None = Field(description="The fitted beta's second shape; None with `alpha`.")
    cases: int = Field(ge=2, description="The cases the fit read.")
    assumes: str = Field(default=MODEL_ASSUMES, description="What the model assumes, carried with every value.")


def model_pass_hat_k(attempts: Mapping[str, Sequence[bool]], k: int) -> ModelPassHatK | None:
    """pass^k at ``k`` from a beta-binomial over the per-case pass chance, fitted by the method of moments.

    Each case's share passed ``c / n`` has mean ``μ`` and variance ``v + (μ(1 − μ) − v) · h`` over cases, ``v`` the
    variance of the per-case chance and ``h`` the mean of ``1 / n``: the spread between cases plus each case's
    binomial noise. Solving for ``v`` gives the beta's shapes, ``α + β = μ(1 − μ) / v − 1``, and pass^k is the beta's
    ``k``-th moment, ``Π_{i<k} (α + i) / (α + β + i)``. Where the shares spread no more than their noise (``v <= 0``),
    every case is read at ``μ`` and pass^k is ``μ^k``; a ``v`` past the beta's limit is held just inside it.

    :param attempts: One arm's scored attempts per case.
    :ptype attempts: Mapping[str, Sequence[bool]]
    :param k: The depth, at least 1.
    :ptype k: int
    :return: The model's reading, or None below two cases, or where every case ran once (the spread between cases
        cannot be told from each case's noise then), or where every case's share is 0 or 1 everywhere alike.
    :rtype: ModelPassHatK | None
    :raises ValueError: ``k`` is below 1.
    """
    if k < 1:
        raise ValueError(f"pass^k needs k >= 1, got {k}")
    cases = [(len(tries), sum(tries)) for tries in attempts.values() if tries]
    if len(cases) < 2 or all(n == 1 for n, _ in cases):
        return None
    shares = [c / n for n, c in cases]
    mean = math.fsum(shares) / len(shares)
    if mean in (0.0, 1.0):
        return ModelPassHatK(k=k, pass_hat_k=mean, alpha=None, beta=None, cases=len(cases))
    spread = statistics.variance(shares)
    h = math.fsum(1 / n for n, _ in cases) / len(cases)
    chance = (spread - mean * (1 - mean) * h) / (1 - h)
    if chance <= 0:
        return ModelPassHatK(k=k, pass_hat_k=mean**k, alpha=None, beta=None, cases=len(cases))
    # The beta's variance is below μ(1 − μ); one at or past it is held just inside, the most spread a beta can have.
    chance = min(chance, mean * (1 - mean) * (1 - 1e-9))
    total = mean * (1 - mean) / chance - 1
    alpha, beta = mean * total, (1 - mean) * total
    value = math.prod((alpha + i) / (total + i) for i in range(k))
    return ModelPassHatK(k=k, pass_hat_k=value, alpha=alpha, beta=beta, cases=len(cases))
