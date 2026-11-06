"""How deep each arm was measured, the depth a comparison of arms is read at, and what limits its precision.

**Depth** is how many times an arm ran each case: its cases, its trials, and its repeats per case (fewest, median,
most). Arms pooled from runs of different ``k``, or a run stopped part-way, differ in depth across arms and across the
cases of one arm, so every reading states each arm's (:class:`Depth`).

**The common depth.** pass^k at depth ``k`` asks that ``k`` attempts at a case all pass, so two arms are compared at
one ``k``: the fewest times any arm ran any case they all share (owner, 2026-10-24: compare at the minimum depth),
over every shared case, so no shared case is left out for depth and a case some arm never ran is left out and counted
(:func:`common_depth`). A deeper arm is read at that ``k`` over all its runs,
``C(c, k) / C(n, k)`` per case, the exact mean over every ``k``-subset of its attempts, so its extra runs tighten its
own estimate with no model. A depth an arm never ran any case to is unmeasured for that arm, and said so. At depth 1 the
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
from collections import Counter
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
    "repeats_sentence",
    "times",
    "unmeasured_sentence",
]

#: Why more repeats of the same cases tighten a comparison little: said once wherever precision is shown.
REPEATS_FLOOR = (
    "Extra repeats of a case shrink only the noise within cases; the spread between cases is a floor that only "
    "more cases lower."
)

# The package's own page text, each sentence one template so a locale can supply it later; nothing here reads words
# back out of what a writer or a host wrote.

#: How many times a case ran: once, twice, a count, or a range.
TIMES_ONCE = "once"
TIMES_TWICE = "twice"
TIMES_MANY = "{count} times"
TIMES_RANGE = "{fewest} to {most} times"

#: The arms besides the deepest, where more than one ran alike.
THE_OTHERS = "the others"

#: One arm ran deeper than the rest, which ran alike (the owner's words, 2026-10-24).
DEPTH_ONE_DEEPER = (
    "{deep} was run {deep_times} on {aspect}, {others} {others_times}; that tightens {deep}'s own estimate, not the "
    "comparison much."
)

#: One arm and how many times it ran each case, as :data:`DEPTH_EACH` lists them.
ARM_TIMES = "{arm} {times}"

#: The arms ran to several depths.
DEPTH_EACH = (
    "On {aspect} the arms ran each case {said}; extra repeats tighten an arm's own estimate, not the comparison much."
)

#: The most arms one sentence names as running to another depth than the rest, in place of the depth table.
REPEATS_NAMED = 3

#: Every arm ran every case alike: on every aspect, or on the one.
EVERY_ARM_EVERY_ASPECT = "Every arm ran each case {times} on every aspect."
EVERY_ARM_ON = "Every arm ran each case {times} on {aspect}."

#: Every arm ran every case alike on every aspect, but for the few the sentence names.
EVERY_ARM_BUT = (
    "Every arm ran each case {times} on every aspect but {exceptions}; extra repeats tighten an arm's own estimate, "
    "not the comparison much."
)

#: One arm and aspect that ran to another depth than the rest, as :data:`EVERY_ARM_BUT` names them.
ARM_TIMES_ON = "{arm} {times} on {aspect}"

#: The one arm a campaign measures ran every case alike: on every aspect, on the one, or on every aspect but a few.
ONE_ARM_EVERY_ASPECT = "{arm} ran each case {times} on every aspect."
ONE_ARM_ON = "{arm} ran each case {times} on {aspect}."
ONE_ARM_BUT = "{arm} ran each case {times} on every aspect but {exceptions}."

#: One aspect the one arm ran to another depth on, as :data:`ONE_ARM_BUT` names them.
TIMES_ON = "{times} on {aspect}"

#: The common depth at 1 on every aspect and every subject, with no case left out and no arm short.
COMMON_DEPTH_ONE_EVERYWHERE = (
    "On every aspect, pass^k is compared at depth 1, the fewest runs any arm gave a shared case, so it compares pass "
    "rates, not reliability."
)

#: The arms share no case, so pass^k compares nothing.
NO_SHARED_CASE = "On {where} the arms share no case, so pass^k compares nothing between them."

#: The common depth at 1, where pass^k is a pass rate.
COMMON_DEPTH_ONE = (
    "On {where}, pass^k is compared at depth 1, the fewest runs any arm gave a shared case, so it compares pass "
    "rates, not reliability."
)

#: The common depth above 1.
COMMON_DEPTH = "On {where}, pass^k is compared at depth {k}, the fewest runs any arm gave a shared case, over all runs."

#: The cases some arm never ran, which the comparison leaves out: one, or a count.
LEFT_OUT_ONE = "1 case some arm never ran is left out."
LEFT_OUT = "{count} cases some arm never ran are left out."

#: pass^k at the planned depth is unmeasured for the arms that ran no case that deep.
UNMEASURED = "On {where}, pass^{k} is unmeasured for {arms}, which ran no case {times}."

#: What the opt-in model-based pass^k assumes, carried with every value it reads.
MODEL_ASSUMES = (
    "model-based, assuming one beta distribution of pass chance across cases, each case's attempts independent"
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
    return TIMES_RANGE.format(fewest=depth.repeats_min, most=depth.repeats_max)


def _times(count: int) -> str:
    """A count of runs in words."""
    return {1: TIMES_ONCE, 2: TIMES_TWICE}.get(count) or TIMES_MANY.format(count=count)


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
        others = THE_OTHERS if len(rest) > 1 else rest[0]
        return DEPTH_ONE_DEEPER.format(
            deep=deep[0],
            deep_times=times(known[deep[0]]),
            aspect=aspect,
            others=others,
            others_times=times(known[rest[0]]),
        )
    said = ", ".join(ARM_TIMES.format(arm=label, times=times(depth)) for label, depth in known.items())
    return DEPTH_EACH.format(aspect=aspect, said=said)


def repeats_sentence(depths: Mapping[str, Mapping[str, Depth]]) -> str | None:
    """The one sentence that says every arm's depth on every aspect, where it loses no figure a depth table would
    print; None where a table is needed.

    It says them where every arm ran each of its cases the same number of times, every arm on one aspect ran as many
    cases, and all but at most :data:`REPEATS_NAMED` arm-and-aspect pairs, fewer than the rest, ran to one depth;
    the sentence names those few. ``Every arm ran each case once on every aspect but v3 twice on tone; extra repeats
    tighten an arm's own estimate, not the comparison much.``

    :param depths: Each aspect's arms' depths, by the names a reader calls them, aspect then arm.
    :ptype depths: Mapping[str, Mapping[str, Depth]]
    :return: The sentence, or None.
    :rtype: str | None
    """
    rows = [(aspect, arm, depth) for aspect, arms in depths.items() for arm, depth in arms.items()]
    if not rows or not all(depth.uniform() for _, _, depth in rows):
        return None
    if any(len({depth.cases for depth in arms.values()}) > 1 for arms in depths.values()):
        return None
    counts = Counter(depth.repeats_min for _, _, depth in rows)
    ((usual, alike),) = counts.most_common(1)
    others = [(aspect, arm, depth) for aspect, arm, depth in rows if depth.repeats_min != usual]
    if others and (len(others) > REPEATS_NAMED or len(others) >= alike):
        return None
    labels = list(dict.fromkeys(arm for _, arm, _ in rows))
    aspects = [aspect for aspect, held_arms in depths.items() if held_arms]
    said = _times(usual)
    if len(labels) == 1:
        if others:
            named = ", ".join(TIMES_ON.format(times=times(depth), aspect=aspect) for aspect, _, depth in others)
            return ONE_ARM_BUT.format(arm=labels[0], times=said, exceptions=named)
        if len(aspects) == 1:
            return ONE_ARM_ON.format(arm=labels[0], times=said, aspect=aspects[0])
        return ONE_ARM_EVERY_ASPECT.format(arm=labels[0], times=said)
    if len(aspects) == 1:
        if others:
            # On one aspect, the sentence its own depth says where its arms differ is the one sentence.
            return depth_sentence(aspects[0], depths[aspects[0]])
        return EVERY_ARM_ON.format(times=said, aspect=aspects[0])
    if others:
        named = ", ".join(ARM_TIMES_ON.format(arm=arm, times=times(depth), aspect=on) for on, arm, depth in others)
        return EVERY_ARM_BUT.format(times=said, exceptions=named)
    return EVERY_ARM_EVERY_ASPECT.format(times=said)


class CommonDepth(NamedTuple):
    """The depth a comparison of arms reads pass^k at, and the cases it counts there."""

    #: The fewest runs any arm gave a case every arm shares (owner, 2026-10-24: compare at min(k)). None where the
    #: arms share no case, or no arm measured any.
    k: int | None
    #: The cases every arm ran, which the comparison counts.
    cases: frozenset[str]
    #: Each arm's measured cases the comparison leaves out: some arm never ran them.
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


def common_depth_sentence(where: str, k: int | None, *, left_out: int) -> str:
    """What depth a comparison of pass^k reads on ``where``, in plain sentences, or that the arms share no case.

    :param where: The aspect, and the subject where an aspect has several, as a sentence names it.
    :ptype where: str
    :param k: The common depth (:func:`common_depth`); None where the arms share no case.
    :ptype k: int | None
    :param left_out: The cases some arm ran that another never did, which the comparison leaves out.
    :ptype left_out: int
    :return: The sentences.
    :rtype: str
    """
    if k is None:
        return NO_SHARED_CASE.format(where=where)
    said = COMMON_DEPTH_ONE.format(where=where) if k == 1 else COMMON_DEPTH.format(where=where, k=k)
    if left_out:
        said += " " + (LEFT_OUT_ONE if left_out == 1 else LEFT_OUT.format(count=left_out))
    return said


def unmeasured_sentence(where: str, k: int, arms: Sequence[str]) -> str | None:
    """That pass^k at ``k`` is unmeasured on ``where`` for the arms that never reached it; None with none.

    :param where: The aspect, and the subject where an aspect has several, as a sentence names it.
    :ptype where: str
    :param k: The depth read.
    :ptype k: int
    :param arms: The arms, by label, that measured no case ``k`` times.
    :ptype arms: Sequence[str]
    :return: The sentence, or None.
    :rtype: str | None
    """
    if not arms:
        return None
    return UNMEASURED.format(where=where, k=k, arms=", ".join(arms), times=_times(k))


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


def _repeats(passed: int, trials: int) -> list[float]:
    """A case's repeats as passes and fails, from its passed and usable trials."""
    return [1.0] * passed + [0.0] * (trials - passed)


def _half_width(interval: tuple[float, float] | None) -> float | None:
    """An interval's half-width in points; None with none."""
    return None if interval is None else 50 * (interval[1] - interval[0])


def precision(
    samples: tuple[Sequence[float], Sequence[float]],
    trials: tuple[Sequence[int], Sequence[int]] | None,
    *,
    passes: tuple[Sequence[int], Sequence[int]] | None = None,
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
    :param passes: Each side's trials passed per case, aligned with ``trials``; None where not carried, and then each
        case's passes are its share times its trials, rounded.
    :ptype passes: tuple[Sequence[int], Sequence[int]] | None
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
    passed = passes or tuple(
        [round(share * count) for share, count in zip(side, counts, strict=True)]
        for side, counts in zip(samples, trials, strict=True)
    )
    cases = [
        [_repeats(hit, count) for hit, count in zip(hits, counts, strict=True)]
        for hits, counts in zip(passed, trials, strict=True)
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

    The method of moments, not empirical Bayes by maximum likelihood: it is closed form and deterministic, so two
    builds read one figure, it needs no optimiser that can fail to converge on a few cases, and it reads only the
    shares and depths the page already states. What it costs: its shapes are noisier than a likelihood fit's on few
    cases, and the two clamps above bend it at the edges. It is opt-in, labelled and never ranked, so that is accepted.

    :param attempts: One arm's scored attempts per case.
    :ptype attempts: Mapping[str, Sequence[bool]]
    :param k: The depth, at least 1.
    :ptype k: int
    :return: The model's reading, or None below two cases, or where every case ran once (the spread between cases
        cannot be told from each case's noise then). Every case passing all, or none, reads that share at every depth.
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
