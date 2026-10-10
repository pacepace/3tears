"""Judge drift: how far each judged dimension's scores moved when the same evidence was scored by another judge.

A judge is a model, the prompt per dimension and a temperature, and a change to any of them can move the scores it
gives without the subject changing at all. A before/after that spans a judge change cannot tell a fix that worked
from a judge that scores differently. The check is to re-score the SAME stored evidence under the new judge
(:func:`~threetears.evals.run.ask_second_judge`, which records each new score beside the first) and read, per dimension,
how far the scores moved:

- **The unit is the case.** A result's first and second scores are a pair; a case judged ``k`` times is one draw,
  so each side is averaged per case first and the pairs are the cases — the unit every separation test in the
  engine reads.
- **A family, corrected.** Every dimension read together is one family, and each is tested at ``1 − α/m``
  (Bonferroni over the ``m`` tested), so a family of ten dimensions does not hand a chance "move" to the reader.
- **One test per dimension, and the verdict is its interval** (:func:`_paired_test`). ``separated`` is exactly the
  interval excluding zero, and the adjusted p is that test's p times ``m``, so a reader never sees "separated"
  beside an interval reaching zero, or "not separated" beside one that excludes it. Holm's step-down would
  separate a little more often, but no interval goes with it: its extra separations would sit beside a Bonferroni
  interval that reaches zero (#597).
- **Three words.** ``separated`` — the movement is shown, its sign the direction; ``not_separated`` — the data
  cannot tell it from noise, which says nothing about whether the judge moved; ``untested`` — no test could
  run (fewer than two cases).

**It detects movement, never which judge is right.** Both judges read the same evidence, so a movement is a
difference between the judges; whether either one agrees with people is the calibration's question
(:func:`~threetears.evals.analysis.judge_agreement`).
"""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from typing import TYPE_CHECKING, Final, Literal

from pydantic import Field

from threetears.evals.analysis.agreement import JudgeKey
from threetears.evals.analysis.stats import (
    SIGNIFICANCE_ALPHA,
    bounded_difference_interval,
    bounded_separation_p,
    difference_interval,
    no_spread_p,
    separation_p,
)
from threetears.evals.schema.base import EvalDocumentModel
from threetears.evals.schema.models import SCALES, JudgeTemperature, RubricScale

if TYPE_CHECKING:
    from threetears.evals.schema.models import EvalResult

#: What a drift reading can and cannot say, stated on every one.
DRIFT_DETECTS_MOVEMENT: Final = (
    "Drift compares two judges on the same evidence: it detects whether the scores moved, never which judge is right. "
    "Whether either judge agrees with people is what calibration measures."
)

#: How one dimension's movement reads — see the module docstring.
DriftVerdict = Literal["separated", "not_separated", "untested"]


class JudgeDriftDimension(EvalDocumentModel):
    """How far one dimension's scores moved from the run's judge to a second judge, over the same evidence."""

    rubric_dim: str = Field(min_length=1, description="The judged dimension.")
    scale: RubricScale = Field(description="The scale both judges scored it on.")
    judge_model: str | None = Field(description="The model that served the first scores; None when it named none.")
    judge_config_id: str | None = Field(description="The JudgeConfig that asked for the first scores; None = built-in.")
    judge_temperature: JudgeTemperature | None = Field(
        default=None, description="The temperature the first scores were sent at; None = not recorded."
    )
    second_model: str = Field(min_length=1, description="The model the second judge was asked as.")
    second_judge_config_id: str | None = Field(
        description="The JudgeConfig that asked the second judge; None = the built-in prompt."
    )
    second_temperature: float | None = Field(
        description="The temperature the second judge was requested at; None = what each prompt asks for."
    )
    n_pairs: int = Field(ge=0, description="Results carrying both judges' scores on the dimension.")
    n_cases: int = Field(ge=0, description="Distinct cases behind those pairs — the unit the test and interval read.")
    n_unanswered: int = Field(
        ge=0,
        description="Second-judge answers with no score (the call failed, or it could not tell), left out of the pairs.",
    )
    first_mean: float | None = Field(description="The first judge's mean over the cases; None with no pair.")
    second_mean: float | None = Field(description="The second judge's mean over the same cases; None with no pair.")
    delta: float | None = Field(
        description="second_mean − first_mean: the movement's point estimate. Never the finding; `verdict` is."
    )
    interval: tuple[float, float] | None = Field(
        description=(
            "The interval on the movement at `interval_level`, over the cases (paired), within ± the scale's span: the "
            "paired t interval, or where every case moved by one amount the bounded test's. The verdict is read off "
            "it: `separated` exactly when it excludes 0. None below two cases."
        ),
    )
    interval_level: float | None = Field(
        description="The interval's coverage: 1 − α/m over the m dimensions tested together. None when none was tested."
    )
    p_adjusted: float | None = Field(
        description=(
            "The two-sided p of the test `interval` inverts, Bonferroni-adjusted (times `family_size`, capped at 1), so "
            "it is below α exactly when the interval excludes 0. None when untested."
        )
    )
    verdict: DriftVerdict = Field(
        description=(
            "`separated`: the scores moved, in the direction of `delta`'s sign; `not_separated`: the movement cannot be "
            "told from noise, which is not a claim that the judge did not move; `untested`: no test could decide."
        )
    )


class JudgeDrift(EvalDocumentModel):
    """Every dimension a second judge re-scored, and how far the scores moved — with what that can and cannot show."""

    dimensions: list[JudgeDriftDimension] = Field(
        default_factory=list,
        description=(
            "One per (dimension, scale, first judge, second judge), ordered by those. Empty when no second judge "
            "re-scored anything: drift is then unmeasured, which is a state, not 'no drift'."
        ),
    )
    family_size: int = Field(default=0, ge=0, description="The dimensions tested together, which the correction spans.")
    disclosure: str = Field(default=DRIFT_DETECTS_MOVEMENT, description="What drift detects, and what it does not.")


_GroupKey = tuple[JudgeKey, str, str | None, float | None]


def _paired_test(
    a: list[Fraction], b: list[Fraction], level: float, scale: tuple[float, float]
) -> tuple[tuple[float, float], float] | None:
    """One test of ``mean(b) − mean(a)`` over paired cases: its interval at ``level`` and its two-sided p.

    The interval and the p are one test's, so the verdict read off either cannot contradict the other: the
    interval excludes zero exactly when ``p < 1 − level``.

    - **Where the differences have spread**, the paired t-test (:func:`~threetears.evals.analysis.stats.separation_p`)
      and the interval it inverts (:func:`~threetears.evals.analysis.stats.difference_interval`, clipped to ± the
      scale's span, which never moves zero in or out).
    - **Where every case moved by one amount** no t exists, and the bounded test by betting reads both
      (:func:`~threetears.evals.analysis.stats.bounded_difference_interval`, and
      :func:`~threetears.evals.analysis.stats.bounded_separation_p` with its stakes at the same tail): valid for the mean at
      every n on the scale's range. The exact sign-flip p this once read here is a test of symmetry, not of the
      mean: twenty cases each up one point arise about one time in ninety from a judge whose mean did not move (up one
      point four times in five, down four the fifth), where the sign flip states one in half a million. The
      price is the truth about a uniform move on a 1-5 scale: twenty cases do not show it, thirty do (with two
      dimensions tested together).

    Spread is decided on exact values (:func:`~threetears.evals.analysis.stats.no_spread_p`), so a float residue
    in per-case means cannot pass for a t-test's spread.

    Returns:
        ``(interval, p)``, or None below two cases.
    """
    if len(a) < 2:
        return None
    if no_spread_p(a, b, paired=True) is None:
        interval = difference_interval(
            [float(x) for x in a], [float(y) for y in b], paired=True, confidence=level, value_range=scale
        )
        p = separation_p(a, b, paired=True)
        if interval is not None and p is not None:
            return interval, p
    bounded = bounded_difference_interval(a, b, paired=True, value_range=scale, confidence=level)
    p = bounded_separation_p(a, b, paired=True, value_range=scale, alpha=1.0 - level)
    if bounded is None or p is None:
        return None
    return bounded, p


def judge_drift(results: Iterable[EvalResult], *, pass_id: str | None = None) -> JudgeDrift:
    """Read how far each dimension's scores moved from the run's judge to a second judge on the same evidence.

    Args:
        results: The results whose second-judge scores to read.
        pass_id: Read only this pass's pairs; ``None`` for every pass.

    Returns:
        The movement per (dimension, scale, first judge, second judge), tested as one family.
    """
    pairs: dict[_GroupKey, dict[str, list[tuple[int, int]]]] = {}
    unanswered: dict[_GroupKey, int] = {}
    for result in results:
        for judging in result.judge_seconds:
            if pass_id is not None and judging.pass_id != pass_id:
                continue
            for entry in judging.scores:
                first = JudgeKey(
                    entry.dim,
                    entry.scale,
                    entry.first_served_model,
                    entry.first_judge_config_id,
                    entry.first_judge_temperature,
                )
                key = (first, judging.judge.model, judging.judge_config_ids.get(entry.dim), judging.judge.temperature)
                cases = pairs.setdefault(key, {})
                if entry.second is None:
                    unanswered[key] = unanswered.get(key, 0) + 1
                    continue
                cases.setdefault(result.test_case_id, []).append((entry.first_score, entry.second.score))
    keys = sorted(
        pairs,
        key=lambda k: (
            k[0].rubric_dim,
            k[0].scale,
            k[0].judge_model or "",
            k[0].judge_config_id or "",
            "" if k[0].judge_temperature is None else str(k[0].judge_temperature),
            k[1],
            k[2] or "",
            "" if k[3] is None else str(k[3]),
        ),
    )
    samples: dict[_GroupKey, tuple[list[Fraction], list[Fraction]]] = {}
    for key in keys:
        # Each case one draw: its repeats' scores averaged on each side, exactly, in one case order.
        by_case = pairs[key]
        samples[key] = (
            [Fraction(sum(first for first, _ in by_case[case]), len(by_case[case])) for case in sorted(by_case)],
            [Fraction(sum(second for _, second in by_case[case]), len(by_case[case])) for case in sorted(by_case)],
        )
    # The family is every dimension a test can read (two cases or more); its size sets every test's level.
    tested = [key for key in keys if len(samples[key][0]) >= 2]
    level = 1.0 - SIGNIFICANCE_ALPHA / len(tested) if tested else None
    dimensions = []
    for key in keys:
        first, model, config_id, temperature = key
        a, b = samples[key]
        low, high = SCALES[first.scale].scores
        test = _paired_test(a, b, level, (float(low), float(high))) if level is not None else None
        interval = None if test is None else test[0]
        # Bonferroni: the p times the family's size, so `p_adjusted < α` is exactly the interval at 1 − α/m
        # excluding zero, and the verdict below reads the interval — one test, never two that could disagree.
        p_adjusted = None if test is None else min(1.0, test[1] * len(tested))
        verdict: DriftVerdict = (
            "untested" if interval is None else "separated" if interval[0] > 0 or interval[1] < 0 else "not_separated"
        )
        first_mean = float(sum(a) / len(a)) if a else None
        second_mean = float(sum(b) / len(b)) if b else None
        dimensions.append(
            JudgeDriftDimension(
                rubric_dim=first.rubric_dim,
                scale=first.scale,
                judge_model=first.judge_model,
                judge_config_id=first.judge_config_id,
                judge_temperature=first.judge_temperature,
                second_model=model,
                second_judge_config_id=config_id,
                second_temperature=temperature,
                n_pairs=sum(len(values) for values in pairs[key].values()),
                n_cases=len(a),
                n_unanswered=unanswered.get(key, 0),
                first_mean=first_mean,
                second_mean=second_mean,
                delta=float(sum(b) / len(b) - sum(a) / len(a)) if a else None,
                interval=interval,
                interval_level=level if interval is not None else None,
                p_adjusted=p_adjusted,
                verdict=verdict,
            )
        )
    return JudgeDrift(dimensions=dimensions, family_size=len(tested))


__all__ = ["DRIFT_DETECTS_MOVEMENT", "DriftVerdict", "JudgeDrift", "JudgeDriftDimension", "judge_drift"]
