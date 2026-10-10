"""Judge drift: how far each judged dimension's scores moved when the same evidence was scored by another judge.

A judge is a model, the prompt per dimension and a temperature, and a change to any of them can move the scores it
gives without the subject changing at all. A before/after that spans a judge change cannot tell a fix that worked
from a judge that scores differently. The check is to re-score the SAME stored evidence under the new judge
(:func:`~threetears.evals.run.ask_second_judge`, which records each new score beside the first) and read, per dimension,
how far the scores moved:

- **The unit is the case.** A result's first and second scores are a pair; a case judged ``k`` times is one draw,
  so each side is averaged per case first and the pairs are the cases — the unit every separation test in the
  engine reads.
- **A family, corrected.** Every dimension read together is one family: each separation p
  (:func:`~threetears.evals.analysis.stats.separation_p`, paired) is Holm-adjusted over it, and each interval on the
  movement is at ``1 − α/m`` (Bonferroni over the ``m`` tested), as the bundle's comparison families are, so a
  family of ten dimensions does not hand a chance "move" to the reader.
- **Three words.** ``separated`` — the movement is shown, its sign the direction; ``not_separated`` — the data
  cannot tell it from noise, which says nothing about whether the judge moved; ``untested`` — no test could
  decide (fewer than two cases, or every case moving by one amount over too few for the exact test to reach α).

**It detects movement, never which judge is right.** Both judges read the same evidence, so a movement is a
difference between the judges; whether either one agrees with people is the calibration's question
(:func:`~threetears.evals.analysis.judge_agreement`).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Final, Literal

from pydantic import Field

from threetears.evals.analysis.agreement import JudgeKey
from threetears.evals.analysis.stats import (
    SIGNIFICANCE_ALPHA,
    bounded_difference_interval,
    difference_interval,
    holm_adjust,
    separation_p,
)
from threetears.evals.contracts.base import EvalDocumentModel
from threetears.evals.contracts.models import SCALES, JudgeTemperature, RubricScale

if TYPE_CHECKING:
    from threetears.evals.contracts.models import EvalResult

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
            "paired t interval, or where every case moved by one amount the bounded test's. None below two cases."
        ),
    )
    interval_level: float | None = Field(
        description="The interval's coverage: 1 − α/m over the m dimensions tested together. None when none was tested."
    )
    p_adjusted: float | None = Field(
        description="The paired separation p, Holm-adjusted over the dimensions read together; None when untested."
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


def _interval(a: list[float], b: list[float], level: float, scale: tuple[float, float]) -> tuple[float, float] | None:
    """The interval on ``mean(b) − mean(a)`` over paired cases at ``level``.

    The paired t interval, clipped to ± the scale's span; where every case moved by the same amount it has no width
    to give, and the bounded test's interval (:func:`~threetears.evals.analysis.stats.bounded_difference_interval`),
    valid at every n on the scale's range, stands in — so a uniform move is never stated without its bounds.
    """
    interval = difference_interval(a, b, paired=True, confidence=level, value_range=scale)
    if interval is None:
        interval = bounded_difference_interval(a, b, paired=True, value_range=scale, confidence=level)
    return interval


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
    samples: dict[_GroupKey, tuple[list[float], list[float]]] = {}
    raw: dict[_GroupKey, float] = {}
    for key in keys:
        # Each case one draw: its repeats' scores averaged on each side, in one case order.
        by_case = pairs[key]
        a = [sum(first for first, _ in by_case[case]) / len(by_case[case]) for case in sorted(by_case)]
        b = [sum(second for _, second in by_case[case]) / len(by_case[case]) for case in sorted(by_case)]
        samples[key] = (a, b)
        if (p := separation_p(a, b, paired=True)) is not None:
            raw[key] = p
    tested = [key for key in keys if key in raw]
    adjusted = dict(zip(tested, holm_adjust([raw[key] for key in tested]), strict=True)) if tested else {}
    level = 1.0 - SIGNIFICANCE_ALPHA / len(tested) if tested else None
    dimensions = []
    for key in keys:
        first, model, config_id, temperature = key
        a, b = samples[key]
        low, high = SCALES[first.scale].scores
        p_adjusted = adjusted.get(key)
        verdict: DriftVerdict = (
            "untested" if p_adjusted is None else "separated" if p_adjusted < SIGNIFICANCE_ALPHA else "not_separated"
        )
        first_mean = sum(a) / len(a) if a else None
        second_mean = sum(b) / len(b) if b else None
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
                delta=None if first_mean is None or second_mean is None else second_mean - first_mean,
                interval=_interval(a, b, level, (float(low), float(high)))
                if level is not None and key in raw
                else None,
                interval_level=level if key in raw else None,
                p_adjusted=p_adjusted,
                verdict=verdict,
            )
        )
    return JudgeDrift(dimensions=dimensions, family_size=len(tested))


__all__ = ["DRIFT_DETECTS_MOVEMENT", "DriftVerdict", "JudgeDrift", "JudgeDriftDimension", "judge_drift"]
