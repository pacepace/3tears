"""Judge-versus-human agreement: how often the judge scored a dimension the way people did.

A judge's consistency across repeats measures its precision; only people can say whether it is
right. People say so in :class:`~threetears.evals.contracts.models.CalibrationRating` documents, one
per rater per dimension per result, and this module sets each beside the score the judge gave the
same dimension of the same result and reads the pairs per dimension:

- **n** — the pairs, because a kappa over four of them is a different claim from one over forty;
- **exact agreement** — the share of pairs where the two gave the same score;
- **Cohen's kappa** — the same agreement net of what the two raters' own score distributions would
  produce by chance;
- **quadratic-weighted kappa** — on a 1-5 dimension only, where a 4 against a 5 is a near miss and
  a 1 against a 5 is not. On pass/fail it would be the unweighted number restated, so it is absent.

**One group per dimension, scale and judge.** The judge is the model that served the score
(:attr:`~threetears.evals.contracts.models.RubricScore.served_model`), so a campaign sweeping its
judge reads each judge's agreement separately — pooling them would credit one judge with the other's
calibration, which is the comparison a judge swap is decided on. A dimension rated on a scale it was
later moved off is two groups for the same reason.

**A rating that cannot be paired is named, never dropped.** Its result may have been deleted, or
re-judged without that dimension, or the dimension's scale may have changed since it was rated; each
is listed with which, so a dimension's n can be reconciled with the ratings people actually wrote.

**Only a person's rating is agreement with people.** A rating an agent wrote (``rater_kind="agent"``) is
another model's opinion of the same output, so it is listed as ``rated_by_an_agent`` and never pooled
into a dimension's pairs, its kappa or its raters.

Pure: the caller hands in the ratings and the results they rate, and nothing here reads storage.
The bundle and a reporter run's calibration read both call :func:`judge_agreement`, so the two never
compute agreement two ways.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from threetears.evals.analysis.stats import cohen_kappa
from threetears.evals.contracts.base import EvalDocumentModel
from threetears.evals.contracts.models import SCALES, RubricScale

if TYPE_CHECKING:
    from threetears.evals.contracts.models import CalibrationRating, EvalResult


#: Why a rating has no judge score to be read against.
UnpairedReason = Literal["result_unresolved", "dimension_unscored", "scale_changed", "rated_by_an_agent"]


class DimensionAgreement(EvalDocumentModel):
    """How one judge's scores on one dimension agreed with people's ratings of the same results."""

    rubric_dim: str = Field(min_length=1, description="The judged dimension.")
    scale: RubricScale = Field(description="The scale both sides scored it on.")
    judge_model: str | None = Field(
        description=(
            "The model that served the judge's scores, as the provider named it. None = the responses named no "
            "model, so which judge these pairs calibrate is unknown — never read as a match for a named one."
        ),
    )
    n: int = Field(ge=1, description="Pairs read: one per rating, so two raters of one result are two pairs.")
    raters: list[str] = Field(
        min_length=1, description="Every person whose ratings are among the pairs, sorted; never an agent."
    )
    exact_agreement: float = Field(
        ge=0.0, le=1.0, description="The share of pairs where judge and person gave the same score."
    )
    kappa: float | None = Field(
        description=(
            "Cohen's kappa, unweighted. None when chance alone predicts no disagreement — judge and people gave "
            "one and the same score to every pair — where it is undefined, not perfect."
        ),
    )
    weighted_kappa: float | None = Field(
        description=(
            "Cohen's kappa with quadratic weights over the 1-5 scale. None on pass/fail, where it equals `kappa`, "
            "and wherever `kappa` is undefined."
        ),
    )


class UnpairedRating(EvalDocumentModel):
    """A rating with no judge score to set it against, and why."""

    rating_id: str = Field(min_length=1, description="The rating.")
    result_id: str = Field(min_length=1, description="The result it rates.")
    rubric_dim: str = Field(min_length=1, description="The dimension it rates.")
    rater: str = Field(min_length=1, description="Who rated.")
    reason: UnpairedReason = Field(
        description=(
            "`result_unresolved`: the rated result is not among those read (deleted since, or not a member of what "
            "was read). `dimension_unscored`: the result carries no judge score on the dimension now (re-judged "
            "without it). `scale_changed`: the judge's score is on another scale than the rating. "
            "`rated_by_an_agent`: an agent wrote it, so it is another model's opinion, never a person's agreement "
            "with the judge."
        ),
    )


class JudgeAgreement(EvalDocumentModel):
    """Every rating read, paired with the judge where it can be, and agreement per dimension and judge."""

    ratings_read: int = Field(default=0, ge=0, description="Every rating read: the pairs plus the unpaired.")
    dimensions: list[DimensionAgreement] = Field(
        default_factory=list,
        description=(
            "One per (dimension, scale, judge) with at least one pair, ordered by those three. Empty when nobody "
            "rated anything — the judge is then uncalibrated against people, which is a state, not a zero."
        ),
    )
    unpaired: list[UnpairedRating] = Field(
        default_factory=list, description="Ratings that could not be paired, in the order they were read."
    )


def judge_agreement(ratings: Iterable[CalibrationRating], results: Iterable[EvalResult]) -> JudgeAgreement:
    """Pair each rating with the judge's score on the same dimension of the same result, and read agreement.

    Args:
        ratings: The ratings to read.
        results: The results they may rate. A rating whose result is not here is unpaired
            (``result_unresolved``), so hand in every result the ratings were read for.

    Returns:
        The agreement per (dimension, scale, judge), and the ratings that could not be paired.
    """
    by_id = {result.id: result for result in results}
    groups: dict[tuple[str, RubricScale, str | None], list[tuple[int, int, str]]] = {}
    unpaired: list[UnpairedRating] = []
    read = 0
    for rating in ratings:
        read += 1
        # Before anything is paired: an agent's rating is not a person's, whatever it would pair with.
        if rating.rater_kind != "person":
            unpaired.append(_unpaired(rating, "rated_by_an_agent"))
            continue
        result = by_id.get(rating.result_id)
        if result is None:
            unpaired.append(_unpaired(rating, "result_unresolved"))
            continue
        score = result.judge_score(rating.rubric_dim)
        if score is None:
            unpaired.append(_unpaired(rating, "dimension_unscored"))
            continue
        if score.scale != rating.scale:
            unpaired.append(_unpaired(rating, "scale_changed"))
            continue
        key = (rating.rubric_dim, rating.scale, score.served_model)
        groups.setdefault(key, []).append((score.score, rating.score, rating.rater))
    dimensions = [
        _dimension_agreement(dim, scale, judge, groups[(dim, scale, judge)])
        for dim, scale, judge in sorted(groups, key=lambda key: (key[0], key[1], key[2] or ""))
    ]
    return JudgeAgreement(ratings_read=read, dimensions=dimensions, unpaired=unpaired)


def _unpaired(rating: CalibrationRating, reason: UnpairedReason) -> UnpairedRating:
    """Name a rating that has no judge score to be read against, and why."""
    return UnpairedRating(
        rating_id=rating.id,
        result_id=rating.result_id,
        rubric_dim=rating.rubric_dim,
        rater=rating.rater,
        reason=reason,
    )


def _dimension_agreement(
    rubric_dim: str, scale: RubricScale, judge_model: str | None, pairs: Sequence[tuple[int, int, str]]
) -> DimensionAgreement:
    """Read one group's pairs.

    Args:
        rubric_dim: The dimension.
        scale: Its scale.
        judge_model: The judge that served the scores.
        pairs: ``(judge score, person's score, rater)`` per rating.

    Returns:
        The group's agreement.
    """
    low, high = SCALES[scale].scores
    categories = list(range(low, high + 1))
    scored = [(judge, person) for judge, person, _ in pairs]
    kappa = cohen_kappa(scored, categories)
    weighted = cohen_kappa(scored, categories, weights="quadratic") if scale == "ordinal" else None
    return DimensionAgreement(
        rubric_dim=rubric_dim,
        scale=scale,
        judge_model=judge_model,
        n=len(pairs),
        raters=sorted({rater for _, _, rater in pairs}),
        exact_agreement=sum(1 for judge, person in scored if judge == person) / len(scored),
        kappa=kappa,
        weighted_kappa=weighted,
    )


__all__ = [
    "DimensionAgreement",
    "JudgeAgreement",
    "UnpairedRating",
    "UnpairedReason",
    "judge_agreement",
]
