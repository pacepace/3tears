"""Judge-versus-human agreement: how often the judge scored a dimension the way people did.

A judge's consistency across repeats measures its precision; only people can say whether it is
right. People say so in :class:`~threetears.evals.contracts.models.CalibrationRating` documents, one
per rater and kind of rater per dimension per result, and this module sets each beside the score the judge gave the
same dimension of the same result and reads the pairs per dimension:

- **n** — the pairs, because a kappa over four of them is a different claim from one over forty;
- **exact agreement** — the share of pairs where the two gave the same score;
- **Cohen's kappa** — the same agreement net of what the two raters' own score distributions would
  produce by chance;
- **quadratic-weighted kappa** — on a 1-5 dimension only, where a 4 against a 5 is a near miss and
  a 1 against a 5 is not. On pass/fail it would be the unweighted number restated, so it is absent.

**Several people, one judge: the judge is set against each person, and the kappas averaged.** Cohen's kappa
is a two-rater statistic. Pooling every (judge, person) pair into one table would enter a result two people
rated twice — the judge's score duplicated, the items no longer independent — and read the people's
disagreement with each other as the judge's with them. So each person's kappa is computed over the results
that person rated, and the dimension's kappa (and weighted kappa) is the UNWEIGHTED mean of those per-person
kappas. Every rater counts once, however many results they rated: a person with 2 ratings moves the mean as
much as one with 200. Each person's kappa is over their own subset of results, so this is a Light-style
average (after Light, 1971), not Light's statistic over one common item set. With one person it is Cohen's
kappa. A person whose kappa is undefined (every pair one score — which includes a person who agreed with the
judge perfectly on a constant score) is EXCLUDED from the mean rather than counted, so such agreement does not
raise it; the dimension's kappa is undefined only when every person's is. ``n`` and ``exact_agreement`` stay
per rating — counts, which nothing double-weights.

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
from typing import TYPE_CHECKING, Literal, NamedTuple

from pydantic import Field

from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.stats import cohen_kappa
from threetears.evals.contracts.base import EvalDocumentModel
from threetears.evals.contracts.evidence_tiers import (
    JudgedEvidenceTier,
    JudgeEvidenceTier,
    TierCriterion,
    agreement_statistic,
    calibration_criterion,
    separation_criterion,
    tier_of,
    weakest_judged_tier,
)
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
    n: int = Field(
        ge=1,
        description=(
            "Pairs read: one per rating, so two raters of one result are two pairs. The kappas are per person and "
            "averaged (see `kappa`), so this count enters no kappa twice."
        ),
    )
    raters: list[str] = Field(
        min_length=1, description="Every person whose ratings are among the pairs, sorted; never an agent."
    )
    exact_agreement: float = Field(
        ge=0.0, le=1.0, description="The share of pairs where judge and person gave the same score."
    )
    kappa: float | None = Field(
        description=(
            "Cohen's kappa, unweighted, of the judge against each person over the results that person rated, "
            "then the unweighted mean over people: each person counts once, however many results they rated "
            "(a Light-style average; with one person, Cohen's kappa). A person whose kappa is undefined is "
            "excluded from the mean. None when every person's is undefined — chance alone predicts no "
            "disagreement, judge and person giving one and the same score to every pair — where it is undefined, "
            "not perfect."
        ),
    )
    weighted_kappa: float | None = Field(
        description=(
            "Cohen's kappa with quadratic weights over the 1-5 scale, per person and averaged as `kappa` is. None on "
            "pass/fail, where it equals `kappa`, and wherever every person's is undefined."
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


class _AgreementNumbers(NamedTuple):
    """One group's agreement, as both reads compute it: pairs, raters, exact agreement and the per-rater kappas."""

    n: int
    raters: list[str]
    exact_agreement: float
    kappa: float | None
    weighted_kappa: float | None


def _agreement_numbers(scale: RubricScale, pairs: Sequence[tuple[int, int, str]]) -> _AgreementNumbers:
    """Read one group's pairs: the ONE computation calibration and self-agreement share.

    Each rater's Cohen's kappa over the pairs that rater gave, then the unweighted mean of the defined
    ones (see the module docstring); on a 1-5 scale the same with quadratic weights.

    Args:
        scale: The group's scale.
        pairs: ``(judge's score, the other side's score, rater)`` per pair, at least one.

    Returns:
        The numbers.
    """
    low, high = SCALES[scale].scores
    categories = list(range(low, high + 1))
    by_rater: dict[str, list[tuple[int, int]]] = {}
    for judge, other, rater in pairs:
        by_rater.setdefault(rater, []).append((judge, other))
    kappa = _mean_kappa([cohen_kappa(own, categories) for own in by_rater.values()])
    weighted = (
        _mean_kappa([cohen_kappa(own, categories, weights="quadratic") for own in by_rater.values()])
        if scale == "ordinal"
        else None
    )
    return _AgreementNumbers(
        n=len(pairs),
        raters=sorted(by_rater),
        exact_agreement=sum(1 for judge, other, _ in pairs if judge == other) / len(pairs),
        kappa=kappa,
        weighted_kappa=weighted,
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
    numbers = _agreement_numbers(scale, pairs)
    return DimensionAgreement(
        rubric_dim=rubric_dim,
        scale=scale,
        judge_model=judge_model,
        n=numbers.n,
        raters=numbers.raters,
        exact_agreement=numbers.exact_agreement,
        kappa=numbers.kappa,
        weighted_kappa=numbers.weighted_kappa,
    )


# ---------------------------------------------------------------------------
# The judge against itself
# ---------------------------------------------------------------------------

#: Why a repeated score has no pair to be read in. ``repeat_failed``: the repeat call failed.
#: ``repeat_cannot_tell``: the repeat answered it could not score a dimension it had scored — a
#: disagreement no kappa can hold, so it is counted here rather than dropped. ``judge_changed``: a
#: different model served the repeat than served the first score, so the pair would measure two
#: judges' agreement, not one judge's.
UnrepeatedReason = Literal["repeat_failed", "repeat_cannot_tell", "judge_changed"]


class SelfAgreementDimension(EvalDocumentModel):
    """How one judge's repeated scores on one dimension agreed with its first scores of the same evidence."""

    rubric_dim: str = Field(min_length=1, description="The judged dimension.")
    scale: RubricScale = Field(description="The scale it was judged on.")
    judge_model: str | None = Field(
        description=(
            "The model that served both the first score and its repeat; None when neither response named one, "
            "a judge nobody observed, never read as a match for a named one."
        ),
    )
    n: int = Field(ge=1, description="First-score/repeat pairs read: one per repeated score.")
    rounds: list[str] = Field(
        min_length=1,
        description=(
            "The repeat rounds among the pairs, sorted: `repeat 1` is each result's first repeat of the "
            "dimension, `repeat 2` its second. Each round is a rater, so the kappas average per round as "
            "calibration averages per person."
        ),
    )
    exact_agreement: float = Field(ge=0.0, le=1.0, description="The share of pairs where the two scores were equal.")
    kappa: float | None = Field(description="Cohen's kappa per round, averaged; None when every round's is undefined.")
    weighted_kappa: float | None = Field(
        description="Quadratic-weighted kappa per round, averaged as `kappa` is. None on pass/fail, and when undefined."
    )


class UnrepeatedScore(EvalDocumentModel):
    """A repeated score with no pair to read, and why."""

    result_id: str = Field(min_length=1, description="The result whose score was repeated.")
    rubric_dim: str = Field(min_length=1, description="The dimension.")
    round: str = Field(min_length=1, description="Which repeat of the dimension it was, as `rounds` names it.")
    reason: UnrepeatedReason


class JudgeSelfAgreement(EvalDocumentModel):
    """Every repeated score read, paired with the first score where it can be, and agreement per dimension and judge."""

    repeats_read: int = Field(default=0, ge=0, description="Every repeated score read: the pairs plus the unpaired.")
    dimensions: list[SelfAgreementDimension] = Field(
        default_factory=list,
        description=(
            "One per (dimension, scale, judge) with at least one pair, ordered by those three. Empty when nothing "
            "was repeated — the judge's consistency is then unmeasured, which is a state, not a zero."
        ),
    )
    unpaired: list[UnrepeatedScore] = Field(
        default_factory=list, description="Repeated scores that could not be paired, in the order they were read."
    )


def judge_self_agreement(results: Iterable[EvalResult]) -> JudgeSelfAgreement:
    """Pair each repeated score with the first score it repeated, and read agreement the way calibration does.

    The pair is the one the repeat recorded (:class:`~threetears.evals.contracts.models.RepeatedScore`),
    so a re-judge that later rewrote the result's score does not split it. The repeat stands where the
    person stands in :func:`judge_agreement` and each round of repeats is a rater, so the figures are the
    same statistic over the same averaging, and the two tiers they decide compare.

    Args:
        results: The results whose repeats to read.

    Returns:
        The agreement per (dimension, scale, judge), and the repeated scores that could not be paired.
    """
    groups: dict[tuple[str, RubricScale, str | None], list[tuple[int, int, str]]] = {}
    unpaired: list[UnrepeatedScore] = []
    read = 0
    for result in results:
        rounds: dict[str, int] = {}
        for repeat in result.judge_repeats:
            for entry in repeat.scores:
                read += 1
                rounds[entry.dim] = rounds.get(entry.dim, 0) + 1
                round_name = f"repeat {rounds[entry.dim]}"
                reason: UnrepeatedReason
                if entry.repeat is None:
                    reason = "repeat_cannot_tell" if entry.cannot_tell is not None else "repeat_failed"
                elif entry.repeat.served_model != entry.first_served_model:
                    reason = "judge_changed"
                else:
                    key = (entry.dim, entry.scale, entry.first_served_model)
                    groups.setdefault(key, []).append((entry.first_score, entry.repeat.score, round_name))
                    continue
                unpaired.append(
                    UnrepeatedScore(result_id=result.id, rubric_dim=entry.dim, round=round_name, reason=reason)
                )
    dimensions = []
    for dim, scale, judge in sorted(groups, key=lambda key: (key[0], key[1], key[2] or "")):
        numbers = _agreement_numbers(scale, groups[(dim, scale, judge)])
        dimensions.append(
            SelfAgreementDimension(
                rubric_dim=dim,
                scale=scale,
                judge_model=judge,
                n=numbers.n,
                rounds=numbers.raters,
                exact_agreement=numbers.exact_agreement,
                kappa=numbers.kappa,
                weighted_kappa=numbers.weighted_kappa,
            )
        )
    return JudgeSelfAgreement(repeats_read=read, dimensions=dimensions, unpaired=unpaired)


# ---------------------------------------------------------------------------
# The tiers the two agreements decide
# ---------------------------------------------------------------------------


def judge_evidence_tiers(
    agreement: JudgeAgreement,
    self_agreement: JudgeSelfAgreement,
    judged: Iterable[tuple[str, RubricScale, str | None]],
) -> list[JudgeEvidenceTier]:
    """Decide the evidence tier of every judge's readings on every dimension, from the two agreements.

    One entry per ``(dimension, scale, judge)`` among ``judged`` and every group either agreement read,
    so a judge nobody rated and nobody repeated is listed — its criteria at ``n=0``, its tier
    ``undetermined`` — rather than absent, which a reader would take for unjudged.

    Args:
        agreement: The judge's agreement with people (:func:`judge_agreement`).
        self_agreement: The judge's agreement with itself (:func:`judge_self_agreement`).
        judged: Every ``(dimension, scale, served model)`` a judged score was read under.

    Returns:
        The tiers, ordered by dimension, scale and judge.
    """
    calibrations = {(d.rubric_dim, d.scale, d.judge_model): d for d in agreement.dimensions}
    repeats = {(d.rubric_dim, d.scale, d.judge_model): d for d in self_agreement.dimensions}
    keys = {*judged, *calibrations, *repeats}
    tiers = []
    for dim, scale, judge in sorted(keys, key=lambda key: (key[0], key[1], key[2] or "")):
        people = calibrations.get((dim, scale, judge))
        itself = repeats.get((dim, scale, judge))
        calibration = calibration_criterion(
            people.n if people else 0,
            agreement_statistic(scale, people.kappa, people.weighted_kappa) if people else None,
        )
        separation = separation_criterion(
            itself.n if itself else 0,
            agreement_statistic(scale, itself.kappa, itself.weighted_kappa) if itself else None,
        )
        tiers.append(
            JudgeEvidenceTier(
                rubric_dim=dim,
                scale=scale,
                judge_model=judge,
                tier=tier_of(calibration, separation),
                calibration=calibration,
                separation=separation,
            )
        )
    return tiers


def tier_for_judges(
    tiers: Iterable[JudgeEvidenceTier], rubric_dim: str, judge_models: Iterable[str | None]
) -> JudgedEvidenceTier:
    """The tier a reading of ``rubric_dim`` stands on when ``judge_models`` served its scores: the weakest of theirs.

    A cell whose scores were served by two models pools two judges' readings, and the composite can
    bear only what the weaker judge can. A judge with no entry is ``undetermined``: nothing measured it.

    Args:
        tiers: The tiers, from :func:`judge_evidence_tiers`.
        rubric_dim: The dimension.
        judge_models: The served models behind the reading's scores.

    Returns:
        The tier; ``undetermined`` when no model is named — a reading with no score behind it has no judge.
    """
    by_judge = {tier.judge_model: tier.tier for tier in tiers if tier.rubric_dim == rubric_dim}
    found: list[JudgedEvidenceTier] = [by_judge.get(model, "undetermined") for model in set(judge_models)]
    return weakest_judged_tier(found) if found else "undetermined"


def tier_sentence(tier: JudgeEvidenceTier) -> str:
    """One sentence a report states for a judge's tier on a dimension: the tier, and the two measurements behind it.

    Args:
        tier: The tier as decided.

    Returns:
        The sentence, naming the judge, the tier and each criterion's agreement and n against its bar.
    """
    judge = tier.judge_model or "an unnamed judge"
    return (
        f"{tier.rubric_dim} ({judge}): {tier.tier} — agreement with people "
        f"{_criterion_words(tier.calibration)}; with its own repeats {_criterion_words(tier.separation)}."
    )


def _criterion_words(criterion: TierCriterion) -> str:
    """A criterion as a clause: its agreement and n against the bar, or why there is nothing to read."""
    bar = f"(bar {format_number(criterion.threshold)} over at least {criterion.min_pairs} pairs)"
    if criterion.n == 0:
        return f"not measured {bar}"
    if criterion.agreement is None:
        return f"undefined over {criterion.n} pairs {bar}"
    verdict = {"met": "meets", "not_met": "misses", "insufficient": "too few pairs for"}[criterion.state]
    return f"{format_number(criterion.agreement)} over {criterion.n} pairs, {verdict} {bar}"


def _mean_kappa(kappas: Sequence[float | None]) -> float | None:
    """The unweighted mean of the defined per-person kappas (each person once; undefined ones excluded), or None."""
    defined = [kappa for kappa in kappas if kappa is not None]
    return sum(defined) / len(defined) if defined else None


__all__ = [
    "DimensionAgreement",
    "JudgeAgreement",
    "JudgeSelfAgreement",
    "SelfAgreementDimension",
    "UnpairedRating",
    "UnpairedReason",
    "UnrepeatedReason",
    "UnrepeatedScore",
    "judge_agreement",
    "judge_evidence_tiers",
    "judge_self_agreement",
    "tier_for_judges",
    "tier_sentence",
]
