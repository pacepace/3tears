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

**Several people, one judge: the judge is set against each person, and the kappas pooled by pairs.** Cohen's
kappa is a two-rater statistic. Pooling every (judge, person) pair into one table would enter a result two people
rated twice — the judge's score duplicated, the items no longer independent — and read the people's
disagreement with each other as the judge's with them. So each person's kappa is computed over the results
that person rated, and the dimension's kappa (and weighted kappa) is the mean of those per-person kappas
**weighted by each person's pairs**: a person with 2 ratings moves it a tenth as far as one with 20. (An
unweighted mean let the small rater outvote the large one — 20 ratings at 0.3 beside 2 at 1.0 read 0.65 — and
since the figure decides the ``calibrated`` tier that was an overclaim, not a presentation choice.) With one
person it is Cohen's kappa. A person whose kappa is undefined (every pair one score — which includes a person
who agreed with the judge perfectly on a constant score) is EXCLUDED from the mean rather than counted, so
such agreement does not raise it, and their results are not among ``results``; the dimension's kappa is
undefined only when every person's is. ``n`` and ``exact_agreement`` stay per rating — counts, which nothing
double-weights. ``results`` is the distinct results the pooled kappa covers, and it is what the evidence
tiers' floor counts (:mod:`threetears.evals.contracts.evidence_tiers`), because ratings can pile onto a few
results and results cannot.

**One group per dimension, scale and judge** (:class:`JudgeKey`). The judge is the model that served the
score (:attr:`~threetears.evals.contracts.models.RubricScore.served_model`) AND the versioned judge config
that asked for it (the result's ``judge_config_ids``; ``None`` = the built-in prompt), so a campaign sweeping
its judge — model or prompt — reads each judge's agreement separately: pooling them would credit one judge
with the other's calibration, which is the comparison a judge swap is decided on. A dimension rated on a
scale it was later moved off is two groups for the same reason.

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


class JudgeKey(NamedTuple):
    """Who judged a reading: the dimension, its scale, the model that served the score and the config that asked.

    The one key both agreements group by, the tiers are listed by, and a reading looks its tier up by — so a
    measurement of one judge can never stand in for another's.
    """

    rubric_dim: str
    scale: RubricScale
    judge_model: str | None
    judge_config_id: str | None


def judge_key(result: EvalResult, dim: str) -> JudgeKey | None:
    """The judge behind ``result``'s score on ``dim``, or None when it holds no score there.

    Args:
        result: The result.
        dim: The dimension, as its score spells it.

    Returns:
        The key: the score's scale and served model, and the config the result records as having scored the dim.
    """
    score = result.judge_score(dim)
    if score is None:
        return None
    return JudgeKey(dim, score.scale, score.served_model, result.judge_config_ids.get(dim))


def _sort_key(key: JudgeKey) -> tuple[str, str, str, str]:
    """Order keys by dimension, scale, judge and config, an unnamed judge or the built-in prompt first."""
    return (key.rubric_dim, key.scale, key.judge_model or "", key.judge_config_id or "")


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
    judge_config_id: str | None = Field(
        description="The versioned JudgeConfig that asked for the scores; None = the built-in prompt."
    )
    n: int = Field(
        ge=1,
        description=(
            "Pairs read: one per rating, so two raters of one result are two pairs. The kappas are per person and "
            "pooled by pairs (see `kappa`), so this count enters no kappa twice."
        ),
    )
    results: int = Field(
        ge=0,
        description=(
            "The distinct results rated by the people whose kappa entered `kappa` — what the `calibrated` tier's "
            "floor counts. 0 when every person's kappa is undefined."
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
            "then the mean over people weighted by each person's pairs, so a person who rated 2 results moves "
            "it a tenth as far as one who rated 20 (with one person, Cohen's kappa). A person whose kappa is "
            "undefined is excluded from the mean. None when every person's is undefined — chance alone predicts "
            "no disagreement, judge and person giving one and the same score to every pair — where it is "
            "undefined, not perfect."
        ),
    )
    weighted_kappa: float | None = Field(
        description=(
            "Cohen's kappa with quadratic weights over the 1-5 scale, per person and pooled as `kappa` is. None on "
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
            "One per (dimension, scale, judge, judge config) with at least one pair, ordered by those four. Empty "
            "when nobody rated anything — the judge is then uncalibrated against people, which is a state, not a "
            "zero."
        ),
    )
    unpaired: list[UnpairedRating] = Field(
        default_factory=list, description="Ratings that could not be paired, in the order they were read."
    )


class _Pair(NamedTuple):
    """One pair either agreement reads: the judge's score, the other side's answer, who gave it, and about what.

    ``other`` is None for a repeat that answered "can't tell" — a pair whose second half is no score at all.
    """

    judge: int
    other: int | None
    rater: str
    result_id: str


def judge_agreement(ratings: Iterable[CalibrationRating], results: Iterable[EvalResult]) -> JudgeAgreement:
    """Pair each rating with the judge's score on the same dimension of the same result, and read agreement.

    Args:
        ratings: The ratings to read.
        results: The results they may rate. A rating whose result is not here is unpaired
            (``result_unresolved``), so hand in every result the ratings were read for.

    Returns:
        The agreement per (dimension, scale, judge, judge config), and the ratings that could not be paired.
    """
    by_id = {result.id: result for result in results}
    groups: dict[JudgeKey, list[_Pair]] = {}
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
        key = judge_key(result, rating.rubric_dim)
        if score is None or key is None:
            unpaired.append(_unpaired(rating, "dimension_unscored"))
            continue
        if score.scale != rating.scale:
            unpaired.append(_unpaired(rating, "scale_changed"))
            continue
        groups.setdefault(key, []).append(_Pair(score.score, rating.score, rating.rater, result.id))
    dimensions = []
    for key in sorted(groups, key=_sort_key):
        numbers = _agreement_numbers(key.scale, groups[key])
        dimensions.append(
            DimensionAgreement(
                rubric_dim=key.rubric_dim,
                scale=key.scale,
                judge_model=key.judge_model,
                judge_config_id=key.judge_config_id,
                n=numbers.n,
                results=numbers.results,
                raters=numbers.raters,
                exact_agreement=numbers.exact_agreement,
                kappa=numbers.kappa,
                weighted_kappa=numbers.weighted_kappa,
            )
        )
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


#: The category a "can't tell" repeat is read in: off the scale, so maximally far from every score.
_CANNOT_TELL_CATEGORY = -1


class _AgreementNumbers(NamedTuple):
    """One group's agreement, as both reads compute it."""

    n: int
    results: int
    raters: list[str]
    exact_agreement: float
    kappa: float | None
    weighted_kappa: float | None
    cannot_tell: int


def _agreement_numbers(scale: RubricScale, pairs: Sequence[_Pair]) -> _AgreementNumbers:
    """Read one group's pairs: the ONE computation calibration and self-agreement share.

    Each rater's Cohen's kappa over the pairs that rater gave, then the mean of the defined ones weighted by
    each rater's pairs (see the module docstring and :mod:`threetears.evals.contracts.evidence_tiers`); on a
    1-5 scale the same with quadratic weights. A "can't tell" answer is its own category, maximally far from
    every score, and a disagreement in exact agreement. ``results`` counts the distinct results among the
    raters whose kappa is defined — the raters the pooled figure actually rests on.

    Args:
        scale: The group's scale.
        pairs: At least one.

    Returns:
        The numbers.
    """
    low, high = SCALES[scale].scores
    categories = list(range(low, high + 1))
    by_rater: dict[str, list[_Pair]] = {}
    for pair in pairs:
        by_rater.setdefault(pair.rater, []).append(pair)

    def kappas(weights: Literal["none", "quadratic"]) -> list[tuple[float | None, list[_Pair]]]:
        return [
            (
                cohen_kappa(
                    [(p.judge, _CANNOT_TELL_CATEGORY if p.other is None else p.other) for p in own],
                    categories,
                    weights=weights,
                    unordered=[_CANNOT_TELL_CATEGORY],
                ),
                own,
            )
            for own in by_rater.values()
        ]

    plain = kappas("none")
    figure = kappas("quadratic") if scale == "ordinal" else plain
    covered = {pair.result_id for kappa, own in figure for pair in own if kappa is not None}
    return _AgreementNumbers(
        n=len(pairs),
        results=len(covered),
        raters=sorted(by_rater),
        exact_agreement=sum(1 for p in pairs if p.other is not None and p.judge == p.other) / len(pairs),
        kappa=_pooled_kappa(plain),
        weighted_kappa=_pooled_kappa(figure) if scale == "ordinal" else None,
        cannot_tell=sum(1 for p in pairs if p.other is None),
    )


def _pooled_kappa(per_rater: Sequence[tuple[float | None, Sequence[_Pair]]]) -> float | None:
    """The mean of the defined per-rater kappas weighted by each rater's pairs (undefined ones excluded), or None.

    Weighted by pairs so that a rater's pull on the figure is the evidence they gave: a rater with 2 pairs
    beside one with 20 moves it a tenth as far, and cannot carry a figure the larger rater's evidence misses.
    """
    defined = [(kappa, len(own)) for kappa, own in per_rater if kappa is not None]
    total = sum(weight for _, weight in defined)
    return sum(kappa * weight for kappa, weight in defined) / total if total else None


# ---------------------------------------------------------------------------
# The judge against itself
# ---------------------------------------------------------------------------

#: Why a repeated score has no pair to be read in. ``repeat_failed``: the repeat call failed — an
#: infrastructure fault, which says nothing about the judge. ``judge_changed``: a different model
#: served the repeat than served the first score, so the pair would measure two judges' agreement, not
#: one judge's. ``config_changed``: a different judge config answered the repeat than scored the first
#: score — a different prompt is a different judge for the same reason. (A repeat answering "can't
#: tell" IS paired: declining to score what it once scored is the judge disagreeing with itself.)
UnrepeatedReason = Literal["repeat_failed", "judge_changed", "config_changed"]


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
    judge_config_id: str | None = Field(
        description="The versioned JudgeConfig that asked both times; None = the built-in prompt."
    )
    n: int = Field(
        ge=1, description='First-score/repeat pairs read: one per repeated score, a "can\'t tell" repeat included.'
    )
    results: int = Field(
        ge=0,
        description=(
            "The distinct results among the rounds whose kappa entered `kappa` — what the `separation` tier's floor "
            "counts, so repeating a few results many times cannot reach it. 0 when every round's kappa is undefined."
        ),
    )
    n_cannot_tell: int = Field(
        ge=0,
        description=(
            "Pairs whose repeat answered it could not tell on a dimension the judge had scored. Counted in `n`, "
            "`results` and as disagreements in `exact_agreement`, and read in both kappas as a category of its own, "
            "maximally far from every score."
        ),
    )
    rounds: list[str] = Field(
        min_length=1,
        description=(
            "The repeat rounds among the pairs, sorted: `repeat 1` is each result's first repeat of the "
            "dimension, `repeat 2` its second. Each round is a rater, so the kappas pool per round, weighted by "
            "the round's pairs, as calibration pools per person."
        ),
    )
    exact_agreement: float = Field(
        ge=0.0,
        le=1.0,
        description='The share of pairs where the repeat gave the same score; a "can\'t tell" never does.',
    )
    kappa: float | None = Field(
        description="Cohen's kappa per round, pooled by pairs; None when every round's is undefined."
    )
    weighted_kappa: float | None = Field(
        description="Quadratic-weighted kappa per round, pooled as `kappa` is. None on pass/fail, and when undefined."
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
            "One per (dimension, scale, judge, judge config) with at least one pair, ordered by those four. Empty "
            "when nothing was repeated — the judge's consistency is then unmeasured, which is a state, not a zero."
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
    same statistic over the same pooling, and the two tiers they decide compare.

    A "can't tell" repeat is paired, as a disagreement (see :class:`SelfAgreementDimension`). Its judge is
    checked by config only — it carries no score, so no served model to compare.

    **Stated limits.** Only a dimension the result holds a SCORE on is repeated, so the reverse flip — "can't
    tell" first, a score on repeat — is never observed. And rounds pool by their pairs, so a few results
    repeated many times beside many results repeated once weigh by their repeats; the floor guarantees the
    figure rests on enough distinct results, and ``results`` beside ``n`` shows how concentrated it is.

    Args:
        results: The results whose repeats to read.

    Returns:
        The agreement per (dimension, scale, judge, judge config), and the repeated scores that could not be paired.
    """
    groups: dict[JudgeKey, list[_Pair]] = {}
    unpaired: list[UnrepeatedScore] = []
    read = 0
    for result in results:
        rounds: dict[str, int] = {}
        for repeat in result.judge_repeats:
            for entry in repeat.scores:
                read += 1
                rounds[entry.dim] = rounds.get(entry.dim, 0) + 1
                round_name = f"repeat {rounds[entry.dim]}"
                reason: UnrepeatedReason | None = None
                if entry.repeat is None and entry.cannot_tell is None:
                    reason = "repeat_failed"
                elif entry.repeat is not None and entry.repeat.served_model != entry.first_served_model:
                    reason = "judge_changed"
                elif repeat.judge_config_ids.get(entry.dim) != entry.first_judge_config_id:
                    reason = "config_changed"
                if reason is not None:
                    unpaired.append(
                        UnrepeatedScore(result_id=result.id, rubric_dim=entry.dim, round=round_name, reason=reason)
                    )
                    continue
                key = JudgeKey(entry.dim, entry.scale, entry.first_served_model, entry.first_judge_config_id)
                again = entry.repeat.score if entry.repeat is not None else None
                groups.setdefault(key, []).append(_Pair(entry.first_score, again, round_name, result.id))
    dimensions = []
    for key in sorted(groups, key=_sort_key):
        numbers = _agreement_numbers(key.scale, groups[key])
        dimensions.append(
            SelfAgreementDimension(
                rubric_dim=key.rubric_dim,
                scale=key.scale,
                judge_model=key.judge_model,
                judge_config_id=key.judge_config_id,
                n=numbers.n,
                results=numbers.results,
                n_cannot_tell=numbers.cannot_tell,
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
    judged: Iterable[JudgeKey],
) -> list[JudgeEvidenceTier]:
    """Decide the evidence tier of every judge's readings on every dimension, from the two agreements.

    One entry per :class:`JudgeKey` among ``judged`` and every group either agreement read, so a judge
    nobody rated and nobody repeated is listed — its criteria at ``n=0``, its tier ``undetermined`` —
    rather than absent, which a reader would take for unjudged.

    Args:
        agreement: The judge's agreement with people (:func:`judge_agreement`).
        self_agreement: The judge's agreement with itself (:func:`judge_self_agreement`).
        judged: Every judge a judged score was read under (:func:`judge_key`).

    Returns:
        The tiers, ordered by dimension, scale, judge and judge config.
    """
    calibrations = {JudgeKey(d.rubric_dim, d.scale, d.judge_model, d.judge_config_id): d for d in agreement.dimensions}
    repeats = {JudgeKey(d.rubric_dim, d.scale, d.judge_model, d.judge_config_id): d for d in self_agreement.dimensions}
    keys = {*(JudgeKey(*key) for key in judged), *calibrations, *repeats}
    tiers = []
    for key in sorted(keys, key=_sort_key):
        people = calibrations.get(key)
        itself = repeats.get(key)
        calibration = calibration_criterion(
            people.n if people else 0,
            people.results if people else 0,
            agreement_statistic(key.scale, people.kappa, people.weighted_kappa) if people else None,
        )
        separation = separation_criterion(
            itself.n if itself else 0,
            itself.results if itself else 0,
            agreement_statistic(key.scale, itself.kappa, itself.weighted_kappa) if itself else None,
        )
        tiers.append(
            JudgeEvidenceTier(
                rubric_dim=key.rubric_dim,
                scale=key.scale,
                judge_model=key.judge_model,
                judge_config_id=key.judge_config_id,
                tier=tier_of(calibration, separation),
                calibration=calibration,
                separation=separation,
            )
        )
    return tiers


def tier_for_judges(tiers: Iterable[JudgeEvidenceTier], judges: Iterable[JudgeKey]) -> JudgedEvidenceTier:
    """The tier a reading stands on when ``judges`` served its scores: the weakest of theirs.

    Looked up by the whole :class:`JudgeKey` — dimension, scale, served model and config — so a reading
    can only ever carry the tier measured for the very judge behind it. A cell whose scores were served by
    two judges pools two judges' readings, and the composite can bear only what the weaker can. A judge
    with no entry is ``undetermined``: nothing measured it.

    Args:
        tiers: The tiers, from :func:`judge_evidence_tiers`.
        judges: The judges behind the reading's scores (:func:`judge_key`).

    Returns:
        The tier; ``undetermined`` when no judge is named — a reading with no score behind it has no judge.
    """
    by_judge = {
        JudgeKey(tier.rubric_dim, tier.scale, tier.judge_model, tier.judge_config_id): tier.tier for tier in tiers
    }
    found: list[JudgedEvidenceTier] = [by_judge.get(JudgeKey(*key), "undetermined") for key in set(judges)]
    return weakest_judged_tier(found) if found else "undetermined"


def tier_sentence(tier: JudgeEvidenceTier) -> str:
    """One sentence a report states for a judge's tier on a dimension: the tier, and the two measurements behind it.

    Args:
        tier: The tier as decided.

    Returns:
        The sentence, naming the judge (and its config, when one asked), the tier and each criterion's
        agreement, pairs and results against its bar.
    """
    judge = tier.judge_model or "an unnamed judge"
    if tier.judge_config_id is not None:
        judge = f"{judge}, config {tier.judge_config_id}"
    return (
        f"{tier.rubric_dim} ({judge}): {tier.tier} — agreement with people "
        f"{_criterion_words(tier.calibration)}; with its own repeats {_criterion_words(tier.separation)}."
    )


def _criterion_words(criterion: TierCriterion) -> str:
    """A criterion as a clause: its agreement, pairs and results against the bar, or why there is nothing to read."""
    bar = f"(bar {format_number(criterion.threshold)} over at least {criterion.min_results} results)"
    if criterion.n == 0:
        return f"not measured {bar}"
    if criterion.agreement is None:
        return f"undefined over {criterion.n} pairs {bar}"
    over = f"over {criterion.n} pairs from {criterion.results} results"
    verdict = {"met": "meets", "not_met": "misses", "insufficient": "too few results for"}[criterion.state]
    return f"{format_number(criterion.agreement)} {over}, {verdict} {bar}"


__all__ = [
    "DimensionAgreement",
    "JudgeAgreement",
    "JudgeKey",
    "JudgeSelfAgreement",
    "SelfAgreementDimension",
    "UnpairedRating",
    "UnpairedReason",
    "UnrepeatedReason",
    "UnrepeatedScore",
    "judge_agreement",
    "judge_evidence_tiers",
    "judge_key",
    "judge_self_agreement",
    "tier_for_judges",
    "tier_sentence",
]
