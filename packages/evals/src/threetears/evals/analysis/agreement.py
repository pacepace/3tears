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

**Several people, one judge: the judge is set against each person, and the kappas pooled by result.** Cohen's
kappa is a two-rater statistic. Pooling every (judge, person) pair into one table would enter a result two people
rated twice — the judge's score duplicated, the items no longer independent — and read the people's
disagreement with each other as the judge's with them. So each person's kappa is computed over the results
that person rated, and the dimension's kappa (and weighted kappa) is the mean of those per-person kappas
**weighted by the results each person measured, each distinct result carrying weight 1** split evenly across
the people who rated it: a person's weight is the sum over their ratings of ``1 / (people who rated that
result)``. The figure therefore weighs what the evidence tiers' floor counts (distinct results), and neither a
small rater nor many small raters can carry it: 20 ratings at 0.44 beside five annotators who each matched the
judge on the same 3 shared anchors read 0.51, where an unweighted mean (0.91) or a pair-weighted one (0.68) put
the judge over the ``calibrated`` bar on the strength of three results. With one person it is Cohen's kappa. A
person whose kappa is undefined (every pair one score — which includes a person who agreed with the judge
perfectly on a constant score) is EXCLUDED from the mean before the weights are split, so such agreement does not
raise it, and results only they rated are not among ``results`` and carry no weight; the dimension's kappa is
undefined only when every person's is. ``n`` and ``exact_agreement`` stay per rating — counts, which nothing
double-weights. ``results`` is the distinct results the pooled kappa covers, and it is what the evidence tiers'
floor counts (:mod:`threetears.evals.contracts.evidence_tiers`).

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

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal, NamedTuple

from pydantic import Field

from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.stats import (
    INTERVAL_LEVEL,
    KappaMoments,
    cohen_kappa,
    kappa_moments,
    t_critical_two_sided,
)
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
            "pooled by result (see `kappa`), so this count enters no kappa twice."
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
            "then the mean over people weighted by result: each distinct result weighs 1, split across the people "
            "who rated it, so many people re-rating a few shared results weigh those few results and no more "
            "(with one person, Cohen's kappa). A person whose kappa is "
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
    agreement_interval: tuple[float, float] | None = Field(
        default=None,
        description=(
            "The 95% interval on the figure the `calibrated` tier reads (`weighted_kappa` on 1-5, `kappa` on "
            "pass/fail), over the distinct results — `agreement_interval`. None when that figure is undefined or "
            "rests on fewer than two results."
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
                agreement_interval=numbers.interval,
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
    interval: tuple[float, float] | None


def _agreement_numbers(scale: RubricScale, pairs: Sequence[_Pair]) -> _AgreementNumbers:
    """Read one group's pairs: the ONE computation calibration and self-agreement share.

    Each rater's Cohen's kappa over the pairs that rater gave, then the mean of the defined ones weighted by
    result — each distinct result weighing 1, split across the raters that measured it (see the module docstring and :mod:`threetears.evals.contracts.evidence_tiers`); on a
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
    weights: Literal["none", "quadratic"] = "quadratic" if scale == "ordinal" else "none"
    figure = kappas("quadratic") if scale == "ordinal" else plain
    covered = {pair.result_id for kappa, own in figure for pair in own if kappa is not None}
    pooled = _pooled_kappa(figure)
    interval = None
    if pooled is not None:
        moments = [
            (
                kappa_moments(
                    [(p.judge, _CANNOT_TELL_CATEGORY if p.other is None else p.other) for p in own],
                    categories,
                    weights=weights,
                    unordered=[_CANNOT_TELL_CATEGORY],
                ),
                [p.result_id for p in own],
            )
            for kappa, own in figure
            if kappa is not None
        ]
        interval = agreement_interval(pooled, [(m, ids) for m, ids in moments if m is not None])
    return _AgreementNumbers(
        n=len(pairs),
        results=len(covered),
        raters=sorted(by_rater),
        exact_agreement=sum(1 for p in pairs if p.other is not None and p.judge == p.other) / len(pairs),
        kappa=_pooled_kappa(plain),
        weighted_kappa=pooled if scale == "ordinal" else None,
        cannot_tell=sum(1 for p in pairs if p.other is None),
        interval=interval,
    )


def agreement_interval(
    estimate: float, raters: Sequence[tuple[KappaMoments, Sequence[str]]]
) -> tuple[float, float] | None:
    """The 95% interval on a pooled agreement figure: a score interval over the distinct results it rests on.

    **Why not the estimate plus or minus a standard error.** At the 20-result floor kappa's sampling spread is
    about 0.2 and its spread shrinks as agreement rises, so a standard error read off the estimate is smallest
    exactly when the estimate is luckiest: a jackknife interval awarded ``calibrated`` to a judge at the bar
    10-25% of the time, and twenty results that all happen to agree read as certainty. A score interval holds
    each candidate value ``κ0`` to the spread kappa WOULD have there — the set of ``κ0`` the estimate is within
    ``t`` of — the Wilson interval's construction, which it reduces to on pass/fail.

    **The spread at ``κ0``.** Kappa is ``1 - D / D_e``: ``D`` the mean disagreement cost over the items, ``D_e``
    the cost chance gives the two raters' marginals. At ``κ0`` the mean cost is ``(1 - κ0) D_e``, and a cost
    ``c`` in ``[0, 1]`` with mean ``m`` has variance ``E[c²] - m²`` with ``E[c²] = ρ m``, where ``ρ`` is how
    large a disagreement is when there is one. On pass/fail every disagreement costs 1 (``ρ = 1``, Wilson
    exactly, with no model). On 1-5 ``ρ`` is the larger of what the observed disagreements show and what
    chance disagreements would (``E_chance[c²] / D_e``), so a judge that agrees exactly or by near misses
    is not credited with a spread its few observed disagreements cannot show, and one that reverses the
    scale is held to the spread it does show.

    **Pooled by result.** Each rater's kappa enters the figure at its result weight (each distinct result
    weighing 1, split across the raters measuring it — :func:`_pooled_kappa`). Results are independent; the
    raters of one result are not, so their contributions to it are added at full correlation (the
    Cauchy-Schwarz bound) — exact when raters' results do not overlap, conservative when they do. The
    multiplier is Student's t at :data:`~threetears.evals.analysis.stats.INTERVAL_LEVEL` on
    ``results - 1`` degrees of freedom.

    Seeded simulation (``tests/test_simulated_agreement.py``) holds the rule it serves: at the 20-result
    floor a judge at the bar earns the tier at most about 1% of the time.

    Args:
        estimate: The pooled figure the interval is around — it always lies inside.
        raters: Per rater whose kappa entered the figure: its disagreement moments under the figure's cost,
            and the result each of its pairs is about.

    Returns:
        ``(lower, upper)``, or None when fewer than two distinct results carry the figure.
    """
    defined = [(moments, ids) for moments, ids in raters if moments.expected > 0]
    measurers: dict[str, int] = {}
    for _, ids in defined:
        for result_id in ids:
            measurers[result_id] = measurers.get(result_id, 0) + 1
    if len(measurers) < 2:
        return None
    weights = [sum(1 / measurers[result_id] for result_id in ids) for _, ids in defined]
    total = sum(weights)
    # Per rater: its coefficient in the pooled figure per pair, its chance cost, and its disagreement size.
    shapes = []
    for (moments, _), weight in zip(defined, weights, strict=True):
        observed_size = moments.observed_square / moments.observed if moments.observed > 0 else 0.0
        size = max(moments.expected_square / moments.expected, observed_size)
        shapes.append((weight / total / (moments.n * moments.expected), moments.expected, size))
    on_result: dict[str, list[int]] = {}
    for index, (_, ids) in enumerate(defined):
        for result_id in ids:
            on_result.setdefault(result_id, []).append(index)
    # Results measured by the same raters contribute alike, so each such group is summed once and counted.
    memberships = Counter(tuple(sorted(members)) for members in on_result.values())
    critical = t_critical_two_sided(INTERVAL_LEVEL, len(measurers) - 1)

    def outside(candidate: float) -> bool:
        spreads = []
        for coefficient, chance, size in shapes:
            mean = (1 - candidate) * chance
            spreads.append(coefficient * math.sqrt(max(size * mean - mean * mean, 0.0)))
        variance = sum(count * sum(spreads[index] for index in members) ** 2 for members, count in memberships.items())
        return (estimate - candidate) ** 2 > critical * critical * variance

    def edge(limit: float) -> float:
        # Walk out from the estimate to the first value outside, then bisect: the innermost crossing.
        step = 0.05 if limit > estimate else -0.05
        inside = estimate
        while inside != limit:
            probe = min(inside + step, limit) if step > 0 else max(inside + step, limit)
            if outside(probe):
                for _ in range(40):
                    middle = (inside + probe) / 2
                    if outside(middle):
                        probe = middle
                    else:
                        inside = middle
                return inside
            inside = probe
        return limit

    return (edge(min(-1.0, estimate)), edge(max(1.0, estimate)))


def _pooled_kappa(per_rater: Sequence[tuple[float | None, Sequence[_Pair]]]) -> float | None:
    """The mean of the defined per-rater kappas, each rater weighted by the results it measured, or None.

    **Each distinct result carries weight 1**, split evenly across the defined raters that measured it, so a
    rater's weight is the sum over its pairs of ``1 / (defined raters measuring that pair's result)``. The
    figure then weighs exactly what the floor counts: twenty results move it as twenty, however many times a
    few of them were re-measured — whether by one more round of repeats over the same two results, or by five
    annotators rating the same three anchors. Undefined raters are excluded first, so a result only they
    measured carries no weight, matching ``results``.
    """
    defined = [(kappa, own) for kappa, own in per_rater if kappa is not None]
    measurers: dict[str, int] = {}
    for _, own in defined:
        for pair in own:
            measurers[pair.result_id] = measurers.get(pair.result_id, 0) + 1
    weighted = [(kappa, sum(1 / measurers[pair.result_id] for pair in own)) for kappa, own in defined]
    total = sum(weight for _, weight in weighted)
    return sum(kappa * weight for kappa, weight in weighted) / total if total else None


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
            "maximally far from every score — deliberately strict: one decline costs what a 1 against a 5 does, so "
            "a declining judge can be under-credited, never over-credited."
        ),
    )
    rounds: list[str] = Field(
        min_length=1,
        description=(
            "The repeat rounds among the pairs, sorted: `repeat 1` is each result's first repeat of the "
            "dimension, `repeat 2` its second. Each round is a rater, so the kappas pool per round, weighted by "
            "the results each round measured (each result weighing 1, split across the rounds that repeated it), as "
            "calibration pools per person."
        ),
    )
    exact_agreement: float = Field(
        ge=0.0,
        le=1.0,
        description='The share of pairs where the repeat gave the same score; a "can\'t tell" never does.',
    )
    kappa: float | None = Field(
        description="Cohen's kappa per round, pooled by result; None when every round's is undefined."
    )
    weighted_kappa: float | None = Field(
        description="Quadratic-weighted kappa per round, pooled as `kappa` is. None on pass/fail, and when undefined."
    )
    agreement_interval: tuple[float, float] | None = Field(
        default=None,
        description=(
            "The 95% interval on the figure the `separation` tier reads, as `DimensionAgreement.agreement_interval`."
        ),
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

    **Stated limit.** Only a dimension the result holds a SCORE on is repeated, so the reverse flip — "can't
    tell" first, a score on repeat — is never observed. (Rounds pool by result, so repeating a few results many
    times weighs those few results and no more.)

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
                agreement_interval=numbers.interval,
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
            people.agreement_interval if people else None,
        )
        separation = separation_criterion(
            itself.n if itself else 0,
            itself.results if itself else 0,
            agreement_statistic(key.scale, itself.kappa, itself.weighted_kappa) if itself else None,
            itself.agreement_interval if itself else None,
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
    """A criterion as a clause: its agreement, interval, pairs and results against the bar, or why there is nothing to read."""
    bar = f"(bar {format_number(criterion.threshold)} over at least {criterion.min_results} results)"
    if criterion.n == 0:
        return f"not measured {bar}"
    if criterion.agreement is None:
        return f"undefined over {criterion.n} pairs {bar}"
    over = f"over {criterion.n} pairs from {criterion.results} results"
    if criterion.interval is not None:
        low, high = criterion.interval
        over = f"(95% interval {format_number(low)} to {format_number(high)}) {over}"
    verdict = {
        "met": "meets",
        "not_met": "misses",
        "undecided": "undecided — the interval straddles",
        "insufficient": "too few results for" if criterion.results < criterion.min_results else "no interval for",
    }[criterion.state]
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
    "agreement_interval",
    "judge_agreement",
    "judge_evidence_tiers",
    "judge_key",
    "judge_self_agreement",
    "tier_for_judges",
    "tier_sentence",
]
