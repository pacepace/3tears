"""Judge-versus-human agreement: how often the judge scored a dimension the way people did.

A judge's consistency across repeats measures its precision; only people can say whether it is
right. People say so in :class:`~threetears.evals.schema.models.CalibrationRating` documents, one
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
floor counts (:mod:`threetears.evals.kernel.evidence_tiers`).

**One group per dimension, scale and judge** (:class:`JudgeKey`). The judge is the model that served the
score (:attr:`~threetears.evals.schema.models.RubricScore.served_model`), the versioned judge config
that asked for it (the result's ``judge_config_ids``; ``None`` = the built-in prompt) AND the temperature the
call was sent at (:attr:`~threetears.evals.schema.models.RubricScore.judge_temperature`), so a campaign
sweeping its judge — model, prompt or sampling — reads each judge's agreement separately: pooling them would
credit one judge with the other's calibration, which is the comparison a judge swap is decided on. A dimension rated on a
scale it was later moved off is two groups for the same reason.

**A label is found by its result and by what was read** (#628). A rating carries the label key of the score it
rated (:class:`~threetears.evals.schema.models.LabelKey`: the fingerprint of the judged output and of the
criterion), and each judge score carries its own. So a rating pairs with its own result's score, AND with every
other result's score whose key is the same — a byte-identical output judged on the same criterion, in another
run, under another judge, or on a result whose own record has since changed. **A label enters each judge's
agreement once.** Within one :class:`JudgeKey` group a rating pairs at most once: with its own result when that
result is in the group, otherwise with the first matching result in the order the results were handed in. Found by
both routes (its own result also matches its key), or on several identical outputs scored by one judge, it is
still one person's one answer, and counting it per copy would inflate ``n`` and the distinct results the
``calibrated`` floor counts with no further human judgement. A rating whose key reached a judged result is paired
even when its own result is gone; one with no key (rated before scores were stamped) is read by its result alone.

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
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Final, Literal, NamedTuple

from pydantic import Field

from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.stats import (
    KappaMoments,
    cohen_kappa,
    kappa_moments,
    t_critical_two_sided,
)
from threetears.evals.schema.base import EvalDocumentModel
from threetears.evals.kernel.evidence_tiers import (
    JudgedEvidenceTier,
    JudgeEvidenceTier,
    TierCriterion,
    agreement_statistic,
    calibration_criterion,
    separation_criterion,
    tier_of,
    weakest_judged_tier,
)
from threetears.evals.schema.models import MODEL_DEFAULT_TEMPERATURE, SCALES, JudgeTemperature, RubricScale

if TYPE_CHECKING:
    from threetears.evals.schema.models import CalibrationRating, EvalResult, JudgeRepeat, LabelKey, RubricScore


#: Why a rating has no judge score to be read against.
UnpairedReason = Literal["result_unresolved", "dimension_unscored", "scale_changed", "rated_by_an_agent"]


class JudgeKey(NamedTuple):
    """Who judged a reading: the dimension, its scale, the model that served the score, the config that asked and
    the temperature the call was sent at.

    The one key both agreements group by, the tiers are listed by, and a reading looks its tier up by — so a
    measurement of one judge can never stand in for another's. A temperature nobody recorded is ``None``, its own
    group: never a match for a recorded one.
    """

    rubric_dim: str
    scale: RubricScale
    judge_model: str | None
    judge_config_id: str | None
    judge_temperature: JudgeTemperature | None = None


def judge_key(result: EvalResult, dim: str) -> JudgeKey | None:
    """The judge behind ``result``'s score on ``dim``, or None when it holds no score there.

    Args:
        result: The result.
        dim: The dimension, as its score spells it.

    Returns:
        The key: the score's scale, served model and temperature, and the config the result records as having
        scored the dim.
    """
    score = result.judge_score(dim)
    if score is None:
        return None
    return JudgeKey(dim, score.scale, score.served_model, result.judge_config_ids.get(dim), score.judge_temperature)


def _sort_key(key: JudgeKey) -> tuple[str, str, str, str, str]:
    """Order keys by dimension, scale, judge, config and temperature, an unnamed judge, the built-in prompt or an
    unrecorded temperature first."""
    temperature = key.judge_temperature
    return (
        key.rubric_dim,
        key.scale,
        key.judge_model or "",
        key.judge_config_id or "",
        "" if temperature is None else str(temperature),
    )


def _key_of(entry: DimensionAgreement | SelfAgreementDimension | JudgeEvidenceTier) -> JudgeKey:
    """The judge an agreement row or a tier was measured for, read back off its fields."""
    return JudgeKey(entry.rubric_dim, entry.scale, entry.judge_model, entry.judge_config_id, entry.judge_temperature)


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
    judge_temperature: JudgeTemperature | None = Field(
        default=None,
        description=(
            "The temperature the judge's calls were sent at ('model_default' = sent none, the model refusing one); "
            "None = not recorded, a judge nobody observed the sampling of, never read as a match for a recorded one."
        ),
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
            "The confidence bounds the `calibrated` tier is decided on — a one-sided 95% lower and a one-sided 97.5% "
            "upper bound (`agreement_interval`) — on the figure it reads (`weighted_kappa` on 1-5, `kappa` on "
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


class _PairedRatings(NamedTuple):
    """Every rating read, as :func:`_pair_ratings` sorted it: the pairs per judge, the unpaired, and the count."""

    groups: dict[JudgeKey, list[_Pair]]
    unpaired: list[UnpairedRating]
    read: int


def _pair_ratings(ratings: Iterable[CalibrationRating], results: Iterable[EvalResult]) -> _PairedRatings:
    """Pair each rating with the judge's score on its dimension, on its result or the same output — the ONE matching rule.

    A rating pairs with its own result's score, then with every other result's score carrying its label key (#628),
    entering each judge (:class:`JudgeKey`) once; it is unpaired only when it entered none, with its own result's
    reason. See the module docstring.

    Read by :func:`judge_agreement` and :func:`person_scores_by_result`, so the ratings agreement reads and the
    ratings a prediction-powered estimate combines with the judge's scores are one set (#598).

    Args:
        ratings: The ratings to read.
        results: The results they may rate. A rating whose result is not here, and whose label key reaches no
            score here, is unpaired (``result_unresolved``).

    Returns:
        The pairs grouped by judge, the ratings that could not be paired (and why), and how many were read.
    """
    results = list(results)
    by_id = {result.id: result for result in results}
    by_label = _scores_by_label_key(results)
    groups: dict[JudgeKey, list[_Pair]] = {}
    unpaired: list[UnpairedRating] = []
    read = 0
    for rating in ratings:
        read += 1
        # Before anything is paired: an agent's rating is not a person's, whatever it would pair with.
        if rating.rater_kind != "person":
            unpaired.append(_unpaired(rating, "rated_by_an_agent"))
            continue
        # The judges this label has entered: one entry per judge, whichever route found it.
        entered: set[JudgeKey] = set()
        reason: UnpairedReason | None = None
        result = by_id.get(rating.result_id)
        score = None if result is None else result.judge_score(rating.rubric_dim)
        key = None if result is None else judge_key(result, rating.rubric_dim)
        if result is None:
            reason = "result_unresolved"
        elif score is None or key is None:
            reason = "dimension_unscored"
        elif score.scale != rating.scale:
            reason = "scale_changed"
        else:
            groups.setdefault(key, []).append(_Pair(score.score, rating.score, rating.rater, result.id))
            entered.add(key)
        if rating.label_key is not None:
            for other, other_score in by_label.get(rating.label_key, ()):
                other_key = judge_key(other, other_score.dim)
                # Its own result was read above; a judge it already entered is not entered twice.
                if other.id == rating.result_id or other_key is None or other_key in entered:
                    continue
                if other_score.scale != rating.scale:
                    continue
                groups.setdefault(other_key, []).append(_Pair(other_score.score, rating.score, rating.rater, other.id))
                entered.add(other_key)
        if not entered:
            assert reason is not None  # its own result paired, or said why not
            unpaired.append(_unpaired(rating, reason))
    return _PairedRatings(groups, unpaired, read)


def judge_agreement(ratings: Iterable[CalibrationRating], results: Iterable[EvalResult]) -> JudgeAgreement:
    """Pair each rating with the judge's score on its dimension, on its result or the same output, and read agreement.

    Args:
        ratings: The ratings to read.
        results: The results they may rate. A rating whose result is not here, and whose label key reaches no
            score here, is unpaired (``result_unresolved``), so hand in every result the ratings were read for.

    Returns:
        The agreement per (dimension, scale, judge, judge config), and the ratings that could not be paired.
    """
    groups, unpaired, read = _pair_ratings(ratings, results)
    dimensions = []
    for key in sorted(groups, key=_sort_key):
        numbers = _agreement_numbers(key.scale, groups[key])
        dimensions.append(
            DimensionAgreement(
                rubric_dim=key.rubric_dim,
                scale=key.scale,
                judge_model=key.judge_model,
                judge_config_id=key.judge_config_id,
                judge_temperature=key.judge_temperature,
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


def person_scores_by_result(
    ratings: Iterable[CalibrationRating], results: Iterable[EvalResult]
) -> dict[tuple[str, str], list[int]]:
    """Every person's score of a judged dimension of a result, keyed ``(result_id, dimension)`` — the human labels.

    Exactly the pairs :func:`judge_agreement` reads (a person's rating, of a result read, on a dimension its judge
    scores on the rating's scale), so an agent's rating, a rating of a result not read and a rating on a changed
    scale label nothing here either. What a prediction-powered estimate combines with the judge's scores (#598).

    A label found by what was read (#628) labels the result it reached: a rating of one result whose label key
    another result's score carries is that other result's person score too. Because the pairing enters each judge
    once, a label labels at most one result per judge — so an arm, whose results share a judge, never counts one
    person's answer on several byte-identical outputs as several labels.

    Args:
        ratings: The ratings to read.
        results: The results they may rate.

    Returns:
        Every person's score per rated ``(result_id, dimension)``, in the order the ratings were read; a result
        two people rated carries both.
    """
    scores: dict[tuple[str, str], list[int]] = {}
    for key, pairs in _pair_ratings(ratings, results).groups.items():
        for pair in pairs:
            if pair.other is not None:
                scores.setdefault((pair.result_id, key.rubric_dim), []).append(pair.other)
    return scores


def _scores_by_label_key(results: Sequence[EvalResult]) -> dict[LabelKey, list[tuple[EvalResult, RubricScore]]]:
    """Every stamped judge score, by the output and criterion it was given on, in result order.

    Args:
        results: The results read.

    Returns:
        ``{label_key: [(result, score), ...]}`` over every judge score — rubric dimensions and the two dual-score
        axes — that carries a key. A score judged before the stamp is absent.
    """
    found: dict[LabelKey, list[tuple[EvalResult, RubricScore]]] = {}
    for result in results:
        for score in (*result.rubric_scores, result.transcript_score, result.outcome_score):
            if score is not None and (key := score.label_key) is not None:
                found.setdefault(key, []).append((result, score))
    return found


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
    result — each distinct result weighing 1, split across the raters that measured it (see the module docstring and :mod:`threetears.evals.kernel.evidence_tiers`); on a
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


#: The one-sided confidence of the bound a tier is AWARDED on: a criterion is met only when its lower bound at
#: this confidence reaches the bar, so a judge exactly at the bar is awarded the tier at most 5% of the time.
#: Measured at most 3.6% at 20-40 results (``tests/test_simulated_agreement.py``).
TIER_LOWER_CONFIDENCE: Final = 0.95

#: The one-sided confidence of the bound a criterion is MISSED on: not met only when its upper bound at this
#: confidence is below the bar. Stricter than the lower bound's, because the score interval's upper side runs
#: looser than nominal on few results: at 95% it showed a judge at the bar below it up to 8% of the time; at
#: 97.5% at most about 5%.
TIER_UPPER_CONFIDENCE: Final = 0.975


def agreement_interval(
    estimate: float, raters: Sequence[tuple[KappaMoments, Sequence[str]]]
) -> tuple[float, float] | None:
    """The bounds a tier is decided on: a score interval on a pooled agreement figure, over its distinct results.

    ``(lower, upper)``: the lower end a one-sided :data:`TIER_LOWER_CONFIDENCE` bound, the upper end a one-sided
    :data:`TIER_UPPER_CONFIDENCE` bound. A tier is a one-sided claim (the judge is at least this good), so it is
    awarded on a one-sided bound.

    **Why a score interval.** Compared by seeded simulation over six marginals at 20-40 results
    (``docs/reading-reports.md`` carries the table): the large-sample analytic standard error of weighted kappa
    (Fleiss, Cohen and Everitt) read on t, and a bootstrap over results (percentile and BCa), awarded the tier
    to a judge at the bar 5-34% of the time and covered the truth as little as half the time, because their
    spread is read off the estimate and shrinks to nothing when a few results happen to agree. A score
    interval holds each candidate value ``κ0`` to the spread kappa WOULD have there — the set of ``κ0`` the
    estimate is within ``t`` of — the Wilson interval's construction, which it reduces to on pass/fail.

    **The spread at ``κ0``.** Kappa is ``1 - D / D_e``: ``D`` the mean disagreement cost over the items, ``D_e``
    the cost chance gives the two raters' marginals. At ``κ0`` the mean cost is ``(1 - κ0) D_e``, and a cost
    ``c`` in ``[0, 1]`` with mean ``m`` has variance ``E[c²] - m²`` with ``E[c²] = ρ m``, where ``ρ`` is how
    large a disagreement is when there is one. On pass/fail every disagreement costs 1 (``ρ = 1``, Wilson
    exactly, with no model). On 1-5 ``ρ`` is the larger of what the observed disagreements show and what
    chance disagreements would (``E_chance[c²] / D_e``), so a judge that agrees exactly or by near misses
    is not credited with a spread its few observed disagreements cannot show, and one that reverses the
    scale is held to the spread it does show. Reading ``ρ`` off the observed disagreements alone awarded the
    tier at the bar 8-18% of the time.

    **Pooled by result, the result the cluster.** Each rater's kappa enters the figure at its result weight
    (each distinct result weighing 1, split across the raters measuring it — :func:`_pooled_kappa`). Results
    are independent; the raters of one result are not — two people who both see the judge misjudge a result
    both disagree with it. Their contributions to a shared result are added at the correlation their centred
    costs show across the results they share, pooled over every pair of raters and held to ``[0, 1]``. Seeded
    simulation with two raters who copy the same truth (correlation 1) held the tier's size at 4.6%, where
    adding them as independent reached 9.5%. With no result shared the question does not arise. The
    multiplier is Student's t on ``results - 1`` degrees of freedom.

    Args:
        estimate: The pooled figure the interval is around — it always lies inside.
        raters: Per rater whose kappa entered the figure: its disagreement moments under the figure's cost
            (with each item's cost), and the result each of its pairs is about.

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
    correlation = _shared_result_correlation(defined, on_result)
    # Results measured by the same raters contribute alike, so each such group is summed once and counted.
    memberships = Counter(tuple(sorted(members)) for members in on_result.values())
    results = len(measurers)

    def outside(candidate: float, critical: float) -> bool:
        spreads = []
        for coefficient, chance, size in shapes:
            mean = (1 - candidate) * chance
            spreads.append(coefficient * math.sqrt(max(size * mean - mean * mean, 0.0)))
        variance = 0.0
        for members, count in memberships.items():
            alone = sum(spreads[index] ** 2 for index in members)
            together = sum(spreads[index] for index in members) ** 2
            variance += count * (alone + correlation * (together - alone))
        return (estimate - candidate) ** 2 > critical * critical * variance

    def edge(limit: float, confidence: float) -> float:
        # Walk out from the estimate to the first value outside, then bisect: the innermost crossing.
        critical = t_critical_two_sided(2 * confidence - 1, results - 1)
        step = 0.05 if limit > estimate else -0.05
        inside = estimate
        while inside != limit:
            probe = min(inside + step, limit) if step > 0 else max(inside + step, limit)
            if outside(probe, critical):
                for _ in range(40):
                    middle = (inside + probe) / 2
                    if outside(middle, critical):
                        probe = middle
                    else:
                        inside = middle
                return inside
            inside = probe
        return limit

    return (
        edge(min(-1.0, estimate), TIER_LOWER_CONFIDENCE),
        edge(max(1.0, estimate), TIER_UPPER_CONFIDENCE),
    )


def _shared_result_correlation(
    defined: Sequence[tuple[KappaMoments, Sequence[str]]], on_result: Mapping[str, Sequence[int]]
) -> float:
    """How alike two raters' disagreements with the judge are on a result both measured: pooled, held to [0, 1].

    Each rater's per-item costs centred on its own mean, cross-multiplied over every pair of raters sharing a
    result and normalised — one correlation for the figure. 1 (the bound) when no result is shared, where it
    multiplies nothing.
    """
    centred: dict[tuple[int, str], float] = {}
    for index, (moments, ids) in enumerate(defined):
        for result_id, cost in zip(ids, moments.costs, strict=True):
            centred[(index, result_id)] = cost - moments.observed
    cross = left = right = 0.0
    for result_id, members in on_result.items():
        for position, first in enumerate(members):
            for second in members[position + 1 :]:
                u, v = centred[(first, result_id)], centred[(second, result_id)]
                cross += u * v
                left += u * u
                right += v * v
    if left == 0 or right == 0:
        return 1.0
    return min(1.0, max(0.0, cross / math.sqrt(left * right)))


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
#: score — a different prompt is a different judge for the same reason. ``temperature_changed``: the
#: repeat was sent at a different temperature than the first score, or only one of the two recorded one — a
#: different (or unknown) sampling is a different judge too; two that both recorded none pair under an
#: unrecorded temperature, as two unnamed models pair under an unnamed judge. (A repeat answering "can't tell" IS paired:
#: declining to score what it once scored is the judge disagreeing with itself.)
UnrepeatedReason = Literal["repeat_failed", "judge_changed", "config_changed", "temperature_changed"]


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
    judge_temperature: JudgeTemperature | None = Field(
        default=None,
        description=(
            "The temperature the judge's calls were sent at ('model_default' = sent none, the model refusing one); "
            "None = not recorded, a judge nobody observed the sampling of, never read as a match for a recorded one."
        ),
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
            "The confidence bounds on the figure the `separation` tier reads, as `DimensionAgreement.agreement_interval`."
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

    The pair is the one the repeat recorded (:class:`~threetears.evals.schema.models.RepeatedScore`),
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
    return self_agreement_of_repeats((result.id, result.judge_repeats) for result in results)


def self_agreement_of_repeats(repeats: Iterable[tuple[str, Sequence[JudgeRepeat]]]) -> JudgeSelfAgreement:
    """:func:`judge_self_agreement` over repeats held apart from any stored result, each beside the result it repeats.

    The one reading, for repeats that were never written to a result — a judge temperature comparison's answers at
    one setting, paired against that setting's first answer (#633) — so they are read by the same code.

    Args:
        repeats: ``(result id, its repeats, oldest first)`` per result.

    Returns:
        As :func:`judge_self_agreement`.
    """
    groups: dict[JudgeKey, list[_Pair]] = {}
    unpaired: list[UnrepeatedScore] = []
    read = 0
    for result_id, judge_repeats in repeats:
        rounds: dict[str, int] = {}
        for repeat in judge_repeats:
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
                elif entry.repeat is not None and entry.repeat.judge_temperature != entry.first_judge_temperature:
                    reason = "temperature_changed"
                if reason is not None:
                    unpaired.append(
                        UnrepeatedScore(result_id=result_id, rubric_dim=entry.dim, round=round_name, reason=reason)
                    )
                    continue
                key = JudgeKey(
                    entry.dim,
                    entry.scale,
                    entry.first_served_model,
                    entry.first_judge_config_id,
                    entry.first_judge_temperature,
                )
                again = entry.repeat.score if entry.repeat is not None else None
                groups.setdefault(key, []).append(_Pair(entry.first_score, again, round_name, result_id))
    dimensions = []
    for key in sorted(groups, key=_sort_key):
        numbers = _agreement_numbers(key.scale, groups[key])
        dimensions.append(
            SelfAgreementDimension(
                rubric_dim=key.rubric_dim,
                scale=key.scale,
                judge_model=key.judge_model,
                judge_config_id=key.judge_config_id,
                judge_temperature=key.judge_temperature,
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
# The judge against a second judge
# ---------------------------------------------------------------------------

#: Why a second judge's score has no pair to be read in: its call failed — an infrastructure fault, which says
#: nothing about either judge. (A second judge answering "can't tell" IS paired, as a disagreement.)
UnpairedSecondReason = Literal["second_failed"]

#: What an undefined kappa means, stated wherever one is: never a zero, never perfect agreement.
KAPPA_UNDEFINED_ONE_SCORE = (
    "undefined: every pair carries one and the same score on both sides, so chance alone predicts no disagreement "
    "and kappa's denominator is zero — this is not perfect agreement, and it is not zero"
)


class InterJudgeDimension(EvalDocumentModel):
    """How a second judge's scores on one dimension agreed with the run's judge's scores of the same evidence."""

    rubric_dim: str = Field(min_length=1, description="The judged dimension.")
    scale: RubricScale = Field(description="The scale both judges scored it on.")
    judge_model: str | None = Field(
        description="The model that served the first scores, as the provider named it; None when it named none."
    )
    judge_config_id: str | None = Field(
        description="The versioned JudgeConfig that asked for the first scores; None = the built-in prompt."
    )
    judge_temperature: JudgeTemperature | None = Field(
        default=None, description="The temperature the first scores were sent at; None = not recorded."
    )
    second_model: str = Field(min_length=1, description="The model the second judge was asked as.")
    second_judge_config_id: str | None = Field(
        description="The versioned JudgeConfig that asked the second judge; None = the built-in prompt."
    )
    second_temperature: float | None = Field(
        description="The temperature the second judge was requested at; None = what each dimension's prompt asks for."
    )
    second_served_models: list[str] = Field(
        default_factory=list,
        description="The models the second judge's responses named as having answered, sorted; empty when none named one.",
    )
    n: int = Field(ge=1, description='Pairs read: one per second score, a "can\'t tell" answer included.')
    results: int = Field(
        ge=0, description="The distinct results among the passes whose kappa entered `kappa`. 0 when it is undefined."
    )
    n_cannot_tell: int = Field(
        ge=0,
        description=(
            "Pairs where the second judge answered it could not tell on a dimension the first scored. Counted in `n` "
            "and as disagreements, and read in `kappa` as a category of its own, maximally far from every score."
        ),
    )
    passes: list[str] = Field(
        min_length=1,
        description=(
            "The second-judge passes among the pairs, by id, sorted. Each pass is a rater: the kappas pool per pass, "
            "weighted by the results each measured, as calibration pools per person."
        ),
    )
    exact_agreement: float = Field(
        ge=0.0, le=1.0, description="The share of pairs where the two judges gave the same score."
    )
    kappa_weighting: Literal["quadratic", "none"] = Field(
        description="How `kappa` weighs a disagreement: quadratic on a 1-5 dimension, unweighted on pass/fail."
    )
    kappa: float | None = Field(
        description=(
            "Cohen's kappa between the two judges — quadratic-weighted on 1-5, unweighted on pass/fail — per pass, "
            "pooled by result. None when it is undefined, and `kappa_undefined` says why: never read as 0."
        ),
    )
    kappa_undefined: str | None = Field(
        default=None, description="Why `kappa` is undefined, when it is; None when it is defined."
    )
    agreement_interval: tuple[float, float] | None = Field(
        default=None,
        description=(
            "Confidence bounds on `kappa` over its distinct results — a one-sided 95% lower and 97.5% upper bound, "
            "the bounds the evidence tiers are decided on (`agreement_interval`). None when `kappa` is undefined or "
            "rests on fewer than two results. The point figure is never the finding; these bounds are."
        ),
    )


class UnpairedSecondScore(EvalDocumentModel):
    """A second judge's score with no pair to read, and why."""

    result_id: str = Field(min_length=1, description="The result asked about.")
    rubric_dim: str = Field(min_length=1, description="The dimension.")
    pass_id: str = Field(min_length=1, description="The pass that asked.")
    reason: UnpairedSecondReason


class InterJudgeAgreement(EvalDocumentModel):
    """Every second-judge score read, paired with the run's judge's where it can be, and agreement per dimension."""

    scores_read: int = Field(default=0, ge=0, description="Every second-judge score read: the pairs plus the unpaired.")
    dimensions: list[InterJudgeDimension] = Field(
        default_factory=list,
        description=(
            "One per (dimension, scale, first judge, second judge) with at least one pair, ordered by those. Empty "
            "when no second judge was asked: agreement between judges is then unmeasured, which is a state, not a zero."
        ),
    )
    unpaired: list[UnpairedSecondScore] = Field(
        default_factory=list, description="Second-judge scores that could not be paired, in the order they were read."
    )


def inter_judge_agreement(results: Iterable[EvalResult], *, pass_id: str | None = None) -> InterJudgeAgreement:
    """Pair each second judge's score with the first score it answers, and read agreement the way calibration does.

    The pair is the one the pass recorded (:class:`~threetears.evals.schema.models.SecondJudgeScore`), so a
    re-judge that later rewrote the result's score does not split it. The second judge stands where the person stands
    in :func:`judge_agreement`, each pass a rater, so the figures are the same statistic over the same pooling
    (:func:`_agreement_numbers`). The figure is quadratic-weighted kappa on 1-5 and unweighted kappa on pass/fail; an
    undefined one is stated as undefined, with why.

    Args:
        results: The results whose second-judge scores to read.
        pass_id: Read only this pass's pairs; ``None`` for every pass.

    Returns:
        The agreement per (dimension, scale, first judge, second judge), and the scores that could not be paired.
    """
    groups: dict[tuple[JudgeKey, str, str | None, float | None], list[_Pair]] = {}
    served: dict[tuple[JudgeKey, str, str | None, float | None], set[str]] = {}
    unpaired: list[UnpairedSecondScore] = []
    read = 0
    for result in results:
        for judging in result.judge_seconds:
            if pass_id is not None and judging.pass_id != pass_id:
                continue
            for entry in judging.scores:
                read += 1
                if entry.second is None and entry.cannot_tell is None:
                    unpaired.append(
                        UnpairedSecondScore(
                            result_id=result.id, rubric_dim=entry.dim, pass_id=judging.pass_id, reason="second_failed"
                        )
                    )
                    continue
                first = JudgeKey(
                    entry.dim,
                    entry.scale,
                    entry.first_served_model,
                    entry.first_judge_config_id,
                    entry.first_judge_temperature,
                )
                key = (first, judging.judge.model, judging.judge_config_ids.get(entry.dim), judging.judge.temperature)
                second = entry.second.score if entry.second is not None else None
                groups.setdefault(key, []).append(_Pair(entry.first_score, second, judging.pass_id, result.id))
                if entry.second is not None and entry.second.served_model is not None:
                    served.setdefault(key, set()).add(entry.second.served_model)
    dimensions = []
    for key in sorted(groups, key=lambda k: (*_sort_key(k[0]), k[1], k[2] or "", "" if k[3] is None else str(k[3]))):
        first, model, config_id, temperature = key
        numbers = _agreement_numbers(first.scale, groups[key])
        figure = numbers.weighted_kappa if first.scale == "ordinal" else numbers.kappa
        dimensions.append(
            InterJudgeDimension(
                rubric_dim=first.rubric_dim,
                scale=first.scale,
                judge_model=first.judge_model,
                judge_config_id=first.judge_config_id,
                judge_temperature=first.judge_temperature,
                second_model=model,
                second_judge_config_id=config_id,
                second_temperature=temperature,
                second_served_models=sorted(served.get(key, set())),
                n=numbers.n,
                results=numbers.results,
                n_cannot_tell=numbers.cannot_tell,
                passes=numbers.raters,
                exact_agreement=numbers.exact_agreement,
                kappa_weighting="quadratic" if first.scale == "ordinal" else "none",
                kappa=figure,
                kappa_undefined=KAPPA_UNDEFINED_ONE_SCORE if figure is None else None,
                agreement_interval=numbers.interval,
            )
        )
    return InterJudgeAgreement(scores_read=read, dimensions=dimensions, unpaired=unpaired)


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
    calibrations = {_key_of(d): d for d in agreement.dimensions}
    repeats = {_key_of(d): d for d in self_agreement.dimensions}
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
                judge_temperature=key.judge_temperature,
                tier=tier_of(calibration, separation),
                calibration=calibration,
                separation=separation,
            )
        )
    return tiers


def tier_for_judges(tiers: Iterable[JudgeEvidenceTier], judges: Iterable[JudgeKey]) -> JudgedEvidenceTier:
    """The tier a reading stands on when ``judges`` served its scores: the weakest of theirs.

    Looked up by the whole :class:`JudgeKey` — dimension, scale, served model, config and temperature — so a reading
    can only ever carry the tier measured for the very judge behind it. A cell whose scores were served by
    two judges pools two judges' readings, and the composite can bear only what the weaker can. A judge
    with no entry is ``undetermined``: nothing measured it.

    Args:
        tiers: The tiers, from :func:`judge_evidence_tiers`.
        judges: The judges behind the reading's scores (:func:`judge_key`).

    Returns:
        The tier; ``undetermined`` when no judge is named — a reading with no score behind it has no judge.
    """
    by_judge = {_key_of(tier): tier.tier for tier in tiers}
    found: list[JudgedEvidenceTier] = [by_judge.get(JudgeKey(*key), "undetermined") for key in set(judges)]
    return weakest_judged_tier(found) if found else "undetermined"


def tier_sentence(tier: JudgeEvidenceTier) -> str:
    """One sentence a report states for a judge's tier on a dimension: the tier, and the two measurements behind it.

    Args:
        tier: The tier as decided.

    Returns:
        The sentence, naming the judge (its config, when one asked, and its temperature), the tier and each criterion's
        agreement, pairs and results against its bar — and, for a tier read from a stored judge profile, the profile,
        when and on which cases it was measured, and what the campaign's own evidence read.
    """
    judge = tier.judge_model or "an unnamed judge"
    if tier.judge_config_id is not None:
        judge = f"{judge}, config {tier.judge_config_id}"
    temperature = tier.judge_temperature
    judge += (
        ", temperature not recorded"
        if temperature is None
        else ", sent no temperature"
        if temperature == MODEL_DEFAULT_TEMPERATURE
        else f", temperature {format_number(temperature)}"
    )
    measured = (
        f"{tier.rubric_dim} ({judge}): {tier.tier} — agreement with people "
        f"{_criterion_words(tier.calibration)}; with its own repeats {_criterion_words(tier.separation)}."
    )
    profile = tier.from_profile
    if profile is None:
        return measured
    # Never silently: a tier read from a stored profile says so, with when and on what it was measured, and what
    # the campaign's own evidence read (#628).
    return (
        f"{tier.rubric_dim} ({judge}): {tier.tier}, read from the judge's stored profile {profile.profile_id}, "
        f"measured at {profile.measured_at} on {format_number(profile.cases)} frozen cases (case set "
        f"{profile.case_set_fingerprint[:12]}, runs {', '.join(profile.run_ids)}) — agreement with the labels "
        f"{_criterion_words(tier.calibration)}; with its own repeats {_criterion_words(tier.separation)}. This "
        f"campaign's own evidence decided no tier: agreement with people {_criterion_words(profile.own_calibration)}; "
        f"with its own repeats {_criterion_words(profile.own_separation)}."
    )


def _criterion_words(criterion: TierCriterion) -> str:
    """A criterion as a clause: its agreement, bounds, pairs and results against the bar, and what it still needs."""
    bar = f"(bar {format_number(criterion.threshold)} over at least {criterion.min_results} results)"
    needed = criterion.results_needed
    if criterion.n == 0:
        return f"not measured {bar} — needs {criterion.min_results} results"
    if criterion.agreement is None:
        # Undefined kappa below the floor is still short of the floor: the results it needs are stated as they are
        # for a defined one, so the sentence never reads as though only the kappa were missing.
        undefined = f"undefined over {criterion.n} pairs from {criterion.results} results {bar}"
        return f"{undefined} — needs {needed} more results" if needed is not None else undefined
    over = f"over {criterion.n} pairs from {criterion.results} results"
    if criterion.interval is not None:
        low, high = criterion.interval
        over = f"(bounds {format_number(low)} to {format_number(high)}) {over}"
    verdict = {
        "met": "meets",
        "not_met": "misses",
        "undecided": "undecided — the bounds straddle",
        "insufficient": "too few results for" if criterion.results < criterion.min_results else "no bounds for",
    }[criterion.state]
    clause = f"{format_number(criterion.agreement)} {over}, {verdict} {bar}"
    if needed is not None and criterion.state == "insufficient":
        clause += f" — needs {needed} more results"
    elif needed is not None:
        clause += f" — about {needed} more results would decide it if agreement holds"
    return clause


__all__ = [
    "DimensionAgreement",
    "InterJudgeAgreement",
    "InterJudgeDimension",
    "JudgeAgreement",
    "JudgeKey",
    "JudgeSelfAgreement",
    "SelfAgreementDimension",
    "KAPPA_UNDEFINED_ONE_SCORE",
    "UnpairedRating",
    "UnpairedReason",
    "UnpairedSecondReason",
    "UnpairedSecondScore",
    "UnrepeatedReason",
    "UnrepeatedScore",
    "TIER_LOWER_CONFIDENCE",
    "TIER_UPPER_CONFIDENCE",
    "agreement_interval",
    "inter_judge_agreement",
    "judge_agreement",
    "judge_evidence_tiers",
    "judge_key",
    "judge_self_agreement",
    "person_scores_by_result",
    "self_agreement_of_repeats",
    "tier_for_judges",
    "tier_sentence",
]
