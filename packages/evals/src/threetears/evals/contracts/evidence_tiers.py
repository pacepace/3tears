"""The evidence tier a judged reading stands on, decided by code from what the judge's reliability was measured to be.

A judged score is one model's opinion of another's output. How far a reader may lean on it depends on
two measurements of the judge, and on nothing a report writer says (owner ruling, 2026-10-06, PD-13):

- **calibrated** — the judge agrees with PEOPLE: on the dimension, its agreement with people's
  calibration ratings of the same results (:func:`threetears.evals.analysis.judge_agreement`, which
  pairs only a person's rating, never an agent's) is at least :data:`CALIBRATION_MIN_AGREEMENT` over at
  least :data:`CALIBRATION_MIN_RESULTS` distinct results. The strongest judged tier: the judge is right
  as people judge rightness.
- **separation** — the judge agrees with ITSELF: re-scoring evidence it already scored, under the
  apparatus the run recorded (:func:`threetears.evals.analysis.judge_self_agreement`), its agreement with
  its own first scores is at least :data:`SEPARATION_MIN_AGREEMENT` over at least
  :data:`SEPARATION_MIN_RESULTS` distinct results. Its scores are repeatable at the item level — a
  statement about precision, never accuracy, and never about any particular gap: whether a gap between
  two arms clears the judge's noise is still that comparison's own test (``multiple_comparisons``).
- **incidental** — a judged reading meeting neither: both measurements were taken over enough results
  and both fell short. The judge was measured and found neither calibrated nor consistent.

**Agreement is one statistic for both tiers, computed by one rule**, so the two thresholds are on one
scale and the tiers compare (:func:`threetears.evals.analysis.agreement` holds the one computation):

- *The figure*: Cohen's kappa with quadratic weights on a 1-5 dimension, Cohen's kappa on pass/fail,
  where the two weightings are the same number (:func:`agreement_statistic`).
- *Per rater, then pooled by result*: each rater's kappa (each person, for calibration; each round of
  repeats, for self-agreement) is computed over the pairs that rater gave, and the dimension's figure is
  the mean of the defined ones **weighted by the results each rater measured**: every distinct result
  carries weight 1, split evenly across the raters that measured it. The figure therefore weighs what
  the floor counts. Neither one small rater nor many small raters re-measuring a few shared results can
  carry it: 20 ratings at 0.44 beside five annotators matching the judge on the same 3 anchors read 0.51,
  where an unweighted mean (0.91) or a pair-weighted one (0.68) published ``calibrated`` on three results;
  and 20 results repeated once at 0.44 beside 2 of them repeated 30 more times read about 0.5, where a
  pair-weighted mean read 0.86 and published ``separation``. The kappa stays per rater rather than pooled
  into one table, because one table would enter a result two people rated twice and read their
  disagreement with each other as the judge's.
- *The floor counts distinct results*, never pairs: :data:`CALIBRATION_MIN_RESULTS` and
  :data:`SEPARATION_MIN_RESULTS` are met by that many different results among the raters whose kappa
  entered the figure. Pairs can be multiplied without new evidence — repeating two results ten times
  is twenty pairs about two results — and results cannot; with the figure weighted by result too, the
  floor and the figure count the same thing.
- *A judge that declines on repeat is disagreeing with itself*: a repeat answering "can't tell" on a
  dimension the judge had scored is a pair whose second half is its own category, at the greatest
  distance from every score in both kappas, and it counts in ``n``, ``results`` and exact agreement like
  any other pair (calibration never holds one: a person's rating is a score). Only a repeat that failed
  for infrastructure, or that another judge or judge config answered, is left out — and named.

**A criterion is decided on confidence bounds for the agreement, never on its point estimate.** A tier is a
claim about the judge, and at 20 results kappa's sampling spread is about 0.2: deciding on the point estimate
awarded ``calibrated`` to a judge whose true weighted kappa is 0.5 a third of the time. So a criterion is
``met`` only when the agreement's one-sided 95% lower bound reaches its threshold, ``not_met`` only when its
one-sided 97.5% upper bound is below it, and ``undecided`` when neither — neither a pass nor a miss
(:func:`criterion_state`). The bounds are a score interval
(:func:`threetears.evals.analysis.agreement.agreement_interval`), chosen over an analytic standard error and a
bootstrap, which seeded simulation showed award the tier at the bar far beyond 5% at these sample sizes. A
criterion that is not yet decided says how many more results it needs (:attr:`TierCriterion.results_needed`),
so a reader plans the ratings instead of reading a bare "undetermined".

**Too little evidence is a state, never a tier.** A measurement over fewer results than its floor, or
whose kappa is undefined, is ``insufficient``, and a reading whose evidence does not decide its tier is
``undetermined`` — never quietly filed as incidental. The rule (:func:`tier_of`): calibrated when the
calibration criterion is met; else separation when the separation criterion is met; else incidental
when BOTH criteria were measured over enough results and shown below their bars; else undetermined. A
met criterion establishes its tier whatever the other one reads, because a tier is a floor the evidence
has reached; ``incidental`` is itself a finding about the judge, so it needs both measurements to show it.
A criterion that is ``undecided`` therefore leaves the reading on the next tier down that IS shown:
``separation`` when that is met, else ``undetermined``.

**A stored analysis says which rule decided its tiers** (:data:`JUDGED_TIER_RULE`, on
``EvalAnalysis.judged_tier_rule``). One stored before intervals carries none: its tiers were decided on the
point estimate, and every surface that renders them says so rather than presenting them as this rule's.

**A judge is a model AND its config.** Every tier is keyed by dimension, scale, the model that served
the scores and the versioned judge config that asked (``None`` = the built-in prompt): a changed judge
prompt is a different judge, so a measurement under one config never sets the tier of readings judged
under another.

**Tiers are flagged, not dropped** (PD-13): every judged reading stays on every surface, carrying its
tier and the two criteria that decided it, so a reader sees how much the number can bear.

A leaf module: the surface and the analysis models import it, and it imports neither.
"""

from __future__ import annotations

from typing import Final, Literal

import math

from pydantic import Field, computed_field, model_validator

from threetears.evals.contracts.base import EvalDocumentModel
from threetears.evals.contracts.models import DimName, JudgeTemperature, RubricScale

#: The least judge–human agreement (weighted kappa; kappa on pass/fail) a dimension's judge must reach
#: to be ``calibrated`` on it. Owner ruling, 2026-10-06.
CALIBRATION_MIN_AGREEMENT: Final = 0.6

#: The fewest distinct results a calibration must cover before it can decide anything: below it the
#: calibration is ``insufficient``, whatever its kappa. The owner's ruling (2026-10-06) reads "20 pairs";
#: counting distinct results is that floor made unforgeable — every result covered is at least one pair,
#: and a pile of pairs about a handful of results is not twenty pieces of evidence.
CALIBRATION_MIN_RESULTS: Final = 20

#: The least agreement a judge must reach with its own repeated scores (the same statistic as
#: calibration) to earn ``separation``. Owner ruling, 2026-10-06.
SEPARATION_MIN_AGREEMENT: Final = 0.8

#: The fewest distinct results a self-agreement must cover before it can decide anything. The ruling sets no
#: floor of its own here. Calibration's 20 was used first, and at 20 results no valid interval can show a
#: kappa of at least 0.8: twenty perfect repeats bound it near 0.6-0.7, and even the exact null distribution
#: puts more than 5% of a judge at 0.8 on perfect agreement. A floor nobody can clear decides nothing, so this
#: is the count at which a near-perfect judge (true self-agreement 0.95) earns the tier at least 80% of the time
#: on five of six simulated marginals (81-98%; 70% on a heavily skewed 1-5 one). At true 0.9, 80% needs more than
#: 200 results. ``tests/test_simulated_agreement.py`` holds it.
SEPARATION_MIN_RESULTS: Final = 120

#: The tier a judged reading stands on: one of the three the evidence can establish, or ``undetermined``
#: when it establishes none of them.
JudgedEvidenceTier = Literal["calibrated", "separation", "incidental", "undetermined"]

#: Weakest first, for composing several judged readings into what all of them can bear.
#: ``undetermined`` sits above ``incidental``: an undetermined reading is SOME judged tier, so its
#: weakest possible value is incidental, and a composite holding a known incidental reading is
#: incidental whatever the undetermined one turns out to be — while one holding an undetermined
#: reading beside stronger ones is undetermined, since it could still be incidental.
JUDGED_TIERS_WEAKEST_FIRST: Final[tuple[JudgedEvidenceTier, ...]] = (
    "incidental",
    "undetermined",
    "separation",
    "calibrated",
)

#: How one criterion read: its interval wholly at or above the threshold over enough results (``met``),
#: wholly below it (``not_met``), across it (``undecided``), or too little evidence to say (fewer distinct
#: results than its floor, or a kappa or interval that is undefined).
CriterionState = Literal["met", "not_met", "undecided", "insufficient"]

#: The rule a stored analysis's judged tiers were decided by. ``interval_lower_bound``: each criterion on
#: confidence bounds for its agreement against the bar. A stored analysis carrying none predates it: its tiers were
#: the point estimate against the bar.
JudgedTierRule = Literal["interval_lower_bound"]

#: The rule this build decides tiers by.
JUDGED_TIER_RULE: Final[JudgedTierRule] = "interval_lower_bound"


class TierCriterion(EvalDocumentModel):
    """One of the two measurements a judged tier is decided by, as it read."""

    state: CriterionState = Field(
        description=(
            "`met`: the interval's lower bound reached `threshold`, over at least `min_results` distinct results. "
            "`not_met`: its upper bound is below `threshold`, over at least `min_results`. `undecided`: the "
            "interval straddles `threshold` — neither met nor missed. `insufficient`: fewer distinct results than "
            "`min_results`, or an undefined kappa or interval — too little evidence to say, never a miss."
        )
    )
    n: int = Field(ge=0, description="The pairs the agreement was read over; 0 when nothing was measured.")
    results: int = Field(
        ge=0,
        description=(
            "The distinct results among the raters whose kappa entered `agreement` — what the floor counts, "
            "because pairs can be multiplied by re-measuring the same results and results cannot."
        ),
    )
    agreement: float | None = Field(
        description=(
            "Weighted kappa on a 1-5 dimension, kappa on pass/fail, per rater and pooled by result (each distinct "
            "result weighing 1, split across its raters); None when nothing was measured or every rater's kappa "
            "is undefined."
        )
    )
    lower: float | None = Field(
        default=None,
        description=(
            "The one-sided 95% lower confidence bound on `agreement` (a score interval over the distinct results — "
            "`threetears.evals.analysis.agreement.agreement_interval`); None when `agreement` is None or fewer than "
            "two results carry it. The criterion is met only when this reaches `threshold`."
        ),
    )
    upper: float | None = Field(
        default=None,
        description=(
            "The one-sided 97.5% upper confidence bound on `agreement`; the criterion is not met only when this is "
            "below `threshold`."
        ),
    )
    threshold: float = Field(description="The agreement the criterion asks for.")
    min_results: int = Field(ge=1, description="The fewest distinct results it is decided over.")

    @model_validator(mode="after")
    def _state_follows_the_numbers(self) -> TierCriterion:
        """Refuse a state its own numbers contradict — the state is arithmetic, never a second opinion.

        Raises:
            ValueError: ``results`` exceeds ``n`` (every result counted is at least one pair); one end of the
                interval without the other, or an interval without an agreement or not containing it; or
                ``state`` is not the one :func:`criterion_state` derives from the numbers.
        """
        if self.results > self.n:
            raise ValueError(f"a criterion over {self.n} pairs cannot cover {self.results} results")
        if (self.lower is None) != (self.upper is None):
            raise ValueError("an interval has both ends or neither")
        if self.lower is not None and self.upper is not None:
            if self.agreement is None or not self.lower <= self.agreement <= self.upper:
                raise ValueError(
                    f"an interval [{self.lower}, {self.upper}] must contain its agreement {self.agreement}"
                )
        derived = criterion_state(
            self.results,
            self.agreement,
            self.interval,
            threshold=self.threshold,
            min_results=self.min_results,
        )
        if self.state != derived:
            raise ValueError(
                f"a criterion over results={self.results}, agreement={self.agreement} is {derived!r}, "
                f"not {self.state!r}"
            )
        return self

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description=(
            "How many more distinct results this criterion needs before it can be decided: for `insufficient`, at "
            "least the rest of the floor; for `undecided`, about how many more would carry the interval clear of "
            "the bar if agreement held at its estimate (the interval's half-width shrinking as one over the square "
            "root of the results) — a planning figure, never a promise. None once decided, and where no estimate "
            "can say (an undecided estimate exactly on the bar)."
        )
    )
    @property
    def results_needed(self) -> int | None:
        """More distinct results before this criterion is decided, as far as its own numbers can project."""
        if self.state in ("met", "not_met"):
            return None
        short = max(self.min_results - self.results, 0)
        if self.agreement is None or self.lower is None or self.upper is None or self.results == 0:
            return short or None
        if self.agreement > self.threshold:
            ratio = (self.agreement - self.lower) / (self.agreement - self.threshold)
        elif self.agreement < self.threshold:
            ratio = (self.upper - self.agreement) / (self.threshold - self.agreement)
        else:
            return short or None
        projected = math.ceil(self.results * ratio * ratio)
        return max(max(projected, self.min_results) - self.results, 1)

    @property
    def interval(self) -> tuple[float, float] | None:
        """``(lower, upper)``, or None when the criterion carries no interval."""
        if self.lower is None or self.upper is None:
            return None
        return (self.lower, self.upper)


class JudgeEvidenceTier(EvalDocumentModel):
    """The evidence tier of one judge's readings on one dimension, and the two measurements that decided it."""

    rubric_dim: DimName = Field(min_length=1, description="The judged dimension.")
    scale: RubricScale = Field(description="The scale it was judged on.")
    judge_model: str | None = Field(
        description=(
            "The model that served the scores, as the provider named it; None when the responses named none — "
            "a judge nobody observed, never read as a match for a named one."
        )
    )
    judge_config_id: str | None = Field(
        description=(
            "The versioned JudgeConfig that asked for the scores; None = the built-in prompt. Part of the judge's "
            "identity: a changed judge prompt is a different judge, so its readings get their own tier."
        )
    )
    judge_temperature: JudgeTemperature | None = Field(
        default=None,
        description=(
            "The temperature the scores' calls were sent at ('model_default' = sent none, the model refusing one). "
            "Part of the judge's identity too: a judge sampled at another temperature gets its own tier. None = not "
            "recorded, never a match for a recorded one."
        ),
    )
    tier: JudgedEvidenceTier = Field(description="The tier the two criteria decide — see `tier_of`.")
    calibration: TierCriterion = Field(description="The judge's agreement with people's ratings of the same results.")
    separation: TierCriterion = Field(description="The judge's agreement with its own repeated scores.")

    @model_validator(mode="after")
    def _tier_follows_the_criteria(self) -> JudgeEvidenceTier:
        """Refuse a tier the criteria do not decide, or criteria held to thresholds other than the ruled ones.

        Raises:
            ValueError: ``tier`` is not :func:`tier_of` of the criteria, or a criterion's threshold or floor is
                not this module's constant.
        """
        if (self.calibration.threshold, self.calibration.min_results) != (
            CALIBRATION_MIN_AGREEMENT,
            CALIBRATION_MIN_RESULTS,
        ):
            raise ValueError("calibration is held to CALIBRATION_MIN_AGREEMENT over CALIBRATION_MIN_RESULTS")
        if (self.separation.threshold, self.separation.min_results) != (
            SEPARATION_MIN_AGREEMENT,
            SEPARATION_MIN_RESULTS,
        ):
            raise ValueError("separation is held to SEPARATION_MIN_AGREEMENT over SEPARATION_MIN_RESULTS")
        derived = tier_of(self.calibration, self.separation)
        if self.tier != derived:
            raise ValueError(f"these criteria decide {derived!r}, not {self.tier!r}")
        return self


def agreement_statistic(scale: RubricScale, kappa: float | None, weighted_kappa: float | None) -> float | None:
    """The one agreement figure both tiers are held to: weighted kappa on 1-5, kappa on pass/fail.

    Over two categories quadratic weights and no weights are the same number, which is why an agreement
    read publishes no weighted kappa on pass/fail; reading ``kappa`` there is reading that number.

    Args:
        scale: The dimension's scale.
        kappa: The unweighted kappa.
        weighted_kappa: The quadratic-weighted kappa (None on pass/fail).

    Returns:
        The figure, or None when it is undefined.
    """
    return weighted_kappa if scale == "ordinal" else kappa


def criterion_state(
    results: int,
    agreement: float | None,
    interval: tuple[float, float] | None,
    *,
    threshold: float,
    min_results: int,
) -> CriterionState:
    """How one criterion reads, decided on the agreement's interval and never on its point estimate.

    Args:
        results: The distinct results the agreement covers.
        agreement: The agreement figure, or None when undefined.
        interval: Its confidence bounds ``(lower, upper)``, or None when there are none.
        threshold: The agreement asked for (inclusive).
        min_results: The fewest distinct results (inclusive).

    Returns:
        ``insufficient`` below the floor, or on an undefined agreement or interval; else ``met`` when the
        lower bound reaches the threshold, ``not_met`` when the upper bound is below it, and ``undecided``
        when the interval straddles it.
    """
    if results < min_results or agreement is None or interval is None:
        return "insufficient"
    lower, upper = interval
    if lower >= threshold:
        return "met"
    if upper < threshold:
        return "not_met"
    return "undecided"


def _criterion(
    n: int,
    results: int,
    agreement: float | None,
    interval: tuple[float, float] | None,
    *,
    threshold: float,
    min_results: int,
) -> TierCriterion:
    """A criterion held to ``threshold`` over ``min_results``, its state derived from the numbers."""
    return TierCriterion(
        state=criterion_state(results, agreement, interval, threshold=threshold, min_results=min_results),
        n=n,
        results=results,
        agreement=agreement,
        lower=None if interval is None else interval[0],
        upper=None if interval is None else interval[1],
        threshold=threshold,
        min_results=min_results,
    )


def calibration_criterion(
    n: int, results: int, agreement: float | None, interval: tuple[float, float] | None = None
) -> TierCriterion:
    """The calibration criterion over ``n`` judge–human pairs covering ``results`` results, at ``agreement``.

    ``interval`` is the agreement's confidence bounds; without them the criterion cannot be decided (``insufficient``).
    """
    return _criterion(
        n, results, agreement, interval, threshold=CALIBRATION_MIN_AGREEMENT, min_results=CALIBRATION_MIN_RESULTS
    )


def separation_criterion(
    n: int, results: int, agreement: float | None, interval: tuple[float, float] | None = None
) -> TierCriterion:
    """The separation criterion over ``n`` first-score/repeat pairs covering ``results`` results, at ``agreement``.

    ``interval`` is the agreement's confidence bounds; without them the criterion cannot be decided (``insufficient``).
    """
    return _criterion(
        n, results, agreement, interval, threshold=SEPARATION_MIN_AGREEMENT, min_results=SEPARATION_MIN_RESULTS
    )


def tier_of(calibration: TierCriterion, separation: TierCriterion) -> JudgedEvidenceTier:
    """The tier two criteria decide.

    Args:
        calibration: The judge's agreement with people.
        separation: The judge's agreement with itself.

    Returns:
        ``calibrated`` when calibration is met; else ``separation`` when separation is met; else
        ``incidental`` when both were shown below their bars (``not_met``); else ``undetermined`` — which an
        ``undecided`` criterion leads to, since an interval across the bar shows neither.
    """
    if calibration.state == "met":
        return "calibrated"
    if separation.state == "met":
        return "separation"
    if calibration.state == "not_met" and separation.state == "not_met":
        return "incidental"
    return "undetermined"


def weakest_judged_tier(tiers: list[JudgedEvidenceTier]) -> JudgedEvidenceTier:
    """What several judged readings can bear together: the weakest of them (see :data:`JUDGED_TIERS_WEAKEST_FIRST`).

    Args:
        tiers: At least one tier.

    Returns:
        The weakest.

    Raises:
        ValueError: ``tiers`` is empty — no reading has no tier to compose, and none is not undetermined.
    """
    if not tiers:
        raise ValueError("composing judged tiers needs at least one")
    return min(tiers, key=JUDGED_TIERS_WEAKEST_FIRST.index)


__all__ = [
    "CALIBRATION_MIN_AGREEMENT",
    "CALIBRATION_MIN_RESULTS",
    "JUDGED_TIERS_WEAKEST_FIRST",
    "JUDGED_TIER_RULE",
    "SEPARATION_MIN_AGREEMENT",
    "SEPARATION_MIN_RESULTS",
    "CriterionState",
    "JudgeEvidenceTier",
    "JudgedEvidenceTier",
    "JudgedTierRule",
    "TierCriterion",
    "agreement_statistic",
    "calibration_criterion",
    "criterion_state",
    "separation_criterion",
    "tier_of",
    "weakest_judged_tier",
]
