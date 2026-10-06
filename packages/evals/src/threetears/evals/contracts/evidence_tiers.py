"""The evidence tier a judged reading stands on, decided by code from what the judge's reliability was measured to be.

A judged score is one model's opinion of another's output. How far a reader may lean on it depends on
two measurements of the judge, and on nothing a report writer says (owner ruling, 2026-10-06, PD-13):

- **calibrated** — the judge agrees with PEOPLE: on the dimension, its agreement with people's
  calibration ratings of the same results (:func:`threetears.evals.analysis.judge_agreement`, which
  pairs only a person's rating, never an agent's) is at least :data:`CALIBRATION_MIN_AGREEMENT` over at
  least :data:`CALIBRATION_MIN_PAIRS` pairs. The strongest judged tier: the judge is right as people
  judge rightness.
- **separation** — the judge agrees with ITSELF: re-scoring evidence it already scored, under the
  apparatus the run recorded (:func:`threetears.evals.analysis.judge_self_agreement`), its agreement with
  its own first scores is at least :data:`SEPARATION_MIN_AGREEMENT` over at least
  :data:`SEPARATION_MIN_PAIRS` pairs. Its scores are repeatable, so a gap between arms is not its own
  noise — which says nothing about whether it is right.
- **incidental** — a judged reading meeting neither: both measurements were taken over enough pairs
  and both fell short. The judge was measured and found neither calibrated nor consistent.

**Agreement is one statistic for both tiers**, so the two thresholds are on one scale and the tiers
compare: Cohen's kappa with quadratic weights on a 1-5 dimension, and Cohen's kappa on pass/fail, where
the two weightings are the same number (:func:`agreement_statistic`). Both measurements average it per
rater exactly as :mod:`threetears.evals.analysis.agreement` describes.

**Too little evidence is a state, never a tier.** A measurement over fewer pairs than its floor, or
whose kappa is undefined, is ``insufficient``, and a reading whose evidence does not decide its tier is
``undetermined`` — never quietly filed as incidental. The rule (:func:`tier_of`): calibrated when the
calibration criterion is met; else separation when the separation criterion is met; else incidental
when BOTH criteria were measured over enough pairs and missed; else undetermined. A met criterion
establishes its tier whatever the other one reads, because a tier is a floor the evidence has reached;
``incidental`` is itself a finding about the judge, so it needs both measurements to make it.

**Tiers are flagged, not dropped** (PD-13): every judged reading stays on every surface, carrying its
tier and the two criteria that decided it, so a reader sees how much the number can bear.

A leaf module: the surface and the analysis models import it, and it imports neither.
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import Field, model_validator

from threetears.evals.contracts.base import EvalDocumentModel
from threetears.evals.contracts.models import DimName, RubricScale

#: The least judge–human agreement (weighted kappa; kappa on pass/fail) a dimension's judge must reach
#: to be ``calibrated`` on it. Owner ruling, 2026-10-06.
CALIBRATION_MIN_AGREEMENT: Final = 0.6

#: The fewest judge–human pairs a calibration is read over before it can decide anything: below it the
#: calibration is ``insufficient``, whatever its kappa. Owner ruling, 2026-10-06.
CALIBRATION_MIN_PAIRS: Final = 20

#: The least agreement a judge must reach with its own repeated scores (the same statistic as
#: calibration) to earn ``separation``. Owner ruling, 2026-10-06.
SEPARATION_MIN_AGREEMENT: Final = 0.8

#: The fewest first-score/repeat pairs a self-agreement is read over before it can decide anything. The
#: ruling sets no floor of its own here; this is calibration's, because the ruling asks for the two
#: agreements to be computed alike so the tiers compare, and a kappa over three pairs decides nothing
#: either way.
SEPARATION_MIN_PAIRS: Final = CALIBRATION_MIN_PAIRS

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

#: How one criterion read: its threshold reached over enough pairs, missed over enough pairs, or too
#: little evidence to say (fewer pairs than its floor, or a kappa that is undefined).
CriterionState = Literal["met", "not_met", "insufficient"]


class TierCriterion(EvalDocumentModel):
    """One of the two measurements a judged tier is decided by, as it read."""

    state: CriterionState = Field(
        description=(
            "`met`: `agreement` reached `threshold` over at least `min_pairs` pairs. `not_met`: it fell short "
            "over at least `min_pairs`. `insufficient`: fewer pairs than `min_pairs`, or an undefined kappa — "
            "too little evidence to say, never a miss."
        )
    )
    n: int = Field(ge=0, description="The pairs the agreement was read over; 0 when nothing was measured.")
    agreement: float | None = Field(
        description=(
            "Weighted kappa on a 1-5 dimension, kappa on pass/fail; None when nothing was measured or the "
            "kappa is undefined."
        )
    )
    threshold: float = Field(description="The agreement the criterion asks for.")
    min_pairs: int = Field(ge=1, description="The fewest pairs it is decided over.")

    @model_validator(mode="after")
    def _state_follows_the_numbers(self) -> TierCriterion:
        """Refuse a state its own numbers contradict — the state is arithmetic, never a second opinion.

        Raises:
            ValueError: ``state`` is not the one :func:`criterion_state` derives from the numbers.
        """
        derived = criterion_state(self.n, self.agreement, threshold=self.threshold, min_pairs=self.min_pairs)
        if self.state != derived:
            raise ValueError(
                f"a criterion over n={self.n}, agreement={self.agreement} is {derived!r}, not {self.state!r}"
            )
        return self


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
        if (self.calibration.threshold, self.calibration.min_pairs) != (
            CALIBRATION_MIN_AGREEMENT,
            CALIBRATION_MIN_PAIRS,
        ):
            raise ValueError("calibration is held to CALIBRATION_MIN_AGREEMENT over CALIBRATION_MIN_PAIRS")
        if (self.separation.threshold, self.separation.min_pairs) != (SEPARATION_MIN_AGREEMENT, SEPARATION_MIN_PAIRS):
            raise ValueError("separation is held to SEPARATION_MIN_AGREEMENT over SEPARATION_MIN_PAIRS")
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


def criterion_state(n: int, agreement: float | None, *, threshold: float, min_pairs: int) -> CriterionState:
    """How one criterion reads: insufficient below its floor or on an undefined kappa, else met or not met.

    Args:
        n: The pairs.
        agreement: The agreement figure, or None when undefined.
        threshold: The agreement asked for (inclusive).
        min_pairs: The fewest pairs (inclusive).

    Returns:
        The state.
    """
    if n < min_pairs or agreement is None:
        return "insufficient"
    return "met" if agreement >= threshold else "not_met"


def calibration_criterion(n: int, agreement: float | None) -> TierCriterion:
    """The calibration criterion over ``n`` judge–human pairs at ``agreement``."""
    return TierCriterion(
        state=criterion_state(n, agreement, threshold=CALIBRATION_MIN_AGREEMENT, min_pairs=CALIBRATION_MIN_PAIRS),
        n=n,
        agreement=agreement,
        threshold=CALIBRATION_MIN_AGREEMENT,
        min_pairs=CALIBRATION_MIN_PAIRS,
    )


def separation_criterion(n: int, agreement: float | None) -> TierCriterion:
    """The separation criterion over ``n`` first-score/repeat pairs at ``agreement``."""
    return TierCriterion(
        state=criterion_state(n, agreement, threshold=SEPARATION_MIN_AGREEMENT, min_pairs=SEPARATION_MIN_PAIRS),
        n=n,
        agreement=agreement,
        threshold=SEPARATION_MIN_AGREEMENT,
        min_pairs=SEPARATION_MIN_PAIRS,
    )


def tier_of(calibration: TierCriterion, separation: TierCriterion) -> JudgedEvidenceTier:
    """The tier two criteria decide.

    Args:
        calibration: The judge's agreement with people.
        separation: The judge's agreement with itself.

    Returns:
        ``calibrated`` when calibration is met; else ``separation`` when separation is met; else
        ``incidental`` when both were measured over enough pairs and missed; else ``undetermined``.
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
    "CALIBRATION_MIN_PAIRS",
    "JUDGED_TIERS_WEAKEST_FIRST",
    "SEPARATION_MIN_AGREEMENT",
    "SEPARATION_MIN_PAIRS",
    "CriterionState",
    "JudgeEvidenceTier",
    "JudgedEvidenceTier",
    "TierCriterion",
    "agreement_statistic",
    "calibration_criterion",
    "criterion_state",
    "separation_criterion",
    "tier_of",
    "weakest_judged_tier",
]
