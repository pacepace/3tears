"""The typed shapes of a check and of what a check produces, as the bundle and the report input both hold them.

Every instrument is checked: a check names an instrument, what it was checked against (people, a model, itself, or
identical arms), on what sample, with one result (a
:data:`~threetears.evals.kernel.evidence_tiers.CriterionState`). A corrected reading is a check's
result carried onto one contrast; a re-score is the judge checked against itself; looks are how often a contrast's
test was read against how often it planned to be.

One class per shape, held by the bundle and the report input alike, so a fact has one spelling at both ends. Every
field a shape declares is optional except what names it, and every state and reason is one of
:mod:`~threetears.evals.schema.reading_states`: no shape here carries a sentence saying why a figure is absent.

A blind read's sample and what its readers preferred are the blind read's own shapes, and none is restated here.

Nothing here computes: each shape is filled where its statistic is computed and copied through from there.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from threetears.evals.schema.reading_states import Coordinate, Reason, SampleKind, StateOf
from threetears.evals.kernel.evidence_tiers import CriterionState
from threetears.evals.analysis.significance import SeparationTest

__all__ = ["Check", "Corrected", "Looks", "Rescore"]


class _Shape(BaseModel):
    """Immutable, and refusing a key it does not declare."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Check(_Shape):
    """One check of one instrument."""

    instrument: str = Field(description="What was checked: a judge by its dimension, a reviewer by its id, or the rig.")
    criterion: str | None = Field(default=None, description="The criterion the check decides, where it names one.")
    against: Literal["people", "model", "itself", "identical_arms"] = Field(description="What it was checked against.")
    against_id: str | None = Field(default=None, description="The model it was checked against, by id.")
    same_family_as: list[str] = Field(
        default_factory=list,
        description="The arms whose candidate model is of the family of the model checked against.",
    )
    sample: SampleKind | None = Field(
        default=None,
        description="Whether the items it was read on were drawn at random, chosen, or no sample stands behind it.",
    )
    sample_results: int | None = Field(default=None, description="Results read.")
    sample_cases: int | None = Field(default=None, description="Distinct cases behind them.")
    skipped: int | None = Field(default=None, description="Items a reader skipped.")
    read_id: str | None = Field(default=None, description="The blind read the sample is, by id.")
    result: CriterionState = Field(description="met, not_met, undecided or insufficient.")
    reason: Reason | None = Field(default=None, description="Why the result is not met, where a shared reason says it.")
    coordinate: Coordinate | None = Field(
        default=None, description="The coordinate an unrecorded identity names, where that is the reason."
    )
    state: StateOf | None = Field(
        default=None,
        description="Where the check could not be made at all (both sides replayed a recording): not available, why.",
    )
    kappa: float | None = Field(default=None, description="Agreement beyond chance.")
    kappa_interval: tuple[float, float] | None = Field(default=None, description="The interval on it.")
    lean: float | None = Field(default=None, description="The instrument's lean toward one arm.")
    lean_interval: tuple[float, float] | None = Field(default=None, description="The interval on it.")
    lean_untested: list[str] = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
        description="The arms, by key, a lean across arms was not tested on, each for lean_untested_reason: so a met "
        "check never reads as one that ruled a lean out on them.",
    )
    lean_untested_reason: Reason | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Why the lean was not tested on those arms: too_few, their answered claims too few to read a rate.",
    )
    half_width: float | None = Field(default=None, description="The half-width the check was read at.")
    reads: int | None = Field(default=None, description="Reads behind the figures.")
    delta: float | None = Field(default=None, description="The difference between identical arms.")
    interval: tuple[float, float] | None = Field(default=None, description="The interval on that difference.")
    p_adjusted: float | None = Field(default=None, description="Its adjusted p.")
    reading: str | None = Field(default=None, description="The reading a check of the rig covers.")
    covers_arms: list[str] = Field(default_factory=list, description="The arms a check of the rig covers.")


class Corrected(_Shape):
    """A check's result on one judged contrast: the contrast as what the judge was checked against would read it."""

    against: Literal["people", "model"] = Field(description="Whose reading corrects the judge's.")
    against_id: str | None = Field(default=None, description="The model, by id.")
    result: CriterionState | None = Field(default=None, description="Whether the corrected reading confirms.")
    state: StateOf | None = Field(default=None, description="Its state where it is not counted.")
    estimate: float | None = Field(default=None, description="The corrected difference.")
    interval: tuple[float, float] | None = Field(default=None, description="The interval on it.")
    level: str | None = Field(default=None, description="The interval's level.")
    lean: float | None = Field(default=None, description="The judge's lean the correction removes.")
    lean_interval: tuple[float, float] | None = Field(default=None, description="The interval on it.")
    labelled_cases: int | None = Field(default=None, description="Cases both the judge and the other reader read.")
    identical_cases: int | None = Field(default=None, description="Of those, the cases they read alike.")
    read_id: str | None = Field(default=None, description="The blind read the correction rests on.")


class Rescore(_Shape):
    """A judge checked against itself on one arm: how much a second scoring of the same reply moves."""

    state: StateOf | None = Field(default=None, description="Its state where it is not counted.")
    variance: float | None = Field(default=None, description="The variance between scorings of one reply.")
    variance_interval: tuple[float, float] | None = Field(
        default=None, exclude_if=lambda value: value is None, description="The interval on it."
    )
    repeated_results: int | None = Field(default=None, description="Results scored more than once.")
    repeated_cases: int | None = Field(default=None, description="Distinct cases behind them.")
    fail_given_pass: float | None = Field(default=None, description="How often a passed reply fails on a re-score.")
    fail_given_pass_interval: tuple[float, float] | None = Field(default=None, description="The interval on it.")
    ceiling_low: float | None = Field(default=None, description="The low end of the ceiling the noise puts on pass^k.")
    ceiling_high: float | None = Field(default=None, description="Its high end.")
    k: int | None = Field(default=None, description="The k the ceiling is read at: the most scores one case holds.")
    ceiling_state: StateOf | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Where the ceiling is read for the dimension and could not be: not available, why.",
    )


class Looks(_Shape):
    """How often a reading's test was looked at, against how often it planned to be.

    A look is an occasion on which the contrast's test could have been read again on more or different evidence
    (:func:`~threetears.evals.analysis.significance.looks_taken`): the first, once both arms held a result; then each
    launch whose results changed what the test reads, each run that joined or left, each re-measure and each re-judge
    made after it. ``taken`` is the most any one contrast of the aspect has had. It is exact where every such fact is
    stored, and a lower bound (``lower_bound``) where one is not: a run that left with no result recorded, an archived
    or unresolved member, a result with no ordered time, or re-judges, which are stored per result, so passes over
    different results cannot be told apart.

    **Looks were not recorded** (``recorded`` false) where a member run's joining was never recorded: it joined before
    durable 17.3 recorded membership. The occasions its stored results give are then artifacts of how the runs were
    stored (each arm launched alone, one group of cases after another), not occasions on which a reading could have
    been taken, and runs joined in one write are dated each by its own launch: the count could be too high or too low,
    so it is neither a bound above nor a bound below, and it is not given as looks. ``evidence_changes`` gives that
    count as what it is: how many times the stored evidence changed. Where looks were recorded since (a run joined,
    left, was launched after it joined, re-measured or re-judged after the first comparison), ``taken`` is the first
    comparison plus those, a true lower count (``lower_bound``): at least that many looks since looks were recorded,
    and earlier ones not recorded. Where none were, ``taken`` is None.
    """

    planned: int | None = Field(
        default=None,
        description=(
            "The looks the design plans, where the plan was declared before the first look; None where none was, or "
            "where it was declared after (``late_plan``)."
        ),
    )
    taken: int | None = Field(
        default=None,
        ge=1,
        description=(
            "The looks taken. Where looks were not recorded (``recorded`` false): the looks recorded since, counting "
            "the first comparison, a lower count; None where none were."
        ),
    )
    # Left out where true, so the looks of a campaign created once looks were counted read {planned, taken} alone.
    plannable: bool = Field(
        default=True,
        exclude_if=lambda value: value,
        description=(
            "Whether the campaign was created once looks were counted, so its design could have declared them. False "
            "for an earlier campaign: its stored verdicts do not move, and its looks qualify the method only."
        ),
    )
    # Left out where false, so an exact count reads {planned, taken} alone.
    lower_bound: bool = Field(
        default=False,
        exclude_if=lambda value: not value,
        description="Whether ``taken`` is a lower bound: a fact a look is counted by is not stored, so more were taken.",
    )
    # Left out where true: only an aspect with a member whose joining was never recorded says it.
    recorded: bool = Field(
        default=True,
        exclude_if=lambda value: value,
        description=(
            "Whether looks were recorded: false where a member run joined before durable 17.3 recorded membership, so "
            "the looks before are not counted; ``taken`` is then only the looks recorded since, a lower count, or None."
        ),
    )
    # Left out where none: only an aspect whose looks were not recorded carries it.
    evidence_changes: int | None = Field(
        default=None,
        ge=1,
        exclude_if=lambda value: value is None,
        description=(
            "Where looks were not recorded: how many times the stored evidence of a contrast changed (its runs' "
            "launches, re-measures and re-judges as stored), context and not a count of looks. None where they were."
        ),
    )
    # Left out where none: the sizes a reference rate's setting is checked against.
    launched: int | None = Field(
        default=None,
        ge=0,
        exclude_if=lambda value: value is None,
        description="The fewest cases any contrast of the aspect read at its first look. None where not counted.",
    )
    largest_step: int | None = Field(
        default=None,
        ge=0,
        exclude_if=lambda value: value is None,
        description="The most cases one later occasion added to any contrast of the aspect. None where not counted.",
    )
    # Left out where empty.
    tests: list[SeparationTest] = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
        description=(
            "The tests the aspect's readings are read by, sorted: ``bounded`` (a pass rate, a judged score, a "
            "guardrail) and ``t`` (a mean, a measure's score, a cost per reply). What a caveat on looks quotes a "
            "measured rate for."
        ),
    )
    late_plan: int | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "The looks a plan declared after the first look named (or over runs that held results before it was "
            "declared): read as no plan, since the looks it would count were already taken. None where no plan was "
            "declared late."
        ),
    )

    @model_validator(mode="after")
    def _a_count_or_not_recorded(self) -> Looks:
        """A count of looks where they were recorded; where they were not, the evidence changes, and the looks
        recorded since only as a lower count."""
        if self.recorded and (self.taken is None or self.evidence_changes is not None):
            raise ValueError("recorded looks carry taken and no evidence_changes")
        if not self.recorded and (self.evidence_changes is None or self.lower_bound != (self.taken is not None)):
            raise ValueError("unrecorded looks carry evidence_changes, and a taken only as a lower_bound")
        return self
