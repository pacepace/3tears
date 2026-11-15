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

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.schema.reading_states import Coordinate, Reason, SampleKind, StateOf
from threetears.evals.kernel.evidence_tiers import CriterionState

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
    """How often a reading's test was looked at, against how often it planned to be."""

    planned: int | None = Field(default=None, description="The looks declared; none where none were.")
    taken: int = Field(ge=1, description="The looks taken.")
