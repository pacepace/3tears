"""What code reads off a campaign, aspect by aspect: each candidate's change against the control, and every floor.

Code reports, the writer recommends, a person decides. This is the report: per aspect and candidate, the change in
pass rate against the control and whether it is **shown better**, **shown worse** or **unclear**; per floor and
candidate, whether it is broken, held or unclear; and for an unclear reading, the cases that would settle it. The
writer's recommendation and the person's call both read it, and neither can change it.

**One family across aspects, two directions.** A campaign tests every aspect × candidate row at once, so its p values
are Holm-adjusted across all of them at the campaign's α, separately for gains and for drops: a gain is the change
tested against zero, a drop the change tested against the aspect's margin. A row's interval is read at the level its
Holm step tested it at, so its reading and its interval never contradict. Floors are read intersection-union, each at
its nominal level with no correction, since a warning that fires less often for every floor added is no warning.

Only the shapes are here yet: nothing computes them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from threetears.evals.analysis.report.decision import SampleProjection
from threetears.evals.schema.base import EvalDocumentModel

__all__ = ["AspectReading", "CampaignReadings", "FloorState", "FloorStatus", "ShownReading"]

#: What one aspect × candidate row shows against the control. ``better``: the gain is shown, its Holm-adjusted p
#: under the campaign's α. ``worse``: a drop past the aspect's margin is shown, likewise — never read on an aspect with
#: no margin. ``unclear``: neither.
ShownReading = Literal["better", "worse", "unclear"]

#: Where one floor stands for one candidate. ``broken``: the evidence shows the candidate past the floor. ``held``: it
#: shows the candidate within it. ``unclear``: neither, the reason saying why.
FloorState = Literal["broken", "held", "unclear"]


class AspectReading(EvalDocumentModel):
    """One candidate on one aspect against the control: the change in pass rate, how sure, and what it shows.

    Every figure is the candidate minus the control, in points of a 0-100 pass rate, so a positive change is a
    candidate passing more.

    **The reading is decided per direction.** The gain and the drop are two Holm families, so each direction has its
    own p values and its own interval, read at the level that direction's Holm step tested it at: ``better`` is the
    gain shown (its interval wholly above zero), ``worse`` the drop past the margin shown (its interval wholly below
    minus the margin). ``unclear`` is neither, and a page showing one shows both intervals, so a reader sees how far
    each side is from settling.
    """

    aspect: str = Field(description="The aspect's label.")
    arm: str = Field(description="The candidate's variant key.")
    arm_label: str = Field(description="What a reader calls the candidate: its arm's label, else its variant's name.")
    change: float = Field(description="The candidate's pass rate minus the control's, in points (positive = better).")
    gain_interval: tuple[float, float] | None = Field(
        description=(
            "The change's interval in points, low then high, at `gain_confidence`: what `better` is read on. None "
            "when the arms' cases cannot bound it."
        )
    )
    gain_confidence: float | None = Field(
        description=(
            "The level `gain_interval` is read at: one minus the α this row's step in the gains' Holm family tested "
            "it at, so the interval and the gain reading never contradict. None with no interval."
        )
    )
    drop_interval: tuple[float, float] | None = Field(
        description=(
            "The change's interval in points, low then high, at `drop_confidence`: what `worse` is read on, against "
            "the margin. None when the arms' cases cannot bound it, or the aspect declares no margin."
        )
    )
    drop_confidence: float | None = Field(
        description=(
            "The level `drop_interval` is read at: one minus the α this row's step in the drops' Holm family tested "
            "it at. None with no interval."
        )
    )
    gain_p: float | None = Field(
        description="The p value of the change against zero, on the case-paired differences, unadjusted. None when untested."
    )
    gain_p_holm: float | None = Field(
        description="`gain_p` Holm-adjusted across every aspect × candidate row's gain. None when untested."
    )
    drop_p: float | None = Field(
        description=(
            "The p value of a drop past the margin, on the case-paired differences, unadjusted. None when untested, "
            "which an aspect with no margin always is."
        )
    )
    drop_p_holm: float | None = Field(
        description="`drop_p` Holm-adjusted across every aspect × candidate row's drop. None when untested."
    )
    reading: ShownReading = Field(description="What the row shows: `better`, `worse` or `unclear`.")
    margin: float | None = Field(
        description="The aspect's margin in points, the drop tested against. None = none declared: the row is never `worse`."
    )
    control_cases: int = Field(ge=0, description="The usable cases the control's rate was read over.")
    arm_cases: int = Field(ge=0, description="The usable cases the candidate's rate was read over.")
    control_passed: int = Field(ge=0, description="Trials the control passed.")
    control_trials: int = Field(ge=0, description="Trials the control was measured on.")
    arm_passed: int = Field(ge=0, description="Trials the candidate passed.")
    arm_trials: int = Field(ge=0, description="Trials the candidate was measured on.")
    projection: SampleProjection | None = Field(
        description=(
            "For an `unclear` row, the drop side only: the cases that would show the candidate no worse than the "
            "control by more than the aspect's margin. None for a settled row, for an aspect with no margin, or "
            "where nothing can be projected. No gain is projected."
        )
    )


class FloorStatus(EvalDocumentModel):
    """One floor for one candidate: broken, held or unclear, and why."""

    floor: str = Field(description="The floor's key (`Floor.key`): what a recorded call names an override by.")
    aspect: str = Field(description="The label of the aspect the floor is read on.")
    arm: str = Field(description="The candidate's variant key.")
    arm_label: str = Field(description="What a reader calls the candidate.")
    status: FloorState = Field(description="`broken`, `held` or `unclear`.")
    reading: str | None = Field(
        description=(
            "The rate that decided the status, named as a clause names a reading (a check's verbatim text, a "
            "measure's or judged dimension's key): for a floor reading every pass rate, the one that broke it, or "
            "held or unclear the one nearest breaking; for a floor on one reading, that reading. None for a cost "
            "floor, and for the trials' own pass rate, which no clause names; the floor key says which kind it is."
        )
    )
    interval: tuple[float, float] | None = Field(
        description=(
            "The interval the status was read on, low then high, in the clause's units: points of pass rate for a "
            "rate floor, dollars per reply for a cost floor; the candidate minus the control. None when none was read."
        )
    )
    reason: str = Field(description="Why, in one plain sentence, with the counts it rests on.")


class CampaignReadings(EvalDocumentModel):
    """Everything code reads off a campaign: one row per aspect × candidate, and one status per floor × candidate."""

    campaign_id: str = Field(description="The campaign read.")
    alpha: float = Field(gt=0.0, lt=1.0, description="The campaign's α, at which every Holm family is tested.")
    control: str | None = Field(description="The control arm's variant key; None when the campaign declares none.")
    control_label: str = Field(description="What a reader calls the control; empty when there is none.")
    aspects: list[str] = Field(description="The aspects' labels, in the campaign's order.")
    readings: list[AspectReading] = Field(description="One row per aspect × candidate, aspect by aspect.")
    floors: list[FloorStatus] = Field(description="One status per floor × candidate, in the floors' order.")

    @field_validator("aspects")
    @classmethod
    def _each_aspect_once(cls, labels: list[str]) -> list[str]:
        """Refuse an aspect named twice: a label is what every row and every floor says it was read on.

        :param labels: The aspects' labels.
        :ptype labels: list[str]
        :return: The labels.
        :rtype: list[str]
        :raises ValueError: A label appears twice.
        """
        if duplicates := sorted({label for label in labels if labels.count(label) > 1}):
            raise ValueError(f"aspects names {', '.join(duplicates)} more than once; an aspect's label is unique")
        return labels

    def broken_floors(self) -> list[FloorStatus]:
        """Every floor a candidate breaks, in order: what the page opens with.

        :return: The broken statuses.
        :rtype: list[FloorStatus]
        """
        return [status for status in self.floors if status.status == "broken"]

    def on_aspect(self, aspect: str) -> list[AspectReading]:
        """The rows read on one aspect, in candidate order.

        :param aspect: The aspect's label.
        :ptype aspect: str
        :return: Its rows; empty for a label the campaign does not hold.
        :rtype: list[AspectReading]
        """
        return [reading for reading in self.readings if reading.aspect == aspect]
