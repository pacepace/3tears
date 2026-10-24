"""What code reads off a campaign, aspect by aspect, by its kind: a change's candidates against the control and its
floors, a choice's pairs of arms, a measurement's measures, and every arm against every absolute bar.

Code reports, the writer recommends, a person decides. This is the report, and what it holds depends on the
campaign's kind, which its arms decide (:func:`campaign_kind`) and nothing declares:

- **A change** (a control and its candidates): per aspect and candidate, the change in pass rate against the control
  and whether it is **shown better**, **shown worse**, **no worse than the margin** or **unclear**; per floor and
  candidate, whether it is broken, held or unclear; and for an unclear reading, the cases that would settle it.
- **A choice** (two or more arms, no control): per aspect, every pair of arms, the difference between them and which
  is **shown ahead**, if either; one family over every pair × aspect, since a pair has no direction.
- **A measurement** (one arm): each aspect's measures for that arm, each with its interval; nothing is compared.

Every kind reads each arm against each absolute bar, as the aspect's bundle adjudicated it. The writer's
recommendation and the person's call both read the report, and neither can change it.

The sections below are a change's; a choice's pairs are read by the same step-down over one family.

**One family per candidate across its aspects, two directions.** Each candidate's rows, one per aspect, are tested
as one family at the campaign's α, separately for gains and for drops: a gain is the change tested against zero, a
drop the change tested against the aspect's margin. Each direction's claims are one-sided, read off a two-sided
interval, so each errs at most α/2 and the gains and drops together at most α, per candidate. **There is no error
control across candidates**: with several candidates, one of them is shown better or worse by chance more often than
α. Floors are read intersection-union, each at its nominal level with no correction, since a warning that fires less
often for every floor added is no warning.

**No worse than the margin is read as a floor is.** A row neither shown better nor shown worse reads ``within_margin``
where the candidate's drop is shown to stay within the aspect's margin: the floor's own reading
(:func:`~threetears.evals.analysis.report.decision.read_floor`) on the pass rate at the nominal level, uncorrected,
so a row and a ``RateDrop`` floor on the same rate and points never disagree. It needs no correction because of what
it feeds: a candidate is no worse only where it holds on every asked aspect, and a claim made only when every part
holds (an intersection-union test) errs at most α however many parts it has. The cases that would settle an unclear
row are projected at the same level, so a row's count and its floor's are one count.

**The reading is read off the interval.** Each direction is Holm's step-down over its family, carried on the
intervals and made monotone by construction (:func:`_step_down`): a family of ``m`` rows is read on each row's
two-sided bounded interval (:func:`~threetears.evals.analysis.stats.bounded_difference_interval`) at the levels
``1 − α/k``, ``k`` from 1 to ``m``, and a row counts as shown at a level only where it is shown there and at every
looser level, so the procedure never relies on the intervals nesting. A row's interval is read at the level its step
showed it at, or at a level where it does not clear, so its reading and its interval never contradict. One aspect
makes a family of one, read at ``1 − α``, the 95% a floor reads, so a one-aspect campaign's floor on its pass rate and
each candidate's reading against the same margin are one computation and cannot disagree: a drop is shown by
:func:`~threetears.evals.analysis.report.decision.read_floor` itself, on the pass rate alone, with the floor's case
floor and its bound over lost trials, and a gain by the same rules mirrored. Each row's p is read on its family's
grid, ``α / k`` for the largest ``k`` it reaches (1 where it reaches none), so its Holm-adjusted p
(:func:`~threetears.evals.analysis.stats.holm_adjust`) is at most α exactly where the row is shown, to rounding.

:func:`campaign_readings` reads every aspect's evidence bundle and nothing else: no storage, so a caller resolves the
arms (named and derived, :func:`~threetears.evals.analysis.campaigns.campaign_arms`) and assembles the bundles first.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from pydantic import Field, field_validator

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.depth import Depth, Precision, precision
from threetears.evals.analysis.kinds import CampaignKind, campaign_kind
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.report.decision import (
    FloorOutcome,
    FloorReadings,
    RateReading,
    SampleProjection,
    aspect_pass_rate,
    case_cost_usd,
    evaluate_floor,
    floor_readings,
    project_rate,
    read_floor,
)
from threetears.evals.analysis.stats import (
    INTERVAL_LEVEL,
    SIGNIFICANCE_ALPHA,
    SMALL_N_BAND_FLOOR,
    bounded_difference_interval,
    holm_adjust,
    pooled_rate_interval,
)
from threetears.evals.analysis.style import measure_name
from threetears.evals.kernel.analysis_measures import BarVerdict
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.declaration import CampaignArm, Floor, RateDrop
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.schema.base import EvalDocumentModel

__all__ = [
    "AspectReading",
    "BarState",
    "BarStatus",
    "CampaignKind",
    "CampaignReadings",
    "FloorState",
    "FloorStatus",
    "MeasureReading",
    "PairReading",
    "ShownReading",
    "campaign_kind",
    "campaign_readings",
]

#: What one aspect × candidate row shows against the control. ``better``: the gain is shown, its Holm-adjusted p
#: under the campaign's α. ``worse``: a drop past the aspect's margin is shown, likewise. ``within_margin``: neither,
#: and the drop is shown to stay within the margin, at the nominal level as a floor reads (the page says "no worse
#: than the margin"). ``unclear``: none of these. Neither ``worse`` nor ``within_margin`` is read on an aspect with no
#: margin.
ShownReading = Literal["better", "worse", "within_margin", "unclear"]

#: Where one floor stands for one candidate. ``broken``: the evidence shows the candidate past the floor. ``held``: it
#: shows the candidate within it. ``unclear``: neither, the reason saying why.
FloorState = Literal["broken", "held", "unclear"]

#: Where one arm stands against one absolute bar. ``met``: its whole interval is on the good side of the bar, less
#: the measure's margin. ``missed``: on the bad side. ``unclear``: neither, or nothing to read; the reason says which.
BarState = Literal["met", "missed", "unclear"]


class AspectReading(EvalDocumentModel):
    """One candidate on one aspect against the control: the change in pass rate, how sure, and what it shows.

    Every figure is the candidate minus the control, in points of a 0-100 pass rate, so a positive change is a
    candidate passing more.

    **The reading is decided per direction.** The gain and the drop are two Holm families per candidate, so each
    direction has its own p values and its own interval, read at the level that direction's step read it at: ``better`` is the
    gain shown (its interval wholly above zero), ``worse`` the drop past the margin shown (its interval wholly below
    minus the margin). A row that is neither reads ``within_margin`` where its interval at the nominal level, the
    floor's own and uncorrected, lies wholly above minus the margin (``within_interval``): the candidate is shown no
    worse than the control by more than the margin. ``unclear`` is none of these, and a page showing one shows the
    intervals, so a reader sees how far each side is from settling.
    """

    aspect: str = Field(description="The aspect's label.")
    arm: str = Field(description="The candidate's variant key.")
    arm_label: str = Field(description="What a reader calls the candidate: its arm's label, else its variant's name.")
    change: float = Field(
        description=(
            "The candidate's pass rate minus the control's, in points (positive = better): the mean of the per-case "
            "shares its intervals read, so it always lies in them; on the trial counts only where no interval was read."
        )
    )
    rig: str | None = Field(
        default=None,
        description=(
            "The test setup (apparatus class) whose per-case shares `change` and the intervals read, where both arms "
            "ran under more than one; None where they share one, or none. The counts are over every rig."
        ),
    )
    gain_interval: tuple[float, float] | None = Field(
        description=(
            "The change's interval in points, low then high, at `gain_confidence`: what `better` is read on. None "
            "when the arms' cases cannot bound it."
        )
    )
    gain_confidence: float | None = Field(
        description=(
            "The level `gain_interval` is read at: the level the gains' step-down showed the row at, or, not shown, "
            "a level at which its interval does not clear, so the interval and the gain reading never contradict. "
            "None with no interval."
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
            "The level `drop_interval` is read at, by the rule `gain_confidence` is, in the drops' family. None with "
            "no interval."
        )
    )
    gain_p: float | None = Field(
        description=(
            "The gain's p, unadjusted, on its family's grid: `α / k` for the largest `k` at which the gain is shown at "
            "`1 − α/k` and every looser level (its interval above zero, at the case floor, past the lost trials), 1 "
            "where it is shown at none. None when untested."
        )
    )
    gain_p_holm: float | None = Field(
        description=(
            "`gain_p` Holm-adjusted across this candidate's aspects: at most α exactly where the row reads `better` "
            "(to rounding). None when untested."
        )
    )
    within_interval: tuple[float, float] | None = Field(
        default=None,
        description=(
            "The change's interval in points, low then high, at the nominal level (`stats.INTERVAL_LEVEL`) with no "
            "correction, as a floor reads it: what `within_margin` is read on, and what an unclear row's "
            "`projection` counts from. None when the arms' cases cannot bound it, or the aspect declares no margin."
        ),
    )
    drop_p: float | None = Field(
        description=(
            "The drop's p past the margin, unadjusted, on its family's grid as `gain_p` is. None when untested, "
            "which an aspect with no margin always is."
        )
    )
    drop_p_holm: float | None = Field(
        description=(
            "`drop_p` Holm-adjusted across this candidate's aspects: at most α exactly where the row reads `worse`. "
            "None when untested."
        )
    )
    reading: ShownReading = Field(
        description="What the row shows: `better`, `worse`, `within_margin` (no worse than the margin) or `unclear`."
    )
    margin: float | None = Field(
        description=(
            "The aspect's margin in points, the drop tested against. None = none declared: the row is never `worse` "
            "and never `within_margin`."
        )
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
            "control by more than the aspect's margin (read `within_margin`), counted at the nominal level from "
            "`within_interval`, as a floor's are. None for a settled row, for an aspect with no margin, or where "
            "nothing can be projected. No gain is projected."
        )
    )
    reason: str = Field(
        default="",
        description=(
            "Why the row reads as it does, in one plain sentence naming the arms: what its interval showed, or what "
            "kept it unclear (too few cases, lost trials, an interval reaching both sides)."
        ),
    )
    confirmatory: bool = Field(
        default=False,
        description=(
            "Whether the reading confirms anything: the campaign asks this aspect a question (the bundle's "
            "`question_declared`), whenever it was asked. False reads it as a lead, which the page says."
        ),
    )
    control_depth: Depth | None = Field(
        default=None,
        description="How deep the control was measured on the aspect, over every rig (`ArmScoreboard.depth`).",
    )
    arm_depth: Depth | None = Field(
        default=None, description="How deep the candidate was measured on the aspect, over every rig."
    )
    depth_differs: int | None = Field(
        default=None,
        description=(
            "Of the paired cases the intervals read, those the two arms ran a different number of times. None where "
            "the cases are unpaired or no trials per case were carried."
        ),
    )
    precision: Precision | None = Field(
        default=None,
        description=(
            "What limits the change's interval at the nominal level: each arm's own half-width, the arm limiting it, "
            "and the between-case share of its variance that extra repeats cannot lower. None with no interval."
        ),
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
    margin: float | None = Field(
        default=None,
        description="A rate floor's margin, in points: the most the candidate's rate may drop. None for a cost floor.",
    )
    worst_drop: float | None = Field(
        default=None,
        description=(
            "The largest drop in points the deciding rate's interval allows: the worst case the floor is read against. "
            "None for a cost floor, or with no interval."
        ),
    )
    short_by: float | None = Field(
        default=None,
        description=(
            "For an unclear floor only: how far the worst case lies past the margin, in points (`worst_drop` less "
            "`margin`), which is how far the floor is from being shown held. None for a held floor, for a broken one "
            "(its breach is the interval and reason it carries, not this figure: `worst_drop` is the far end of the "
            "interval, not how far the floor is broken), where the worst case lies within the margin, and for a cost "
            "floor. A floor speaks for the one rate it read, never its whole aspect."
        ),
    )


class PairReading(EvalDocumentModel):
    """Two arms of a choice on one aspect: the difference between them, how sure, and which is shown ahead.

    A pair is direction-free: ``first`` and ``second`` are in the campaign's arm order, and nothing about which is
    first is a claim. Every figure is ``first`` minus ``second`` in points of a 0-100 pass rate.
    """

    aspect: str = Field(description="The aspect's label.")
    first: str = Field(description="The first arm's variant key, in the campaign's arm order.")
    first_label: str = Field(description="What a reader calls the first arm.")
    second: str = Field(description="The second arm's variant key.")
    second_label: str = Field(description="What a reader calls the second arm.")
    change: float = Field(
        description=(
            "The first arm's pass rate minus the second's, in points: the mean of the per-case shares `interval` "
            "reads, so it always lies in it; on the trial counts only where no interval was read."
        )
    )
    rig: str | None = Field(
        default=None,
        description=(
            "The test setup whose per-case shares `change` and `interval` read, where both arms ran under more than "
            "one; None where they share one, or none. The counts are over every rig."
        ),
    )
    interval: tuple[float, float] | None = Field(
        description="The difference's two-sided bounded interval in points at `confidence`; None when unbounded."
    )
    confidence: float | None = Field(
        description="The level `interval` is read at, by the step-down over every pair × aspect. None with none."
    )
    p: float | None = Field(description="The pair's p on its family's grid, unadjusted. None when untested.")
    p_holm: float | None = Field(
        description="`p` Holm-adjusted across every pair × aspect: at most α exactly where an arm is ahead."
    )
    ahead: str | None = Field(
        description=(
            "The variant key of the arm shown ahead (`first` or `second`): the interval excludes zero, every arm at "
            "the case floor, and past the lost trials. None = no separation. A reading, never a pick."
        )
    )
    ahead_label: str | None = Field(
        default=None, description="What a reader calls the arm shown ahead; None with no separation."
    )
    reason: str = Field(default="", description="Why, in one plain sentence naming the arms.")
    confirmatory: bool = Field(
        default=False, description="Whether the campaign asks this aspect a question, whenever it was asked."
    )
    first_cases: int = Field(ge=0, description="The first arm's usable cases.")
    second_cases: int = Field(ge=0, description="The second arm's usable cases.")
    first_passed: int = Field(ge=0, description="Trials the first arm passed.")
    first_trials: int = Field(ge=0, description="Trials the first arm was measured on.")
    second_passed: int = Field(ge=0, description="Trials the second arm passed.")
    second_trials: int = Field(ge=0, description="Trials the second arm was measured on.")
    first_depth: Depth | None = Field(
        default=None, description="How deep the first arm was measured on the aspect, over every rig."
    )
    second_depth: Depth | None = Field(
        default=None, description="How deep the second arm was measured on the aspect, over every rig."
    )
    depth_differs: int | None = Field(
        default=None,
        description=(
            "Of the paired cases the interval reads, those the two arms ran a different number of times. None where "
            "the cases are unpaired or no trials per case were carried."
        ),
    )
    precision: Precision | None = Field(
        default=None,
        description=(
            "What limits the interval at the nominal level: each arm's own half-width (by variant key), the arm "
            "limiting it, and the between-case share of its variance. None with no interval."
        ),
    )


class BarStatus(EvalDocumentModel):
    """One arm against one absolute bar on one aspect: met, missed or unclear, as the bundle adjudicated it."""

    aspect: str = Field(description="The aspect's label.")
    arm: str = Field(description="The arm's variant key.")
    arm_label: str = Field(description="What a reader calls the arm.")
    measure: str = Field(description="What the bar is read on, as the bar names it.")
    bar: float = Field(description="The bar's threshold, in the measure's own units.")
    direction: Literal["higher_is_better", "lower_is_better"] = Field(description="Which way clearing it runs.")
    status: BarState = Field(description="`met`, `missed` or `unclear`.")
    value: float | None = Field(description="The arm's value on the measure; None with no observation.")
    interval: tuple[float, float] | None = Field(
        description="The interval the status was read on, in the measure's units; None below two observations."
    )
    margin: float | None = Field(description="The measure's declared margin the bar was read with; None with none.")
    reason: str = Field(default="", description="Why, in one plain sentence naming the arm.")


class MeasureReading(EvalDocumentModel):
    """One arm's value on one measure on one aspect, with its interval: what a measurement reports."""

    aspect: str = Field(description="The aspect's label.")
    arm: str = Field(description="The arm's variant key.")
    arm_label: str = Field(description="What a reader calls the arm.")
    measure: str = Field(description="The measure's key; `passed` for the trials' own pass rate.")
    name: str = Field(description="The measure as a reader names it.")
    rig: str | None = Field(
        default=None,
        description=(
            "The test setup the value was measured under: a measure is read per cell, an arm under each rig apart. "
            "None for the trials' pass rate, which is counted over every rig."
        ),
    )
    value: float | None = Field(description="The arm's value: a mean, or a share for a rate. None with none.")
    interval: tuple[float, float] | None = Field(
        description=(
            "The value's interval: the cell summary's own, or for the pass rate the Wilson interval on its count with "
            "the cases as the independent draws (`stats.pooled_rate_interval`). None below two cases."
        )
    )
    passed: int | None = Field(default=None, description="For the pass rate, the trials passed; None otherwise.")
    trials: int | None = Field(default=None, description="For the pass rate, the usable trials; None otherwise.")
    cases: int = Field(ge=0, description="The independent cases behind the value.")
    higher_is_better: bool | None = Field(default=None, description="Which way is better; None where undeclared.")


class CampaignReadings(EvalDocumentModel):
    """Everything code reads off a campaign, by its kind: for a change, one row per aspect × candidate and one status
    per floor × candidate; for a choice, one reading per pair of arms per aspect; for a measurement, each arm's
    measures; for every kind, each arm against each absolute bar.

    Each candidate's rows are corrected across its own aspects (Holm, gains and drops apart, each direction erring at
    most α/2); nothing corrects across candidates, so with several, one is shown by chance more often than α. A row
    no worse than the margin is read at the nominal level with no correction, as a floor is.
    """

    campaign_id: str = Field(description="The campaign read.")
    alpha: float = Field(
        gt=0.0, lt=1.0, description="The campaign's α, at which each candidate's two Holm families are tested."
    )
    control: str | None = Field(description="The control arm's variant key; None when the campaign declares none.")
    control_label: str = Field(description="What a reader calls the control; empty when there is none.")
    aspects: list[str] = Field(description="The aspects' labels, in the campaign's order.")
    readings: list[AspectReading] = Field(description="One row per aspect × candidate, aspect by aspect.")
    floors: list[FloorStatus] = Field(description="One status per floor × candidate, in the floors' order.")
    kind: CampaignKind = Field(
        default="change",
        description="What the campaign is, from its arms (`campaign_kind`): which of the lists below code fills.",
    )
    pairs: list[PairReading] = Field(
        default_factory=list, description="A choice's pairs, aspect by aspect, each pair once. Empty otherwise."
    )
    bars: list[BarStatus] = Field(
        default_factory=list, description="Every arm against every absolute bar, aspect by aspect, for every kind."
    )
    measures: list[MeasureReading] = Field(
        default_factory=list, description="A measurement's measures, aspect by aspect. Empty otherwise."
    )

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

    def missed_bars(self) -> list[BarStatus]:
        """Every bar an arm misses, in order: what the page opens with, as it does a broken floor.

        :return: The missed statuses.
        :rtype: list[BarStatus]
        """
        return [status for status in self.bars if status.status == "missed"]

    def broken_floors(self) -> list[FloorStatus]:
        """Every floor a candidate breaks, in order: what the page opens with.

        :return: The broken statuses.
        :rtype: list[FloorStatus]
        """
        return [status for status in self.floors if status.status == "broken"]

    def arm_labels(self) -> dict[str, str]:
        """What a reader calls each arm the readings name, by variant key, for every kind: the control, every candidate
        with a row, both arms of every pair, every arm measured or held to a bar. What an aspect's own page is built
        with (``build_code_only_report(..., arm_labels=...)``), so a fold names the arms as the campaign's table does.

        :return: Variant key → label.
        :rtype: dict[str, str]
        """
        labels = {self.control: self.control_label} if self.control is not None else {}
        labels.update({row.arm: row.arm_label for row in self.readings})
        for pair in self.pairs:
            labels.update({pair.first: pair.first_label, pair.second: pair.second_label})
        labels.update({measure.arm: measure.arm_label for measure in self.measures})
        labels.update({status.arm: status.arm_label for status in self.bars})
        return labels

    def on_aspect(self, aspect: str) -> list[AspectReading]:
        """The rows read on one aspect, in candidate order.

        :param aspect: The aspect's label.
        :ptype aspect: str
        :return: Its rows; empty for a label the campaign does not hold.
        :rtype: list[AspectReading]
        """
        return [reading for reading in self.readings if reading.aspect == aspect]


# =============================================================================
# Reading a campaign
# =============================================================================

#: The fewest usable cases on every arm before a row can be shown either way: the floors' case floor.
_CASE_FLOOR = SMALL_N_BAND_FLOOR

#: The share range the pass rate's per-case values lie in.
_RATE_RANGE = (0.0, 1.0)


@dataclass
class _Row:
    """One aspect × candidate row while it is read: its evidence, and its pass rate at each level it was read at."""

    aspect: str
    margin: float | None
    control_key: str
    control_label: str
    arm: CampaignArm
    base: FloorReadings
    rate: RateReading
    confirmatory: bool
    per_case_usd: float | None
    rigs: list[str]
    levels: dict[float, RateReading]
    drops: dict[float, tuple[FloorOutcome, str]]
    depths: Mapping[str, Depth | None]

    def at(self, confidence: float) -> RateReading:
        """The pass rate with its bounded interval at ``confidence``, computed again from the same per-case values a
        floor reads, by the same function, once per level; the nominal level's is the floor's own reading."""
        if confidence == self.rate.confidence or self.rate.samples is None:
            return self.rate
        if confidence not in self.levels:
            control, contrast = self.rate.samples
            interval = bounded_difference_interval(
                control, contrast, paired=self.rate.paired, value_range=_RATE_RANGE, confidence=confidence
            )
            self.levels[confidence] = replace(self.rate, interval=interval, confidence=confidence)
        return self.levels[confidence]

    def fewest(self) -> int | None:
        """The fewer usable cases of the two arms; None when either was not counted."""
        return self.base.fewest_cases()

    def gain_shown(self, confidence: float) -> bool:
        """Whether the gain is shown at ``confidence``: every arm at the case floor, the interval wholly above zero,
        and still a gain with every lost trial given its worst outcome for the candidate (the floor's rules, mirrored)."""
        rate = self.at(confidence)
        fewest = self.fewest()
        if rate.interval is None or fewest is None or fewest < _CASE_FLOOR:
            return False
        return rate.interval[0] > 0 and rate.drop_bounds()[1] < 0

    def drop_verdict(self, confidence: float) -> tuple[FloorOutcome, str]:
        """The drop against the margin at ``confidence``, and the reason: the floor's own reading
        (:func:`~threetears.evals.analysis.report.decision.read_floor`) on the pass rate alone, at that level.
        ``broken`` is the drop shown past the margin, ``held`` the drop shown within it, ``unclear`` neither; with no
        margin, ``unclear`` with no reason."""
        if self.margin is None:
            return "unclear", ""
        if confidence not in self.drops:
            rate = self.at(confidence)
            readings = replace(self.base, rates=(rate,), confidence=confidence)
            floor = Floor(aspect=self.aspect, clause=RateDrop(max_points=self.margin))
            verdict = read_floor(floor, readings, control=self.control_label, candidate=self.arm.label)
            self.drops[confidence] = (verdict.outcome, verdict.reason)
        return self.drops[confidence]


def _high(rate: RateReading) -> float:
    """The interval's high end as a share, or infinity with none."""
    return rate.interval[1] if rate.interval is not None else math.inf


@dataclass(frozen=True)
class _Decision:
    """One row in one direction's family: whether it is shown, the level its interval is read at, and its p."""

    shown: bool
    level: float
    p: float


def _step_down(rows: Sequence[int], shown: Callable[[int, float], bool], alpha: float) -> dict[int, _Decision]:
    """Holm's step-down on one family, read on the intervals and made monotone by construction.

    The family of ``m`` rows is read at the levels ``1 − α/k``, ``k`` from 1 to ``m``, loosest first. A row's
    **reach** is the largest ``k`` at which it is shown at that level and at every looser one, so whether a row
    counts as shown never rises as the level tightens, whatever the interval does between levels. With ``k`` rows
    not yet shown, a step shows every one whose reach is at least ``k``; they leave, and the next step reads the rest
    at their new ``k``; a step showing none ends it. This is Holm's procedure exactly on the p values ``α / reach``
    (1 where the reach is 0): a row is shown exactly where its Holm-adjusted p is at most ``α``. Each such p is valid
    (a row reaches ``k`` only where its interval at ``1 − α/k`` clears), so the family's error is Holm's.

    A shown row's interval is read at the level its step showed it at; one not shown at the last step's level, or,
    where its interval clears there but not at some looser level, at the loosest level it does not clear, so its
    interval never says it was shown.

    :return: Each row's decision.
    :rtype: dict[int, _Decision]
    """
    size = len(rows)
    grid = [1.0 - alpha / k for k in range(1, size + 1)]
    reach: dict[int, int] = {}
    for index in rows:
        reached = 0
        for k, level in enumerate(grid, start=1):
            if not shown(index, level):
                break
            reached = k
        reach[index] = reached
    decided: dict[int, _Decision] = {}
    remaining = list(rows)
    while remaining:
        k = len(remaining)
        hits = [index for index in remaining if reach[index] >= k]
        if not hits:
            for index in remaining:
                level = grid[k - 1] if not shown(index, grid[k - 1]) else grid[reach[index]]
                decided[index] = _Decision(shown=False, level=level, p=_p(alpha, reach[index]))
            break
        for index in hits:
            decided[index] = _Decision(shown=True, level=grid[k - 1], p=_p(alpha, reach[index]))
        remaining = [index for index in remaining if index not in hits]
    return decided


def _p(alpha: float, reach: int) -> float:
    """A row's p on its family's grid: ``α / reach``, 1 for a row shown at no level."""
    return alpha / reach if reach else 1.0


def _families(rows: Sequence[_Row], keep: Callable[[_Row], bool]) -> list[list[int]]:
    """One family per candidate, in candidate order: the rows of that candidate ``keep`` admits, across its aspects."""
    by_arm: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        if keep(row):
            by_arm.setdefault(row.arm.variant_key, []).append(index)
    return list(by_arm.values())


def _decided(
    rows: Sequence[_Row], keep: Callable[[_Row], bool], shown: Callable[[int, float], bool], alpha: float
) -> dict[int, _Decision]:
    """Every family's step-down, each candidate's apart."""
    decided: dict[int, _Decision] = {}
    for family in _families(rows, keep):
        decided.update(_step_down(family, shown, alpha))
    return decided


def _bundle_control(bundle: AnalysisContextBundle) -> str | None:
    """The control's variant key in one aspect's evidence, when it ran there."""
    arm = bundle.design.control_arm
    key = arm.variant_key if arm is not None else None
    return key if any(board.variant_key == key for board in bundle.scoreboard) else None


def _checked_bundles(
    campaign: EvalCampaign, bundles: Mapping[str, AnalysisContextBundle]
) -> dict[str, AnalysisContextBundle]:
    """Every aspect's bundle, refused unless there is exactly one per aspect and each reads the aspect it is keyed by."""
    labels = [aspect.label for aspect in campaign.aspects]
    if missing := [label for label in labels if label not in bundles]:
        raise RefusedInput(
            f"campaign {campaign.id} has no evidence bundle for aspect(s) {', '.join(missing)}",
            remedy="pass one bundle per aspect, each assembled for that aspect and keyed by its label",
        )
    if stray := sorted(set(bundles) - set(labels)):
        raise RefusedInput(
            f"campaign {campaign.id} holds no aspect labelled {', '.join(stray)}",
            remedy="key each bundle by the label of the aspect it was assembled for",
        )
    if wrong := [label for label in labels if bundles[label].aspect != label]:
        raise RefusedInput(
            "the bundle keyed by " + ", ".join(f"{label} reads aspect {bundles[label].aspect}" for label in wrong),
            remedy="key each bundle by the label of the aspect it was assembled for",
        )
    return {label: bundles[label] for label in labels}


def _distinct(arms: Sequence[CampaignArm]) -> list[CampaignArm]:
    """The arms, each once by variant key, the first naming it kept, in the order given."""
    seen: set[str] = set()
    found: list[CampaignArm] = []
    for arm in arms:
        if arm.variant_key not in seen:
            seen.add(arm.variant_key)
            found.append(arm)
    return found


def _candidates(campaign: EvalCampaign, arms: Sequence[CampaignArm]) -> list[CampaignArm]:
    """The candidate arms, each once by variant key, in the order given; never the control."""
    control = campaign.control
    seen = {control.variant_key} if control is not None else set()
    found: list[CampaignArm] = []
    for arm in arms:
        if arm.role == "candidate" and arm.variant_key not in seen:
            seen.add(arm.variant_key)
            found.append(arm)
    return found


def _shared_rigs(bundle: AnalysisContextBundle, first: str, second: str) -> list[str]:
    """The test setups both arms ran under on the aspect, in the bundle's order of cells."""
    theirs = {cell.apparatus_class_id for cell in bundle.cells if cell.variant_key == second}
    return [
        rig
        for rig in dict.fromkeys(c.apparatus_class_id for c in bundle.cells if c.variant_key == first)
        if rig in theirs
    ]


def _change(rate: RateReading) -> float:
    """The second arm's pass rate minus the first's (the candidate's minus the control's), in points: the mean of the
    per-case shares the interval reads, so the change always lies in its interval; on the trial counts only where the
    rate carries no per-case shares."""
    if rate.samples is None:
        return -rate.point_drop()
    control, contrast = rate.samples
    return 100 * (math.fsum(contrast) / len(contrast) - math.fsum(control) / len(control))


def _rig_read(rate: RateReading, rigs: Sequence[str]) -> str | None:
    """The rig a rate's shares were read under, said only where the arms shared more than one."""
    return rate.rig if len(rigs) > 1 else None


def _no_interval(first: str, second: str, rigs: Sequence[str]) -> str:
    """Why no interval was read: no rig both arms ran under, or too few cases with a usable trial on the one read."""
    if not rigs:
        return f"no interval could be read: {first} and {second} ran under no test setup in common"
    return "no interval could be read: a side has fewer than two cases with a usable trial"


def _percent(level: float) -> str:
    """A level as a percent, to the precision that tells two levels apart: ``95%``, ``98.33%``."""
    return format_number(round(100 * level, 2)) + "%"


def _reason(
    row: _Row, reading: ShownReading, gain_level: float, drop_level: float, drop_reason: str, within_reason: str
) -> str:
    """Why a row reads as it does, in one sentence naming the arms: for ``worse`` the drop family's reason at the
    row's level, for ``within_margin`` the floor's reason at the nominal level, and for an unclear row with a margin
    whose interval reaches both sides, the floor's reason for what keeps the drop from being shown within it."""
    arm, control = row.arm.label, row.control_label
    if reading == "better":
        return f"{arm}'s gain over {control} is shown: its whole interval is above zero"
    if reading == "worse":
        return drop_reason
    if reading == "within_margin":
        nominal, corrected = row.at(INTERVAL_LEVEL), row.at(gain_level)
        if corrected.interval is not None and corrected.interval[0] > 0:
            # Its gain clears zero at the level its family read it at: only the lost trials keep it from better.
            return f"{within_reason}; its gain clears zero, but the lost trials could still take it back"
        if nominal.interval is not None and nominal.interval[0] > 0:
            return (
                f"{within_reason}; its gain clears zero at {_percent(INTERVAL_LEVEL)}, not at the "
                f"{_percent(gain_level)} its gains are read at, corrected across {arm}'s aspects"
            )
        return within_reason
    fewest = row.fewest()
    rate = row.at(gain_level)
    if rate.interval is None:
        return _no_interval(arm, control, row.rigs)
    if fewest is None:
        return "the usable cases were not counted on both arms"
    if fewest < _CASE_FLOOR:
        return f"{fewest} cases are too few: an interval is read only from {_CASE_FLOOR} independent cases"
    if rate.interval[0] > 0:
        return f"{arm}'s gain clears zero, but the lost trials could still take it back"
    if row.margin is None:
        if rate.interval[1] < 0:
            return f"{arm}'s drop is shown on its interval, but no margin was declared, so it is never shown worse"
        return "its interval reaches zero, and no margin was declared to show a drop against"
    points = format_number(row.margin)
    if _high(row.at(drop_level)) < -row.margin / 100:
        return f"{arm}'s drop is past the margin on its interval, but the lost trials could still bring it within"
    if _high(row.at(drop_level)) < 0:
        return f"{arm}'s drop is shown, but not past the {points}-point margin on its whole interval"
    if within_reason:
        return f"its interval reaches zero, and {within_reason}"
    return f"its interval reaches zero and does not lie past the {points}-point margin"


def campaign_readings(
    campaign: EvalCampaign,
    bundles: Mapping[str, AnalysisContextBundle],
    *,
    arms: Sequence[CampaignArm],
) -> CampaignReadings:
    """What code reads off a campaign, by its kind (:func:`campaign_kind`, over ``arms``).

    **A change.** Per aspect and candidate, the aspect's pass rate
    (:func:`~threetears.evals.analysis.report.decision.aspect_pass_rate`): the change in points, shown better, shown
    worse or unclear, read by the step-down on intervals the module describes, Holm across each candidate's aspects
    at :data:`~threetears.evals.analysis.stats.SIGNIFICANCE_ALPHA`, gains and drops apart; nothing corrects across
    candidates. An aspect with no margin
    is never shown worse and tests no drop. A row neither shown better nor shown worse reads ``within_margin`` where
    the floor's own reading of its pass rate against the margin, at the nominal level, is ``held``. An unclear row
    with a margin carries the cases that would show the candidate within it
    (:func:`~threetears.evals.analysis.report.decision.project_rate`, at the nominal level, as a floor's are).
    Per floor and candidate, :func:`~threetears.evals.analysis.report.decision.evaluate_floor`, at the nominal level
    with no correction. A row is confirmatory where the campaign asks that aspect a question (the bundle's
    ``question_declared``), whenever it was asked: a criterion set after the aspect's readings were visible reads as
    one set before them, and the bundle discloses it (``reading_scope.late_criteria``).

    A candidate that did not run on an aspect, or an aspect its control did not run on, has no row there. Each row's
    change and intervals are read on one rig's per-case shares, named where both arms ran under several.

    **A choice** (no control, two or more arms) reads no row and no floor: every pair of arms on every aspect both ran
    on, the difference and the arm shown ahead, by one step-down over every pair × aspect. **A measurement** (one arm,
    a control alone among them) reads no row, pair or floor: each aspect's measures for its arm. **Every kind** reads
    each arm against each absolute bar the bundle adjudicated.

    :param campaign: The campaign.
    :ptype campaign: EvalCampaign
    :param bundles: Each aspect's evidence, keyed by its label: one per aspect, each assembled for it.
    :ptype bundles: Mapping[str, AnalysisContextBundle]
    :param arms: The campaign's arms as read (:func:`~threetears.evals.analysis.campaigns.campaign_arms`: the named,
        then the derived candidates). Required: a change whose candidates are derived from its runs names only its
        control, so the named arms alone would read it as a measurement.
    :ptype arms: Sequence[CampaignArm]
    :return: The readings.
    :rtype: CampaignReadings
    :raises RefusedInput: An aspect has no bundle, a bundle is keyed by a label the campaign does not hold, or a bundle
        reads another aspect than the one it is keyed by.
    """
    checked = _checked_bundles(campaign, bundles)
    every = _distinct(arms)
    kind = campaign_kind(every)
    alpha = SIGNIFICANCE_ALPHA
    labels = [aspect.label for aspect in campaign.aspects]
    bars = _bar_statuses(campaign, checked, every)
    control = campaign.control or next((arm for arm in every if arm.role == "control"), None)
    if kind != "change" or control is None:
        return CampaignReadings(
            campaign_id=campaign.id,
            alpha=alpha,
            control=None,
            control_label="",
            aspects=labels,
            readings=[],
            floors=[],
            kind=kind,
            pairs=_pair_readings(campaign, checked, every, alpha) if kind == "choice" else [],
            bars=bars,
            measures=_measure_readings(campaign, checked, every) if kind == "measurement" else [],
        )
    candidates = _candidates(campaign, every)
    rows: list[_Row] = []
    for aspect in campaign.aspects:
        bundle = checked[aspect.label]
        control_key = _bundle_control(bundle)
        if control_key is None:
            continue
        per_case = case_cost_usd(bundle)
        for arm in candidates:
            if arm.variant_key == control_key:
                continue
            rate = aspect_pass_rate(bundle, control_key, arm.variant_key)
            if rate is None:
                continue
            rows.append(
                _Row(
                    aspect=aspect.label,
                    margin=aspect.margin,
                    control_key=control_key,
                    control_label=control.label,
                    arm=arm,
                    base=floor_readings(bundle, control_key, arm.variant_key),
                    rate=rate,
                    confirmatory=bundle.question_declared,
                    per_case_usd=per_case,
                    rigs=_shared_rigs(bundle, control_key, arm.variant_key),
                    levels={},
                    drops={},
                    depths=_depths(bundle),
                )
            )

    def gain_kept(row: _Row) -> bool:
        return row.rate.interval is not None

    def drop_kept(row: _Row) -> bool:
        return row.rate.interval is not None and row.margin is not None

    gains = _decided(rows, gain_kept, lambda index, level: rows[index].gain_shown(level), alpha)
    drops = _decided(rows, drop_kept, lambda index, level: rows[index].drop_verdict(level)[0] == "broken", alpha)
    gain_holm = _holm(rows, gain_kept, gains, alpha)
    drop_holm = _holm(rows, drop_kept, drops, alpha)
    readings: list[AspectReading] = []
    for index, row in enumerate(rows):
        gain, drop = gains.get(index), drops.get(index)
        gained, dropped = gain is not None and gain.shown, drop is not None and drop.shown
        # The nominal level's reading is the floor's own: held is the drop shown within the margin, uncorrected.
        within, within_reason = row.drop_verdict(INTERVAL_LEVEL) if drop_kept(row) else ("unclear", "")
        # A gain and a drop past the margin on one estimate's two intervals would contradict each other; should the
        # two ever both be shown, neither is claimed.
        reading: ShownReading
        if gained != dropped:
            reading = "better" if gained else "worse"
        elif not gained and within == "held":
            reading = "within_margin"
        else:
            reading = "unclear"
        gain_level = gain.level if gain is not None else None
        drop_level = drop.level if drop is not None else None
        drop_reason = row.drop_verdict(drop_level)[1] if drop_level is not None else ""
        gain_rate = row.at(gain_level) if gain_level is not None else None
        drop_rate = row.at(drop_level) if drop_level is not None else None
        within_rate = row.at(INTERVAL_LEVEL) if drop_kept(row) else None
        fewest = row.fewest()
        projection = (
            project_rate(within_rate, cases=fewest, margin=row.margin, per_case_usd=row.per_case_usd)
            if reading == "unclear" and within_rate is not None and row.margin is not None and fewest
            else None
        )
        readings.append(
            AspectReading(
                aspect=row.aspect,
                arm=row.arm.variant_key,
                arm_label=row.arm.label,
                change=_change(row.rate),
                rig=_rig_read(row.rate, row.rigs),
                gain_interval=_points(gain_rate),
                gain_confidence=gain_level,
                drop_interval=_points(drop_rate),
                drop_confidence=drop_level,
                within_interval=_points(within_rate),
                gain_p=gain.p if gain is not None else None,
                gain_p_holm=gain_holm.get(index),
                drop_p=drop.p if drop is not None else None,
                drop_p_holm=drop_holm.get(index),
                reading=reading,
                margin=row.margin,
                control_cases=row.base.control_cases or 0,
                arm_cases=row.base.candidate_cases or 0,
                control_passed=row.rate.control_passed,
                control_trials=row.rate.control_n,
                arm_passed=row.rate.candidate_passed,
                arm_trials=row.rate.candidate_n,
                projection=projection,
                reason=_reason(
                    row,
                    reading,
                    gain_level or row.rate.confidence or INTERVAL_LEVEL,
                    drop_level or INTERVAL_LEVEL,
                    drop_reason,
                    within_reason,
                ),
                confirmatory=row.confirmatory,
                control_depth=row.depths.get(row.control_key),
                arm_depth=row.depths.get(row.arm.variant_key),
                depth_differs=_depth_differs(row.rate),
                precision=_precision(row.rate, (row.control_key, row.arm.variant_key)),
            )
        )
    floors = [
        evaluate_floor(floor, row.base, arm=row.arm.variant_key, arm_label=row.arm.label, control_label=control.label)
        for floor in campaign.floors
        for row in rows
        if row.aspect == floor.aspect
    ]
    return CampaignReadings(
        campaign_id=campaign.id,
        alpha=alpha,
        control=control.variant_key,
        control_label=control.label,
        aspects=labels,
        readings=readings,
        floors=floors,
        kind=kind,
        bars=bars,
    )


# =============================================================================
# A choice, its bars and a measurement
# =============================================================================


@dataclass
class _Pair:
    """One pair of a choice on one aspect while it is read: the two arms in the campaign's order, and the pass rate
    as the bundle carries it, the arm first in variant-key order its ``control``."""

    aspect: str
    first: CampaignArm
    second: CampaignArm
    flipped: bool
    base: FloorReadings
    rate: RateReading
    confirmatory: bool
    rigs: list[str]
    levels: dict[float, RateReading]
    depths: Mapping[str, Depth | None]

    def at(self, confidence: float) -> RateReading:
        """The pass rate's bounded interval at ``confidence``, computed once per level."""
        if confidence == self.rate.confidence or self.rate.samples is None:
            return self.rate
        if confidence not in self.levels:
            control, contrast = self.rate.samples
            interval = bounded_difference_interval(
                control, contrast, paired=self.rate.paired, value_range=_RATE_RANGE, confidence=confidence
            )
            self.levels[confidence] = replace(self.rate, interval=interval, confidence=confidence)
        return self.levels[confidence]

    def leader(self, confidence: float) -> CampaignArm | None:
        """The arm shown ahead at ``confidence``, by the rules a candidate's gain is read by, either way round: every
        arm at the case floor, the interval wholly one side of zero, and still so with every lost trial given its
        worst outcome for the arm ahead. None where neither is."""
        rate = self.at(confidence)
        fewest = self.base.fewest_cases()
        if rate.interval is None or fewest is None or fewest < _CASE_FLOOR:
            return None
        least, most = rate.drop_bounds()
        # The rate reads the arm second in key order minus the first.
        later, earlier = (self.first, self.second) if self.flipped else (self.second, self.first)
        if rate.interval[0] > 0 and most < 0:
            return later
        if rate.interval[1] < 0 and least > 0:
            return earlier
        return None


def _pair_readings(
    campaign: EvalCampaign, bundles: Mapping[str, AnalysisContextBundle], arms: Sequence[CampaignArm], alpha: float
) -> list[PairReading]:
    """Every pair of a choice's arms on every aspect both ran on: the difference, its interval and which arm is shown
    ahead, by one step-down over every pair × aspect (a pair has no direction, so one family, not two)."""
    pairs: list[_Pair] = []
    for aspect in campaign.aspects:
        bundle = bundles[aspect.label]
        ran = {board.variant_key for board in bundle.scoreboard}
        present = [arm for arm in arms if arm.variant_key in ran]
        for index, first in enumerate(present):
            for second in present[index + 1 :]:
                flipped = first.variant_key > second.variant_key
                low, high = (second, first) if flipped else (first, second)
                rate = aspect_pass_rate(bundle, low.variant_key, high.variant_key)
                if rate is None:
                    continue
                pairs.append(
                    _Pair(
                        aspect=aspect.label,
                        first=first,
                        second=second,
                        flipped=flipped,
                        base=floor_readings(bundle, low.variant_key, high.variant_key),
                        rate=rate,
                        confirmatory=bundle.question_declared,
                        rigs=_shared_rigs(bundle, low.variant_key, high.variant_key),
                        levels={},
                        depths=_depths(bundle),
                    )
                )
    family = [index for index, pair in enumerate(pairs) if pair.rate.interval is not None]
    decided = _step_down(family, lambda index, level: pairs[index].leader(level) is not None, alpha)
    adjusted = _snapped(family, holm_adjust([decided[index].p for index in family]), decided, alpha)
    read: list[PairReading] = []
    for index, pair in enumerate(pairs):
        decision = decided.get(index)
        rate = pair.at(decision.level) if decision is not None else pair.rate
        leader = pair.leader(decision.level) if decision is not None and decision.shown else None
        sign = -1.0 if pair.flipped else 1.0
        # The rate reads the arm second in key order minus the first; the pair reads its first arm minus its second.
        interval = (
            None
            if rate.interval is None
            else tuple(sorted((-sign * 100 * rate.interval[0], -sign * 100 * rate.interval[1])))
        )
        firsts = (rate.candidate_passed, rate.candidate_n) if pair.flipped else (rate.control_passed, rate.control_n)
        seconds = (rate.control_passed, rate.control_n) if pair.flipped else (rate.candidate_passed, rate.candidate_n)
        cases = (pair.base.control_cases or 0, pair.base.candidate_cases or 0)
        first_cases, second_cases = (cases[1], cases[0]) if pair.flipped else cases
        read.append(
            PairReading(
                aspect=pair.aspect,
                first=pair.first.variant_key,
                first_label=pair.first.label,
                second=pair.second.variant_key,
                second_label=pair.second.label,
                # The rate reads the arm second in key order minus the first.
                change=-sign * _change(rate),
                rig=_rig_read(rate, pair.rigs),
                interval=None if interval is None else (interval[0], interval[1]),
                confidence=decision.level if decision is not None else None,
                p=decision.p if decision is not None else None,
                p_holm=adjusted.get(index),
                ahead=leader.variant_key if leader is not None else None,
                ahead_label=leader.label if leader is not None else None,
                reason=_pair_reason(pair, leader, rate),
                confirmatory=pair.confirmatory,
                first_cases=first_cases,
                second_cases=second_cases,
                first_passed=firsts[0],
                first_trials=firsts[1],
                second_passed=seconds[0],
                second_trials=seconds[1],
                first_depth=pair.depths.get(pair.first.variant_key),
                second_depth=pair.depths.get(pair.second.variant_key),
                depth_differs=_depth_differs(pair.rate),
                # The rate reads the arm first in key order as its control.
                precision=_precision(pair.rate, tuple(sorted((pair.first.variant_key, pair.second.variant_key)))),
            )
        )
    return read


def _pair_reason(pair: _Pair, leader: CampaignArm | None, rate: RateReading) -> str:
    """Why a pair reads as it does, in one sentence naming both arms."""
    first, second = pair.first.label, pair.second.label
    if leader is not None:
        behind = second if leader is pair.first else first
        return f"{leader.label} is ahead of {behind}: the whole interval on the difference is one side of zero"
    fewest = pair.base.fewest_cases()
    if rate.interval is None:
        return _no_interval(first, second, pair.rigs)
    if fewest is not None and fewest < _CASE_FLOOR:
        return f"{fewest} cases are too few: an interval is read only from {_CASE_FLOOR} independent cases"
    if rate.interval[0] > 0 or rate.interval[1] < 0:
        return f"the interval between {first} and {second} clears zero, but the lost trials could still close it"
    return f"no separation: the interval between {first} and {second} reaches zero"


#: What each bar decision reads as, and why.
_BAR_STATES: dict[str, tuple[BarState, str]] = {
    "cleared": ("met", "its whole interval is on the good side of the bar"),
    "missed": ("missed", "its whole interval is on the bad side of the bar"),
    "undecided": ("unclear", "its interval straddles the bar"),
    "no_interval": ("unclear", "too few observations for an interval"),
    "no_data": ("unclear", "no observation of it"),
}


#: How a bar's reason says where an arm stands.
_BAR_VERBS: dict[BarState, str] = {"met": "meets", "missed": "misses", "unclear": "is unclear on"}


def _bar_statuses(
    campaign: EvalCampaign, bundles: Mapping[str, AnalysisContextBundle], arms: Sequence[CampaignArm]
) -> list[BarStatus]:
    """Every arm against every absolute bar on every aspect, as the bundle adjudicated it (``bar_adjudications``),
    never computed again. An arm read under several rigs misses a bar where any rig misses it, meets it where every
    rig meets it, and is otherwise unclear."""
    labels = {arm.variant_key: arm.label for arm in arms}
    statuses: list[BarStatus] = []
    for aspect in campaign.aspects:
        for bar in bundles[aspect.label].bar_adjudications:
            if bar.state != "adjudicated":
                continue
            by_arm: dict[str, list[BarVerdict]] = {}
            for verdict in bar.verdicts:
                by_arm.setdefault(verdict.variant_key, []).append(verdict)
            order = [arm.variant_key for arm in arms if arm.variant_key in by_arm]
            order += [key for key in by_arm if key not in order]
            descriptor = bundles[aspect.label].measure_catalog.get(bar.measure_id)
            heading = measure_name(bar.measure_id, descriptor.reader_name if descriptor is not None else None)
            for key in order:
                verdicts = by_arm[key]
                decisions = [verdict.decision for verdict in verdicts]
                decided = (
                    "missed"
                    if "missed" in decisions
                    else "cleared"
                    if all(decision == "cleared" for decision in decisions)
                    else next(decision for decision in decisions if decision != "cleared")
                )
                shown = next(verdict for verdict in verdicts if verdict.decision == decided)
                status, why = _BAR_STATES[decided]
                label = labels.get(key, f"arm {key[:12]}")
                statuses.append(
                    BarStatus(
                        aspect=aspect.label,
                        arm=key,
                        arm_label=label,
                        measure=bar.measure_id,
                        bar=bar.threshold,
                        direction=bar.direction,
                        status=status,
                        value=shown.value,
                        interval=(shown.ci_low, shown.ci_high)
                        if shown.ci_low is not None and shown.ci_high is not None
                        else None,
                        margin=shown.margin,
                        reason=f"{label} {_BAR_VERBS[status]} the bar on {heading}: {why}",
                    )
                )
    return statuses


def _measure_readings(
    campaign: EvalCampaign, bundles: Mapping[str, AnalysisContextBundle], arms: Sequence[CampaignArm]
) -> list[MeasureReading]:
    """A measurement's measures on every aspect: each arm's pass rate, its count over every rig with the Wilson
    interval on it, the cases the independent draws (:func:`~threetears.evals.analysis.stats.pooled_rate_interval`);
    then each measure each of its cells (one per rig) summarises with an interval, the summary's own."""
    labels = {arm.variant_key: arm.label for arm in arms}
    found: list[MeasureReading] = []
    for aspect in campaign.aspects:
        bundle = bundles[aspect.label]
        for board in bundle.scoreboard:
            label = labels.get(board.variant_key, f"arm {board.variant_key[:12]}")
            found.append(
                MeasureReading(
                    aspect=aspect.label,
                    arm=board.variant_key,
                    arm_label=label,
                    measure="passed",
                    name="pass rate",
                    value=board.passed / board.usable if board.usable else None,
                    interval=pooled_rate_interval(board.passed, board.usable, cases=board.usable_cases),
                    passed=board.passed,
                    trials=board.usable,
                    cases=board.usable_cases,
                    higher_is_better=True,
                )
            )
            for cell in bundle.cell_measures:
                if cell.variant_key != board.variant_key:
                    continue
                for summary in cell.measures.measures:
                    value = summary.rate if summary.rate is not None else summary.mean
                    # A figure with no interval is a derived one (an f1 over the cells), not a reading to report.
                    if value is None or summary.ci_low is None or summary.ci_high is None:
                        continue
                    descriptor = bundle.measure_catalog.get(summary.name)
                    found.append(
                        MeasureReading(
                            aspect=aspect.label,
                            arm=board.variant_key,
                            arm_label=label,
                            measure=summary.name,
                            name=measure_name(summary.name, descriptor.reader_name if descriptor else None),
                            rig=cell.apparatus_class_id,
                            value=value,
                            interval=(summary.ci_low, summary.ci_high)
                            if summary.ci_low is not None and summary.ci_high is not None
                            else None,
                            cases=summary.n_independent,
                            higher_is_better=summary.higher_is_better,
                        )
                    )
    return found


def _holm(
    rows: Sequence[_Row], keep: Callable[[_Row], bool], decided: Mapping[int, _Decision], alpha: float
) -> dict[int, float]:
    """Each row's p Holm-adjusted within its candidate's family: at most α exactly where the step-down showed it."""
    adjusted: dict[int, float] = {}
    for family in _families(rows, keep):
        adjusted.update(_snapped(family, holm_adjust([decided[index].p for index in family]), decided, alpha))
    return adjusted


def _snapped(
    family: Sequence[int], adjusted: Sequence[float], decided: Mapping[int, _Decision], alpha: float
) -> dict[int, float]:
    """Each row's Holm-adjusted p, a shown row's held to at most α: its p is ``α / k`` on the grid and Holm multiplies
    it back by ``k``, which in floating point can land a hair above α (``(0.05 / 11) * 11``), so a host filtering on
    ``p_holm <= α`` would disagree with the reading."""
    return {index: min(p, alpha) if decided[index].shown else p for index, p in zip(family, adjusted, strict=True)}


def _depths(bundle: AnalysisContextBundle) -> dict[str, Depth | None]:
    """Each arm's depth on the aspect, over every rig, as its scoreboard counts it."""
    return {board.variant_key: board.depth for board in bundle.scoreboard}


def _depth_differs(rate: RateReading) -> int | None:
    """The paired cases the two arms ran a different number of times; None unpaired or with no trials carried."""
    if not rate.paired or rate.trials is None:
        return None
    return sum(1 for first, second in zip(*rate.trials, strict=True) if first != second)


def _precision(rate: RateReading, keys: tuple[str, ...]) -> Precision | None:
    """What limits a pass rate's interval at the nominal level it was read at (:func:`~threetears.evals.analysis.
    depth.precision`), its sides named by ``keys`` in its samples' order; None with no per-case shares."""
    if rate.samples is None:
        return None
    return precision(
        rate.samples,
        rate.trials,
        passes=rate.passes,
        keys=(keys[0], keys[1]),
        interval=rate.interval,
        paired=rate.paired,
    )


def _points(rate: RateReading | None) -> tuple[float, float] | None:
    """A rate's interval in points, low then high; None with none."""
    if rate is None or rate.interval is None:
        return None
    return 100 * rate.interval[0], 100 * rate.interval[1]
