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

**The reading is read off the interval, never off a p.** Each direction is a step-down over its family's rows, Holm's
procedure carried on the intervals themselves: with ``k`` rows not yet shown, each is read on its two-sided bounded
interval (:func:`~threetears.evals.analysis.stats.bounded_difference_interval`) at ``1 − α/k``, every row it shows
leaves the family, and the step repeats at the next ``k`` until a step shows none; a row's interval is the one its last
step read. One row is read at ``1 − α``, the 95% a floor reads, so a one-aspect campaign's floor on its pass rate and
its aspect's reading against the same margin are one computation and cannot disagree: a drop is shown by
:func:`~threetears.evals.analysis.report.decision.read_floor` itself, on the pass rate alone, with the floor's case
floor and its bound over lost trials, and a gain by the same rules mirrored. The p values beside each row invert that
same interval (the smallest α at which it would clear), Holm-adjusted across the family
(:func:`~threetears.evals.analysis.stats.holm_adjust`), for a program to read; nothing on the page is decided by them.

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
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.report.decision import (
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
    SIGNIFICANCE_ALPHA,
    SMALL_N_BAND_FLOOR,
    bounded_difference_interval,
    holm_adjust,
)
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.declaration import CampaignArm, Floor, RateDrop
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.schema.base import EvalDocumentModel

__all__ = [
    "AspectReading",
    "CampaignReadings",
    "FloorState",
    "FloorStatus",
    "ShownReading",
    "campaign_readings",
]

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
            "Whether the reading confirms anything: the campaign's question was asked on this aspect before its first "
            "run (`asked_in_time`). False reads it as a lead, which the page says."
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


# =============================================================================
# Reading a campaign
# =============================================================================

#: The fewest usable cases on every arm before a row can be shown either way: the floors' case floor.
_CASE_FLOOR = SMALL_N_BAND_FLOOR

#: The share range the pass rate's per-case values lie in.
_RATE_RANGE = (0.0, 1.0)

#: The smallest p an inverted interval reports: one clearing even at ``1 - _P_FLOOR`` is reported at it.
_P_FLOOR = 1e-12

#: The largest α an inversion tries; an interval not clearing even there reads p 1.
_P_CEILING = 1.0 - 1e-9

#: Bisection steps on log α for an inverted p: enough for three significant figures across the twelve decades.
_P_STEPS = 40


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
    levels: dict[float, RateReading]

    def at(self, confidence: float) -> RateReading:
        """The pass rate with its bounded interval at ``confidence``, computed again from the same per-case values a
        floor reads, by the same function; the nominal level's is the floor's own reading."""
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

    def drop_verdict(self, confidence: float) -> tuple[bool, str]:
        """Whether a drop past the margin is shown at ``confidence``, and the reason: the floor's own reading
        (:func:`~threetears.evals.analysis.report.decision.read_floor`) on the pass rate alone, at that level."""
        if self.margin is None:
            return False, ""
        rate = self.at(confidence)
        readings = replace(self.base, rates=(rate,), confidence=confidence)
        floor = Floor(aspect=self.aspect, clause=RateDrop(max_points=self.margin))
        verdict = read_floor(floor, readings, control=self.control_label, candidate=self.arm.label)
        return verdict.outcome == "broken", verdict.reason

    def gain_p(self) -> float | None:
        """The smallest two-sided α at which the interval lies wholly above zero; None with no interval."""
        if self.rate.interval is None:
            return None
        return _inverted_p(lambda confidence: _low(self.at(confidence)) > 0)

    def drop_p(self) -> float | None:
        """The smallest two-sided α at which the interval lies wholly past the margin; None with no interval or no
        margin."""
        if self.rate.interval is None or self.margin is None:
            return None
        bound = -self.margin / 100
        return _inverted_p(lambda confidence: _high(self.at(confidence)) < bound)


def _low(rate: RateReading) -> float:
    """The interval's low end as a share, or minus infinity with none."""
    return rate.interval[0] if rate.interval is not None else -math.inf


def _high(rate: RateReading) -> float:
    """The interval's high end as a share, or infinity with none."""
    return rate.interval[1] if rate.interval is not None else math.inf


def _inverted_p(clears: Callable[[float], bool]) -> float:
    """The smallest two-sided α at which ``clears(1 − α)`` holds: a p read off the interval itself, bisected on log α.

    :param clears: Whether the interval at a two-sided level clears the null on the side tested.
    :ptype clears: Callable[[float], bool]
    :return: The p, in ``[_P_FLOOR, 1]``.
    :rtype: float
    """
    if clears(1.0 - _P_FLOOR):
        return _P_FLOOR
    if not clears(1.0 - _P_CEILING):
        return 1.0
    low, high = math.log(_P_FLOOR), math.log(_P_CEILING)
    for _ in range(_P_STEPS):
        middle = (low + high) / 2
        if clears(1.0 - math.exp(middle)):
            high = middle
        else:
            low = middle
    return min(1.0, math.exp(high))


def _step_down(rows: Sequence[int], shown: Callable[[int, float], bool], alpha: float) -> dict[int, tuple[float, bool]]:
    """Holm's step-down carried on the intervals: each step reads every row not yet shown at ``1 − α/k``, ``k`` of
    them, and every row it shows leaves the family; a step showing none ends it, its level every remaining row's.

    Equal to Holm on the inverted p values wherever a row's test is monotone in its level, which the bounded interval
    is to the precision it is computed at; and the reading is the interval's by construction, so the two never
    contradict.

    :return: Each row's level and whether it was shown there.
    :rtype: dict[int, tuple[float, bool]]
    """
    remaining = list(rows)
    decided: dict[int, tuple[float, bool]] = {}
    while remaining:
        level = 1.0 - alpha / len(remaining)
        hits = [index for index in remaining if shown(index, level)]
        if not hits:
            decided.update({index: (level, False) for index in remaining})
            break
        decided.update({index: (level, True) for index in hits})
        remaining = [index for index in remaining if index not in hits]
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


def _candidates(campaign: EvalCampaign, arms: Sequence[CampaignArm] | None) -> list[CampaignArm]:
    """The candidate arms, each once by variant key, in the order given; never the control."""
    control = campaign.control
    seen = {control.variant_key} if control is not None else set()
    found: list[CampaignArm] = []
    for arm in campaign.arms if arms is None else arms:
        if arm.role == "candidate" and arm.variant_key not in seen:
            seen.add(arm.variant_key)
            found.append(arm)
    return found


def _reason(row: _Row, reading: ShownReading, gain_level: float, drop_level: float, drop_reason: str) -> str:
    """Why a row reads as it does, in one sentence naming the arms."""
    arm, control = row.arm.label, row.control_label
    if reading == "better":
        return f"{arm}'s gain over {control} is shown: its whole interval is above zero"
    if reading == "worse":
        return drop_reason
    fewest = row.fewest()
    rate = row.at(gain_level)
    if rate.interval is None:
        return "no interval could be read: a side has fewer than two cases with a usable trial"
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
    return f"its interval reaches zero and does not lie past the {points}-point margin"


def campaign_readings(
    campaign: EvalCampaign,
    bundles: Mapping[str, AnalysisContextBundle],
    *,
    arms: Sequence[CampaignArm] | None = None,
) -> CampaignReadings:
    """What code reads off a campaign: each candidate against the control on every aspect, and every floor.

    Per aspect and candidate, the aspect's pass rate
    (:func:`~threetears.evals.analysis.report.decision.aspect_pass_rate`): the change in points, shown better, shown
    worse or unclear, read by the step-down on intervals the module describes, Holm across every aspect × candidate
    row at :data:`~threetears.evals.analysis.stats.SIGNIFICANCE_ALPHA`, gains and drops apart. An aspect with no margin
    is never shown worse and tests no drop. An unclear row with a margin carries the cases that would show the
    candidate within it (:func:`~threetears.evals.analysis.report.decision.project_rate`, at the row's drop level).
    Per floor and candidate, :func:`~threetears.evals.analysis.report.decision.evaluate_floor`, at the nominal level
    with no correction. A row is confirmatory only where the campaign's question was asked on that aspect before its
    first run (the bundle's ``question_declared``).

    A candidate that did not run on an aspect, or an aspect its control did not run on, has no row there. A campaign
    naming no control reads no row and no floor.

    :param campaign: The campaign.
    :ptype campaign: EvalCampaign
    :param bundles: Each aspect's evidence, keyed by its label: one per aspect, each assembled for it.
    :ptype bundles: Mapping[str, AnalysisContextBundle]
    :param arms: The campaign's arms as read (:func:`~threetears.evals.analysis.campaigns.campaign_arms`: the named,
        then the derived candidates); None reads the named arms alone.
    :ptype arms: Sequence[CampaignArm] | None
    :return: The readings.
    :rtype: CampaignReadings
    :raises RefusedInput: An aspect has no bundle, a bundle is keyed by a label the campaign does not hold, or a bundle
        reads another aspect than the one it is keyed by.
    """
    checked = _checked_bundles(campaign, bundles)
    control = campaign.control
    alpha = SIGNIFICANCE_ALPHA
    labels = [aspect.label for aspect in campaign.aspects]
    if control is None:
        return CampaignReadings(
            campaign_id=campaign.id,
            alpha=alpha,
            control=None,
            control_label="",
            aspects=labels,
            readings=[],
            floors=[],
        )
    candidates = _candidates(campaign, arms)
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
                    levels={},
                )
            )
    gains = _step_down(
        [index for index, row in enumerate(rows) if row.rate.interval is not None],
        lambda index, level: rows[index].gain_shown(level),
        alpha,
    )
    drops = _step_down(
        [index for index, row in enumerate(rows) if row.rate.interval is not None and row.margin is not None],
        lambda index, level: rows[index].drop_verdict(level)[0],
        alpha,
    )
    gain_ps = {index: p for index, row in enumerate(rows) if (p := row.gain_p()) is not None}
    drop_ps = {index: p for index, row in enumerate(rows) if (p := row.drop_p()) is not None}
    gain_holm = dict(zip(gain_ps, holm_adjust(list(gain_ps.values())), strict=True))
    drop_holm = dict(zip(drop_ps, holm_adjust(list(drop_ps.values())), strict=True))
    readings: list[AspectReading] = []
    for index, row in enumerate(rows):
        gain_level, gained = gains.get(index, (None, False))
        drop_level, dropped = drops.get(index, (None, False))
        # Both cannot hold on intervals about one estimate; were rounding ever to make them, neither is claimed.
        reading: ShownReading = "unclear" if gained == dropped else "better" if gained else "worse"
        drop_reason = row.drop_verdict(drop_level)[1] if drop_level is not None else ""
        gain_rate = row.at(gain_level) if gain_level is not None else None
        drop_rate = row.at(drop_level) if drop_level is not None else None
        fewest = row.fewest()
        projection = (
            project_rate(drop_rate, cases=fewest, margin=row.margin, per_case_usd=row.per_case_usd)
            if reading == "unclear" and drop_rate is not None and row.margin is not None and fewest
            else None
        )
        readings.append(
            AspectReading(
                aspect=row.aspect,
                arm=row.arm.variant_key,
                arm_label=row.arm.label,
                change=-row.rate.point_drop(),
                gain_interval=_points(gain_rate),
                gain_confidence=gain_level,
                drop_interval=_points(drop_rate),
                drop_confidence=drop_level,
                gain_p=gain_ps.get(index),
                gain_p_holm=gain_holm.get(index),
                drop_p=drop_ps.get(index),
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
                    row, reading, gain_level or row.rate.confidence or 0.95, drop_level or 0.95, drop_reason
                ),
                confirmatory=row.confirmatory,
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
    )


def _points(rate: RateReading | None) -> tuple[float, float] | None:
    """A rate's interval in points, low then high; None with none."""
    if rate is None or rate.interval is None:
        return None
    return 100 * rate.interval[0], 100 * rate.interval[1]
