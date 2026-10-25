"""The report plan: what a campaign page says, in what order, with which charts, as one typed document.

The plan is the one writer of a campaign's report (``docs/eval-report-kit/GENERATOR.md`` §2): the planner (a model)
chooses the mode, the headline, the decision slots, the options, the charts, the explore controls and every sentence,
and the renderer draws it. ``docs/eval-report-kit/schema/report-plan.schema.json`` is its documentation copy; a test
holds the two to the same fields and vocabularies.

**Two writers, one document.** Code computes facts and the planner writes choices and prose:

- **The planner writes** :class:`PlannedReport`: mode, headline kind, headline, lede, flag, decision (with its stated
  confidence), options, summary and section charts (kind, title, a structured :class:`ChartQuery`, reference,
  emphasis), explore controls, which full-results tables to print, method, durable claims and answers to the
  warnings code raised. :func:`planner_response_format` is its strict projection, and nothing in it is a field code
  fills.
- **Code fills** the rest of :class:`ReportPlan`: the schema id, the :class:`Signals` and the shape (GENERATOR §1
  read as a lookup over the report input), the :class:`Masthead`, each arm's colour slot, whether the arms key
  shows, and each chart's level, unit and domain (:class:`ChartSpec`). The stored plan carries both, so a page
  renders from the plan alone.

**Sized like the memo's contract.** Nothing in the planner's contract is nullable or a union: an optional part is a
list of at most one (the flag, the explore part, a chart's reference) and an optional scalar is an empty string.
Prose refers to figures by key (``{reading.<arm>.change|pts}``, GENERATOR §6.2), never by typed numbers, so no prose
field carries a number code would read.

**Field-level constraints only.** Lengths, enums, non-empty ids and list bounds are checked here. Whether the parts
agree with each other and with the report input (a headline kind that suits the mode, slot keys in the mode's order,
a key that resolves, a flag that discloses what the caveats imply) is :mod:`threetears.evals.analysis.report_plan_checks`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from threetears.evals.kernel.authored import Confidence, strict_response_format
from threetears.evals.schema.prose import ModelProse

#: The schema id a stored plan carries.
REPORT_PLAN_SCHEMA = "report-plan/1"

#: The name the planner's contract is sent under.
PLAN_SCHEMA_NAME = "report_plan"

#: An id the plan names something by: never empty.
_PlanId = Annotated[str, StringConstraints(min_length=1)]

#: GENERATOR §1: how the candidates relate.
_CandidateStructure = Literal["unordered", "ordered_versions", "factorial", "control_plus_candidates"]
#: GENERATOR §1: whether an arm is the control.
_Baseline = Literal["none", "control"]
#: GENERATOR §1: what a trial's score is.
_ScoreType = Literal["graded", "pass_fail", "class_label", "mixed"]
#: GENERATOR §1: how the cases are grouped.
_StrataSignal = Literal["none", "tasks", "many_strata", "target_matrix"]
#: GENERATOR §1: what the campaign declared to decide by.
_CriteriaSignal = Literal["hard_gates", "margin", "none_declared"]
#: GENERATOR §1: the shape the signals look up.
_Shape = Literal["A", "B", "C", "D", "E"]
#: GENERATOR §1.1: decide, trade off, or make no call.
_Mode = Literal["decide", "tradeoff", "no_call"]
#: What the headline is: a decision, a trade or a stated no-call. Code checks it agrees with the mode; whether the
#: sentence is one is a model judge's to score.
_HeadlineKind = Literal["decision", "trade", "no_call"]
#: GENERATOR §2.5: the decision block's slots, decide mode's four then tradeoff mode's four. A no-call uses decide
#: mode's slots.
_SlotKey = Literal[
    "recommendation", "why_not", "watch", "next", "the_trade", "where_it_flips", "our_lean", "what_settles_it"
]
#: GENERATOR §2.6: why an option is in the options table.
_OptionRole = Literal["recommended", "lean", "control", "near_miss", "most_accurate", "cheapest", "alternative"]
#: The kit's chart kinds (``analysis.report.charts.KIT_CHARTS``, one name each) plus a plain table.
_PlanChartKind = Literal[
    "forest",
    "bars",
    "diverging_bars",
    "lines",
    "scatter_frontier",
    "gate_scatter",
    "dot_interval",
    "range_bars",
    "gate_matrix",
    "lattice",
    "heatmap",
    "diff_heatmap",
    "confusion",
    "examples",
    "table",
]
#: The part of the report input a chart reads.
_ChartSource = Literal[
    "readings",
    "contrasts",
    "measures",
    "judged",
    "scoreboard",
    "floors",
    "strata",
    "examples",
    "tasks",
    "configs",
    "cells",
    "confusion",
    "human_pref",
]
#: What a chart does to the values it reads: per 1,000 replies, each arm less the control, or nothing.
_ChartTransform = Literal["none", "per_1k", "arm_minus_control"]
#: What a chart repeats its panel by.
_SmallMultiplesBy = Literal["none", "arm", "aspect", "task"]
#: What a reference line is: the kit's reference kinds (``analysis.report.charts.ChartReference.kind``).
_ReferenceKind = Literal["floor", "budget", "margin", "no_preference"]
#: An explore control's widget.
_ControlKind = Literal["slider", "segmented", "toggle", "select"]
#: The tables Full results prints.
_FullResultsTable = Literal["arms_by_measure", "contrasts", "strata", "gate_matrix_full", "configs", "cells"]
#: What code warned of on the recommended arm, which the plan must answer: a floor it broke, a floor read unclear (a
#: borderline, GENERATOR §3, answered with the sample that would settle it), an aspect a choice's pair readings show
#: another arm ahead of it on, an absolute bar it missed.
_WarningKind = Literal["floor_broken", "floor_unclear", "behind_on", "bar_missed"]

#: What a flag discloses, by kind: criteria set after the results, a judge with no human calibration, one confound
#: (``confound:<id>``, its id in the report input's ``caveats.confounds``) and one short run (``short_run:<run id>``).
_DISCLOSURE_KINDS: tuple[str, ...] = ("late_criteria", "judge_uncalibrated", "confound", "short_run")
#: The disclosure kinds that name one thing after a colon.
_NAMED_DISCLOSURES = ("confound", "short_run")
#: A disclosure as the flag writes it: a bare kind, or a named kind with its id.
_DISCLOSURE_PATTERN = (
    "^("
    + "|".join(kind for kind in _DISCLOSURE_KINDS if kind not in _NAMED_DISCLOSURES)
    + "|"
    + "|".join(f"{kind}:.+" for kind in _NAMED_DISCLOSURES)
    + ")$"
)
#: One disclosure.
_Disclosure = Annotated[str, StringConstraints(pattern=_DISCLOSURE_PATTERN)]


class _Plan(BaseModel):
    """A closed object: the schema sent is strict, the check on return is strict, and so is a stored read."""

    model_config = ConfigDict(extra="forbid")


# --- what the planner writes ---------------------------------------------------------------------------------------


class Flag(_Plan):
    """The one amber line under the lede that discloses every caveat at once (GENERATOR §2.4). Planner-written.

    It carries no confidence: neither a late criterion nor an uncalibrated judge lowers the recommendation's
    confidence (owner ruling, 2026-10-25); the flag says what it weakens and what survives, and the reader weighs it.
    """

    title: ModelProse = Field(min_length=1, description="The flag in one line.")
    text: ModelProse = Field(min_length=1, description="What it weakens and what survives, in a sentence or two.")
    discloses: list[_Disclosure] = Field(
        min_length=1,
        description=(
            "Each caveat it discloses, once: `late_criteria`, `judge_uncalibrated`, `confound:<id>` or "
            "`short_run:<run id>`."
        ),
    )
    weakens: list[str] = Field(description="The measures whose findings it weakens, as the input names them.")
    survives: list[str] = Field(description="The measures whose findings survive it, as the input names them.")


class DecisionSlot(_Plan):
    """One of the decision block's four slots: which, and what it says. Planner-written."""

    key: _SlotKey
    text: ModelProse = Field(min_length=1, description="The slot's text, figures by key.")


class Decision(_Plan):
    """The decision block (GENERATOR §2.5): four slots in the mode's order, and how sure the planner is. Planner-written.

    ``confidence`` is stated, never capped: a late criterion or an uncalibrated judge is disclosed by the flag and does
    not lower it. The tiers are the memo's (:data:`threetears.evals.kernel.authored.Confidence`).
    """

    slots: list[DecisionSlot] = Field(min_length=4, max_length=4, description="The four slots, in the mode's order.")
    confidence: Confidence = Field(description="How sure the planner is of the decision, as stated.")


class OptionRow(_Plan):
    """One row of the options table: an arm (or a modeled option) and why it is shown. Planner-written."""

    option: _PlanId = Field(description="An arm id, or a modeled option's id.")
    role: _OptionRole


class Options(_Plan):
    """The options table (GENERATOR §2.6): rows in reading order, the recommended one first. Planner-written."""

    rows: list[OptionRow] = Field(min_length=1, max_length=5)
    columns: list[_PlanId] = Field(
        min_length=1,
        description="A figure key family per column: pass_rate, reading.change, judged:<dim>, cost_per_1k, gate, ...",
    )


class ChartQuery(_Plan):
    """What a chart reads from the report input, structured so code can check it names things that exist.

    Planner-written. An empty list or string means no filter there: every arm, every measure, every stratum, no
    aspect, no limit (``limit`` 0).
    """

    source: _ChartSource = Field(description="The part of the report input the chart reads.")
    aspect: str = Field(description="The aspect id it reads, for readings and anything per aspect; else empty.")
    measures: list[str] = Field(description="Measure or judged-dimension keys, as the source spells them.")
    arms: list[str] = Field(description="Arm ids to show, in order; empty for every arm.")
    strata: list[str] = Field(description="Stratum or task ids to keep; empty for all.")
    limit: int = Field(ge=0, description="The most rows to show (strata by cases, examples); 0 for no limit.")
    transform: _ChartTransform
    small_multiples_by: _SmallMultiplesBy


class ChartReference(_Plan):
    """A reference line on a chart: which figure it sits at and what kind of line it is. Planner-written; the
    renderer reads the value from the key and writes the label."""

    key: _PlanId = Field(description="The figure key of the line's value, e.g. aspect.margin.")
    kind: _ReferenceKind


class PlannedChart(_Plan):
    """A chart as the planner chooses it: kind, title, what it reads, a reference line and the one thing it stresses."""

    kind: _PlanChartKind
    title: ModelProse = Field(description="The chart's title; empty when the section headline says it.")
    query: ChartQuery
    reference: list[ChartReference] = Field(max_length=1, description="At most one reference line.")
    emphasis: str = Field(description="The one arm or row bolded or ringed, by id; empty for none.")


class PlannedSection(_Plan):
    """One evidence section (GENERATOR §2.9): the finding it states and the chart that shows it. Planner-written."""

    id: _PlanId
    eyebrow: ModelProse = Field(min_length=1, description="The question the section answers, in a few words.")
    headline: ModelProse = Field(min_length=1, description="A finding with subject and direction.")
    takeaway: ModelProse = Field(description="One or two sentences on how to read the chart; empty for none.")
    chart: PlannedChart
    caption: ModelProse = Field(description="One line under the chart; empty for none.")


class ExploreControl(_Plan):
    """One explore control (GENERATOR §2.10): a business choice the reader may move. Planner-written.

    A slider reads ``min``, ``max``, ``step`` and ``default``; a segmented control or a select offers ``options`` and
    starts on the first; a toggle starts off. Unused numbers are 0 and unused options empty.
    """

    id: _PlanId
    kind: _ControlKind
    title: ModelProse = Field(min_length=1, description="What the control sets, as its label reads.")
    binds: str = Field(
        description="The criterion it replaces: margin, floor:<id>, p95, price_per_point, weighting, lever:<factor>."
    )
    min: float
    max: float
    step: float
    default: float
    options: list[str]


class Explore(_Plan):
    """The explore part under the divider: one to three controls and what they recompute. Planner-written."""

    title: ModelProse = Field(min_length=1)
    controls: list[ExploreControl] = Field(min_length=1, max_length=3)
    recomputes: ModelProse = Field(
        min_length=1, description="What the controls recompute and the page states, and where the answer flips."
    )


class MethodItem(_Plan):
    """One term and its definition in Method. Planner-written."""

    term: ModelProse = Field(min_length=1)
    text: ModelProse = Field(min_length=1)


class DurableClaim(_Plan):
    """A claim about the subject that later campaigns should carry as a prior insight. Planner-written.

    Code turns it into an insight, reading the runs it rests on from the arms and measures its ``keys`` name.
    """

    text: ModelProse = Field(min_length=1, description="The claim, figures by key.")
    confidence: Confidence
    keys: list[_PlanId] = Field(min_length=1, description="The figure keys it rests on.")


class WarningRef(_Plan):
    """One warning code raised on the recommended arm, named by what it is about in the report input's ids.

    - ``floor_broken`` and ``floor_unclear``: ``aspect`` is the floor's aspect, ``name`` its ``floor`` key.
    - ``behind_on``: ``aspect`` is where another arm is shown ahead, ``name`` that arm's id.
    - ``bar_missed``: ``aspect`` is the bar's aspect, ``name`` its measure key.

    Held as parts, not one joined string, because the ids themselves contain ``/`` (a floor key does).
    """

    kind: _WarningKind
    aspect: _PlanId
    name: _PlanId


class WarningAnswer(_Plan):
    """Why the recommendation stands over one warning code raised on its arm. Planner-written; every warning raised
    gets exactly one."""

    warning: WarningRef
    text: ModelProse = Field(min_length=1, description="Why recommending the arm over it is acceptable here.")


class PlannedReport(_Plan):
    """What the planner writes: the planner's contract, sent strictly (:func:`planner_response_format`).

    It holds no field code fills; :class:`ReportPlan` adds those.
    """

    mode: _Mode
    headline_kind: _HeadlineKind
    headline: ModelProse = Field(
        min_length=1, description="The decision as an instruction with its reason; in tradeoff mode, the trade."
    )
    lede: ModelProse = Field(min_length=1, max_length=420, description="At most two sentences.")
    flag: list[Flag] = Field(max_length=1, description="The flag, or none when nothing needs disclosing.")
    decision: Decision
    options: Options
    summary: Sequence[PlannedChart] = Field(max_length=3, description="Summary charts beside the options table.")
    sections: Sequence[PlannedSection] = Field(description="The evidence sections, in reading order.")
    explore: list[Explore] = Field(max_length=1, description="The explore part, or none.")
    full_results: list[_FullResultsTable]
    method: list[MethodItem] = Field(min_length=3, max_length=9)
    durable_claims: list[DurableClaim] = Field(description="Claims for later campaigns' prior insights; may be empty.")
    answers: list[WarningAnswer] = Field(description="One per warning code raised on the recommended arm.")


# --- what code fills ------------------------------------------------------------------------------------------------


class Signals(_Plan):
    """GENERATOR §1's signals, read off the report input. Code-filled."""

    candidate_structure: _CandidateStructure
    baseline: _Baseline
    score_type: _ScoreType
    strata: _StrataSignal
    criteria: _CriteriaSignal
    cross_cutting: list[str] = Field(description="Aspects that occur inside every task.")
    log_cost_axis: bool = Field(description="Whether cost spans more than 10x across arms.")


class Masthead(_Plan):
    """The page's one-line masthead (GENERATOR §2.1). Code-filled."""

    kind: str
    system: str
    counts: str
    run_date: str
    synthetic: bool


class PlanArm(_Plan):
    """An arm's colour on the page. Code-filled; its name is the report input's, declared, never the planner's."""

    id: _PlanId
    color_slot: int = Field(ge=-1, description="-1 for the control (ink gray); 0.. for series colours in fixed order.")


class ChartSpec(PlannedChart):
    """A chart as stored: the planner's choice plus what code reads off the data. ``level``, ``unit`` and ``domain``
    are code-filled; an empty ``domain`` lets the renderer fit the axis."""

    level: str = Field(description="The interval level printed on the chart, e.g. 95%; empty without intervals.")
    unit: str = Field(description="The chart's one unit.")
    domain: list[float] = Field(max_length=2, description="The value axis, [low, high], or empty.")

    @field_validator("domain")
    @classmethod
    def _domain_is_a_range_or_empty(cls, domain: list[float]) -> list[float]:
        """Refuse a domain of one number: an axis has two ends, or the renderer fits it.

        :raises ValueError: The domain holds one number.
        """
        if len(domain) == 1:
            raise ValueError("`domain` is [low, high] or empty, never one number")
        return domain


class Section(PlannedSection):
    """An evidence section as stored, its chart with code's facts."""

    chart: ChartSpec


class ReportPlan(PlannedReport):
    """The stored plan: the planner's :class:`PlannedReport` with the fields code fills, which the renderer draws."""

    model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True, serialize_by_alias=True)

    schema_id: Literal["report-plan/1"] = Field(alias="schema")
    signals: Signals
    shape: _Shape
    masthead: Masthead
    arms: list[PlanArm] = Field(min_length=1)
    show_arms_key: bool
    summary: Sequence[ChartSpec] = Field(max_length=3, description="Summary charts beside the options table.")
    sections: Sequence[Section] = Field(description="The evidence sections, in reading order.")


def planner_response_format() -> dict[str, Any]:
    """The strict ``json_schema`` directive that asks the planner for exactly :class:`PlannedReport`.

    :return: The OpenAI-shaped directive.
    :rtype: dict[str, Any]
    """
    return strict_response_format(PLAN_SCHEMA_NAME, PlannedReport.model_json_schema())


__all__ = [
    "PLAN_SCHEMA_NAME",
    "REPORT_PLAN_SCHEMA",
    "ChartQuery",
    "ChartReference",
    "ChartSpec",
    "Decision",
    "DecisionSlot",
    "DurableClaim",
    "Explore",
    "ExploreControl",
    "Flag",
    "Masthead",
    "MethodItem",
    "OptionRow",
    "Options",
    "PlanArm",
    "PlannedChart",
    "PlannedReport",
    "PlannedSection",
    "ReportPlan",
    "Section",
    "Signals",
    "WarningAnswer",
    "WarningRef",
    "planner_response_format",
]
