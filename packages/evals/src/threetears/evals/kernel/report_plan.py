"""The report plan: what a campaign page says, in what order, with which charts, as one typed document.

The plan is the one writer of a campaign's report (``docs/eval-report-kit/GENERATOR.md`` §2): the planner (a model)
chooses the mode, the headline, the decision slots, the options, the charts, the explore controls and every sentence,
and the renderer draws it. ``docs/eval-report-kit/schema/report-plan.schema.json`` is its documentation copy; a test
holds the two to the same fields and vocabularies.

**Two writers, one document.** Code computes facts and the planner writes choices and prose:

- **The planner writes** :class:`PlannedReport`: mode, headline kind, headline, lede, flag, decision (with its
  conditional confidence), options, summary and section charts (kind, title, a structured :class:`ChartQuery`, reference kind,
  emphasis), explore controls, method, durable claims and the answers to the warnings code raised, each answer naming
  its warning by position in code's list.
- **Code fills** the rest of :class:`ReportPlan`: the schema id, the :class:`Signals` and the shape (GENERATOR §1
  read as a lookup over the report input), the :class:`Masthead`, each arm's colour slot, whether the arms key
  shows, each chart's level, unit and domain (:class:`ChartSpec`), the flag's disclosures and the measures it weakens
  and leaves standing (:class:`Flag`), the options table's columns (:class:`Options`), each chart reference's figure
  key (:class:`ChartReference`), which full-results tables print, and the warning each answer names
  (:class:`WarningAnswer`). The stored plan carries both, so a page renders from the plan alone;
  :mod:`threetears.evals.analysis.report_plan_facts` fills it.

**The planner writes in two calls, so neither contract grows with the page.** A page's prose slots multiply with its
sections, and a strict schema naming each slot outgrew what strict-schema writers accept
(:data:`~threetears.evals.kernel.authored.SLOT_BUDGET`). So the planner first writes the page's
:class:`PlanStructure` (every choice, no prose: :func:`structure_response_format`), then its :class:`PlanProse`, a
flat list of ``{key, text}`` pairs addressing the prose slots that structure created (:func:`prose_response_format`),
and code assembles :class:`PlannedReport` from the two (:mod:`threetears.evals.analysis.plan_assembly`). Nothing in
either contract is nullable or a union: an optional part is a list of at most one and an optional scalar an empty
string. Prose refers to figures by key (``{reading.<arm>.change|pts}``, GENERATOR §6.2), never by typed numbers.

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

#: The name the planner's structure contract is sent under.
_STRUCTURE_SCHEMA_NAME = "report_plan_structure"

#: The name the planner's prose contract is sent under.
_PROSE_SCHEMA_NAME = "report_plan_prose"

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

#: Every caveat a report input carries, by a stable id code assigns
#: (:func:`threetears.evals.analysis.report_decision.report_caveats`): ``late:<criterion>`` or
#: ``late:<criterion>:<name>`` for each criterion set after the results (``late:margin``,
#: ``late:floor:<floor key>``), ``judge_uncalibrated``, ``confound:<id>`` for each confound,
#: ``apparatus:<dimension>`` for each apparatus confound, ``short_run:<run id>`` for each short run,
#: ``unsettled:<aspect>`` for each aspect a surviving candidate's reading leaves unsettled past its margin, ``latency``
#: and ``unreported:<observation>`` for each observation recorded but not reported.
_CAVEAT_KINDS: tuple[str, ...] = (
    "late",
    "judge_uncalibrated",
    "confound",
    "apparatus",
    "short_run",
    "unsettled",
    "latency",
    "unreported",
)
#: The caveat kinds that name one thing after a colon.
_NAMED_CAVEATS = ("late", "confound", "apparatus", "short_run", "unsettled", "unreported")
#: What a flag may disclose: the report-wide caveat kinds, plus ``no_rule``, a tradeoff the campaign declared no rule
#: to rank (GENERATOR §4.2: the flag names who makes the call). ``latency`` and ``unreported`` are method-level and
#: never the flag's.
_FLAG_KINDS: tuple[str, ...] = (
    "no_rule",
    "late",
    "judge_uncalibrated",
    "confound",
    "apparatus",
    "short_run",
    "unsettled",
)


def _id_pattern(kinds: Sequence[str]) -> str:
    """The pattern a caveat id of one of ``kinds`` matches: a bare kind, or a named kind with its id."""
    bare = [kind for kind in kinds if kind not in _NAMED_CAVEATS]
    named = [f"{kind}:.+" for kind in kinds if kind in _NAMED_CAVEATS]
    return "^(" + "|".join([*bare, *named]) + ")$"


#: One caveat, by its id.
_CaveatId = Annotated[str, StringConstraints(pattern=_id_pattern(_CAVEAT_KINDS))]
#: One disclosure as the flag holds it: a report-wide caveat's id, or ``no_rule``.
_Disclosure = Annotated[str, StringConstraints(pattern=_id_pattern(_FLAG_KINDS))]
#: GENERATOR §2.5: how sure the decision is. ``high``: no open caveat remains; ``high_if_immaterial``: the evidence is
#: strong, and open caveats remain, each with the condition that would make it material; ``low``: the evidence is weak
#: even if every caveat is immaterial.
_ConfidenceForm = Literal["high", "high_if_immaterial", "low"]


class _Plan(BaseModel):
    """A closed object: the schema sent is strict, the check on return is strict, and so is a stored read."""

    model_config = ConfigDict(extra="forbid")


# --- what the planner writes ---------------------------------------------------------------------------------------


class _PlannedFlag(_Plan):
    """The flag as the planner writes it: its line and what it weakens and what survives, in words.

    It carries no confidence: the decision's :class:`DecisionConfidence` does.
    """

    title: ModelProse = Field(min_length=1, description="The flag in one line.")
    text: ModelProse = Field(min_length=1, description="What it weakens and what survives, in a sentence or two.")


class Flag(_PlannedFlag):
    """The one amber line under the lede for the report-wide disclosures no confidence clause carries (GENERATOR §2.4).

    The planner writes the words; code fills what it discloses and the measures whose findings it weakens and leaves
    standing. It discloses ``no_rule`` in a tradeoff with no declared ranking rule, and each open report-wide caveat the
    decision's confidence does not list, which only a ``low`` confidence may leave out
    (:func:`threetears.evals.analysis.report_plan_facts.flag_disclosures`).
    """

    discloses: list[_Disclosure] = Field(
        min_length=1,
        description=(
            "Each report-wide caveat it discloses, once, by id (`late:<criterion>`, `late:<criterion>:<name>`, "
            "`judge_uncalibrated`, `confound:<id>`, `apparatus:<dimension>`, `short_run:<run id>`), or `no_rule`."
        ),
    )
    weakens: list[str] = Field(description="The measures whose findings it weakens, as the input names them.")
    survives: list[str] = Field(description="The measures whose findings survive it, as the input names them.")


class DecisionSlot(_Plan):
    """One of the decision block's four slots: which, what it says, and the caveats said beside its claim.
    Planner-written."""

    key: _SlotKey
    text: ModelProse = Field(min_length=1, description="The slot's text, figures by key.")
    caveats: list[_CaveatId] = Field(
        description="The ids of the caveats this slot's text states beside the claim they qualify; may be empty."
    )


class CaveatClause(_Plan):
    """One open caveat the decision's confidence is conditional on, and what would make it material. Planner-written.

    ``material_if`` is the clause the page prints for it: what the caveat is, then the condition that would make it
    material ("The judge is uncalibrated. Material only if it misreads in-character replies"), and, where the deciding
    finding does not rest on the caveat, that it does not.
    """

    id: _CaveatId = Field(description="The caveat's id, as code assigns it.")
    material_if: ModelProse = Field(
        description="The clause the page prints: what the caveat is, and what would make it material."
    )


class DecisionConfidence(_Plan):
    """How sure the decision is, in one of three forms, with the open caveats it is conditional on. Planner-written.

    Code enforces the form against the caveats (:mod:`threetears.evals.analysis.report_plan_checks`): plain ``high``
    only when no open report-wide caveat remains; ``high_if_immaterial`` lists every open report-wide caveat, each
    with its clause; ``low`` is an evidence judgement allowed whatever the caveats. A caveat code proved immaterial is
    never listed: code renders its checked line.
    """

    form: _ConfidenceForm
    caveats: list[CaveatClause] = Field(description="The open caveats the confidence is conditional on, each once.")


class Decision(_Plan):
    """The decision block (GENERATOR §2.5): four slots in the mode's order, and how sure the decision is.
    Planner-written."""

    slots: list[DecisionSlot] = Field(min_length=4, max_length=4, description="The four slots, in the mode's order.")
    confidence: DecisionConfidence = Field(description="How sure the decision is, and on which open caveats.")


class OptionRow(_Plan):
    """One row of the options table: an arm (or a modeled option) and why it is shown. Planner-written."""

    option: _PlanId = Field(description="An arm id, or a modeled option's id.")
    role: _OptionRole


class _PlannedOptions(_Plan):
    """The options table's rows as the planner chooses them: in reading order, the recommended one first."""

    rows: list[OptionRow] = Field(min_length=1, max_length=5)


class Options(_PlannedOptions):
    """The options table (GENERATOR §2.6): the planner's rows, and the columns code reads off the input's criteria
    and cost (§2.6: "the criteria plus cost")."""

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


class _PlannedChartReference(_Plan):
    """A reference line as the planner asks for it: what kind of line it is."""

    kind: _ReferenceKind


class ChartReference(_PlannedChartReference):
    """A reference line on a chart: what kind of line it is, and the figure it sits at. The planner picks the kind;
    code fills the key from the chart's aspect, and the renderer reads the value from it and writes the label."""

    key: _PlanId = Field(description="The figure key of the line's value, e.g. aspect.margin.")


class PlannedChart(_Plan):
    """A chart as the planner chooses it: kind, title, what it reads, a reference line and the one thing it stresses."""

    kind: _PlanChartKind
    title: ModelProse = Field(description="The chart's title; empty when the section headline says it.")
    query: ChartQuery
    reference: Sequence[_PlannedChartReference] = Field(max_length=1, description="At most one reference line.")
    emphasis: str = Field(description="The one arm or row bolded or ringed, by id; empty for none.")


class _PlannedSection(_Plan):
    """One evidence section (GENERATOR §2.9): the finding it states and the chart that shows it. Planner-written."""

    id: _PlanId
    eyebrow: ModelProse = Field(min_length=1, description="The question the section answers, in a few words.")
    headline: ModelProse = Field(min_length=1, description="A finding with subject and direction.")
    takeaway: ModelProse = Field(description="One or two sentences on how to read the chart; empty for none.")
    chart: PlannedChart
    caption: ModelProse = Field(description="One line under the chart; empty for none.")
    caveats: list[_CaveatId] = Field(
        description="The ids of the caveats the section states beside its finding; may be empty."
    )


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
    """One term and its definition in Method, and the method-level caveats it states. Planner-written."""

    term: ModelProse = Field(min_length=1)
    text: ModelProse = Field(min_length=1)
    caveats: list[_CaveatId] = Field(
        description="The ids of the method-level caveats (apparatus, latency, unreported) it states; may be empty."
    )


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


class _PlannedAnswer(_Plan):
    """Why the recommendation stands over one warning code raised on its arm, as the planner writes it: the warning by
    its position in code's list of the warnings raised on that arm, and the answer."""

    warning: int = Field(ge=0, description="The warning's position in code's list for the recommended arm.")
    text: ModelProse = Field(min_length=1, description="Why recommending the arm over it is acceptable here.")


class WarningAnswer(_Plan):
    """Why the recommendation stands over one warning code raised on its arm, the warning named in the input's ids.
    The planner writes the text; code names the warning from the position the planner gave. Every warning raised
    gets exactly one."""

    warning: WarningRef
    text: ModelProse = Field(min_length=1, description="Why recommending the arm over it is acceptable here.")


class _PlanCore(_Plan):
    """The parts the planner writes whole, which the planner's reply and the stored plan share as they are."""

    mode: _Mode
    headline_kind: _HeadlineKind
    headline: ModelProse = Field(
        min_length=1, description="The decision as an instruction with its reason; in tradeoff mode, the trade."
    )
    lede: ModelProse = Field(min_length=1, max_length=420, description="At most two sentences.")
    decision: Decision
    method: list[MethodItem] = Field(min_length=3, max_length=9)


class PlannedReport(_PlanCore):
    """What the planner writes, assembled from its two replies (:class:`PlanStructure` and :class:`PlanProse`).

    It holds no field code fills; :class:`ReportPlan` adds those, and completes each unit code has a part of.
    """

    flag: list[_PlannedFlag] = Field(max_length=1, description="The flag, or none when nothing needs disclosing.")
    options: _PlannedOptions
    summary: Sequence[PlannedChart] = Field(max_length=3, description="Summary charts beside the options table.")
    sections: Sequence[_PlannedSection] = Field(description="The evidence sections, in reading order.")
    explore: list[Explore] = Field(max_length=1, description="The explore part, or none.")
    durable_claims: list[DurableClaim] = Field(description="Claims for later campaigns' prior insights; may be empty.")
    answers: list[_PlannedAnswer] = Field(description="One per warning code raised on the recommended arm.")


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

    reference: Sequence[ChartReference] = Field(max_length=1, description="At most one reference line.")
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


class Section(_PlannedSection):
    """An evidence section as stored, its chart with code's facts."""

    chart: ChartSpec


class ReportPlan(_PlanCore):
    """The stored plan: the planner's :class:`PlannedReport` completed with the fields code fills, which the renderer
    draws."""

    model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True, serialize_by_alias=True)

    schema_id: Literal["report-plan/1"] = Field(alias="schema")
    signals: Signals
    shape: _Shape
    masthead: Masthead
    arms: list[PlanArm] = Field(min_length=1)
    show_arms_key: bool
    flag: list[Flag] = Field(max_length=1, description="The flag, or none when nothing needs disclosing.")
    options: Options
    summary: Sequence[ChartSpec] = Field(max_length=3, description="Summary charts beside the options table.")
    sections: Sequence[Section] = Field(description="The evidence sections, in reading order.")
    explore: list[Explore] = Field(max_length=1, description="The explore part, or none.")
    full_results: list[_FullResultsTable]
    durable_claims: list[DurableClaim] = Field(description="Claims for later campaigns' prior insights; may be empty.")
    answers: list[WarningAnswer] = Field(description="One per warning code raised on the recommended arm.")


def plan_recommended_arm(plan: ReportPlan) -> str | None:
    """The arm id of the plan's ``recommended`` option, or None when the plan recommends nothing.

    One reading for the analysis the plan is stored with (its recommended variant) and the page that draws it (the
    arm its charts stress).

    :param plan: The plan.
    :ptype plan: ReportPlan
    :return: The arm id, or None.
    :rtype: str | None
    """
    return next((row.option for row in plan.options.rows if row.role == "recommended"), None)


# --- the planner's two calls ---------------------------------------------------------------------------------------


class StructureChart(_Plan):
    """A chart as the structure call chooses it: kind, what it reads, a reference line and the one thing it stresses.
    Its title is prose, written by the prose call."""

    kind: _PlanChartKind
    query: ChartQuery
    reference: list[_PlannedChartReference] = Field(max_length=1, description="At most one reference line.")
    emphasis: str = Field(description="The one arm or row bolded or ringed, by id; empty for none.")


class _StructureSection(_Plan):
    """An evidence section as the structure call chooses it: its id, which its prose keys address, and its chart."""

    id: _PlanId
    chart: StructureChart


class _StructureControl(_Plan):
    """An explore control as the structure call chooses it: everything but its title, which is prose."""

    id: _PlanId
    kind: _ControlKind
    binds: str = Field(
        description="The criterion it replaces: margin, floor:<id>, p95, price_per_point, weighting, lever:<factor>."
    )
    min: float
    max: float
    step: float
    default: float
    options: list[str]


class _StructureExplore(_Plan):
    """The explore part's controls, as the structure call chooses them."""

    controls: list[_StructureControl] = Field(min_length=1, max_length=3)


class _StructureClaim(_Plan):
    """A durable claim's skeleton: the figure keys it rests on and its confidence. Its text is prose."""

    confidence: Confidence
    keys: list[_PlanId] = Field(min_length=1, description="The figure keys it rests on, as paths without filters.")


class _CaveatLanding(_Plan):
    """Where the structure lands one caveat: in the decision's confidence (its clause is prose), beside a decision
    slot's claim, beside a section's finding, or in a Method pair."""

    caveat: _CaveatId = Field(description="The caveat's id, as code lists it.")
    at: _PlanId = Field(
        description="`confidence`, `decision.<slot key>`, `sections.<section id>` or `method.<pair index from 0>`."
    )


class PlanStructure(_Plan):
    """The planner's first reply: every choice the page makes and no prose (:func:`structure_response_format`)."""

    mode: _Mode = Field(description="One of the modes code allows for this input.")
    headline_kind: _HeadlineKind
    confidence: _ConfidenceForm = Field(
        description="high: no open report-wide caveat; high_if_immaterial: every open one landed at `confidence`; low."
    )
    caveat_landings: list[_CaveatLanding] = Field(
        description="Where each caveat code lists as open lands, each exactly once; none for one code checked."
    )
    options: _PlannedOptions
    summary: list[StructureChart] = Field(max_length=3, description="Summary charts beside the options table.")
    sections: list[_StructureSection] = Field(description="The evidence sections, in reading order.")
    explore: list[_StructureExplore] = Field(max_length=1, description="The explore part, or none.")
    method_items: int = Field(ge=3, le=9, description="How many term and definition pairs Method holds.")
    durable_claims: list[_StructureClaim] = Field(description="Claims for later campaigns; may be empty.")


class _ProseItem(_Plan):
    """One prose slot the structure created, by its key, and its text."""

    key: _PlanId = Field(description="The slot's key, from the list of keys this call is asked for.")
    text: ModelProse = Field(description="The slot's text, figures by key; empty only where the slot is optional.")


class PlanProse(_Plan):
    """The planner's second reply: the page's prose, one item per key (:func:`prose_response_format`)."""

    prose: list[_ProseItem]


def structure_response_format() -> dict[str, Any]:
    """The strict ``json_schema`` directive that asks the planner for exactly :class:`PlanStructure`.

    :return: The OpenAI-shaped directive.
    :rtype: dict[str, Any]
    """
    return strict_response_format(_STRUCTURE_SCHEMA_NAME, PlanStructure.model_json_schema())


def prose_response_format() -> dict[str, Any]:
    """The strict ``json_schema`` directive that asks the planner for exactly :class:`PlanProse`.

    :return: The OpenAI-shaped directive.
    :rtype: dict[str, Any]
    """
    return strict_response_format(_PROSE_SCHEMA_NAME, PlanProse.model_json_schema())


__all__ = [
    "REPORT_PLAN_SCHEMA",
    "CaveatClause",
    "ChartQuery",
    "ChartReference",
    "ChartSpec",
    "Decision",
    "DecisionConfidence",
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
    "PlanProse",
    "PlanStructure",
    "PlannedChart",
    "PlannedReport",
    "ReportPlan",
    "Section",
    "Signals",
    "StructureChart",
    "WarningAnswer",
    "WarningRef",
    "plan_recommended_arm",
    "prose_response_format",
    "structure_response_format",
]
