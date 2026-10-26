"""The report plan's checks: code classifies the eval, and refuses a plan whose parts disagree with each other or with
the report input.

:mod:`threetears.evals.kernel.report_plan` checks each field on its own (lengths, enums, list bounds). This module
checks what only the whole plan and its report input can show, and every refusal is a :class:`PlanRefusal`, a
:class:`~threetears.evals.analysis.errors.SoundnessRefusal` the planner's repair round feeds back. All the faults a
plan holds are found in one pass and refused together, each named by its field path (``decision.slots[2].key``), so
one repair round can mend every one of them.

**Code classifies (GENERATOR §1).** :func:`classify` reads the signals off the report input and looks the shape up.
It is a lookup, read top to bottom, and the first row that matches is the shape:

====  ===============================================================================================
E     the candidates are ordered versions (``meta.design_shape`` is ``ordered_versions``)
C     a score is a class label, or the input carries a target matrix
B     the candidates are a factorial grid (``factors`` with ``configs``, or ``design_shape`` ``factorial``)
A     the campaign declares a hard gate, with or without a control
D     a control, read on a margin or on no declared rule
A     anything else: an unordered set of candidates
====  ===============================================================================================

The criteria signal wins the one conflict the lookup can meet, a control under hard gates, which is candidate
selection (A), not a forest of changes (D). A factorial grid or a classifier keeps its own shape whatever gates it
declares, since its spine (a gate-space scatter, a config lattice) is what draws those gates.

The signals: **candidate structure** is ordered versions, then a factorial grid, then one control plus candidates,
else unordered. **Score type** is a class label where any measure is graded by one; mixed where pass/fail and
ordinal grades both occur (judged scales and measure grades together); graded where only ordinal ones do; else
pass/fail (a trial passes or fails). **Strata** is a target matrix; else the named tasks (six or fewer read as
tasks, more as many strata); else the strata, counted across every stratification (none for fewer than two, tasks
for two to six, many strata past six). **Criteria** is hard gates where a criterion is a ``gate``; a margin where an
aspect declares one, a floor was read, or a criterion is a floor or a guardrail; else none declared. A bar alone
declares none: it warns and decides nothing. **Cross-cutting** aspects are the tasks marked so. The **cost axis** is
logarithmic when the arms' cost per 1,000 replies spans more than ten times.

**Code reads the spine (GENERATOR §1).** :func:`spine` looks the shape's spine chart up (A a gate matrix, B a
gate-space scatter, C a config lattice, D a forest, E lines) and names the parts of its query the input settles, which
the planner copies:

====  ==============================================================================================================
A     ``gate_matrix`` over the ``tasks`` where the input names tasks
B     ``gate_scatter`` over the ``configs``, its two axes the measures the gates read where they read exactly two
C     ``lattice`` over the ``configs``, its metric the measure of the first gate read over the whole comparison
D     ``forest`` of the asked aspect's ``readings`` with a margin line where it declares one; on a measured question
      (no candidate separated by an asked reading, one by a contrast on an asked aspect) of those ``contrasts``
      and their measures; else of the ``contrasts``
E     ``lines`` over the ``tasks``, else the ``strata``
====  ==============================================================================================================

A part the input does not settle is the planner's: a source where it holds none of those (the planner picks one the
kind draws), the measures where no rule names them, and on every spine the arms and their order, strata, limit,
transform, small multiples and emphasis. Some of what a spine draws is no query field at all: a gate matrix's binding
reasons and cheapest passing cell, a gate scatter's gate region and one path per factor level, a lattice's row and
column factors and its tiles' cost and latency. The renderer reads those off the criteria and the configs (P4).

**The checks** (:func:`validate_plan`), each refusing at the field it names:

- *Facts code filled*: ``signals`` and ``shape`` are :func:`classify`'s.
- *The spine*: the first evidence section's chart is :func:`spine`'s kind, with every query field and reference line
  code fixed; a plan with no evidence section is refused.
- *Allowed mode*: ``mode`` is one :func:`~threetears.evals.analysis.report_decision.allowed_modes` allows, and a
  decide plan whose input has one dominant survivor recommends that one.
- *Mode*: ``headline_kind`` suits ``mode`` (decide: decision; tradeoff: trade; no call: no_call). The decision slots
  are the mode's, in its order: decide ``recommendation, why_not, watch, next``; tradeoff ``the_trade,
  where_it_flips, our_lean, what_settles_it``; a no call takes decide mode's four, since GENERATOR §1.1 has it say
  what is missing (``recommendation``: what to do meanwhile), why no option survives (``why_not``), the risk
  (``watch``) and what would settle it (``next``). Option roles: decide has exactly one ``recommended`` row, first,
  and no ``lean``; tradeoff no ``recommended`` and at most one ``lean``; a no call neither.
- *The recommendation* (the memo's refusals, ported onto the report input): a measurement (one arm) recommends
  nothing; a recommended option is an arm the input holds (a change's candidate or its control); a choice's pick that
  no pair reading shows ahead takes the ``low`` confidence form, unless the campaign declares a gate (a criterion of
  kind ``gate`` or ``guardrail``) and the pick is the one dominant survivor
  (:func:`~threetears.evals.analysis.report_decision.dominant_options`): then gates and cost, not pair readings, settled
  it, and the confidence states what they earn; a change switches to a candidate only when some aspect
  is asked, no guardrail on it breached, and it is shown better (a confirmatory ``better`` reading, or an
  ``improved`` contrast on an asked aspect) or no worse (every floor on every asked aspect ``held``, each declaring
  one). A change in tradeoff mode or a no call recommends nothing: GENERATOR §1.1's modes replace the memo's "a
  change always recommends".
- *Survivors*: a ``recommended`` or ``lean`` row naming an arm names one that survives every gate and guardrail
  (:func:`~threetears.evals.analysis.report_decision.surviving_options`), in every mode: a decide plan picking by a
  ranking rule, a tradeoff's lean, a change's candidate shown better that failed a gate, and a choice's low-confidence
  pick a gate removed are all refused off the survivors.
- *Answers*: every warning :func:`raised_warnings` raises on the recommended arm has exactly one answer, and no
  answer names a warning not raised.
- *Confidence* (GENERATOR §2.5): plain ``high`` only while no open report-wide caveat remains, and listing none;
  ``high_if_immaterial`` lists at least one caveat and every open report-wide one, none of them stated elsewhere;
  ``low`` is allowed whatever the caveats. Every listed caveat has a non-empty clause (``material_if``).
- *Flag*: holds ``no_rule`` exactly when it applies; otherwise only report-wide caveats, each once; a flag on an input
  with no report-wide disclosure is refused.
- *Coverage*: every caveat :func:`~threetears.evals.analysis.report_decision.report_caveats` lists lands exactly
  once, at a place its scope allows, read off the plan's structure and never its prose: a report-wide caveat in the
  confidence's clauses or the flag; a scoped one in the ``caveats`` of the decision slot or section whose claim it
  qualifies, or the confidence; a method-level one (latency, an unreported observation, an apparatus confound not
  report-wide) in a Method pair's ``caveats``, beside a claim, or the confidence. A caveat code proved immaterial
  (:func:`~threetears.evals.analysis.report_decision.caveat_materiality`) lands nowhere: code renders its checked
  line. An id that names no caveat of the input is refused wherever it lands.
- *Charts*: each chart's kind suits its source (a forest reads readings or contrasts, every row with an interval; a
  lattice reads configs; …); its query names only arms, aspects, measures and strata the input holds, and selects at
  least one row; its reference key and its emphasis name something present. The summary charts show the options
  table's arm rows in its order, a change-from-control chart (readings, contrasts, or ``arm_minus_control``) leaving
  the control out.
- *Ids*: section ids, explore control ids and option rows are each unique.
- *Figure keys*: every key in every prose field, every durable claim's ``keys`` and every chart reference parses,
  resolves against the report input and suits its filters (:mod:`~threetears.evals.analysis.report.figure_keys`).
- *D3, no typed digit*: :func:`~threetears.evals.analysis.report.figure_keys.literal_numbers` over every prose
  field. **The prose fields are the plan's fields typed** :data:`~threetears.evals.schema.prose.ModelProse`, found
  by walking the plan's model, so a prose field added later is covered without a list to keep: the headline, the
  lede, the flag's title and text, each decision slot's text, each confidence clause, each answer's text, each durable claim's text, each
  section's eyebrow, headline, takeaway and caption, every chart's title, the explore part's title and
  ``recomputes``, each control's title, and each method term and text. The masthead is code's and not prose.
  **Explore option labels are layout**: printed as written, so a label holds no figure key (a brace is refused) and,
  since a reader still reads a digit there as a number, no typed digit either. A control's ``min``, ``max``,
  ``step`` and ``default`` are the control's own scale, numbers and not prose, and are not checked here.

- *Watch names the weakest margin*: a decide plan's Watch slot cites the weakest margin
  (:func:`~threetears.evals.analysis.report_margins.weakest_margin`) by at least one ``weakest.*`` figure key, where the
  input reads any figure against a criterion, margin or floor. Which margin is weakest is code's fact; what Watch says
  about it is the prose's.
- *Reader names above Full results* (GENERATOR §6.3, §8 item 9): no prose field above Full results (every one but
  Method's and the durable claims', and the explore option labels included) types an internal id: an aspect,
  measure, judged dimension, stratum, task or criterion id that reads as an id (it holds an underscore, a dot, a
  colon, a slash, an equals sign, a bracket or a quote) and differs from its reader name
  (:func:`~threetears.evals.analysis.report.figure_keys.reader_name`). It is matched as D3 matches a name: keys set
  aside, then every name a reader is shown (arms, reader names), then the id where a non-word character or the text's
  edge bounds it, in any case. A plain-word id (``billing``) reads as a word and is the rubric's to judge. Nor does
  such prose quote the campaign's question (``{meta.question}``): the campaign writes it in its own words and ids,
  which no reader name can replace, so the page says what it asks in reader names instead.

**Warnings** (:func:`raised_warnings`), what code raises on an arm and the plan must answer: a floor read ``broken``
(``floor_broken``) or ``unclear`` (``floor_unclear``: GENERATOR §3's borderline, whose answer says what sample
would settle it) on the arm, by its aspect and ``floor`` key; an aspect where a pair reading shows another arm ahead
of it (``behind_on``), by aspect and that arm's id; an absolute bar it ``missed`` (``bar_missed``), by aspect and
measure.

**Disclosures** (:func:`implied_disclosures`), the report-wide disclosures a plan must make (GENERATOR §2.4):
``no_rule`` when the plan is in tradeoff mode and the campaign declared no rule ranking the axes (GENERATOR §4.2:
the flag says "no rule set in advance" and names who makes the call). The report input's one declared rule is a
contrast family's ``merit_axes`` (the merit priority); it carries no exchange rate, so a tradeoff with every family's
merit priority empty, or no family at all, is the case. Decide mode and a no call never disclose it. Then every
report-wide caveat code did not prove immaterial, by id (:mod:`~threetears.evals.analysis.report_decision` defines
the ids and which caveats are report-wide: every late criterion, an uncalibrated judge, and a confound, apparatus
confound or short run touching every comparison). ``no_rule`` is always the flag's; each caveat lands once, in the
decision's confidence or, where the confidence does not list it, the flag.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.analysis.errors import PlanRefusal, UnresolvableReference
from threetears.evals.analysis.report.figure_keys import (
    PERCENTILE_NAMES,
    WEAKEST_PREFIX,
    format_figure,
    keys_in,
    literal_numbers,
    parse_key,
    reader_names,
    render,
    resolve,
)
from threetears.evals.analysis.report_decision import (
    Caveat,
    Materiality,
    allowed_modes,
    caveat_materiality,
    control_arm,
    dominant_options,
    floors_not_held,
    report_caveats,
    report_kind,
    separated_by,
    separating_contrasts,
    shown_better,
    surviving_options,
)
from threetears.evals.analysis.report_input import Aspect, Judged, Measure, ReportInput
from threetears.evals.analysis.report_margins import criterion_place, weakest_margin
from threetears.evals.kernel.report_plan import ChartSpec, OptionRow, ReportPlan, Signals, WarningRef
from threetears.evals.schema.prose import field_is_prose

__all__ = [
    "MODE_SLOTS",
    "SPINE_KINDS",
    "Spine",
    "classify",
    "implied_disclosures",
    "raised_warnings",
    "spine",
    "validate_plan",
]

#: The ``meta.design_shape`` that marks the candidates as ordered versions of one thing.
_ORDERED_VERSIONS = "ordered_versions"
#: The ``meta.design_shape`` that marks a factorial grid.
_FACTORIAL = "factorial"
#: The most named tasks or strata read as tasks; more read as many strata.
_FEW_STRATA = 6
#: The spread of cost across arms past which the cost axis is logarithmic.
_LOG_SPREAD = 10.0
#: The caveat kinds Method may carry: method-level ones (R1 of the 2026-10-25 review).
_METHOD_KINDS = frozenset({"apparatus", "latency", "unreported"})

#: The headline kind each mode takes.
_HEADLINE_KIND = {"decide": "decision", "tradeoff": "trade", "no_call": "no_call"}
#: The decision slots each mode takes, in order. A no call takes decide mode's.
_DECIDE_SLOTS = ("recommendation", "why_not", "watch", "next")
MODE_SLOTS: dict[str, tuple[str, ...]] = {
    "decide": _DECIDE_SLOTS,
    "tradeoff": ("the_trade", "where_it_flips", "our_lean", "what_settles_it"),
    "no_call": _DECIDE_SLOTS,
}

#: The sources each chart kind can draw (GENERATOR §5.1). A table draws any.
_KIND_SOURCES: dict[str, frozenset[str]] = {
    "forest": frozenset({"readings", "contrasts"}),
    "bars": frozenset({"measures", "judged", "scoreboard", "strata", "tasks", "configs", "human_pref"}),
    "diverging_bars": frozenset({"readings", "contrasts", "measures", "judged", "strata", "tasks", "human_pref"}),
    "lines": frozenset({"measures", "judged", "strata", "tasks"}),
    "scatter_frontier": frozenset({"measures", "judged", "scoreboard", "configs", "tasks"}),
    "gate_scatter": frozenset({"measures", "judged", "configs", "tasks"}),
    "dot_interval": frozenset({"readings", "contrasts", "measures", "judged", "floors", "tasks"}),
    "range_bars": frozenset({"measures"}),
    "gate_matrix": frozenset({"tasks", "floors", "measures", "configs"}),
    "lattice": frozenset({"configs"}),
    "heatmap": frozenset({"cells", "strata", "tasks"}),
    "diff_heatmap": frozenset({"cells", "strata", "tasks"}),
    "confusion": frozenset({"confusion"}),
    "examples": frozenset({"examples"}),
}
#: Sources whose chart shows each candidate's change from the control, so it leaves the control out.
_FROM_CONTROL = frozenset({"readings", "contrasts"})
#: Sources a query names no measure of.
_NO_MEASURES = frozenset({"readings", "examples", "human_pref"})
#: A scoreboard's figures, which a scoreboard chart's measures name.
_SCOREBOARD_FIGURES = frozenset(
    {"ran", "usable", "passed", "cases", "lost", "failed", "reply_cost_usd", "test_cost_usd"}
)

#: One fault: the field path it sits at and why it is refused.
_Fault = tuple[str, str]


# --- classification ------------------------------------------------------------------------------------------------


def classify(report_input: ReportInput) -> tuple[Signals, str]:
    """GENERATOR §1's signals read off the report input, and the shape they look up (this module's lookup table).

    :param report_input: The report input the plan is written against.
    :ptype report_input: ReportInput
    :return: The signals, and the shape (``A`` to ``E``).
    :rtype: tuple[Signals, str]
    """
    control = control_arm(report_input)
    signals = Signals.model_validate(
        {
            "candidate_structure": _structure(report_input, control),
            "baseline": "control" if control is not None else "none",
            "score_type": _score_type(report_input),
            "strata": _strata_signal(report_input),
            "criteria": _criteria(report_input),
            "cross_cutting": [task.id for task in report_input.tasks if task.cross_cutting],
            "log_cost_axis": _log_cost_axis(report_input),
        }
    )
    return signals, _shape(signals)


def _structure(report_input: ReportInput, control: str | None) -> str:
    design = report_input.meta.design_shape
    if design == _ORDERED_VERSIONS:
        return "ordered_versions"
    if (report_input.factors and report_input.configs) or design == _FACTORIAL:
        return "factorial"
    return "control_plus_candidates" if control is not None else "unordered"


def _score_type(report_input: ReportInput) -> str:
    grades: set[str] = {judged.scale for judged in report_input.judged.values() if judged.scale is not None}
    grades |= {measure.grade for measure in report_input.measures.values() if measure.grade is not None}
    if "class_label" in grades:
        return "class_label"
    if {"pass_fail", "ordinal"} <= grades:
        return "mixed"
    return "graded" if "ordinal" in grades else "pass_fail"


def _strata_signal(report_input: ReportInput) -> str:
    if report_input.target_matrix is not None:
        return "target_matrix"
    if report_input.tasks:
        return "tasks" if len(report_input.tasks) <= _FEW_STRATA else "many_strata"
    count = len({stratum.id for stratum in report_input.strata})
    if count < 2:
        return "none"
    return "tasks" if count <= _FEW_STRATA else "many_strata"


def _criteria(report_input: ReportInput) -> str:
    kinds = {criterion.kind for criterion in report_input.criteria}
    if "gate" in kinds:
        return "hard_gates"
    if (
        any(aspect.margin is not None for aspect in report_input.aspects)
        or report_input.floors
        or kinds & {"floor", "guardrail"}
    ):
        return "margin"
    return "none_declared"


def _log_cost_axis(report_input: ReportInput) -> bool:
    costs: list[float] = []
    for arm in report_input.arms:
        try:
            figure = resolve(f"cost_per_1k.{arm.id}", report_input)
        except UnresolvableReference:
            continue  # NOSILENT: an arm that states no cost spans nothing; the cost axis is read over the others
        if isinstance(figure.value, float) and figure.value > 0:
            costs.append(figure.value)
    return len(costs) > 1 and max(costs) / min(costs) > _LOG_SPREAD


def _shape(signals: Signals) -> str:
    if signals.candidate_structure == "ordered_versions":
        return "E"
    if signals.score_type == "class_label" or signals.strata == "target_matrix":
        return "C"
    if signals.candidate_structure == "factorial":
        return "B"
    if signals.criteria == "hard_gates":
        return "A"
    return "D" if signals.baseline == "control" else "A"


# --- the spine -----------------------------------------------------------------------------------------------------

#: GENERATOR §1: the spine chart each shape opens its evidence on.
SPINE_KINDS: dict[str, str] = {"A": "gate_matrix", "B": "gate_scatter", "C": "lattice", "D": "forest", "E": "lines"}


class Spine(BaseModel):
    """The shape's spine (GENERATOR §1): the chart the first evidence section draws, a fact code reads off the input.

    ``kind`` is the shape's spine chart. The query fields code can name are fixed with it, and the planner copies them:
    ``source`` (empty where the input holds no source the spine reads, and the planner picks one the kind draws),
    ``aspect`` (empty for none), ``measures`` (None where the measures are the planner's to choose) and the reference
    line kinds. Everything else on the chart (its arms and their order, strata, limit, transform, small multiples,
    emphasis and title) is the planner's.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    shape: str = Field(description="The shape the spine is read for.")
    kind: str = Field(description="The spine chart's kind.")
    source: str = Field(description="What it reads; empty where the planner picks a source the kind draws.")
    aspect: str = Field(description="The aspect it reads; empty for none.")
    measures: tuple[str, ...] | None = Field(description="Its measures, in order; None where the planner picks them.")
    reference: tuple[str, ...] = Field(description="The kinds of its reference lines, in order; empty for none.")

    def as_fact(self) -> dict[str, Any]:
        """The spine as the planner is told it: its kind, the query fields code fixes, and its reference lines."""
        query: dict[str, Any] = {"source": self.source} if self.source else {}
        if self.source:
            query["aspect"] = self.aspect
        if self.measures is not None:
            query["measures"] = list(self.measures)
        return {"kind": self.kind, "query": query, "reference": [{"kind": kind} for kind in self.reference]}

    def said(self) -> str:
        """The spine in words, for a refusal."""
        parts = [f"a {self.kind!r}"]
        if self.source:
            parts.append(f"of {self.source!r}")
        if self.aspect:
            parts.append(f"on aspect {self.aspect!r}")
        if self.measures:
            parts.append(f"over the measures {list(self.measures)}")
        parts.append(
            "with a " + " and a ".join(self.reference) + " line" if self.reference else "with no reference line"
        )
        return " ".join(parts)


def spine(report_input: ReportInput) -> Spine:
    """The spine chart of the input's shape, with the query fields code can name (this module's Spine).

    :param report_input: The report input the plan is written against.
    :ptype report_input: ReportInput
    :return: The spine.
    :rtype: Spine
    """
    _, shape = classify(report_input)
    kind = SPINE_KINDS[shape]
    gates = [criterion for criterion in report_input.criteria if criterion.kind == "gate"]
    figures = set(report_input.measures) | set(report_input.judged)
    source, aspect = "", ""
    measures: tuple[str, ...] | None = None
    reference: tuple[str, ...] = ()
    if shape == "A":
        source = "tasks" if report_input.tasks else ""
    elif shape == "B":
        source = "configs" if report_input.configs else ""
        axes = tuple(dict.fromkeys(gate.measure for gate in gates))
        measures = axes if len(axes) == 2 and set(axes) <= figures else None
    elif shape == "C":
        source = "configs" if report_input.configs else ""
        overall = [gate.measure for gate in gates if criterion_place(gate) == "overall"]
        measures = (overall[0],) if overall and overall[0] in figures else None
    elif shape == "D":
        source, aspect, measures, reference = _forest_spine(report_input)
    else:
        source = "tasks" if report_input.tasks else ("strata" if report_input.strata else "")
    if source == "readings":
        measures = ()
    return Spine(shape=shape, kind=kind, source=source, aspect=aspect, measures=measures, reference=reference)


def _forest_spine(report_input: ReportInput) -> tuple[str, str, tuple[str, ...] | None, tuple[str, ...]]:
    """Shape D's forest: what it reads, on which aspect, which measures, and its reference lines.

    The asked aspect's readings (the one asked aspect read with intervals, where several are asked), with the margin
    line where it declares one; else, on a measured question, the
    contrasts that decided it: where no candidate is separated by an asked reading and at least one is separated by a
    contrast on an asked aspect (:func:`~threetears.evals.analysis.report_decision.separated_by`, the test the modes
    read), the forest reads those contrasts, their measures, on their aspect where they share one, with no margin line
    (the margin is the readings' own scale); else any contrasts; else nothing code names."""
    control = control_arm(report_input)
    candidates = [arm.id for arm in report_input.arms if arm.id != control]
    separation = [separated_by(report_input, arm) for arm in candidates]
    if not any("reading" in found for found in separation) and any("contrast" in found for found in separation):
        deciding = [contrast for arm in candidates for contrast in separating_contrasts(report_input, arm)]
        aspects = {contrast.aspect or "" for contrast in deciding}
        measures = tuple(dict.fromkeys(contrast.measure for contrast in deciding))
        return "contrasts", aspects.pop() if len(aspects) == 1 else "", measures, ()
    read = [
        aspect for aspect in report_input.aspects if aspect.asked and any(r.within_interval for r in aspect.readings)
    ]
    asked = read[0] if len(read) == 1 else _aspect_of("", report_input)
    margin = ("margin",) if asked is not None and asked.margin is not None else ()
    if asked is not None and any(reading.within_interval for reading in asked.readings):
        return "readings", asked.id, (), margin
    if any(contrast.interval is not None for contrast in report_input.contrasts):
        return "contrasts", "", None, margin
    return "", "", None, ()


def _spine_faults(plan: ReportPlan, report_input: ReportInput) -> Iterator[_Fault]:
    """The first evidence section draws the spine code read off the input, with every query field code fixed."""
    wanted = spine(report_input)
    told = f"shape {wanted.shape}'s spine, code's fact, is {wanted.said()}"
    if not plan.sections:
        yield "sections", f"hold no evidence section, and the first draws the spine: {told}"
        return
    chart = plan.sections[0].chart
    if chart.kind != wanted.kind:
        yield "sections[0].chart.kind", f"is {chart.kind!r}, and the first evidence section draws the spine: {told}"
        return
    query = chart.query
    if wanted.source and query.source != wanted.source:
        yield "sections[0].chart.query.source", f"is {query.source!r}, and {told}"
    if wanted.source and query.aspect != wanted.aspect:
        yield "sections[0].chart.query.aspect", f"is {query.aspect!r}, and {told}"
    if wanted.measures is not None and tuple(query.measures) != wanted.measures:
        yield "sections[0].chart.query.measures", f"are {list(query.measures)}, and {told}"
    if tuple(reference.kind for reference in chart.reference) != wanted.reference:
        yield "sections[0].chart.reference", f"draws {[r.kind for r in chart.reference]}, and {told}"


# --- what code raises and implies ----------------------------------------------------------------------------------


def raised_warnings(report_input: ReportInput, arm: str) -> list[WarningRef]:
    """Every warning code raises on an arm, which a plan recommending it must answer (this module's Warnings).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param arm: The arm's id.
    :ptype arm: str
    :return: Each warning once, floors first, then pair readings, then bars.
    :rtype: list[WarningRef]
    """
    raised: dict[tuple[str, str, str], None] = {}
    for floor in report_input.floors:
        if floor.arm == arm and floor.status in ("broken", "unclear"):
            raised.setdefault((f"floor_{floor.status}", floor.aspect, floor.floor))
    for aspect in report_input.aspects:
        for pair in aspect.pairs:
            if arm in pair.arms and pair.ahead is not None and pair.ahead != arm:
                raised.setdefault(("behind_on", aspect.id, pair.ahead))
    for bar in report_input.bars:
        if bar.arm == arm and bar.status == "missed":
            raised.setdefault(("bar_missed", bar.aspect, bar.measure))
    return [WarningRef.model_validate({"kind": k, "aspect": a, "name": n}) for k, a, n in raised]


def implied_disclosures(report_input: ReportInput, mode: str) -> list[str]:
    """The report-wide disclosures a plan in ``mode`` must make (this module's Disclosures): ``no_rule`` where it
    applies, then every report-wide caveat code did not prove immaterial. ``no_rule`` is the flag's; each caveat lands
    once, in the decision's confidence or, under a ``low`` confidence, in the flag.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param mode: The plan's mode: ``decide``, ``tradeoff`` or ``no_call``.
    :ptype mode: str
    :return: Each disclosure by id: no rule, then the caveats in :func:`report_caveats`' order.
    :rtype: list[str]
    """
    implied: list[str] = []
    if mode == "tradeoff" and not any(family.merit_axes for family in report_input.contrast_families):
        implied.append("no_rule")
    materiality = caveat_materiality(report_input)
    implied += [
        caveat.id
        for caveat in report_caveats(report_input)
        if caveat.reach == "report" and not materiality[caveat.id].immaterial
    ]
    return implied


# --- the plan check ------------------------------------------------------------------------------------------------


def validate_plan(plan: ReportPlan, report_input: ReportInput) -> None:
    """Refuse a plan whose parts disagree with each other or with its report input (this module's checks).

    :param plan: The stored plan: the planner's part with the fields code fills.
    :ptype plan: ReportPlan
    :param report_input: The report input it was written against.
    :ptype report_input: ReportInput
    :raises PlanRefusal: The plan holds one or more faults; each names its field path and why.
    """
    faults: list[_Fault] = []
    faults += _classification_faults(plan, report_input)
    faults += _spine_faults(plan, report_input)
    faults += _mode_faults(plan)
    faults += _allowed_mode_faults(plan, report_input)
    faults += _id_faults(plan)
    faults += _recommendation_faults(plan, report_input)
    faults += _survivor_faults(plan, report_input)
    faults += _answer_faults(plan, report_input)
    faults += _confidence_faults(plan, report_input)
    faults += _flag_faults(plan, report_input)
    faults += _coverage_faults(plan, report_input)
    faults += _chart_faults(plan, report_input)
    faults += _key_faults(plan, report_input)
    faults += _literal_faults(plan, report_input)
    faults += _watch_faults(plan, report_input)
    faults += _reader_id_faults(plan, report_input)
    if faults:
        raise PlanRefusal(faults)


def _classification_faults(plan: ReportPlan, report_input: ReportInput) -> Iterator[_Fault]:
    signals, shape = classify(report_input)
    for name in Signals.model_fields:
        mine, code = getattr(plan.signals, name), getattr(signals, name)
        if mine != code:
            yield f"signals.{name}", f"is {mine!r}, and code reads {code!r} off the report input"
    if plan.shape != shape:
        yield "shape", f"is {plan.shape!r}, and the signals look up shape {shape!r}"


def _mode_faults(plan: ReportPlan) -> Iterator[_Fault]:
    mode = plan.mode
    if plan.headline_kind != _HEADLINE_KIND[mode]:
        yield "headline_kind", f"is {plan.headline_kind!r}, and a {mode} plan's headline is a {_HEADLINE_KIND[mode]!r}"
    keys = tuple(slot.key for slot in plan.decision.slots)
    if keys != MODE_SLOTS[mode]:
        yield (
            "decision.slots",
            f"are {list(keys)}, and a {mode} plan's slots are {list(MODE_SLOTS[mode])}, in that order",
        )
    rows = plan.options.rows
    recommended = [index for index, row in enumerate(rows) if row.role == "recommended"]
    lean = [index for index, row in enumerate(rows) if row.role == "lean"]
    if mode == "decide":
        if len(recommended) != 1 or recommended[0] != 0:
            yield (
                "options.rows",
                (
                    f"mark rows {recommended} recommended, and a decide plan recommends exactly one option, its first row"
                ),
            )
        for index in lean:
            yield f"options.rows[{index}].role", "is 'lean', and a decide plan recommends; a lean is tradeoff mode's"
        return
    for index in recommended:
        yield f"options.rows[{index}].role", f"is 'recommended', and a {mode} plan recommends no option"
    if mode == "tradeoff" and len(lean) > 1:
        yield "options.rows", f"mark rows {lean} 'lean', and a tradeoff plan leans toward one option at most"
    if mode == "no_call":
        for index in lean:
            yield f"options.rows[{index}].role", "is 'lean', and a no call leans toward no option"


def _id_faults(plan: ReportPlan) -> Iterator[_Fault]:
    yield from _repeated("sections", [section.id for section in plan.sections], "section id")
    yield from _repeated("options.rows", [row.option for row in plan.options.rows], "option")
    for index, explore in enumerate(plan.explore):
        yield from _repeated(f"explore[{index}].controls", [control.id for control in explore.controls], "control id")


def _repeated(path: str, ids: Sequence[str], what: str) -> Iterator[_Fault]:
    for name, count in Counter(ids).items():
        if count > 1:
            yield path, f"name the {what} {name!r} {count} times; each is named once"


def _recommended(plan: ReportPlan) -> OptionRow | None:
    return next((row for row in plan.options.rows if row.role == "recommended"), None)


def _recommendation_faults(plan: ReportPlan, report_input: ReportInput) -> Iterator[_Fault]:
    """The memo's recommendation refusals, read off the plan's recommended row and the report input."""
    row = _recommended(plan)
    if row is None:
        return
    index = plan.options.rows.index(row)
    path = f"options.rows[{index}].option"
    kind = report_kind(report_input)
    arms = [arm.id for arm in report_input.arms]
    if kind == "measurement":
        yield (
            f"options.rows[{index}].role",
            (
                f"recommends {row.option!r}, and this campaign measures one arm, so there is no arm to choose and nothing "
                "to recommend: make no call, and say what was measured and what it means"
            ),
        )
        return
    if row.option not in arms:
        yield path, f"is {row.option!r}, which is no arm; a recommended option is an arm, and the arms are {arms}"
        return
    if kind == "choice":
        ahead = any(pair.ahead == row.option for aspect in report_input.aspects for pair in aspect.pairs)
        if not ahead and not _settled_by_gates(report_input, row.option) and plan.decision.confidence.form != "low":
            yield (
                "decision.confidence.form",
                (
                    f"is {plan.decision.confidence.form!r}, and the plan picks {row.option!r}, which no pair reading shows ahead "
                    "of any arm, and no declared gate leaves it the one dominant survivor: a pick nothing separated rests "
                    "on cost or simplicity, so its confidence is 'low'. Lower it, pick an arm a pair reading shows ahead, "
                    "or make no call"
                ),
            )
        return
    yield from _adoption_faults(row.option, path, report_input)


def _survivor_faults(plan: ReportPlan, report_input: ReportInput) -> Iterator[_Fault]:
    """A recommended or lean row naming an arm names one that survives every gate and guardrail, in every mode."""
    arms = {arm.id for arm in report_input.arms}
    picked = [(index, row) for index, row in enumerate(plan.options.rows) if row.role in ("recommended", "lean")]
    if not any(row.option in arms for _, row in picked):
        return
    survivors = surviving_options(report_input)
    for index, row in picked:
        if row.option in arms and row.option not in survivors:
            yield (
                f"options.rows[{index}].option",
                (
                    f"is {row.option!r}, marked {row.role!r}, and it does not survive the campaign's gates and "
                    f"guardrails (the options that do are {survivors}): a plan recommends or leans toward a survivor "
                    "only, and says in Why not what kept this one out"
                ),
            )


def _settled_by_gates(report_input: ReportInput, arm: str) -> bool:
    """Whether the campaign's declared gates settle the pick: it declares a gate or guardrail, and ``arm`` is the one
    survivor at least as good as every other on cost, the asked aspect and every gate. Its basis is then the gates and
    cost, which code read, and not a pair reading that does not separate it."""
    gated = any(criterion.kind in ("gate", "guardrail") for criterion in report_input.criteria)
    return gated and dominant_options(report_input) == [arm]


def _adoption_faults(arm: str, path: str, report_input: ReportInput) -> Iterator[_Fault]:
    """A change's switch to a candidate: refused when exploratory, past a breached guardrail, or neither shown better
    nor shown no worse on the asked aspects (the memo's exploratory and unseparated adoption refusals)."""
    control = control_arm(report_input)
    if control is None or arm == control:
        return
    remedy = f"recommend the control, {control!r}, and say in `next` what would settle it"
    asked = [aspect for aspect in report_input.aspects if aspect.asked]
    if not asked:
        yield (
            path,
            (
                f"is {arm!r}, a candidate, and this campaign asks no question of any aspect, so it is exploratory: nothing "
                f"in it is a decision to switch on. Recommend the control, {control!r}, and say in `next` what declared "
                "question would settle the switch"
            ),
        )
        return
    breached = [
        f"{guardrail.get('name')} on aspect {guardrail.get('aspect')!r}"
        for guardrail in report_input.guardrails
        if guardrail.get("arm") == arm and guardrail.get("decision") == "breached"
    ]
    if breached:
        yield (
            path,
            (
                f"is {arm!r}, which breached the guardrail(s) {', '.join(breached)} (shown worse than the control by more "
                f"than the margin), and no gain elsewhere pays for a guardrail; {remedy}"
            ),
        )
        return
    if shown_better(report_input, arm):
        return
    open_floors = floors_not_held(report_input, arm, asked)
    if not open_floors:
        return
    yield (
        path,
        (
            f"is {arm!r}, and on the aspects the question asks about no confirmatory reading shows it better and no "
            f"contrast shows it improved. Nor is it shown no worse than its floors: {'; '.join(open_floors)}. "
            f"{remedy[:1].upper()}{remedy[1:]}"
        ),
    )


def _answer_faults(plan: ReportPlan, report_input: ReportInput) -> Iterator[_Fault]:
    row = _recommended(plan)
    known = row is not None and row.option in {arm.id for arm in report_input.arms}
    raised = raised_warnings(report_input, row.option) if row is not None and known else []
    wanted = {_warning_key(warning) for warning in raised}
    seen: Counter[tuple[str, str, str]] = Counter()
    for index, answer in enumerate(plan.answers):
        key = _warning_key(answer.warning)
        seen[key] += 1
        if key not in wanted:
            arm = f"on {row.option!r}" if row is not None else "with no option recommended"
            yield f"answers[{index}].warning", f"names {_said(key)}, which code did not raise {arm}"
        elif seen[key] == 2:
            yield f"answers[{index}].warning", f"answers {_said(key)} again; each warning gets exactly one answer"
    for warning in raised:
        key = _warning_key(warning)
        if not seen[key]:
            assert row is not None
            yield (
                "answers",
                (
                    f"hold no answer to {_said(key)}, raised on the recommended {row.option!r}: a warning only warns, so "
                    "the arm may still be recommended, but the plan says why it stands over each one"
                    + (", and what sample would settle it" if key[0] == "floor_unclear" else "")
                ),
            )


def _warning_key(warning: WarningRef) -> tuple[str, str, str]:
    return warning.kind, warning.aspect, warning.name


def _said(key: tuple[str, str, str]) -> str:
    kind, aspect, name = key
    return f"{kind} {name!r} on aspect {aspect!r}"


def _allowed_mode_faults(plan: ReportPlan, report_input: ReportInput) -> Iterator[_Fault]:
    """The mode is one code allows, and a decide plan recommends the one option that dominates, where one does."""
    allowed = allowed_modes(report_input)
    if plan.mode not in allowed:
        survivors = surviving_options(report_input)
        yield (
            "mode",
            (
                f"is {plan.mode!r}, and code allows only {sorted(allowed)} for this input: the options that survive "
                f"every gate and guardrail are {survivors}, and {_dominance_said(report_input)}"
            ),
        )
        return
    row = _recommended(plan)
    dominant = dominant_options(report_input)
    if plan.mode != "decide" or row is None or len(dominant) != 1 or row.option == dominant[0]:
        return
    if row.option in {arm.id for arm in report_input.arms}:
        yield (
            f"options.rows[{plan.options.rows.index(row)}].option",
            (
                f"is {row.option!r}, and {dominant[0]!r} is the one surviving option at least as good as every other on "
                f"every axis that matters (cost, the asked aspect, every gate): recommend {dominant[0]!r}"
            ),
        )


def _dominance_said(report_input: ReportInput) -> str:
    dominant = dominant_options(report_input)
    if len(dominant) == 1:
        return f"{dominant[0]!r} dominates them"
    if dominant:
        return f"{dominant} tie on every axis, so none dominates alone"
    return "none is at least as good as every other on every axis that matters"


def _confidence_faults(plan: ReportPlan, report_input: ReportInput) -> Iterator[_Fault]:
    """The confidence's form suits the open report-wide caveats, and each clause states its condition."""
    confidence = plan.decision.confidence
    known = {caveat.id for caveat in report_caveats(report_input)}
    for index, clause in enumerate(confidence.caveats):
        if clause.id in known and not clause.material_if.strip():
            yield (
                f"decision.confidence.caveats[{index}].material_if",
                f"is empty: the open caveat {clause.id!r} needs the condition that would make it material",
            )
    wide = [disclosure for disclosure in implied_disclosures(report_input, plan.mode) if disclosure != "no_rule"]
    listed = {clause.id for clause in confidence.caveats}
    if confidence.form == "high":
        if wide:
            yield (
                "decision.confidence.form",
                (
                    f"is 'high', and open caveats remain: {wide}. Say 'high_if_immaterial' and give each its clause, "
                    "or 'low' when the evidence is weak even if they are immaterial"
                ),
            )
        elif confidence.caveats:
            yield "decision.confidence.caveats", "list caveats, and a 'high' confidence is conditional on none"
        return
    if confidence.form != "high_if_immaterial":
        return
    if not confidence.caveats:
        yield (
            "decision.confidence.form",
            "is 'high_if_immaterial' and lists no open caveat: with none open, the confidence is 'high'",
        )
    landed = _landings(plan)
    if elsewhere := [item for item in wide if item not in listed and landed.get(item)]:
        yield (
            "decision.confidence.caveats",
            (
                f"leave out {elsewhere}, stated elsewhere ({', '.join(landed[elsewhere[0]])}): a 'high_if_immaterial' "
                "confidence lists every open report-wide caveat, each with what would make it material"
            ),
        )


def _flag_faults(plan: ReportPlan, report_input: ReportInput) -> Iterator[_Fault]:
    """The flag holds ``no_rule`` exactly when it applies, and otherwise only report-wide caveats."""
    implied = implied_disclosures(report_input, plan.mode)
    if not plan.flag:
        if "no_rule" in implied:
            yield "flag", f"is empty, and this {plan.mode} plan must disclose 'no_rule': no rule was set in advance"
        return
    discloses = plan.flag[0].discloses
    if not implied:
        yield (
            "flag[0]",
            (
                f"discloses {discloses}, and this {plan.mode} plan has no report-wide disclosure: drop the flag, and say "
                "any scoped caveat beside the claim it qualifies"
            ),
        )
        return
    yield from _repeated("flag[0].discloses", discloses, "disclosure")
    if "no_rule" in implied and "no_rule" not in discloses:
        yield "flag[0].discloses", f"leaves out ['no_rule'], which this {plan.mode} plan must disclose"
    if "no_rule" in discloses and "no_rule" not in implied:
        yield (
            "flag[0].discloses",
            (
                f"holds 'no_rule', and a {plan.mode} plan makes the call: only a tradeoff plan with no declared ranking "
                "rule discloses that no rule was set in advance"
                if plan.mode != "tradeoff"
                else "holds 'no_rule', and the campaign declared a merit priority that ranks the axes"
            ),
        )
    scopes = {caveat.id: caveat.reach for caveat in report_caveats(report_input)}
    if scoped := [item for item in discloses if scopes.get(item) in ("scoped", "method")]:
        yield (
            "flag[0].discloses",
            (
                f"holds {scoped}, which are not report-wide disclosures of this input: a caveat touching one arm or one "
                "finding is said beside the claim it qualifies, usually in Watch, a method-level one in Method, and the "
                f"flag holds only what touches the whole comparison ({implied})"
            ),
        )


#: Where a caveat may land, by scope; the flag is checked by :func:`_flag_faults`.
_PLACES: dict[str, frozenset[str]] = {
    "report": frozenset({"confidence", "flag"}),
    "scoped": frozenset({"confidence", "claim"}),
    "method": frozenset({"confidence", "claim", "method"}),
}


def _landing_places(plan: ReportPlan) -> Iterator[tuple[str, str, str]]:
    """Every place the plan lands a caveat: the id, the field path, and the kind of place."""
    for index, clause in enumerate(plan.decision.confidence.caveats):
        yield clause.id, f"decision.confidence.caveats[{index}].id", "confidence"
    for flag in plan.flag:
        for index, item in enumerate(flag.discloses):
            if item != "no_rule":
                yield item, f"flag[0].discloses[{index}]", "flag"
    for index, slot in enumerate(plan.decision.slots):
        for number, item in enumerate(slot.caveats):
            yield item, f"decision.slots[{index}].caveats[{number}]", "claim"
    for index, section in enumerate(plan.sections):
        for number, item in enumerate(section.caveats):
            yield item, f"sections[{index}].caveats[{number}]", "claim"
    for index, method in enumerate(plan.method):
        for number, item in enumerate(method.caveats):
            yield item, f"method[{index}].caveats[{number}]", "method"


def _landings(plan: ReportPlan) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for item, path, _place in _landing_places(plan):
        found.setdefault(item, []).append(path)
    return found


def _coverage_faults(plan: ReportPlan, report_input: ReportInput) -> Iterator[_Fault]:
    """Every caveat lands exactly once where its scope allows, or nowhere when code proved it immaterial."""
    caveats = {caveat.id: caveat for caveat in report_caveats(report_input)}
    materiality = caveat_materiality(report_input)
    seen: dict[str, list[tuple[str, str]]] = {}
    for item, path, place in _landing_places(plan):
        if item not in caveats:
            yield path, f"names {item!r}, which is no caveat of this input; the caveats are {list(caveats)}"
            continue
        seen.setdefault(item, []).append((path, place))
    for item, caveat in caveats.items():
        where = seen.get(item, [])
        if materiality[item].immaterial:
            for path, _place in where:
                yield path, f"names {item!r}, which code proved immaterial: {_checked(materiality[item])}"
            continue
        if not where:
            yield _home(caveat), f"drop the caveat {item!r}: {_where_it_lands(caveat, plan)}"
            continue
        for path, _place in where[1:]:
            yield path, f"states {item!r} again (also at {where[0][0]}): a caveat is stated exactly once"
        path, place = where[0]
        if place != "flag" and place not in _PLACES[caveat.reach]:
            yield path, f"holds {item!r}, and {_where_it_lands(caveat, plan)}"


def _checked(materiality: Materiality) -> str:
    return f"code renders its checked line ({materiality.checked!r}), so the plan says nothing more of it"


def _home(caveat: Caveat) -> str:
    """The field a dropped caveat belongs at, named in its refusal."""
    if caveat.reach == "report":
        return "decision.confidence.caveats"
    return "method" if caveat.reach == "method" else "decision.slots"


def _where_it_lands(caveat: Caveat, plan: ReportPlan) -> str:
    if caveat.reach == "report":
        if plan.decision.confidence.form == "low":
            return "it is report-wide: list it in `decision.confidence.caveats` with its clause, or disclose it in the flag"
        return "it is report-wide: list it in `decision.confidence.caveats` with what would make it material"
    if caveat.reach == "scoped":
        return (
            "it touches some comparisons, not all: name it in the `caveats` of the decision slot or section whose claim "
            "it qualifies (usually Watch), or list it in `decision.confidence.caveats`; Method takes only method-level "
            f"caveats ({sorted(_METHOD_KINDS)})"
        )
    return (
        "it is method-level: name it in a Method item's `caveats`, in the `caveats` of the slot or section whose claim "
        "it qualifies, or in `decision.confidence.caveats`"
    )


# --- charts --------------------------------------------------------------------------------------------------------


def _charts(plan: ReportPlan) -> Iterator[tuple[str, ChartSpec]]:
    for index, chart in enumerate(plan.summary):
        yield f"summary[{index}]", chart
    for index, section in enumerate(plan.sections):
        yield f"sections[{index}].chart", section.chart


def _chart_faults(plan: ReportPlan, report_input: ReportInput) -> Iterator[_Fault]:
    for path, chart in _charts(plan):
        yield from _query_faults(path, chart, plan, report_input)
    control = control_arm(report_input)
    arms = {arm.id for arm in report_input.arms}
    rows = [row.option for row in plan.options.rows if row.option in arms]
    for index, chart in enumerate(plan.summary):
        from_control = _from_control(chart)
        shown = list(chart.query.arms) or [
            arm.id for arm in report_input.arms if not (from_control and arm.id == control)
        ]
        wanted = [arm for arm in rows if not (from_control and arm == control)]
        if shown != wanted:
            yield (
                f"summary[{index}].query.arms",
                (
                    f"show {shown}, and a summary chart shows the options table's arm rows in its order, {wanted}"
                    + (" (a change-from-control chart leaves out the control)" if from_control else "")
                ),
            )


def _from_control(chart: ChartSpec) -> bool:
    return chart.query.source in _FROM_CONTROL or chart.query.transform == "arm_minus_control"


def _query_faults(path: str, chart: ChartSpec, plan: ReportPlan, report_input: ReportInput) -> Iterator[_Fault]:
    query = chart.query
    source = query.source
    if chart.kind != "table" and source not in _KIND_SOURCES[chart.kind]:
        yield (
            f"{path}.kind",
            (f"is {chart.kind!r}, which cannot draw {source!r}; it draws {sorted(_KIND_SOURCES[chart.kind])}"),
        )
    arms = [arm.id for arm in report_input.arms]
    if stray := [arm for arm in query.arms if arm not in arms]:
        yield f"{path}.query.arms", f"name {stray}, which are no arms; the arms are {arms}"
    aspects = [aspect.id for aspect in report_input.aspects]
    if query.aspect and query.aspect not in aspects:
        yield f"{path}.query.aspect", f"is {query.aspect!r}, which is no aspect; the aspects are {aspects}"
        return
    measures = _measure_vocabulary(source, report_input)
    if stray := [measure for measure in query.measures if measure not in measures]:
        held = f"it reads {sorted(measures)}" if measures else "it names none"
        yield f"{path}.query.measures", f"name {stray}, which the {source!r} source does not hold; {held}"
    strata = _strata_vocabulary(report_input)
    if stray := [stratum for stratum in query.strata if stratum not in strata]:
        yield f"{path}.query.strata", f"name {stray}, which are no strata or tasks of the report input"
    rows = list(_source_rows(chart, report_input))
    if not rows:
        yield f"{path}.query", f"selects nothing from {source!r}: the chart would be empty"
    elif chart.kind == "forest" and not all(interval for _, interval in rows):
        yield f"{path}.kind", "is 'forest', and some row it selects carries no interval: a forest draws intervals"
    for index, reference in enumerate(chart.reference):
        yield from _bare_key_faults(f"{path}.reference[{index}].key", reference.key, report_input)
    if chart.emphasis and chart.emphasis not in _emphasizable(plan, report_input):
        yield f"{path}.emphasis", f"is {chart.emphasis!r}, which is no arm, option, stratum or task to stress"


def _measure_vocabulary(source: str, report_input: ReportInput) -> set[str]:
    if source in _NO_MEASURES:
        return set()
    if source == "contrasts":
        return {contrast.measure for contrast in report_input.contrasts}
    if source == "measures":
        return set(report_input.measures)
    if source == "judged":
        return set(report_input.judged)
    if source == "configs":
        return set(report_input.measures) | set(report_input.judged)
    if source == "scoreboard":
        return set(_SCOREBOARD_FIGURES)
    if source == "floors":
        return {floor.floor for floor in report_input.floors}
    if source == "confusion":
        return set(report_input.confusion)
    holders: Iterable[Mapping[str, Mapping[str, object]]]
    if source == "strata":
        holders = [stratum.per_arm for stratum in report_input.strata]
    elif source == "tasks":
        holders = [task.per_arm for task in report_input.tasks]
    else:
        holders = [cell.per_arm for cell in report_input.cells.values()]
    return {name for per_arm in holders for figures in per_arm.values() for name in figures}


def _strata_vocabulary(report_input: ReportInput) -> set[str]:
    names = {stratum.id for stratum in report_input.strata} | {task.id for task in report_input.tasks}
    names |= {example.stratum for example in report_input.examples if example.stratum}
    names |= set(report_input.cells)
    if report_input.human_pref is not None:
        names |= {row.id for row in report_input.human_pref.rows}
    return names


def _source_rows(chart: ChartSpec, report_input: ReportInput) -> Iterator[tuple[str, bool]]:
    """Each row the query selects, by a label, with whether it carries an interval."""
    query = chart.query
    arms = set(query.arms) or {arm.id for arm in report_input.arms}
    measures = set(query.measures)
    strata = set(query.strata)
    source = query.source
    if source == "readings":
        aspect = _aspect_of(query.aspect, report_input)
        for reading in aspect.readings if aspect is not None else []:
            if reading.arm in arms:
                yield reading.arm, reading.within_interval is not None
    elif source == "contrasts":
        for contrast in report_input.contrasts:
            if (
                contrast.arm in arms
                and (not measures or contrast.measure in measures)
                and (not query.aspect or contrast.aspect == query.aspect)
            ):
                yield f"{contrast.arm}/{contrast.measure}", contrast.interval is not None
    elif source in ("measures", "judged", "configs"):
        yield from _figure_rows(source, query.aspect, measures, arms, report_input)
    elif source == "scoreboard":
        aspect = _aspect_of(query.aspect, report_input) if query.aspect else None
        board = aspect.scoreboard if aspect is not None and aspect.scoreboard is not None else report_input.scoreboard
        yield from ((arm, False) for arm in board if arm in arms)
    elif source == "floors":
        for floor in report_input.floors:
            if floor.arm in arms and (not query.aspect or floor.aspect == query.aspect):
                if not measures or floor.floor in measures:
                    yield f"{floor.floor}/{floor.arm}", floor.interval is not None
    elif source in ("strata", "tasks"):
        slices = report_input.strata if source == "strata" else report_input.tasks
        for item in slices:
            if (not strata or item.id in strata) and any(arm in arms for arm in item.per_arm):
                yield item.id, False
    elif source == "examples":
        for example in report_input.examples:
            if (
                (example.arm is None or example.arm in arms)
                and (not strata or example.stratum in strata)
                and (not query.aspect or example.aspect in (None, query.aspect))
            ):
                yield example.trial or example.case or "example", False
    elif source == "cells":
        yield from ((cell, False) for cell in report_input.cells if not strata or cell in strata)
    elif source == "confusion":
        yield from ((axis, False) for axis in report_input.confusion if not measures or axis in measures)
    elif report_input.human_pref is not None:
        yield from ((row.id, row.half_width is not None) for row in report_input.human_pref.rows)


def _figure_rows(
    source: str, aspect: str, measures: set[str], arms: set[str], report_input: ReportInput
) -> Iterator[tuple[str, bool]]:
    """Each arm's figure on each measure or judged dimension the query keeps; a configs chart keeps the arms that
    ran a config."""
    tables: list[Mapping[str, Measure | Judged]] = []
    if source in ("measures", "configs"):
        tables.append(report_input.measures)
    if source in ("judged", "configs"):
        tables.append(report_input.judged)
    if source == "configs":
        arms = arms & {config.arm for config in report_input.configs}
    for table in tables:
        for key, figures in table.items():
            if measures and key not in measures:
                continue
            per_arm = figures.by_aspect.get(aspect, figures.per_arm) if aspect else figures.per_arm
            for arm, stat in per_arm.items():
                if arm in arms:
                    yield f"{key}/{arm}", stat.ci is not None


def _aspect_of(aspect_id: str, report_input: ReportInput) -> Aspect | None:
    """The aspect a query names, or with none named the one asked aspect (the only one asked, else the only one)."""
    if aspect_id:
        return next((aspect for aspect in report_input.aspects if aspect.id == aspect_id), None)
    asked = [aspect for aspect in report_input.aspects if aspect.asked]
    if len(asked) == 1:
        return asked[0]
    return report_input.aspects[0] if not asked and len(report_input.aspects) == 1 else None


def _emphasizable(plan: ReportPlan, report_input: ReportInput) -> set[str]:
    return (
        {arm.id for arm in report_input.arms}
        | {row.option for row in plan.options.rows}
        | _strata_vocabulary(report_input)
    )


# --- figure keys and typed digits ----------------------------------------------------------------------------------


def _prose(node: BaseModel, path: str = "") -> Iterator[tuple[str, str]]:
    """Every prose field the plan holds (a field typed ``ModelProse``), by its path, walking the plan's model."""
    for name, field in type(node).model_fields.items():
        value = getattr(node, name)
        here = f"{path}.{name}" if path else name
        if isinstance(value, str):
            if field_is_prose(field):
                yield here, value
        elif isinstance(value, BaseModel):
            yield from _prose(value, here)
        elif isinstance(value, Sequence):
            for index, item in enumerate(value):
                if isinstance(item, BaseModel):
                    yield from _prose(item, f"{here}[{index}]")


def _option_labels(plan: ReportPlan) -> Iterator[tuple[str, str]]:
    for index, explore in enumerate(plan.explore):
        for number, control in enumerate(explore.controls):
            for at, label in enumerate(control.options):
                yield f"explore[{index}].controls[{number}].options[{at}]", label


def _key_faults(plan: ReportPlan, report_input: ReportInput) -> Iterator[_Fault]:
    for path, text in _prose(plan):
        try:
            keys = keys_in(text)
        except UnresolvableReference as refused:
            yield path, str(refused)
            continue
        for key in keys:
            try:
                figure = resolve(key.path, report_input)
                if figure.kind != "arm":
                    format_figure(figure, key.filters)
            except UnresolvableReference as refused:
                yield path, str(refused)
    for index, claim in enumerate(plan.durable_claims):
        for number, claimed in enumerate(claim.keys):
            yield from _bare_key_faults(f"durable_claims[{index}].keys[{number}]", claimed, report_input)
    for path, label in _option_labels(plan):
        if "{" in label or "}" in label:
            yield path, "is an option label, printed as written, so it holds no figure key: name the option in words"


def _bare_key_faults(path: str, key: str, report_input: ReportInput) -> Iterator[_Fault]:
    try:
        parsed = parse_key(key)
        figure = resolve(parsed.path, report_input)
        if figure.kind != "arm":
            format_figure(figure, parsed.filters)
    except UnresolvableReference as refused:
        yield path, str(refused)


def _literal_faults(plan: ReportPlan, report_input: ReportInput) -> Iterator[_Fault]:
    for path, text in [*_prose(plan), *_option_labels(plan)]:
        if typed := [text[start:end] for start, end in literal_numbers(text, report_input)]:
            yield (
                path,
                (
                    f"types the number(s) {typed}: every figure in prose is a figure key, and a level, a count or a unit "
                    "the input does not hold is written in words or left out"
                ),
            )


# --- the weakest margin in Watch -----------------------------------------------------------------------------------


def _watch_faults(plan: ReportPlan, report_input: ReportInput) -> Iterator[_Fault]:
    """A decide plan's Watch cites the weakest margin by a ``weakest.*`` key, where the input has one."""
    if plan.mode != "decide" or weakest_margin(report_input) is None:
        return
    for index, slot in enumerate(plan.decision.slots):
        if slot.key != "watch":
            continue
        try:
            cited = any(key.path.startswith(WEAKEST_PREFIX) for key in keys_in(slot.text))
        except UnresolvableReference:
            # NOSILENT: a Watch key that does not parse is refused by the key check (_key_faults) at this same path,
            # naming it; whether that text cites the weakest margin is asked again once the key is mended.
            continue
        if not cited:
            said = render(_WEAKEST_SAID, report_input, {})
            yield (
                f"decision.slots[{index}].text",
                f"names no weakest margin: Watch opens with it, cited by a `weakest.` key, and code found it: {said}",
            )


#: How a Watch refusal says the weakest margin code found.
_WEAKEST_SAID = "{weakest.arm}, {weakest.value} against {weakest.criterion} at {weakest.threshold}"


# --- internal ids above Full results -------------------------------------------------------------------------------

#: A character no reader name carries and an id often does: an id holding one reads as an id.
_ID_MARKS = re.compile(r"[_.:/=()\[\]\"]")
#: Where the page's Full results begin: prose at these paths sits below it, where ids may appear.
_BELOW_FULL_RESULTS = ("method", "durable_claims")
#: The key that quotes the campaign's question, in the campaign's own words and ids.
_QUESTION_KEY = "meta.question"


def _above_full_results(plan: ReportPlan) -> Iterator[tuple[str, str]]:
    for path, text in [*_prose(plan), *_option_labels(plan)]:
        if not path.startswith(_BELOW_FULL_RESULTS):
            yield path, text


def _internal_ids(report_input: ReportInput) -> dict[str, tuple[str, str]]:
    """Every id that reads as an id and differs from its reader name: id, then its kind and reader name.

    An id reads as an id when it holds a character no reader name carries (an underscore, a dot, a colon, a slash, an
    equals sign, a bracket or a quote); a plain word (``billing``, ``overall``) reads as a word, and a percentile's
    name (``p95``) is the reader's own."""
    found: dict[str, tuple[str, str]] = {}
    for kind, names in reader_names(report_input).items():
        for identifier, name in names.items():
            if identifier.lower() in PERCENTILE_NAMES or identifier.casefold() == name.casefold():
                continue
            if _ID_MARKS.search(identifier):
                found.setdefault(identifier, (kind, name))
    return found


def _reader_id_faults(plan: ReportPlan, report_input: ReportInput) -> Iterator[_Fault]:
    """No prose above Full results types an internal id (GENERATOR §8 item 9), matched as D3 matches a name, nor quotes
    the campaign's question, which the campaign wrote in its own ids."""
    ids = _internal_ids(report_input)
    shown = {name for names in reader_names(report_input).values() for name in names.values()}
    shown |= {
        name for arm in report_input.arms for name in (arm.id, arm.name, arm.short_name, arm.served_model) if name
    }
    shown -= set(ids)
    for path, text in _above_full_results(plan):
        try:
            keys = keys_in(text)
        except UnresolvableReference:
            keys = []  # the key check refuses a key that does not parse
        if any(key.path == _QUESTION_KEY for key in keys):
            yield (
                path,
                f"quotes the campaign's question ({{{_QUESTION_KEY}}}) above Full results, in the campaign's own words "
                "and ids: say what it asks in reader names, or leave it to Method",
            )
        for identifier in _typed_ids(text, ids, shown):
            kind, name = ids[identifier]
            yield path, f"types the {kind} id {identifier!r} above Full results: write its reader name, {name!r}"


def _typed_ids(text: str, ids: Mapping[str, tuple[str, str]], shown: Iterable[str]) -> list[str]:
    """The ids ``text`` types outside its keys and outside any name a reader is shown, each once, in order found.

    Names and ids are matched in one pass, longest first, each masked where it matched, so a shown name that is a
    bounded prefix of an id (``Tone`` in ``tone.kept``) never hides the id, and an id inside a longer shown name is
    that name."""
    masked = list(_KEY_SPAN.sub(lambda match: "_" * len(match.group()), text))
    candidates = {name: False for name in shown} | dict.fromkeys(ids, True)
    found: dict[str, int] = {}
    for name in sorted(candidates, key=len, reverse=True):
        for match in _bounded(name).finditer("".join(masked)):
            masked[match.start() : match.end()] = "_" * (match.end() - match.start())
            if candidates[name]:
                found.setdefault(name, match.start())
    return sorted(found, key=found.__getitem__)


#: A key as prose holds it, set aside before ids are looked for.
_KEY_SPAN = re.compile(r"\{[^{}]*\}")


def _bounded(name: str) -> re.Pattern[str]:
    """``name`` where a non-word character or the text's edge bounds it on both sides, in any case: D3's rule."""
    return re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE)
