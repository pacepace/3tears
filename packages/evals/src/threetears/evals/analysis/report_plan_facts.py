"""Code's facts about a report: what the planner is told before it writes, and what completes its reply into a plan.

The planner writes choices and prose (:class:`~threetears.evals.kernel.report_plan.PlannedReport`); code computes the
facts (owner ruling, 2026-10-25: code computes facts). This module computes them from the report input alone, once,
as :class:`PlanFacts`, so the planner's user message and the plan code completes from its reply state the same facts:

- **The classification** (GENERATOR §1): the signals and the shape, from
  :func:`~threetears.evals.analysis.report_plan_checks.classify`, and the shape's spine, the chart the first evidence
  section draws, with the query fields the input settles
  (:func:`~threetears.evals.analysis.report_plan_checks.spine`), which the plan copies.
- **The masthead** (§2.1): the campaign's kind, the system under test, the counts and the run date, counts and date
  as figure keys where the input holds the figure.
- **The arms** (§5.2, §6.1): each arm's colour slot (-1 for the control, then series colours in the input's order),
  its letter tag, and whether the arms key shows (§2.8: any arm's full configuration is longer than 24 characters,
  or a setting other than the model varied).
- **The options table's columns** (§2.6: "the criteria plus cost"), each one of
  :data:`~threetears.evals.analysis.report_plan_checks.OPTION_COLUMN_FAMILIES`, the vocabulary the plan checks hold
  a stored plan's columns to: the pass rate where one resolves; the asked reading's change and verdict where an
  aspect is asked and read; the primary judged dimension; a gate column where a gate is declared; each bar's or
  guardrail's measure; then the cost per 1,000 replies where it resolves.
- **The full-results tables** (§2.11: every arm by measure, contrasts, strata), each where the input holds its rows.
- **What the planner must answer and disclose**: every warning code raises on each arm
  (:func:`~threetears.evals.analysis.report_plan_checks.raised_warnings`, which the plan answers by position for the
  arm it recommends), and the report-wide disclosures each mode implies
  (:func:`~threetears.evals.analysis.report_plan_checks.implied_disclosures`).
- **The modes and the caveats** (:mod:`~threetears.evals.analysis.report_decision`): the modes the evidence allows
  (:func:`~threetears.evals.analysis.report_decision.allowed_modes`), and every caveat by id, with its scope (where it
  may land) and, for one code proved immaterial, the checked line code renders in its place.
- **The weakest margin** (:func:`~threetears.evals.analysis.report_margins.weakest_margin`): the figure nearest the
  line it is read against, rendered, with the ``weakest.*`` figure keys that name it, which a decide plan's Watch cites.
- **The reader names** (:func:`~threetears.evals.analysis.report.figure_keys.reader_names`): the one name a reader is
  shown for each aspect, measure, judged dimension, stratum, task and criterion, which the plan writes in place of
  the id above Full results.

:func:`complete_plan` turns a reply into the stored :class:`~threetears.evals.kernel.report_plan.ReportPlan`. Besides
the facts above it fills each unit's code part: the flag's disclosures (:func:`flag_disclosures`) and the measures it
weakens and leaves standing; each answer's warning, from the position the planner gave; each chart reference's key
(this module's reference lines); and each chart's interval level, unit and value axis. Then it runs
:func:`~threetears.evals.analysis.report_plan_checks.validate_plan`, draws every chart no fault names through the
page's own resolvers (:func:`~threetears.evals.analysis.report.plan_page.drawable.undrawn_charts`: a chart the page
cannot draw on this input is refused here, never stored), and refuses every fault it, the checks and the page found,
together, as one :class:`~threetears.evals.analysis.errors.PlanRefusal`, so the planner's one repair round can mend
them all.

**What the flag weakens** is read off the disclosures: an uncalibrated judge weakens every judged dimension a
contrast reads; a criterion set after the results weakens the readings those criteria decide (every contrast that is
not a measure's, which the margin and the floors are read on, and a named late criterion's own measure); a confound or
an apparatus confound weakens the measures whose contrasts name it; a short run of the whole comparison weakens every
measure; an aspect left unsettled past its margin weakens the measures its contrasts read. What survives is
every other measure a contrast reads. Judged dimensions are written ``judged:<dim>``, as the options columns write
them.

**A chart's reference line** sits at a line the input declares, and code fills its figure key:

- a **margin** at the chart's aspect's margin (``aspect.margin`` for the one asked aspect, else
  ``aspects.<id>.margin``);
- a **floor** or a **budget** at ``criteria.<id>.threshold``, the threshold of the one criterion that reads the chart's
  one measure (the measures its query names, else the ones its rows read): a floor's criterion is higher-is-better and
  a budget's lower-is-better, as the kit draws them (pass at or above a floor, at or below a budget); its statistic is
  one the chart draws (range bars draw p50 to p95, every other kind the mean); and it reads where the chart does, over
  the whole comparison or each task, or within the one task the chart keeps, or each cell on a chart of cells.

No such criterion, or several, is a fault at completion, naming why, and so is any other kind of line.

What the chart reads is :mod:`~threetears.evals.analysis.report_rows`' selection (:func:`query_aspect`,
:func:`query_rows`), the one the plan checks and the page read too.

**A chart's interval level** is the one its rows carry: a reading's confidence, a contrast's family level, a measure's
or judged dimension's declared level (a stratum's with "uncorrected"). A diverging chart over judged figures or strata
whose figures carry a standard error and no interval is drawn at the descriptive level, "95% uncorrected"
(:data:`~threetears.evals.analysis.stats.INTERVAL_LEVEL`): a bar is separated from zero where its change exceeds
1.96 times the two arms' standard errors combined (GENERATOR §5.2).

**A chart's axis** is computed only where code holds a rule: a forest of readings spans every interval it draws, zero
and the margin line, one point beyond each end. Every other chart leaves its domain empty, which lets the renderer fit
it. A chart that would mix interval levels (§5.2: one interval system per chart) is refused at its query. Which
measures a chart draws, and whether it can draw them in their units, is the page's own rule and the only unit rule
here (:func:`~threetears.evals.analysis.report.plan_page.charting.unit_refusal`, §5.2: one unit per axis): a chart of
strata or tasks naming no measure where its slices hold several (a heatmap reads every one), or measures or contrasts
in more than one unit on one axis, is refused here by the same function the page refuses it by; a kind that draws each
measure in its own panel (a heatmap's matrix, a dot strip's strip, bars in panels by slice or measure) may read
several units, and its stored unit is then empty.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.stats import INTERVAL_LEVEL
from threetears.evals.analysis.errors import PlanRefusal, SoundnessRefusal, UnresolvableReference
from threetears.evals.analysis.report.plan_page.charting import unit_refusal
from threetears.evals.analysis.report.figure_keys import (
    DIRECTION_WORDS,
    WEAKEST_PREFIX,
    reader_names,
    render,
    resolve,
)
from threetears.evals.analysis.report.plan_page.drawable import undrawn_charts
from threetears.evals.analysis.report.words import arm_tags
from threetears.evals.analysis.report_decision import allowed_modes, caveat_materiality, report_caveats
from threetears.evals.analysis.report_input import Cell, Contrast, Criterion, Reading, ReportInput, Stat, Stratum, Task
from threetears.evals.analysis.report_margins import criterion_place, weakest_margin
from threetears.evals.analysis.report_rows import query_aspect, query_rows
from threetears.evals.analysis.report_plan_checks import (
    Spine,
    classify,
    contrast_level,
    contrast_unit,
    implied_disclosures,
    raised_warnings,
    spine,
    validate_plan,
)
from threetears.evals.kernel.report_plan import (
    REPORT_PLAN_SCHEMA,
    ChartReference,
    ChartSpec,
    Flag,
    Masthead,
    Options,
    PlanArm,
    PlannedChart,
    PlannedReport,
    ReportPlan,
    Section,
    Signals,
    WarningAnswer,
    WarningRef,
)

__all__ = [
    "PLAN_MODES",
    "FactArm",
    "FactCaveat",
    "PlanFacts",
    "complete_plan",
    "flag_disclosures",
    "plan_facts",
]

#: GENERATOR §2.8: the arms key shows when any arm's full configuration is longer than this.
_ARMS_KEY_CHARS = 24

#: The modes a plan may take, in the order the facts list what each implies.
PLAN_MODES: tuple[str, ...] = ("decide", "tradeoff", "no_call")

#: The masthead's kind, by what the campaign is (GENERATOR §2.1).
_KIND_LABELS = {
    "change": "Change · control vs candidates",
    "choice": "Choice · candidates compared",
    "measurement": "Measurement · one arm",
}

#: A fault: the field path it sits at, and why.
_Fault = tuple[str, str]


class FactArm(BaseModel):
    """One arm as the planner is told of it: its id, how the page names and tags it, its role and its colour."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="The arm's id, as the plan names it.")
    tag: str = Field(description="Its letter tag: P for the control, then A, B, C in the input's order.")
    name: str = Field(description="What the page calls it: its declared short name, else its alias, else its id.")
    role: str = Field(description="control or challenger.")
    color_slot: int = Field(description="-1 for the control (ink gray); 0.. for series colours in fixed order.")


class FactCaveat(BaseModel):
    """One caveat as the planner is told of it: its id, where it may land, and whether code proved it immaterial."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="The caveat's id, as the plan names it.")
    reach: str = Field(description="report (the confidence's or the flag's), scoped (beside a claim) or method.")
    checked: str = Field(description="The line code renders for a caveat it proved immaterial; empty for an open one.")


class _FactWeakest(BaseModel):
    """The weakest margin as the planner is told of it: what it is, rendered, and the keys that name it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    said: str = Field(description="The weakest margin in words, its figures rendered.")
    keys: tuple[str, ...] = Field(description="The figure keys that name it; a decide plan's Watch cites one.")


#: The weakest-margin keys, in the order the facts list them, and how the facts render it.
_WEAKEST_KEYS = tuple(f"{WEAKEST_PREFIX}{field}" for field in ("arm", "criterion", "value", "threshold", "margin"))
_WEAKEST_SAID = (
    "{weakest.arm}: {weakest.value} against {weakest.criterion} at {weakest.threshold}, a gap of {weakest.margin}"
)


class PlanFacts(BaseModel):
    """Everything code decides about a report before the planner writes, read off the report input alone."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    signals: Signals
    shape: str
    spine: Spine = Field(description="The shape's spine: the chart the first evidence section draws, as code fixes it.")
    masthead: Masthead
    arms: tuple[FactArm, ...]
    show_arms_key: bool
    columns: tuple[str, ...] = Field(description="The options table's columns, as figure key families.")
    full_results: tuple[str, ...] = Field(description="The full-results tables the page prints.")
    warnings: Mapping[str, tuple[WarningRef, ...]] = Field(
        description="Every warning code raises on each arm, by arm id, in the order an answer's position counts."
    )
    disclosures: Mapping[str, tuple[str, ...]] = Field(
        description=(
            "The report-wide disclosures each mode implies, by mode: no_rule, then each open report-wide caveat. Each "
            "lands once, in the decision's confidence or the flag (:func:`flag_disclosures`)."
        )
    )
    allowed_modes: tuple[str, ...] = Field(description="The modes the evidence allows, in PLAN_MODES' order.")
    caveats: tuple[FactCaveat, ...] = Field(description="Every caveat the input carries, by id.")
    weakest: _FactWeakest | None = Field(
        description="The weakest margin under the call; None where the input reads no figure against a line."
    )
    names: Mapping[str, Mapping[str, str]] = Field(
        description="Each id's one reader name, by kind (aspect, measure, judged, stratum, task, criterion), then id."
    )


def plan_facts(report_input: ReportInput) -> PlanFacts:
    """Code's facts about a report, read off its input (this module's list).

    :param report_input: The report input the plan is written against.
    :ptype report_input: ReportInput
    :return: The facts.
    :rtype: PlanFacts
    """
    signals, shape = classify(report_input)
    materiality = caveat_materiality(report_input)
    control = next((arm.id for arm in report_input.arms if arm.role == "control"), None)
    tags = arm_tags([arm.id for arm in report_input.arms], control)
    slots = iter(range(len(report_input.arms)))
    arms = tuple(
        FactArm(
            id=arm.id,
            tag=tags[arm.id],
            name=arm.short_name or arm.name or arm.id,
            role=arm.role,
            color_slot=-1 if arm.id == control else next(slots),
        )
        for arm in report_input.arms
    )
    return PlanFacts(
        signals=signals,
        shape=shape,
        spine=spine(report_input),
        masthead=_masthead(report_input, control),
        arms=arms,
        show_arms_key=_shows_arms_key(report_input),
        columns=tuple(_columns(report_input)),
        full_results=tuple(_full_results(report_input)),
        warnings={arm.id: tuple(raised_warnings(report_input, arm.id)) for arm in report_input.arms},
        disclosures={mode: tuple(implied_disclosures(report_input, mode)) for mode in PLAN_MODES},
        allowed_modes=tuple(mode for mode in PLAN_MODES if mode in allowed_modes(report_input)),
        caveats=tuple(
            FactCaveat(id=caveat.id, reach=caveat.reach, checked=materiality[caveat.id].checked or "")
            for caveat in report_caveats(report_input)
        ),
        weakest=_weakest(report_input),
        names={str(kind): found for kind, found in reader_names(report_input).items()},
    )


def _weakest(report_input: ReportInput) -> _FactWeakest | None:
    if weakest_margin(report_input) is None:
        return None
    return _FactWeakest(said=render(_WEAKEST_SAID, report_input, {}), keys=_WEAKEST_KEYS)


def flag_disclosures(facts: PlanFacts, mode: str, listed: Iterable[str]) -> tuple[str, ...]:
    """What a plan's flag discloses: its mode's implied disclosures less the caveats its confidence lists. The flag is
    present exactly when this is not empty.

    :param facts: Code's facts about the input.
    :ptype facts: PlanFacts
    :param mode: The plan's mode.
    :ptype mode: str
    :param listed: The ids the decision's confidence lists.
    :ptype listed: Iterable[str]
    :return: ``no_rule`` where it applies, then each open report-wide caveat no clause carries.
    :rtype: tuple[str, ...]
    """
    carried = set(listed)
    return tuple(disclosure for disclosure in facts.disclosures[mode] if disclosure not in carried)


def _resolves(key: str, report_input: ReportInput) -> bool:
    try:
        resolve(key, report_input)
    except UnresolvableReference:
        return False
    return True


def _first_resolving(keys: Iterable[str], report_input: ReportInput) -> str | None:
    return next((key for key in keys if _resolves(key, report_input)), None)


def _campaign_kind(report_input: ReportInput, control: str | None) -> str:
    if len({arm.id for arm in report_input.arms}) <= 1:
        return "measurement"
    return "change" if control is not None else "choice"


def _masthead(report_input: ReportInput, control: str | None) -> Masthead:
    """GENERATOR §2.1: kind · system under test · counts · run date, counts as figure keys where they resolve."""
    meta = report_input.meta
    kind = _campaign_kind(report_input, control)
    counts: list[str] = []
    if kind == "change":
        name = _first_resolving(("control.name", "control.id"), report_input)
        counts.append(f"{{{name}}} vs {{n_candidates}} candidates" if name else "{n_candidates} candidates")
    elif kind == "choice":
        counts.append(f"{len(report_input.arms)} arms")
    if report_input.tasks:
        counts.append(f"{len(report_input.tasks)} tasks")
    elif len(report_input.aspects) > 1:
        counts.append(f"{len(report_input.aspects)} aspects")
    elif aspect := _first_resolving(("aspect.name", "aspect.id"), report_input):
        counts.append(f"{{{aspect}}}")
    if _resolves("cases", report_input) and _resolves("repeats", report_input):
        counts.append("{cases} cases × {repeats} repeats")
    return Masthead(
        kind=_KIND_LABELS[kind],
        system=meta.behavior or meta.campaign,
        counts=" · ".join(counts),
        run_date="{meta.run_date}" if meta.run_date else "",
        synthetic=bool(meta.synthetic),
    )


def _shows_arms_key(report_input: ReportInput) -> bool:
    """GENERATOR §2.8: a full configuration over ``_ARMS_KEY_CHARS`` characters, or a setting beside the model that
    varied."""
    for arm in report_input.arms:
        shown = " · ".join(
            str(value) if setting == "model" else f"{setting} {value}" for setting, value in arm.settings.items()
        )
        if len(shown or arm.served_model or "") > _ARMS_KEY_CHARS:
            return True
    others = {setting for arm in report_input.arms for setting in arm.settings if setting != "model"}
    return any(len({repr(arm.settings.get(setting)) for arm in report_input.arms}) > 1 for setting in others)


def _columns(report_input: ReportInput) -> Iterator[str]:
    """GENERATOR §2.6: the criteria plus cost, as figure key families (this module's list), each one of
    :data:`~threetears.evals.analysis.report_plan_checks.OPTION_COLUMN_FAMILIES`."""
    arms = [arm.id for arm in report_input.arms]
    if any(_resolves(f"pass_rate.{arm}", report_input) for arm in arms):
        yield "pass_rate"
    asked = query_aspect("", report_input)
    if asked is not None and asked.readings:
        yield from ("reading.change", "reading.verdict")
    columns: dict[str, None] = {}
    if report_input.primary_judged is not None:
        columns[f"judged:{report_input.primary_judged}"] = None
    if any(criterion.kind == "gate" for criterion in report_input.criteria):
        columns["gate"] = None
    for criterion in report_input.criteria:
        if criterion.kind in ("bar", "guardrail"):
            family = "judged" if criterion.measure in report_input.judged else "measure"
            columns[f"{family}:{criterion.measure}"] = None
    yield from columns
    if any(_resolves(f"cost_per_1k.{arm}", report_input) for arm in arms):
        yield "cost_per_1k"


def _full_results(report_input: ReportInput) -> Iterator[str]:
    """GENERATOR §2.11: every arm by measure, then each table the input holds rows for."""
    yield "arms_by_measure"
    if report_input.contrasts:
        yield "contrasts"
    if report_input.strata or report_input.tasks:
        yield "strata"
    if any(criterion.kind == "gate" for criterion in report_input.criteria):
        yield "gate_matrix_full"
    if report_input.configs:
        yield "configs"
    if report_input.cells:
        yield "cells"


# --- completing a reply --------------------------------------------------------------------------------------------


def complete_plan(planned: PlannedReport, report_input: ReportInput, facts: PlanFacts | None = None) -> ReportPlan:
    """Complete the planner's reply with code's facts and parts into the stored plan, and check it (this module's
    account).

    :param planned: The planner's validated reply.
    :ptype planned: PlannedReport
    :param report_input: The report input it was written against.
    :ptype report_input: ReportInput
    :param facts: The facts the planner was told; None computes them (:func:`plan_facts`).
    :ptype facts: PlanFacts | None
    :return: The stored plan, which passes :func:`~threetears.evals.analysis.report_plan_checks.validate_plan`.
    :rtype: ReportPlan
    :raises PlanRefusal: A part code could not complete (an answer naming no warning, a reference line the input holds
        no figure for, a chart mixing units or levels, a flag where nothing needs disclosing), a fault the plan checks
        found, or a chart the page cannot draw on this input, all together.
    :raises SoundnessRefusal: The completed plan does not build (repaired once, like any refusal of a finished call).
    """
    facts = facts if facts is not None else plan_facts(report_input)
    faults: list[_Fault] = []
    flag = _flag(planned, report_input, facts, faults)
    try:
        plan = ReportPlan.model_validate(
            {
                "schema": REPORT_PLAN_SCHEMA,
                "signals": facts.signals,
                "shape": facts.shape,
                "masthead": facts.masthead,
                "arms": [PlanArm(id=arm.id, color_slot=arm.color_slot) for arm in facts.arms],
                "show_arms_key": facts.show_arms_key,
                "mode": planned.mode,
                "headline_kind": planned.headline_kind,
                "headline": planned.headline,
                "lede": planned.lede,
                "flag": flag,
                "decision": planned.decision,
                "options": Options(rows=planned.options.rows, columns=list(facts.columns)),
                "summary": [
                    _chart(chart, report_input, f"summary[{index}]", faults)
                    for index, chart in enumerate(planned.summary)
                ],
                "sections": [
                    Section(
                        id=section.id,
                        eyebrow=section.eyebrow,
                        headline=section.headline,
                        takeaway=section.takeaway,
                        chart=_chart(section.chart, report_input, f"sections[{index}].chart", faults),
                        caption=section.caption,
                        caveats=section.caveats,
                    )
                    for index, section in enumerate(planned.sections)
                ],
                "explore": planned.explore,
                "full_results": list(facts.full_results),
                "method": planned.method,
                "durable_claims": planned.durable_claims,
                "answers": list(_answers(planned, report_input, facts, faults)),
            }
        )
    except ValidationError as refused:
        raise SoundnessRefusal(f"the planner's reply could not be completed into a report plan: {refused}") from refused
    try:
        validate_plan(plan, report_input)
    except PlanRefusal as refused:
        faults.extend(refused.faults)
    faults += undrawn_charts(plan, report_input, facts, faulted=[path for path, _ in faults])
    if faults:
        raise PlanRefusal(faults)
    return plan


def _flag(planned: PlannedReport, report_input: ReportInput, facts: PlanFacts, faults: list[_Fault]) -> list[Flag]:
    """The planner's flag with what it discloses (:func:`flag_disclosures`) and the measures it weakens and leaves
    standing; a flag where nothing needs disclosing is refused and left out."""
    if not planned.flag:
        return []
    listed = [clause.id for clause in planned.decision.confidence.caveats]
    discloses = list(flag_disclosures(facts, planned.mode, listed))
    if not discloses:
        faults.append(
            (
                "flag[0]",
                f"is written, and this {planned.mode} plan has no report-wide disclosure: drop the flag, and say any "
                "scoped caveat beside the claim it qualifies",
            )
        )
        return []
    weakens = _weakened(report_input, discloses)
    survives = [measure for measure in _contrasted(report_input) if measure not in weakens]
    (written,) = planned.flag
    return [Flag(title=written.title, text=written.text, discloses=discloses, weakens=weakens, survives=survives)]


def _is_judged(report_input: ReportInput, measure: str, kind: str | None) -> bool:
    return kind == "judged" or (kind is None and measure in report_input.judged)


def _measure_name(report_input: ReportInput, measure: str, kind: str | None) -> str:
    return f"judged:{measure}" if _is_judged(report_input, measure, kind) else measure


def _contrasted(report_input: ReportInput) -> list[str]:
    """Every measure a contrast reads, judged ones as ``judged:<dim>``, in the contrasts' order."""
    names = (_measure_name(report_input, contrast.measure, contrast.kind) for contrast in report_input.contrasts)
    return list(dict.fromkeys(names))


def _weakened(report_input: ReportInput, discloses: Sequence[str]) -> list[str]:
    """The measures whose findings the disclosures weaken (this module's account), in the contrasts' order."""
    weak: set[str] = set()
    contrasts = report_input.contrasts

    def named(keep: Iterable[bool]) -> set[str]:
        return {_measure_name(report_input, c.measure, c.kind) for c, kept in zip(contrasts, keep, strict=True) if kept}

    for disclosure in discloses:
        if disclosure == "judge_uncalibrated":
            weak |= named(_is_judged(report_input, c.measure, c.kind) for c in contrasts)
        elif disclosure.startswith("late:"):
            weak |= named(c.kind != "measure" for c in contrasts)
            name = disclosure.removeprefix("late:").partition(":")[2]
            weak |= {
                _measure_name(report_input, criterion.measure, None)
                for criterion in report_input.criteria
                if name and criterion.id == name
            }
        elif disclosure.startswith(("confound:", "apparatus:")):
            confound = disclosure.partition(":")[2]
            weak |= named(confound in c.confounds for c in contrasts)
        elif disclosure.startswith("short_run:"):
            weak |= set(_contrasted(report_input))
        elif disclosure.startswith("unsettled:"):
            aspect = disclosure.removeprefix("unsettled:")
            weak |= named(c.aspect == aspect for c in contrasts)
    ordered = _contrasted(report_input)
    return [measure for measure in ordered if measure in weak] + sorted(weak - set(ordered))


def _answers(
    planned: PlannedReport, report_input: ReportInput, facts: PlanFacts, faults: list[_Fault]
) -> Iterator[WarningAnswer]:
    """Each answer with the warning its position names in code's list for the recommended arm."""
    row = next((row for row in planned.options.rows if row.role == "recommended"), None)
    raised = facts.warnings.get(row.option, ()) if row is not None else ()
    for index, answer in enumerate(planned.answers):
        if answer.warning < len(raised):
            yield WarningAnswer(warning=raised[answer.warning], text=answer.text)
        elif row is None:
            faults.append((f"answers[{index}].warning", "answers a warning, and the plan recommends no option"))
        else:
            held = f"positions 0 to {len(raised) - 1}" if raised else "none"
            faults.append(
                (
                    f"answers[{index}].warning",
                    f"is {answer.warning}, and code raised {len(raised)} warning(s) on the recommended "
                    f"{row.option!r} ({held})",
                )
            )


# --- a chart's facts -----------------------------------------------------------------------------------------------


def _chart(chart: PlannedChart, report_input: ReportInput, path: str, faults: list[_Fault]) -> ChartSpec:
    """The planner's chart with its reference key, interval level, unit and value axis."""
    references = list(_references(chart, report_input, path, faults))
    level = _level(chart, report_input, path, faults)
    unit = _unit(chart, report_input, path, faults)
    return ChartSpec(
        kind=chart.kind,
        title=chart.title,
        query=chart.query,
        reference=references,
        emphasis=chart.emphasis,
        level=level,
        unit=unit,
        domain=_domain(chart, report_input, references),
    )


def _references(
    chart: PlannedChart, report_input: ReportInput, path: str, faults: list[_Fault]
) -> Iterator[ChartReference]:
    """Each reference line with the figure key it sits at (this module's reference lines)."""
    for index, reference in enumerate(chart.reference):
        where = f"{path}.reference[{index}].kind"
        if reference.kind == "margin":
            key = _margin_key(chart, report_input, where, faults)
        elif reference.kind in _LINE_DIRECTIONS:
            key = _criterion_key(chart, reference.kind, report_input, where, faults)
        else:
            faults.append(
                (
                    where,
                    f"is {reference.kind!r}, and the report input holds no figure a {reference.kind} line sits at; "
                    "a margin, floor or budget line is what it can draw",
                )
            )
            continue
        if key is not None:
            yield ChartReference(kind=reference.kind, key=key)


def _margin_key(chart: PlannedChart, report_input: ReportInput, where: str, faults: list[_Fault]) -> str | None:
    """A margin line's key: its chart's aspect's margin."""
    aspect = query_aspect(chart.query.aspect, report_input)
    asked = query_aspect("", report_input)
    candidates = ["aspect.margin"] if aspect is not None and aspect is asked else []
    if aspect is not None:
        candidates.append(f"aspects.{aspect.id}.margin")
    key = _first_resolving(candidates, report_input)
    if key is None:
        named = f"aspect {aspect.id!r}" if aspect is not None else "the chart, which names no single aspect,"
        faults.append((where, f"is 'margin', and {named} declares no margin to draw"))
    return key


#: The direction of the criterion a floor or a budget line sits at: the kit passes a floor at or above, a budget at
#: or below.
_LINE_DIRECTIONS = {"floor": "higher_is_better", "budget": "lower_is_better"}
#: The statistics a kind draws a measure at, where they are not the mean: range bars run from p50 to p95.
_KIND_STATISTICS: dict[str, frozenset[str]] = {"range_bars": frozenset({"p50", "p95"})}
_MEAN = frozenset({"mean"})


def _chart_measures(chart: PlannedChart, report_input: ReportInput) -> list[str]:
    """The measures a chart reads: those its query names, else those its rows read (a figure's, a contrast's, a
    floor's or an axis's key; a slice's or cell's figures for the arms it keeps), each once, in order."""
    if chart.query.measures:
        return list(dict.fromkeys(chart.query.measures))
    arms = set(chart.query.arms) or {arm.id for arm in report_input.arms}
    found: list[str] = []
    for row in query_rows(chart.query, report_input):
        if row.key:
            found.append(row.key)
        elif isinstance(row.item, Stratum | Task | Cell):
            found += [name for arm, figures in row.item.per_arm.items() if arm in arms for name in figures]
    return list(dict.fromkeys(found))


def _reads_where_the_chart_does(criterion: Criterion, chart: PlannedChart, report_input: ReportInput) -> bool:
    """Whether a criterion reads where the chart draws: over the whole comparison or each task, within the one task
    the chart keeps, or each cell on a chart of cells."""
    place = criterion_place(criterion)
    if place == "overall" or (place == "task" and criterion.task is None):
        return True
    if place == "cell":
        return chart.query.source == "cells"
    source = chart.query.source
    kept = [row.label for row in query_rows(chart.query, report_input)] if source == "tasks" else chart.query.strata
    return list(kept) == [criterion.task]


def _criterion_key(
    chart: PlannedChart, kind: str, report_input: ReportInput, where: str, faults: list[_Fault]
) -> str | None:
    """A floor or budget line's key: the threshold of the one criterion that reads the chart's one measure (this
    module's reference lines)."""
    measures = _chart_measures(chart, report_input)
    if len(measures) != 1:
        held = f"the measures {measures}" if measures else "no measure"
        faults.append(
            (
                where,
                f"is {kind!r}, and the chart reads {held}: a {kind} line sits at the criterion that reads the "
                "chart's one measure, so name that one measure in the query, or draw no line",
            )
        )
        return None
    (measure,) = measures
    drawn = _KIND_STATISTICS.get(chart.kind, _MEAN)
    direction = _LINE_DIRECTIONS[kind]
    found = [
        criterion
        for criterion in report_input.criteria
        if criterion.measure == measure
        and criterion.direction == direction
        and criterion.statistic in drawn
        and _reads_where_the_chart_does(criterion, chart, report_input)
    ]
    if len(found) == 1:
        return f"criteria.{found[0].id}.threshold"
    said = DIRECTION_WORDS[direction]
    if not found:
        faults.append(
            (
                where,
                f"is {kind!r}, and no criterion reads {measure!r} as a {kind} does ({said}, at a statistic a "
                f"{chart.kind!r} draws, {sorted(drawn)}, where the chart reads): draw no line, or the line a "
                "declared criterion gives",
            )
        )
        return None
    faults.append(
        (
            where,
            f"is {kind!r}, and the criteria {[criterion.id for criterion in found]} all read {measure!r} where the "
            "chart does: a line sits at one criterion's threshold, so draw no line, or narrow the chart to where only "
            "one reads (the one task a task's criterion reads)",
        )
    )
    return None


def _percent(level: float) -> str:
    """A confidence level as the chart prints it: a share (0.95) or a percent (95) as "95%"."""
    return format_number(level * 100 if level <= 1 else level) + "%"


def _one(path: str, what: str, values: Iterable[str], faults: list[_Fault]) -> str:
    """The one value a chart's rows agree on, empty for none; several are refused (§5.2: one per chart)."""
    found = list(dict.fromkeys(value for value in values if value))
    if len(found) > 1:
        faults.append((f"{path}.query", f"selects rows on {found}, and a chart carries one {what} (GENERATOR §5.2)"))
    return found[0] if found else ""


def _selected_contrasts(chart: PlannedChart, report_input: ReportInput) -> list[tuple[str, str]]:
    """The contrasts a chart of contrasts reads: each one's unit and interval level (:func:`contrast_unit`,
    :func:`contrast_level`)."""
    contrasts = [row.item for row in query_rows(chart.query, report_input) if isinstance(row.item, Contrast)]
    return [(contrast_unit(report_input, contrast), contrast_level(report_input, contrast)) for contrast in contrasts]


def _selected_readings(chart: PlannedChart, report_input: ReportInput) -> list[Reading]:
    """The readings a chart of readings reads."""
    return [row.item for row in query_rows(chart.query, report_input) if isinstance(row.item, Reading)]


def _figure_levels(chart: PlannedChart, report_input: ReportInput) -> list[str]:
    measures = set(chart.query.measures)
    levels = [m.level for key, m in report_input.measures.items() if not measures or key in measures]
    levels += [j.level for key, j in report_input.judged.items() if not measures or key in measures]
    return [level for level in levels if level]


def _level(chart: PlannedChart, report_input: ReportInput, path: str, faults: list[_Fault]) -> str:
    """The interval level printed on the chart, from the rows it reads; empty where they carry none."""
    source = chart.query.source
    if source == "readings":
        readings = _selected_readings(chart, report_input)
        levels = [_percent(r.gain_confidence) for r in readings if r.gain_confidence is not None]
        return _one(path, "interval level", levels, faults)
    if source == "contrasts":
        return _one(path, "interval level", (level for _, level in _selected_contrasts(chart, report_input)), faults)
    if source in ("measures", "judged"):
        table = report_input.measures if source == "measures" else report_input.judged
        measures = set(chart.query.measures)
        levels = [figures.level or "" for key, figures in table.items() if not measures or key in measures]
        level = _one(path, "interval level", levels, faults)
        return level or _descriptive_level(chart, report_input)
    if source == "strata":
        level = _one(path, "interval level", _figure_levels(chart, report_input), faults)
        return f"{level} {_UNCORRECTED}" if level else _descriptive_level(chart, report_input)
    return ""


def _descriptive_level(chart: PlannedChart, report_input: ReportInput) -> str:
    """The descriptive level a diverging chart over judged figures or strata is drawn at where its figures carry a
    standard error and no interval: a bar is separated from zero when its change exceeds the interval level's normal
    multiple of the two arms' standard errors combined (GENERATOR §5.2: descriptive, uncorrected); empty otherwise."""
    if chart.kind != "diverging_bars" or chart.query.source not in _DESCRIPTIVE_SOURCES:
        return ""
    stats = _selected_stats(chart, report_input)
    if stats and all(stat.sem is not None and stat.ci is None and stat.half_width is None for stat in stats):
        return f"{_percent(INTERVAL_LEVEL)} {_UNCORRECTED}"
    return ""


#: How a fault opens where the page cannot draw a chart in the units its query selects.
_UNDRAWABLE = "cannot be drawn as it asks:"

#: The sources a diverging chart may draw at the descriptive level, and the word that labels it.
_DESCRIPTIVE_SOURCES = frozenset({"judged", "strata"})
_UNCORRECTED = "uncorrected"


def _selected_stats(chart: PlannedChart, report_input: ReportInput) -> list[Stat]:
    """Every figure a chart of judged figures or strata reads: a judged row's, or each kept arm's on each measure the
    query keeps in a stratum."""
    measures = set(chart.query.measures)
    arms = set(chart.query.arms) or {arm.id for arm in report_input.arms}
    found: list[Stat] = []
    for row in query_rows(chart.query, report_input):
        if isinstance(row.item, Stat):
            found.append(row.item)
        elif isinstance(row.item, Stratum):
            found += [
                stat
                for arm, figures in row.item.per_arm.items()
                if arm in arms
                for key, stat in figures.items()
                if not measures or key in measures
            ]
    return found


def _judged_unit(report_input: ReportInput, measures: Sequence[str]) -> list[str]:
    scales = [judged.scale for key, judged in report_input.judged.items() if not measures or key in measures]
    return ["percent" if scale == "pass_fail" else "score" for scale in scales]


def _unit(chart: PlannedChart, report_input: ReportInput, path: str, faults: list[_Fault]) -> str:
    """The unit the chart's rows are in: a change from the control in points, else the rows' own unit, empty where
    they are in several (each drawn in its own panel) or none. Whether the page can draw the chart in its units is the
    page's one rule, refused here by the same function and by no other (this module's account)."""
    query = chart.query
    source = query.source
    refusal = unit_refusal(chart.kind, query, report_input)
    if refusal:
        faults.append(
            (
                f"{path}.query",
                f"{_UNDRAWABLE} {refusal} (GENERATOR §5.2: one unit per axis, a panel each where the kind panels by measure)",
            )
        )
    if source in ("readings", "floors"):
        return "points"
    if source == "contrasts":
        return _only(unit for unit, _ in _selected_contrasts(chart, report_input))
    if source == "measures":
        return _only(
            m.unit or "" for key, m in report_input.measures.items() if not query.measures or key in query.measures
        )
    if source in ("judged", "strata"):
        if query.transform == "arm_minus_control":
            return "points"
        return _only(_judged_unit(report_input, query.measures))
    return ""


def _only(values: Iterable[str]) -> str:
    """The one value the rows agree on; empty for none, or for several."""
    found = list(dict.fromkeys(value for value in values if value))
    return found[0] if len(found) == 1 else ""


def _domain(chart: PlannedChart, report_input: ReportInput, references: Sequence[ChartReference]) -> list[float]:
    """A forest of readings spans its intervals, zero and its margin line, a point beyond each end; any other chart
    is left to the renderer (empty)."""
    if chart.query.source != "readings":
        return []
    aspect = query_aspect(chart.query.aspect, report_input)
    if aspect is None:
        return []
    ends = [end for r in _selected_readings(chart, report_input) if r.within_interval for end in r.within_interval]
    if not ends:
        return []
    ends.append(0.0)
    if any(reference.kind == "margin" for reference in references) and aspect.margin is not None:
        ends.append(-float(aspect.margin))
    return [float(math.floor(min(ends)) - 1), float(math.ceil(max(ends)) + 1)]
