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
  guardrail's measure; then the cost per 1,000 (of what it counts) where it resolves.
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
weakens and leaves standing; each answer's warning, from the position the planner gave; and each chart's facts (below).
Then it runs
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

**A chart's facts** (its reference line's key, interval level, unit and axis) are one chart's rules, apart from any
plan: :func:`~threetears.evals.analysis.report.block_spec.complete_chart` fills them, the same completion an analyst's
``block_render`` runs, and refuses what it cannot fill.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from threetears.evals.analysis.errors import PlanRefusal, SoundnessRefusal, UnresolvableReference
from threetears.evals.analysis.report.block_spec import complete_chart
from threetears.evals.analysis.report.figure_keys import (
    WEAKEST_PREFIX,
    printed_arm_name,
    reader_names,
    render,
    resolve,
)
from threetears.evals.analysis.report.plan_page.drawable import undrawn_charts
from threetears.evals.analysis.report.words import arm_tags
from threetears.evals.analysis.report_decision import allowed_modes, caveat_materiality, report_caveats
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_margins import criterion_aspect, lead_aspect, weakest_margin
from threetears.evals.analysis.report_rows import query_aspect
from threetears.evals.analysis.report_plan_checks import (
    Spine,
    classify,
    implied_disclosures,
    raised_warnings,
    spine,
    validate_plan,
)
from threetears.evals.kernel.report_plan import (
    REPORT_PLAN_SCHEMA,
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
    "fact_arms",
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
    arms = fact_arms(report_input)
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


def fact_arms(report_input: ReportInput) -> tuple[FactArm, ...]:
    """Each arm as the planner is told of it, and as a page draws it: its tag, name, role and colour slot (-1 for the
    control, then series colours in the input's order).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The arms, in the input's order.
    :rtype: tuple[FactArm, ...]
    """
    control = next((arm.id for arm in report_input.arms if arm.role == "control"), None)
    tags = arm_tags([arm.id for arm in report_input.arms], control)
    slots = iter(range(len(report_input.arms)))
    return tuple(
        FactArm(
            id=arm.id,
            tag=tags[arm.id],
            name=printed_arm_name(arm),
            role=arm.role,
            color_slot=-1 if arm.id == control else next(slots),
        )
        for arm in report_input.arms
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
    lead = lead_aspect(report_input)
    for criterion in report_input.criteria:
        # A column prints the lead aspect's figures, so a criterion on another aspect has none of its own to print.
        if criterion.kind in ("bar", "guardrail") and criterion_aspect(report_input, criterion) == lead:
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
    """The planner's chart with its reference key, interval level, unit and value axis
    (:func:`~threetears.evals.analysis.report.block_spec.complete_chart`), its faults added to the plan's."""
    spec, found = complete_chart(chart, report_input, path)
    faults += found
    return spec
