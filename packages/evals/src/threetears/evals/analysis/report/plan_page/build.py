"""The one planned-page builder: a stored report plan and the report input it was written against, to the page.

:func:`plan_page` assembles the page in GENERATOR §2 order, which is fixed here; what each part says is the plan's.
It needs no campaign and no bundle, so any plan renders through it from its input alone. It never re-checks the plan
(``validate_plan`` is generation's): a stored plan is drawn as it was accepted.

**Part by part** (the P4 design, §2):

- *Masthead*: the plan's, its keys written out; *headline*, *lede*: the plan's prose, keys written out with each
  arm's name and tag (:func:`~threetears.evals.analysis.report.figure_keys.render`).
- *Flag*: the plan's flag where it has one, with what code filled it to disclose.
- *Confidence*: the plan's form in GENERATOR §2.5's words, the planner's clause for each open caveat, then code's
  checked line for each caveat it proved immaterial (``plan_facts(...).caveats``).
- *Decision*: the four slots in the plan's (the mode's) order, titled by mode and kind, the recommendation (decide)
  or the lean (tradeoff) set apart; a no-call sets none apart.
- *Answers*: each warning code raised on the recommended arm, in code's words, and the planner's answer.
- *Options* and the *arms key*: :mod:`~threetears.evals.analysis.report.plan_page.tables`; the arms key only where
  the plan shows it.
- *Summary* and *sections*: each chart resolved from its query
  (:func:`~threetears.evals.analysis.report.plan_page.resolve.resolve_chart`).
- *Explore*: a static flip table where the plan's control moves the margin
  (:mod:`~threetears.evals.analysis.report.plan_page.explore`); none otherwise.
- *Full results*: every table the plan lists (:mod:`~threetears.evals.analysis.report.plan_page.full_results`).
- *Method*: the plan's terms. *Made*: the caller's one line.
"""

from __future__ import annotations

from collections.abc import Callable

from threetears.evals.analysis.planned_analysis import plan_recommended_arm
from threetears.evals.analysis.report.figure_keys import render
from threetears.evals.analysis.report.plan_page.charting import measure_words
from threetears.evals.analysis.report.plan_page.explore import explore_part
from threetears.evals.analysis.report.plan_page.full_results import full_results
from threetears.evals.analysis.report.plan_page.look import PageLook, page_look
from threetears.evals.analysis.report.plan_page.model import (
    PageAnswer,
    PageClause,
    PageConfidence,
    PageFlag,
    PageMasthead,
    PageSection,
    PageSlot,
    PageTerm,
    PlannedPage,
)
from threetears.evals.analysis.report.plan_page.resolve import resolve_chart
from threetears.evals.analysis.report.plan_page.tables import arms_key, options_table
from threetears.evals.analysis.report.plan_page.words import ANSWER_TEMPLATES, CONFIDENCE_WORDS, slot_title
from threetears.evals.analysis.report_decision import report_kind, surviving_options
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_plan_facts import FactCaveat, PlanFacts, plan_facts
from threetears.evals.analysis.style import humanize
from threetears.evals.kernel.report_plan import (
    CaveatClause,
    Decision,
    DecisionConfidence,
    DecisionSlot,
    MethodItem,
    ReportPlan,
    WarningRef,
)

__all__ = ["plan_page"]

#: The slot each mode sets apart: the recommendation, or the lean. A no-call sets none apart.
_SET_APART = {"decide": "recommendation", "tradeoff": "our_lean"}

#: Plan prose, its keys written out.
_Written = Callable[[str], str]


def _look(plan: ReportPlan, report_input: ReportInput, facts: PlanFacts) -> PageLook:
    return page_look(
        facts.arms,
        {arm.id: arm.color_slot for arm in plan.arms},
        order=[row.option for row in plan.options.rows],
        survivors=surviving_options(report_input),
        recommended=plan_recommended_arm(plan),
        names=facts.names,
    )


def _masthead(plan: ReportPlan, written: _Written) -> PageMasthead:
    masthead = plan.masthead
    return PageMasthead(
        kind=masthead.kind,
        system=masthead.system,
        counts=written(masthead.counts),
        run_date=written(masthead.run_date),
        synthetic=masthead.synthetic,
    )


def _checked(caveats: tuple[FactCaveat, ...]) -> list[str]:
    """Code's line for each caveat it proved immaterial, in the input's order."""
    return [caveat.checked for caveat in caveats if caveat.checked]


def _clause(clause: CaveatClause, written: _Written) -> PageClause:
    return PageClause(caveat=clause.id, text=written(clause.material_if))


def _confidence(confidence: DecisionConfidence, facts: PlanFacts, written: _Written) -> PageConfidence:
    return PageConfidence(
        form=confidence.form,
        words=CONFIDENCE_WORDS[confidence.form],
        clauses=[_clause(clause, written) for clause in confidence.caveats],
        checked=_checked(facts.caveats),
    )


def _slot(slot: DecisionSlot, mode: str, *, change: bool, written: _Written) -> PageSlot:
    return PageSlot(
        key=slot.key,
        title=slot_title(slot.key, change=change),
        text=written(slot.text),
        highlighted=_SET_APART.get(mode) == slot.key,
    )


def _decision(decision: Decision, mode: str, report_input: ReportInput, written: _Written) -> list[PageSlot]:
    change = report_kind(report_input) == "change"
    return [_slot(slot, mode, change=change, written=written) for slot in decision.slots]


def _warning(warning: WarningRef, report_input: ReportInput, look: PageLook) -> str:
    """A warning code raised, in its kind's template (:data:`~.words.ANSWER_TEMPLATES`)."""
    aspect = look.reader("aspect", warning.aspect)
    if warning.kind == "behind_on":
        name = look.arm(warning.name).name if warning.name in look.arms else warning.name
    elif warning.kind == "bar_missed":
        name = measure_words(report_input, warning.name)
    else:
        name = humanize("dimension", warning.name)
    return ANSWER_TEMPLATES[warning.kind].format(name=name, aspect=aspect)


def _term(item: MethodItem, written: _Written) -> PageTerm:
    return PageTerm(term=written(item.term), text=written(item.text))


def plan_page(plan: ReportPlan, report_input: ReportInput, *, made: str = "") -> PlannedPage:
    """Build the page a stored plan draws, in GENERATOR §2 order (this module's account).

    :param plan: The stored plan.
    :ptype plan: ReportPlan
    :param report_input: The report input the plan was written against, which its keys and queries read.
    :ptype report_input: ReportInput
    :param made: How the page was made, in one line, for its foot; empty for none.
    :ptype made: str
    :return: The page.
    :rtype: PlannedPage
    :raises UnresolvableReference: A figure key no longer resolves against the input, or a chart cannot be drawn
        from it (its kind has no resolver, or its query selects nothing it can draw).
    """
    facts = plan_facts(report_input)
    look = _look(plan, report_input, facts)
    arm_names = {arm.id: arm.name for arm in facts.arms}

    def written(text: str) -> str:
        return render(text, report_input, arm_names)

    flag = (
        PageFlag(title=written(plan.flag[0].title), text=written(plan.flag[0].text), discloses=plan.flag[0].discloses)
        if plan.flag
        else None
    )
    summary = [
        resolve_chart(spec, report_input, look, title=written(spec.title), summary=True) for spec in plan.summary
    ]
    sections = []
    for section in plan.sections:
        headline = written(section.headline)
        sections.append(
            PageSection(
                id=section.id,
                eyebrow=written(section.eyebrow),
                headline=headline,
                takeaway=written(section.takeaway),
                chart=resolve_chart(
                    section.chart, report_input, look, title=written(section.chart.title), fallback_title=headline
                ),
                caption=written(section.caption),
            )
        )
    return PlannedPage(
        title=report_input.meta.campaign,
        masthead=_masthead(plan, written),
        headline=written(plan.headline),
        lede=written(plan.lede),
        flag=flag,
        confidence=_confidence(plan.decision.confidence, facts, written),
        decision=_decision(plan.decision, plan.mode, report_input, written),
        answers=[
            PageAnswer(warning=_warning(answer.warning, report_input, look), text=written(answer.text))
            for answer in plan.answers
        ],
        options=options_table(plan, report_input, look),
        summary=summary,
        arms_key=arms_key(report_input, look) if plan.show_arms_key else None,
        sections=sections,
        explore=explore_part(plan, report_input, look, written),
        full_results=full_results(plan, report_input, look),
        method=[_term(item, written) for item in plan.method],
        made=made,
    )
