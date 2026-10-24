"""What a campaign's page says of how its criteria came to be: the question, quoted whole as the campaign asks it,
and one line for each criterion (an aspect, the control, a margin, a bar or a floor) set after the readings it judges
were visible, saying when, or that its time is not recorded. Disclosed, never a demotion, since its readings read as
any others. When the runs ran is never said."""

from __future__ import annotations

from collections.abc import Sequence

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, CriterionKind, LateCriterion
from threetears.evals.analysis.report.decision import floor_statement
from threetears.evals.analysis.report.model import DisclosureBlock, ReportBlock
from threetears.evals.analysis.style import humanize, measure_name
from threetears.evals.kernel.campaign import EvalCampaign

__all__ = ["late_criteria_lines", "question_blocks", "timing_blocks"]


def question_blocks(campaign: EvalCampaign) -> list[ReportBlock]:
    """The campaign's question, quoted whole on one line; nothing where it asks none.

    :param campaign: The campaign.
    :ptype campaign: EvalCampaign
    :return: The question's line, or none.
    :rtype: list[ReportBlock]
    """
    if not campaign.question.strip():
        return []
    return [DisclosureBlock(section="header", source="basis", text=f'"{" ".join(campaign.question.split())}"')]


def timing_blocks(bundles: Sequence[AnalysisContextBundle]) -> list[ReportBlock]:
    """The first screen's line for each criterion set after its readings were visible (:func:`late_criteria_lines`).

    :param bundles: Each aspect's evidence, in the campaign's order.
    :ptype bundles: Sequence[AnalysisContextBundle]
    :return: The lines; none where every criterion came before its readings.
    :rtype: list[ReportBlock]
    """
    return [DisclosureBlock(section="header", source="timing", text=line) for line in late_criteria_lines(bundles)]


def late_criteria_lines(bundles: Sequence[AnalysisContextBundle]) -> list[str]:
    """The lines a page and a listing say for each criterion set after the readings it judges were visible, or whose
    time is not recorded (each aspect's ``reading_scope.late_criteria``): disclosed, never a demotion, so a reader
    can judge what the readings rest on. When the runs ran, or when the campaign was declared, is never said.

    One line per kind of criterion, criterion and date, naming the aspects it holds on (``every aspect`` where it holds
    on all of several), so each late criterion is named once and a campaign of seven adopted aspects is not told the
    same thing seven times.

    :param bundles: Each aspect's evidence, in the campaign's order; one for an aspect's own page.
    :ptype bundles: Sequence[AnalysisContextBundle]
    :return: The lines; none where every criterion came before its readings.
    :rtype: list[str]
    """
    groups: dict[tuple[CriterionKind, str, str | None], list[str]] = {}
    for bundle in bundles:
        for late in bundle.reading_scope.late_criteria:
            key = (late.criterion, _criterion_words(bundle, late), late.page_date())
            groups.setdefault(key, []).append(bundle.aspect)
    every = len(bundles) > 1
    lines = [
        _late_line(kind, what, at, labels, every=every and len(labels) == len(bundles))
        for (kind, what, at), labels in groups.items()
    ]
    untimed = [bundle for bundle in bundles if bundle.reading_scope.untimed_results]
    if untimed:
        count = sum(bundle.reading_scope.untimed_results for bundle in untimed)
        on = "" if len(untimed) == len(bundles) else f" on {_listed([bundle.aspect for bundle in untimed])}"
        plural = count != 1
        lines.append(
            f"{count} result{'s' if plural else ''}{on} carr{'y' if plural else 'ies'} no time that can be ordered, so "
            f"whether a criterion was set after {'they were' if plural else 'it was'} visible is not known."
        )
    return lines


def _criterion_words(bundle: AnalysisContextBundle, late: LateCriterion) -> str:
    """Which criterion of its kind ``late`` is, as a line names it: a bar by its measure, a guardrail by its dimension,
    a floor by what it says."""
    if late.criterion == "bar" and late.name is not None:
        described = bundle.measure_catalog.get(late.name)
        return measure_name(late.name, described.reader_name if described is not None else None)
    if late.criterion == "guardrail" and late.name is not None:
        return humanize("dimension", late.name)
    if late.criterion == "floor" and bundle.declared is not None:
        floor = next((floor for floor in bundle.declared.floors if floor.key == late.name), None)
        if floor is not None:
            return floor_statement(floor.clause, candidate="the arm")
    return late.name or ""


def _late_line(kind: CriterionKind, what: str, at: str | None, labels: Sequence[str], *, every: bool) -> str:
    """One line naming a criterion set after its readings were visible and when, or saying its time is not
    recorded."""
    on = "every aspect" if every else _listed(labels)
    several = len(labels) > 1
    were, its = ("were", "their") if several else ("was", "its")
    if kind == "control":
        set_it = "The control was named"
        after = "after every aspect's readings were visible" if every else f"after the readings on {on} were visible"
    elif kind == "aspect":
        subject = "Every aspect was" if every else f"The aspect{'s' if several else ''} {on} {were}"
        set_it = f"{subject} declared"
        after = f"after {'its' if every else its} readings were visible"
    else:
        noun = {
            "margin": "margin",
            "bar": f"{what} bar",
            "guardrail": f"{what} guardrail's margin",
            "floor": f"floor that {what}",
        }[kind]
        if several and kind in ("margin", "bar"):
            noun += "s"
        set_it = f"The {noun} on {on} {were if kind in ('margin', 'bar') else 'was'} set"
        after = f"after {its} readings were visible"
    if at is None:
        return f"When {set_it[0].lower()}{set_it[1:]} is not recorded."
    return f"{set_it} on {at}, {after}."


def _listed(labels: Sequence[str]) -> str:
    """Aspects as a sentence lists them: ``billing``, ``billing and tone``, ``billing, tone and refunds``."""
    return labels[0] if len(labels) == 1 else f"{', '.join(labels[:-1])} and {labels[-1]}"
