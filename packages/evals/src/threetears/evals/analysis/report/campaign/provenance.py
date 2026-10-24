"""What a campaign's page says of how it came to be: the question, quoted whole as the campaign asks it; one caveat
line for each disclosure that holds across the report; and the Design & provenance section, where each is stated as
data.

A report-wide disclosure is said once, at the level it holds on: one line on the first screen, worded as what it
means for the readings, and its rows in Design & provenance. It is never recited per aspect or per finding, and it
never lowers a reading or a recommendation: a criterion set after its readings were visible is disclosed, never a
demotion, since its readings read as any others. When the runs ran is never said. Four disclosures are report-wide:

- **criteria set late**: a criterion (an aspect, the control, a margin, a bar, a guardrail's margin or a floor) set
  after the readings it judges were visible, or whose time is not recorded;
- **results with no ordered time**, of which whether a criterion came after them is not known;
- **latency not compared**, read while other runs executed beside it;
- **request settings not confirmed**, for the judge or the simulated user.
"""

from __future__ import annotations

from collections.abc import Sequence

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, CriterionKind, LateCriterion
from threetears.evals.analysis.report.campaign.names import aspect_name
from threetears.evals.analysis.report.decision import floor_statement
from threetears.evals.analysis.report.model import (
    DisclosureBlock,
    Report,
    ReportBlock,
    TableBlock,
    TableCell,
    TableColumn,
    fold_ids,
)
from threetears.evals.analysis.style import humanize, measure_name
from threetears.evals.kernel.campaign import EvalCampaign

__all__ = [
    "caveat_blocks",
    "provenance_blocks",
    "question_blocks",
    "stated_for_the_campaign",
    "timing_caveats",
]


def question_blocks(campaign: EvalCampaign) -> list[ReportBlock]:
    """The campaign's question, quoted whole on one line, the first under the title; nothing where it asks none.

    :param campaign: The campaign.
    :ptype campaign: EvalCampaign
    :return: The question's line, or none.
    :rtype: list[ReportBlock]
    """
    if not campaign.question.strip():
        return []
    return [DisclosureBlock(section="answer", source="basis", text=f'"{" ".join(campaign.question.split())}"')]


def timing_caveats(bundles: Sequence[AnalysisContextBundle]) -> list[str]:
    """The report's caveat lines on when its criteria were set: one if any criterion was set after the readings it
    judges were visible or has no recorded time, one if any result carries no time that can be ordered. Each names
    where it holds, never which criterion or when, which Design & provenance states.

    A page of one aspect, a campaign's page and an analysis's listing line say these same lines.

    :param bundles: Each aspect's evidence, in the campaign's order; one for an aspect's own page.
    :ptype bundles: Sequence[AnalysisContextBundle]
    :return: The lines; none where every criterion came before its readings and every result is timed.
    :rtype: list[str]
    """
    lines: list[str] = []
    late = [bundle for bundle in bundles if bundle.reading_scope.late_criteria]
    if late:
        where = _where([aspect_name(bundle) for bundle in late], len(bundles))
        unrecorded = any(
            criterion.set_at is None for bundle in late for criterion in bundle.reading_scope.late_criteria
        )
        lines.append(
            f"Criteria on {where} were set after the results there existed or have no recorded time, so they are not "
            "shown fixed in advance."
            if unrecorded
            else f"Criteria on {where} were set after the results there existed, so they were not fixed in advance."
        )
    untimed = [bundle for bundle in bundles if bundle.reading_scope.untimed_results]
    if untimed:
        count = sum(bundle.reading_scope.untimed_results for bundle in untimed)
        on = "" if len(untimed) == len(bundles) else f" on {_where([aspect_name(b) for b in untimed], len(bundles))}"
        plural = count != 1
        lines.append(
            f"{count} result{'s' if plural else ''}{on} carr{'y' if plural else 'ies'} no time that can be ordered, so "
            f"whether a criterion was set after {'they were' if plural else 'it was'} visible is not known."
        )
    return lines


def caveat_blocks(bundles: Sequence[AnalysisContextBundle]) -> list[ReportBlock]:
    """A campaign's first screen's caveat lines, after its warnings: when its criteria were set
    (:func:`timing_caveats`), then latency not compared and request settings not confirmed, each once for the whole
    campaign, naming where it holds.

    :param bundles: Each aspect's evidence, in the campaign's order.
    :ptype bundles: Sequence[AnalysisContextBundle]
    :return: The lines; none where nothing holds.
    :rtype: list[ReportBlock]
    """
    blocks: list[ReportBlock] = [
        DisclosureBlock(section="warnings", source="timing", text=line) for line in timing_caveats(bundles)
    ]
    if contended := [aspect_name(bundle) for bundle in bundles if bundle.latency_contended]:
        blocks.append(
            DisclosureBlock(
                section="warnings",
                source="measurement",
                text=f"Latency on {_where(contended, len(bundles))} was read while other runs executed beside it, so "
                "it is not compared; launching with `measure_latency=True` reads it clean.",
            )
        )
    if unconfirmed := [aspect_name(bundle) for bundle in bundles if bundle.request_settings_disclosure]:
        blocks.append(
            DisclosureBlock(
                section="warnings",
                source="measurement",
                text=f"Request settings on {_where(unconfirmed, len(bundles))} are not confirmed as sent, so the judge "
                "or simulated user may have run on other settings than asked.",
            )
        )
    return blocks


def provenance_blocks(campaign: EvalCampaign, bundles: Sequence[AnalysisContextBundle]) -> list[ReportBlock]:
    """The page's Design & provenance section: each criterion set after its readings were visible, with when and on
    which aspects, then the test setup each report-wide disclosure names, with its aspects; nothing where neither
    holds.

    :param campaign: The campaign, whose folds an aspect's cell links.
    :ptype campaign: EvalCampaign
    :param bundles: Each aspect's evidence, in the campaign's order.
    :ptype bundles: Sequence[AnalysisContextBundle]
    :return: The tables, each only where it has a row.
    :rtype: list[ReportBlock]
    """
    labels = [aspect.label for aspect in campaign.aspects]
    folds = dict(zip(labels, fold_ids(labels), strict=True))
    by_name = {aspect_name(bundle): bundle.aspect for bundle in bundles}

    def aspects(names: Sequence[str]) -> TableCell:
        # One aspect's cell links its fold; a list of several links none, since a cell links one place.
        every = len(bundles) > 1 and len(names) == len(bundles)
        one = by_name.get(names[0]) if len(names) == 1 else None
        return TableCell(
            value="every aspect" if every else _listed(names),
            ref=f"aspect-{folds[one]}" if one in folds else None,
        )

    blocks: list[ReportBlock] = []
    groups: dict[tuple[CriterionKind, str, str | None], list[str]] = {}
    for bundle in bundles:
        for late in bundle.reading_scope.late_criteria:
            key = (late.criterion, _criterion_words(bundle, late), late.page_date())
            groups.setdefault(key, []).append(aspect_name(bundle))
    if groups:
        rows = [
            {
                "criterion": TableCell(value=_criterion_noun(kind, what)),
                "set_on": TableCell(value=at or "not recorded"),
                "aspects": aspects(names),
            }
            for (kind, what, at), names in groups.items()
        ]
        blocks.append(
            TableBlock(
                section="provenance",
                name="criteria_set",
                title="Criteria set late",
                columns=[
                    TableColumn(key="criterion", header="Criterion"),
                    TableColumn(key="set_on", header="Set on"),
                    TableColumn(key="aspects", header="Results existed on"),
                ],
                rows=rows,
                order="each criterion as the aspects declare it, in the campaign's order",
                total_rows=len(rows),
            )
        )
    setup: list[tuple[str, list[str]]] = []
    if untimed := [bundle for bundle in bundles if bundle.reading_scope.untimed_results]:
        count = sum(bundle.reading_scope.untimed_results for bundle in untimed)
        setup.append(
            (f"{count} result{'' if count == 1 else 's'} with no ordered time", [aspect_name(b) for b in untimed])
        )
    if contended := [aspect_name(bundle) for bundle in bundles if bundle.latency_contended]:
        setup.append(("latency not compared", contended))
    if unconfirmed := [aspect_name(bundle) for bundle in bundles if bundle.request_settings_disclosure]:
        setup.append(("request settings not confirmed", unconfirmed))
    if setup:
        rows = [{"setup": TableCell(value=what), "aspects": aspects(names)} for what, names in setup]
        blocks.append(
            TableBlock(
                section="provenance",
                name="test_setup",
                title="Test setup",
                columns=[
                    TableColumn(key="setup", header="Disclosure"),
                    TableColumn(key="aspects", header="Aspects", term="aspect"),
                ],
                rows=rows,
                order="results with no ordered time, latency, then request settings",
                total_rows=len(rows),
            )
        )
    return blocks


def stated_for_the_campaign(block: ReportBlock, report: Report) -> bool:
    """Whether ``block`` is one of an aspect's own disclosures that its campaign's page states once for every aspect
    (:func:`caveat_blocks`, :func:`provenance_blocks`), so the aspect's fold leaves it out: the aspect's latency read
    under concurrency, and its request settings not confirmed as sent. Read by what the aspect's own evidence says,
    never by the words of anything else.

    :param block: A block of the aspect's own page.
    :ptype block: ReportBlock
    :param report: The aspect's own page.
    :ptype report: Report
    :return: Whether the campaign's page states it.
    :rtype: bool
    """
    if not isinstance(block, DisclosureBlock) or block.section != "methods" or block.source != "measurement":
        return False
    bundle = report.bundle
    settings = (bundle.request_settings_disclosure or "").strip()
    latency = (bundle.latency_contended or "").strip()
    return bool(settings and block.text == settings) or bool(latency and block.text.startswith(latency))


def _where(names: Sequence[str], total: int) -> str:
    """Where a caveat holds, as its line says it: the aspect's name on a page of one, ``every aspect``, or how many
    of the campaign's (``2 of 5 aspects``), Design & provenance naming them."""
    if total == 1:
        return names[0]
    return "every aspect" if len(names) == total else f"{len(names)} of {total} aspects"


def _criterion_words(bundle: AnalysisContextBundle, late: LateCriterion) -> str:
    """Which criterion of its kind ``late`` is, as a row names it: a bar by its measure, a guardrail by its dimension,
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


def _criterion_noun(kind: CriterionKind, what: str) -> str:
    """A criterion as its row names it: ``aspect``, ``control``, ``margin``, ``bar on cost``, ``guardrail margin on
    tone``, ``floor that no pass rate drops by more than 10 points``."""
    if kind in ("aspect", "control", "margin") or not what:
        return kind
    return {"bar": f"bar on {what}", "guardrail": f"guardrail margin on {what}", "floor": f"floor that {what}"}[kind]


def _listed(labels: Sequence[str]) -> str:
    """Aspects as a cell lists them: ``billing``, ``billing and tone``, ``billing, tone and refunds``."""
    return labels[0] if len(labels) == 1 else f"{', '.join(labels[:-1])} and {labels[-1]}"
