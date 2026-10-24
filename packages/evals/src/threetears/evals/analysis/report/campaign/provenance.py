"""What a page says of how it came to be: the question, quoted whole as the campaign asks it; one caveat line for each
disclosure that holds across the report; and where each is stated as data, once.

A report-wide disclosure is said once, at the level it holds on: one line on the first screen, worded as what it
means for the readings, and its detail in one place. On a campaign's page the detail is a row of Design &
provenance, and no aspect's fold repeats it; on an aspect's own page it is the aspect's methods. It is never recited
per aspect or per finding, and it never lowers a reading or a recommendation: a criterion set after its readings
were visible is disclosed, never a demotion, since its readings read as any others. When the runs ran is never said.
Four disclosures are report-wide:

- **criteria set late**: a criterion (an aspect, the control, a margin, a bar, a guardrail's margin or a floor) set
  after the readings it judges were visible, or whose time is not recorded;
- **results with no ordered time**, of which whether a criterion came after them is not known;
- **latency read under concurrency**, which every comparison leaves out;
- **request settings not confirmed**, for the judge or the simulated user.

Each is built here and only here: its line (:func:`caveat_blocks`), its detail (:func:`provenance_blocks`,
:func:`aspect_provenance_blocks`, :func:`setup_disclosure`), and what an aspect's fold on a campaign's page leaves out
because the campaign states it (:func:`for_the_campaign`), which reads the marks those carry, never their words.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from threetears.evals.analysis.arms import short_digest
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, CriterionKind, LateCriterion
from threetears.evals.analysis.report.campaign.common import named_arms
from threetears.evals.analysis.report.campaign.names import aspect_name
from threetears.evals.analysis.report.campaign.warnings import listed, where
from threetears.evals.analysis.report.decision import floor_statement
from threetears.evals.analysis.report.model import (
    DisclosureBlock,
    Report,
    ReportBlock,
    ReportSection,
    TableBlock,
    TableCell,
    TableColumn,
    fold_ids,
)
from threetears.evals.analysis.style import ArmName, humanize, measure_name
from threetears.evals.kernel.campaign import EvalCampaign

__all__ = [
    "aspect_provenance_blocks",
    "caveat_blocks",
    "for_the_campaign",
    "provenance_blocks",
    "question_blocks",
    "setup_disclosure",
    "timing_caveats",
]

#: A report-wide disclosure's kind.
_Kind = Literal["late", "untimed", "latency", "settings"]

#: The id each report-wide caveat line carries, by its kind: the mark a fold leaves it out by.
_CAVEAT_ANCHORS: dict[_Kind, str] = {
    "late": "caveat-late",
    "untimed": "caveat-untimed",
    "latency": "caveat-latency",
    "settings": "caveat-settings",
}

#: The id an aspect's own methods line stating a test-setup disclosure in full carries, by its kind.
_SETUP_ANCHORS: dict[Literal["latency", "settings"], str] = {"latency": "setup-latency", "settings": "setup-settings"}

#: The tables stating the report-wide disclosures' detail, on a campaign's page and an aspect's own.
_PROVENANCE_TABLES = frozenset({"criteria_set", "test_setup"})


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


def _lines(bundles: Sequence[AnalysisContextBundle]) -> list[tuple[_Kind, str]]:
    """Each report-wide disclosure that holds, with its one line: the one construction every surface says it by."""
    total = len(bundles)
    lines: list[tuple[_Kind, str]] = []
    if late := [bundle for bundle in bundles if bundle.reading_scope.late_criteria]:
        on = where([aspect_name(bundle) for bundle in late], total)
        unrecorded = any(
            criterion.set_at is None for bundle in late for criterion in bundle.reading_scope.late_criteria
        )
        lines.append(("late", _LATE_LINE[unrecorded].format(where=on)))
    if untimed := [bundle for bundle in bundles if bundle.reading_scope.untimed_results]:
        count = sum(bundle.reading_scope.untimed_results for bundle in untimed)
        on = where([aspect_name(bundle) for bundle in untimed], total)
        lines.append(("untimed", _UNTIMED_LINE["one" if count == 1 else "other"].format(count=count, where=on)))
    if contended := [aspect_name(bundle) for bundle in bundles if bundle.latency_contended]:
        lines.append(("latency", _LATENCY_LINE.format(where=where(contended, total))))
    if unconfirmed := [aspect_name(bundle) for bundle in bundles if bundle.request_settings_disclosure]:
        lines.append(("settings", _SETTINGS_LINE.format(where=where(unconfirmed, total))))
    return lines


#: The line for criteria set after their readings were visible, by whether any has no recorded time.
_LATE_LINE = {
    False: "Criteria on {where} were set after the results there existed, so they were not fixed in advance.",
    True: "Criteria on {where} were set after the results there existed or have no recorded time, so they are not "
    "shown fixed in advance.",
}

#: The line for results with no time that can be ordered, by count form.
_UNTIMED_LINE = {
    "one": "{count} result on {where} carries no time that can be ordered, so whether a criterion was set after it "
    "was visible is not known.",
    "other": "{count} results on {where} carry no time that can be ordered, so whether a criterion was set after "
    "they were visible is not known.",
}

#: The line for latency read under concurrency, which every comparison leaves out, whatever share of it that is.
_LATENCY_LINE = (
    "Latency read while other runs executed beside it is left out of every comparison on {where}; launching with "
    "`measure_latency=True` reads it clean."
)

#: The line for request settings a judge's or a simulated user's client did not confirm as sent.
_SETTINGS_LINE = (
    "Request settings on {where} were not confirmed as sent, so the judge or the simulated user there may have run on "
    "other settings than asked."
)


def timing_caveats(bundles: Sequence[AnalysisContextBundle]) -> list[str]:
    """The report's caveat lines on when its criteria were set, as its first screen says them: one if any criterion
    was set after the readings it judges were visible or has no recorded time, one if any result carries no time that
    can be ordered. An analysis's listing line says these after its warnings.

    :param bundles: Each aspect's evidence, in the campaign's order; one for an aspect's own page.
    :ptype bundles: Sequence[AnalysisContextBundle]
    :return: The lines; none where every criterion came before its readings and every result is timed.
    :rtype: list[str]
    """
    return [line for kind, line in _lines(bundles) if kind in ("late", "untimed")]


def caveat_blocks(bundles: Sequence[AnalysisContextBundle], *, section: ReportSection) -> list[ReportBlock]:
    """A page's first-screen caveat lines, after what warns: when its criteria were set, results with no ordered time,
    latency read under concurrency and request settings not confirmed, each once for the whole page, naming where it
    holds; each carries the id its kind marks it by.

    :param bundles: Each aspect's evidence, in the campaign's order; one for an aspect's own page.
    :ptype bundles: Sequence[AnalysisContextBundle]
    :param section: The first-screen section they sit in: ``warnings`` on a campaign's page, ``header`` on an
        aspect's.
    :ptype section: ReportSection
    :return: The lines; none where nothing holds.
    :rtype: list[ReportBlock]
    """
    return [
        DisclosureBlock(
            section=section,
            source="timing" if kind in ("late", "untimed") else "measurement",
            text=line,
            anchor=_CAVEAT_ANCHORS[kind],
        )
        for kind, line in _lines(bundles)
    ]


def setup_disclosure(
    kind: Literal["latency", "settings"], bundle: AnalysisContextBundle, arm: Callable[[str], str]
) -> DisclosureBlock | None:
    """An aspect's own methods line stating a test-setup disclosure in full: the bundle's sentence, verbatim, and for
    latency the arms it holds for where it does not hold for every arm; None where it does not hold.

    :param kind: Which disclosure.
    :ptype kind: Literal['latency', 'settings']
    :param bundle: The aspect's evidence.
    :ptype bundle: AnalysisContextBundle
    :param arm: An arm's name on the page, by its variant key.
    :ptype arm: Callable[[str], str]
    :return: The line, folded in the methods' disclosures, or None.
    :rtype: DisclosureBlock | None
    """
    text = _setup_detail(kind, bundle, arm)
    if text is None:
        return None
    return DisclosureBlock(
        section="methods", source="measurement", text=text, fold="Disclosures", anchor=_SETUP_ANCHORS[kind]
    )


def _setup_detail(
    kind: Literal["latency", "settings"], bundle: AnalysisContextBundle, arm: Callable[[str], str]
) -> str | None:
    """A test-setup disclosure's detail as every surface states it: the bundle's own sentence, verbatim (the request
    settings' names the role and what was sent against what was asked), and for latency the arms it holds for, where
    it does not hold for every arm."""
    if kind == "settings":
        return (bundle.request_settings_disclosure or "").strip() or None
    sentence = (bundle.latency_contended or "").strip()
    if not sentence:
        return None
    keys = sorted({cell.variant_key for cell in bundle.latency_contended_cells})
    if not keys or set(keys) >= {board.variant_key for board in bundle.scoreboard}:
        return sentence
    return f"{sentence} Read under concurrency: {'; '.join(arm(key) for key in keys)}."


def provenance_blocks(
    campaign: EvalCampaign, bundles: Sequence[AnalysisContextBundle], labels: Mapping[str, str]
) -> list[ReportBlock]:
    """A campaign's Design & provenance section: each criterion set after its readings were visible, with when and on
    which aspects, then each test-setup disclosure in full with the aspects it holds on; nothing where neither holds.

    :param campaign: The campaign, whose folds an aspect's cell links.
    :ptype campaign: EvalCampaign
    :param bundles: Each aspect's evidence, in the campaign's order.
    :ptype bundles: Sequence[AnalysisContextBundle]
    :param labels: The campaign's label for each arm, by variant key, which the detail names arms by.
    :ptype labels: Mapping[str, str]
    :return: The tables, each only where it has a row.
    :rtype: list[ReportBlock]
    """
    folds = dict(zip([a.label for a in campaign.aspects], fold_ids([a.label for a in campaign.aspects]), strict=True))
    by_name = {aspect_name(bundle): bundle.aspect for bundle in bundles}

    def aspects(names: Sequence[str]) -> TableCell:
        # One aspect's cell links its fold; a list of several links none, since a cell links one place.
        one = by_name.get(names[0]) if len(names) == 1 else None
        return TableCell(value=_aspects_cell(names, len(bundles)), ref=f"aspect-{folds[one]}" if one in folds else None)

    blocks = _criteria_table(bundles, section="provenance", cell=aspects)
    setup: dict[tuple[str, str | int], list[str]] = {}
    if untimed := [bundle for bundle in bundles if bundle.reading_scope.untimed_results]:
        count = sum(bundle.reading_scope.untimed_results for bundle in untimed)
        setup[("results with no ordered time", count)] = [aspect_name(bundle) for bundle in untimed]
    kinds: tuple[tuple[Literal["latency", "settings"], str], ...] = (
        ("latency", "latency read under concurrency"),
        ("settings", "request settings not confirmed"),
    )
    for kind, what in kinds:
        for bundle in bundles:
            if (detail := _setup_detail(kind, bundle, _namer(named_arms(bundle, labels)))) is not None:
                setup.setdefault((what, detail), []).append(aspect_name(bundle))
    if setup:
        rows = [
            {"setup": TableCell(value=what), "detail": TableCell(value=detail), "aspects": aspects(names)}
            for (what, detail), names in setup.items()
        ]
        blocks.append(
            TableBlock(
                section="provenance",
                name="test_setup",
                title="Test setup",
                columns=[
                    TableColumn(key="setup", header="Disclosure"),
                    TableColumn(key="detail", header="Detail"),
                    TableColumn(key="aspects", header="Aspects", term="aspect"),
                ],
                rows=rows,
                order="results with no ordered time, latency, then request settings; aspects saying one alike share it",
                total_rows=len(rows),
            )
        )
    return blocks


def _namer(named: Mapping[str, ArmName]) -> Callable[[str], str]:
    """An arm's name by its variant key, as the aspect's fold names it; an arm it does not name by its digest."""
    return lambda key: named[key].label if key in named else short_digest(key)


def aspect_provenance_blocks(bundle: AnalysisContextBundle) -> list[ReportBlock]:
    """An aspect's own page's statement of each criterion set after its readings were visible, and when, in its
    methods: the detail its first screen's one line on them points at.

    :param bundle: The aspect's evidence.
    :ptype bundle: AnalysisContextBundle
    :return: The table, or nothing where every criterion came before its readings.
    :rtype: list[ReportBlock]
    """
    return _criteria_table([bundle], section="methods", cell=lambda names: TableCell(value=_aspects_cell(names, 1)))


def _criteria_table(
    bundles: Sequence[AnalysisContextBundle], *, section: ReportSection, cell: Callable[[Sequence[str]], TableCell]
) -> list[ReportBlock]:
    """Each criterion set late, one row per criterion and time with the aspects whose results existed then."""
    groups: dict[tuple[CriterionKind, str, str | None], list[str]] = {}
    for bundle in bundles:
        for late in bundle.reading_scope.late_criteria:
            key = (late.criterion, _criterion_words(bundle, late), late.page_date())
            groups.setdefault(key, []).append(aspect_name(bundle))
    if not groups:
        return []
    rows = [
        {
            "criterion": TableCell(value=_criterion_noun(kind, what)),
            "set_on": TableCell(value=at or "not recorded"),
            "aspects": cell(names),
        }
        for (kind, what, at), names in groups.items()
    ]
    return [
        TableBlock(
            section=section,
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
    ]


def for_the_campaign(report: Report) -> Report:
    """An aspect's own page as its fold on a campaign's page lays it out: without the report-wide disclosures the
    campaign states once for every aspect, found by the marks they carry (each caveat line's id, each full test-setup
    line's id, each provenance table's name), never by their words.

    :param report: The aspect's own page.
    :ptype report: Report
    :return: The page without them.
    :rtype: Report
    """
    marks = {*_CAVEAT_ANCHORS.values(), *_SETUP_ANCHORS.values()}
    kept = [
        block
        for block in report.blocks
        if not (isinstance(block, DisclosureBlock) and block.anchor in marks)
        and not (isinstance(block, TableBlock) and block.name in _PROVENANCE_TABLES)
    ]
    return report.model_copy(update={"blocks": kept})


def _aspects_cell(names: Sequence[str], total: int) -> str:
    """The aspects a row holds on, as its cell lists them: ``every aspect`` where it holds on all of several."""
    return "every aspect" if total > 1 and len(names) == total else listed(names)


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
