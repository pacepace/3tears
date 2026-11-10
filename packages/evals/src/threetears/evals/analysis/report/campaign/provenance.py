"""What a page says of how it came to be: the question, quoted whole as the campaign asks it; one caveat line for each
disclosure that holds across the report; and where each is stated as data, once.

A report-wide disclosure is said once, at the level it holds on: one line on the first screen, worded as what it
means for the readings, and its detail in one place. On a campaign's page the detail is a row of Design &
provenance, and no aspect's fold repeats it; on an aspect's own page it is the aspect's methods. It is never recited
per aspect or per finding, and it never lowers a reading or a recommendation: a criterion set after its readings
were visible is disclosed, never a demotion, since its readings read as any others. When the runs ran is never said.
Six disclosures are report-wide:

- **criteria set late**: a criterion (an aspect, the control, a margin, a bar, a guardrail's margin or a floor) set
  after the readings it judges were visible, or whose time is not recorded;
- **results with no ordered time**, of which whether a criterion came after them is not known;
- **latency read under concurrency**, which every comparison leaves out;
- **request settings not confirmed**, for the judge or the simulated user;
- **a judge not yet calibrated** against people's labels, whose scores are then directional
  (``AnalysisContextBundle.judge_evidence_tiers``): a state with its next step, never a failure.

Each of these has its line on the first screen (:func:`caveat_blocks`) and its detail one step down
(:func:`provenance_blocks`, :func:`aspect_provenance_blocks`, :func:`setup_disclosure`, :func:`calibration_blocks`).
A sixth disclosure has its detail only: **cost as routed**, that every cost figure is what the runs' provider
routing charged, not the model's alone, is one row of Design & provenance on a campaign's page and one methods line
on an aspect's, never a first-screen line (owner reading, 2026-11-06: to a business reader the cost is what it costs).
What an aspect's fold on a campaign's page leaves out because the campaign states it (:func:`for_the_campaign`) is
found by the marks those carry, never by their words.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from threetears.evals.analysis.arms import short_digest
from threetears.evals.analysis.contention import reader_latency_sentence
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, CriterionKind, LateCriterion
from threetears.evals.analysis.report.campaign.common import named_arms
from threetears.evals.analysis.report.campaign.names import AspectNames, measure_name_of
from threetears.evals.analysis.report.campaign.warnings import listed, where
from threetears.evals.analysis.report.decision import floor_statement
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.model import (
    DisclosureBlock,
    DisclosureSource,
    Report,
    ReportBlock,
    ReportSection,
    TableBlock,
    TableCell,
    TableColumn,
)
from threetears.evals.analysis.report.words import judged_name
from threetears.evals.analysis.style import ArmName
from threetears.evals.kernel.campaign import EvalCampaign

__all__ = [
    "aspect_provenance_blocks",
    "calibration_blocks",
    "caveat_blocks",
    "for_the_campaign",
    "provenance_blocks",
    "question_blocks",
    "setup_disclosure",
    "timing_caveats",
]

#: The line in the question's place where a campaign that compares arms asks none: what the page is, said once, so
#: no reading under it says it again. The glossary's ``exploratory`` says what that makes its readings.
_EXPLORATORY = "Exploratory comparison."

#: A report-wide disclosure's kind, each with a line on the first screen.
_Kind = Literal["late", "untimed", "latency", "settings", "judge"]

#: The id each report-wide caveat line carries, by its kind: the mark a fold leaves it out by.
_CAVEAT_ANCHORS: dict[_Kind, str] = {
    "late": "caveat-late",
    "untimed": "caveat-untimed",
    "latency": "caveat-latency",
    "settings": "caveat-settings",
    "judge": "caveat-judge",
}

#: What each report-wide caveat line speaks for, by its kind. A judge's is said as how its scores were measured;
#: ``calibration`` is its detail's, the state with its next step.
_CAVEAT_SOURCES: dict[_Kind, DisclosureSource] = {
    "late": "timing",
    "untimed": "timing",
    "latency": "measurement",
    "settings": "measurement",
    "judge": "measurement",
}

#: A disclosure whose detail is stated in full one step down: on an aspect's own page in its methods (the judge's in
#: what was not tested), on a campaign's page as a row of Design & provenance.
_SetupKind = Literal["latency", "settings", "judge", "cost"]

#: The id an aspect's own line stating a disclosure in full carries, by its kind.
_SETUP_ANCHORS: dict[_SetupKind, str] = {
    "latency": "setup-latency",
    "settings": "setup-settings",
    "judge": "setup-judge",
    "cost": "setup-cost",
}

#: What an aspect's own methods line stating a disclosure in full speaks for, by its kind.
_SETUP_SOURCES: dict[Literal["latency", "settings", "cost"], DisclosureSource] = {
    "latency": "measurement",
    "settings": "measurement",
    "cost": "cost",
}

#: Each disclosure's name in the Design & provenance table, in the order its rows are laid out.
_SETUP_NAMES: tuple[tuple[_SetupKind, str], ...] = (
    ("latency", "latency read under concurrency"),
    ("settings", "request settings not confirmed"),
    ("judge", "judge scores directional"),
    ("cost", "cost as routed"),
)

#: The tables stating the report-wide disclosures' detail, on a campaign's page and an aspect's own.
_PROVENANCE_TABLES = frozenset({"criteria_set", "test_setup"})


def question_blocks(campaign: EvalCampaign, *, compares: bool) -> list[ReportBlock]:
    """The campaign's question, quoted whole on one line, the first under the title; where it asks none and the page
    compares arms, what the page is (``Exploratory comparison.``), once, its word linked to the glossary; where it
    asks none of one arm measured, nothing: the lead says what the page is, the arm's figure and its count.

    :param campaign: The campaign.
    :ptype campaign: EvalCampaign
    :param compares: Whether the page compares arms (a change or a choice), not one arm measured.
    :ptype compares: bool
    :return: The question's line, the line saying the page is exploratory, or nothing.
    :rtype: list[ReportBlock]
    """
    if not campaign.question.strip():
        if not compares:
            return []
        return [DisclosureBlock(section="answer", source="comparisons", text=_EXPLORATORY, term="exploratory")]
    return [DisclosureBlock(section="answer", source="basis", text=f'"{" ".join(campaign.question.split())}"')]


def _lines(bundles: Sequence[AnalysisContextBundle], names: AspectNames | None = None) -> list[tuple[_Kind, str]]:
    """Each report-wide disclosure that holds, with its one line: the one construction every surface says it by, each
    aspect by what ``names`` calls it (else by :func:`aspect_name`)."""
    total = len(bundles)
    named = names if names is not None else AspectNames.of(bundles)
    lines: list[tuple[_Kind, str]] = []
    if late := [bundle for bundle in bundles if bundle.reading_scope.late_criteria]:
        on = where([named(bundle.aspect) for bundle in late], total)
        unrecorded = any(
            criterion.set_at is None for bundle in late for criterion in bundle.reading_scope.late_criteria
        )
        held = {criterion.criterion for bundle in late for criterion in bundle.reading_scope.late_criteria}
        # In the order a design declares them: the aspect, the control, the margin, then its bars, guardrails, floors.
        kinds = [kind for kind in _LATE_NOUNS if kind in held]
        # Past three kinds the line says criteria, and the table one step down names each.
        what = f"The {listed([_LATE_NOUNS[kind] for kind in kinds])}" if len(kinds) <= 3 else "Criteria"
        several = len(kinds) > 1 or any(
            sum(criterion.criterion == kinds[0] for criterion in bundle.reading_scope.late_criteria) > 1
            for bundle in late
        )
        line = _LATE_LINE[unrecorded]["other" if several else "one"]
        lines.append(("late", line.format(what=what, where=on)))
    if untimed := [bundle for bundle in bundles if bundle.reading_scope.untimed_results]:
        count = sum(bundle.reading_scope.untimed_results for bundle in untimed)
        on = where([named(bundle.aspect) for bundle in untimed], total)
        lines.append(("untimed", _UNTIMED_LINE["one" if count == 1 else "other"].format(count=count, where=on)))
    if contended := [named(bundle.aspect) for bundle in bundles if bundle.latency_contended]:
        lines.append(("latency", _LATENCY_LINE.format(where=where(contended, total))))
    if unconfirmed := [named(bundle.aspect) for bundle in bundles if bundle.request_settings_disclosure]:
        lines.append(("settings", _SETTINGS_LINE.format(where=where(unconfirmed, total))))
    if directional := [named(bundle.aspect) for bundle in bundles if _calibration_sentences(bundle)]:
        lines.append(("judge", _JUDGE_LINE.format(where=where(directional, total))))
    return lines


#: What a late criterion of each kind is called where the late line names what was set late.
_LATE_NOUNS: dict[CriterionKind, str] = {
    "aspect": "aspect",
    "control": "control",
    "margin": "margin",
    "bar": "bar",
    "guardrail": "guardrail margin",
    "floor": "floor",
}

#: The line for criteria set after their readings were visible, naming what was set, by whether any has no recorded
#: time and by count. It leads the report-wide caveats, the one set apart as a flag
#: (:data:`~threetears.evals.analysis.report.blocks.FLAGGED_CAVEATS`).
_LATE_LINE = {
    False: {
        "one": "{what} on {where} was set after the results there existed, so it was not fixed in advance.",
        "other": "{what} on {where} were set after the results there existed, so they were not fixed in advance.",
    },
    True: {
        "one": "{what} on {where} was set after its results existed or has no recorded time, so it is not shown "
        "fixed in advance.",
        "other": "{what} on {where} were set after its results existed or have no recorded time, so they are not "
        "shown fixed in advance.",
    },
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


#: The line for a judge not yet calibrated against people's labels: what that makes its scores, and what changes it.
#: Its detail, which judge and how many outputs to rate, is one step down (:func:`calibration_blocks`).
_JUDGE_LINE = "Judged scores on {where} are directional until people's labels calibrate the judge."


def timing_caveats(bundles: Sequence[AnalysisContextBundle], names: AspectNames | None = None) -> list[str]:
    """The report's caveat lines on when its criteria were set, as its first screen says them: one if any criterion
    was set after the readings it judges were visible or has no recorded time, one if any result carries no time that
    can be ordered. An analysis's listing line says these after its warnings.

    :param bundles: Each aspect's evidence, in the campaign's order; one for an aspect's own page.
    :ptype bundles: Sequence[AnalysisContextBundle]
    :param names: What the page calls each aspect; None names each by :func:`aspect_name`.
    :ptype names: AspectNames | None
    :return: The lines; none where every criterion came before its readings and every result is timed.
    :rtype: list[str]
    """
    return [line for kind, line in _lines(bundles, names) if kind in ("late", "untimed")]


def caveat_blocks(
    bundles: Sequence[AnalysisContextBundle], *, section: ReportSection, names: AspectNames | None = None
) -> list[ReportBlock]:
    """A page's first-screen caveat lines, after what warns: when its criteria were set, results with no ordered time,
    latency read under concurrency, request settings not confirmed, a judge not yet calibrated, each once for the
    whole page, naming where it holds; each carries the id its kind marks it by.

    :param bundles: Each aspect's evidence, in the campaign's order; one for an aspect's own page.
    :ptype bundles: Sequence[AnalysisContextBundle]
    :param section: The first-screen section they sit in: ``warnings`` on a campaign's page, ``header`` on an
        aspect's.
    :ptype section: ReportSection
    :param names: What the page calls each aspect; None names each by :func:`aspect_name`.
    :ptype names: AspectNames | None
    :return: The lines; none where nothing holds.
    :rtype: list[ReportBlock]
    """
    return [
        DisclosureBlock(
            section=section,
            source=_CAVEAT_SOURCES[kind],
            text=line,
            anchor=_CAVEAT_ANCHORS[kind],
            # The judged scale is one click from the first screen: the judge's line links its glossary entry.
            term="judged score" if kind == "judge" else None,
        )
        for kind, line in _lines(bundles, names)
    ]


#: The detail on a page that prints any cost figure: what an eval spent depends on which providers served its calls,
#: which can differ by several times for one model, while its quality does not (owner, 2026-10-25). It is said one step
#: down, never on the first screen (owner reading, 2026-11-06).
COST_ROUTING_CAVEAT = (
    "Each cost figure reflects the provider routing of the runs it was measured on, not the model alone."
)


def priced_aspects(bundles: Sequence[AnalysisContextBundle]) -> list[str]:
    """The aspects on which any arm carries a cost, by label, in the campaign's order: where the page prints a cost
    figure and the writer's view hands one.

    :param bundles: Each aspect's evidence, in the campaign's order.
    :ptype bundles: Sequence[AnalysisContextBundle]
    :return: The labels; none where no arm on any aspect was priced.
    :rtype: list[str]
    """
    return [
        bundle.aspect
        for bundle in bundles
        if any(board.test_cost_usd is not None or board.reply_cost_usd is not None for board in bundle.scoreboard)
    ]


def setup_disclosure(
    kind: Literal["latency", "settings", "cost"],
    bundle: AnalysisContextBundle,
    arm: Callable[[str], str],
    *,
    forms: PageForms,
) -> DisclosureBlock | None:
    """An aspect's own methods line stating a disclosure in full: the bundle's sentence, verbatim, and for latency the
    arms it holds for where it does not hold for every arm; for cost, where any arm was priced, that each cost figure
    is its runs' provider routing's; None where it does not hold.

    :param kind: Which disclosure.
    :ptype kind: Literal['latency', 'settings', 'cost']
    :param bundle: The aspect's evidence.
    :ptype bundle: AnalysisContextBundle
    :param arm: An arm's name on the page, by its variant key.
    :ptype arm: Callable[[str], str]
    :param forms: The page's forms, which print the counts a stored sentence holds.
    :ptype forms: PageForms
    :return: The line, folded in the methods' disclosures, or None.
    :rtype: DisclosureBlock | None
    """
    text = _setup_detail(kind, bundle, arm, forms)
    if text is None:
        return None
    return DisclosureBlock(
        section="methods", source=_SETUP_SOURCES[kind], text=text, fold="Disclosures", anchor=_SETUP_ANCHORS[kind]
    )


def calibration_blocks(bundle: AnalysisContextBundle) -> list[ReportBlock]:
    """An aspect's own line on each judge not yet calibrated against people's labels, in what was not tested: where
    each judge stands and what to do next, the detail its first screen's caveat line points at; nothing where every
    judge is calibrated or has met or missed its criterion.

    :param bundle: The aspect's evidence.
    :ptype bundle: AnalysisContextBundle
    :return: The line, or nothing.
    :rtype: list[ReportBlock]
    """
    said = _calibration_sentences(bundle)
    if not said:
        return []
    return [
        DisclosureBlock(
            section="not_tested",
            source="calibration",
            text=" ".join(said),
            term="calibrated",
            anchor=_SETUP_ANCHORS["judge"],
        )
    ]


def _calibration_sentences(bundle: AnalysisContextBundle) -> list[str]:
    """Each judge not yet calibrated against people, as a state and its next step, never as a failure.

    A new judge has no labels, so on a first campaign this is said on every report: where the judge stands and what
    to do next, one sentence per judge and state, every dimension it holds for named once.
    """
    by_judge: dict[tuple[str, bool, int], list[str]] = {}
    for tier in bundle.judge_evidence_tiers:
        criterion = tier.calibration
        if tier.tier == "calibrated" or criterion.state in ("met", "not_met"):
            continue
        key = (
            tier.judge_model or "an unnamed judge",
            criterion.n == 0,
            criterion.results_needed or criterion.min_results,
        )
        dims = by_judge.setdefault(key, [])
        if (dim := judged_name(tier.rubric_dim)) not in dims:
            dims.append(dim)
    said = []
    for (judge, none_yet, more), dims in by_judge.items():
        state = "is uncalibrated" if none_yet else "is not yet calibrated"
        why = "no human labels yet" if none_yet else "too few human labels so far"
        said.append(
            f"Judge {judge} {state} on {', '.join(dims)}: {why}, so its scores are directional. "
            f"Rate about {more} of its outputs to calibrate it."
        )
    return said


def _setup_detail(
    kind: _SetupKind, bundle: AnalysisContextBundle, arm: Callable[[str], str], forms: PageForms
) -> str | None:
    """A disclosure's detail as every surface states it: the bundle's own sentence, verbatim (the request settings'
    names the role and what was sent against what was asked), and for latency the arms it holds for, where it does
    not hold for every arm; each judge's state and next step; the cost's routing, where any arm was priced."""
    if kind == "settings":
        return _settings_in_words(forms.stored_counts((bundle.request_settings_disclosure or "").strip())) or None
    if kind == "judge":
        return " ".join(_calibration_sentences(bundle)) or None
    if kind == "cost":
        return COST_ROUTING_CAVEAT if priced_aspects([bundle]) else None
    sentence = reader_latency_sentence((bundle.latency_contended or "").strip())
    if not sentence:
        return None
    keys = sorted({cell.variant_key for cell in bundle.latency_contended_cells})
    if not keys or set(keys) >= {board.variant_key for board in bundle.scoreboard}:
        return sentence
    return f"{sentence} Read under concurrency: {'; '.join(arm(key) for key in keys)}."


#: A stored sentence's request settings as the engine quotes them (``{max_tokens=10240, reasoning_effort=high,
#: strict}``), and the field its reports name them by.
_SETTINGS_PHRASE = re.compile(r"\{([^{}]*=[^{}]*)\}")
_SETTINGS_FIELD = " (sent_settings)"

#: Each quoted setting as a sentence says it.
_SETTING_WORDS = {
    "max_tokens": "max tokens",
    "reasoning_effort": "reasoning effort",
    "reasoning_max_tokens": "reasoning max tokens",
}


def _settings_in_words(sentence: str) -> str:
    """A stored request-settings sentence as a page prints it: each quoted set of settings in words (``max tokens
    10240, reasoning max tokens 8192, strict output``), never ``key=value``, and no field name. The stored sentence
    stays as frozen; only its printing changes."""

    def words(match: re.Match[str]) -> str:
        said = []
        for part in match.group(1).split(", "):
            key, equals, value = part.partition("=")
            if equals:
                said.append(f"{_SETTING_WORDS.get(key, key.replace('_', ' '))} {value}")
            else:
                said.append({"strict": "strict output", "not strict": "no strict output"}.get(part, part))
        return f"({', '.join(said)})"

    return _SETTINGS_PHRASE.sub(words, sentence).replace(_SETTINGS_FIELD, "")


def provenance_blocks(
    bundles: Sequence[AnalysisContextBundle], labels: Mapping[str, str], *, names: AspectNames, forms: PageForms
) -> list[ReportBlock]:
    """A campaign's Design & provenance section: each criterion set after its readings were visible, with when and on
    which aspects, then each report-wide disclosure in full with the aspects it holds on (results with no ordered
    time, latency, request settings, each judge not yet calibrated, cost as routed); nothing where none holds.

    :param bundles: Each aspect's evidence, in the campaign's order.
    :ptype bundles: Sequence[AnalysisContextBundle]
    :param labels: The campaign's label for each arm, by variant key, which the detail names arms by.
    :ptype labels: Mapping[str, str]
    :param names: What the page calls each aspect, and where each one's fold is, which a cell of one aspect links.
    :ptype names: AspectNames
    :param forms: The page's forms, which state each floor's margin, the counts a stored sentence holds, and a count a
        detail states (results with no ordered time).
    :ptype forms: PageForms
    :return: The tables, each only where it has a row.
    :rtype: list[ReportBlock]
    """

    def aspects(held: Sequence[str]) -> TableCell:
        # One aspect's cell links its fold; a list of several links none, since a cell links one place.
        ref = names.folds.get(held[0]) if len(held) == 1 else None
        return TableCell(value=_aspects_cell([names(label) for label in held], len(bundles)), ref=ref)

    blocks = _criteria_table(bundles, section="provenance", cell=aspects, forms=forms)
    setup: dict[tuple[str, str | int], list[str]] = {}
    if untimed := [bundle for bundle in bundles if bundle.reading_scope.untimed_results]:
        count = sum(bundle.reading_scope.untimed_results for bundle in untimed)
        setup[("results with no ordered time", count)] = [bundle.aspect for bundle in untimed]
    for kind, what in _SETUP_NAMES:
        for bundle in bundles:
            if (detail := _setup_detail(kind, bundle, _namer(named_arms(bundle, labels)), forms)) is not None:
                setup.setdefault((what, detail), []).append(bundle.aspect)
    if setup:
        # A disclosure's name is its row's label, so it starts with a capital as every row label does.
        rows = [
            {"setup": TableCell(value=_label(what)), "detail": forms.count_cell(detail), "aspects": aspects(held)}
            for (what, detail), held in setup.items()
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
                order=(
                    "results with no ordered time, latency, request settings, the judge, then cost; aspects saying one "
                    "alike share it"
                ),
                total_rows=len(rows),
            )
        )
    return blocks


def _label(words: str) -> str:
    """A row's label as a table prints it: its words, starting with a capital (``Latency read under concurrency``,
    ``Aspect; control; margin``); each starts with a word of code's, never a name or an expression."""
    return words[:1].upper() + words[1:]


def _namer(named: Mapping[str, ArmName]) -> Callable[[str], str]:
    """An arm's name by its variant key, as the aspect's fold names it; an arm it does not name by its digest."""
    return lambda key: named[key].label if key in named else short_digest(key)


def aspect_provenance_blocks(bundle: AnalysisContextBundle, *, forms: PageForms) -> list[ReportBlock]:
    """An aspect's own page's statement of each criterion set after its readings were visible, and when, in its
    methods: the detail its first screen's one line on them points at.

    :param bundle: The aspect's evidence.
    :ptype bundle: AnalysisContextBundle
    :param forms: The page's forms, which state each floor's margin.
    :ptype forms: PageForms
    :return: The table, or nothing where every criterion came before its readings.
    :rtype: list[ReportBlock]
    """
    names = AspectNames.of([bundle])
    return _criteria_table(
        [bundle],
        section="methods",
        cell=lambda held: TableCell(value=_aspects_cell([names(label) for label in held], 1)),
        forms=forms,
    )


def _criteria_table(
    bundles: Sequence[AnalysisContextBundle],
    *,
    section: ReportSection,
    cell: Callable[[Sequence[str]], TableCell],
    forms: PageForms,
) -> list[ReportBlock]:
    """Each criterion set late, one row per criterion and time with the aspects whose results existed then, the
    aspects by label for ``cell`` to name and link."""
    groups: dict[tuple[CriterionKind, str, str | None], list[str]] = {}
    for bundle in bundles:
        for late in bundle.reading_scope.late_criteria:
            key = (late.criterion, _criterion_words(bundle, late, forms), late.page_date())
            groups.setdefault(key, []).append(bundle.aspect)
    if not groups:
        return []
    # Criteria set at one time on the same aspects share a row, each named once in the order declared.
    merged: dict[tuple[str | None, tuple[str, ...]], list[str]] = {}
    for (kind, what, at), held in groups.items():
        merged.setdefault((at, tuple(held)), []).append(_criterion_noun(kind, what))
    # One criterion's words may hold a comma (a measure's name), so a row's criteria are set apart by semicolons; the
    # row's label starts with a capital as every row label does.
    rows = [
        {
            "criterion": TableCell(value=_label("; ".join(nouns))),
            "set_on": TableCell(value=at or "not recorded"),
            "aspects": cell(list(held)),
        }
        for (at, held), nouns in merged.items()
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
            order=(
                "each criterion as the aspects declare it, in the campaign's order; those set at one time on the same "
                "aspects share a row"
            ),
            total_rows=len(rows),
        )
    ]


def for_the_campaign(report: Report, *, alone: bool = False) -> Report:
    """An aspect's own page as its fold on a campaign's page lays it out: without the report-wide disclosures the
    campaign states once for every aspect, found by the marks they carry (each caveat line's id, each full test-setup
    line's id, each provenance table's name), never by their words.

    A fold laid out ``alone``, without the campaign's own blocks (a report document's evidence record, which is the
    folds and the glossary), keeps its judge's calibration line (``setup-judge``): no other line on that page names
    the judge, its dimensions or how many of its outputs to rate.

    :param report: The aspect's own page.
    :ptype report: Report
    :param alone: Whether the fold is laid out without the campaign's own blocks.
    :ptype alone: bool
    :return: The page without them.
    :rtype: Report
    """
    marks = {*_CAVEAT_ANCHORS.values(), *_SETUP_ANCHORS.values()} - ({_SETUP_ANCHORS["judge"]} if alone else set())
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


def _criterion_words(bundle: AnalysisContextBundle, late: LateCriterion, forms: PageForms) -> str:
    """Which criterion of its kind ``late`` is, as a row names it: a bar by its measure, a guardrail by its dimension,
    a floor by what it says, in the page's forms."""
    if late.criterion == "bar" and late.name is not None:
        return measure_name_of(bundle, late.name)
    if late.criterion == "guardrail" and late.name is not None:
        return judged_name(late.name)
    if late.criterion == "floor" and bundle.declared is not None:
        floor = next((floor for floor in bundle.declared.floors if floor.key == late.name), None)
        if floor is not None:
            return forms.said(floor_statement(floor.clause, candidate="the arm", check_names=bundle.named_checks()))
    return late.name or ""


def _criterion_noun(kind: CriterionKind, what: str) -> str:
    """A criterion as its row names it: ``aspect``, ``control``, ``margin``, ``bar on cost``, ``guardrail margin on
    tone``, ``floor that no pass rate drops by more than 10 points``."""
    if kind in ("aspect", "control", "margin") or not what:
        return kind
    return {"bar": f"bar on {what}", "guardrail": f"guardrail margin on {what}", "floor": f"floor that {what}"}[kind]
