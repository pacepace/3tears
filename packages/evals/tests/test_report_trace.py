"""A finding's trace: from what the memo says to the figures and trials behind it, drawn by code, on both pages.

Each figure in a finding's readings table links to a cell printing the same figure of the same reading at the same
cell: a contrast's arm or control, a guardrail's mean, a one-arm page's reading, an arm's cost per reply; or, where
that cell sits in a table shown closed (the bars), to the table's fold. Never to another number: a contrast's paired
mean over fewer cases is not the cell's figure, and is not linked to. Each trial id a finding names links to the
Trials section's heading. On a campaign's page each finding's ``Evidence`` fold lists what it cites, with its figure,
linked to the heading of the aspect's fold that prints it. No trace link lands inside a closed ``<details>``, where a
browser that does not open it would leave the reader nowhere. A finding whose trace reaches nothing says so, and the
page model refuses a link to a place it does not hold.
"""

from __future__ import annotations

from html import escape
import json
import re
from collections.abc import Mapping, Sequence
from html.parser import HTMLParser
from typing import Any

import pytest

from threetears.evals import Answer, Eval, Result, grader
from threetears.evals.analysis import (
    AnalysisContextBundle,
    CampaignReport,
    DisclosureBlock,
    Report,
    TableBlock,
    TableCell,
    TableColumn,
    analysis_report,
    build_report,
    campaign_report,
    campaign_report_html,
    generate_analysis,
    report_html,
)
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.references import campaign_cell_aliases, cell_aliases, resolve_reading
from threetears.evals.analysis.report.model import fold_ids, reading_anchor
from threetears.evals.analysis.report.words import UNTRACED
from threetears.evals.kernel.authored import EvidenceRef, Finding
from threetears.evals.kernel.campaign import EvalAnalysis, EvidenceRow, FindingResolution
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.campaign_bundle_support import memo as authored_memo
from packages.evals.tests.g2_support import CANDIDATE, PRODUCTION, answering
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.report_support import toy_report
from packages.evals.tests.toyhost_memo import MODEL, PROMPT, PROMPT_ID, FixturedClient

#: Production passes 30 of its 36 trials, the candidate 6: a contrast on accuracy the page tests and states.
_CASES = [{"text": f"queue request {index}", "label": "queued"} for index in range(12)]

#: The findings' headlines: one on a reading the contrasts state, one on a reading no table states, one on a trial.
TESTED = "The candidate answers fewer requests right"
UNSTATED = "The candidate matched fewer labels"
ON_A_TRIAL = "One request went wrong"

#: The tables a finding's trace is drawn from: its readings, its facts, and a campaign page's trace.
_TRACE_TABLES = frozenset({"evidence", "finding_facts", "trace"})


def _finding(title: str, measure: str, cells: Mapping[str, str], trials: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "title": title,
        "body": "",
        "confidence": "low",
        "axes": [],
        "evidence": [{"cell": cell, "measure_id": measure, "reading": "measure"} for cell in cells.values()],
        "trials": list(trials),
        "caveats": [],
        "invalidates": [],
        "durable": "",
    }


async def _generated(
    comparison: Result, bundle: CampaignBundle | AnalysisContextBundle, findings: Sequence[dict[str, Any]] = ()
) -> EvalAnalysis:
    """A stored analysis of ``findings`` over ``bundle``, written by a fixtured writer; a campaign of one arm
    recommends nothing."""
    campaign = bundle if isinstance(bundle, CampaignBundle) else campaign_bundle(bundle)
    memo = authored_memo(campaign, list(findings), means="Keep production.")
    if len(campaign.arms) == 1:
        memo["recommendation"] = []
    analysis, _ = await generate_analysis(
        campaign,
        prompt=PROMPT,
        model=MODEL,
        client=FixturedClient(json.dumps(memo)),
        prompt_id=PROMPT_ID,
        bundle_assembled_at="2026-10-14T19:23:41+00:00",
        profile=comparison.host.profile,
    )
    off_loop(comparison.host.storage.save_analysis, analysis)
    return analysis


async def _analysed() -> tuple[Result, EvalAnalysis]:
    """Two arms, production the control, and a stored analysis of three findings: on accuracy, which the contrasts
    test; on the label match, which no table on the page states; and on one trial alone."""
    comparison = await Eval(
        _CASES, {PRODUCTION: answering(30, 0.003092), CANDIDATE: answering(6, 0.003301)}, ["label"], k=3, name="queue"
    ).run(store=":memory:")
    bundle: AnalysisContextBundle = comparison.report.bundle
    by_variant = {ref.split(":")[0]: alias for alias, ref in cell_aliases(bundle.cell_measures).items()}
    cells = {name: by_variant[comparison.arm_variants[name]] for name in (PRODUCTION, CANDIDATE)}
    findings = [
        _finding(TESTED, "accuracy", cells),
        _finding(UNSTATED, "match", cells),
        _finding(ON_A_TRIAL, "match", {}, ("T40",)),
    ]
    return comparison, await _generated(comparison, bundle, findings)


def _citing(analysis: EvalAnalysis, measure: str, title: str = "The arms read differently here") -> EvalAnalysis:
    """``analysis`` holding one finding, citing ``measure`` on every cell of its one aspect, its figures as code
    resolves them."""
    (held,) = analysis.aspects
    surface = held.decision_surface
    rows: list[EvidenceRow] = []
    for cell in surface.cells:
        ref = cell_ref(cell.variant_key, cell.apparatus_class_id)
        resolved = resolve_reading(surface, ref, measure, "measure")
        rows.append(
            EvidenceRow(
                aspect=held.label,
                cell_ref=ref,
                measure_id=measure,
                reading="measure",
                value=resolved.mean,
                n=resolved.n,
                n_cases=resolved.n_cases,
                dispersion=resolved.dispersion,
            )
        )
    finding = Finding(
        title=title,
        body="",
        confidence="low",
        axes=[],
        evidence=[EvidenceRef(cell=row.cell_ref, measure_id=measure, reading="measure") for row in rows],
        trials=[],
        caveats=[],
        invalidates=[],
        durable="",
    )
    recommendation = [item.model_copy(update={"rests_on": []}) for item in analysis.document.recommendation]
    document = analysis.document.model_copy(update={"findings": [finding], "recommendation": recommendation})
    return analysis.model_copy(update={"document": document, "resolutions": [FindingResolution(evidence=rows)]})


def _tables(blocks: Sequence[Any], name: str, finding: int | None = None) -> list[TableBlock]:
    return [
        block
        for block in blocks
        if isinstance(block, TableBlock) and block.name == name and (finding is None or block.finding == finding)
    ]


def _figures(page: Report, finding: int = 0) -> list[TableCell]:
    """A finding's figures in its readings table, row by row."""
    (evidence,) = _tables(page.blocks, "evidence", finding=finding)
    return [cell for row in evidence.rows for key, cell in row.items() if key.startswith("reading_")]


def _carrying(page: Report, anchor: str) -> tuple[TableBlock, TableCell]:
    """The one cell on the page carrying ``anchor``, and its table."""
    (found,) = [
        (block, cell)
        for block in page.blocks
        if isinstance(block, TableBlock)
        for row in block.rows
        for cell in row.values()
        if cell.anchor == anchor
    ]
    return found


def _untraced(blocks: Sequence[Any]) -> list[int | None]:
    """The findings whose trace says it reaches nothing, by position."""
    return [block.finding for block in blocks if isinstance(block, DisclosureBlock) and block.text == UNTRACED]


class _Landings(HTMLParser):
    """Every id with the closed ``<details>`` holding it, and every link with the closed ``<details>`` holding it and
    the table it sits in."""

    _VOID = frozenset({"meta", "br", "img", "input", "hr", "link", "wbr"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, int | None, str | None]] = []
        self.folds = 0
        self.held: dict[str, frozenset[int]] = {}
        self.links: list[tuple[str, frozenset[int], str | None]] = []

    def _shut(self) -> frozenset[int]:
        return frozenset(fold for _, fold, _ in self.stack if fold is not None)

    def _start(self, tag: str, attrs: list[tuple[str, str | None]]) -> tuple[int | None, str | None]:
        found = dict(attrs)
        if (name := found.get("id")) is not None:
            assert name not in self.held, f"the id {name} is carried twice"
            self.held[name] = self._shut()
        table = found.get("data-table") or (self.stack[-1][2] if self.stack else None)
        if tag == "a" and (href := found.get("href")) is not None:
            # A link in a table's body is a figure's or a trial's; a header's links its glossary term.
            in_cell = any(open_tag == "td" for open_tag, _, _ in self.stack)
            self.links.append((href, self._shut(), table if in_cell else None))
        fold = None
        if tag == "details" and "open" not in found:
            self.folds += 1
            fold = self.folds
        return fold, table

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        fold, table = self._start(tag, attrs)
        if tag not in self._VOID:
            self.stack.append((tag, fold, table))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._start(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        assert any(open_tag == tag for open_tag, _, _ in self.stack), f"</{tag}> closes nothing open"
        while self.stack.pop()[0] != tag:
            pass


def _trace_links_land_in_sight(page: str) -> list[str]:
    """Every trace link's target, after checking every link on the page is fragment-only and lands where every
    browser shows it: on an id the page carries, held by no closed ``<details>`` the reader has not already opened to
    see the link (a fold's own ``<details>``, or the section around it, is outside it)."""
    parsed = _Landings()
    parsed.feed(page)
    parsed.close()
    assert not parsed.stack, "an element is left open"
    targets = []
    for href, shut, table in parsed.links:
        assert href.startswith("#"), f"{href} is not a link to a place on the page"
        assert href[1:] in parsed.held, f"{href} lands nowhere"
        assert parsed.held[href[1:]] <= shut, f"{href} lands inside a closed fold"
        if table in _TRACE_TABLES:
            targets.append(href[1:])
    assert "<base" not in page
    return targets


# =============================================================================
# An aspect's page
# =============================================================================


async def test_a_finding_s_figures_link_to_the_contrast_that_prints_each() -> None:
    _, analysis = await _analysed()
    page = build_report(analysis, campaign_name="queue")
    linked = [cell.ref for cell in _figures(page)]
    assert len(linked) == 2 and None not in linked
    # Each figure lands on its arm's own figure in the contrast on that reading: the control's and the candidate's.
    (contrasts,) = _tables(page.blocks, "comparisons")
    (accuracy,) = [row for row in contrasts.rows if row["reading"].text == "Accuracy"]
    assert linked == [accuracy["control"].anchor, accuracy["arm"].anchor]
    assert 0 not in _untraced(page.blocks)
    html = report_html(page)
    for place in linked:
        assert f'<a href="#{place}">' in html and f'<td id="{place}"' in html
    assert set(_trace_links_land_in_sight(html)) >= set(linked)


async def test_a_finding_whose_figures_no_table_prints_links_nothing_and_says_so_in_its_fold() -> None:
    _, analysis = await _analysed()
    page = build_report(analysis, campaign_name="queue")
    assert all(cell.ref is None for cell in _figures(page, finding=1))
    (said,) = [block for block in page.blocks if isinstance(block, DisclosureBlock) and block.text == UNTRACED]
    assert said.finding == 1 and said.fold == "Evidence" and said.section == "memo"
    html = report_html(page)
    fold = html[html.index('id="F2"') :].split("</details>", 1)[0]
    assert escape(UNTRACED) in fold and 'href="#r-' not in fold


async def test_a_finding_s_trial_ids_link_to_the_trials_heading_and_trace_it() -> None:
    _, analysis, page = await toy_report()
    # The toy campaign has no control, so no table prints the turn time its one finding cites: the trace says so.
    assert _untraced(page.blocks) == [0]
    finding = analysis.document.findings[0].model_copy(update={"trials": ["T1"]})
    traced = build_report(
        analysis.model_copy(update={"document": analysis.document.model_copy(update={"findings": [finding]})})
    )
    assert _untraced(traced.blocks) == []
    html = report_html(traced)
    facts = html[html.index('data-table="finding_facts"') :].split("</table>", 1)[0]
    # The record is in the closed Trials section; the link lands on its heading, which every browser shows.
    assert '<a href="#trials">T1</a>' in facts and '<section id="trials">' in html
    assert "trials" in _trace_links_land_in_sight(html)


async def test_a_trial_id_the_page_does_not_hold_is_not_linked() -> None:
    _, analysis, _ = await toy_report()
    finding = analysis.document.findings[0].model_copy(update={"trials": ["T1", "T999"]})
    page = build_report(
        analysis.model_copy(update={"document": analysis.document.model_copy(update={"findings": [finding]})})
    )
    html = report_html(page)
    facts = html[html.index('data-table="finding_facts"') :].split("</table>", 1)[0]
    assert '<a href="#trials">T1</a>' in facts and "T999" in facts and 'href="#T999"' not in facts
    _trace_links_land_in_sight(html)


async def test_every_link_on_both_pages_lands_in_sight() -> None:
    """The nav, a glossary term, a case's trial ids: each lands on the nearest place outside every closed fold the
    reader has not opened, on the aspect page and the campaign page alike."""
    host, analysis, page = await toy_report()
    html = report_html(page)
    nav = html[html.index("<nav>") : html.index("</nav>")]
    assert 'href="#cases"' in nav and "-body" not in nav
    assert 'href="#glossary"' in html and 'href="#g-' not in html
    _trace_links_land_in_sight(html)
    campaign = off_loop(campaign_report, host, analysis.campaign_id, analysis.scope_id)
    campaign_html = campaign_report_html(campaign)
    (fold_id,) = campaign.fold_ids
    nav = campaign_html[campaign_html.index("<nav>") : campaign_html.index("</nav>")]
    assert f'href="#aspect-{fold_id}"' in nav and "-body" not in nav
    # A case's trial ids, inside the aspect's fold, land on that fold's Trials section, which the reader then opens.
    assert f'href="#a-{fold_id}-trials"' in campaign_html
    _trace_links_land_in_sight(campaign_html)


async def test_a_link_keeps_its_exact_target_where_the_reader_sees_it() -> None:
    """The Trials table's own trial ids sit in the Trials fold with their records, so once the reader opens it to see
    the table, each id lands on its record: ``#T<n>`` on the aspect page, ``#a-<fold>-T<n>`` inside a campaign's
    fold. The nav names each section itself."""
    host, analysis, page = await toy_report()
    html = report_html(page)
    (trials,) = _tables(page.blocks, "trials")
    first = str(trials.rows[0]["trial"].value)
    table = html[html.index('data-table="trials"') :].split("</table>", 1)[0]
    assert f'<a href="#{first}">{first}</a>' in table and f'id="{first}"' in html
    nav = html[html.index("<nav>") : html.index("</nav>")]
    assert re.findall(r'href="#([^"]+)"', nav) == [
        "memo",
        "contrasts",
        "cases",
        "trials",
        "not_tested",
        "methods",
        "glossary",
    ]
    campaign = off_loop(campaign_report, host, analysis.campaign_id, analysis.scope_id)
    campaign_html = campaign_report_html(campaign)
    (fold_id,) = campaign.fold_ids
    folded = campaign_html[campaign_html.index(f'id="a-{fold_id}-trials"') :]
    table = folded[folded.index('data-table="trials"') :].split("</table>", 1)[0]
    assert f'<a href="#a-{fold_id}-{first}">{first}</a>' in table and f'id="a-{fold_id}-{first}"' in campaign_html


async def test_a_figure_never_lands_on_another_number() -> None:
    """The candidate never answers the last two cases, so the contrast on cost reads production over the ten cases
    both ran: its paired mean, not production's own figure over twelve. The finding's figure lands where production's
    own figure is printed, on its cost per reply, and the contrast's paired mean carries no id."""

    async def production(case: Mapping[str, Any]) -> Answer:
        index = int(case["text"].rsplit(" ", 1)[1])
        return Answer(case["label"], model="deepseek", input_tokens=100, output_tokens=10, cost_usd=0.001 * index)

    async def partial(case: Mapping[str, Any]) -> Answer:
        if int(case["text"].rsplit(" ", 1)[1]) >= 10:
            raise RuntimeError("the candidate is down")
        return Answer("skipped", model="deepseek", input_tokens=100, output_tokens=10, cost_usd=0.0033)

    comparison = await Eval(_CASES, {PRODUCTION: production, CANDIDATE: partial}, ["label"], k=1, name="queue").run(
        store=":memory:"
    )
    page = build_report(_citing(await _generated(comparison, comparison.report.bundle), "production_replicating_cost"))
    (contrasts,) = _tables(page.blocks, "comparisons")
    (cost,) = [row for row in contrasts.rows if row["reading"].text.startswith("Production cost")]
    assert cost["control"].anchor is None, "a paired mean over ten cases is not production's figure over twelve"
    for figure in _figures(page):
        assert figure.ref is not None
        table, landed = _carrying(page, figure.ref)
        assert landed.text.startswith(figure.text), f"{figure.text} lands on {landed.text} in {table.name}"
    _trace_links_land_in_sight(report_html(page))


async def test_two_contrasts_sharing_a_control_anchor_its_figure_on_the_first() -> None:
    comparison = await Eval(
        _CASES,
        {PRODUCTION: answering(30, 0.003092), CANDIDATE: answering(6, 0.003301), "third": answering(20, 0.0031)},
        ["label"],
        k=3,
        name="queue",
    ).run(store=":memory:")
    page = build_report(_citing(await _generated(comparison, comparison.report.bundle), "accuracy"))
    (contrasts,) = _tables(page.blocks, "comparisons")
    first, second = [row for row in contrasts.rows if row["reading"].text == "Accuracy"]
    assert first["control"].anchor is not None and second["control"].anchor is None
    assert first["control"].anchor in [figure.ref for figure in _figures(page)]
    _trace_links_land_in_sight(report_html(page))


async def test_a_figure_lands_on_its_guardrail() -> None:
    def correct(case: Mapping[str, Any], answer: str) -> bool:
        return answer.startswith("right")

    def no_leak(case: Mapping[str, Any], answer: str) -> bool:
        return "SECRET" not in answer

    def answers(misses: int, leaks: int) -> Any:
        async def run(case: Mapping[str, Any]) -> str:
            return ("wrong" if case["n"] < misses else "right") + (" SECRET" if case["n"] < leaks else "")

        return run

    comparison = await Eval(
        [{"n": index} for index in range(50)],
        {"current": answers(10, 0), "leaky": answers(0, 20)},
        [correct, grader(no_leak, guard=0.1)],
        k=1,
    ).run(store=":memory:", scope="trace-guardrails")
    page = build_report(_citing(await _generated(comparison, comparison.report.bundle), "no_leak"))
    (guardrails,) = _tables(page.blocks, "guardrails")
    (row,) = guardrails.rows
    assert [figure.ref for figure in _figures(page)] == [row["control_mean"].anchor, row["arm_mean"].anchor]
    assert None not in [row["control_mean"].anchor, row["arm_mean"].anchor]
    _trace_links_land_in_sight(report_html(page))


async def test_a_figure_on_a_one_arm_page_lands_on_its_reading() -> None:
    comparison = await Eval(_CASES, {PRODUCTION: answering(30, 0.003092)}, ["label"], k=3, name="queue").run(
        store=":memory:"
    )
    page = build_report(_citing(await _generated(comparison, comparison.report.bundle), "accuracy"))
    (readings,) = _tables(page.blocks, "readings")
    (accuracy,) = [row for row in readings.rows if row["reading"].text == "Accuracy"]
    (figure,) = _figures(page)
    assert figure.ref is not None and figure.ref == accuracy["value"].anchor
    _trace_links_land_in_sight(report_html(page))


async def test_a_figure_printed_only_in_the_closed_bars_lands_on_their_fold() -> None:
    _, analysis, _ = await toy_report()
    page = build_report(_citing(analysis, "field_accuracy"))
    (bars,) = _tables(page.blocks, "bars")
    assert bars.details and bars.anchor is not None
    assert [figure.ref for figure in _figures(page)] == [bars.anchor, bars.anchor]
    html = report_html(page)
    assert f'<details id="{bars.anchor}" data-table="bars">' in html
    assert _trace_links_land_in_sight(html) == [bars.anchor, bars.anchor]


async def test_a_cost_figure_lands_on_the_scoreboard_where_no_contrast_prints_it() -> None:
    _, analysis, _ = await toy_report()
    page = build_report(_citing(analysis, "production_replicating_cost"))
    (scoreboard,) = _tables(page.blocks, "scoreboard")
    anchors = {row["cost"].anchor for row in scoreboard.rows}
    figures = _figures(page)
    assert {figure.ref for figure in figures} == anchors and None not in anchors
    for figure in figures:
        assert figure.ref is not None and _carrying(page, figure.ref)[1].text == figure.text
    assert _untraced(page.blocks) == []


# =============================================================================
# A campaign's page
# =============================================================================


async def test_the_campaign_page_states_each_figure_and_links_the_aspect_s_fold() -> None:
    comparison, analysis = await _analysed()
    page = off_loop(analysis_report, comparison.host.storage, analysis.id, analysis.scope_id)
    (fold_id,) = page.fold_ids
    (tested,) = _tables(page.blocks, "trace", finding=0)
    assert [column.key for column in tested.columns] == ["arm", "reading", "value"], "one aspect needs no column"
    assert [(row["reading"].text, row["value"].text) for row in tested.rows] == [
        ("Accuracy", "83.3%"),
        ("Accuracy", "16.7%"),
    ]
    assert [row["value"].ref for row in tested.rows] == [f"aspect-{fold_id}"] * 2
    (trial,) = _tables(page.blocks, "trace", finding=2)
    (row,) = trial.rows
    assert (row["reading"].text, row["value"].text, row["value"].ref) == ("trial", "T40", f"aspect-{fold_id}")
    assert _untraced(page.blocks) == [1]
    html = campaign_report_html(page)
    assert set(_trace_links_land_in_sight(html)) == {f"aspect-{fold_id}"}


async def test_a_campaign_of_two_aspects_traces_each_figure_and_trial_to_its_own_aspect() -> None:
    first = await Eval(
        _CASES, {PRODUCTION: answering(30, 0.003092), CANDIDATE: answering(6, 0.003301)}, ["label"], k=3, name="queue"
    ).run(store=":memory:")
    second = await Eval(
        _CASES, {PRODUCTION: answering(20, 0.003092), CANDIDATE: answering(33, 0.003301)}, ["label"], k=3, name="queue"
    ).run(store=":memory:")
    billing = first.report.bundle
    tone = second.report.bundle.model_copy(
        update={"aspect": "tone", "campaign_id": billing.campaign_id, "scope_id": billing.scope_id}
    )
    bundle = campaign_bundle(billing, tone)
    aliases = campaign_cell_aliases([(aspect.aspect, aspect.cell_measures) for aspect in (billing, tone)])
    on_tone = {alias: alias for alias, where in aliases.items() if where.aspect == "tone"}
    # The memo names the second aspect's first trial by its campaign id, after every one of the first aspect's.
    named = f"T{len(billing.trials) + 1}"
    finding = _finding("The candidate answers more requests right", "accuracy", on_tone, (named,))
    analysis = await _generated(first, bundle, [finding])
    page = off_loop(analysis_report, first.host.storage, analysis.id, analysis.scope_id)
    assert page.fold_ids[1] == "tone"
    (trace,) = _tables(page.blocks, "trace", finding=0)
    assert trace.columns[0].key == "aspect", "two aspects name each row's"
    readings = [row for row in trace.rows if row["reading"].text == "Accuracy"]
    assert [row["aspect"].text for row in readings] == ["tone", "tone"]
    assert {row["value"].ref for row in readings} == {"aspect-tone"}
    tone_fold = page.aspects[1].report
    for row in readings:
        printed = [
            cell.text
            for block in tone_fold.blocks
            if isinstance(block, TableBlock)
            for line in block.rows
            for cell in line.values()
            if cell.anchor is not None
        ]
        assert any(text.startswith(row["value"].text) for text in printed)
    (trial,) = [row for row in trace.rows if row["reading"].text == "trial"]
    assert (trial["aspect"].text, trial["value"].text, trial["value"].ref) == ("tone", "T1", "aspect-tone")
    assert tone.trials[0].trial_id == "T1"
    _trace_links_land_in_sight(campaign_report_html(page))


async def test_a_campaign_page_finding_citing_nothing_its_folds_print_says_so() -> None:
    host, analysis, _ = await toy_report()
    page = off_loop(campaign_report, host, analysis.campaign_id, analysis.scope_id)
    (trace,) = _tables(page.blocks, "trace", finding=0)
    assert trace.rows and all(row["value"].ref is None for row in trace.rows)
    assert _untraced(page.blocks) == [0]
    assert escape(UNTRACED) in campaign_report_html(page)


# =============================================================================
# The model: a link lands, or there is none
# =============================================================================


def _with_cells(report: Report, *cells: TableCell) -> Report:
    table = TableBlock(
        section="contrasts",
        name="extra",
        title="Extra",
        columns=[TableColumn(key=f"c{index}", header=f"C{index}") for index in range(len(cells))],
        rows=[{f"c{index}": cell for index, cell in enumerate(cells)}],
        order="as given",
        total_rows=1,
    )
    return Report.model_validate({**dict(report), "blocks": [*report.blocks, table]})


async def test_a_report_refuses_a_link_to_a_place_it_does_not_hold() -> None:
    _, _, page = await toy_report()
    with pytest.raises(ValueError, match="which nothing on the page carries"):
        _with_cells(page, TableCell(value=1, ref="r-000000000000"))
    assert _with_cells(page, TableCell(value=1, ref="r-1"), TableCell(value=2, anchor="r-1"))


async def test_a_report_refuses_one_id_on_two_cells() -> None:
    _, _, page = await toy_report()
    with pytest.raises(ValueError, match="more than once"):
        _with_cells(page, TableCell(value=1, anchor="r-1"), TableCell(value=2, anchor="r-1"))
    # A block's id counts too: the first finding's headline is F1.
    with pytest.raises(ValueError, match="more than once"):
        _with_cells(page, TableCell(value=1, anchor="F1"))


async def test_a_campaign_page_refuses_a_link_anywhere_but_a_fold() -> None:
    host, analysis, _ = await toy_report()
    page = off_loop(campaign_report, host, analysis.campaign_id, analysis.scope_id)
    (trace,) = _tables(page.blocks, "trace", finding=0)
    place = reading_anchor("v:r", "total_ms", "measure")
    stray = trace.model_copy(update={"rows": [{**trace.rows[0], "value": TableCell(value=1, ref=place)}]})
    blocks = [stray if block is trace else block for block in page.blocks]
    with pytest.raises(ValueError, match="which is no aspect's fold"):
        CampaignReport.model_validate({**dict(page), "blocks": blocks})


def test_fold_ids_never_repeat_whatever_the_labels() -> None:
    assert fold_ids(["Billing", "billing-3", "billing"]) == ["billing", "billing-3", "billing-4"]
    assert fold_ids(["a b", "a-b", "A B"]) == ["a-b", "a-b-2", "a-b-3"]
    labels = ["x", "x", "x-2", "x"]
    assert len(set(fold_ids(labels))) == len(labels)


def test_every_trace_id_is_one_a_link_can_name() -> None:
    assert re.fullmatch(r"r-[0-9a-f]{12}", reading_anchor("a:b", "measure id with spaces", "judged"))
