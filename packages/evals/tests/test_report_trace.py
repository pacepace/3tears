"""Where a reading's figure is anchored on code's pages, and that every link on them lands where a reader sees it.

Every cell printing a reading's own figure at a cell carries that reading's id (``reading_anchor``), and one cell keeps
it: a cell shown open before one in a closed table, a contrast before any other table, else the first on the page, and
a table shown closed (the bars) carries the id of its own fold where a cell in it keeps one. A contrast's paired mean
over fewer cases is not the cell's figure, and carries no id. Every link on an aspect's page and on a campaign's is
fragment-only and lands on an id the page carries, outside any closed ``<details>`` the reader has not opened, and the
page models refuse a link to a place they do not hold and one id carried twice.

The stored analysis read here is the toy campaign's retired memo: its aspect's page is code's over the bundle frozen
with it (``report_support.toy_report``), and its campaign's page is the one ``analysis_report`` lays out by its id.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from html.parser import HTMLParser
from typing import Any

import pytest

from threetears.evals import Answer, Eval, Result, grader
from threetears.evals.analysis import (
    AnalysisContextBundle,
    CampaignReport,
    Report,
    TableBlock,
    TableCell,
    TableColumn,
    analysis_report,
    build_code_only_report,
    campaign_report_html,
    report_html,
)
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.report.model import fold_ids, reading_anchor
from threetears.evals.kernel.campaign import EvalAnalysis
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.surface import DecisionSurface
from packages.evals.tests.g2_support import CANDIDATE, PRODUCTION, answering
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.report_support import toy_report
from packages.evals.tests.page_contract_support import without_kit_script

#: Production passes 30 of its 36 trials, the candidate 6: a contrast on accuracy the page tests and states.
_CASES = [{"text": f"queue request {index}", "label": "queued"} for index in range(12)]


def _page(bundle: AnalysisContextBundle) -> Report:
    """Code's page over ``bundle``."""
    return build_code_only_report(bundle, assembled_at="2026-10-14T19:23:41+00:00", campaign_name="queue")


def _anchors(surface: DecisionSurface, measure: str, variants: Sequence[str] | None = None) -> list[str]:
    """The id of ``measure``'s reading at each cell of ``surface`` (or at each of ``variants``' cells, in order)."""
    cells = {cell.variant_key: cell for cell in surface.cells}
    keys = list(variants) if variants is not None else list(cells)
    return [
        reading_anchor(cell_ref(cells[key].variant_key, cells[key].apparatus_class_id), measure, "measure")
        for key in keys
    ]


def _arms(comparison: Result) -> list[str]:
    """Production's variant key, then the candidate's."""
    return [comparison.arm_variants[name] for name in (PRODUCTION, CANDIDATE)]


def _tables(blocks: Sequence[Any], name: str) -> list[TableBlock]:
    return [block for block in blocks if isinstance(block, TableBlock) and block.name == name]


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


def _campaign_page(host: EvalHost, analysis: EvalAnalysis) -> CampaignReport:
    """The campaign's page ``analysis_report`` lays out by the analysis's id, over the bundles frozen with it."""
    return analysis_report(host.storage, analysis.id, analysis.scope_id)


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


def _links_land_in_sight(page: str) -> None:
    """Check every link on the page is fragment-only and lands where every browser shows it: on an id the page
    carries, held by no closed ``<details>`` the reader has not already opened to see the link (a fold's own
    ``<details>``, or the section around it, is outside it)."""
    parsed = _Landings()
    parsed.feed(page)
    parsed.close()
    assert not parsed.stack, "an element is left open"
    for href, shut, _ in parsed.links:
        assert href.startswith("#"), f"{href} is not a link to a place on the page"
        assert href[1:] in parsed.held, f"{href} lands nowhere"
        # The one link that lands inside a closed fold: a goal check's name, on its own glossary row, the one place
        # its expression stands (coordinator ruling, 2026-11-07).
        if href.startswith("#g-check-"):
            continue
        assert parsed.held[href[1:]] <= shut, f"{href} lands inside a closed fold"
    assert "<base" not in without_kit_script(page)


# =============================================================================
# An aspect's page
# =============================================================================


async def test_a_contrast_s_figures_carry_their_readings_ids_on_the_page() -> None:
    comparison = await Eval(
        _CASES, {PRODUCTION: answering(30, 0.003092), CANDIDATE: answering(6, 0.003301)}, ["label"], k=3, name="queue"
    ).run(store=":memory:")
    bundle = comparison.report.bundle
    page = _page(bundle)
    (contrasts,) = _tables(page.blocks, "comparisons")
    (accuracy,) = [row for row in contrasts.rows if row["reading"].text == "Accuracy"]
    # Each arm's own figure in the contrast on that reading carries its reading's id: the control's and the
    # candidate's.
    expected = _anchors(bundle_decision_surface(bundle), "accuracy", _arms(comparison))
    assert [accuracy["control"].anchor, accuracy["arm"].anchor] == expected
    html = report_html(page)
    for place in expected:
        assert f'<td id="{place}"' in html
    _links_land_in_sight(html)


async def test_every_link_on_both_pages_lands_in_sight() -> None:
    """The nav, a glossary term, a case's trial ids: each lands on the nearest place outside every closed fold the
    reader has not opened, on the aspect page and the campaign page alike."""
    host, analysis, page = await toy_report()
    html = report_html(page)
    nav = html[html.index("<nav>") : html.index("</nav>")]
    assert 'href="#cases"' in nav and "-body" not in nav
    # A term of art lands on the glossary's heading; only a goal check's name lands on its row.
    assert 'href="#glossary"' in html and 'href="#g-' not in html.replace('href="#g-check-', "")
    _links_land_in_sight(html)
    campaign = off_loop(_campaign_page, host, analysis)
    campaign_html = campaign_report_html(campaign)
    (fold_id,) = campaign.fold_ids
    nav = campaign_html[campaign_html.index("<nav>") : campaign_html.index("</nav>")]
    assert f'href="#aspect-{fold_id}"' in nav and "-body" not in nav
    # A case's trial ids, inside the aspect's fold, land on that fold's Trials section, which the reader then opens.
    assert f'href="#a-{fold_id}--trials"' in campaign_html
    _links_land_in_sight(campaign_html)


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
        "contrasts",
        "cases",
        "trials",
        "not_tested",
        "methods",
        "glossary",
    ]
    campaign = off_loop(_campaign_page, host, analysis)
    campaign_html = campaign_report_html(campaign)
    (fold_id,) = campaign.fold_ids
    folded = campaign_html[campaign_html.index(f'id="a-{fold_id}--trials"') :]
    table = folded[folded.index('data-table="trials"') :].split("</table>", 1)[0]
    assert f'<a href="#a-{fold_id}--{first}">{first}</a>' in table and f'id="a-{fold_id}--{first}"' in campaign_html


async def test_a_paired_mean_never_carries_the_cell_s_id() -> None:
    """The candidate never answers the last two cases, so the contrast on cost reads production over the ten cases
    both ran: its paired mean, not production's own figure over twelve. The contrast's paired mean carries no id;
    production's own figure keeps it, on its cost per reply."""

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
    bundle = comparison.report.bundle
    page = _page(bundle)
    (contrasts,) = _tables(page.blocks, "comparisons")
    (cost,) = [row for row in contrasts.rows if row["reading"].text.startswith("Production cost")]
    assert cost["control"].anchor is None, "a paired mean over ten cases is not production's figure over twelve"
    (own,) = _anchors(bundle_decision_surface(bundle), "production_replicating_cost", _arms(comparison)[:1])
    table, _ = _carrying(page, own)
    assert table.name == "scoreboard", "production's own figure over twelve keeps its id, on its cost per reply"
    _links_land_in_sight(report_html(page))


async def test_two_contrasts_sharing_a_control_anchor_its_figure_on_the_first() -> None:
    comparison = await Eval(
        _CASES,
        {PRODUCTION: answering(30, 0.003092), CANDIDATE: answering(6, 0.003301), "third": answering(20, 0.0031)},
        ["label"],
        k=3,
        name="queue",
    ).run(store=":memory:")
    bundle = comparison.report.bundle
    page = _page(bundle)
    (contrasts,) = _tables(page.blocks, "comparisons")
    first, second = [row for row in contrasts.rows if row["reading"].text == "Accuracy"]
    assert first["control"].anchor is not None and second["control"].anchor is None
    assert [first["control"].anchor] == _anchors(bundle_decision_surface(bundle), "accuracy", _arms(comparison)[:1])
    _links_land_in_sight(report_html(page))


async def test_a_guardrail_s_figures_carry_their_readings_ids() -> None:
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
        name="current vs leaky",
    ).run(store=":memory:", scope="trace-guardrails")
    bundle = comparison.report.bundle
    page = _page(bundle)
    (guardrails,) = _tables(page.blocks, "guardrails")
    (row,) = guardrails.rows
    variants = [comparison.arm_variants[name] for name in ("current", "leaky")]
    assert [row["control_mean"].anchor, row["arm_mean"].anchor] == _anchors(
        bundle_decision_surface(bundle), "no_leak", variants
    )
    _links_land_in_sight(report_html(page))


async def test_a_one_arm_page_s_reading_carries_its_id() -> None:
    comparison = await Eval(_CASES, {PRODUCTION: answering(30, 0.003092)}, ["label"], k=3, name="queue").run(
        store=":memory:"
    )
    bundle = comparison.report.bundle
    page = _page(bundle)
    (readings,) = _tables(page.blocks, "readings")
    (accuracy,) = [row for row in readings.rows if row["reading"].text == "Accuracy"]
    assert [accuracy["value"].anchor] == _anchors(bundle_decision_surface(bundle), "accuracy")
    _links_land_in_sight(report_html(page))


async def test_a_figure_printed_only_in_the_closed_bars_anchors_their_fold() -> None:
    _, analysis, page = await toy_report()
    (bars,) = _tables(page.blocks, "bars")
    assert bars.details and bars.anchor is not None
    for anchor in _anchors(analysis.aspects[0].decision_surface, "field_accuracy"):
        assert _carrying(page, anchor)[0] is bars, "no open table prints the field accuracy, so the bars keep its id"
    html = report_html(page)
    assert f'<details id="{bars.anchor}" data-table="bars">' in html
    _links_land_in_sight(html)


async def test_a_cost_figure_is_anchored_on_the_scoreboard_where_no_contrast_prints_it() -> None:
    _, analysis, page = await toy_report()
    (scoreboard,) = _tables(page.blocks, "scoreboard")
    anchors = {row["cost"].anchor for row in scoreboard.rows}
    assert anchors == set(_anchors(analysis.aspects[0].decision_surface, "production_replicating_cost"))
    assert None not in anchors


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
        _with_cells(page, TableCell(value=1, value_text="1", ref="r-000000000000"))
    assert _with_cells(
        page, TableCell(value=1, value_text="1", ref="r-1"), TableCell(value=2, value_text="2", anchor="r-1")
    )


async def test_a_report_refuses_one_id_on_two_cells() -> None:
    _, _, page = await toy_report()
    with pytest.raises(ValueError, match="more than once"):
        _with_cells(
            page, TableCell(value=1, value_text="1", anchor="r-1"), TableCell(value=2, value_text="2", anchor="r-1")
        )
    # A block's id counts too: the closed bars table carries its fold's.
    (bars,) = _tables(page.blocks, "bars")
    assert bars.anchor is not None
    with pytest.raises(ValueError, match="more than once"):
        _with_cells(page, TableCell(value=1, value_text="1", anchor=bars.anchor))


async def test_a_campaign_page_refuses_a_link_anywhere_but_a_fold() -> None:
    host, analysis, _ = await toy_report()
    page = off_loop(_campaign_page, host, analysis)
    place = reading_anchor("v:r", "total_ms", "measure")
    stray = TableBlock(
        section="scorecard",
        name="stray",
        title="Stray",
        columns=[TableColumn(key="value", header="Value")],
        rows=[{"value": TableCell(value=1, value_text="1", ref=place)}],
        order="as given",
        total_rows=1,
    )
    blocks = [*page.blocks, stray]
    with pytest.raises(ValueError, match="which is no aspect's fold"):
        CampaignReport.model_validate({**dict(page), "blocks": blocks})


def test_fold_ids_never_repeat_whatever_the_labels() -> None:
    assert fold_ids(["Billing", "billing-3", "billing"]) == ["billing", "billing-3", "billing-4"]
    assert fold_ids(["a b", "a-b", "A B"]) == ["a-b", "a-b-2", "a-b-3"]
    labels = ["x", "x", "x-2", "x"]
    assert len(set(fold_ids(labels))) == len(labels)


def test_every_trace_id_is_one_a_link_can_name() -> None:
    assert re.fullmatch(r"r-[0-9a-f]{12}", reading_anchor("a:b", "measure id with spaces", "judged"))
