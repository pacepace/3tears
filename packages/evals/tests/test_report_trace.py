"""A finding's trace: from what the memo says to the readings and trials behind it, drawn by code, on both pages.

Each figure in a finding's readings table links to the place on the page where code states that reading at that cell
in full (a contrast's row, a guardrail's, a bar's, a one-arm page's readings); each trial id it names links to the
trial's record. On a campaign's page each finding's ``Evidence`` fold lists what it cites, linked into the aspect's
fold that states it. A finding whose trace reaches nothing says so in words rather than linking nowhere, and the
page model refuses a link to a place it does not hold.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from threetears.evals import Eval, Result
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
from threetears.evals.analysis.references import cell_aliases
from threetears.evals.analysis.report.words import UNTRACED
from threetears.evals.kernel.campaign import EvalAnalysis
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


def _finding(title: str, measure: str, cells: dict[str, str], trials: tuple[str, ...] = ()) -> dict[str, Any]:
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
    analysis, _ = await generate_analysis(
        campaign_bundle(bundle),
        prompt=PROMPT,
        model=MODEL,
        client=FixturedClient(json.dumps(authored_memo(bundle, findings, means="Keep production."))),
        prompt_id=PROMPT_ID,
        bundle_assembled_at="2026-10-14T19:23:41+00:00",
        profile=comparison.host.profile,
    )
    off_loop(comparison.host.storage.save_analysis, analysis)
    return comparison, analysis


def _tables(blocks: list[Any], name: str, finding: int | None = None) -> list[TableBlock]:
    return [
        block
        for block in blocks
        if isinstance(block, TableBlock) and block.name == name and (finding is None or block.finding == finding)
    ]


def _untraced(blocks: list[Any]) -> list[int | None]:
    """The findings whose trace says it reaches nothing, by position."""
    return [block.finding for block in blocks if isinstance(block, DisclosureBlock) and block.text == UNTRACED]


def _lands(page: str) -> None:
    """Every in-page link lands on exactly one id the page carries."""
    ids = re.findall(r'\bid="([^"]+)"', page)
    assert len(ids) == len(set(ids)), "an id is carried twice"
    assert set(re.findall(r'href="#([^"]+)"', page)) <= set(ids), "a link lands nowhere"


# =============================================================================
# An aspect's page
# =============================================================================


async def test_a_finding_s_figures_link_to_the_contrast_that_states_each_reading() -> None:
    _, analysis = await _analysed()
    page = build_report(analysis, campaign_name="queue")
    (evidence,) = _tables(page.blocks, "evidence", finding=0)
    linked = [cell.ref for row in evidence.rows for key, cell in row.items() if key.startswith("reading_")]
    assert len(linked) == 2 and None not in linked
    # Each figure lands on its arm's own figure in the contrast on that reading: the control's and the candidate's.
    (contrasts,) = _tables(page.blocks, "comparisons")
    (accuracy,) = [row for row in contrasts.rows if row["reading"].text == "Accuracy"]
    assert linked == [accuracy["control"].anchor, accuracy["arm"].anchor]
    assert 0 not in _untraced(page.blocks)
    html = report_html(page)
    for place in linked:
        assert f'<a href="#{place}">' in html and f'<td id="{place}"' in html
    _lands(html)


async def test_a_finding_whose_readings_no_table_states_links_nothing_and_says_so_in_its_fold() -> None:
    _, analysis = await _analysed()
    page = build_report(analysis, campaign_name="queue")
    (evidence,) = _tables(page.blocks, "evidence", finding=1)
    assert all(cell.ref is None for row in evidence.rows for cell in row.values())
    (said,) = [block for block in page.blocks if isinstance(block, DisclosureBlock) and block.text == UNTRACED]
    assert said.finding == 1 and said.fold == "Evidence" and said.section == "memo"
    html = report_html(page)
    fold = html[html.index('id="F2"') :].split("</details>", 1)[0]
    assert UNTRACED in fold and 'href="#r-' not in fold


async def test_a_finding_s_trial_ids_link_to_their_records_and_trace_it() -> None:
    _, analysis, page = await toy_report()
    # The toy campaign has no control, so no table states the turn time its one finding cites: the trace says so.
    assert _untraced(page.blocks) == [0]
    finding = analysis.document.findings[0].model_copy(update={"trials": ["T1"]})
    traced = build_report(
        analysis.model_copy(update={"document": analysis.document.model_copy(update={"findings": [finding]})})
    )
    assert _untraced(traced.blocks) == []
    html = report_html(traced)
    facts = html[html.index('data-table="finding_facts"') :].split("</table>", 1)[0]
    assert '<a href="#T1">T1</a>' in facts
    _lands(html)


# =============================================================================
# A campaign's page
# =============================================================================


async def test_the_campaign_page_traces_each_finding_into_its_aspect_s_fold() -> None:
    comparison, analysis = await _analysed()
    page = off_loop(analysis_report, comparison.host.storage, analysis.id, analysis.scope_id)
    (fold_id,) = page.fold_ids
    (tested,) = _tables(page.blocks, "trace", finding=0)
    places = [row["cites"].ref for row in tested.rows]
    assert [row["cites"].text for row in tested.rows] == ["Accuracy", "Accuracy"]
    # The places the aspect's own page states the readings, inside the fold.
    (fold,) = page.aspects
    (contrasts,) = _tables(fold.report.blocks, "comparisons")
    (accuracy,) = [row for row in contrasts.rows if row["reading"].text == "Accuracy"]
    assert places == [f"a-{fold_id}-{accuracy['control'].anchor}", f"a-{fold_id}-{accuracy['arm'].anchor}"]
    (trial,) = _tables(page.blocks, "trace", finding=2)
    (row,) = trial.rows
    own = dict(campaign_bundle(fold.report.bundle).trial_ids())["T40"][1]
    assert row["cites"].text == "trial T40" and row["cites"].ref == f"a-{fold_id}-{own}"
    assert _untraced(page.blocks) == [1]
    html = campaign_report_html(page)
    for place in [*places, f"a-{fold_id}-{own}"]:
        assert f'<a href="#{place}">' in html and f'id="{place}"' in html
    _lands(html)


async def test_a_campaign_page_finding_citing_nothing_its_folds_state_says_so() -> None:
    host, analysis, _ = await toy_report()
    page = off_loop(campaign_report, host, analysis.campaign_id, analysis.scope_id)
    (trace,) = _tables(page.blocks, "trace", finding=0)
    assert trace.rows and all(row["cites"].ref is None for row in trace.rows)
    assert _untraced(page.blocks) == [0]
    assert UNTRACED in campaign_report_html(page)


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
    with pytest.raises(ValueError, match="which no cell on the page carries"):
        _with_cells(page, TableCell(value=1, ref="r-000000000000"))
    assert _with_cells(page, TableCell(value=1, ref="r-1"), TableCell(value=2, anchor="r-1"))


async def test_a_report_refuses_one_id_on_two_cells() -> None:
    _, _, page = await toy_report()
    with pytest.raises(ValueError, match="more than once"):
        _with_cells(page, TableCell(value=1, anchor="r-1"), TableCell(value=2, anchor="r-1"))


async def test_a_campaign_page_refuses_a_link_no_fold_holds() -> None:
    host, analysis, _ = await toy_report()
    page = off_loop(campaign_report, host, analysis.campaign_id, analysis.scope_id)
    (trace,) = _tables(page.blocks, "trace", finding=0)
    stray = trace.model_copy(
        update={"rows": [{**trace.rows[0], "cites": TableCell(value="Turn time", ref="a-nowhere-r-1")}]}
    )
    blocks = [stray if block is trace else block for block in page.blocks]
    with pytest.raises(ValueError, match="which no aspect's fold holds"):
        CampaignReport.model_validate({**dict(page), "blocks": blocks})
