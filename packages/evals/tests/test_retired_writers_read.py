"""An analysis a retired report writer stored reads, through every door, as code's page with one line saying so.

The report planner no longer writes analyses (``analysis_generate`` runs the report writer), and its planned page is
retired with it. A host still holds the planned analyses it stored: the one the planner stored over the toy corpus
campaign (``report_support.stored_plan_analysis``), its plan written beside it as the host's document store holds it
(``report_support.stored_plan_document``). Contracts held here (``threetears.evals.analysis.service``,
``threetears.evals.ops``):

- **every door reads it**, never refusing and never blank: ``report_read`` in its four forms, ``campaign_report``,
  ``analysis_report`` by its id and ``analyses_list``. Each is code's page (``basis="code_only"``) standing in for the
  analysis (``unread_analysis_id``), with the retired writer's line (``retired_generation``, naming the report planner)
  on its first screen and as ``needs_regeneration``; the listing counts no findings. ``analysis_report`` lays the page
  over the bundles frozen with the analysis, every aspect's fold kept;
- **the plan is never read**: the reads are the same whether the plan stored beside it is the planner's own, a host's
  plan the planned page could not draw, a plan no model reads, or no plan at all;
- **the plan is still swept**: ``drop_derived`` deletes it, and deleting its analysis deletes it (its insights stay);
- an aspect's page of a planned analysis is refused, as it holds no memo;
- an analysis the memo writer stored still lists, reads and renders as written;
- the actions round-trip a generated analysis as a report document, with no plan member.
"""

from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path
from typing import Any, Final

import pytest

from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture, settled
from packages.evals.tests.report_support import stored_memo_analysis, stored_plan_analysis, stored_plan_document
from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.analysis import CampaignReport, build_report
from threetears.evals.analysis.report.build import analysis_campaign_bundle
from threetears.evals.analysis.report.campaign.memo_blocks import retired_generation
from threetears.evals.analysis.report.serialize import serialize_report
from threetears.evals.analysis.service import analysis_report, campaign_report
from threetears.evals.kernel.campaign import EvalAnalysis
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.ops import analyses_list, analysis_delete, report_read
from threetears.evals.storage import InMemoryDocumentStore

#: A host's planned analysis of Kairo round 2, its plan as the host stored it: one chart of it the planned page
#: refused to draw (a forest mixing a score change with changes in points).
_HOST_PLAN: Final = Path(__file__).parent / "fixtures" / "retired_writers" / "01a177f6.plan.json"

#: The line every door prints for a planned analysis.
_LINE: Final = (
    "written by a retired report writer (the report planner); regenerate with analysis_generate to read a report "
    "over it"
)

#: What the store holds beside the analysis: the planner's own plan, the host's, a plan no model reads, or none.
_STORED_PLANS = ("planners", "hosts", "unreadable", "none")


def _plan_body(stored: str) -> dict[str, Any] | None:
    if stored == "hosts":
        loaded: dict[str, Any] = json.loads(_HOST_PLAN.read_text(encoding="utf-8"))
        return loaded
    return {"schema": "report-plan/0", "headline": {"not": "a plan"}} if stored == "unreadable" else None


async def _planned(stored: str = "planners") -> tuple[OpsFixture, InMemoryDocumentStore, EvalAnalysis]:
    """The toy campaign's host holding the planner's analysis, and beside it the plan ``stored`` names."""
    store = InMemoryDocumentStore()
    fixture = off_loop(ops_fixture, store=store)
    analysis, _ = await stored_plan_analysis(fixture.host.eval_host, fixture.campaign)
    if stored != "none":
        document = off_loop(stored_plan_document, store, fixture.campaign, plan=_plan_body(stored))
        assert document["id"] == analysis.report_plan_id
    return fixture, store, analysis


def _first_screen(body: str, form: str) -> str:
    """What a reader sees before the page's first section: the HTML page's header, the Markdown before its first
    section heading; the JSON form, which carries no blocks, says it in its ``needs_regeneration``."""
    if form == "html":
        return body[body.index("<header>") : body.index("</header>")]
    if form == "markdown":
        return body.split("\n## ", 1)[0]
    said: str = json.loads(body)["needs_regeneration"] or ""
    return said


def _said(analysis: EvalAnalysis, form: str) -> str:
    if form == "json":
        return _LINE
    said = f"This campaign's analysis {analysis.id} was {_LINE}."
    return escape(said) if form == "html" else said


def _assert_stands_in(page: CampaignReport, analysis: EvalAnalysis) -> None:
    assert (page.basis, page.analysis_id, page.unread_analysis_id) == ("code_only", None, analysis.id)
    assert page.needs_regeneration is not None and page.needs_regeneration.startswith(_LINE)
    assert page.plan_page is None and page.doc_page is None
    for form in ("markdown", "html", "json"):
        assert _said(analysis, form) in _first_screen(serialize_report(page, form), form), form


# -- every door ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("stored", _STORED_PLANS)
async def test_report_read_is_code_s_page_with_the_line_on_its_first_screen_in_every_form(stored: str) -> None:
    fixture, _, analysis = await _planned(stored)
    host = fixture.host.eval_host

    for form, static in (("markdown", False), ("html", False), ("html", True), ("json", False)):
        read = off_loop(report_read, host, fixture.campaign.id, TOYHOST_SCOPE, format=form, static=static)

        assert (read.basis, read.analysis_id, read.analysis_basis, read.doc) == ("code_only", analysis.id, "plan", None)
        assert read.needs_regeneration == f"{_LINE} (analysis {analysis.id})", form
        assert _said(analysis, form) in _first_screen(read.body, form), form
        assert "No recommendation was written" not in read.body, form
        if form == "json":
            assert json.loads(read.body)["needs_regeneration"] == read.needs_regeneration


@pytest.mark.parametrize("stored", _STORED_PLANS)
async def test_the_campaign_s_page_and_the_page_by_id_stand_in_for_it_with_the_line(stored: str) -> None:
    fixture, _, analysis = await _planned(stored)
    host = fixture.host.eval_host

    campaigns = off_loop(campaign_report, host, fixture.campaign.id, TOYHOST_SCOPE)
    by_id = off_loop(analysis_report, host.storage, analysis.id, TOYHOST_SCOPE)
    everything = off_loop(analysis_report, host.storage, analysis.id, TOYHOST_SCOPE, trials="all")

    _assert_stands_in(campaigns, analysis)
    _assert_stands_in(by_id, analysis)
    assert by_id.needs_regeneration == _LINE, "by its id, the line alone: the page names the analysis"
    # By its id, the page is laid over the bundles frozen with the analysis, every aspect's fold kept.
    assert [fold.aspect for fold in by_id.aspects] == [
        bundle.aspect for bundle in analysis_campaign_bundle(analysis).aspects
    ]
    assert serialize_report(everything, "html").count("<details") > serialize_report(by_id, "html").count("<details")


@pytest.mark.parametrize("stored", _STORED_PLANS)
async def test_the_listing_leads_its_line_with_the_line_and_counts_no_findings(stored: str) -> None:
    fixture, _, analysis = await _planned(stored)
    memo, _ = await stored_memo_analysis(fixture.host.eval_host, fixture.campaign)

    lines = {
        line.id: line
        for line in off_loop(analyses_list, fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE).analyses
    }

    planned = lines[analysis.id]
    assert (planned.basis, planned.report_plan_id, planned.findings) == ("plan", analysis.report_plan_id, 0)
    assert planned.needs_regeneration == _LINE
    assert lines[memo.id].needs_regeneration is None, "a retired writer's analysis flags no other line"


async def test_the_line_is_said_once_and_every_form_s_page_reads_the_same_whatever_plan_is_stored() -> None:
    bodies: dict[str, list[str]] = {}
    for stored in _STORED_PLANS:
        fixture, _, analysis = await _planned(stored)
        page = off_loop(analysis_report, fixture.host.eval_host.storage, analysis.id, TOYHOST_SCOPE)
        bodies[stored] = [_without_instants(serialize_report(page, form)) for form in ("markdown", "html", "json")]
        markdown, html, _ = bodies[stored]
        assert markdown.count(_LINE) == 1 and html.count(escape(_LINE)) == 1, stored
    assert len({tuple(forms) for forms in bodies.values()}) == 1, "no stored plan is read"


def _without_instants(body: str) -> str:
    return re.sub(r"\d{4}-\d{2}-\d{2}T[0-9:.+]+", "", body)


def test_the_line_is_worded_once_and_names_only_a_retired_writer() -> None:
    assert retired_generation("plan") == _LINE
    assert retired_generation("memo") is None and retired_generation("doc") is None, "a writer this version reads"


# -- the plan is still swept --------------------------------------------------------------------------------------


async def test_drop_derived_sweeps_the_plan() -> None:
    fixture, store, analysis = await _planned()
    assert analysis.report_plan_id is not None

    dropped = off_loop(fixture.host.eval_host.storage.drop_derived, [TOYHOST_SCOPE])

    assert dropped.counts["eval_report_plan"] == 1 and dropped.counts["eval_analysis"] == 1
    assert store.get(analysis.report_plan_id, TOYHOST_SCOPE) is None


async def test_deleting_the_analysis_deletes_its_plan_and_keeps_its_insights() -> None:
    fixture, store, analysis = await _planned()
    assert analysis.report_plan_id is not None

    off_loop(analysis_delete, fixture.host.eval_host, analysis.id, TOYHOST_SCOPE, confirm=analysis.id)

    assert store.get(analysis.report_plan_id, TOYHOST_SCOPE) is None
    insights = off_loop(fixture.host.eval_host.storage.query_insights, TOYHOST_SCOPE)
    assert insights and {insight.source_analysis_id for insight in insights} == {analysis.id}


# -- the other writers --------------------------------------------------------------------------------------------


async def test_an_aspect_page_of_a_planned_analysis_is_refused() -> None:
    _, _, analysis = await _planned()

    with pytest.raises(RefusedInput, match="written by the report planner"):
        build_report(analysis)


async def test_a_memo_analysis_lists_reads_and_renders_as_it_did() -> None:
    fixture = off_loop(ops_fixture)
    memo, _ = await stored_memo_analysis(fixture.host.eval_host, fixture.campaign)
    assert memo.document is not None

    (line,) = off_loop(analyses_list, fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE).analyses
    assert (line.id, line.basis, line.report_plan_id) == (memo.id, "memo", None)
    assert (line.findings, line.recommendations, line.needs_regeneration) == (len(memo.document.findings), 1, None)
    read = off_loop(report_read, fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE, format="markdown")
    assert (read.basis, read.analysis_id, read.analysis_basis, read.doc) == ("analysis", memo.id, "memo", None)
    assert read.needs_regeneration is None
    assert "**Recommended:**" in read.body, "the memo's recommendation, as written"
    assert build_report(memo).basis == "analysis"


async def test_the_actions_round_trip_a_generated_analysis_as_a_report_document() -> None:
    """Generate, poll, list and read through the mounted tool: ``analysis_generate`` reaches the report writer, so the
    listing and the read say a report document wrote it."""
    fixture = off_loop(ops_fixture)
    evals = eval_catalogue().mount_all(standard_tools())[0]

    started = await evals.call(
        {"action": "analysis_generate", "campaign_id": fixture.campaign.id}, host=fixture.host, caller=CALLER
    )
    assert not started.is_error and started.structured is not None, started.text
    (job,) = started.structured["jobs"]
    status = await settled(fixture.host, job["job_id"])
    assert (status.state, status.status) == ("completed", "stored")

    listed = await evals.call(
        {"action": "analyses_list", "campaign_id": fixture.campaign.id}, host=fixture.host, caller=CALLER
    )
    assert listed.structured is not None, listed.text
    (line,) = listed.structured["analyses"]
    assert (line["id"], line["basis"], line["report_plan_id"]) == (status.analysis_id, "doc", None)
    assert line["report_doc_id"] is not None
    assert (line["findings"], line["recommendations"]) == (1, 0)

    read = await evals.call(
        {"action": "report_read", "campaign_id": fixture.campaign.id, "format": "json"},
        host=fixture.host,
        caller=CALLER,
    )
    assert read.structured is not None, read.text
    # The page is the stored report document's, and the result carries the document.
    assert (read.structured["basis"], read.structured["analysis_id"], read.structured["analysis_basis"]) == (
        "analysis",
        status.analysis_id,
        "doc",
    )
    assert "plan" not in read.structured and read.structured["doc"]["schema"] == "report-doc/1"
    body = json.loads(read.structured["body"])
    assert body["analysis_id"] == status.analysis_id and body["report_doc"]["id"] == line["report_doc_id"]
