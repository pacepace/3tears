"""A planned analysis, as the retired report planner stored it: its page, its plan and how both are swept.

The report planner no longer writes analyses (``analysis_generate`` runs the report writer), so every planned analysis
is one stored before it retired. These read the one the planner stored over the toy corpus campaign
(``report_support.stored_plan_analysis``). Contracts held here (``threetears.evals.analysis.service``,
``threetears.evals.ops``):

- a planned analysis read by its id (``analysis_report``) is the page its stored plan draws over the input built from
  its frozen bundle (no blocks of its own, its made line naming the planner), the aspects' folds a closed appendix
  after Method in HTML and Markdown, and ``trials="all"`` still reaches every failed trial's record through the folds;
  a key the frozen evidence no longer holds refuses that page naming the regeneration; a planned page is an
  analysis's and holds no blocks of its own;
- the campaign's page and ``report_read`` of a planned analysis are the code-only page with the one regenerate line,
  static where asked, never refused (``test_report_read_doc.py`` holds every form);
- an analysis the memo writer wrote still lists, reads and renders as it did, and says it is a memo;
- the plan is regenerable: ``drop_derived`` sweeps it, and deleting its analysis deletes it (its insights stay);
- the actions round-trip: a generation started through ``analysis_generate`` is the report writer's, listed by
  ``analyses_list`` as a document and read by ``report_read`` with no plan;
- a key's subjects (the arms and aspects it names) are read off its one reading of the input.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.analysis import CampaignReport, build_report
from pydantic import ValidationError

from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.build import analysis_campaign_bundle
from threetears.evals.analysis.report.figure_keys import key_subjects
from threetears.evals.analysis.report.plan_page.build import plan_page
from threetears.evals.analysis.report.serialize import serialize_report
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.analysis.service import analysis_report, campaign_report, get_report_plan, report_plan_of
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.ops import analyses_list, analysis_delete, report_read
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, ops_fixture, settled
from packages.evals.tests.report_plan_support import r5_input
from packages.evals.tests.report_support import stored_memo_analysis, stored_plan_analysis, toy_campaign_host
from packages.evals.tests.static_page_support import assert_static, less_drawings, less_what_static_leaves_out
from threetears.evals.analysis.report.campaign.memo_blocks import NO_LONGER_READS


def _without_instants(body: str) -> str:
    return re.sub(r"\d{4}-\d{2}-\d{2}T[0-9:.+]+", "", body)


async def _stored() -> tuple[Any, Any]:
    """The toy campaign's host, holding the analysis the planner stored over it, its plan and its insights."""
    host, campaign = off_loop(toy_campaign_host)
    analysis, _ = await stored_plan_analysis(host, campaign)
    return host, analysis


class TestAStoredPlan:
    async def test_its_page_is_the_page_its_plan_draws_its_folds_a_closed_appendix(self) -> None:
        host, analysis = await _stored()
        stored = off_loop(get_report_plan, host.storage, analysis)
        assert stored is not None
        bundle = analysis_campaign_bundle(analysis)

        page = off_loop(analysis_report, host.storage, analysis.id, analysis.scope_id)

        assert (page.basis, page.analysis_id, page.finding_count, page.blocks) == ("analysis", analysis.id, 0, [])
        assert page.plan_page is not None
        # The page is the one the stored plan draws over the input it was written against, made by the planner.
        assert page.plan_page == plan_page(report_plan_of(stored), report_input(bundle), made=page.plan_page.made)
        assert analysis.id in page.plan_page.made and stored.generation.generator_model in page.plan_page.made
        assert page.fold_ids and len(page.aspects) == len(bundle.aspects), "every aspect keeps its fold"

        html = serialize_report(page, "html")
        assert 'data-page="planned"' in html and "{" + "arm:" not in html
        assert html.index('class="method"') < html.index('data-part="appendix"')
        assert '<details class="section" data-part="appendix">' in html, "the appendix is closed"
        for slug in page.fold_ids:
            assert html.index('data-part="appendix"') < html.index(f'id="aspect-{slug}"'), "each fold is in it"
        assert "No recommendation was written" not in html, "the code-only page's first screen is gone"

        markdown = serialize_report(page, "markdown")
        assert page.plan_page.masthead.system in markdown.splitlines()[0] and f"# {page.plan_page.headline}" in markdown
        assert markdown.index(page.plan_page.made) < markdown.index("## Evidence record")
        assert all(f"## {fold.aspect}" in markdown.split("## Evidence record", 1)[1] for fold in page.aspects)

    async def test_its_report_read_serves_the_code_only_page_with_the_line_static_where_asked(self) -> None:
        host, analysis = await _stored()

        static = off_loop(report_read, host, analysis.campaign_id, analysis.scope_id, format="html", static=True)
        default = off_loop(report_read, host, analysis.campaign_id, analysis.scope_id, format="html")

        assert (static.basis, static.analysis_id, static.analysis_basis) == ("code_only", analysis.id, "plan")
        assert '<article data-page="planned"' not in static.body and NO_LONGER_READS in static.body
        assert_static(static.body)
        assert _without_instants(less_drawings(static.body)) == _without_instants(
            less_what_static_leaves_out(default.body)
        )

    async def test_trials_all_lays_each_failed_trial_s_record_out_in_its_fold(self) -> None:
        host, analysis = await _stored()

        table = off_loop(analysis_report, host.storage, analysis.id, analysis.scope_id)
        everything = off_loop(analysis_report, host.storage, analysis.id, analysis.scope_id, trials="all")

        assert table.plan_page == everything.plan_page, "the plan's page reads the same either way"
        assert [fold.report for fold in table.aspects] != [fold.report for fold in everything.aspects]
        assert serialize_report(everything, "html").count("<details") > serialize_report(table, "html").count(
            "<details"
        )

    async def test_a_plan_key_its_frozen_evidence_no_longer_holds_refuses_naming_the_regeneration(self) -> None:
        host, analysis = await _stored()
        stored = off_loop(get_report_plan, host.storage, analysis)
        assert stored is not None
        unheld = {**stored.plan, "headline": "Keep it: {pass_rate.nobody}."}
        off_loop(host.storage.save_report_plan, stored.model_copy(update={"plan": unheld}))

        with pytest.raises(RefusedInput, match=r"generate the analysis again \(analysis_generate\)") as refused:
            off_loop(analysis_report, host.storage, analysis.id, analysis.scope_id)
        assert "pass_rate.nobody" in str(refused.value)
        page = off_loop(campaign_report, host, analysis.campaign_id, analysis.scope_id)
        assert (page.basis, page.unread_analysis_id) == ("code_only", analysis.id), "the campaign's page never refuses"
        lines = off_loop(analyses_list, host, analysis.campaign_id, analysis.scope_id).analyses
        assert [line.needs_regeneration for line in lines] == [NO_LONGER_READS], "the plan still parses: the line alone"

    async def test_a_planned_page_is_an_analysis_s_and_holds_no_blocks_of_its_own(self) -> None:
        host, analysis = await _stored()
        page = off_loop(analysis_report, host.storage, analysis.id, analysis.scope_id)
        assert page.plan_page is not None

        with pytest.raises(ValidationError, match="a planned page lays out an analysis"):
            CampaignReport.model_validate({**dict(page), "basis": "code_only"})
        with pytest.raises(ValidationError, match="a planned page is the page's whole first part"):
            CampaignReport.model_validate({**dict(page), "blocks": [page.plan_page.options.table]})

    async def test_an_aspect_page_of_a_planned_analysis_is_refused(self) -> None:
        _, analysis = await _stored()

        with pytest.raises(RefusedInput, match="written by the report planner"):
            build_report(analysis)


class TestStoredMemos:
    async def test_a_memo_analysis_lists_reads_and_renders_as_it_did(self) -> None:
        fixture = off_loop(ops_fixture)
        memo, _ = await stored_memo_analysis(fixture.host.eval_host, fixture.campaign)
        assert memo.document is not None

        (line,) = off_loop(analyses_list, fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE).analyses
        assert (line.id, line.basis, line.report_plan_id) == (memo.id, "memo", None)
        assert (line.findings, line.recommendations) == (len(memo.document.findings), 1)
        read = off_loop(report_read, fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE, format="markdown")
        assert (read.basis, read.analysis_id, read.analysis_basis, read.doc) == ("analysis", memo.id, "memo", None)
        assert read.needs_regeneration is None
        assert "**Recommended:**" in read.body, "the memo's recommendation, as written"
        assert off_loop(get_report_plan, fixture.host.eval_host.storage, memo) is None
        assert build_report(memo).basis == "analysis"


class TestAStoredPlanThisBuildCannotRead:
    """A plan stored under a shape the plan model no longer accepts (it moves inside regenerable v13) is listed as
    needing regeneration, the one regenerate line then its reason, beside every analysis that reads; its report and the
    campaign's page are the code-only page with that line, never refused, and its page by id refuses, naming the
    regeneration."""

    async def _unreadable(self) -> tuple[Any, str]:
        fixture = off_loop(ops_fixture)
        analysis_id = (await stored_plan_analysis(fixture.host.eval_host, fixture.campaign))[0].id
        storage = fixture.host.eval_host.storage
        analysis = off_loop(storage.load_analysis, analysis_id, TOYHOST_SCOPE)
        stored = off_loop(get_report_plan, storage, analysis)
        assert stored is not None
        moved = {key: value for key, value in stored.plan.items() if key != "headline"}
        off_loop(storage.save_report_plan, stored.model_copy(update={"plan": moved}))
        return fixture, analysis_id

    async def test_it_is_listed_as_needing_regeneration_with_the_reason(self) -> None:
        fixture, analysis_id = await self._unreadable()
        memo, _ = await stored_memo_analysis(fixture.host.eval_host, fixture.campaign)

        lines = {
            line.id: line
            for line in off_loop(analyses_list, fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE).analyses
        }

        unreadable = lines[analysis_id].needs_regeneration
        assert unreadable is not None and unreadable.startswith(f"{NO_LONGER_READS} (")
        assert "report plan this build cannot read" in unreadable and "headline" in unreadable
        assert (lines[analysis_id].basis, lines[analysis_id].findings) == ("plan", 0)
        assert lines[memo.id].needs_regeneration is None, "one unreadable plan refuses no other analysis's line"

    async def test_its_report_reads_code_only_with_the_line_and_its_page_by_id_refuses(self) -> None:
        fixture, analysis_id = await self._unreadable()
        host = fixture.host.eval_host

        read = off_loop(report_read, host, fixture.campaign.id, TOYHOST_SCOPE, format="markdown")
        assert (read.basis, read.analysis_id) == ("code_only", analysis_id)
        assert read.needs_regeneration is not None and read.needs_regeneration.startswith(f"{NO_LONGER_READS} (")
        assert NO_LONGER_READS in read.body
        assert off_loop(campaign_report, host, fixture.campaign.id, TOYHOST_SCOPE).unread_analysis_id == analysis_id
        with pytest.raises(RefusedInput, match="report plan this build cannot read"):
            off_loop(analysis_report, host.storage, analysis_id, TOYHOST_SCOPE)


class TestThePlanIsDerived:
    async def test_drop_derived_sweeps_it(self) -> None:
        host, analysis = await _stored()

        dropped = off_loop(host.storage.drop_derived, [analysis.scope_id])

        assert dropped.counts["eval_report_plan"] == 1 and dropped.counts["eval_analysis"] == 1
        assert off_loop(host.storage.load_report_plan, analysis.report_plan_id, analysis.scope_id) is None

    async def test_deleting_its_analysis_deletes_it_and_keeps_the_insights(self) -> None:
        fixture = off_loop(ops_fixture)
        analysis_id = (await stored_plan_analysis(fixture.host.eval_host, fixture.campaign))[0].id
        storage = fixture.host.eval_host.storage
        analysis = off_loop(storage.load_analysis, analysis_id, TOYHOST_SCOPE)
        assert analysis is not None and analysis.report_plan_id is not None

        off_loop(analysis_delete, fixture.host.eval_host, analysis_id, TOYHOST_SCOPE, confirm=analysis_id)

        assert off_loop(storage.load_report_plan, analysis.report_plan_id, TOYHOST_SCOPE) is None
        assert [insight.source_analysis_id for insight in off_loop(storage.query_insights, TOYHOST_SCOPE)] == [
            analysis_id
        ]


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


class TestKeySubjects:
    @pytest.fixture
    def report(self) -> ReportInput:
        return ReportInput.model_validate(r5_input())

    def test_a_reading_names_its_arm_the_asked_aspect_and_the_control(self, report: ReportInput) -> None:
        subjects = key_subjects("reading.qwen3.8-flash.change", report)
        assert (subjects.arms, subjects.aspects, subjects.against_control) == (
            ("qwen3.8-flash",),
            ("character adherence (v2)",),
            True,
        )

    def test_an_arm_is_its_own_subject(self, report: ReportInput) -> None:
        subjects = key_subjects("arm:1017", report)
        assert (subjects.arms, subjects.aspects, subjects.against_control) == (("1017",), (), False)

    def test_a_per_arm_figure_names_its_arm_and_no_aspect(self, report: ReportInput) -> None:
        subjects = key_subjects("cost_per_1k.September", report)
        assert (subjects.arms, subjects.aspects, subjects.against_control) == (("September",), (), False)

    def test_a_key_that_names_nothing_is_refused(self, report: ReportInput) -> None:
        with pytest.raises(UnresolvableReference, match="names no figure"):
            key_subjects("reading.nobody.change", report)
