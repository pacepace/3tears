"""A generation is written by the report planner: a planned analysis, its stored plan, and the insights its claims mint.

Contracts held here (``threetears.evals.analysis.planned_analysis``, ``threetears.evals.analysis.service``,
``threetears.evals.ops``):

- a generation stores a planned analysis (no memo; ``report_plan_id`` names its plan), the plan beside it
  (``eval_report_plan``, with how it was generated and the digest of the report input it was written against), and
  one insight per durable claim, its statement the claim with every figure key written out and its runs those behind
  the arms and aspects the claim's keys name;
- ``recommended_variant_key`` is the variant key of the plan's recommended option;
- an analysis the memo writer wrote, as every analysis stored before the planner was, still lists, reads and renders
  as it did, and says it is a memo; a reporter case freezes it as a memo case, and a planned analysis as a plan case
  (``test_reporter_plan_cases.py`` holds the plan case);
- the plan is regenerable: ``drop_derived`` sweeps it, and deleting its analysis deletes it (its insights stay);
- the campaign's page for a planned analysis carries the plan's first screen, and the actions round-trip: a generation
  started through ``analysis_generate`` is listed by ``analyses_list`` as a plan and read by ``report_read`` with its
  plan;
- a key's subjects (the arms and aspects it names) are read off its one reading of the input.

The model is the one non-production part: a fixtured planner writes a structure and its prose over whatever report
input the generation built (``toyhost_plan``).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.analysis import build_report
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.planned_analysis import plan_recommended_arm, report_input_digest, report_plan_of
from threetears.evals.analysis.plan_prompt import EVAL_REPORT_PLAN_DEFAULT
from threetears.evals.analysis.report.figure_keys import key_subjects
from threetears.evals.analysis.report.serialize import serialize_report
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.service import (
    campaign_report,
    get_report_plan,
    prepare_analysis_generation,
    run_analysis_generation,
)
from threetears.evals.kernel.campaign import report_plan_id
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.kernel.prompts.catalogue import REPORT_PLAN_PROMPT
from threetears.evals.ops import analyses_list, analysis_delete, analysis_generate, report_read
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, ops_fixture, settled
from packages.evals.tests.report_plan_support import r5_input
from packages.evals.tests.report_support import stored_memo_analysis, toy_campaign_host
from packages.evals.tests.reporter_case_support import freeze_corpus_campaign, reporter_fixture
from packages.evals.tests.toyhost_plan import PLAN_CALLS, PlanWriter, contract_name
from threetears.evals.kernel.inner_ceilings import MAX_GENERATION_CALLS


async def _prompt() -> str:
    return EVAL_REPORT_PLAN_DEFAULT


async def _generated() -> tuple[Any, Any, Any, Any, list[Any]]:
    """The toy campaign generated through the service: its host, prepared generation, analysis, writer and insights."""
    built: list[PlanWriter] = []

    def clients(role: str, model: str | None, *, temperature: float | None = None) -> PlanWriter:
        client = PlanWriter()
        built.append(client)
        return client

    host, campaign = off_loop(toy_campaign_host, clients)
    prepared = await prepare_analysis_generation(
        host, campaign.id, campaign.scope_id, model=None, resolve_prompt=_prompt, out_of_run_cap_usd=None
    )
    built[0].report_input = prepared.report_input
    analysis, insights = await run_analysis_generation(
        host, prepared, prompt_id=REPORT_PLAN_PROMPT, max_output_tokens=4000
    )
    return host, prepared, analysis, built[0], insights


class TestAGeneration:
    async def test_it_stores_a_planned_analysis_and_its_plan_beside_it(self) -> None:
        host, prepared, analysis, writer, _ = await _generated()

        assert (analysis.basis, analysis.document, analysis.resolutions) == ("plan", None, [])
        assert analysis.report_plan_id == report_plan_id(analysis.id)
        assert len(writer.calls) == PLAN_CALLS, "a structure, then its prose"
        stored = off_loop(host.storage.load_analysis, analysis.id, analysis.scope_id)
        assert stored == analysis
        plan = off_loop(get_report_plan, host.storage, stored)
        assert plan is not None
        assert (plan.id, plan.analysis_id, plan.campaign_id) == (
            analysis.report_plan_id,
            analysis.id,
            analysis.campaign_id,
        )
        assert plan.report_input_digest == report_input_digest(prepared.report_input)
        assert (plan.generation.prompt_id, plan.generation.calls, plan.generation.repair_attempts) == (
            REPORT_PLAN_PROMPT,
            PLAN_CALLS,
            0,
        )
        assert plan.generation.token_cost == pytest.approx(analysis.generation.token_cost)
        assert plan.plan["schema"] == "report-plan/1" and report_plan_of(plan).mode == "decide"

    async def test_the_recommended_variant_key_is_the_plans_recommended_option(self) -> None:
        host, prepared, analysis, _, _ = await _generated()
        stored = off_loop(get_report_plan, host.storage, analysis)
        assert stored is not None
        recommended = plan_recommended_arm(report_plan_of(stored))
        (arm,) = [arm for arm in prepared.report_input.arms if arm.id == recommended]
        assert analysis.recommended_variant_key == arm.variant_key is not None

    async def test_each_durable_claim_mints_an_insight_citing_the_runs_behind_its_keys(self) -> None:
        host, prepared, analysis, _, insights = await _generated()
        plan = off_loop(get_report_plan, host.storage, analysis)
        assert plan is not None

        (claim,) = report_plan_of(plan).durable_claims
        (insight,) = insights
        assert "{" not in insight.statement and "%" in insight.statement, "every figure key is written out"
        assert (insight.confidence, insight.source_analysis_id, insight.source_campaign_id) == (
            claim.confidence,
            analysis.id,
            analysis.campaign_id,
        )
        # The claim's key names the recommended arm alone, so the insight cites that arm's runs and no other's.
        recommended = plan_recommended_arm(report_plan_of(plan))
        (arm,) = [arm for arm in prepared.report_input.arms if arm.id == recommended]
        behind = {
            run
            for aspect in prepared.bundle.aspects
            for cell in aspect.cell_measures
            if cell.variant_key == arm.variant_key
            for run in cell.run_ids
        }
        assert insight.evidence_run_ids == sorted(behind) and behind
        assert set(insight.evidence_run_ids) < set(prepared.bundle.run_ids)
        assert analysis.id in insight.invalidation_trigger
        ledger = off_loop(host.storage.query_insights, analysis.scope_id, subject_id=analysis.subject_id)
        assert [each.id for each in ledger] == [insight.id]

    async def test_its_page_carries_the_plans_first_screen(self) -> None:
        host, _, analysis, _, _ = await _generated()

        page = off_loop(campaign_report, host, analysis.campaign_id, analysis.scope_id)
        assert (page.basis, page.analysis_id, page.finding_count) == ("analysis", analysis.id, 0)
        markdown = serialize_report(page, "markdown")
        assert "it is the option the evidence favours." in markdown and "**How sure:** Low." in markdown
        assert "**Recommendation:**" in markdown and "{arm:" not in markdown
        assert "No recommendation was written" not in markdown

    async def test_an_aspect_page_of_a_planned_analysis_is_refused(self) -> None:
        _, _, analysis, _, _ = await _generated()

        with pytest.raises(RefusedInput, match="written by the report planner"):
            build_report(analysis)


class _RefusedOnceEach(PlanWriter):
    """A planner whose first reply to each call is refused (not JSON), so each call buys its one repair."""

    def __init__(self) -> None:
        super().__init__()
        self.refused: set[str | None] = set()

    async def generate(self, *, system: str, user: str, response_format: Any = None, tools: Any = None) -> Any:
        name = contract_name(response_format)
        if name not in self.refused:
            self.refused.add(name)
            self.calls.append({"system": system, "user": user})
            self.response_formats.append(response_format)
            self.completion.content = "not json at all"
            return self.completion
        return await super().generate(system=system, user=user, response_format=response_format, tools=tools)


async def test_a_generation_makes_at_most_max_generation_calls() -> None:
    """The structure, its repair, the prose, its repair: the bound the estimate, the cap and the job budget read."""
    built: list[PlanWriter] = []

    def clients(role: str, model: str | None, *, temperature: float | None = None) -> PlanWriter:
        client = _RefusedOnceEach()
        built.append(client)
        return client

    host, campaign = off_loop(toy_campaign_host, clients)
    prepared = await prepare_analysis_generation(
        host, campaign.id, campaign.scope_id, model=None, resolve_prompt=_prompt, out_of_run_cap_usd=None
    )
    built[0].report_input = prepared.report_input
    analysis, _ = await run_analysis_generation(host, prepared, prompt_id=REPORT_PLAN_PROMPT, max_output_tokens=4000)

    assert len(built[0].calls) == MAX_GENERATION_CALLS == 4
    plan = off_loop(get_report_plan, host.storage, analysis)
    assert plan is not None and (plan.generation.calls, plan.generation.repair_attempts) == (4, 2)
    assert plan.generation.structure_refusal and plan.generation.prose_refusal
    assert analysis.generation.repair_attempts == 2 and analysis.generation.repaired_refusal


class TestStoredMemos:
    async def test_a_memo_analysis_lists_reads_and_renders_as_it_did(self) -> None:
        fixture = off_loop(ops_fixture)
        memo, _ = await stored_memo_analysis(fixture.host.eval_host, fixture.campaign)
        assert memo.document is not None

        (line,) = off_loop(analyses_list, fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE).analyses
        assert (line.id, line.basis, line.report_plan_id) == (memo.id, "memo", None)
        assert (line.findings, line.recommendations) == (len(memo.document.findings), 1)
        read = off_loop(report_read, fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE, format="markdown")
        assert (read.basis, read.analysis_id, read.analysis_basis, read.plan) == ("analysis", memo.id, "memo", None)
        assert "**Recommended:**" in read.body, "the memo's recommendation, as written"
        assert off_loop(get_report_plan, fixture.host.eval_host.storage, memo) is None
        assert build_report(memo).basis == "analysis"

    async def test_a_reporter_case_pins_a_memo_as_a_memo_case_and_a_plan_as_a_plan_case(self) -> None:
        fixture = off_loop(reporter_fixture)
        (job,) = (await _generate_through_the_action(fixture)).jobs
        planned = (await settled(fixture.host, job.job_id)).analysis_id
        assert planned is not None
        memo, _ = await stored_memo_analysis(fixture.host.eval_host, fixture.campaign)

        assert off_loop(freeze_corpus_campaign, fixture, recorded_analysis_id=memo.id).kind_version == 1
        assert off_loop(freeze_corpus_campaign, fixture, recorded_analysis_id=planned).kind_version == 2


async def _generate_through_the_action(fixture: Any) -> Any:
    return await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)


class TestThePlanIsDerived:
    async def test_drop_derived_sweeps_it(self) -> None:
        host, _, analysis, _, _ = await _generated()

        dropped = off_loop(host.storage.drop_derived, [analysis.scope_id])

        assert dropped.counts["eval_report_plan"] == 1 and dropped.counts["eval_analysis"] == 1
        assert off_loop(host.storage.load_report_plan, analysis.report_plan_id, analysis.scope_id) is None

    async def test_deleting_its_analysis_deletes_it_and_keeps_the_insights(self) -> None:
        fixture = off_loop(ops_fixture)
        (job,) = (await _generate_through_the_action(fixture)).jobs
        analysis_id = (await settled(fixture.host, job.job_id)).analysis_id
        assert analysis_id is not None
        storage = fixture.host.eval_host.storage
        analysis = off_loop(storage.load_analysis, analysis_id, TOYHOST_SCOPE)
        assert analysis is not None and analysis.report_plan_id is not None

        off_loop(analysis_delete, fixture.host.eval_host, analysis_id, TOYHOST_SCOPE, confirm=analysis_id)

        assert off_loop(storage.load_report_plan, analysis.report_plan_id, TOYHOST_SCOPE) is None
        assert [insight.source_analysis_id for insight in off_loop(storage.query_insights, TOYHOST_SCOPE)] == [
            analysis_id
        ]


async def test_the_actions_round_trip_a_planned_analysis() -> None:
    """Generate, poll, list and read through the mounted tool: the listing and the read say a plan wrote it."""
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
    assert (line["id"], line["basis"], line["report_plan_id"]) == (
        status.analysis_id,
        "plan",
        report_plan_id(str(status.analysis_id)),
    )
    assert (line["findings"], line["recommendations"]) == (1, 1)

    read = await evals.call(
        {"action": "report_read", "campaign_id": fixture.campaign.id, "format": "json"},
        host=fixture.host,
        caller=CALLER,
    )
    assert read.structured is not None, read.text
    assert (read.structured["basis"], read.structured["analysis_id"], read.structured["analysis_basis"]) == (
        "analysis",
        status.analysis_id,
        "plan",
    )
    plan = read.structured["plan"]
    assert plan["schema"] == "report-plan/1" and plan["mode"] == "decide" and plan["durable_claims"]
    assert json.loads(read.structured["body"])["analysis_id"] == status.analysis_id


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
