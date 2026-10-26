"""The reporter eval scores plans: a planned analysis freezes into a plan reporter case, judged as a reader reads it.

Contracts held here (``threetears.evals.analysis.reporter_kind``, ``threetears.evals.analysis.service``,
``threetears.evals.analysis.reporter_bank``, ``threetears.evals.ops``):

- **What the judge reads** (:func:`render_plan_for_judge`): the planner's own part of the plan (``PlannedReport``'s
  fields, at every depth, nothing code fills beside it), as JSON, every prose field written out by the renderer the
  campaign's page uses, against the same report input, so the judged headline is the page's headline;
- **a plan case** (reporter kind version 2) freezes the bundle frozen with the analysis, the report input built from
  it and its digest, the plan in key form, the planner's prompt version, the plan as the judge reads it, and what the
  planner read (``plan_user_message``) with the digest its generation recorded; a memo case (version 1) carries none
  of it and is stored exactly as before; a case holding half of a plan, or a frozen input that does not digest to
  what it names, is refused on read;
- **the freeze** accepts a planned analysis, of a campaign of several aspects too; it is idempotent, takes an excerpt
  from the judged text and refuses one that is not in it; a memo analysis still freezes as a memo case;
- **the as-recorded replay** hands the judge the frozen plan text against what its planner read; a generating
  reporter (a memo writer) never runs a plan case: its launch is refused, and a cell excludes it as the harness's;
- **calibration** groups a run's cases by kind version and prompt version and never pools agreement across groups;
- **end to end**: a two-aspect campaign generated through the service with a fixtured planner, its stored plan, its
  page, its ``report_read``, and a plan case frozen of it, agree on every figure: the judged text is the plan rendered
  against the frozen input, every number in it comes from a key, and the judge's material is what the planner read.

The model is the one non-production part: a fixtured planner writes a structure and its prose over whatever report
input the generation built (``toyhost_plan``).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.analysis.generator import user_message_digest
from threetears.evals.analysis.planned_analysis import report_input_digest, report_plan_of
from threetears.evals.analysis.planner import plan_user_message
from threetears.evals.analysis.plan_prompt import EVAL_REPORT_PLAN_DEFAULT
from threetears.evals.analysis.report.md_parts import table_lines
from threetears.evals.analysis.report.plan_page.build import plan_page
from threetears.evals.analysis.report.plan_page.drawable import plan_look
from threetears.evals.analysis.report.plan_page.tables import arms_key
from threetears.evals.analysis.report.figure_keys import render
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_plan_checks import validate_plan
from threetears.evals.analysis.report_plan_facts import plan_facts
from threetears.evals.analysis.reporter_bank import read_calibration
from threetears.evals.analysis.reporter_kind import (
    AS_RECORDED_MODEL,
    MEMO_CASE_VERSION,
    PLAN_CASE_VERSION,
    REPORTER_CASE_KEY,
    AsRecordedReporterKind,
    LabelCriterion,
    PreparedReporter,
    ReporterCase,
    ReporterKind,
    ReporterLabel,
    render_plan_for_judge,
    reporter_case_of,
    reporter_case_payload,
)
from threetears.evals.analysis.service import (
    campaign_report,
    get_report_plan,
    prepare_analysis_generation,
    run_analysis_generation,
)
from threetears.evals.kernel import Aspect, ValidationFailedError
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.identity import IDENTITY_VERSION
from threetears.evals.kernel.prompts.catalogue import REPORT_PLAN_PROMPT
from threetears.evals.kernel.report_plan import PlannedReport, ReportPlan
from threetears.evals.ops import ReporterCaseFreeze, analysis_generate, report_read, reporter_case_freeze
from threetears.evals.schema.models import EvalResult, EvalTestCase, RubricScore
from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_NARROW, TOYHOST_WIDE, toyhost_aspect_design
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_INSTANT, toyhost_batch, toyhost_measurements
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import TOYHOST_SCOPE, OpsFixture, settled
from packages.evals.tests.report_plan_support import r5_input, r5_plan
from packages.evals.tests.scripted_table import FakeCellSink
from packages.evals.tests.report_support import stored_memo_analysis, toy_campaign_host
from packages.evals.tests.reporter_case_support import (
    GROUNDED,
    READER,
    attests_reader,
    freeze_corpus_campaign,
    reporter_fixture,
    reporter_template,
)
from packages.evals.tests.toyhost_memo import MODEL, PROMPT, PROMPT_ID
from packages.evals.tests.toyhost_plan import PlanWriter

_ORDINAL = LabelCriterion(description="Every claim traces to the evidence.", scale="ordinal")


# --- what the judge reads ---------------------------------------------------------------------------------------------


def _judged_json(text: str) -> Any:
    """The plan's JSON out of the judged text: it follows the arms key where the page shows one."""
    return json.loads(text[text.index("\n{\n") + 1 :] if not text.startswith("{") else text)


class TestWhatTheJudgeReads:
    @pytest.fixture
    def pair(self) -> tuple[ReportPlan, ReportInput]:
        return ReportPlan.model_validate(r5_plan()), ReportInput.model_validate(r5_input())

    def test_it_is_the_planners_part_and_nothing_code_fills(self, pair: tuple[ReportPlan, ReportInput]) -> None:
        plan, report = pair
        read = _judged_json(render_plan_for_judge(plan, report))

        assert list(read) == list(PlannedReport.model_fields)
        assert "columns" not in read["options"], "the options columns are code's"
        assert all({"level", "unit", "domain"}.isdisjoint(section["chart"]) for section in read["sections"])
        assert "discloses" not in (read["flag"][0] if read["flag"] else {})

    def test_every_prose_field_is_written_out_as_the_page_writes_it(self, pair: tuple[ReportPlan, ReportInput]) -> None:
        plan, report = pair
        read = _judged_json(render_plan_for_judge(plan, report))
        page = plan_page(plan, report)

        assert read["headline"] == page.headline == render(plan.headline, report, {})
        assert read["lede"] == page.lede == render(plan.lede, report, {})
        assert [slot["text"] for slot in read["decision"]["slots"]] == [slot.text for slot in page.decision]
        assert [slot.text for slot in page.decision] == [render(slot.text, report, {}) for slot in plan.decision.slots]
        assert [claim["keys"] for claim in read["durable_claims"]] == [claim.keys for claim in plan.durable_claims]

    def test_the_arms_key_the_page_shows_leads_and_explains_every_tag(
        self, pair: tuple[ReportPlan, ReportInput]
    ) -> None:
        plan, report = pair
        plan = plan.model_copy(update={"show_arms_key": True})
        text = render_plan_for_judge(plan, report)
        facts = plan_facts(report)
        key = arms_key(report, plan_look(plan, report, facts))
        shown = "\n".join([f"## {key.eyebrow}", "", key.takeaway, "", *table_lines(key.table)])

        assert text.startswith(shown + "\n\n{\n")
        assert plan_page(plan, report).arms_key == key
        for arm in facts.arms:
            tag = plan_look(plan, report, facts).arm(arm.id).tag
            assert f"| {tag} · " in shown, "the key says which arm each tag is"
        assert any(
            f"({plan_look(plan, report, facts).arm(arm.id).tag})" in text.split("\n{\n", 1)[1] for arm in facts.arms
        ), "the prose cites a tagged arm, and the key above explains it"

    def test_a_page_with_no_arms_key_renders_the_bare_json(self, pair: tuple[ReportPlan, ReportInput]) -> None:
        plan, report = pair
        plan = plan.model_copy(update={"show_arms_key": False})
        text = render_plan_for_judge(plan, report)

        assert text.startswith("{\n")
        assert json.loads(text)["headline"] == render(plan.headline, report, {})
        assert plan_page(plan, report).arms_key is None

    def test_one_plan_renders_to_one_string(self, pair: tuple[ReportPlan, ReportInput]) -> None:
        plan, report = pair
        assert render_plan_for_judge(plan, report) == render_plan_for_judge(plan.model_copy(deep=True), report)


# --- the case's shape -------------------------------------------------------------------------------------------------


def _plan_fields(**changes: Any) -> dict[str, Any]:
    # Frozen as the freeze freezes it: the input's own JSON by alias.
    report = ReportInput.model_validate(r5_input()).model_dump(mode="json", by_alias=True)
    message = "what the planner read"
    fields: dict[str, Any] = {
        "bundle": {"campaign_id": "camp-1"},
        "bundle_fingerprint": "fp",
        "bundle_assembled_at": "2026-01-01T00:00:00+00:00",
        "recorded_analysis_id": "plan-1",
        "recorded_memo": "the plan as the judge reads it",
        "writer_message": message,
        "recorded_writer_message_digest": user_message_digest(message),
        "kind_version": PLAN_CASE_VERSION,
        "report_input": report,
        "report_input_digest": report_input_digest(ReportInput.model_validate(report)),
        "recorded_report_input_digest": report_input_digest(ReportInput.model_validate(report)),
        "recorded_plan": r5_plan(),
        "planner_prompt_version": "abc123",
    }
    return fields | changes


class TestTheCaseShape:
    def test_a_plan_case_round_trips_through_its_payload(self) -> None:
        case = ReporterCase.model_validate(_plan_fields())
        stored = EvalTestCase(scope_id="s", template_id="t", host_payload=reporter_case_payload(case))

        assert reporter_case_of(stored) == case

    @pytest.mark.parametrize("dropped", ["report_input", "recorded_plan", "planner_prompt_version", "recorded_memo"])
    def test_a_plan_case_missing_a_part_is_refused(self, dropped: str) -> None:
        fields = _plan_fields()
        fields[dropped] = None
        if dropped == "recorded_memo":
            fields |= {"writer_message": None, "recorded_writer_message_digest": None}
        with pytest.raises(ValidationError, match=f"lacks .*{dropped}"):
            ReporterCase.model_validate(fields)

    def test_a_memo_case_holding_a_plan_part_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="freezes no plan, and this case holds recorded_plan"):
            ReporterCase.model_validate(
                _plan_fields(
                    kind_version=MEMO_CASE_VERSION,
                    report_input=None,
                    report_input_digest=None,
                    recorded_report_input_digest=None,
                    planner_prompt_version=None,
                )
            )

    def test_a_frozen_input_that_does_not_digest_to_what_it_names_is_refused(self) -> None:
        frozen = _plan_fields()["report_input"]
        tampered = frozen | {"arms": frozen["arms"][:1]}
        with pytest.raises(ValidationError, match="does not hold the input it names"):
            ReporterCase.model_validate(_plan_fields(report_input=tampered))

    def test_a_memo_case_is_stored_exactly_as_before(self) -> None:
        message = "message"
        case = ReporterCase(
            bundle={"campaign_id": "camp-1"},
            bundle_fingerprint="fp",
            bundle_assembled_at="2026-01-01T00:00:00+00:00",
            recorded_analysis_id="memo-1",
            recorded_memo="memo",
            writer_message=message,
            recorded_writer_message_digest=user_message_digest(message),
        )

        assert case.kind_version == MEMO_CASE_VERSION
        assert set(reporter_case_payload(case)[REPORTER_CASE_KEY]) == {
            "bundle",
            "bundle_fingerprint",
            "bundle_assembled_at",
            "recorded_analysis_id",
            "recorded_memo",
            "writer_message",
            "recorded_writer_message_digest",
            "labels",
            "limits",
            "supersedes",
        }


# --- the freeze -------------------------------------------------------------------------------------------------------


async def _planned_analysis_id(fixture: OpsFixture) -> str:
    (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    analysis_id = (await settled(fixture.host, job.job_id)).analysis_id
    assert analysis_id is not None
    return analysis_id


def _stored_case(fixture: OpsFixture, test_case_id: str) -> ReporterCase:
    stored = off_loop(fixture.host.eval_host.storage.load_test_case, test_case_id, TOYHOST_SCOPE)
    assert stored is not None
    case = reporter_case_of(stored)
    assert case is not None
    return case


class TestTheFreeze:
    async def test_a_planned_analysis_freezes_as_a_plan_case(self) -> None:
        fixture = off_loop(reporter_fixture)
        planned = await _planned_analysis_id(fixture)

        receipt = off_loop(freeze_corpus_campaign, fixture, recorded_analysis_id=planned)

        case = _stored_case(fixture, receipt.test_case_id)
        storage = fixture.host.eval_host.storage
        analysis = off_loop(storage.load_analysis, planned, TOYHOST_SCOPE)
        assert analysis is not None
        stored = off_loop(get_report_plan, storage, analysis)
        assert stored is not None
        assert (receipt.kind_version, case.kind_version) == (PLAN_CASE_VERSION, PLAN_CASE_VERSION)
        assert case.recorded_plan == stored.plan
        assert receipt.planner_prompt_version == case.planner_prompt_version == stored.generation.prompt_version
        assert receipt.report_input_digest == case.report_input_digest == stored.report_input_digest
        assert case.bundle_fingerprint == analysis.generation.bundle_fingerprint, "the bundle the planner planned over"
        report = ReportInput.model_validate(case.report_input)
        assert case.recorded_memo == render_plan_for_judge(report_plan_of(stored), report)
        assert case.writer_message == plan_user_message(report)
        assert receipt.writer_message_check == "verified", "the planner's own message, by its recorded digest"
        assert case.limits == []

    async def test_a_re_freeze_returns_the_case_and_an_excerpt_follows_the_judged_text(self) -> None:
        fixture = off_loop(reporter_fixture)
        planned = await _planned_analysis_id(fixture)
        first = off_loop(freeze_corpus_campaign, fixture, recorded_analysis_id=planned)
        text = _stored_case(fixture, first.test_case_id).recorded_memo
        assert text is not None
        excerpt = _judged_json(text)["headline"]

        again = off_loop(freeze_corpus_campaign, fixture, recorded_analysis_id=planned)
        assert again.test_case_id == first.test_case_id

        label = {"dimension": GROUNDED, "direction": "high", "quote": "it checks", "memo_excerpt": "not in the plan"}
        with pytest.raises(ValidationFailedError, match="is not a passage of the recorded memo"):
            off_loop(freeze_corpus_campaign, fixture, recorded_analysis_id=planned, labels=[label])
        labelled = off_loop(
            freeze_corpus_campaign,
            fixture,
            recorded_analysis_id=planned,
            labels=[label | {"memo_excerpt": excerpt}],
            supersedes=[first.test_case_id],
        )
        (stamped,) = labelled.labels
        assert (stamped.memo_excerpt, labelled.kind_version) == (excerpt, PLAN_CASE_VERSION)

    async def test_a_memo_analysis_still_freezes_as_a_memo_case(self) -> None:
        fixture = off_loop(reporter_fixture)
        memo, _ = await stored_memo_analysis(fixture.host.eval_host, fixture.campaign)

        receipt = off_loop(freeze_corpus_campaign, fixture, recorded_analysis_id=memo.id)

        assert (receipt.kind_version, receipt.report_input_digest, receipt.planner_prompt_version) == (
            MEMO_CASE_VERSION,
            None,
            None,
        )
        stored = off_loop(fixture.host.eval_host.storage.load_test_case, receipt.test_case_id, TOYHOST_SCOPE)
        assert stored is not None and "kind_version" not in stored.host_payload[REPORTER_CASE_KEY]


# --- running a plan case ----------------------------------------------------------------------------------------------


# parity-with: threetears.evals.kernel.candidate_kind.CellSpanWindow
class _FakeSpans:
    """Both of a cell's tracing windows, opened with nothing watching."""

    @contextmanager
    def identity(self) -> Iterator[None]:
        yield

    @contextmanager
    def collecting(self) -> Iterator[None]:
        yield


def _plan_test_case() -> EvalTestCase:
    return EvalTestCase(
        id="plan-case",
        scope_id=TOYHOST_SCOPE,
        template_id="t",
        host_payload=reporter_case_payload(ReporterCase.model_validate(_plan_fields())),
    )


class TestRunningAPlanCase:
    async def test_the_replay_hands_the_judge_the_plan_text_against_what_its_planner_read(self) -> None:
        test_case = _plan_test_case()
        case = reporter_case_of(test_case)
        assert case is not None

        output = await AsRecordedReporterKind().invoke(
            PreparedReporter(model=AS_RECORDED_MODEL, span_window=_FakeSpans()),
            test_case,
            FakeCellSink(),
        )

        assert output.infra_errors == [] and output.judge_evidence is not None
        assert output.judge_evidence.artifact == case.recorded_memo
        assert output.judge_evidence.case_material == case.writer_message
        (recorded,) = output.output
        assert (recorded["kind_version"], recorded["plan"]) == (PLAN_CASE_VERSION, case.recorded_plan)

    async def test_a_memo_writer_excludes_a_plan_case_as_the_harness_s(self) -> None:
        host, _ = off_loop(toy_campaign_host)
        kind = ReporterKind(
            prompt=PROMPT, prompt_id=PROMPT_ID, prompt_version=None, client=None, model=MODEL, host=host
        )

        assert kind.price_case(_plan_test_case()) is None
        output = await kind.invoke(
            PreparedReporter(model=MODEL, span_window=_FakeSpans()), _plan_test_case(), FakeCellSink()
        )
        (error,) = output.infra_errors
        assert "judges a report plan" in error and "runs as recorded" in error


# --- calibration --------------------------------------------------------------------------------------------------------


def _result(result_id: str, test_case_id: str) -> EvalResult:
    return EvalResult(
        id=result_id,
        scope_id=TOYHOST_SCOPE,
        eval_run_id="run-1",
        test_case_id=test_case_id,
        model=AS_RECORDED_MODEL,
        k_iteration=1,
        rubric_scores=[RubricScore(dim=GROUNDED, scale="ordinal", score=4, axis="capability", judge_temperature=0.0)],
        termination="completed",
        cost_usd=0.0,
        cost_roles=["candidate", "inner_agent", "judge", "simulator"],
        usage=[],
        covariates={},
        phase_timings={},
        host_measures={},
        variant_key="vk-1",
        identity_version=IDENTITY_VERSION,
    )


def _memo_case() -> ReporterCase:
    message = "message"
    return ReporterCase(
        bundle={"campaign_id": "camp-2"},
        bundle_fingerprint="fp",
        bundle_assembled_at="2026-01-01T00:00:00+00:00",
        recorded_analysis_id="memo-1",
        recorded_memo="memo",
        writer_message=message,
        recorded_writer_message_digest=user_message_digest(message),
        labels=[ReporterLabel(dimension=GROUNDED, direction="high", quote="q", criterion=_ORDINAL)],
    )


def _calibration(cases: list[tuple[str, ReporterCase]]) -> Any:
    return read_calibration(
        run_id="run-1",
        template_id="tpl",
        status="completed",
        dimensions=[GROUNDED],
        cases=cases,
        results=[_result(f"r-{case_id}", case_id) for case_id, _ in cases],
        superseded_by={},
        archived={},
        live_criteria=None,
        judge_model="judge-x",
        effective_judges=None,
        ratings=[],
        template_digest="td-1",
    )


class TestCalibration:
    def test_memo_and_plan_cases_are_read_in_their_own_groups_and_never_pooled(self) -> None:
        plan = ReporterCase.model_validate(_plan_fields())

        read = _calibration([("memo-case", _memo_case()), ("plan-case", plan)])

        assert [(case.test_case_id, case.kind_version, case.prompt_version) for case in read.cases] == [
            ("memo-case", MEMO_CASE_VERSION, None),
            ("plan-case", PLAN_CASE_VERSION, "abc123"),
        ]
        assert [(group.kind_version, group.prompt_version, group.test_case_ids) for group in read.groups] == [
            (MEMO_CASE_VERSION, None, ["memo-case"]),
            (PLAN_CASE_VERSION, "abc123", ["plan-case"]),
        ]
        assert {group.template_digest for group in read.groups} == {"td-1"}
        assert read.rating_agreement is None, "agreement across a memo and a plan would pool two subjects"

    def test_two_planner_prompts_are_two_groups(self) -> None:
        one = ReporterCase.model_validate(_plan_fields())
        other = ReporterCase.model_validate(
            _plan_fields(planner_prompt_version="def456", bundle={"campaign_id": "camp-3"})
        )

        read = _calibration([("one", one), ("other", other)])

        assert [group.prompt_version for group in read.groups] == ["abc123", "def456"]
        assert read.rating_agreement is None

    def test_one_group_reads_the_run_wide_agreement(self) -> None:
        read = _calibration([("memo-case", _memo_case())])

        (group,) = read.groups
        assert read.rating_agreement is not None and read.rating_agreement == group.rating_agreement


# --- end to end: a campaign of two aspects, through the planner -------------------------------------------------------

#: The second aspect's label, and what each arm measured there: the wide chunk still ahead, both a little lower.
_SECOND_ASPECT = "receipts"
_SECOND_MEASUREMENTS: dict[int, dict[str, float]] = {
    TOYHOST_NARROW: {"cost_usd": 0.02, "total_ms": 950.0, "field_accuracy": 0.74, "layout_fidelity": 3},
    TOYHOST_WIDE: {"cost_usd": 0.03, "total_ms": 1450.0, "field_accuracy": 0.86, "layout_fidelity": 4},
}
_SECOND_CONTEXT_TOKENS: dict[int, int] = {TOYHOST_NARROW: 2_300, TOYHOST_WIDE: 7_400}


def _two_aspect_host(clients: Any) -> tuple[EvalHost, str]:
    """The toy host holding its corpus campaign with a second aspect over two more batches of the same two arms.

    The second aspect's batches record one more host key (``intake``), no lever, so its arms are the corpus's arms.
    """
    host, campaign = toy_campaign_host(clients)
    batches = [
        toyhost_batch(
            chunk_tokens=level,
            retriever_top_k=3,
            extraction_schema="v1",
            ocr_engine_version="tess-5.3.1",
            reviewer_pool="pool-a",
            intake=_SECOND_ASPECT,
        )
        for level in (TOYHOST_NARROW, TOYHOST_WIDE)
    ]
    for batch, level in zip(batches, (TOYHOST_NARROW, TOYHOST_WIDE), strict=True):
        host.storage.save_eval_run(batch)
        measured = toyhost_measurements(
            batch,
            profile=toyhost_profile(),
            covariates={"context_tokens_in": _SECOND_CONTEXT_TOKENS[level]},
            cost_usd=_SECOND_MEASUREMENTS[level]["cost_usd"],
            total_ms=_SECOND_MEASUREMENTS[level]["total_ms"],
            field_accuracy=_SECOND_MEASUREMENTS[level]["field_accuracy"],
            layout_fidelity=int(_SECOND_MEASUREMENTS[level]["layout_fidelity"]),
        )
        for result in measured:
            host.storage.save_eval_result(result)
    (first,) = campaign.aspects
    second = Aspect(
        label=_SECOND_ASPECT,
        template_id=first.template_id,
        run_ids=[batch.id for batch in batches],
        design=toyhost_aspect_design(),
        declared_at=TOYHOST_INSTANT,
        asked_at=TOYHOST_INSTANT,
    )
    host.storage.save_campaign(campaign.model_copy(update={"aspects": [first, second]}))
    host.storage.save_template(reporter_template(campaign.scope_id))
    return host, campaign.id


async def _plan_prompt() -> str:
    return EVAL_REPORT_PLAN_DEFAULT


async def test_a_two_aspect_campaign_planned_paged_read_and_frozen_agrees_on_every_figure() -> None:
    writers: list[PlanWriter] = []

    def clients(role: str, model: str | None, *, temperature: float | None = None) -> PlanWriter:
        writers.append(PlanWriter())
        return writers[-1]

    host, campaign_id = off_loop(_two_aspect_host, clients)
    prepared = await prepare_analysis_generation(
        host, campaign_id, TOYHOST_SCOPE, model=None, resolve_prompt=_plan_prompt, out_of_run_cap_usd=None
    )
    assert [aspect.id for aspect in prepared.report_input.aspects] == ["extract_invoice_fields", _SECOND_ASPECT]
    writers[0].report_input = prepared.report_input
    analysis, _ = await run_analysis_generation(host, prepared, prompt_id=REPORT_PLAN_PROMPT, max_output_tokens=4000)

    # The stored plan, beside its analysis.
    stored = off_loop(get_report_plan, host.storage, analysis)
    assert stored is not None and stored.report_input_digest == report_input_digest(prepared.report_input)
    plan = report_plan_of(stored)

    # The campaign's page, and its report_read, are that plan.
    page = off_loop(campaign_report, host, campaign_id, TOYHOST_SCOPE)
    assert (page.basis, page.analysis_id) == ("analysis", analysis.id)
    assert page.plan_page is not None
    page_text = {"answer": page.plan_page.headline, "summary": page.plan_page.lede}
    read = off_loop(report_read, host, campaign_id, TOYHOST_SCOPE, format="json")
    assert (read.analysis_id, read.analysis_basis, read.plan) == (analysis.id, "plan", stored.plan)

    # Frozen through the operation Discodon calls, with a reader's verdict on the judged headline.
    judged_headline = render(plan.headline, prepared.report_input, {})
    receipt = off_loop(
        reporter_case_freeze,
        replace(host, attests_person=attests_reader),
        ReporterCaseFreeze.model_validate(
            {
                "template_id": reporter_template().id,
                "campaign_id": campaign_id,
                "recorded_analysis_id": analysis.id,
                "labels": [
                    {"dimension": GROUNDED, "direction": "high", "quote": "it checks", "memo_excerpt": judged_headline}
                ],
            }
        ),
        TOYHOST_SCOPE,
        collected_by=READER,
    )
    assert (receipt.kind_version, receipt.writer_message_check, receipt.limits) == (PLAN_CASE_VERSION, "verified", [])
    stored_case = off_loop(host.storage.load_test_case, receipt.test_case_id, TOYHOST_SCOPE)
    assert stored_case is not None
    case = reporter_case_of(stored_case)
    assert case is not None and case.recorded_memo is not None
    frozen_input = ReportInput.model_validate(case.report_input)
    frozen_plan = ReportPlan.model_validate(case.recorded_plan)

    # One evidence throughout: the frozen input is the one the plan was written against, over both aspects.
    assert report_input_digest(frozen_input) == stored.report_input_digest == case.report_input_digest
    assert frozen_plan == plan
    # The judged text is the plan rendered against the frozen input, and its headline is the page's.
    assert case.recorded_memo == render_plan_for_judge(frozen_plan, frozen_input)
    assert _judged_json(case.recorded_memo)["headline"] == page_text["answer"] == judged_headline
    assert _judged_json(case.recorded_memo)["lede"] == page_text["summary"]
    # Every number in it comes from a key resolved against the frozen input: the plan types no digit of its own and
    # every key resolves (the checks that refuse a typed number and an unresolvable key pass on the frozen pair).
    validate_plan(frozen_plan, frozen_input)
    # The judge's material is what the planner read of that input, byte for byte, by its recorded digest.
    assert case.writer_message == plan_user_message(frozen_input)
    assert user_message_digest(case.writer_message) == stored.generation.user_message_digest
    output = await AsRecordedReporterKind().invoke(
        PreparedReporter(model=AS_RECORDED_MODEL, span_window=_FakeSpans()), stored_case, FakeCellSink()
    )
    assert output.judge_evidence is not None
    assert (output.judge_evidence.case_material, output.judge_evidence.artifact) == (
        case.writer_message,
        case.recorded_memo,
    )
