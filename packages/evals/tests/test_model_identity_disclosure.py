"""Discodon #2025: a run whose record cannot say which model answered says so, once, on every surface.

A run records the model id each call asked for, and a provider may answer an id with a model of its choosing (a
floating alias resolves on its side). Only the response names the model that answered — the candidate's usage rows
and each judge score carry it as ``served_model`` — and where the completion client reported none, the requested id
is all the record holds. The package never refuses such a run and never guesses which ids float; it discloses:

* the run's summary (``run_get``), ``run_summary``, the analysis bundle and the campaign report each say once that
  which model answered is unconfirmed, in one sentence;
* a run whose every response named its model says nothing, and neither does one whose candidate called nothing;
* a run some of whose calls were confirmed and some not counts precisely which, per role.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from threetears.evals.analysis import (
    DisclosureBlock,
    EvalSummary,
    Report,
    UnconfirmedModelIdentity,
    assemble_context_bundle,
    build_code_only_report,
    build_report,
    report_markdown,
    run_summary,
    summarize_run,
)
from threetears.evals.analysis.reporting import model_identity_disclosure, unconfirmed_model_identities
from threetears.evals.kernel import EvalCampaign, EvalStorage
from threetears.evals.quick import Answer, callable_host
from threetears.evals.quick.answer import unwrap_answer
from threetears.evals.schema import EvalResult, EvalRun, RoleUsage, RubricScore
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_eval_result, make_eval_run
from packages.evals.tests.fixtures.toyhost.corpus import (
    TOYHOST_SCOPE,
    TOYHOST_SUBJECT,
    ToyhostStorage,
    toyhost_batch,
    toyhost_measurements,
)
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.report_support import toy_report

_SCOPE = "identity-scope"
_RUN = "run-identity"
#: What the launch asked for, and the model a response named as having answered it.
_ALIAS = "~vendor/model-latest"
_SERVED = "vendor/model-2026-05"
_JUDGE = "vendor/judge-latest"
_JUDGE_SERVED = "vendor/judge-2026-04"
_DIM = "answer.helpful"

_HEADLINE = "Model identity unconfirmed:"


def _result(case: str, *, served: str | None, judge_served: str | None = _JUDGE_SERVED) -> EvalResult:
    """One result whose candidate call's response named ``served``, and whose one score's named ``judge_served``."""
    return make_eval_result(
        id=f"res-{case}",
        scope_id=_SCOPE,
        eval_run_id=_RUN,
        test_case_id=case,
        model=_ALIAS,
        usage=[RoleUsage(role="candidate", model=_ALIAS, served_model=served, cost_usd=0.01, call_count=1)],
        rubric_scores=[RubricScore(dim=_DIM, score=4, scale="ordinal", served_model=judge_served)],
    )


def _run(results: list[EvalResult]) -> EvalRun:
    return make_eval_run(
        id=_RUN,
        scope_id=_SCOPE,
        candidate_model=_ALIAS,
        judge_model=_JUDGE,
        effective_judges={_DIM: _JUDGE},
        effective_judges_source="recorded",
        test_case_ids=sorted({result.test_case_id for result in results}),
        status="completed",
    )


def _summary(results: list[EvalResult]) -> EvalSummary:
    storage = EvalStorage(InMemoryDocumentStore())
    storage.save_eval_run(_run(results))
    for result in results:
        storage.save_eval_result(result)
    return summarize_run(dataclasses.replace(callable_host(), storage=storage), _RUN, _SCOPE)


class TestTheRunSummary:
    def test_a_candidate_whose_responses_named_no_model_is_disclosed_with_the_id_it_asked_for(self) -> None:
        summary = _summary([_result("tc-1", served=None), _result("tc-2", served=None)])

        assert summary.unconfirmed_models == [
            UnconfirmedModelIdentity(role="candidate", requested=[_ALIAS], n_unrecorded=2, n_total=2)
        ]
        assert summary.model_identity_warning == (
            f"{_HEADLINE} the candidate (asked for {_ALIAS}) made calls whose response named no model in 2 of 2 "
            "result(s). The record keeps the model id asked for, and a provider may answer an id with a model of "
            "its choosing (a floating alias), so these numbers cannot say which model produced them; a completion "
            "client that reports the model its provider's response named (served_model) confirms it."
        )

    def test_a_run_whose_every_response_named_its_model_says_nothing(self) -> None:
        summary = _summary([_result("tc-1", served=_SERVED), _result("tc-2", served=_SERVED)])

        assert summary.unconfirmed_models == []
        assert summary.model_identity_warning is None
        assert "unconfirmed" not in summary.render()

    def test_a_mixed_run_counts_precisely_which_calls_and_scores_were_unconfirmed(self) -> None:
        summary = _summary(
            [
                _result("tc-1", served=_SERVED),
                _result("tc-2", served=None, judge_served=None),
                _result("tc-3", served=_SERVED),
            ]
        )

        candidate, judge = summary.unconfirmed_models
        assert (candidate.role, candidate.n_unrecorded, candidate.n_total, candidate.served_models) == (
            "candidate",
            1,
            3,
            [_SERVED],
        )
        assert (judge.role, judge.requested, judge.n_unrecorded, judge.n_total, judge.served_models) == (
            "judge",
            [_JUDGE],
            1,
            3,
            [_JUDGE_SERVED],
        )
        warning = summary.model_identity_warning
        assert warning is not None
        assert (
            f"the candidate (asked for {_ALIAS}) made calls whose response named no model in 1 of 3 result(s) "
            f"(the responses that named one named {_SERVED}); the judge (asked for {_JUDGE}) produced 1 of 3 "
            f"score(s) whose response named no model (the responses that named one named {_JUDGE_SERVED})"
        ) in warning

    def test_a_result_with_one_confirmed_and_one_unconfirmed_call_counts_as_unconfirmed(self) -> None:
        result = _result("tc-1", served=_SERVED).model_copy(
            update={
                "usage": [
                    RoleUsage(role="candidate", model=_ALIAS, served_model=_SERVED, call_count=1),
                    RoleUsage(role="candidate", model=_ALIAS, served_model=None, call_count=1),
                ]
            }
        )

        (candidate,) = _summary([result]).unconfirmed_models
        assert (candidate.n_unrecorded, candidate.n_total, candidate.served_models) == (1, 1, [_SERVED])

    def test_a_candidate_that_called_nothing_says_nothing(self) -> None:
        result = _result("tc-1", served=None).model_copy(update={"usage": []})

        assert _summary([result]).unconfirmed_models == []

    def test_the_rendered_text_says_it_once(self) -> None:
        text = _summary([_result("tc-1", served=None, judge_served=None)]).render()

        assert text.count(_HEADLINE) == 1
        assert "the candidate (asked for" in text and "the judge (asked for" in text

    def test_a_judge_score_whose_run_recorded_no_attribution_names_no_requested_judge(self) -> None:
        """The run's pin is never put in its place: a per-dimension config overrides it."""
        result = _result("tc-1", served=_SERVED, judge_served=None)
        run = _run([result]).model_copy(update={"effective_judges": None})

        (judge,) = unconfirmed_model_identities([run], [result])
        assert (judge.role, judge.requested) == ("judge", [])
        assert "the judge produced 1 of 1 score(s)" in (model_identity_disclosure([judge]) or "")

    def test_an_answer_that_names_its_served_model_confirms_the_candidate(self) -> None:
        _, confirmed = unwrap_answer(Answer("x", model=_ALIAS, served_model=_SERVED, cost_usd=0.01))
        _, unconfirmed = unwrap_answer(Answer("x", model=_ALIAS, cost_usd=0.01))

        assert [row.served_model for row in confirmed.usage] == [_SERVED]
        assert [row.served_model for row in unconfirmed.usage] == [None]


class TestRunSummaryRead:
    def _read(self, results: list[EvalResult]) -> dict[str, Any]:
        run = _run(results)
        return run_summary(
            ToyhostStorage([run], {run.id: results}),
            run.id,
            run.scope_id,
            load_run_listed=lambda _id, _scope: run,
            row_columns=lambda _results: {},
            profile=toyhost_profile(),
        )

    def test_it_carries_the_sentence_where_the_candidate_is_unconfirmed(self) -> None:
        disclosure = self._read([_result("tc-1", served=None)])["model_identity_disclosure"]

        assert disclosure is not None and disclosure.startswith(_HEADLINE)

    def test_it_carries_none_where_every_response_named_its_model(self) -> None:
        assert self._read([_result("tc-1", served=_SERVED)])["model_identity_disclosure"] is None


def _campaign_bundle(served: str | None) -> Any:
    """The toy corpus batch, its candidate rows naming ``served`` and its scores naming a judge."""
    batch = toyhost_batch(chunk_tokens=512, retriever_top_k=3, extraction_schema="v1", ocr_engine_version="tess-5.3.1")
    results = [
        result.model_copy(
            update={
                "usage": [
                    row.model_copy(update={"served_model": served}) if row.role == "candidate" else row
                    for row in result.usage
                ],
                "rubric_scores": [
                    score.model_copy(update={"served_model": _JUDGE_SERVED}) for score in result.rubric_scores
                ],
            }
        )
        for result in toyhost_measurements(
            batch, profile=toyhost_profile(), cost_usd=0.02, total_ms=900.0, field_accuracy=0.8
        )
    ]
    campaign = EvalCampaign(
        id="5f0c9a3e-2b7d-4e41-8c6a-9d3e1f2a7b40",
        scope_id=TOYHOST_SCOPE,
        name="model-identity",
        subject_id=TOYHOST_SUBJECT.subject_id,
        subject_kind="extractor_config",
        behavior="extract_invoice_fields",
        template_id="",
        run_ids=[batch.id],
        created_by="test:fixture",
    )
    storage = ToyhostStorage([batch], {batch.id: results})
    return assemble_context_bundle(campaign, storage=storage, profile=toyhost_profile())


def _said(report: Report) -> list[str]:
    return [block.text for block in report.blocks if isinstance(block, DisclosureBlock) and _HEADLINE in block.text]


class TestTheBundleAndItsCodeOnlyReport:
    def test_the_bundle_says_it_once_and_the_report_renders_it_once(self) -> None:
        bundle = _campaign_bundle(served=None)

        assert bundle.model_identity_disclosure is not None
        assert bundle.model_identity_disclosure.startswith(f"{_HEADLINE} the candidate (asked for ")
        report = build_code_only_report(
            bundle, measures=toyhost_profile().measures, assembled_at="2026-10-12T00:00:00+00:00"
        )
        assert _said(report) == [bundle.model_identity_disclosure]
        assert report_markdown(report).count(_HEADLINE) == 1

    def test_a_campaign_whose_every_response_named_its_model_says_nothing(self) -> None:
        bundle = _campaign_bundle(served=_SERVED)

        assert bundle.model_identity_disclosure is None
        report = build_code_only_report(
            bundle, measures=toyhost_profile().measures, assembled_at="2026-10-12T00:00:00+00:00"
        )
        assert _said(report) == []


class TestTheAnalysisReport:
    async def test_build_report_says_the_disclosure_once_among_the_methods(self) -> None:
        sentence = f"{_HEADLINE} the candidate (asked for {_ALIAS}) made calls whose response named no model."
        _, analysis, _ = await toy_report()

        report = build_report(analysis, model_identity=sentence)

        (block,) = [block for block in report.blocks if isinstance(block, DisclosureBlock) and block.text == sentence]
        assert (block.section, block.source) == ("methods", "measurement")
        assert report_markdown(report).count(_HEADLINE) == 1
        assert _said(build_report(analysis)) == []

    async def test_reading_an_analysis_reads_the_disclosure_off_its_runs_results(self) -> None:
        host, analysis, report = await toy_report()

        def read() -> str | None:
            runs = [host.storage.load_eval_run(run_id, analysis.scope_id) for run_id in analysis.observation_refs]
            results = [
                result
                for run_id in analysis.observation_refs
                for result in host.storage.query_eval_results_by_run(run_id, analysis.scope_id)
            ]
            return model_identity_disclosure(unconfirmed_model_identities([run for run in runs if run], results))

        expected = off_loop(read)

        assert expected is not None, "the toy corpus records no served model, so its report must say so"
        assert _said(report) == [expected]
