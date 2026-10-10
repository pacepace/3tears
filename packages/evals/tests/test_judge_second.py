"""A second judge on a finished run: agreement between judges (#646) and drift across a judge change (#597).

Every cell here runs through the real runner and the real storage, so the evidence a second judge reads is the one
the runner wrote, and the first scores it pairs with are the ones the judge phase stored.

What is pinned:

* a toy-host run with a second judge on half its results reports per-dimension n, exact agreement and kappa beside
  each judged dimension's score (``run_get``'s summary), and the second judge's spend on its own line — never in the
  candidate's ``cost_usd``, never in the run's judge spend;
* the sample is seeded and recorded: the same seed draws the same results;
* an undefined kappa is stated as undefined, with why — never 0;
* a stub judge B shifting one dimension by +1 reads that dimension ``separated`` with an interval excluding 0, and an
  unshifted dimension ``not_separated``; the stored scores are unchanged afterwards;
* every call is priced against the out-of-run cap before the first; a second judge that IS the run's judge is refused;
* an analysis whose arms were judged by different judges names the change, and links the drift reading that spans it.

Mutations that turn this file red (each made in a scratch copy, the file restored from it, 2026-10-10):

- ``ops.summary._judged_dimensions``: dropping ``second_judges`` (the agreement-beside-the-score test);
- ``run.judge_second._prepare``: building the budgeted client under purpose ``judge`` (the spend-line test);
- ``run.judge_second._record``: writing the second scores over the result's (the scores-untouched assertions);
- ``analysis.judge_drift.judge_drift``: pairing per result instead of per case, or dropping the family correction —
  neither moves these toy numbers, which is why the simulation file exists (``test_simulated_judge_drift.py``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import pytest

from threetears.evals.analysis import inter_judge_agreement, judge_drift
from threetears.evals.analysis.agreement import KAPPA_UNDEFINED_ONE_SCORE
from threetears.evals.kernel import EvalCampaign
from threetears.evals.kernel.candidate_kind import CandidateOutput, CellSink, CellSpanWindow, VariantConfig
from threetears.evals.kernel.cassettes import CellCassettes
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.kernel.host.eval_host import EvalHost
from threetears.evals.kernel.identity import resolve_variant_identity
from threetears.evals.schema.models import (
    EvalResult,
    EvalTemplate,
    EvalTestCase,
    JudgedArtifact,
    JudgeEvidence,
    RubricDim,
    SecondJudge,
)
from threetears.evals.kernel.provider import withhold_failure_detail
from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.analysis.summary import summarize_run
from threetears.evals.run import ask_second_judge, estimate_second_judge
from threetears.evals.run.judge import JUDGE_CALL_ATTEMPTS, JUDGE_REQUEST_SETTINGS
from threetears.evals.run.judge_service import JudgeService
from threetears.evals.run.launch import build_judge_service
from threetears.evals.run.runner import RunnerOptions, run_one_result
from threetears.evals.storage import InMemoryDocumentStore
from threetears.evals.kernel import EvalStorage
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.fixtures.toyhost.host import toyhost_host

_SCOPE = "scope-second"
_KIND = "second-probe"
_FIRST = "judge/first"
_SECOND = "judge/second"
_SHIFTED = "doc.faithful"
_STEADY = "doc.clear"
_CEILING = 0.01
_CASES = 20

_EVIDENCE = JudgeEvidence(
    subject="The game master. Rules as written.",
    case_material="GM ONLY: the third flagstone is a trap.",
    artifact="Player: I check the floor.\nGM: Roll Perception.",
)


@dataclass(frozen=True)
class _Completion:
    content: str
    model: str
    served_model: str
    stop_reason: str = "end_turn"
    input_tokens: int = 10
    output_tokens: int = 5
    reasoning_tokens: int | None = None
    cost_usd: float | None = 0.001


@dataclass
class _ScriptedJudge:
    """A judge that scores each case by its party number, shifted per dim — a judge whose answers are known.

    Score ``2 + party % 4 + shift[dim]``, held to 1-5: twenty parties carry first scores across 2-5, so a kappa is
    defined, and a shift of +1 moves fifteen of twenty cases (a case already at 5 stays there).
    """

    model_name: str
    shift: dict[str, int] = field(default_factory=dict)
    #: A score every dim answers with, whatever the case — the judge that makes chance agreement total.
    constant: int | None = None
    calls: list[tuple[str, str]] = field(default_factory=list)
    temperatures: list[float | None] = field(default_factory=list)

    def price_ceiling(self, *, system: str, user: str, response_format: Any = None) -> float | None:
        """The most one call costs."""
        return _CEILING

    async def generate(self, *, system: str, user: str, response_format: Any = None) -> _Completion:
        """Record the call and answer it."""
        dim = re.search(r'the single key "(.+?)"', system).group(1)  # type: ignore[union-attr]
        party = int(re.search(r"party: p(\d+)", user).group(1))  # type: ignore[union-attr]
        self.calls.append((dim, user))
        score = self.constant if self.constant is not None else max(1, min(5, 2 + party % 4 + self.shift.get(dim, 0)))
        body = json.dumps({"reasoning": "read it", "criteria_scores": {dim: score}})
        return _Completion(content=body, model=self.model_name, served_model=self.model_name)

    async def aclose(self) -> None:
        """Nothing to release."""


class _Kind:
    """A kind rendering :data:`_EVIDENCE` for its one output."""

    judged_artifact = JudgedArtifact.DOCUMENT

    async def prepare(
        self,
        *,
        subject_snapshot: Any,
        variant_config: VariantConfig,
        world_seed: Any,
        span_window: CellSpanWindow,
        cassettes: CellCassettes | None,
        world: Any,
    ) -> None:
        """Nothing to build."""

    async def invoke(self, instance: None, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        """Hand back one output and the evidence a judge reads of it."""
        return CandidateOutput(output=[{"rendered": "elsewhere"}], judge_evidence=_EVIDENCE)


@dataclass
class _Judges:
    """The two judges a host serves, by model: the run's, and the second."""

    first: _ScriptedJudge
    second: _ScriptedJudge

    def clients(self, role: str, model: str | None, *, temperature: float | None = None) -> _ScriptedJudge:
        judge = self.second if model == _SECOND else self.first
        judge.temperatures.append(temperature)
        return judge


async def _judged_run(
    judges: _Judges, *, cases: int = _CASES, storage: EvalStorage | None = None, judge_model: str = _FIRST
) -> tuple[EvalHost, str]:
    """Run ``cases`` cells through the real runner, each judged on two dims by ``judge_model``, stored as finished."""
    host = toyhost_host(storage=storage, clients=judges.clients)
    template = EvalTemplate(
        scope_id=_SCOPE,
        name="second",
        intent="run a fair encounter",
        candidate_kind=_KIND,
        rubric=[
            RubricDim(name=_SHIFTED, description="the GM rules what the dice say", scale="ordinal"),
            RubricDim(name=_STEADY, description="the GM is clear", scale="ordinal"),
        ],
    )
    if host.storage.load_template(template.id, _SCOPE) is None:
        host.storage.save_template(template)
    test_cases = [
        EvalTestCase(template_id=template.id, scope_id=_SCOPE, variation_params={"party": f"p{index}"})
        for index in range(cases)
    ]
    for case in test_cases:
        host.storage.save_test_case(case)
    built = build_judge_service(host, template, judge_model, judged_artifact=JudgedArtifact.DOCUMENT)
    run = make_eval_run(
        scope_id=_SCOPE,
        template_id=template.id,
        candidate_kind=_KIND,
        candidate_model="candidate/m",
        test_case_ids=[case.id for case in test_cases],
        judge_model=judge_model,
        status="completed",
        effective_judges=built.effective_judges,
        judge_request_settings=JUDGE_REQUEST_SETTINGS,
    )
    host.storage.save_eval_run(run)
    scorer = judges.second if judge_model == _SECOND else judges.first
    for case in test_cases:
        outcome = await run_one_result(
            host,
            template=template,
            test_case=case,
            subject_id="subject-1",
            model=run.candidate_model,
            k_iteration=1,
            eval_run_id=run.id,
            scope_id=_SCOPE,
            judge_service=JudgeService(
                client_factory=lambda _model, _temperature: scorer, failure_describer=withhold_failure_detail
            ),
            judge_model=judge_model,
            options=RunnerOptions(candidate_kinds={_KIND: lambda _cell: _Kind()}),
            variant=resolve_variant_identity(run=run, profile=host.profile),
        )
        host.storage.save_eval_result(outcome.result, outcome.trace)
    return host, run.id


def _results(host: EvalHost, run_id: str) -> list[EvalResult]:
    return sorted(host.storage.query_eval_results_by_run(run_id, _SCOPE), key=lambda result: result.id)


class TestASecondJudgeOnHalfTheResults:
    """#646's Done-when: per-dimension n, agreement and kappa beside the scores, and the spend on its own line."""

    async def test_agreement_is_reported_beside_each_judged_dimension_and_the_spend_apart(self) -> None:
        judges = _Judges(first=_ScriptedJudge(_FIRST), second=_ScriptedJudge(_SECOND, shift={_SHIFTED: 1}))
        host, run_id = await _judged_run(judges)
        before = _results(host, run_id)
        summary_before = summarize_run(host, run_id, _SCOPE)

        report = await ask_second_judge(
            host, run_id, _SCOPE, judge=SecondJudge(model=_SECOND), out_of_run_cap_usd=10.0, sample_fraction=0.5
        )

        assert len(report.sampled) == _CASES // 2 and sorted(report.judged) == report.sampled
        after = _results(host, run_id)
        for old, new in zip(before, after, strict=True):
            assert new.rubric_scores == old.rubric_scores, "a second judge never rewrites the scores it pairs with"
            assert new.cost_usd == old.cost_usd, "a second judge's spend is never the candidate's"
            assert new.usage == old.usage
        summary = summarize_run(host, run_id, _SCOPE)
        by_name = {dim.name: dim for dim in summary.judged}
        for name in (_SHIFTED, _STEADY):
            (row,) = by_name[name].second_judges
            assert (row.n, row.second_model, row.kappa_weighting) == (_CASES // 2, _SECOND, "quadratic")
            assert row.kappa is not None and row.kappa_undefined is None
        assert by_name[_STEADY].second_judges[0].exact_agreement == 1.0
        assert by_name[_SHIFTED].second_judges[0].exact_agreement < 1.0
        # The spend: its own line, from the ledger, and nowhere in the run's judge spend or the candidate's.
        rows = [row for row in host.storage.query_out_of_run_spend(_SCOPE, purpose="second_judge")]
        assert len(rows) == report.calls_made == (_CASES // 2) * 2
        assert {(row.run_id, row.purpose) for row in rows} == {(run_id, "second_judge")}
        assert summary.second_judge_calls == len(rows)
        assert summary.second_judge_cost_usd == pytest.approx(sum(row.cost_usd or 0 for row in rows))
        assert (summary.judge_calls, summary.judge_cost_usd) == (
            summary_before.judge_calls,
            summary_before.judge_cost_usd,
        )
        assert (summary.candidate_calls, summary.candidate_cost_usd) == (
            summary_before.candidate_calls,
            summary_before.candidate_cost_usd,
        )
        assert host.storage.query_out_of_run_spend(_SCOPE, purpose="judge") == []

    async def test_the_sample_is_seeded_recorded_and_drawn_again_by_the_same_seed(self) -> None:
        judges = _Judges(first=_ScriptedJudge(_FIRST), second=_ScriptedJudge(_SECOND))
        host, run_id = await _judged_run(judges)

        first = await ask_second_judge(
            host, run_id, _SCOPE, judge=SecondJudge(model=_SECOND), out_of_run_cap_usd=10.0, sample_fraction=0.5, seed=7
        )
        again = await estimate_second_judge(
            host, run_id, _SCOPE, judge=SecondJudge(model=_SECOND), out_of_run_cap_usd=10.0, sample_fraction=0.5, seed=7
        )

        assert again.sampled == first.sampled
        for result in _results(host, run_id):
            for judging in result.judge_seconds:
                assert (judging.pass_id, judging.sample_fraction, judging.sample_seed) == (first.pass_id, 0.5, 7)
        assert {result.id for result in _results(host, run_id) if result.judge_seconds} == set(first.sampled)

    async def test_an_undefined_kappa_says_why_and_is_never_zero(self) -> None:
        judges = _Judges(first=_ScriptedJudge(_FIRST, constant=4), second=_ScriptedJudge(_SECOND, constant=4))
        host, run_id = await _judged_run(judges, cases=6)

        report = await ask_second_judge(host, run_id, _SCOPE, judge=SecondJudge(model=_SECOND), out_of_run_cap_usd=10.0)

        agreement = inter_judge_agreement(_results(host, run_id), pass_id=report.pass_id)
        for row in agreement.dimensions:
            assert row.kappa is None and row.kappa_undefined == KAPPA_UNDEFINED_ONE_SCORE
            assert row.exact_agreement == 1.0 and row.agreement_interval is None

    async def test_the_second_judge_is_sent_every_dim_at_its_temperature(self) -> None:
        judges = _Judges(first=_ScriptedJudge(_FIRST), second=_ScriptedJudge(_SECOND))
        host, run_id = await _judged_run(judges, cases=2)

        await ask_second_judge(
            host, run_id, _SCOPE, judge=SecondJudge(model=_SECOND, temperature=0.7), out_of_run_cap_usd=10.0
        )

        assert judges.second.temperatures == [0.7]
        assert {dim for dim, _ in judges.second.calls} == {_SHIFTED, _STEADY}


class TestNothingIsPaidForThatCannotBeRead:
    async def test_a_cap_below_the_price_refuses_the_pass_before_any_call(self) -> None:
        judges = _Judges(first=_ScriptedJudge(_FIRST), second=_ScriptedJudge(_SECOND))
        host, run_id = await _judged_run(judges, cases=2)
        priced = 2 * 2 * JUDGE_CALL_ATTEMPTS * _CEILING

        with pytest.raises(ValidationFailedError):
            await ask_second_judge(
                host, run_id, _SCOPE, judge=SecondJudge(model=_SECOND), out_of_run_cap_usd=priced - 0.001
            )

        assert judges.second.calls == []
        assert host.storage.query_out_of_run_spend(_SCOPE, purpose="second_judge") == []

    async def test_the_runs_own_judge_is_refused_as_a_second_one(self) -> None:
        judges = _Judges(first=_ScriptedJudge(_FIRST), second=_ScriptedJudge(_SECOND))
        host, run_id = await _judged_run(judges, cases=2)

        with pytest.raises(ValidationFailedError, match="judge repeat"):
            await ask_second_judge(host, run_id, _SCOPE, judge=SecondJudge(model=_FIRST), out_of_run_cap_usd=10.0)

    async def test_a_fraction_outside_zero_to_one_is_refused(self) -> None:
        judges = _Judges(first=_ScriptedJudge(_FIRST), second=_ScriptedJudge(_SECOND))
        host, run_id = await _judged_run(judges, cases=2)

        with pytest.raises(ValidationFailedError, match="sample_fraction"):
            await ask_second_judge(
                host, run_id, _SCOPE, judge=SecondJudge(model=_SECOND), out_of_run_cap_usd=10.0, sample_fraction=0.0
            )


class TestDriftAcrossAJudgeChange:
    """#597's Done-when: the shifted dimension moved with an interval excluding 0, the unshifted one not separated."""

    async def test_a_judge_shifting_one_dimension_by_one_is_read_as_moved_and_the_other_as_not_separated(self) -> None:
        judges = _Judges(first=_ScriptedJudge(_FIRST), second=_ScriptedJudge(_SECOND, shift={_SHIFTED: 1}))
        host, run_id = await _judged_run(judges)
        before = _results(host, run_id)

        report = await ask_second_judge(host, run_id, _SCOPE, judge=SecondJudge(model=_SECOND), out_of_run_cap_usd=10.0)

        drift = {row.rubric_dim: row for row in judge_drift(_results(host, run_id), pass_id=report.pass_id).dimensions}
        moved, steady = drift[_SHIFTED], drift[_STEADY]
        assert moved.verdict == "separated" and moved.delta is not None and moved.delta > 0
        assert moved.interval is not None and moved.interval[0] > 0, "the interval on the movement excludes 0"
        assert (moved.n_cases, moved.interval_level) == (_CASES, pytest.approx(1 - 0.05 / 2))
        assert steady.verdict == "not_separated" and steady.delta == 0
        after = _results(host, run_id)
        assert [result.rubric_scores for result in after] == [result.rubric_scores for result in before], (
            "re-scoring under the new judge leaves every stored score as it was"
        )

    async def test_an_analysis_spanning_a_judge_change_names_it_and_links_the_drift_reading(self) -> None:
        judges = _Judges(first=_ScriptedJudge(_FIRST), second=_ScriptedJudge(_SECOND, shift={_SHIFTED: 1}))
        storage = EvalStorage(InMemoryDocumentStore())
        host, before_id = await _judged_run(judges, cases=6, storage=storage)
        _, after_id = await _judged_run(judges, cases=6, storage=storage, judge_model=_SECOND)
        campaign = EvalCampaign(
            scope_id=_SCOPE,
            name="a judge swap",
            subject_id="subject-1",
            subject_kind="s",
            behavior="b",
            run_ids=[before_id, after_id],
            created_by="test:fixture",
        )

        unlinked = assemble_context_bundle(campaign, storage=storage, profile=host.profile).judge_change
        await ask_second_judge(host, before_id, _SCOPE, judge=SecondJudge(model=_SECOND), out_of_run_cap_usd=10.0)
        linked = assemble_context_bundle(campaign, storage=storage, profile=host.profile).judge_change

        assert [level.judge_model for level in unlinked.levels] == [_FIRST, _SECOND]
        assert unlinked.drift_links == [] and unlinked.sentence is not None
        assert "Nothing measured" in unlinked.sentence
        (link,) = linked.drift_links
        assert (link.from_level, link.to_level, link.run_ids) == (0, 1, [before_id])
        moved = {row.rubric_dim: row.verdict for row in link.drift.dimensions}
        assert moved == {_SHIFTED: "separated", _STEADY: "not_separated"}
        assert linked.sentence is not None and "drift reading" in linked.sentence
