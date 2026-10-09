"""Repeating a finished run's judge scores: the measurement the ``separation`` tier reads, priced before it is paid for.

Every cell here runs through the real runner and the real storage, so the evidence a repeat reads is the one
the runner wrote, and the first scores it repeats are the ones the judge phase stored.

What is pinned:

* a repeat sends each scored dim's prompt exactly as the first judge was sent it, records the first score
  beside the answer, and never changes the result's scores;
* every call is priced and admitted against the out-of-run cap BEFORE the first is made, parse retries
  included — at the cap it starts, a cent under it nothing is called and nothing is ledgered;
* an unpriceable call under an enforced cap is refused, and with no cap enforced it is made and ledgered;
* each call made is a ledger row under purpose ``judge``, stamped with the run;
* the estimate answers exactly what the repeat would do, making no call;
* what cannot be reproduced is refused (the run) or named and left out (a result), before any spend;
* a repeat agreeing with twenty first scores puts the dimension's judge on ``separation``; two results
  repeated ten times, or a judge declining a third of its repeats, does not;
* a write race keeps the other writer's change; a result that cannot be read back is reported unwritten
  and the report still returns; an account refusal stops the repeat before any later call.

Mutations that turn this file red (each made in a scratch copy, the file restored from it, 2026-10-06):

- ``repeat_judge_scores``: dropping the up-front admission loop (every call is then refused as unadmitted);
- ``_prepare``: planning one attempt per dim instead of ``JUDGE_CALL_ATTEMPTS`` (the retry test's second
  call is refused as unadmitted);
- ``_BudgetedJudgeClient.generate``: calling the host's client directly (no ledger rows);
- ``_record``: writing the repeat's scores over the result's (the scores-untouched assertion); returning on
  a conflict instead of re-reading (the conflict test's repeat is not stored); re-sending the copy read first
  with a fresh etag (the other writer's repeat is lost); reading the result outside the guard (the report never
  returns);
- ``repeat_judge_scores``: not breaking on an account refusal (a call is made after it).
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field
from typing import Any

import pytest

from threetears.evals.analysis import JudgeKey, judge_agreement, judge_evidence_tiers, judge_self_agreement
from threetears.evals.contracts import EvalStorage
from threetears.evals.contracts.candidate_kind import CandidateOutput, CellSink, CellSpanWindow, VariantConfig
from threetears.evals.contracts.cassettes import CellCassettes
from threetears.evals.contracts.errors import ConflictError, StorageError, ValidationFailedError
from threetears.evals.contracts.evidence_tiers import SEPARATION_MIN_RESULTS
from threetears.evals.contracts.host.eval_host import EvalHost
from threetears.evals.contracts.identity import resolve_variant_identity
from threetears.evals.contracts.models import (
    OUTCOME_DIM_ID,
    TRANSCRIPT_DIM_ID,
    EvalResult,
    EvalTemplate,
    EvalTestCase,
    JudgedArtifact,
    JudgeEvidence,
    JudgeRepeat,
    RepeatedScore,
    RubricDim,
    RubricScore,
)
from threetears.evals.contracts.provider import ProviderFailure, withhold_failure_detail
from threetears.evals.run import estimate_judge_repeat, repeat_judge_scores
from threetears.evals.run.judge import CANNOT_TELL, JUDGE_CALL_ATTEMPTS, JUDGE_REQUEST_SETTINGS
from threetears.evals.run.judge_service import JudgeService
from threetears.evals.run.launch import build_judge_service
from threetears.evals.run.runner import RunnerOptions, run_one_result
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.fixtures.toyhost.host import toyhost_host

_SCOPE = "scope-repeat"
_KIND = "repeat-probe"
_JUDGE = "judge/scripted"
_DIM = "doc.faithful"
_CEILING = 0.01
#: The judge every scored dim here is read under: the run's one dim, served by the scripted judge, no config.
_KEY = JudgeKey(_DIM, "ordinal", _JUDGE, None)

_EVIDENCE = JudgeEvidence(
    subject="The game master. Rules as written.",
    case_material="GM ONLY: the third flagstone is a trap.",
    artifact="Player: I check the floor.\nGM: Roll Perception.",
)


@dataclass(frozen=True)
class _Completion:
    content: str
    stop_reason: str = "end_turn"
    input_tokens: int = 10
    output_tokens: int = 5
    reasoning_tokens: int | None = None
    cost_usd: float | None = 0.001
    model: str = _JUDGE
    served_model: str | None = _JUDGE


@dataclass
class _PricedJudge:
    """A judge client that prices every call at ``ceiling`` and scores each case by its variation.

    The score for a case is read off the ``party=`` value its prompt carries, so twenty cases can carry
    twenty first scores across the scale — what makes an agreement's kappa defined.
    """

    ceiling: float | None = _CEILING
    #: Added to every score (clamped to the scale) — a judge that disagrees with its earlier self.
    drift: int = 0
    #: Dims answered unparseable text once, then scored — the case a parse retry exists for.
    unparseable_once: set[str] = field(default_factory=set)
    cannot_tell: set[str] = field(default_factory=set)
    #: Parties (cases) answered "can't tell" on every dim.
    cannot_tell_parties: set[int] = field(default_factory=set)
    #: Parties whose call raises — the account behind the judge refusing it, as the host describes it.
    refused_parties: set[int] = field(default_factory=set)
    calls: list[tuple[str, str, str]] = field(default_factory=list)
    model_name: str = _JUDGE

    def price_ceiling(self, *, system: str, user: str, response_format: Any = None) -> float | None:
        """The most one call costs: ``ceiling``, whatever the prompt."""
        return self.ceiling

    async def generate(self, *, system: str, user: str, response_format: Any = None) -> _Completion:
        """Record the call and answer it."""
        dim = re.search(r'the single key "(.+?)"', system).group(1)  # type: ignore[union-attr]
        self.calls.append((dim, system, user))
        if dim in self.unparseable_once:
            self.unparseable_once.discard(dim)
            return _Completion(content="not json")
        party = int(re.search(r"party: p(\d+)", user).group(1))  # type: ignore[union-attr]
        if party in self.refused_parties:
            raise _AccountRefused("out of credit")
        declines = dim in self.cannot_tell or party in self.cannot_tell_parties
        score: Any = CANNOT_TELL if declines else max(1, min(5, 2 + party % 4 + self.drift))
        return _Completion(content=json.dumps({"reasoning": "read it", "criteria_scores": {dim: score}}))

    async def aclose(self) -> None:
        """Nothing to release."""


class _Kind:
    """A kind rendering :data:`_EVIDENCE` for its one output, under ``judged_artifact``."""

    def __init__(self, judged_artifact: JudgedArtifact) -> None:
        self.judged_artifact = judged_artifact

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


class _AccountRefused(Exception):
    """What the scripted judge raises for a refused party."""


def _describe_account_refusal(exc: BaseException) -> ProviderFailure:
    """The host's describer: :class:`_AccountRefused` is a refusal for the account, anything else is not."""
    return ProviderFailure(
        f"{type(exc).__name__}", payload_withheld=True, account_refused=isinstance(exc, _AccountRefused)
    )


#: What the other writer appends while the repeat is between its read and its write.
_OTHER_WRITERS_REPEAT = JudgeRepeat(
    judge_model="judge/other-writer",
    scores=[
        RepeatedScore(
            dim=_DIM,
            scale="ordinal",
            first_score=1,
            first_served_model="judge/other-writer",
            first_judge_config_id=None,
            repeat=RubricScore(dim=_DIM, scale="ordinal", score=1, served_model="judge/other-writer"),
        )
    ],
)


class _ConflictOnce(EvalStorage):
    """The real store, except that another writer changes the result between the repeat's read and its first write.

    The other writer's change really lands — it appends its own repeat — so a re-send of the stale copy would
    delete it, and only a re-read that re-applies the repeat to the newer result keeps both.
    """

    def __init__(self) -> None:
        super().__init__(InMemoryDocumentStore())
        self.conflicts = 0

    def replace_eval_result(self, result: EvalResult, /, *, if_match: str | None) -> None:
        """On the first rewrite, let another writer change the result first, then refuse this one as the lost race."""
        if self.conflicts == 0:
            self.conflicts += 1
            current, etag = self.load_eval_result_with_etag(result.id, result.scope_id)
            assert current is not None
            super().replace_eval_result(
                current.model_copy(update={"judge_repeats": [*current.judge_repeats, _OTHER_WRITERS_REPEAT]}),
                if_match=etag,
            )
            raise ConflictError("another writer got there first")
        super().replace_eval_result(result, if_match=if_match)


class _UnreadableOnce(EvalStorage):
    """The real store, except that reading ``unreadable`` back with its etag fails in the backend."""

    def __init__(self) -> None:
        super().__init__(InMemoryDocumentStore())
        self.unreadable: str | None = None

    def load_eval_result_with_etag(self, result_id: str, scope_id: str) -> tuple[EvalResult | None, str | None]:
        """Fail for the chosen result; read as the store does otherwise."""
        if result_id == self.unreadable:
            raise StorageError("the backend timed out")
        return super().load_eval_result_with_etag(result_id, scope_id)


async def _judged_run(
    judge: _PricedJudge,
    *,
    cases: int = 2,
    judged_artifact: JudgedArtifact = JudgedArtifact.DOCUMENT,
    storage: EvalStorage | None = None,
    failure_describer: Any = None,
    **run_fields: Any,
) -> tuple[EvalHost, str]:
    """Run ``cases`` judged cells through the real runner and store them as a finished run would.

    Returns:
        The host holding the records, and the run's id.
    """
    host = toyhost_host(storage=storage, clients=lambda role, model, *, temperature=None: judge)
    if failure_describer is not None:
        host = dataclasses.replace(host, failure_describer=failure_describer)
    template = EvalTemplate(
        scope_id=_SCOPE,
        name="repeat",
        intent="run a fair encounter",
        candidate_kind=_KIND,
        rubric=[RubricDim(name=_DIM, description="the GM rules what the dice say", scale="ordinal")],
    )
    host.storage.save_template(template)
    test_cases = [
        EvalTestCase(template_id=template.id, scope_id=_SCOPE, variation_params={"party": f"p{index}"})
        for index in range(cases)
    ]
    for case in test_cases:
        host.storage.save_test_case(case)
    built = build_judge_service(host, template, _JUDGE, judged_artifact=judged_artifact)
    run = make_eval_run(
        scope_id=_SCOPE,
        template_id=template.id,
        candidate_kind=_KIND,
        candidate_model="candidate/m",
        test_case_ids=[case.id for case in test_cases],
        judge_model=_JUDGE,
        **{
            "status": "completed",
            "effective_judges": built.effective_judges,
            "judge_request_settings": JUDGE_REQUEST_SETTINGS,
            **run_fields,
        },
    )
    host.storage.save_eval_run(run)
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
                client_factory=lambda _model, _temperature: judge, failure_describer=withhold_failure_detail
            ),
            judge_model=_JUDGE,
            options=RunnerOptions(candidate_kinds={_KIND: lambda _cell: _Kind(judged_artifact)}),
            variant=resolve_variant_identity(run=run, profile=host.profile),
        )
        host.storage.save_eval_result(outcome.result, outcome.trace)
    return host, run.id


def _results(host: EvalHost, run_id: str) -> list[EvalResult]:
    return sorted(host.storage.query_eval_results_by_run(run_id, _SCOPE), key=lambda result: result.id)


def _total(cases: int, dims: int = 1) -> float:
    """What a repeat of ``cases`` results of ``dims`` scored dims each is priced at: every attempt, at the ceiling."""
    return cases * dims * JUDGE_CALL_ATTEMPTS * _CEILING


class TestARepeatMeasuresTheJudgeAndChangesNothing:
    async def test_each_scored_dim_is_asked_again_exactly_as_first_asked_and_recorded_beside_its_score(self):
        judge = _PricedJudge()
        host, run_id = await _judged_run(judge)
        before = _results(host, run_id)
        first_prompts = sorted((dim, user) for dim, _, user in judge.calls)
        judge.calls.clear()
        judge.drift = -1

        report = await repeat_judge_scores(host, run_id, _SCOPE, out_of_run_cap_usd=1.0)

        assert sorted((dim, user) for dim, _, user in judge.calls) == first_prompts
        after = _results(host, run_id)
        assert sorted(report.repeated) == [result.id for result in before]
        for old, new in zip(before, after, strict=True):
            assert new.rubric_scores == old.rubric_scores, "a repeat never rewrites the scores it measures"
            ((entry,),) = [repeat.scores for repeat in new.judge_repeats]
            first = old.judge_score(_DIM)
            assert first is not None and entry.repeat is not None
            assert (entry.dim, entry.first_score, entry.first_served_model) == (_DIM, first.score, _JUDGE)
            assert entry.repeat.score == max(1, first.score - 1)
        assert (report.scores_repeated, report.scores_unanswered) == (2, 0)

    async def test_a_dim_the_judge_could_not_tell_on_is_not_repeated(self):
        judge = _PricedJudge(cannot_tell={_DIM})
        host, run_id = await _judged_run(judge, judged_artifact=JudgedArtifact.TRANSCRIPT)
        judge.calls.clear()
        judge.cannot_tell.clear()

        await repeat_judge_scores(host, run_id, _SCOPE, out_of_run_cap_usd=1.0)

        assert {dim for dim, _, _ in judge.calls} == {TRANSCRIPT_DIM_ID, OUTCOME_DIM_ID}

    async def test_every_call_is_a_ledger_row_under_judge_stamped_with_the_run(self):
        judge = _PricedJudge()
        host, run_id = await _judged_run(judge)
        judge.calls.clear()

        report = await repeat_judge_scores(host, run_id, _SCOPE, out_of_run_cap_usd=1.0)

        rows = host.storage.query_out_of_run_spend(_SCOPE, purpose="judge")
        assert len(rows) == len(judge.calls) == report.calls_made == 2
        assert {(row.run_id, row.purpose, row.outcome, row.priced_ceiling_usd) for row in rows} == {
            (run_id, "judge", "completed", _CEILING)
        }
        assert report.cost_usd == pytest.approx(sum(row.cost_usd or 0 for row in rows))

    async def test_a_parse_retry_is_priced_and_ledgered_too(self):
        judge = _PricedJudge()
        host, run_id = await _judged_run(judge, cases=1)
        judge.calls.clear()
        judge.unparseable_once = {_DIM}

        report = await repeat_judge_scores(host, run_id, _SCOPE, out_of_run_cap_usd=_total(1))

        assert len(judge.calls) == 2, "the retry was made — which it can only be when it was admitted"
        assert len(host.storage.query_out_of_run_spend(_SCOPE, purpose="judge")) == 2
        assert report.scores_repeated == 1

    async def test_a_repeat_losing_a_write_race_is_re_applied_to_the_newer_result(self):
        judge = _PricedJudge()
        storage = _ConflictOnce()
        host, run_id = await _judged_run(judge, cases=1, storage=storage)

        report = await repeat_judge_scores(host, run_id, _SCOPE, out_of_run_cap_usd=1.0)

        assert storage.conflicts == 1
        assert report.unwritten == [] and len(report.repeated) == 1
        (result,) = _results(host, run_id)
        # Both writers' changes stand: the other writer's repeat, then this one re-applied on top of it.
        assert [repeat.judge_model for repeat in result.judge_repeats] == ["judge/other-writer", _JUDGE]


class TestWhatWasPaidForIsReported:
    async def test_a_result_that_cannot_be_read_back_is_unwritten_and_the_report_still_returns(self):
        judge = _PricedJudge()
        storage = _UnreadableOnce()
        host, run_id = await _judged_run(judge, cases=2, storage=storage)
        unreadable, readable = _results(host, run_id)
        storage.unreadable = unreadable.id
        judge.calls.clear()

        report = await repeat_judge_scores(host, run_id, _SCOPE, out_of_run_cap_usd=1.0)

        assert report.unwritten == [unreadable.id]
        assert report.repeated == [readable.id]
        assert report.calls_made == len(judge.calls) == 2, "both paid calls are on the report"

    async def test_an_account_refusal_stops_the_repeat_before_any_later_call(self):
        judge = _PricedJudge()
        host, run_id = await _judged_run(judge, cases=3, failure_describer=_describe_account_refusal)
        order = [planned_id for planned_id in (r.id for r in host.storage.query_eval_results_by_run(run_id, _SCOPE))]
        refused_party = 1
        judge.refused_parties = {refused_party}
        judge.calls.clear()

        report = await repeat_judge_scores(host, run_id, _SCOPE, out_of_run_cap_usd=1.0)

        assert report.stopped is not None and "refused" in report.stopped
        parties_called = [int(re.search(r"party: p(\d+)", user).group(1)) for _, _, user in judge.calls]  # type: ignore[union-attr]
        assert parties_called[-1] == refused_party, "nothing is called after the refusal"
        repeated_up_to_refusal = order[: parties_called.index(refused_party) + 1]
        assert report.repeated == repeated_up_to_refusal, "the refused result's answer is recorded; nothing after it"
        assert len(report.repeated) < 3, "the refusal must come before the last result, or the stop is vacuous"
        assert report.calls_made == len(judge.calls)


class TestNothingIsPaidForThatTheCapWouldRefuse:
    async def test_at_the_cap_every_call_is_admitted(self):
        judge = _PricedJudge()
        host, run_id = await _judged_run(judge, cases=3)
        judge.calls.clear()

        report = await repeat_judge_scores(host, run_id, _SCOPE, out_of_run_cap_usd=_total(3))

        assert len(report.repeated) == 3 and len(judge.calls) == 3

    async def test_a_cent_under_it_nothing_is_called_or_ledgered_or_stored(self):
        judge = _PricedJudge()
        host, run_id = await _judged_run(judge, cases=3)
        judge.calls.clear()

        with pytest.raises(ValidationFailedError, match="above the out-of-run cap"):
            await repeat_judge_scores(host, run_id, _SCOPE, out_of_run_cap_usd=_total(3) - 0.01)

        assert judge.calls == []
        assert host.storage.query_out_of_run_spend(_SCOPE) == []
        assert all(result.judge_repeats == [] for result in _results(host, run_id))

    async def test_an_unpriceable_call_under_an_enforced_cap_is_refused_before_any_call(self):
        judge = _PricedJudge()
        host, run_id = await _judged_run(judge)
        judge.calls.clear()
        judge.ceiling = None

        with pytest.raises(ValidationFailedError, match="cannot be priced before they are made"):
            await repeat_judge_scores(host, run_id, _SCOPE, out_of_run_cap_usd=100.0)
        assert judge.calls == []

    async def test_with_no_cap_enforced_an_unpriceable_call_is_made_and_ledgered(self):
        judge = _PricedJudge()
        host, run_id = await _judged_run(judge)
        judge.calls.clear()
        judge.ceiling = None

        report = await repeat_judge_scores(host, run_id, _SCOPE, out_of_run_cap_usd=None)

        rows = host.storage.query_out_of_run_spend(_SCOPE, purpose="judge")
        assert len(rows) == 2 and {(row.priced_ceiling_usd, row.cap_usd) for row in rows} == {(None, None)}
        assert report.cap_usd is None


class TestTheEstimateIsTheRepeatsOwnAnswer:
    @pytest.mark.parametrize(("cap_delta", "starts"), [(0.0, True), (-0.01, False)])
    async def test_it_prices_every_attempt_and_makes_no_call(self, cap_delta: float, starts: bool):
        judge = _PricedJudge()
        host, run_id = await _judged_run(judge, cases=3)
        judge.calls.clear()

        estimate = await estimate_judge_repeat(host, run_id, _SCOPE, out_of_run_cap_usd=_total(3) + cap_delta)

        assert judge.calls == [] and host.storage.query_out_of_run_spend(_SCOPE) == []
        assert (estimate.results, estimate.dims, estimate.max_calls) == (3, 3, 3 * JUDGE_CALL_ATTEMPTS)
        assert estimate.ceiling_usd == pytest.approx(_total(3))
        assert estimate.would_start is starts
        assert (estimate.refusal is None) is starts

    async def test_an_unpriceable_call_has_no_ceiling(self):
        judge = _PricedJudge(ceiling=None)
        host, run_id = await _judged_run(judge)

        estimate = await estimate_judge_repeat(host, run_id, _SCOPE, out_of_run_cap_usd=None)

        assert estimate.ceiling_usd is None and estimate.would_start


class TestWhatCannotBeReproducedIsRefusedBeforeAnySpend:
    async def test_a_run_still_running_is_refused(self):
        judge = _PricedJudge()
        host, run_id = await _judged_run(judge, status="running")
        judge.calls.clear()

        with pytest.raises(ValidationFailedError, match="still being written"):
            await repeat_judge_scores(host, run_id, _SCOPE, out_of_run_cap_usd=1.0)
        assert judge.calls == []

    async def test_a_run_that_recorded_no_request_settings_is_refused(self):
        judge = _PricedJudge()
        host, run_id = await _judged_run(judge, judge_request_settings=None)
        judge.calls.clear()

        with pytest.raises(ValidationFailedError, match="recorded no judge request settings"):
            await repeat_judge_scores(host, run_id, _SCOPE, out_of_run_cap_usd=1.0)
        assert judge.calls == []

    async def test_a_result_without_stored_evidence_is_named_and_left_out(self):
        judge = _PricedJudge()
        host, run_id = await _judged_run(judge)
        bare, kept = _results(host, run_id)
        trace = host.storage.load_eval_trace(bare.id, _SCOPE)
        assert trace is not None and trace.judge_evidence is not None
        host.storage.save_eval_result(bare, trace.model_copy(update={"judge_evidence": None, "judged_artifact": None}))
        judge.calls.clear()

        report = await repeat_judge_scores(host, run_id, _SCOPE, out_of_run_cap_usd=1.0)

        assert report.repeated == [kept.id]
        (skip,) = report.skipped
        assert skip.result_id == bare.id and "stores no judge evidence" in skip.reason

    async def test_a_result_outside_the_run_is_refused(self):
        judge = _PricedJudge()
        host, run_id = await _judged_run(judge)

        with pytest.raises(ValidationFailedError, match="are not results of run"):
            await repeat_judge_scores(host, run_id, _SCOPE, out_of_run_cap_usd=1.0, result_ids=["elsewhere"])


class TestAgreeingRepeatsSeparateTheJudge:
    async def test_the_stored_repeats_decide_separation(self):
        # Separation's floor: 120 results.
        judge = _PricedJudge()
        host, run_id = await _judged_run(judge, cases=SEPARATION_MIN_RESULTS)

        await repeat_judge_scores(host, run_id, _SCOPE, out_of_run_cap_usd=3.0)

        results = _results(host, run_id)
        (tier,) = judge_evidence_tiers(judge_agreement([], results), judge_self_agreement(results), {_KEY})
        assert (tier.separation.n, tier.separation.agreement, tier.tier) == (SEPARATION_MIN_RESULTS, 1.0, "separation")

    async def test_twenty_agreeing_repeats_are_short_of_the_floor_and_say_by_how_much(self):
        judge = _PricedJudge()
        host, run_id = await _judged_run(judge, cases=20)

        await repeat_judge_scores(host, run_id, _SCOPE, out_of_run_cap_usd=1.0)

        results = _results(host, run_id)
        (tier,) = judge_evidence_tiers(judge_agreement([], results), judge_self_agreement(results), {_KEY})
        assert (tier.separation.agreement, tier.separation.state, tier.tier) == (1.0, "insufficient", "undetermined")
        assert tier.separation.results_needed == SEPARATION_MIN_RESULTS - 20

    async def test_nineteen_repeats_do_not(self):
        judge = _PricedJudge()
        host, run_id = await _judged_run(judge, cases=20)
        ids = [result.id for result in _results(host, run_id)][:19]

        await repeat_judge_scores(host, run_id, _SCOPE, out_of_run_cap_usd=1.0, result_ids=ids)

        results = _results(host, run_id)
        (tier,) = judge_evidence_tiers(judge_agreement([], results), judge_self_agreement(results), {_KEY})
        assert (tier.separation.state, tier.tier) == ("insufficient", "undetermined")

    async def test_two_results_repeated_ten_times_do_not(self):
        # Twenty agreeing pairs about two results — the shape that reached separation when the floor counted pairs.
        judge = _PricedJudge()
        host, run_id = await _judged_run(judge, cases=4)
        first, *rest = _results(host, run_id)
        other = next(r for r in rest if r.judge_score(_DIM) != first.judge_score(_DIM))

        for _ in range(10):
            await repeat_judge_scores(host, run_id, _SCOPE, out_of_run_cap_usd=1.0, result_ids=[first.id, other.id])

        results = _results(host, run_id)
        (tier,) = judge_evidence_tiers(judge_agreement([], results), judge_self_agreement(results), {_KEY})
        assert (tier.separation.n, tier.separation.results, tier.separation.agreement) == (20, 2, 1.0)
        assert (tier.separation.state, tier.tier) == ("insufficient", "undetermined")

    async def test_a_judge_declining_a_third_of_its_repeats_does_not(self):
        judge = _PricedJudge()
        host, run_id = await _judged_run(judge, cases=SEPARATION_MIN_RESULTS)
        judge.cannot_tell_parties = set(range(SEPARATION_MIN_RESULTS // 3))

        report = await repeat_judge_scores(host, run_id, _SCOPE, out_of_run_cap_usd=3.0)

        third = SEPARATION_MIN_RESULTS // 3
        assert (report.scores_repeated, report.scores_unanswered) == (SEPARATION_MIN_RESULTS - third, third)
        results = _results(host, run_id)
        read = judge_self_agreement(results)
        (dimension,) = read.dimensions
        assert (dimension.n, dimension.results, dimension.n_cannot_tell) == (
            SEPARATION_MIN_RESULTS,
            SEPARATION_MIN_RESULTS,
            third,
        )
        (tier,) = judge_evidence_tiers(judge_agreement([], results), read, {_KEY})
        assert (tier.separation.state, tier.tier) == ("not_met", "undetermined")
