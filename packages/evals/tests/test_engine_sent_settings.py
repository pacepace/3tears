"""What each judge and simulator call was SENT with is recorded, pooled on and disclosed, never assumed from the ask.

A run stamps what the engine ASKED each apparatus role for (``judge_asked_settings``, ``simulator_asked_settings``).
A client may send the closest setting a model accepts, resend a strict call without enforcement, or say nothing, so
what each call carried is a fact of its own, reported per completion (``sent_settings``) and stored where the call's
output lands. Host-neutral: every client here is a test double, and no provider is called.

* **Recorded.** A judge call's report lands on its score (``RubricScore.judge_sent_settings``) and its usage; a
  repeat, a second judge and a judge-kind case carry the first score's; an out-of-run ledger row stores its own.
  Absent is unverified (``None``), never refused; a wrong-typed member is the host's defect.
* **Pooled on.** ``judge_sent_settings`` and ``simulator_sent_settings`` are apparatus beside the asked dimensions:
  cells pool only when asked AND sent match, unverified never pools with verified, a mixed run only with an identical
  mixture (the six rows of the design's D3 table).
* **A judge sent differently is another judge.** :class:`~threetears.evals.analysis.JudgeKey` carries it; a repeat
  sent differently is unpaired (``settings_changed``); a tier names what its judge was sent.
* **Disclosed.** One sentence, per role, where asked and sent differ, where a call is unverified, where settings went
  unenforced, and where a run's calls were sent more than one way, with counts. Never about the candidate.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Self

import pytest

from threetears.evals.analysis import inter_judge_agreement, judge_drift, judge_self_agreement, tier_sentence
from threetears.evals.analysis.agreement import (
    judge_agreement,
    judge_evidence_tiers,
    judge_key,
    sent_settings_from_level,
    sent_settings_key,
    sent_settings_level,
)
from threetears.evals.analysis.cells import apparatus_class_of
from threetears.evals.analysis.reporting import (
    RequestSettingsReading,
    request_settings_disclosure,
    request_settings_readings,
)
from threetears.evals.kernel.host.sweepables import SHARED_CORE
from threetears.evals.kernel.judge_cases import JudgeTrial
from threetears.evals.kernel.out_of_run import OutOfRunBudget, PlannedCall
from threetears.evals import Eval, Judge
from threetears.evals.quick import as_host
from threetears.evals.run.judge import JUDGE_REQUEST_SETTINGS, run_judge_llm
from threetears.evals.schema import CompletionProtocolDefect, RubricDim
from threetears.evals.schema.hashing import canonical_json
from threetears.evals.schema.models import (
    CalibrationRating,
    ClientRequestSettings,
    EvalResult,
    EvalRun,
    JudgeRepeat,
    RepeatedScore,
    RoleUsage,
    RubricScore,
    SecondJudge,
    SecondJudgeScore,
    SecondJudging,
    request_settings_words,
)
from threetears.evals.schema.out_of_run_spend import OutOfRunSpend
from packages.evals.tests.factories import make_eval_result, make_eval_run
from packages.evals.tests.off_loop import off_loop

#: What the engine asks its judge for (a reasoning budget), and what a client serving a words-only model sends instead.
_ASKED = JUDGE_REQUEST_SETTINGS
_WORD = ClientRequestSettings(max_tokens=10240, reasoning_effort="high", strict_output=True)
#: The same ask, resent without enforcement after a router could not place it.
_UNENFORCED = ClientRequestSettings(max_tokens=10240, reasoning_max_tokens=8192, strict_output=False)
_DIM = "answer.helpful"
_JUDGE = "vendor/judge-2026-04"


# ---------------------------------------------------------------------------
# Recorded: the judge call, its pairs, the judge kind, the out-of-run ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Reply:
    """A judge completion that reports its temperature and, optionally, what its request carried."""

    content: str
    input_tokens: int | None = 1
    output_tokens: int | None = 1
    cost_usd: float | None = 0.25
    price_source: str | None = "provider"
    model: str = _JUDGE
    served_model: str | None = _JUDGE
    reasoning_tokens: int | None = None
    stop_reason: str = "end_turn"
    temperature: float | None = 0.0


@dataclass(frozen=True)
class _ReportingReply(_Reply):
    sent_settings: Any = None


def _scored(dim: str) -> str:
    return json.dumps({"reasoning": "ok", "criteria_scores": {dim: 4}})


# parity-with: threetears.evals.schema.CompletionClient
class _FakeJudge:
    """A judge client answering each call with the next of ``replies`` (the last one repeats)."""

    model_name = _JUDGE

    def __init__(self, replies: Sequence[_Reply]) -> None:
        self.replies = list(replies)
        self.calls = 0

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        if reply.content == "<score>":
            match = re.search(r'single key "([^"]+)"', system)
            reply = replace(reply, content=_scored(match[1] if match is not None else "d"))
        return reply

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


async def _judge(client: _FakeJudge) -> dict[str, Any]:
    parsed = await run_judge_llm(
        client=client,
        system_prompt='Score the single key "d".',
        user_prompt="u",
        criteria_dicts=[{"name": "d", "weight": 1.0, "scale": "ordinal"}],
        label="Judge",
        case_id="c",
        cannot_tell_offered=True,
    )
    assert parsed is not None
    return parsed


class TestTheJudgeCallRecordsWhatItWasSent:
    async def test_a_reported_send_lands_on_the_reply_and_its_usage(self) -> None:
        parsed = await _judge(_FakeJudge([_ReportingReply(_scored("d"), sent_settings=_WORD)]))

        assert parsed["judge_sent_settings"] == _WORD
        assert parsed["judge_usage"].sent_settings == _WORD

    async def test_a_client_that_says_nothing_is_unverified_never_refused(self) -> None:
        parsed = await _judge(_FakeJudge([_Reply(_scored("d"))]))

        assert "error" not in parsed
        assert parsed["judge_sent_settings"] is None
        assert parsed["judge_usage"].sent_settings is None

    async def test_attempts_sent_differently_leave_the_usage_unverified_and_the_score_its_own(self) -> None:
        """One usage record cannot name two values; the score is the parsing attempt's."""
        parsed = await _judge(
            _FakeJudge(
                [
                    _ReportingReply("not json", sent_settings=_ASKED),
                    _ReportingReply(_scored("d"), sent_settings=_WORD),
                ]
            )
        )

        assert parsed["judge_sent_settings"] == _WORD
        assert (parsed["judge_usage"].calls, parsed["judge_usage"].sent_settings) == (2, None)

    async def test_a_wrong_typed_member_is_the_hosts_defect_carrying_the_spend(self) -> None:
        with pytest.raises(CompletionProtocolDefect, match="sent_settings") as raised:
            await _judge(_FakeJudge([_ReportingReply(_scored("d"), sent_settings={"max_tokens": 10})]))

        assert (raised.value.calls, raised.value.cost_usd) == (1, 0.25)

    async def test_a_judged_run_stores_it_on_every_score_and_discloses_the_gap(self) -> None:
        judge = Judge(
            _FakeJudge([_ReportingReply("<score>", sent_settings=_WORD)]),
            [RubricDim(name="answer.helpful", description="It helps.", scale="ordinal")],
        )
        ev = Eval([{"q": "a"}, {"q": "b"}], _answer, [judge], k=1)
        host = as_host([ev], store=":memory:")
        summary = (await ev.run(host=host, scope="sent-settings")).summary

        results = off_loop(lambda: host.storage.query_eval_results_by_run(summary.run_id, summary.scope_id))
        assert [score.judge_sent_settings for result in results for score in result.rubric_scores] == [_WORD] * 2
        (judge,) = summary.request_settings
        assert (judge.role, judge.state, judge.n_not_as_asked, judge.n_total) == ("judge", "sent_otherwise", 2, 2)
        assert judge.sent_display() == request_settings_words(_WORD)
        assert summary.request_settings_warning is not None
        assert summary.render().count("Request settings not confirmed as asked:") == 1


class TestThePairsCarryTheFirstScoresSend:
    def test_a_judge_kind_trial_records_it_beside_its_temperature(self) -> None:
        assert "judge_sent_settings" in JudgeTrial.model_fields

    async def test_an_out_of_run_row_stores_what_its_call_reported(self) -> None:
        rows: list[OutOfRunSpend] = []
        budget = OutOfRunBudget(store=_FakeLedger(rows), scope_id="s", cap_usd=None, blocking_executor=None)
        client = _FakePricedJudge(_ReportingReply("{}", sent_settings=_WORD))

        (admitted,) = budget.admit(client, "judge", [PlannedCall(system="s", user="u")])
        await budget.generate(client, admitted)

        assert [(row.sent_settings, row.unreadable) for row in rows] == [(_WORD, [])]

    async def test_an_out_of_run_row_whose_member_is_mistyped_is_kept_and_names_it_unreadable(self) -> None:
        rows: list[OutOfRunSpend] = []
        budget = OutOfRunBudget(store=_FakeLedger(rows), scope_id="s", cap_usd=None, blocking_executor=None)
        client = _FakePricedJudge(_ReportingReply("{}", sent_settings="high"))

        (admitted,) = budget.admit(client, "judge", [PlannedCall(system="s", user="u")])
        await budget.generate(client, admitted)

        assert [(row.sent_settings, row.unreadable) for row in rows] == [(None, ["sent_settings"])]


# parity-with: threetears.evals.schema.out_of_run_spend.OutOfRunSpendStore
class _FakeLedger:
    def __init__(self, rows: list[OutOfRunSpend]) -> None:
        self.rows = rows

    def save_out_of_run_spend(self, spend: OutOfRunSpend, /) -> None:
        self.rows.append(spend)


# parity-with: threetears.evals.schema.completion.PricedCompletion
class _FakePricedJudge:
    model_name = _JUDGE

    def __init__(self, reply: _Reply) -> None:
        self.reply = reply

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        return 0.5

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
        return self.reply


async def _answer(case: Mapping[str, Any]) -> str:
    """Answer in a word."""
    return str(case["q"])


# ---------------------------------------------------------------------------
# A judge sent differently is another judge
# ---------------------------------------------------------------------------


def _score(sent: ClientRequestSettings | None, score: int = 4) -> RubricScore:
    return RubricScore(
        dim=_DIM,
        score=score,
        scale="ordinal",
        axis="capability",
        served_model=_JUDGE,
        judge_temperature=0.0,
        judge_sent_settings=sent,
    )


def _repeated(first: ClientRequestSettings | None, again: ClientRequestSettings | None) -> EvalResult:
    entry = RepeatedScore(
        dim=_DIM,
        scale="ordinal",
        first_score=4,
        first_served_model=_JUDGE,
        first_judge_config_id=None,
        first_judge_temperature=0.0,
        first_judge_sent_settings=first,
        repeat=_score(again),
    )
    return make_eval_result(
        id=f"res-{id(entry)}",
        rubric_scores=[_score(first)],
        judge_repeats=[JudgeRepeat(judge_model=_JUDGE, scores=[entry])],
    )


class TestAJudgeSentDifferentlyIsAnotherJudge:
    def test_the_key_carries_the_send(self) -> None:
        verified = judge_key(make_eval_result(rubric_scores=[_score(_ASKED)]), _DIM)
        budget = judge_key(make_eval_result(rubric_scores=[_score(_WORD)]), _DIM)
        unverified = judge_key(make_eval_result(rubric_scores=[_score(None)]), _DIM)

        assert verified is not None and budget is not None and unverified is not None
        assert verified.judge_sent_settings == sent_settings_key(_ASKED)
        assert len({verified, budget, unverified}) == 3

    @pytest.mark.parametrize(
        ("first", "again", "paired"),
        [
            (_ASKED, _ASKED, True),
            (None, None, True),
            (_ASKED, _WORD, False),
            (_ASKED, None, False),
            (None, _ASKED, False),
        ],
    )
    def test_a_repeat_sent_differently_is_unpaired(
        self, first: ClientRequestSettings | None, again: ClientRequestSettings | None, paired: bool
    ) -> None:
        agreement = judge_self_agreement([_repeated(first, again)])

        if paired:
            (dimension,) = agreement.dimensions
            assert (dimension.n, dimension.judge_sent_settings, agreement.unpaired) == (1, first, [])
        else:
            assert agreement.dimensions == []
            assert [entry.reason for entry in agreement.unpaired] == ["settings_changed"]

    def test_agreement_with_people_is_read_per_send(self) -> None:
        results = [
            make_eval_result(id="r-1", rubric_scores=[_score(_ASKED)]),
            make_eval_result(id="r-2", rubric_scores=[_score(_WORD)]),
        ]
        ratings = [
            CalibrationRating(
                scope_id="uni-1",
                run_id="run-1",
                result_id=result.id,
                rubric_dim=_DIM,
                rater="reviewer",
                rater_kind="person",
                scale="ordinal",
                score=4,
                reason="it helps",
                rated_withheld="not rebuilt in this test",
            )
            for result in results
        ]

        sent = [dimension.judge_sent_settings for dimension in judge_agreement(ratings, results).dimensions]
        assert len(sent) == 2 and _ASKED in sent and _WORD in sent

    def test_a_tier_names_what_its_judge_was_sent_or_that_it_is_unconfirmed(self) -> None:
        empty = judge_agreement([], [])
        alone = judge_self_agreement([])
        budget_judge = judge_key(make_eval_result(rubric_scores=[_score(_WORD)]), _DIM)
        silent_judge = judge_key(make_eval_result(rubric_scores=[_score(None)]), _DIM)
        assert budget_judge is not None and silent_judge is not None
        (sent,) = judge_evidence_tiers(empty, alone, [budget_judge])
        (silent,) = judge_evidence_tiers(empty, alone, [silent_judge])

        assert f", sent {request_settings_words(_WORD)}" in tier_sentence(sent)
        assert ", request settings unconfirmed" in tier_sentence(silent)


def _seconded(result_id: str, case: str, pass_id: str, sent: ClientRequestSettings | None, score: int) -> EvalResult:
    """A result whose first score was sent as asked, carrying one second-judge pass whose score was sent ``sent``."""
    second = RubricScore(
        dim=_DIM,
        scale="ordinal",
        score=score,
        served_model="judge/second",
        axis="capability",
        judge_temperature=0.0,
        judge_sent_settings=sent,
    )
    entry = SecondJudgeScore(
        dim=_DIM,
        scale="ordinal",
        first_score=4,
        first_served_model=_JUDGE,
        first_judge_config_id=None,
        first_judge_temperature=0.0,
        first_judge_sent_settings=_ASKED,
        second=second,
    )
    judging = SecondJudging(
        pass_id=pass_id,
        judge=SecondJudge(model="judge/second"),
        sample_fraction=1.0,
        sample_seed=0,
        scores=[entry],
    )
    return make_eval_result(id=result_id, test_case_id=case, rubric_scores=[_score(_ASKED)], judge_seconds=[judging])


#: Two passes of one second judge, asked alike: the first sent as asked, the second sent a budget instead.
_TWO_SECOND_SENDS = [
    _seconded("r-1", "case-1", "pass-asked", _ASKED, 4),
    _seconded("r-2", "case-2", "pass-asked", _ASKED, 3),
    _seconded("r-3", "case-1", "pass-budget", _WORD, 2),
    _seconded("r-4", "case-2", "pass-budget", _WORD, 2),
    _seconded("r-5", "case-1", "pass-silent", None, 5),
]


class TestASecondJudgeSentDifferentlyIsAnotherSecondJudge:
    def test_inter_judge_agreement_keys_and_names_the_second_judges_send(self) -> None:
        dimensions = inter_judge_agreement(_TWO_SECOND_SENDS).dimensions

        named = {tuple(d.passes): (d.second_sent_settings, d.n) for d in dimensions}
        assert named == {
            ("pass-asked",): ([_ASKED], 2),
            ("pass-budget",): ([_WORD], 2),
            ("pass-silent",): (None, 1),
        }

    def test_drift_keys_and_names_the_second_judges_send(self) -> None:
        dimensions = judge_drift(_TWO_SECOND_SENDS).dimensions

        sent = [
            (None if d.second_sent_settings is None else [request_settings_words(x) for x in d.second_sent_settings])
            for d in dimensions
        ]
        assert [d.n_pairs for d in dimensions] == [1, 2, 2], "unverified first, then each send in canonical order"
        assert sent[0] is None
        assert sorted(sent[1:]) == sorted([[request_settings_words(_ASKED)], [request_settings_words(_WORD)]])

    def test_a_pass_reads_as_the_level_the_bundle_names_it_by(self) -> None:
        """One reading names a pass's send for the bundle and for both readers, so a linked drift groups as its level."""
        for result in _TWO_SECOND_SENDS:
            (judging,) = result.judge_seconds
            level = sent_settings_level(entry.second for entry in judging.scores)
            (dimension,) = judge_drift([result]).dimensions
            assert dimension.second_sent_settings == sent_settings_from_level(level)


# ---------------------------------------------------------------------------
# Pooled on: the D3 rows
# ---------------------------------------------------------------------------

_JUDGE_DIMENSIONS = ("judge_request_settings", "judge_sent_settings")
_SIMULATOR_DIMENSIONS = ("simulator_request_settings", "simulator_sent_settings")


def _class_id(run: EvalRun, results: Sequence[EvalResult], dimensions: Sequence[str]) -> str:
    """The apparatus class the run's results fall in over ``dimensions``, read as the bundle reads a rig."""
    values = SHARED_CORE.read_all(run, results)
    recorded = {
        name: None if SHARED_CORE.is_indeterminate(name, values[name]) else canonical_json(values[name])
        for name in dimensions
    }
    return apparatus_class_of(recorded, dimensions=dimensions, provenance=run.apparatus_provenance).apparatus_class_id


def _judged(asked: ClientRequestSettings, *sent: ClientRequestSettings | None) -> tuple[EvalRun, list[EvalResult]]:
    run = make_eval_run(judge_model=_JUDGE, judge_asked_settings=asked)
    return run, [
        make_eval_result(id=f"res-{i}", eval_run_id=run.id, rubric_scores=[_score(value)])
        for i, value in enumerate(sent)
    ]


def _pools(a: tuple[EvalRun, list[EvalResult]], b: tuple[EvalRun, list[EvalResult]]) -> bool:
    return _class_id(*a, _JUDGE_DIMENSIONS) == _class_id(*b, _JUDGE_DIMENSIONS)


class TestCellsPoolOnlyWhenAskedAndSentMatch:
    def test_asked_alike_and_sent_alike_pool(self) -> None:
        assert _pools(_judged(_ASKED, _WORD), _judged(_ASKED, _WORD))

    def test_asked_alike_and_sent_differently_do_not(self) -> None:
        assert not _pools(_judged(_ASKED, _ASKED), _judged(_ASKED, _WORD))

    def test_asked_differently_and_sent_alike_do_not(self) -> None:
        assert not _pools(_judged(_ASKED, _ASKED), _judged(_WORD, _ASKED))

    def test_asked_alike_and_unverified_on_both_sides_pool_carrying_the_gap(self) -> None:
        run, results = _judged(_ASKED, None)
        values = SHARED_CORE.read_all(run, results)
        assert values["judge_sent_settings"] is None
        assert _pools(_judged(_ASKED, None), _judged(_ASKED, None))

    def test_unverified_never_pools_with_verified(self) -> None:
        assert not _pools(_judged(_ASKED, _ASKED), _judged(_ASKED, None))
        assert not _pools(_judged(_ASKED, _ASKED, None), _judged(_ASKED, _ASKED))

    def test_asked_differently_with_one_unverified_does_not(self) -> None:
        assert not _pools(_judged(_ASKED, None), _judged(_WORD, _ASKED))

    def test_a_mixed_run_pools_only_with_an_identical_mixture(self) -> None:
        mixed = _judged(_ASKED, _ASKED, _WORD)

        assert _pools(mixed, _judged(_ASKED, _WORD, _ASKED, _WORD))
        assert not _pools(mixed, _judged(_ASKED, _ASKED))
        assert not _pools(mixed, _judged(_ASKED, _ASKED, _UNENFORCED))

    def test_the_simulator_pools_on_its_usage_rows_by_the_same_rule(self) -> None:
        def simulated(*sent: list[ClientRequestSettings] | None) -> tuple[EvalRun, list[EvalResult]]:
            run = make_eval_run(simulator_model="vendor/sim", simulator_asked_settings=_ASKED)
            rows = [
                RoleUsage(role="simulator", model="vendor/sim", call_count=1, sent_settings=value) for value in sent
            ]
            return run, [make_eval_result(usage=rows)]

        def pools(a: tuple[EvalRun, list[EvalResult]], b: tuple[EvalRun, list[EvalResult]]) -> bool:
            return _class_id(*a, _SIMULATOR_DIMENSIONS) == _class_id(*b, _SIMULATOR_DIMENSIONS)

        assert pools(simulated([_ASKED]), simulated([_ASKED], [_ASKED]))
        assert not pools(simulated([_ASKED]), simulated([_UNENFORCED]))
        assert not pools(simulated([_ASKED]), simulated([_ASKED], None))


# ---------------------------------------------------------------------------
# Disclosed: one sentence, per role, each case
# ---------------------------------------------------------------------------


def _disclosed(run: EvalRun, results: list[EvalResult]) -> tuple[list[RequestSettingsReading], str | None]:
    readings = request_settings_readings([run], results)
    return readings, request_settings_disclosure(readings)


class TestTheDisclosure:
    def test_sent_as_asked_says_nothing(self) -> None:
        readings, said = _disclosed(*_judged(_ASKED, _ASKED, _ASKED))

        assert [reading.state for reading in readings] == ["as_asked"]
        assert said is None

    def test_sent_other_than_asked_names_both(self) -> None:
        readings, said = _disclosed(*_judged(_ASKED, _WORD, _WORD))

        assert readings[0].state == "sent_otherwise"
        assert said == (
            f"Request settings not confirmed as asked: the judge was asked for {request_settings_words(_ASKED)} and "
            f"sent {request_settings_words(_WORD)} on 2 of 2 score(s). A client may send the closest setting a model "
            "accepts, or resend a call without enforcement, and reports what each request carried (sent_settings); "
            "runs sent differently, or one reported and one not, are not pooled as one apparatus."
        )

    def test_unverified_says_the_scores_cannot_say(self) -> None:
        readings, said = _disclosed(*_judged(_ASKED, _ASKED, None))

        assert (readings[0].state, readings[0].sent_display()) == (
            "unverified",
            "unverified (1 of 2 score(s) not reported)",
        )
        assert said is not None
        assert (
            "the judge's client did not report what it sent for 1 of 2 score(s), so those cannot say whether what "
            f"was asked ({request_settings_words(_ASKED)}) was applied"
        ) in said

    def test_unenforced_says_the_provider_may_have_ignored_settings(self) -> None:
        _, said = _disclosed(*_judged(_ASKED, _UNENFORCED))

        assert said is not None
        assert (
            "the judge's settings were sent without enforcement on 1 of 1 score(s); the provider may have ignored any "
            "the serving model does not take"
        ) in said

    def test_a_mixed_run_names_each_way_with_its_count(self) -> None:
        readings, said = _disclosed(*_judged(_ASKED, _ASKED, _ASKED, _WORD))

        (reading,) = readings
        assert reading.state == "mixed"
        assert sorted((share.count, share.settings == _ASKED) for share in reading.sent) == [(1, False), (2, True)]
        assert reading.sent_display().startswith("mixed: ")
        assert said is not None
        assert "the judge's calls were sent two ways: " in said
        # In canonical order (the effort word's JSON sorts before the budget's null word), the unit named once.
        assert (
            f"the judge's calls were sent two ways: 1 score(s) with {request_settings_words(_WORD)}, 2 with "
            f"{request_settings_words(_ASKED)} (asked for {request_settings_words(_ASKED)})"
        ) in said

    def test_the_simulator_is_read_off_its_usage_rows(self) -> None:
        run = make_eval_run(simulator_model="vendor/sim", simulator_asked_settings=_ASKED)
        result = make_eval_result(
            rubric_scores=[],
            usage=[RoleUsage(role="simulator", model="vendor/sim", call_count=3, sent_settings=[_UNENFORCED])],
        )

        readings, said = _disclosed(run, [result])

        assert [(reading.role, reading.state) for reading in readings] == [("simulator", "sent_otherwise")]
        assert said is not None and "the simulated user was asked for" in said and "on 1 of 1 result(s)" in said

    def test_the_candidate_is_never_read(self) -> None:
        """A host builds its candidate's client itself: no sentence may say its settings were checked."""
        result = make_eval_result(
            rubric_scores=[],
            usage=[RoleUsage(role="candidate", model="sonnet", call_count=1, sent_settings=[_WORD])],
        )

        readings, said = _disclosed(make_eval_run(), [result])

        assert readings == [] and said is None

    def test_a_score_made_outside_a_judge_phase_is_not_a_judge_call(self) -> None:
        result = make_eval_result(
            rubric_scores=[RubricScore(dim=_DIM, score=4, scale="ordinal", axis="capability", judge_temperature=None)]
        )

        assert request_settings_readings([make_eval_run(judge_model=_JUDGE)], [result]) == []
