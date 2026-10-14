"""A judge reply cut off at its output cap is an exclusion with cause ``truncated``, carrying the cap it was sent.

Before this, a judge reply the provider stopped at ``max_tokens`` became ``judge_error = "<dim>: ... cut short ..."``,
read as ``infra`` like any fault of the rig, and reached the report as "faulted": the word "truncated" and the cap
appeared nowhere but inside a free-text string, and an arm whose long answers ran its judge out of room lost trials
nobody could see were lost differentially. Now the cut is a fact on the result (``judge_cut_short``: the stop reason,
the settings the call was SENT with, the token counts), the trial is excluded as ``truncated``, the bundle counts
each arm's exclusions by cause with the caps the truncated calls were sent, and flags an arm losing more than 10% of
its trials to one cause. The reply is never retried or bought again.

The judge's reasoning is asked as a bound now (``reasoning_max_tokens`` under a derived cap), which a model taking
budgets enforces, and which a model taking effort words only is sent as the same word it was sent before.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

import pytest

from threetears.evals.analysis import assemble_context_bundle, get_campaign
from threetears.evals.kernel.result_condition import (
    DIFFERENTIAL_EXCLUSION_SHARE,
    ResultOutcome,
    classify_result,
    judge_truncated_dims,
    trial_exclusion,
)
from threetears.evals.negotiation import SettingFacts, closest_settings, wire_settings_for
from threetears.evals import Eval, Judge
from threetears.evals.run import JUDGE_REQUEST_SETTINGS, list_results
from threetears.evals.run.judge import (
    JUDGE_ANSWER_BUDGET_TOKENS,
    JUDGE_REASONING_EFFORT,
    JUDGE_REASONING_HEADROOM_TOKENS,
    run_judge_llm,
)
from threetears.evals.schema import ClientRequestSettings, JudgeCutShort, RubricDim, StopReason
from packages.evals.tests.factories import make_eval_result
from packages.evals.tests.off_loop import off_loop

DIM = "answer.helpful"
CAP = JUDGE_REQUEST_SETTINGS.max_tokens
CUT = JudgeCutShort(
    stop_reason="max_tokens", sent_settings=JUDGE_REQUEST_SETTINGS, output_tokens=CAP, reasoning_tokens=9918
)


@dataclass(frozen=True)
class _Reply:
    """One judge completion, with what its request was sent with."""

    content: str
    stop_reason: StopReason = "end_turn"
    output_tokens: int | None = 20
    reasoning_tokens: int | None = None
    sent_settings: ClientRequestSettings | None = None
    input_tokens: int | None = 50
    cost_usd: float | None = None
    price_source: str | None = None
    model: str = "scripted/judge"
    served_model: str | None = "scripted/judge"
    temperature: float | None = None


# parity-with: threetears.evals.schema.CompletionClient
class _FakeCuttingJudge:
    """A judge that runs out of room on every answer saying LONG, and scores every other answer 4."""

    model_name = "scripted/judge"

    def __init__(self, *, sent: ClientRequestSettings | None = JUDGE_REQUEST_SETTINGS) -> None:
        self.calls: list[str] = []
        self._sent = sent

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
        output = user.split("# Output under review\n", 1)[1] if "# Output under review\n" in user else user
        self.calls.append(output)
        if "LONG" in output:
            return _Reply(
                content='{"reasoning": "The answer runs',
                stop_reason="max_tokens",
                output_tokens=CAP,
                reasoning_tokens=9918,
                sent_settings=self._sent,
            )
        match = re.search(r'single key "([^"]+)"', system)
        name = match[1] if match else DIM
        return _Reply(json.dumps({"reasoning": "fine", "criteria_scores": {name: "4"}}), sent_settings=self._sent)

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


# --- the call: the cut is a fact, reported once ------------------------------------------------------------------


async def test_a_cut_off_reply_carries_its_stop_reason_and_sent_cap_and_is_not_retried() -> None:
    judge = _FakeCuttingJudge()

    reply = await run_judge_llm(
        client=judge,
        system_prompt="judge",
        user_prompt="# Output under review\nLONG",
        criteria_dicts=[{"name": DIM, "weight": 1.0}],
        label="EvalRubric",
        case_id="c-1",
    )

    assert reply is not None and "cut short by the provider and was not retried" in reply["error"]
    assert reply["judge_cut_short"] == CUT
    assert len(judge.calls) == 1, "a cut-off reply is never bought again"


# --- the exclusion: truncated, below a candidate failure and a harness fault -------------------------------------


def _cut_result(cut: JudgeCutShort = CUT, **overrides: Any) -> Any:
    return make_eval_result(
        rubric_scores=[], judge_error=f"{DIM}: EvalRubric reply was cut short", judge_cut_short={DIM: cut}, **overrides
    )


def test_a_dim_cut_off_at_its_cap_excludes_the_trial_as_truncated() -> None:
    result = _cut_result()

    assert classify_result(result) is ResultOutcome.INFRA_EXCLUDE
    assert trial_exclusion(result) == "truncated"
    assert result.judge_cut_short[DIM].sent_settings is not None
    assert result.judge_cut_short[DIM].sent_settings.max_tokens == CAP


def test_a_candidate_failure_outranks_a_cut_off_judge() -> None:
    result = _cut_result(candidate_error="the model returned 500")

    assert classify_result(result) is ResultOutcome.CANDIDATE_FAIL and trial_exclusion(result) is None


def test_a_harness_fault_outranks_a_cut_off_judge() -> None:
    assert trial_exclusion(_cut_result(infra_error="the simulator failed")) == "infra"


def test_a_filtered_reply_is_recorded_and_stays_a_fault_of_the_rig() -> None:
    filtered = CUT.model_copy(update={"stop_reason": "content_filter"})
    result = _cut_result(filtered)

    assert trial_exclusion(result) == "infra" and judge_truncated_dims(result) == []


def test_a_stale_entry_beside_a_rescored_dim_is_not_read() -> None:
    """An older build's re-judge scores the dim and clears judge_error without knowing the field."""
    rescored = make_eval_result(judge_cut_short={"conversation.tone": CUT})

    assert rescored.judge_error is None and trial_exclusion(rescored) is None
    assert judge_truncated_dims(rescored.model_copy(update={"judge_error": "another dim failed"})) == []


def test_end_turn_is_not_a_cut() -> None:
    with pytest.raises(ValueError, match="was not cut short"):
        JudgeCutShort(stop_reason="end_turn", sent_settings=None, output_tokens=None, reasoning_tokens=None)


# --- the bundle: exclusions per arm by cause, the cap, and the differential flag ---------------------------------

_CASES = [{"n": index} for index in range(10)]


def _challenger(long_cases: int) -> Any:
    async def answer(case: Mapping[str, Any]) -> str:
        return "LONG " * 5 if case["n"] < long_cases else "short"

    return answer


async def _control(case: Mapping[str, Any]) -> str:
    return "short"


async def _compare(long_cases: int, judge: _FakeCuttingJudge) -> Any:
    # Scored 1-5, as the judge answers every answer it can finish.
    rubric = [RubricDim(name=DIM, description="It helps.", scale="ordinal")]
    comparison = await Eval(
        _CASES, {"control": _control, "challenger": _challenger(long_cases)}, [Judge(judge, rubric)], k=1
    ).run(store=":memory:")
    campaign = off_loop(get_campaign, comparison.host.storage, comparison.campaign_id, comparison.scope_id)
    bundle = off_loop(
        assemble_context_bundle, campaign, storage=comparison.host.storage, profile=comparison.host.profile
    )
    return comparison, bundle


async def test_a_challenger_whose_judge_runs_out_of_room_is_counted_truncated_and_flagged() -> None:
    judge = _FakeCuttingJudge()
    comparison, bundle = await _compare(7, judge)
    challenger = comparison.arm_variants["challenger"]
    control = comparison.arm_variants["control"]

    by_arm = {arm.variant_key: arm for arm in bundle.exclusions}
    assert by_arm[challenger].by_cause == {"truncated": 7}
    assert (by_arm[challenger].n_trials, by_arm[challenger].n_usable) == (10, 3)
    assert by_arm[challenger].truncated_sent_caps == [CAP] and not by_arm[challenger].truncated_cap_unreported
    assert by_arm[control].by_cause == {}

    (flag,) = bundle.differential_exclusions
    assert (flag.variant_key, flag.cause, flag.share) == (challenger, "truncated", pytest.approx(0.7))
    assert flag.threshold == DIFFERENTIAL_EXCLUSION_SHARE and flag.other_arms == {control: 0.0}

    assert sum("LONG" in output for output in judge.calls) == 7, "each cut-off call was made once"
    results = off_loop(list_results, comparison.host.storage, comparison.arms["challenger"].run_id, comparison.scope_id)
    truncated = [result for result in results if trial_exclusion(result) == "truncated"]
    assert len(truncated) == 7
    assert all(result.judge_cut_short[DIM].sent_settings == JUDGE_REQUEST_SETTINGS for result in truncated)


async def test_one_cut_in_ten_is_counted_and_not_flagged() -> None:
    comparison, bundle = await _compare(1, _FakeCuttingJudge())

    by_arm = {arm.variant_key: arm for arm in bundle.exclusions}
    assert by_arm[comparison.arm_variants["challenger"]].by_cause == {"truncated": 1}
    assert bundle.differential_exclusions == []


async def test_a_cap_the_client_did_not_report_is_said_unreported_not_guessed() -> None:
    comparison, bundle = await _compare(2, _FakeCuttingJudge(sent=None))

    (arm,) = [arm for arm in bundle.exclusions if arm.variant_key == comparison.arm_variants["challenger"]]
    assert arm.truncated_sent_caps == [] and arm.truncated_cap_unreported


# --- the request: reasoning asked as a bound under a derived cap -------------------------------------------------


def test_the_judge_asks_its_reasoning_as_a_budget_under_a_cap_one_answer_above_it() -> None:
    assert JUDGE_REQUEST_SETTINGS.reasoning_max_tokens == JUDGE_REASONING_HEADROOM_TOKENS == 8192
    assert JUDGE_REQUEST_SETTINGS.reasoning_effort is None
    assert JUDGE_REQUEST_SETTINGS.max_tokens == JUDGE_REASONING_HEADROOM_TOKENS + JUDGE_ANSWER_BUDGET_TOKENS == 10240
    assert JUDGE_REQUEST_SETTINGS.strict_output


@pytest.mark.parametrize("family", ["openai", "router"])
def test_a_model_taking_words_only_is_sent_the_word_it_was_sent_before(family: Any) -> None:
    """Wire-neutral where words are all a model takes: 8192 of 10240 is 0.80, which is `high`."""
    sent, _notes = closest_settings(wire_settings_for(JUDGE_REQUEST_SETTINGS, 0.0), SettingFacts(), family)

    assert (sent.reasoning_effort, sent.reasoning_max_tokens, sent.max_tokens) == (JUDGE_REASONING_EFFORT, None, CAP)


def test_an_anthropic_model_taking_a_budget_is_sent_the_bound() -> None:
    sent, _notes = closest_settings(
        wire_settings_for(JUDGE_REQUEST_SETTINGS, 0.0), SettingFacts(reasoning_budget=True), "anthropic"
    )

    assert (sent.reasoning_max_tokens, sent.reasoning_effort, sent.max_tokens) == (8192, None, CAP)
    assert sent.temperature is None, "Anthropic takes no temperature beside thinking"


def test_an_anthropic_model_taking_no_budget_is_sent_the_word_under_the_same_cap() -> None:
    sent, _notes = closest_settings(
        wire_settings_for(JUDGE_REQUEST_SETTINGS, 0.0), SettingFacts(reasoning_budget=False), "anthropic"
    )

    assert (sent.reasoning_effort, sent.reasoning_max_tokens, sent.max_tokens) == (JUDGE_REASONING_EFFORT, None, CAP)
