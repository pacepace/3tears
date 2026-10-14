"""What a request was asked for and what it was sent with are two facts under two names; the port carries the second.

A completion MAY report the request settings its request actually carried, as the optional ``sent_settings`` member:
not a protocol property, so every host's completion keeps conforming, read by ``sent_settings_of`` alone, absent read
as unverified and never refused, a wrong type refused by name. Pinned here:

- the reader and the optional-member declaration, and the conformance kit's check of the member (its type when
  present, its presence when asked);
- usage capture: one call's record reads it, a ledger row keeps every distinct value its calls reported in canonical
  order, one unverified call makes the row unverified, and the value never splits a row;
- the stored fields named for what they hold (``RubricScore.judge_sent_settings``, ``RoleUsage.sent_settings``, the
  first-score halves, ``OutOfRunSpend.sent_settings``, ``JudgeCaseSource.first_judge_sent_settings``, the run's
  ``judge_asked_settings``/``simulator_asked_settings``), and a judge case's digest untouched while its field is None;
- the engine's own constants, which ask in effort words.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import analysis_gen_request_settings_for
from threetears.evals.analysis.generator import (
    DEFAULT_ANALYSIS_GEN_REASONING_EFFORT,
    DEFAULT_ANALYSIS_GEN_REASONING_HEADROOM_TOKENS,
    MIN_ANALYSIS_GEN_REASONING_HEADROOM_TOKENS,
)
from threetears.evals.kernel.judge_cases import JudgeCase, JudgeCaseSource
from threetears.evals.kernel.usage_capture import CallUsage, RoleUsageLedger
from threetears.evals.run import JUDGE_MAX_TOKENS, JUDGE_REQUEST_SETTINGS
from threetears.evals.run.judge import JUDGE_ANSWER_BUDGET_TOKENS, JUDGE_REASONING_HEADROOM_TOKENS
from threetears.evals.schema import (
    ClientRequestSettings,
    CompletionProtocolDefect,
    EvalRun,
    OutOfRunSpend,
    StopReason,
    sent_settings_of,
)
from threetears.evals.schema.completion import COMPLETION_RESULT_ATTRIBUTES, OPTIONAL_COMPLETION_ATTRIBUTES
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.models import (
    TRANSCRIPT_DIM_ID,
    JudgedArtifact,
    JudgeEvidence,
    RepeatedScore,
    RoleUsage,
    RubricScore,
    SecondJudgeScore,
)
from threetears.evals.testing import CompletionConformanceFailure, check_completion_conformance

from packages.evals.tests.factories import make_eval_run

#: What a client sent in place of a budget it was asked for, and two other values to tell apart.
EFFORT_HIGH = ClientRequestSettings(max_tokens=10240, reasoning_effort="high", strict_output=True)
BUDGET = ClientRequestSettings(max_tokens=10240, reasoning_max_tokens=8192, strict_output=True)
UNENFORCED = ClientRequestSettings(max_tokens=10240, reasoning_effort="high", strict_output=False)


@dataclass(frozen=True, kw_only=True)
class _Completion:
    """A host's completion: the ten protocol members, and no ``sent_settings``."""

    content: str = "ok"
    input_tokens: int | None = 10
    output_tokens: int | None = 20
    cost_usd: float | None = 0.001
    price_source: str | None = "provider:reported"
    model: str = "acme/large"
    served_model: str | None = "acme/large-2026-09"
    reasoning_tokens: int | None = None
    temperature: float | None = 0.0
    stop_reason: StopReason = "end_turn"


@dataclass(frozen=True, kw_only=True)
class _Reporting(_Completion):
    """A host's completion that reports what its request carried."""

    sent_settings: Any = None


# --- the port: an optional member and its one reader ------------------------------------------------------------


def test_sent_settings_is_an_optional_member_never_a_protocol_property() -> None:
    assert OPTIONAL_COMPLETION_ATTRIBUTES == ("sent_settings",)
    assert not set(OPTIONAL_COMPLETION_ATTRIBUTES) & set(COMPLETION_RESULT_ATTRIBUTES)


@pytest.mark.parametrize("completion", [_Completion(), _Reporting(sent_settings=None)], ids=["absent", "none"])
def test_a_completion_that_does_not_say_what_it_sent_reads_as_unverified(completion: object) -> None:
    assert sent_settings_of(completion) is None


def test_a_completion_that_says_what_it_sent_reads_as_that() -> None:
    assert sent_settings_of(_Reporting(sent_settings=EFFORT_HIGH)) == EFFORT_HIGH


def test_a_completion_whose_sent_settings_are_not_request_settings_is_a_protocol_defect() -> None:
    with pytest.raises(CompletionProtocolDefect, match="_Reporting's optional `sent_settings`") as defect:
        sent_settings_of(_Reporting(sent_settings={"max_tokens": 10240}))
    assert defect.value.member == "sent_settings" and "it holds a dict" in str(defect.value)
    assert "check_completion_conformance" in str(defect.value)


# --- the conformance kit -----------------------------------------------------------------------------------------


def test_a_completion_without_sent_settings_conforms_unless_the_member_is_required() -> None:
    check_completion_conformance(_Completion())
    with pytest.raises(CompletionConformanceFailure, match="missing sent_settings, which this check was asked"):
        check_completion_conformance(_Completion(), require_sent_settings=True)


@pytest.mark.parametrize("sent", [None, EFFORT_HIGH], ids=["unverified", "reported"])
def test_a_completion_carrying_sent_settings_conforms_when_it_holds_settings_or_none(sent: Any) -> None:
    check_completion_conformance(_Reporting(sent_settings=sent))


def test_a_required_sent_settings_must_hold_settings_not_none() -> None:
    """Required, the member must say what was sent: ``None`` is a client that stopped saying, caught by name."""
    check_completion_conformance(_Reporting(sent_settings=EFFORT_HIGH), require_sent_settings=True)
    with pytest.raises(CompletionConformanceFailure, match="sent_settings is None, and this check was asked"):
        check_completion_conformance(_Reporting(sent_settings=None), require_sent_settings=True)


def test_a_completion_carrying_sent_settings_of_another_type_fails_conformance_by_name() -> None:
    with pytest.raises(CompletionConformanceFailure, match="sent_settings holds a str, not a ClientRequestSettings"):
        check_completion_conformance(_Reporting(sent_settings="high"))


# --- usage capture -----------------------------------------------------------------------------------------------


def test_one_call_s_usage_record_reads_what_it_was_sent_with() -> None:
    assert CallUsage.of(_Reporting(sent_settings=EFFORT_HIGH)).sent_settings == EFFORT_HIGH
    assert CallUsage.of(_Completion()).sent_settings is None


def test_one_call_s_usage_record_refuses_sent_settings_of_another_type() -> None:
    with pytest.raises(CompletionProtocolDefect, match="sent_settings"):
        CallUsage.of(_Reporting(sent_settings=7))


def test_a_usage_row_records_what_its_calls_were_sent_with() -> None:
    ledger = RoleUsageLedger(role="simulator")
    ledger.add_llm_result(_Reporting(sent_settings=EFFORT_HIGH))
    ledger.add_llm_result(_Reporting(sent_settings=EFFORT_HIGH))

    (row,) = ledger.rows()
    assert row.sent_settings == [EFFORT_HIGH] and row.call_count == 2


def test_calls_sent_two_ways_land_in_one_row_holding_both_in_canonical_order() -> None:
    """Sent settings are not in a row's key: spend is never split by them, and the row says it was sent both ways."""
    ledger = RoleUsageLedger(role="judge")
    ledger.add_llm_result(_Reporting(sent_settings=UNENFORCED))
    ledger.add_llm_result(_Reporting(sent_settings=BUDGET))
    ledger.add_llm_result(_Reporting(sent_settings=EFFORT_HIGH))

    (row,) = ledger.rows()
    assert row.call_count == 3
    assert row.sent_settings is not None and set(map(repr, row.sent_settings)) == {
        repr(EFFORT_HIGH),
        repr(BUDGET),
        repr(UNENFORCED),
    }
    again = RoleUsageLedger(role="judge")
    for sent in (EFFORT_HIGH, BUDGET, UNENFORCED):
        again.add_llm_result(_Reporting(sent_settings=sent))
    assert again.rows()[0].sent_settings == row.sent_settings, "the order is canonical, not the order of the calls"


def test_one_unverified_call_makes_the_row_unverified() -> None:
    """Like one unpriced call's dollars: the row cannot say what ALL its calls were sent with."""
    ledger = RoleUsageLedger(role="simulator")
    ledger.add_llm_result(_Reporting(sent_settings=EFFORT_HIGH))
    ledger.add_llm_result(_Completion())

    (row,) = ledger.rows()
    assert row.sent_settings is None and row.call_count == 2


def test_a_row_of_contributions_that_are_not_completions_is_unverified() -> None:
    ledger = RoleUsageLedger(role="inner_agent")
    ledger.add(model="m", prompt_tokens=1, completion_tokens=1, reasoning_tokens=None, cost_usd=0.0)
    assert ledger.rows()[0].sent_settings is None


@pytest.mark.parametrize(
    ("sent", "problem"),
    [
        ([], "lists every distinct value"),
        ([EFFORT_HIGH, EFFORT_HIGH], "each distinct value once"),
        (None, None),
    ],
)
def test_a_usage_row_holds_each_distinct_sent_value_once_in_canonical_order(sent: Any, problem: str | None) -> None:
    if problem is None:
        assert RoleUsage(role="judge", sent_settings=sent).sent_settings is None
        return
    with pytest.raises(ValidationError, match=problem):
        RoleUsage(role="judge", sent_settings=sent)


def test_a_usage_row_written_out_of_canonical_order_is_put_into_it() -> None:
    """A host lists the values in the order it met them; the row stores them in the one canonical order, so two rows
    naming the same values are equal, and a value repeated out of order is still refused."""
    (ordered,) = [row.sent_settings for row in _two_way_row()]
    assert ordered is not None and len(ordered) == 2

    assert RoleUsage(role="judge", sent_settings=list(reversed(ordered))).sent_settings == ordered
    assert RoleUsage(role="judge", sent_settings=list(reversed(ordered))) == RoleUsage(
        role="judge", sent_settings=ordered
    )
    with pytest.raises(ValidationError, match="each distinct value once"):
        RoleUsage(role="judge", sent_settings=[ordered[1], ordered[0], ordered[1]])


def _two_way_row() -> list[RoleUsage]:
    ledger = RoleUsageLedger(role="judge")
    ledger.add_llm_result(_Reporting(sent_settings=EFFORT_HIGH))
    ledger.add_llm_result(_Reporting(sent_settings=BUDGET))
    return ledger.rows()


# --- the stored fields, named for what they hold -----------------------------------------------------------------


def test_a_score_and_its_paired_halves_carry_what_the_first_call_was_sent_with() -> None:
    score = RubricScore(
        dim="conversation.tone", score=4, scale="ordinal", axis="capability", judge_sent_settings=EFFORT_HIGH
    )
    assert score.judge_sent_settings == EFFORT_HIGH
    assert RubricScore(dim="conversation.tone", score=4, scale="ordinal", axis="capability").judge_sent_settings is None
    halves = {
        "dim": "conversation.tone",
        "scale": "ordinal",
        "first_score": 4,
        "first_served_model": "m",
        "first_judge_config_id": None,
    }
    assert RepeatedScore(**halves, repeat=score, first_judge_sent_settings=BUDGET).first_judge_sent_settings == BUDGET
    assert SecondJudgeScore(**halves, second=score).first_judge_sent_settings is None


def test_a_run_records_what_its_roles_were_asked_under_names_that_say_so() -> None:
    run = make_eval_run(judge_model="judge-a", judge_asked_settings=JUDGE_REQUEST_SETTINGS)
    assert run.judge_asked_settings == JUDGE_REQUEST_SETTINGS
    assert "judge_request_settings" not in EvalRun.model_fields
    assert "simulator_request_settings" not in EvalRun.model_fields
    assert {"judge_asked_settings", "simulator_asked_settings"} <= set(EvalRun.model_fields)


def _spend(**fields: Any) -> OutOfRunSpend:
    return OutOfRunSpend(scope_id="uni-1", purpose="proposer", model="m", **fields)


def test_an_out_of_run_call_records_what_it_was_sent_with_and_a_raised_one_none() -> None:
    assert _spend(outcome="completed", stop_reason="end_turn", sent_settings=EFFORT_HIGH).sent_settings == EFFORT_HIGH
    assert _spend(outcome="raised", failure="TimeoutError").sent_settings is None
    with pytest.raises(ValidationError, match="sent_settings"):
        _spend(outcome="raised", failure="TimeoutError", sent_settings=EFFORT_HIGH)


def test_a_judge_case_s_digest_leaves_out_its_first_sent_settings_while_none() -> None:
    """A field a later build added stays out of a whole-shape digest at its default: frozen cases keep their digest."""
    case = JudgeCase(
        dim=TRANSCRIPT_DIM_ID,
        scale="ordinal",
        intent="Answer the member's question as the bot would.",
        judged_artifact=JudgedArtifact.TRANSCRIPT,
        evidence=JudgeEvidence(
            case_material="The member asked about the raid schedule.", artifact="assistant: Friday."
        ),
        source=JudgeCaseSource(run_id="source-run", result_id="result-1", test_case_id="source-case-1"),
    )
    unset = case.model_dump(mode="json")
    del unset["source"]["first_judge_sent_settings"]

    assert case.source.first_judge_sent_settings is None
    assert case.content_digest() == canonical_digest(unset)
    sent = case.model_copy(update={"source": case.source.model_copy(update={"first_judge_sent_settings": BUDGET})})
    assert sent.content_digest() != case.content_digest()


def test_a_judge_case_source_names_the_first_call_s_sent_settings() -> None:
    source = JudgeCaseSource(run_id="r", result_id="res", test_case_id="tc", first_judge_sent_settings=EFFORT_HIGH)
    assert source.first_judge_sent_settings == EFFORT_HIGH


# --- the engine's own constants: the judge asks a reasoning budget, the others effort words -----------------------


def test_the_judge_asks_for_a_reasoning_budget_strictly_under_a_derived_cap() -> None:
    assert JUDGE_REQUEST_SETTINGS == ClientRequestSettings(
        max_tokens=10240, reasoning_max_tokens=8192, strict_output=True
    )
    assert JUDGE_REQUEST_SETTINGS.reasoning_max_tokens == JUDGE_REASONING_HEADROOM_TOKENS
    assert JUDGE_MAX_TOKENS == JUDGE_REASONING_HEADROOM_TOKENS + JUDGE_ANSWER_BUDGET_TOKENS == 10240


def test_the_analysis_generator_asks_for_medium_effort_with_its_cap_sized_from_answer_and_headroom() -> None:
    settings = analysis_gen_request_settings_for(answer_budget_tokens=32768)
    assert (DEFAULT_ANALYSIS_GEN_REASONING_EFFORT, DEFAULT_ANALYSIS_GEN_REASONING_HEADROOM_TOKENS) == ("medium", 8192)
    assert settings == ClientRequestSettings(max_tokens=32768 + 8192, reasoning_effort="medium", strict_output=True)


def test_the_analysis_generator_takes_a_host_s_effort_and_headroom() -> None:
    settings = analysis_gen_request_settings_for(
        answer_budget_tokens=4000, reasoning_effort="high", reasoning_headroom_tokens=2000
    )
    assert settings == ClientRequestSettings(max_tokens=6000, reasoning_effort="high", strict_output=True)
    assert settings.reasoning_max_tokens is None, "the headroom sizes the cap and is never sent"


def test_a_headroom_too_small_to_think_in_is_refused_by_name() -> None:
    least = MIN_ANALYSIS_GEN_REASONING_HEADROOM_TOKENS
    assert least == 1024
    analysis_gen_request_settings_for(answer_budget_tokens=4000, reasoning_headroom_tokens=least)
    with pytest.raises(ValueError, match="reasoning_headroom_tokens=1023 leaves the analysis generator too little"):
        analysis_gen_request_settings_for(answer_budget_tokens=4000, reasoning_headroom_tokens=least - 1)


@pytest.mark.parametrize("role", ["candidate", "inner_agent"])
def test_a_role_pooling_never_reads_carries_its_spend_and_no_sent_settings(role: str) -> None:
    """Only the judge and the simulator are read for what was sent: no other role's row can ever read as verified."""
    ledger = RoleUsageLedger(role=role)  # type: ignore[arg-type]  # a KnownUsageRole literal chosen by the parametrize
    ledger.add_llm_result(_Reporting(sent_settings=EFFORT_HIGH))
    ledger.add_llm_result(_Reporting(sent_settings=7))  # a mistyped value is not read, so it cannot fail the cell

    (row,) = ledger.rows()
    assert row.sent_settings is None and row.call_count == 2
