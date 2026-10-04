"""The engine's judge path, pinned on inputs that name no host.

The judge is engine code that the toy host never reaches: it has no model-scored axis by
design, so no toy test can fail without a change here. The substitute evidence is a test
whose input carries no host type or vocabulary. Every fixture below is an invoice-summary
candidate scored on a ``clarity`` dimension by a scripted client. There is no adapter
import and no conversational domain.

Three engine behaviours are pinned:

* a judge reply the provider cut short is refused after one call and never re-bought, even
  when its content happens to parse (``judge.run_judge_llm``);
* a document candidate is judged against its case material, and the judge is told so in its
  instructions and its JSON contract (``judge_service``);
* the kind contract carries the judge evidence and the "nothing to time" reason as typed
  fields, with the evidence's whitespace kept intact (``candidate_kind``, ``models``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import ValidationError

import threetears.evals.contracts as contracts
from threetears.evals.contracts.candidate_kind import CandidateOutput, CandidateTelemetry
from threetears.evals.contracts.models import GoalStateOutcome, JudgedArtifact, JudgeEvidence, RubricDim
from threetears.evals.contracts.provider import INCOMPLETE_STOP_REASONS, withhold_failure_detail
from threetears.evals.run.judge import run_judge_llm
from threetears.evals.run.judge_service import JudgeContext, JudgeService


_DIM = "summary.clarity"

#: A reply that parses: salvage would score it if the truncation check came after parsing.
_PARSEABLE = json.dumps({"reasoning": "the totals line is legible", "criteria_scores": {_DIM: 4}})


@dataclass
class _Reply:
    """One completion, shaped as the engine's completion port describes it."""

    content: str
    stop_reason: str
    input_tokens: int = 120
    output_tokens: int = 64
    reasoning_tokens: int | None = 40
    cost_usd: float | None = 0.01
    served_model: str | None = None
    model: str = "judge/scripted"


@dataclass
class _ScriptedJudge:
    """Answers every call with one fixed reply and records what it was sent."""

    reply: _Reply
    calls: list[tuple[str, str]] = field(default_factory=list)

    async def generate(self, *, system: str, user: str, response_format: Any = None) -> _Reply:
        """Record the prompt pair and hand back the fixed reply.

        Args:
            system: The system prompt.
            user: The user prompt.
            response_format: Ignored.

        Returns:
            The scripted reply.
        """
        self.calls.append((system, user))
        return self.reply


async def _judge(reply: _Reply) -> tuple[dict[str, Any], _ScriptedJudge]:
    client = _ScriptedJudge(reply)
    result = await run_judge_llm(
        client=client,
        system_prompt="Score the summary.",
        user_prompt="Invoice INV-7 totals 1,240.00.",
        criteria_dicts=[{"name": _DIM, "weight": 1.0}],
        label="Judge",
        case_id="case-1",
    )
    assert result is not None
    return result, client


class TestACutShortReplyIsNotReBought:
    """``run_judge_llm`` reads the provider's own stop reason before it parses anything."""

    @pytest.mark.parametrize("stop_reason", sorted(INCOMPLETE_STOP_REASONS))
    async def test_a_cut_short_reply_is_refused_after_one_call_even_when_it_parses(self, stop_reason: str) -> None:
        result, client = await _judge(_Reply(content=_PARSEABLE, stop_reason=stop_reason))

        assert len(client.calls) == 1, "the identical request meets the identical cap, so it is not re-sent"
        assert "criteria_scores" not in result, "a reply the provider cut short must not be scored"
        assert "cut short by the provider" in result["error"]
        assert result["judge_usage"].calls == 1

    async def test_a_finished_reply_that_parses_is_scored_on_the_same_fixture(self) -> None:
        """The accepting side of the refusal above: only the stop reason differs."""
        result, client = await _judge(_Reply(content=_PARSEABLE, stop_reason="end_turn"))

        assert len(client.calls) == 1
        assert "error" not in result
        assert result["criteria_scores"] == {_DIM: pytest.approx(0.75)}

    async def test_a_finished_reply_that_will_not_parse_is_asked_again(self) -> None:
        """A malformed finished reply is re-askable, so it spends the bounded retry."""
        result, client = await _judge(_Reply(content="the totals look right", stop_reason="end_turn"))

        assert len(client.calls) == 2
        assert "Failed to parse" in result["error"]
        assert result["judge_usage"].calls == 2

    async def test_the_refusal_names_the_token_counts_behind_it(self) -> None:
        result, _ = await _judge(
            _Reply(content="", stop_reason="max_tokens", output_tokens=2048, reasoning_tokens=2000)
        )

        assert "Failed to parse" not in result["error"]
        assert "2048 output token(s), of which 2000 were reasoning" in result["error"]


def _evidence() -> JudgeEvidence:
    # Leading indentation and a blank line are part of what the judge reads.
    return JudgeEvidence(
        case_material="  line  | amount\n  rent  | 1000.00\n\n  total | 1240.00",
        artifact="  The invoice totals 1240.00.\n",
    )


def _context(*, judged_artifact: JudgedArtifact) -> JudgeContext:
    return JudgeContext(
        case_id="case-1",
        intent="summarise the invoice",
        variation={"layout": "two-column"},
        goal_outcomes=[GoalStateOutcome(expression="fields_extracted >= 3", passed=True, detail="3 fields")],
        judged_artifact=judged_artifact,
        judge_evidence=_evidence(),
    )


async def _score(context: JudgeContext) -> tuple[str, str, Any]:
    client = _ScriptedJudge(_Reply(content=_PARSEABLE, stop_reason="end_turn"))
    service = JudgeService(client_factory=lambda model, temperature: client, failure_describer=withhold_failure_detail)
    outcome = await service.score_dimension(
        RubricDim(name=_DIM, description="Is the summary easy to read?", scale="ordinal"), context
    )
    (system, user) = client.calls[0]
    return system, user, outcome


class TestADocumentCandidateIsJudgedAgainstItsCaseMaterial:
    """``JudgeService`` switches the prompt on the kind's declaration; the evidence is the same either way."""

    async def test_the_prompt_carries_the_material_and_the_output_and_no_transcript(self) -> None:
        system, user, outcome = await _score(_context(judged_artifact=JudgedArtifact.DOCUMENT))

        assert outcome.score is not None and outcome.score.score == 4
        assert "# Case material" in user and "# Output under review" in user
        assert _evidence().case_material in user, "the material reaches the judge byte for byte"
        assert _evidence().artifact in user
        assert "# Transcript" not in user
        assert "a candidate's output" in system and "conversation" not in system
        assert "passages of the output and of the case material" in system
        assert "moments in the transcript" not in system

    async def test_declared_a_transcript_the_same_evidence_is_judged_as_a_conversation(self) -> None:
        """The other side of the switch, on the same fixture: only the declaration differs."""
        system, user, _ = await _score(_context(judged_artifact=JudgedArtifact.TRANSCRIPT))

        assert "# Transcript" in user and "# Output under review" not in user
        assert _evidence().case_material in user, "a conversation is judged against its case material too"
        assert user.endswith(f"# Transcript\n{_evidence().artifact}")
        assert "a candidate's side of a conversation" in system
        assert "moments in the transcript" in system


class TestTheKindContractCarriesTheNewFields:
    """``candidate_kind``'s two additions are typed fields, not free-form extras."""

    def test_the_evidence_keeps_its_whitespace(self) -> None:
        evidence = _evidence()

        assert evidence.case_material.startswith("  line")
        assert evidence.artifact.endswith("1240.00.\n")

    def test_the_evidence_is_one_immutable_object(self) -> None:
        evidence = _evidence()
        with pytest.raises(ValidationError):
            evidence.artifact = "rewritten"  # type: ignore[misc]
        with pytest.raises(ValidationError):
            JudgeEvidence(case_material="m", artifact="a", notes="x")  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            JudgeEvidence(case_material="m")  # type: ignore[call-arg]

    def test_an_output_carries_its_evidence_and_an_unjudged_one_carries_none(self) -> None:
        assert CandidateOutput(output=[{"summary": "x"}], judge_evidence=_evidence()).judge_evidence == _evidence()
        assert CandidateOutput(output=[{"summary": "x"}]).judge_evidence is None

    def test_telemetry_can_say_there_was_nothing_to_time(self) -> None:
        telemetry = CandidateTelemetry(untimed_reason="a stored summary was returned")
        assert telemetry.untimed_reason == "a stored summary was returned"
        assert CandidateTelemetry().untimed_reason is None

    def test_the_evidence_type_is_part_of_the_published_contract(self) -> None:
        assert {"JudgeEvidence", "JudgedArtifact"} <= set(contracts.__all__)
