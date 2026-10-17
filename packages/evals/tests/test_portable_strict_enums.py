"""The judge asks for an ordinal score as a string and reads it back as its integer; a host's integer enum is refused.

``google/gemini-3.5-flash-lite`` through OpenRouter, with ``require_parameters`` on, accepted the judge's strict reply
format while it scored an ordinal criterion as ``{"type": "integer", "enum": [1, ..., 5]}``, and did not enforce it:
it answered ``"criteria_scores": {}`` with the score in prose, every ordinal call failed its parse twice, and a whole
run's results were excluded. Gemini's structured output supports string enums only, so the reply format asks for
``"1"`` to ``"5"`` and the parse reads them back; a bare integer, which a provider not enforcing the schema may still
send, is read as before. The strict projection the engine and a host share refuses any other non-string enum by name,
with the remedy, instead of changing the reply type a host's own parser expects.
"""

from __future__ import annotations

import enum
import json
import logging
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from threetears.evals.kernel import UnportableSchema, strict_response_format
from threetears.evals.run.judge import ORDINAL_ANSWERS, SCALE_READERS, run_judge_llm

_ORDINAL = [{"name": "tone.warm", "weight": 1.0}]


class _Replying:
    """A judge client that answers every call with one fixed reply, counting the calls."""

    def __init__(self, score: Any) -> None:
        self.content = json.dumps({"reasoning": "r", "criteria_scores": {"tone.warm": score}})
        self.calls = 0

    async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
        self.calls += 1
        return SimpleNamespace(
            served_model=None, stop_reason="end_turn", content=self.content, input_tokens=1, output_tokens=1,
            cost_usd=0.0, model="judge-fake", temperature=None,
        )  # fmt: skip


async def _judged(score: Any) -> tuple[dict[str, Any], int]:
    client = _Replying(score)
    parsed = await run_judge_llm(client, "s", "u", _ORDINAL, "Judge", "tc-1")
    assert parsed is not None
    return parsed, client.calls


def test_the_ordinal_answers_are_the_scale_written_as_strings() -> None:
    assert ORDINAL_ANSWERS == ("1", "2", "3", "4", "5")


@pytest.mark.parametrize(("answer", "score"), [("1", 1), ("3", 3), ("5", 5)])
async def test_a_reply_with_a_string_score_parses_to_its_integer(answer: str, score: int) -> None:
    assert SCALE_READERS["ordinal"]("tone.warm", answer) == score
    parsed, calls = await _judged(answer)
    assert parsed["criteria_ordinal_scores"] == {"tone.warm": score}
    assert calls == 1


@pytest.mark.parametrize("score", [1, 4, 5])
async def test_a_reply_with_an_integer_score_still_parses(score: int) -> None:
    """A provider that does not enforce the schema may answer the integer the prompt once asked for."""
    parsed, calls = await _judged(score)
    assert parsed["criteria_ordinal_scores"] == {"tone.warm": score}
    assert calls == 1


@pytest.mark.parametrize("answer", ["0", "6", "4.0", " 4", "four", ""])
async def test_an_out_of_scale_string_is_refused_by_name(answer: str, caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(ValueError, match=r"criterion 'tone\.warm' is not one of the 1-5 scale's answers"):
        SCALE_READERS["ordinal"]("tone.warm", answer)

    with caplog.at_level(logging.WARNING):
        parsed, _ = await _judged(answer)

    assert parsed["error"].startswith("Failed to parse") and "criteria_scores" not in parsed
    assert f"criterion 'tone.warm' is not one of the 1-5 scale's answers ['1', '2', '3', '4', '5']: {answer!r}" in (
        caplog.text
    )


# =============================================================================
# A host's schema: refused at the projection, by name, with the remedy
# =============================================================================


class _Verdict(enum.IntEnum):
    LOW = 1
    HIGH = 2


class _IntegerLiteral(BaseModel):
    rating: Literal[1, 2, 3]


class _IntegerEnum(BaseModel):
    verdict: _Verdict


class _IntegerConst(BaseModel):
    version: Literal[2]


class _StringLiteral(BaseModel):
    rating: Literal["1", "2", "3"]


@pytest.mark.parametrize(
    ("model", "where", "values"),
    [
        (_IntegerLiteral, "/properties/rating/enum", "[1, 2, 3]"),
        (_IntegerEnum, "/$defs/_Verdict/enum", "[1, 2]"),
        (_IntegerConst, "/properties/version/const", "[2]"),
    ],
)
def test_a_hosts_schema_with_a_non_string_enum_is_refused_with_the_remedy(
    model: type[BaseModel], where: str, values: str
) -> None:
    """The projection does not know a host's reader, so it refuses rather than change the reply type it parses."""
    with pytest.raises(UnportableSchema) as refused:
        strict_response_format("host_reply", model.model_json_schema())

    message = str(refused.value)
    assert message.startswith(f"{where} carries") and values in message
    assert "a string enum is the one enum form every provider's strict structured-output mode enforces" in message
    assert "declare the values as strings" in message


def test_a_hosts_schema_with_a_string_enum_is_projected_as_declared() -> None:
    directive = strict_response_format("host_reply", _StringLiteral.model_json_schema())

    assert directive["json_schema"]["schema"]["properties"]["rating"] == {"type": "string", "enum": ["1", "2", "3"]}
