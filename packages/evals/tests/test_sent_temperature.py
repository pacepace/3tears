"""A judge completion's temperature is part of the completion contract: a host's completion lacking it fails loudly.

``sent_temperature`` read ``CompletionResult.temperature`` through ``hasattr`` and answered "not recorded" for a
completion that had none, so a host whose completion type omitted it made every judge unrecordable as a profile
and every judge comparison "unknown", with no error anywhere. The attribute is a member of the protocol, which
``check_completion_conformance`` holds a host's type to; reading it is no longer optional. And its absence is the
host's defect, not a judging outcome: it is raised (``CompletionProtocolDefect``) and fails the run, carrying what
the billed calls cost, rather than becoming an error on every dimension of a run that finishes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

import pytest

from threetears.evals.quick import Judge, callable_host, run_eval
from threetears.evals.run.judge import run_judge_llm, sent_temperature
from threetears.evals.schema import CompletionProtocolDefect
from threetears.evals.schema.models import MODEL_DEFAULT_TEMPERATURE
from threetears.evals.testing import CompletionConformanceFailure, check_completion_conformance


@dataclass(frozen=True)
class _NoTemperature:
    """A host's completion that reports everything but the temperature it was sent at."""

    content: str = "{}"
    input_tokens: int | None = 1
    output_tokens: int | None = 1
    cost_usd: float | None = 0.0
    price_source: str | None = "provider"
    model: str = "judge-model"
    served_model: str | None = "judge-model"
    reasoning_tokens: int | None = None
    stop_reason: str = "end_turn"


@dataclass(frozen=True)
class _Sent(_NoTemperature):
    temperature: float | None = 0.0


def test_a_completion_with_no_temperature_is_a_defect_named_where_it_is_read() -> None:
    with pytest.raises(TypeError, match="check_completion_conformance"):
        sent_temperature(_NoTemperature())  # type: ignore[arg-type]


def test_the_conformance_check_names_it() -> None:
    with pytest.raises(CompletionConformanceFailure, match="temperature"):
        check_completion_conformance(_NoTemperature())


def test_a_temperature_sent_or_withheld_is_read() -> None:
    assert sent_temperature(_Sent(temperature=0.0)) == 0.0  # type: ignore[arg-type]
    assert sent_temperature(_Sent(temperature=None)) == MODEL_DEFAULT_TEMPERATURE  # type: ignore[arg-type]


# parity-with: threetears.evals.schema.CompletionClient
class _FakeUntemperedJudge:
    """A judge client whose completions are billed and carry no ``temperature``."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
        self.calls += 1
        match = re.search(r'single key "([^"]+)"', system)
        dim = match[1] if match is not None else "d"
        return _NoTemperature(content=json.dumps({"reasoning": "ok", "criteria_scores": {dim: 4}}), cost_usd=_COST)

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


#: What each billed call costs.
_COST = 0.5


async def test_a_reply_with_no_temperature_is_raised_as_the_host_s_defect_carrying_its_spend() -> None:
    """The host's completion type, not a judging outcome, so it is raised; the billed call goes up with it."""
    with pytest.raises(CompletionProtocolDefect, match="check_completion_conformance") as raised:
        await run_judge_llm(
            client=_FakeUntemperedJudge(),
            system_prompt='Score the single key "d".',
            user_prompt="u",
            criteria_dicts=[{"name": "d", "weight": 1.0, "scale": "ordinal"}],
            label="Judge",
            case_id="c",
            cannot_tell_offered=True,
        )
    assert isinstance(raised.value, TypeError)
    assert (raised.value.member, raised.value.calls, raised.value.cost_usd) == ("temperature", 1, _COST)


async def test_a_judged_run_on_such_a_client_fails_naming_the_check_rather_than_finishing_with_every_dim_failed() -> (
    None
):
    """Every call through the client returns the same type, so the run fails on it, its error naming the check."""
    judge = _FakeUntemperedJudge()
    summary = await run_eval(
        [{"q": "a"}, {"q": "b"}],
        _answer,
        judge=Judge(client=judge, model="judge-model", rubric={"helpful": "It helps."}),
        scope_id="untempered-judge",
        host=callable_host(),
        k=1,
    )
    assert summary.status == "failed"
    assert any("check_completion_conformance" in error for error in summary.errors)


async def _answer(case: Mapping[str, Any]) -> str:
    """Answer in a word."""
    return str(case["q"])
