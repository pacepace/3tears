"""``run.Ceilings`` carries the package's inner ceilings as one value, each the name it stands for.

A host derives its outer budgets from :class:`~threetears.evals.run.Ceilings`, so a field that drifted from the
constant or function the engine bounds by would derive a budget from a number nothing holds. Each field is held
to the exported original here, by identity for the two functions.
"""

from __future__ import annotations

import dataclasses

import pytest

from threetears.evals.analysis import generation_ceiling_s, judge_phase_ceiling_s
from threetears.evals.gen import PROPOSER_MAX_TOKENS
from threetears.evals.run import (
    SCHEDULER_CALL_ATTEMPTS,
    SIMULATOR_ANSWER_BUDGET_TOKENS,
    SIMULATOR_MAX_TOKENS,
    Ceilings,
)


def test_each_field_is_the_ceiling_it_names() -> None:
    ceilings = Ceilings()

    assert ceilings.scheduler_call_attempts == SCHEDULER_CALL_ATTEMPTS
    assert ceilings.simulator_max_tokens == SIMULATOR_MAX_TOKENS
    assert ceilings.simulator_answer_budget_tokens == SIMULATOR_ANSWER_BUDGET_TOKENS
    assert ceilings.proposer_max_tokens == PROPOSER_MAX_TOKENS
    assert ceilings.generation_ceiling_s is generation_ceiling_s
    assert ceilings.judge_phase_ceiling_s is judge_phase_ceiling_s


def test_the_fields_are_the_six_ceilings_and_none_is_set_by_the_caller() -> None:
    fields = dataclasses.fields(Ceilings)

    assert [f.name for f in fields] == [
        "scheduler_call_attempts",
        "simulator_max_tokens",
        "simulator_answer_budget_tokens",
        "proposer_max_tokens",
        "generation_ceiling_s",
        "judge_phase_ceiling_s",
    ]
    assert not any(f.init for f in fields)
    with pytest.raises(TypeError):
        Ceilings(proposer_max_tokens=1)  # type: ignore[call-arg]


def test_a_budget_derived_from_the_ceilings_is_the_ceiling_it_wraps() -> None:
    ceilings = Ceilings()

    def request_s(max_tokens: int) -> float:
        return max_tokens / 100

    generation_s = ceilings.generation_ceiling_s(request_s=request_s, generator_max_tokens=1000)
    judging_s = ceilings.judge_phase_ceiling_s(
        judge_dims=3, judge_concurrency=2, judge_call_attempts=2, judge_max_tokens=500, request_s=request_s
    )

    assert generation_s == generation_ceiling_s(request_s=request_s, generator_max_tokens=1000)
    assert judging_s == 2 * 2 * 5.0
