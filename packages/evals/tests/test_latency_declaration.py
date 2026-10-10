"""Latency under test is declared, never inferred, and a contradiction is refused (#701).

One declaration says latency is under test: ``measure_latency=True`` on a launch, on the quick path, and on a
campaign's design. Declared, the cells run one at a time; not declared, they run concurrently. A campaign
that asks about latency — a bar on a latency measure, a question on the latency axis, latency in its merit
priority — without declaring it is refused when the design is declared, naming the setting to turn on, and
the engine never quietly switches a run to serial behind it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals.contracts import Question
from threetears.evals.contracts.declaration import BarOverride, refuse_an_undeclarable_design
from threetears.evals.quick import compare, run_eval
from threetears.evals.run.executor import DEFAULT_MAX_CONCURRENT_CELLS
from packages.evals.tests.factories import minimal_declaration
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.run import toyhost_template

_PROFILE = toyhost_profile()
_LATENCY_BAR = BarOverride(measure_id="total_ms", threshold=2000.0, direction="lower_is_better")
_LATENCY_QUESTION = Question(id="q-fast", text="is it fast enough?", merit_axes=["latency"])


def _refuse(**design: Any) -> None:
    declared = minimal_declaration().model_copy(update=design)
    refuse_an_undeclarable_design(declared, behavior="b", template=toyhost_template(), profile=_PROFILE)


@pytest.mark.parametrize(
    ("design", "named"),
    [
        pytest.param({"bars": [_LATENCY_BAR]}, "the bar on total_ms", id="a-latency-bar"),
        pytest.param({"questions": [_LATENCY_QUESTION]}, "the question 'is it fast enough?'", id="a-latency-question"),
        pytest.param({"merit_priority": ["latency", "quality"]}, "latency in merit_priority", id="a-latency-ranking"),
    ],
)
def test_a_design_asking_about_latency_without_declaring_it_is_refused_naming_the_setting(
    design: dict[str, Any], named: str
) -> None:
    with pytest.raises(ValueError, match="does not declare latency under test") as refused:
        _refuse(**design)

    assert named in str(refused.value)
    assert "measure_latency=True" in str(refused.value)


def test_the_same_design_declaring_latency_under_test_is_accepted() -> None:
    _refuse(bars=[_LATENCY_BAR], questions=[_LATENCY_QUESTION], merit_priority=["latency"], measure_latency=True)


def test_a_design_asking_nothing_of_latency_needs_no_declaration() -> None:
    _refuse(questions=[Question(id="q-good", text="is it right?", merit_axes=["quality"])])


def test_a_stored_design_asking_about_latency_still_loads() -> None:
    """The refusal is the declaration's gate, never a read's: a design stored before the field loads as it was."""
    stored = minimal_declaration().model_dump(mode="json") | {"bars": [_LATENCY_BAR.model_dump(mode="json")]}

    loaded = type(minimal_declaration()).model_validate(stored)

    assert loaded.measure_latency is False and loaded.bars == [_LATENCY_BAR]


# =============================================================================
# The quick path: the keyword decides the mode, and the run and the campaign record it
# =============================================================================


class _Clock:
    """How many cases a candidate is answering at once."""

    def __init__(self) -> None:
        self.running = 0
        self.peak = 0

    def candidate(self) -> Any:
        async def answer(case: Mapping[str, Any]) -> str:
            self.running += 1
            self.peak = max(self.peak, self.running)
            try:
                await asyncio.sleep(0.01)
                return str(case["label"])
            finally:
                self.running -= 1

        return answer


_CASES = [{"text": f"case {index}", "label": "yes" if index % 2 else "no"} for index in range(8)]


def _expected(case: Mapping[str, Any]) -> str:
    return str(case["label"])


async def test_run_eval_runs_its_cases_concurrently_unless_latency_is_declared() -> None:
    concurrent, serial = _Clock(), _Clock()

    fast = await run_eval(_CASES, concurrent.candidate(), expected=_expected, scope_id="latency-quick", model="a")
    slow = await run_eval(
        _CASES, serial.candidate(), expected=_expected, scope_id="latency-quick", model="b", measure_latency=True
    )

    assert concurrent.peak == DEFAULT_MAX_CONCURRENT_CELLS
    assert serial.peak == 1, "a run declaring latency under test ran two cases at once"
    assert fast.status == slow.status == "completed"


async def test_compare_declaring_latency_runs_every_arm_serially_and_its_campaign_records_it() -> None:
    # One clock for both arms: what is under test is that nothing of the comparison ran beside anything else.
    clock = _Clock()

    comparison = await compare(
        _CASES,
        {"a": clock.candidate(), "b": clock.candidate()},
        expected=_expected,
        control="a",
        scope_id="latency-compare",
        measure_latency=True,
    )

    assert clock.peak == 1, "an arm or a case of a latency comparison ran beside another"
    storage = comparison.host.storage
    for summary in comparison.arms.values():
        run = storage.load_eval_run(summary.run_id, "latency-compare")
        assert run is not None and (run.measure_latency, run.cell_concurrency) == (True, 1)
    campaign = storage.load_campaign(comparison.campaign_id, "latency-compare")
    assert campaign is not None and campaign.declared_design is not None
    assert campaign.declared_design.measure_latency is True
