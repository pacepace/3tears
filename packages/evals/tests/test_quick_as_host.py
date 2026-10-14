"""``as_host``: the host bridge — one host for a set of evals, over a store of the product's own.

What a host adopter does with it: build the host once over every eval it serves, run each eval into it under a scope
it names, and read the runs back through the engine's other surfaces. Its refusals keep one host one vocabulary: one
measure per grader name, one set of levers, one world.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field

from threetears.evals import Eval, World, grader
from threetears.evals.quick import as_host
from threetears.evals.run import list_runs
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.off_loop import off_loop

CASES = [{"id": "two", "n": 2, "label": "even"}, {"id": "three", "n": 3, "label": "odd"}]


async def exact(case: Mapping[str, Any]) -> str:
    """Say whether the number is even."""
    return "even" if int(case["n"]) % 2 == 0 else "odd"


async def sloppy(case: Mapping[str, Any]) -> str:
    """Say whether the number is even."""
    return "even"


def correct(case: Mapping[str, Any], out: Any) -> bool:
    """The answer is the case's label."""
    return out == case["label"]


async def test_one_host_holds_the_runs_of_every_eval_it_was_built_for_in_the_store_given() -> None:
    store = InMemoryDocumentStore()
    exact_eval = Eval(CASES, {"arm-exact": exact}, [correct], k=1)
    sloppy_eval = Eval(CASES, {"arm-sloppy": sloppy}, [correct], k=1)
    host = as_host([exact_eval, sloppy_eval], store=store)

    first = await exact_eval.run(host=host, scope="shared")
    second = await sloppy_eval.run(host=host, scope="shared")

    assert first.template_id == second.template_id, "one case set, one template"
    assert (first.summary.candidate_model, second.summary.candidate_model) == ("arm-exact", "arm-sloppy")
    assert {run.id for run in off_loop(list_runs, host, "shared")} == {first.run_id, second.run_id}


def test_one_host_reads_one_measure_of_a_name() -> None:
    def correct_too(case: Mapping[str, Any], out: Any) -> bool:
        """Another grade under the same name."""
        return True

    correct_too.__name__ = "correct"

    with pytest.raises(ValueError, match="two graders named 'correct' declare different readings"):
        as_host(
            [Eval(CASES, exact, [grader(correct, margin=0.1)]), Eval(CASES, exact, [correct_too])],
            store=":memory:",
        )


def test_one_host_states_one_set_of_levers_and_one_world() -> None:
    class Light(BaseModel):
        on: bool = Field(description="whether the light is on")

    def flip(state: Light, to: Literal["on", "off"]) -> str:
        """Flip the light."""
        state.on = to == "on"
        return to

    async def agent(case: Mapping[str, Any], tools: Any) -> str:
        """Flip it."""
        return str(await tools["flip"](to="on"))

    with pytest.raises(ValueError, match="states a level of every lever"):
        as_host([Eval(CASES, {"prompt=v1": exact}, ["label"]), Eval(CASES, exact, ["label"])], store=":memory:")
    on_one = Eval(CASES, World(Light, tools=[flip])(agent), [correct])
    on_another = Eval(CASES, World(Light, tools=[flip])(agent), [correct])
    with pytest.raises(ValueError, match="one host declares one world"):
        as_host([on_one, on_another], store=":memory:")
    with pytest.raises(ValueError, match="none was"):
        as_host([], store=":memory:")
