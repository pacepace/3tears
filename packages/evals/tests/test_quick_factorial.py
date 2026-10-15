"""A second factor is a lever of its own: an arm named as ``k=v`` pairs states each beside its model.

An arm's run is keyed by its model; a lever named beside it (``{"model=m, prompt=v2": runner}``) is frozen onto the
run as an overlay of the callable kind, so two runs of one model at two prompts are two variants, and the campaign can
declare the prompt as an axis. A mapping of such arms is a factorial: each cell of the design runs as one arm, the
campaign declares one axis per factor, the first key is the control, and :meth:`Result.against` reads the same runs
against another control without running anything again.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals import Eval, Result
from threetears.evals.analysis import variant_key_of_run
from threetears.evals.kernel import ValidationFailedError
from threetears.evals.kernel.host import CANDIDATE_MODEL_LEVER, EvalHost
from threetears.evals.schema import SweepableValue
from threetears.evals.quick import as_host
from threetears.evals.quick.one_call import (
    CALLABLE_KIND_CONTRACT,
    SHARED_ARM_MODEL,
    callable_host,
    callable_kind_contracts,
)
from threetears.evals.run import list_runs, list_templates
from packages.evals.tests.off_loop import off_loop

SCOPE = "compare-factors"

#: Eight cases; the last four carry a second label's word, which only the careful rule reads past.
CASES = [
    {"text": "refund please", "label": "billing"},
    {"text": "charged twice", "label": "billing"},
    {"text": "app crashes", "label": "bug"},
    {"text": "button broken", "label": "bug"},
    {"text": "charged, but the page crashes", "label": "billing"},
    {"text": "refund page crashes", "label": "billing"},
    {"text": "crash after refund", "label": "billing"},
    {"text": "charged then crashed", "label": "billing"},
]


async def careful(case: Mapping[str, Any]) -> str:
    """Billing whenever money is mentioned, else a bug."""
    text = case["text"]
    return "billing" if "refund" in text or "charged" in text else "bug"


async def hasty(case: Mapping[str, Any]) -> str:
    """A bug whenever anything crashed, else billing."""
    text = case["text"]
    return "bug" if "crash" in text or "broken" in text else "billing"


async def _one(runner: Any, host: EvalHost, *, k: int = 1) -> Any:
    """One arm's summary, its case graded on its ``label`` field, run in ``host``."""
    return (await Eval(CASES, runner, ["label"], k=k).run(host=host, scope=SCOPE)).summary


# --- one arm at a lever -----------------------------------------------------------------------------------------


async def test_a_lever_beside_the_model_is_frozen_on_the_run_and_splits_the_variant() -> None:
    host = callable_host(levers=("prompt",))
    one = await _one({"model=m, prompt=v1": careful}, host)
    two = await _one({"model=m, prompt=v2": careful}, host)
    runs = {run.id: run for run in off_loop(list_runs, host, SCOPE)}
    assert runs[one.run_id].overlays == {"prompt": "v1"} and runs[two.run_id].overlays == {"prompt": "v2"}
    keys = [variant_key_of_run(off_loop(host.storage.query_eval_results_by_run, s.run_id, SCOPE)) for s in (one, two)]
    assert keys[0] != keys[1], "same model, same cases, different prompt: two variants"
    assert host.profile.sweepables.get("callable.prompt") is not None


async def test_with_no_host_an_eval_declares_exactly_the_levers_its_arms_name() -> None:
    result = await Eval(CASES, {"prompt=v1": careful}, ["label"], k=1).run(store=":memory:", scope=SCOPE)
    assert result.summary.status == "completed"
    assert result.summary.candidate_model == SHARED_ARM_MODEL


@pytest.mark.parametrize(
    ("runner", "said"),
    [
        ({"prompt=v1, temperature=0": careful}, "temperature"),
        (careful, "prompt"),
    ],
    ids=["an undeclared lever", "a declared lever left out"],
)
async def test_a_launch_must_state_every_declared_lever_and_no_other(runner: Any, said: str) -> None:
    host = callable_host(levers=("prompt",))
    with pytest.raises(ValidationFailedError, match=said):
        await _one(runner, host)


def test_a_blank_level_is_refused_where_the_eval_is_built() -> None:
    with pytest.raises(ValueError, match="is not one"):
        Eval(CASES, {"prompt= ": careful}, ["label"])


@pytest.mark.parametrize(
    ("levers", "said"),
    [
        (("model",), "is not 'model'"),
        (("_private",), "does not start with '_'"),
        (("two words",), "Python identifier"),
        (("prompt", "prompt"), "more than once"),
        ("prompt", "not one string"),
    ],
    ids=["the model", "a private name", "not an identifier", "repeated", "one string"],
)
def test_an_unusable_lever_name_is_refused_where_the_host_is_built(levers: Any, said: str) -> None:
    with pytest.raises(ValueError, match=said):
        callable_host(levers=levers)


async def test_a_callers_host_declares_its_levers_with_callable_kind_contracts() -> None:
    base = callable_host()
    unjudged, judged = callable_kind_contracts(("prompt",))
    assert (unjudged.seats, judged.seats) == (CALLABLE_KIND_CONTRACT.seats, frozenset({"judge"}))
    assert callable_kind_contracts(()) == (CALLABLE_KIND_CONTRACT, base.profile.kinds[1])
    host = dataclasses.replace(base, profile=dataclasses.replace(base.profile, kinds=(unjudged, judged)))
    summary = await _one({"prompt=v1": careful}, host)
    assert summary.status == "completed"


# --- a factorial ----------------------------------------------------------------------------------------------

#: The design's four cells, the control first.
ARMS = {
    "model=a, prompt=v1": hasty,
    "model=a, prompt=v2": careful,
    "model=b, prompt=v1": careful,
    "model=b, prompt=v2": careful,
}


async def _factorial(arms: Mapping[str, Any] | None = None, *, host: EvalHost | None = None) -> Result:
    ev = Eval(CASES, ARMS if arms is None else arms, ["label"], k=2)
    if host is not None:
        return await ev.run(host=host, scope=SCOPE)
    return await ev.run(store=":memory:", scope=SCOPE)


async def test_each_cell_runs_as_one_arm_at_its_model_and_its_lever() -> None:
    comparison = await _factorial()
    assert comparison.factors == ("model", "prompt")
    assert comparison.name == " vs ".join(ARMS)
    assert [summary.candidate_model for summary in comparison.arms.values()] == ["a", "a", "b", "b"]
    runs = {run.id: run for run in off_loop(list_runs, comparison.host, SCOPE)}
    assert [runs[s.run_id].overlays for s in comparison.arms.values()] == [{"prompt": p} for p in ("v1", "v2") * 2]


async def test_the_campaign_declares_one_axis_per_factor_at_the_levels_the_runs_carry() -> None:
    comparison = await _factorial()
    campaign = off_loop(comparison.host.storage.load_campaign, comparison.campaign_id, SCOPE)
    assert campaign is not None and campaign.declared_design is not None
    design = campaign.declared_design
    model_axis, prompt_axis = design.axes
    assert (model_axis.axis_id, prompt_axis.axis_id) == (CANDIDATE_MODEL_LEVER, "callable.prompt")
    assert [value.content_hash for value in prompt_axis.values] == [
        SweepableValue.of(level, display=level).content_hash for level in ("v1", "v2")
    ]
    assert design.control == variant_key_of_run(
        off_loop(comparison.host.storage.query_eval_results_by_run, comparison.arms["model=a, prompt=v1"].run_id, SCOPE)
    )
    assert design.intended_repetitions == 2


async def test_every_arm_is_named_by_both_coordinates_and_tested_against_the_control() -> None:
    comparison = await _factorial()
    rows = {row["contrast"]: row for row in comparison.contrasts("accuracy")}
    assert set(rows) == {"callable prompt v2, model a", "callable prompt v1, model b", "callable prompt v2, model b"}
    assert {row["control"] for row in rows.values()} == {"callable prompt v1, model a"}
    assert all(row["delta"] == pytest.approx(0.5) for row in rows.values()), "the careful rule reads every case"
    # The arms return plain answers, so no spend was observed and cost is not a reading to test.
    assert {row["reading"] for row in comparison.contrasts()} == {"Accuracy"}


async def test_against_reads_the_same_runs_against_another_control_and_runs_nothing() -> None:
    comparison = await _factorial()
    n_runs = len(off_loop(list_runs, comparison.host, SCOPE))
    on_b = await comparison.against("model=b, prompt=v1")
    assert len(off_loop(list_runs, comparison.host, SCOPE)) == n_runs
    assert on_b.campaign_id != comparison.campaign_id
    assert on_b.arms == comparison.arms and on_b.control == "model=b, prompt=v1"
    assert on_b.name == f"{comparison.name}, against model=b, prompt=v1"
    rows = {row["contrast"]: row for row in on_b.contrasts("accuracy")}
    assert rows["callable prompt v2, model b"]["control"] == "callable prompt v1, model b"
    assert rows["callable prompt v2, model b"]["verdict"] == "not separated from the control"
    assert rows["callable prompt v1, model a"]["delta"] == pytest.approx(-0.5)
    with pytest.raises(ValueError, match="names no arm"):
        await comparison.against("model=c, prompt=v1")


async def test_a_one_factor_comparison_is_read_against_another_control_too() -> None:
    comparison = await _factorial({"hasty": hasty, "careful": careful})
    assert comparison.factors == ("candidate",)
    flipped = await comparison.against("careful")
    (row,) = flipped.contrasts("accuracy")
    assert (row["contrast"], row["control"], row["verdict"]) == (
        "hasty",
        "careful",
        "regressed from the control",
    )


async def test_arms_naming_no_model_run_at_the_one_shared_model() -> None:
    comparison = await _factorial({"prompt=v1": hasty, "prompt=v2": careful})
    assert comparison.factors == ("model", "prompt")
    assert {summary.candidate_model for summary in comparison.arms.values()} == {SHARED_ARM_MODEL}
    (row,) = comparison.contrasts("accuracy")
    assert row["delta"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("arms", "said"),
    [
        ({"model=a, model=b": hasty, "model=c": careful}, "twice"),
        ({"model=a, _p=v1": hasty, "model=a, _p=v2": careful}, "does not start with '_'"),
        ({"model=a, prompt=v1": hasty, "model=a": careful}, "same factors in the same order"),
        ({"model=a, prompt=v1": hasty, "model=a, prompt= ": careful}, "is not one"),
        ({"model=a, prompt=v1": hasty, "a": careful}, "not both"),
    ],
    ids=["a factor twice", "unusable lever", "short key", "blank level", "plain name"],
)
async def test_an_unusable_factorial_is_refused_before_anything_runs(arms: dict[str, Any], said: str) -> None:
    host = callable_host(levers=("prompt",))
    with pytest.raises(ValueError, match=said):
        await _factorial(arms, host=host)
    assert off_loop(list_templates, host.storage, SCOPE) == []


async def test_arms_named_as_models_keep_their_names_as_the_runs_models() -> None:
    comparison = await _factorial({"model=hasty": hasty, "model=careful": careful})
    assert comparison.factors == ("model",)
    assert comparison.arms["model=careful"].candidate_model == "careful"
    assert comparison.arms["model=careful"].arm is None
    (row,) = comparison.contrasts("accuracy")
    assert (row["arm"], row["contrast"], row["control"]) == ("model=careful", "careful", "hasty")


async def test_a_callers_host_without_the_arm_lever_names_single_factor_arms_as_models() -> None:
    comparison = await _factorial({"hasty": hasty, "careful": careful}, host=callable_host())
    assert comparison.factors == ("model",)
    (row,) = comparison.contrasts("accuracy")
    assert (row["arm"], row["contrast"]) == ("careful", "careful")


async def test_a_callers_host_declaring_the_arm_lever_names_arms_on_it() -> None:
    host = as_host([Eval(CASES, {"hasty": hasty, "careful": careful}, ["label"])], store=":memory:")
    comparison = await _factorial({"hasty": hasty, "careful": careful}, host=host)
    (row,) = comparison.contrasts("accuracy")
    assert (row["arm"], row["contrast"], row["control"]) == ("careful", "careful", "hasty")
