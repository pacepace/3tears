"""A second factor is a lever of its own: ``run_eval(levers=)`` states it, ``compare(factors=)`` crosses it with the model.

``run_eval`` keys a run's variant by its model; a lever stated beside it (``levers={"prompt": "v2"}``) is
frozen onto the run as an overlay of the callable kind, so two runs of one model at two prompts are two
variants, and the campaign can declare the prompt as an axis. ``compare`` keyed by ``(model, prompt)``
runs each cell of the design as one arm, declares one axis per factor, and :meth:`Comparison.against`
reads the same runs against another control without running anything again.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals.analysis import variant_key_of_run
from threetears.evals.contracts import ValidationFailedError
from threetears.evals.contracts.host import CANDIDATE_MODEL_LEVER, SweepableValue
from threetears.evals.quick import (
    CALLABLE_KIND_CONTRACT,
    Comparison,
    callable_host,
    callable_kind_contracts,
    compare,
    run_eval,
)
from threetears.evals.run import list_runs, list_templates

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


def _expected(case: Mapping[str, Any]) -> str:
    return str(case["label"])


# --- run_eval(levers=) ---------------------------------------------------------------------------------


async def test_a_lever_beside_the_model_is_frozen_on_the_run_and_splits_the_variant() -> None:
    host = callable_host(levers=("prompt",))
    one = await run_eval(
        CASES, careful, expected=_expected, scope_id=SCOPE, host=host, k=1, model="m", levers={"prompt": "v1"}
    )
    two = await run_eval(
        CASES, careful, expected=_expected, scope_id=SCOPE, host=host, k=1, model="m", levers={"prompt": "v2"}
    )
    runs = {run.id: run for run in list_runs(host, SCOPE)}
    assert runs[one.run_id].overlays == {"prompt": "v1"} and runs[two.run_id].overlays == {"prompt": "v2"}
    keys = [variant_key_of_run(host.storage.query_eval_results_by_run(s.run_id, SCOPE)) for s in (one, two)]
    assert keys[0] != keys[1], "same model, same cases, different prompt: two variants"
    assert host.profile.sweepables.get("callable.prompt") is not None


async def test_with_no_host_run_eval_declares_exactly_the_levers_it_is_handed() -> None:
    summary = await run_eval(CASES, careful, expected=_expected, scope_id=SCOPE, k=1, levers={"prompt": "v1"})
    assert summary.status == "completed"


@pytest.mark.parametrize(
    ("levers", "said"),
    [
        ({"prompt": "v1", "temperature": "0"}, "temperature"),
        ({}, "prompt"),
        ({"prompt": " "}, "prompt"),
    ],
    ids=["an undeclared lever", "a declared lever left out", "a blank level"],
)
async def test_a_launch_must_state_every_declared_lever_and_no_other(levers: dict[str, str], said: str) -> None:
    host = callable_host(levers=("prompt",))
    with pytest.raises(ValidationFailedError, match=said):
        await run_eval(CASES, careful, expected=_expected, scope_id=SCOPE, host=host, k=1, levers=levers or None)


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
    summary = await run_eval(
        CASES, careful, expected=_expected, scope_id=SCOPE, host=host, k=1, levers={"prompt": "v1"}
    )
    assert summary.status == "completed"


# --- compare(factors=) ---------------------------------------------------------------------------------


async def _factorial(**overrides: Any) -> Comparison:
    arguments: dict[str, Any] = {
        "candidates": {("a", "v1"): hasty, ("a", "v2"): careful, ("b", "v1"): careful, ("b", "v2"): careful},
        "factors": ("model", "prompt"),
        "expected": _expected,
        "control": ("a", "v1"),
        "scope_id": SCOPE,
        "k": 2,
    }
    arguments.update(overrides)
    candidates = arguments.pop("candidates")
    return await compare(CASES, candidates, **arguments)


async def test_each_cell_runs_as_one_arm_at_its_model_and_its_lever() -> None:
    comparison = await _factorial()
    assert comparison.factors == ("model", "prompt")
    assert comparison.name == "model × prompt"
    assert [summary.candidate_model for summary in comparison.arms.values()] == ["a", "a", "b", "b"]
    runs = {run.id: run for run in list_runs(comparison.host, SCOPE)}
    assert [runs[s.run_id].overlays for s in comparison.arms.values()] == [{"prompt": p} for p in ("v1", "v2") * 2]


async def test_the_campaign_declares_one_axis_per_factor_at_the_levels_the_runs_carry() -> None:
    comparison = await _factorial()
    campaign = comparison.host.storage.load_campaign(comparison.campaign_id, SCOPE)
    assert campaign is not None and campaign.declared_design is not None
    design = campaign.declared_design
    model_axis, prompt_axis = design.axes
    assert (model_axis.axis_id, prompt_axis.axis_id) == (CANDIDATE_MODEL_LEVER, "callable.prompt")
    assert [value.content_hash for value in prompt_axis.values] == [
        SweepableValue.of(level, display=level).content_hash for level in ("v1", "v2")
    ]
    assert design.control == variant_key_of_run(
        comparison.host.storage.query_eval_results_by_run(comparison.arms[("a", "v1")].run_id, SCOPE)
    )
    assert design.intended_repetitions == 2


async def test_every_arm_is_named_by_both_coordinates_and_tested_against_the_control() -> None:
    comparison = await _factorial()
    rows = {row["contrast"]: row for row in comparison.contrasts("accuracy")}
    assert set(rows) == {"callable.prompt=v2, model=a", "callable.prompt=v1, model=b", "callable.prompt=v2, model=b"}
    assert {row["control"] for row in rows.values()} == {"callable.prompt=v1, model=a"}
    assert all(row["delta"] == pytest.approx(0.5) for row in rows.values()), "the careful rule reads every case"
    # The arms return plain answers, so no spend was observed and cost is not a reading to test.
    assert {row["reading"] for row in comparison.contrasts()} == {"accuracy"}


async def test_against_reads_the_same_runs_against_another_control_and_runs_nothing() -> None:
    comparison = await _factorial()
    n_runs = len(list_runs(comparison.host, SCOPE))
    on_b = comparison.against(("b", "v1"))
    assert len(list_runs(comparison.host, SCOPE)) == n_runs
    assert on_b.campaign_id != comparison.campaign_id
    assert on_b.arms == comparison.arms and on_b.control == ("b", "v1")
    assert on_b.name == "model × prompt, against model=b, prompt=v1"
    rows = {row["contrast"]: row for row in on_b.contrasts("accuracy")}
    assert rows["callable.prompt=v2, model=b"]["control"] == "callable.prompt=v1, model=b"
    assert rows["callable.prompt=v2, model=b"]["verdict"] == "not separated from the control"
    assert rows["callable.prompt=v1, model=a"]["delta"] == pytest.approx(-0.5)
    with pytest.raises(ValueError, match="names no arm"):
        comparison.against(("c", "v1"))


async def test_a_one_factor_comparison_is_read_against_another_control_too() -> None:
    comparison = await compare(
        CASES, {"hasty": hasty, "careful": careful}, expected=_expected, control="hasty", scope_id=SCOPE, k=2
    )
    assert comparison.factors == ("candidate",)
    flipped = comparison.against("careful")
    (row,) = flipped.contrasts("accuracy")
    assert (row["contrast"], row["control"], row["verdict"]) == (
        "candidate=hasty",
        "candidate=careful",
        "regressed from the control",
    )


@pytest.mark.parametrize(
    ("overrides", "said"),
    [
        ({"factors": ("prompt",), "candidates": {("v1",): hasty, ("v2",): careful}, "control": ("v1",)}, "leave out"),
        ({"factors": ("model", "model")}, "more than once"),
        ({"factors": "model"}, "not one string"),
        ({"factors": ("model", "_p")}, "does not start with '_'"),
        ({"candidates": {("a", "v1"): hasty, ("a",): careful}}, r"\('a',\) is not"),
        ({"candidates": {("a", "v1"): hasty, ("a", " "): careful}}, "non-blank level"),
        ({"candidates": {("a", "v1"): hasty, "a": careful}}, "'a' is not"),
        ({"control": ("z", "v1")}, "names no arm"),
    ],
    ids=[
        "no model factor",
        "model twice",
        "one string",
        "unusable lever",
        "short key",
        "blank level",
        "str key",
        "control",
    ],
)
async def test_an_unusable_factorial_is_refused_before_anything_runs(overrides: dict[str, Any], said: str) -> None:
    host = callable_host(levers=("prompt",))
    with pytest.raises(ValueError, match=said):
        await _factorial(host=host, **overrides)
    assert list_templates(host.storage, SCOPE) == []


async def test_arms_keyed_as_models_with_factors_model_keep_their_names_as_the_runs_models() -> None:
    comparison = await compare(
        CASES,
        {"hasty": hasty, "careful": careful},
        expected=_expected,
        control="hasty",
        factors=("model",),
        scope_id=SCOPE,
        k=2,
    )
    assert comparison.factors == ("model",)
    assert comparison.arms["careful"].candidate_model == "careful" and comparison.arms["careful"].arm is None
    (row,) = comparison.contrasts("accuracy")
    assert (row["arm"], row["contrast"], row["control"]) == ("careful", "model=careful", "model=hasty")


async def test_a_callers_host_without_the_arm_lever_names_single_factor_arms_as_models() -> None:
    comparison = await compare(
        CASES,
        {"hasty": hasty, "careful": careful},
        expected=_expected,
        control="hasty",
        host=callable_host(),
        scope_id=SCOPE,
        k=2,
    )
    assert comparison.factors == ("model",)
    (row,) = comparison.contrasts("accuracy")
    assert (row["arm"], row["contrast"]) == ("careful", "model=careful")


async def test_a_callers_host_declaring_the_arm_lever_names_arms_on_it() -> None:
    comparison = await compare(
        CASES,
        {"hasty": hasty, "careful": careful},
        expected=_expected,
        control="hasty",
        host=callable_host(arms=True),
        scope_id=SCOPE,
        k=2,
    )
    (row,) = comparison.contrasts("accuracy")
    assert (row["arm"], row["contrast"], row["control"]) == ("careful", "candidate=careful", "candidate=hasty")
