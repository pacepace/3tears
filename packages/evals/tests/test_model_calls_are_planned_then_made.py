"""Every model call the package makes outside a run is planned, then admitted by its caller, then made.

A host prices and refuses spend in the call and makes the model calls in a job. That holds only when the package
hands it each call before making it: a public planner returning the
:class:`~threetears.evals.kernel.PlannedCall`, so a host can quote it and admit it under its budget in the call,
and an executor that takes the :class:`~threetears.evals.kernel.AdmittedCall` and only makes it. A function that
plans, admits and makes in one step (as ``propose_draft`` did) leaves the host nothing to price before the job.

Two holds:

* **No function both admits and makes.** Read from the source: no function's own body (its nested functions
  apart) calls both ``.admit(`` and ``.generate(``. An admission and the making of what it admitted are two
  steps, in two places, so the call half and the job half can be told apart.
* **Each out-of-run caller is a planner and an executor.** The audit, as a table: every package function that
  calls a model outside a run, its planner (returning planned calls) and its executor (taking admitted calls, or
  returning work whose calls were admitted in the call). A new model caller must join it.

A run's own calls (the per-dimension judge, the simulated user, a reporter or judge kind's cell) are not here: the
run's cost cap reserves each at its ceiling as it is sent (``CellSink.reserve``), inside work a launch admitted.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import pytest

from threetears.evals import analysis, gen, kernel, run
from threetears.evals.kernel import AdmittedCall, PlannedCall

_SOURCE = Path(kernel.__file__).resolve().parent.parent


def _own_calls(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """The attribute names ``function``'s own body calls — ``x.admit(...)`` gives ``admit`` — its nested defs apart."""
    called: set[str] = set()
    stack: list[ast.AST] = list(function.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)
        stack.extend(ast.iter_child_nodes(node))
    return called


def _admits_and_makes() -> list[str]:
    found = []
    for path in sorted(_SOURCE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and {"admit", "generate"} <= _own_calls(node):
                found.append(f"{path.relative_to(_SOURCE)}:{node.lineno} {node.name}")
    return found


def test_no_function_both_admits_a_call_and_makes_it() -> None:
    assert _admits_and_makes() == [], "admit in the call, make in the job: split these into planner and executor"


def _mentions(annotation: Any, wanted: type) -> bool:
    """Whether ``annotation`` is ``wanted``, holds it as an argument, or is a package class with a field of it."""
    if annotation is wanted:
        return True
    if inspect.isclass(annotation) and annotation.__module__.startswith("threetears.evals"):
        fields = getattr(annotation, "__annotations__", {})
        if any(wanted.__name__ in str(field) for field in fields.values()):
            return True
    return any(_mentions(arg, wanted) for arg in get_args(annotation)) if get_origin(annotation) else False


def _namespace() -> dict[str, Any]:
    """Every public name, and what the package imports only for type checking, to read annotations against."""
    import concurrent.futures

    from threetears.evals import PUBLIC_ROOTS

    names: dict[str, Any] = {"Executor": concurrent.futures.Executor}
    for root in PUBLIC_ROOTS:
        module = importlib.import_module(root)
        names |= {name: getattr(module, name) for name in module.__all__}
    return names


def _hints(function: Callable[..., Any]) -> dict[str, Any]:
    return get_type_hints(function, localns=_namespace())


def _returns(function: Callable[..., Any]) -> Any:
    return _hints(function)["return"]


def _takes(function: Callable[..., Any]) -> list[Any]:
    hints = _hints(function)
    return [hints[name] for name in inspect.signature(function).parameters if name in hints]


#: The audit: every package function that calls a model outside a run, with its planner and its executor.
#: A planner returns the calls (``PlannedCall``, or a mapping of them); an executor takes what was admitted.
OUT_OF_RUN_CALLERS: Mapping[str, tuple[Callable[..., Any], Callable[..., Any]]] = {
    "rubric proposer (capability and boundary)": (gen.plan_draft, gen.propose_draft),
    "case generation (an llm variation axis)": (kernel.plan_variation_calls, gen.generate_variations),
}

#: Out-of-run work the engine itself starts as a job: its builder admits every call before the first is made and
#: returns work that only makes them, and its estimate quotes the same plan. A host calls the builder in its call,
#: except for an analysis generation, whose builder assembles the campaign's whole bundle and so runs in its job,
#: before the job's first call (``analysis_generate``).
ADMITTED_IN_THE_CALL: Mapping[str, tuple[Callable[..., Any], Callable[..., Any]]] = {
    "analysis generation": (analysis.estimate_analysis_generation, analysis.prepare_analysis_generation),
    "second judge and drift check": (run.estimate_second_judge, run.second_judge_work),
    "judge repeat": (run.estimate_judge_repeat, run.judge_repeat_work),
    "judge temperature comparison": (run.estimate_judge_temperature_comparison, run.judge_temperature_work),
}


@pytest.mark.parametrize("name", list(OUT_OF_RUN_CALLERS))
def test_each_out_of_run_caller_has_a_public_planner_and_an_executor_of_admitted_calls(name: str) -> None:
    planner, executor = OUT_OF_RUN_CALLERS[name]
    assert _mentions(_returns(planner), PlannedCall), f"{name}: the planner returns its PlannedCalls"
    assert any(_mentions(taken, AdmittedCall) for taken in _takes(executor)), (
        f"{name}: the executor takes the calls its caller admitted, and admits none of its own"
    )


@pytest.mark.parametrize("name", list(ADMITTED_IN_THE_CALL))
def test_each_engine_job_admits_in_the_call_and_its_estimate_quotes_the_same(name: str) -> None:
    estimate, admits = ADMITTED_IN_THE_CALL[name]
    assert inspect.iscoroutinefunction(estimate) and inspect.iscoroutinefunction(admits)
    assert "estimate" in estimate.__name__
