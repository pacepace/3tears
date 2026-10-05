"""One repeat count when a launch names none, on every surface that launches or prices one.

``DEFAULT_LAUNCH_K_RUNS`` is the default; each surface below reads it rather than restating a number, and
each is pinned by what it DOES with an omitted ``k`` — the run it stores, the estimate it prices, the
launch it hands on — not by the literal in its signature, since a surface can default correctly and still
drop the value on the way to the launch.

Surfaces: the ``run_launch`` and ``launch_estimate`` actions, ops ``run_launch`` (``LaunchArguments``) and
ops ``launch_estimate``, the command line's ``run``, and ``run_eval``. Each was written against a constant
of 3; a test asserting ``k_runs == 3`` would pass for a surface hard-coding 3, so every assertion here
reads the constant.

Mutations that turn this file red (each applied to a saved copy and restored from it): the literal ``1``
in place of the constant in ``ops.runs.LaunchArguments.k_runs``, ``ops.lenses.launch_estimate``,
``quick.cli``'s ``--k``, ``quick.one_call.run_eval``, and ``actions.engine``'s ``RunLaunchParams`` and
``LaunchEstimateParams``; and ``quick.cli._launch`` handing on ``k_runs=1`` in place of ``args.k``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals.actions import MountedTool, eval_catalogue, standard_tools
from threetears.evals.analysis import CostEstimate
from threetears.evals.contracts import DEFAULT_LAUNCH_K_RUNS
from threetears.evals.ops import JobsStarted, LaunchArguments, launch_estimate, run_launch
from threetears.evals.quick import run_cli, run_eval
from threetears.evals.quick import cli as quick_cli
from threetears.evals.run import start_run
from packages.evals.tests.factories import make_test_case
from packages.evals.tests.fixtures.courierhost import (
    COURIER_SCOPE,
    COURIER_SUBJECT,
    COURIER_TEMPLATE_ID,
    courier_launch_host,
)
from packages.evals.tests.fixtures.toyhost.run import toyhost_template
from packages.evals.tests.ops_support import CALLER, RUN_MODELS, TOYHOST_SCOPE, TOYHOST_SUBJECT, OpsFixture, ops_fixture


def test_the_default_is_more_than_one_observation() -> None:
    """The constant's reason: one observation per case cannot tell a setting from the model's own variance.

    Also what makes every other assertion here discriminate: were the default 1, a surface still
    hard-coding 1 would agree with it.
    """
    assert DEFAULT_LAUNCH_K_RUNS > 1


@pytest.fixture
def evals() -> MountedTool:
    return eval_catalogue().mount_all(standard_tools())[0]


def _stored_k(fixture: OpsFixture, started: JobsStarted) -> set[int]:
    storage = fixture.host.eval_host.storage
    runs = [storage.load_eval_run(job.target_id, TOYHOST_SCOPE) for job in started.jobs]
    assert runs and all(run is not None for run in runs)
    return {run.k_runs for run in runs if run is not None}


async def test_ops_run_launch_stores_the_default_when_the_arguments_name_none() -> None:
    fixture = ops_fixture()
    arguments = LaunchArguments(
        template_id=toyhost_template().id, subject_id=TOYHOST_SUBJECT.subject_id, models=list(RUN_MODELS)
    )

    started = await run_launch(fixture.host, arguments, TOYHOST_SCOPE)

    assert _stored_k(fixture, started) == {DEFAULT_LAUNCH_K_RUNS}


async def test_the_run_launch_action_stores_the_default_when_the_call_names_none(evals: MountedTool) -> None:
    fixture = ops_fixture()
    arguments = {
        "action": "run_launch",
        "template_id": toyhost_template().id,
        "subject_id": TOYHOST_SUBJECT.subject_id,
        "models": list(RUN_MODELS),
    }

    outcome = await evals.call(arguments, host=fixture.host, caller=CALLER)

    assert not outcome.is_error, outcome.text
    assert _stored_k(fixture, JobsStarted.model_validate(outcome.structured)) == {DEFAULT_LAUNCH_K_RUNS}


def _with_cases(fixture: OpsFixture) -> str:
    """Store two of the toy template's cases, so an estimate has a case count to multiply; return its id."""
    template = toyhost_template()
    for index in range(2):
        fixture.host.eval_host.storage.save_test_case(
            make_test_case(id=f"case-{index}", scope_id=TOYHOST_SCOPE, template_id=template.id)
        )
    return template.id


def test_ops_launch_estimate_prices_the_default_when_the_call_names_none() -> None:
    fixture = ops_fixture()

    estimate = launch_estimate(fixture.host, TOYHOST_SCOPE, template_id=_with_cases(fixture), models=["m"])

    assert estimate.k_runs == DEFAULT_LAUNCH_K_RUNS


async def test_the_launch_estimate_action_prices_the_default_when_the_call_names_none(evals: MountedTool) -> None:
    fixture = ops_fixture()
    arguments = {"action": "launch_estimate", "template_id": _with_cases(fixture), "models": ["m"]}

    outcome = await evals.call(arguments, host=fixture.host, caller=CALLER)

    assert not outcome.is_error, outcome.text
    assert CostEstimate.model_validate(outcome.structured).k_runs == DEFAULT_LAUNCH_K_RUNS


def test_the_command_line_launches_the_default_when_run_names_no_k(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pinned at the launch the command hands on AND at the run it prints, so a parser default the
    handler then ignores goes red too."""
    seen: list[int] = []

    async def recording(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs["k_runs"])
        return await start_run(*args, **kwargs)

    monkeypatch.setattr(quick_cli, "start_run", recording)
    argv = ["run", "--scope", COURIER_SCOPE, "--template", COURIER_TEMPLATE_ID, "--subject", COURIER_SUBJECT.subject_id]

    assert run_cli([*argv, "--model", "planner-lite"], host_factory=courier_launch_host) == 0

    assert seen == [DEFAULT_LAUNCH_K_RUNS]
    assert f"x k={DEFAULT_LAUNCH_K_RUNS}" in capsys.readouterr().out


async def _answer(case: Mapping[str, Any]) -> int:
    return 1


def _graded(case: Mapping[str, Any], answer: Any) -> float:
    return 1.0


async def test_run_eval_repeats_each_case_the_default_number_of_times_when_given_no_k() -> None:
    summary = await run_eval([{"q": 1}, {"q": 2}], _answer, [_graded], scope_id="k-default")

    assert summary.k_runs == DEFAULT_LAUNCH_K_RUNS
    assert summary.n_results == 2 * DEFAULT_LAUNCH_K_RUNS
