"""The frontier over a quick comparison: one subject, so its models rank on one frontier, from Python, an action or the CLI.

``compare`` used to make each arm's model its run's subject, and the frontier ranks within a subject, so on
the quick path it only ever compared arms that shared a model: "the cheapest model that clears the bar" could
not be asked. The arms of one comparison now share its subject, the model a lever the frontier ranks across;
and the frontier is reachable as the ``scope_frontier`` action and the ``frontier`` command, both answering
from :func:`~threetears.evals.ops.scope_frontier`.

Mutations that turn this file red: a subject per arm again; the action or the CLI reading a different lens, a
different bar or a different status from the Python call; the CLI not mounting ``frontier``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest

from threetears.evals.actions import Caller, eval_catalogue, standard_tools
from threetears.evals.ops import OpsHost, frontier_text, scope_frontier
from threetears.evals.quick import Answer, Judge, compare, run_cli
from threetears.evals.run import LaunchHost, LaunchSettings, default_job_timeout

CASES = [{"n": index} for index in range(24)]


async def big(case: Mapping[str, Any]) -> Answer:
    return Answer("GOOD", cost_usd=0.01)


async def small(case: Mapping[str, Any]) -> Answer:
    return Answer("BAD" if case["n"] == 0 else "GOOD", cost_usd=0.001)


async def _stand_in_judge(*, system: str, user: str, response_format: dict[str, Any] | None = None) -> SimpleNamespace:
    """An offline judge passing an answer that says GOOD and failing one that does not."""
    reply = {"reasoning": "stand-in", "criteria_scores": {"answer.sound": 5 if "GOOD" in user else 1}}
    return SimpleNamespace(
        content=json.dumps(reply),
        input_tokens=None,
        output_tokens=None,
        reasoning_tokens=None,
        cost_usd=0.0,
        price_source="stand-in",
        model="stand-in",
        served_model=None,
        stop_reason="end_turn",
        temperature=0.0,
    )


def _judge() -> Judge:
    return Judge(
        client=SimpleNamespace(generate=_stand_in_judge),
        model="stand-in",
        rubric={"sound": "The answer is sound."},
        case_material=lambda case: f"Case {case['n']}",
    )


async def test_a_comparison_of_two_models_is_one_frontier_naming_the_cheaper_one() -> None:
    comparison = await compare(
        CASES, {"big": big, "small": small}, control="big", scope_id="ranked", k=2, factors=("model",), judge=_judge()
    )
    ranked = scope_frontier(comparison.host, "ranked", bar=0.5)
    (subject,) = ranked.subjects
    assert subject.subject_id == comparison.name
    assert sorted(point.model for point in subject.points) == ["big", "small"]
    assert subject.verdict is not None and subject.verdict.model == "small"
    assert subject.verdict.cost_decision == "shown_cheapest"
    assert "verdict: small · " in frontier_text(ranked)


async def test_named_arms_share_the_comparison_s_subject_too() -> None:
    comparison = await compare(CASES[:4], {"v1": big, "v2": small}, control="v1", scope_id="named", k=1, judge=_judge())
    (subject,) = scope_frontier(comparison.host, "named").subjects
    assert len(subject.points) == 2
    assert {summary.arm for summary in comparison.arms.values()} == {"v1", "v2"}


async def test_the_action_and_the_cli_return_the_python_call_s_frontier(capsys: pytest.CaptureFixture[str]) -> None:
    comparison = await compare(
        CASES, {"big": big, "small": small}, control="big", scope_id="ranked", k=2, factors=("model",), judge=_judge()
    )
    host = comparison.host
    expected = scope_frontier(host, "ranked", bar=0.5).model_dump(mode="json")

    tool = eval_catalogue().mount_all(standard_tools())[0]
    settings = LaunchSettings(
        max_launch_arms=1,
        max_admitted_runs=1,
        judge_concurrency=1,
        enforcement_enabled=False,
        max_cost_usd=1.0,
        max_metered_calls=None,
        max_out_of_run_cost_usd=1.0,
    )
    ops_host = OpsHost(
        launch=LaunchHost(eval_host=host, kinds={}, settings=lambda: settings, job_timeout_factory=default_job_timeout)
    )
    called = await tool.call(
        {"action": "scope_frontier", "bar": 0.5}, host=ops_host, caller=Caller(scope_id="ranked", identity="t")
    )
    assert not called.is_error, called.text
    assert called.structured == expected
    assert called.text == frontier_text(scope_frontier(host, "ranked", bar=0.5))

    assert run_cli(["frontier", "--scope", "ranked", "--bar", "0.5", "--json"], host_factory=lambda: host) == 0
    assert json.loads(capsys.readouterr().out) == expected
