"""A judge template is authored by the package, through an action: ``judge_template_create``.

``judge_cases_freeze`` freezes cases into a template of the judge kind, and a judge campaign launches one — but no
action authored one, so an agent could not start a judge campaign at all. A judge template is wholly the package's:
the judge kind, a name and an intent, nothing a host's kind would read (its cases carry everything a trial reads, and
the dispatch refuses any other field on it). So the action writes exactly that, on a host that launches the judge
kind, and the freeze's refusal of a template of another kind points at it.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from threetears.evals.actions import Caller, MountedTool, eval_catalogue, standard_tools
from threetears.evals.kernel import JUDGE_KIND, JUDGE_KIND_CONTRACT
from threetears.evals.ops import OpsHost, templates_list
from threetears.evals.run import LaunchHost, default_job_timeout, launchable_judge_kind
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.ops_support import ops_fixture

_CALLER = Caller(scope_id="judge-templates", identity="agent:test")


def _judging_host() -> OpsHost:
    profile = toyhost_profile()
    eval_host = dataclasses.replace(
        toyhost_host(),
        profile=dataclasses.replace(
            profile, kinds=(*profile.kinds, dataclasses.replace(JUDGE_KIND_CONTRACT, seats=None))
        ),
    )
    holder: list[LaunchHost] = []
    launch = LaunchHost(
        eval_host=eval_host,
        kinds={JUDGE_KIND: launchable_judge_kind(lambda: holder[0])},
        settings=lambda: TOYHOST_LAUNCH_SETTINGS,
        resolve_subject=lambda _subject, _scope: None,
        job_timeout_factory=default_job_timeout,
        world_placements=lambda _run: {},
    )
    holder.append(launch)
    return OpsHost(launch=launch)


def _tool(name: str) -> MountedTool:
    return next(tool for tool in eval_catalogue().mount_all(standard_tools()) if tool.action(name) is not None)


async def _create(host: OpsHost, **arguments: Any) -> Any:
    call = {
        "action": "judge_template_create",
        "template_name": "faithfulness judge",
        "intent": "Score as a person would.",
    }
    return await _tool("judge_template_create").call({**call, **arguments}, host=host, caller=_CALLER)


async def test_it_writes_a_template_of_the_judge_kind_and_nothing_else() -> None:
    host = _judging_host()
    outcome = await _create(host)
    assert not outcome.is_error, outcome.text
    stored = host.eval_host.storage.load_template(outcome.structured["id"], _CALLER.scope_id)
    assert stored is not None
    assert (stored.candidate_kind, stored.name, stored.intent) == (
        JUDGE_KIND,
        "faithfulness judge",
        "Score as a person would.",
    )
    assert (stored.rubric, stored.goal_state_checks, stored.world_seed.namespaces) == ([], [], {})


async def test_a_second_of_one_name_is_refused_with_a_remedy() -> None:
    host = _judging_host()
    await _create(host)
    outcome = await _create(host)
    assert outcome.is_error and "already exists" in outcome.text and "remedy:" in outcome.text


async def test_a_host_that_does_not_launch_the_judge_kind_is_refused_naming_what_registers_it() -> None:
    outcome = await _create(ops_fixture().host)
    assert outcome.is_error and "launchable_judge_kind" in outcome.text


async def test_the_freeze_of_another_kinds_template_points_at_the_action() -> None:
    fixture = ops_fixture()
    (template, *_) = templates_list(fixture.host.eval_host, fixture.campaign.scope_id).templates
    outcome = await _tool("judge_cases_freeze").call(
        {"action": "judge_cases_freeze", "template_id": template.id, "judged_run_ids": list(fixture.campaign.run_ids)},
        host=fixture.host,
        caller=Caller(scope_id=fixture.campaign.scope_id, identity="agent:test"),
    )
    assert outcome.is_error and "judge_template_create" in outcome.text
