"""Everything a campaign needs is an action: its runs, its design and its control, from an ad hoc start (#2673).

An operator on MCP or REST could create a campaign and archive it, and nothing else: a run launched later could not
join it, and a control could be named only at creation. Their refusals sent the operator to Python functions
(``analysis.add_runs_to_campaign``, ``set_campaign_control``) no action reaches. Pinned here, driven through
:meth:`MountedTool.call` as every transport drives it:

- **The whole path is actions**: create exploratory, ``campaign_runs_add`` a later run, ``campaign_update`` a design
  and its control from that run — the key resolved as ``set_campaign_control`` resolves it.
- **The design and its control are one write**: a refused control leaves the design unwritten too.
- **Membership keeps its rules**: a run outside the scope is not added, a run the campaign does not hold is not
  removed, and removing takes nothing but membership.
- **A refusal names an action**: the judge configs a second judge names are listed by ``judge_configs_list``.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.actions import MountedTool, eval_catalogue, standard_tools
from threetears.evals.analysis.campaigns import set_campaign_control
from threetears.evals.kernel import EvalCampaign
from threetears.evals.kernel.remedies import FIND_JUDGE_CONFIG
from threetears.evals.ops import CampaignLine, JudgeConfigListing
from threetears.evals.schema.models import JudgeConfig
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_design
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, TOYHOST_SUBJECT, OpsFixture, ops_fixture
from packages.evals.tests.off_loop import off_loop


@pytest.fixture
def evals() -> MountedTool:
    return eval_catalogue().mount_all(standard_tools())[0]


def _design(**changes: Any) -> dict[str, Any]:
    design = toyhost_design().model_dump(mode="json", exclude={"declared_at", "declared_by"})
    design.update(changes)
    return design


async def _call(tool: MountedTool, fixture: OpsFixture, action: str, **arguments: Any) -> Any:
    return await tool.call({"action": action, **arguments}, host=fixture.host, caller=CALLER)


def _stored(fixture: OpsFixture, campaign_id: str) -> EvalCampaign:
    campaign = off_loop(fixture.host.eval_host.storage.load_campaign, campaign_id, TOYHOST_SCOPE)
    assert campaign is not None
    return campaign


async def _create(tool: MountedTool, fixture: OpsFixture, run_ids: list[str], **arguments: Any) -> str:
    created = await _call(
        tool,
        fixture,
        "campaign_create",
        name="by actions",
        subject_id=TOYHOST_SUBJECT.subject_id,
        behavior=fixture.campaign.behavior,
        run_ids=run_ids,
        **({"exploratory": True} if "declared_design" not in arguments else {}),
        **arguments,
    )
    assert not created.is_error, created.text
    return str(created.structured["id"])


def _key_set_campaign_control_resolves(fixture: OpsFixture, run_id: str) -> str:
    """The control key the Python path resolves from ``run_id``, on a campaign of its own."""
    eval_host = fixture.host.eval_host
    designated = set_campaign_control(
        eval_host.storage, fixture.campaign.id, TOYHOST_SCOPE, run_id, set_by="op", profile=eval_host.profile
    )
    assert designated.declared_design is not None and designated.declared_design.control is not None
    return designated.declared_design.control


async def test_an_operator_using_only_actions_starts_ad_hoc_and_ends_declared_with_a_control(
    evals: MountedTool,
) -> None:
    fixture = off_loop(ops_fixture)
    first, later = fixture.campaign.run_ids[0], fixture.campaign.run_ids[1]
    campaign_id = await _create(evals, fixture, [first])

    added = await _call(evals, fixture, "campaign_runs_add", campaign_id=campaign_id, run_ids=[later])
    declared = await _call(
        evals, fixture, "campaign_update", campaign_id=campaign_id, declared_design=_design(), control_from_run_id=later
    )

    assert not added.is_error and added.structured["run_count"] == 2, added.text
    assert not declared.is_error, declared.text
    assert not CampaignLine.model_validate(declared.structured).exploratory
    stored = _stored(fixture, campaign_id)
    assert stored.run_ids == [first, later]
    assert stored.declared_design is not None and stored.declared_design.declared_by == CALLER.identity
    assert stored.declared_design.control == off_loop(_key_set_campaign_control_resolves, fixture, later)


async def test_control_from_run_id_alone_re_names_the_control_of_a_declared_campaign(evals: MountedTool) -> None:
    fixture = off_loop(ops_fixture)
    first, second = fixture.campaign.run_ids[0], fixture.campaign.run_ids[1]
    campaign_id = await _create(evals, fixture, [first, second], declared_design=_design(), control_from_run_id=first)
    before = _stored(fixture, campaign_id)

    outcome = await _call(evals, fixture, "campaign_update", campaign_id=campaign_id, control_from_run_id=second)

    assert not outcome.is_error, outcome.text
    after = _stored(fixture, campaign_id)
    assert after.declared_design is not None and before.declared_design is not None
    assert after.declared_design.control == off_loop(_key_set_campaign_control_resolves, fixture, second)
    assert after.declared_design.axes == before.declared_design.axes, "the rest of the declaration is kept"


async def test_a_null_control_clears_it_and_on_an_undeclared_campaign_is_no_refusal(evals: MountedTool) -> None:
    fixture = off_loop(ops_fixture)
    run = fixture.campaign.run_ids[0]
    declared = await _create(evals, fixture, [run], declared_design=_design(), control_from_run_id=run)
    exploratory = await _create(evals, fixture, [run])

    cleared = await _call(evals, fixture, "campaign_update", campaign_id=declared, control_from_run_id=None)
    nothing_to_clear = await _call(evals, fixture, "campaign_update", campaign_id=exploratory, control_from_run_id=None)

    assert not cleared.is_error and not nothing_to_clear.is_error, (cleared.text, nothing_to_clear.text)
    stored = _stored(fixture, declared).declared_design
    assert stored is not None and stored.control is None
    assert _stored(fixture, exploratory).declared_design is None


@pytest.mark.parametrize(
    ("start", "arguments", "says"),
    [
        pytest.param("exploratory", {"control_from_run_id": "{run}"}, "declared_design", id="no-declaration"),
        pytest.param(
            "exploratory",
            {"declared_design": "{design}", "control_from_run_id": "run-not-in-the-campaign"},
            "is not attached",
            id="a-run-the-campaign-does-not-hold",
        ),
        pytest.param(
            "exploratory",
            {"declared_design": "{typed}", "control_from_run_id": "{run}"},
            "give one",
            id="a-typed-control-beside-a-control-run",
        ),
        pytest.param(
            "declared",
            {"declared_design": None, "exploratory": True, "control_from_run_id": "{run}"},
            "withdrawing",
            id="a-control-on-a-withdrawn-declaration",
        ),
    ],
)
async def test_a_control_with_nothing_to_hold_it_is_refused_and_the_design_is_not_written(
    evals: MountedTool, start: str, arguments: dict[str, Any], says: str
) -> None:
    fixture = off_loop(ops_fixture)
    run = fixture.campaign.run_ids[0]
    starting = {"declared_design": _design()} if start == "declared" else {}
    campaign_id = await _create(evals, fixture, [run], **starting)
    before = _stored(fixture, campaign_id)
    values = {"{run}": run, "{design}": _design(), "{typed}": _design(control="a" * 64)}
    resolved = {key: values.get(value, value) if isinstance(value, str) else value for key, value in arguments.items()}

    outcome = await _call(evals, fixture, "campaign_update", campaign_id=campaign_id, **resolved)

    assert outcome.is_error and says in outcome.text, outcome.text
    assert _stored(fixture, campaign_id) == before, "the design and its control are one write"


async def test_a_run_outside_the_scope_is_not_added_and_none_beside_it_is(evals: MountedTool) -> None:
    fixture = off_loop(ops_fixture)
    elsewhere = make_eval_run(id="run-elsewhere", scope_id="another-scope")
    off_loop(fixture.host.eval_host.storage.save_eval_run, elsewhere)
    campaign_id = await _create(evals, fixture, [fixture.campaign.run_ids[0]])
    before = _stored(fixture, campaign_id)

    outcome = await _call(
        evals,
        fixture,
        "campaign_runs_add",
        campaign_id=campaign_id,
        run_ids=[fixture.campaign.run_ids[1], elsewhere.id],
    )

    assert outcome.is_error and "run-elsewhere" in outcome.text and "runs_list" in outcome.text, outcome.text
    assert _stored(fixture, campaign_id) == before


async def test_adding_a_run_twice_holds_it_once(evals: MountedTool) -> None:
    fixture = off_loop(ops_fixture)
    run = fixture.campaign.run_ids[0]
    campaign_id = await _create(evals, fixture, [run])

    outcome = await _call(evals, fixture, "campaign_runs_add", campaign_id=campaign_id, run_ids=[run, run])

    assert not outcome.is_error, outcome.text
    assert _stored(fixture, campaign_id).run_ids == [run]


async def test_removing_takes_only_membership_and_refuses_a_run_the_campaign_does_not_hold(
    evals: MountedTool,
) -> None:
    fixture = off_loop(ops_fixture)
    first, second = fixture.campaign.run_ids[0], fixture.campaign.run_ids[1]
    campaign_id = await _create(evals, fixture, [first, second])

    stranger = await _call(evals, fixture, "campaign_runs_remove", campaign_id=campaign_id, run_ids=[first, "nope"])
    removed = await _call(evals, fixture, "campaign_runs_remove", campaign_id=campaign_id, run_ids=[first])

    assert stranger.is_error and "'nope'" in stranger.text, stranger.text
    assert not removed.is_error and removed.structured["run_count"] == 1, removed.text
    assert _stored(fixture, campaign_id).run_ids == [second]
    assert off_loop(fixture.host.eval_host.storage.load_eval_run, first, TOYHOST_SCOPE) is not None, "a run stays"
    assert first in _stored(fixture, fixture.campaign.id).run_ids, "and stays in every other campaign"


@pytest.mark.parametrize("action", ["campaign_runs_add", "campaign_runs_remove"])
async def test_a_membership_change_naming_no_run_is_refused(evals: MountedTool, action: str) -> None:
    fixture = off_loop(ops_fixture)
    campaign_id = await _create(evals, fixture, [fixture.campaign.run_ids[0]])

    outcome = await _call(evals, fixture, action, campaign_id=campaign_id, run_ids=[])

    assert outcome.is_error and "no run was named" in outcome.text, outcome.text


async def test_judge_configs_list_names_the_configs_a_second_judge_can_map_a_dimension_to(evals: MountedTool) -> None:
    fixture = off_loop(ops_fixture)
    config = JudgeConfig(
        scope_id=TOYHOST_SCOPE, name="strict accuracy", rubric_dim_id="toy.accuracy", prompt_template="Grade it."
    )
    other = JudgeConfig(scope_id=TOYHOST_SCOPE, name="tone", rubric_dim_id="toy.tone", prompt_template="Grade it.")
    for each in (config, other):
        off_loop(fixture.host.eval_host.storage.save_judge_config, each)

    every = await _call(evals, fixture, "judge_configs_list")
    one_dim = await _call(evals, fixture, "judge_configs_list", dim_filter="toy.accuracy")

    assert not every.is_error and not one_dim.is_error, (every.text, one_dim.text)
    assert {line.id for line in JudgeConfigListing.model_validate(every.structured).judge_configs} >= {
        config.id,
        other.id,
    }
    assert [line.id for line in JudgeConfigListing.model_validate(one_dim.structured).judge_configs] == [config.id]
    assert f"{config.id}: strict accuracy — dimension toy.accuracy" in one_dim.text
    assert "judge_configs_list" in FIND_JUDGE_CONFIG and "run.list_judge_configs" not in FIND_JUDGE_CONFIG


async def test_a_second_judge_naming_a_dimension_wrongly_is_a_refusal_not_a_defect(evals: MountedTool) -> None:
    """``second_judge_config_ids`` naming a dimension without its context raised out of the handler as a defect."""
    fixture = off_loop(ops_fixture)

    outcome = await _call(
        evals,
        fixture,
        "judge_second_estimate",
        run_id=fixture.campaign.run_ids[0],
        second_judge_model="another-model",
        second_judge_config_ids={"accuracy": "some-config"},
    )

    assert outcome.is_error and outcome.refusal is not None and outcome.refusal.kind == "validation_failed"
    assert "invalid second judge" in outcome.text and "judge_configs_list" in outcome.text, outcome.text


async def test_a_host_that_cannot_run_judge_campaigns_sends_the_caller_to_its_operator(evals: MountedTool) -> None:
    fixture = off_loop(ops_fixture)

    outcome = await _call(evals, fixture, "judge_template_create", template_name="judges", intent="measure judges")

    assert outcome.is_error, outcome.text
    remedy = outcome.text.split("remedy: ", 1)[1]
    assert "operator" in remedy and "run." not in remedy and "judge_second" in remedy, remedy
