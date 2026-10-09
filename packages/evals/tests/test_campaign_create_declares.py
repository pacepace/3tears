"""``campaign_create`` declares a design: the action and the operation take one, held to the gates every declaration meets.

Driven over the toy host through :meth:`MountedTool.call`, the path every transport takes. Pinned here:

- **A declared design is stored as declared, stamped by the surface**: its axes, what it held fixed and its
  questions, with ``declared_by`` the caller rather than anything the payload typed.
- **The control is addressed from a run**, and resolves to the same variant key ``set_campaign_control`` gives.
- **Every refusal names what was wrong**: an axis the host does not declare (with its vocabulary), a control run
  with no design or outside the campaign, a typed control beside a control run, and the old ``controls`` name.
- **Without a design the campaign is undeclared**, as it always was.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.actions import Caller, MountedTool, eval_catalogue, standard_tools
from threetears.evals.analysis.campaigns import set_campaign_control
from threetears.evals.ops import CampaignLine
from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_AXIS, toyhost_design
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, TOYHOST_SUBJECT, OpsFixture, ops_fixture


@pytest.fixture
def evals() -> MountedTool:
    return eval_catalogue().mount_all(standard_tools())[0]


def _design(**changes: Any) -> dict[str, Any]:
    """The toy host's declaration as an author sends it: no authorship, no date — the surface stamps both."""
    design = toyhost_design().model_dump(mode="json", exclude={"declared_at", "declared_by"})
    design.update(changes)
    return design


async def _create(tool: MountedTool, fixture: OpsFixture, caller: Caller = CALLER, **arguments: Any) -> Any:
    call = {
        "action": "campaign_create",
        "name": "declared at birth",
        "subject_id": TOYHOST_SUBJECT.subject_id,
        "behavior": fixture.campaign.behavior,
        "run_ids": list(fixture.campaign.run_ids),
        **arguments,
    }
    return await tool.call(call, host=fixture.host, caller=caller)


async def test_a_campaign_created_with_a_design_carries_it_stamped_by_the_caller(evals: MountedTool) -> None:
    fixture = ops_fixture()

    outcome = await _create(evals, fixture, declared_design=_design())

    assert not outcome.is_error, outcome.text
    stored = fixture.host.eval_host.storage.load_campaign(
        CampaignLine.model_validate(outcome.structured).id, TOYHOST_SCOPE
    )
    assert stored is not None and stored.declared_design is not None
    declared = stored.declared_design
    assert [axis.axis_id for axis in declared.axes] == [TOYHOST_AXIS]
    assert declared.held_fixed == toyhost_design().held_fixed
    assert [q.id for q in declared.questions] == [q.id for q in toyhost_design().questions]
    assert declared.declared_by == CALLER.identity, "authorship is the surface's record, not the payload's"
    assert declared.control is None


async def test_the_control_is_resolved_from_a_run_as_set_campaign_control_resolves_it(evals: MountedTool) -> None:
    fixture = ops_fixture()
    storage = fixture.host.eval_host.storage
    control_run = fixture.campaign.run_ids[0]

    created = await _create(evals, fixture, declared_design=_design(), control_run_id=control_run)
    designated_later = await _create(evals, fixture, declared_design=_design())

    assert not created.is_error, created.text
    with_control = storage.load_campaign(created.structured["id"], TOYHOST_SCOPE)
    later = set_campaign_control(
        storage,
        designated_later.structured["id"],
        TOYHOST_SCOPE,
        control_run,
        set_by="agent:test",
        profile=fixture.host.eval_host.profile,
    )
    assert with_control is not None and with_control.declared_design is not None
    assert later.declared_design is not None
    assert with_control.declared_design.control == later.declared_design.control
    assert with_control.declared_design.control is not None


@pytest.mark.parametrize(
    ("arguments", "says"),
    [
        pytest.param(
            {
                "declared_design": _design(
                    axes=[{"axis_id": "no_such_lever", "values": [{"content": "x", "display": "x"}]}]
                )
            },
            "no_such_lever",
            id="an-axis-the-host-does-not-declare",
        ),
        pytest.param({"control_run_id": "{run}"}, "declare a design", id="a-control-run-with-no-design"),
        pytest.param(
            {"declared_design": _design(), "control_run_id": "run-not-in-the-campaign"},
            "is not attached",
            id="a-control-run-outside-the-campaign",
        ),
        pytest.param(
            {"declared_design": _design(control="a" * 64), "control_run_id": "{run}"},
            "give one",
            id="a-typed-control-beside-a-control-run",
        ),
        pytest.param(
            {"declared_design": {**_design(), "controls": _design()["held_fixed"]}},
            "renamed `held_fixed`",
            id="the-old-name",
        ),
        pytest.param({"declared_design": _design(axes=[])}, "axes", id="no-axes"),
    ],
)
async def test_a_design_the_host_cannot_honour_is_refused_naming_why(
    evals: MountedTool, arguments: dict[str, Any], says: str
) -> None:
    fixture = ops_fixture()
    run = fixture.campaign.run_ids[0]
    resolved = {key: (value.format(run=run) if isinstance(value, str) else value) for key, value in arguments.items()}
    before = fixture.host.eval_host.storage.list_campaigns(TOYHOST_SCOPE)

    outcome = await _create(evals, fixture, **resolved)

    assert outcome.is_error and says in outcome.text, outcome.text
    assert fixture.host.eval_host.storage.list_campaigns(TOYHOST_SCOPE) == before, "a refused create writes nothing"


async def test_without_a_design_the_campaign_is_undeclared(evals: MountedTool) -> None:
    fixture = ops_fixture()

    outcome = await _create(evals, fixture)

    assert not outcome.is_error, outcome.text
    stored = fixture.host.eval_host.storage.load_campaign(outcome.structured["id"], TOYHOST_SCOPE)
    assert stored is not None and stored.declared_design is None
