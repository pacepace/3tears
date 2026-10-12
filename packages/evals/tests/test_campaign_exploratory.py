"""A campaign is declared or exploratory, the caller says which, and ``campaign_update`` moves it between the two (#2673).

A campaign created with no design used to be exploratory by default, so a caller who forgot the design got a report
saying its readings confirm nothing without ever having chosen that; and nothing on MCP or REST could declare one
afterwards, though ``update_campaign`` could and the control refusal sent the operator to it. Pinned here:

- **A create says which**: ``declared_design`` or ``exploratory``, exactly one — neither is refused naming both,
  both are refused as a contradiction — through ``analysis.create_campaign`` and the ``campaign_create`` action.
- **``exploratory`` is never stored**: the campaign carries no such field, and being exploratory is read off
  ``declared_design is None``, so declaring a design through ``campaign_update`` makes a campaign declared.
- **The way back is said too**: ``declared_design: null`` withdraws a declaration only beside ``exploratory: true``;
  ``exploratory: true`` beside a design, or on a declared campaign the update leaves declared, is refused.
- **``campaign_update`` is an action** like the others, and the refusal that sends an operator to declare a design
  names it.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.actions import MountedTool, eval_catalogue, standard_tools
from threetears.evals.analysis.campaigns import create_campaign, set_campaign_control, update_campaign
from threetears.evals.kernel import EvalCampaign
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.ops import CampaignLine
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_design
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, TOYHOST_SUBJECT, OpsFixture, ops_fixture
from packages.evals.tests.off_loop import off_loop


@pytest.fixture
def evals() -> MountedTool:
    return eval_catalogue().mount_all(standard_tools())[0]


def _design() -> dict[str, Any]:
    """The toy host's declaration as an author sends it: no authorship, no date — the surface stamps both."""
    return toyhost_design().model_dump(mode="json", exclude={"declared_at", "declared_by"})


def _definition(fixture: OpsFixture, **fields: Any) -> dict[str, Any]:
    return {
        "name": "said at birth",
        "subject_id": TOYHOST_SUBJECT.subject_id,
        "behavior": fixture.campaign.behavior,
        "run_ids": list(fixture.campaign.run_ids),
        **fields,
    }


def _stored(fixture: OpsFixture, campaign_id: str) -> EvalCampaign:
    campaign = off_loop(fixture.host.eval_host.storage.load_campaign, campaign_id, TOYHOST_SCOPE)
    assert campaign is not None
    return campaign


def _python_create(fixture: OpsFixture, *, exploratory: bool = False, **fields: Any) -> EvalCampaign:
    eval_host = fixture.host.eval_host
    return off_loop(
        create_campaign,
        eval_host.storage,
        _definition(fixture, **fields),
        scope_id=TOYHOST_SCOPE,
        created_by=CALLER.identity,
        profile=eval_host.profile,
        exploratory=exploratory,
    )


async def _call(tool: MountedTool, fixture: OpsFixture, action: str, **arguments: Any) -> Any:
    return await tool.call({"action": action, **arguments}, host=fixture.host, caller=CALLER)


async def _exploratory_campaign(tool: MountedTool, fixture: OpsFixture) -> str:
    created = await _call(tool, fixture, "campaign_create", **_definition(fixture), exploratory=True)
    assert not created.is_error, created.text
    return str(created.structured["id"])


async def _declared_campaign(tool: MountedTool, fixture: OpsFixture) -> str:
    created = await _call(tool, fixture, "campaign_create", **_definition(fixture), declared_design=_design())
    assert not created.is_error, created.text
    return str(created.structured["id"])


# --- create says which ----------------------------------------------------------------------------------------------


def test_a_python_create_that_says_neither_is_refused_naming_both_and_writes_nothing() -> None:
    fixture = off_loop(ops_fixture)
    before = off_loop(fixture.host.eval_host.storage.list_campaigns, TOYHOST_SCOPE)

    with pytest.raises(ValidationFailedError) as refused:
        _python_create(fixture)

    said = str(refused.value)
    assert "says neither" in said and "declared_design" in said and "exploratory: true" in said, said
    assert off_loop(fixture.host.eval_host.storage.list_campaigns, TOYHOST_SCOPE) == before


def test_a_python_create_that_says_both_is_refused_as_a_contradiction() -> None:
    fixture = off_loop(ops_fixture)

    with pytest.raises(ValidationFailedError, match="both declares a design and says it is exploratory"):
        _python_create(fixture, exploratory=True, declared_design=_design())


def test_a_python_create_with_a_design_is_declared_and_one_saying_exploratory_is_not() -> None:
    fixture = off_loop(ops_fixture)

    declared = _python_create(fixture, declared_design=_design())
    exploratory = _python_create(fixture, exploratory=True)

    assert declared.declared_design is not None
    assert exploratory.declared_design is None
    assert "exploratory" not in EvalCampaign.model_fields, "an assertion checked, never stored"
    assert "exploratory" not in _stored(fixture, exploratory.id).model_dump()


@pytest.mark.parametrize(
    ("arguments", "says"),
    [
        pytest.param({}, "says neither", id="neither"),
        pytest.param({"exploratory": False}, "says neither", id="exploratory-false-and-no-design"),
        pytest.param({"exploratory": True, "declared_design": "{design}"}, "both declares a design", id="both"),
    ],
)
async def test_the_create_action_refuses_a_campaign_that_does_not_say_which_once(
    evals: MountedTool, arguments: dict[str, Any], says: str
) -> None:
    fixture = off_loop(ops_fixture)
    resolved = {key: (_design() if value == "{design}" else value) for key, value in arguments.items()}
    before = off_loop(fixture.host.eval_host.storage.list_campaigns, TOYHOST_SCOPE)

    outcome = await _call(evals, fixture, "campaign_create", **_definition(fixture), **resolved)

    assert outcome.is_error and says in outcome.text, outcome.text
    assert outcome.refusal is not None and outcome.refusal.kind == "validation_failed"
    assert off_loop(fixture.host.eval_host.storage.list_campaigns, TOYHOST_SCOPE) == before


async def test_the_create_action_takes_either_and_its_line_says_which(evals: MountedTool) -> None:
    fixture = off_loop(ops_fixture)

    exploratory = await _call(evals, fixture, "campaign_create", **_definition(fixture), exploratory=True)
    declared = await _call(evals, fixture, "campaign_create", **_definition(fixture), declared_design=_design())

    assert not exploratory.is_error and not declared.is_error, (exploratory.text, declared.text)
    assert CampaignLine.model_validate(exploratory.structured).exploratory and "exploratory" in exploratory.text
    assert not CampaignLine.model_validate(declared.structured).exploratory and "declared" in declared.text


# --- campaign_update moves a campaign between the two ---------------------------------------------------------------


async def test_declaring_a_design_on_an_exploratory_campaign_makes_it_declared(evals: MountedTool) -> None:
    fixture = off_loop(ops_fixture)
    campaign_id = await _exploratory_campaign(evals, fixture)

    outcome = await _call(evals, fixture, "campaign_update", campaign_id=campaign_id, declared_design=_design())

    assert not outcome.is_error, outcome.text
    assert not CampaignLine.model_validate(outcome.structured).exploratory
    stored = _stored(fixture, campaign_id)
    assert stored.declared_design is not None and stored.declared_design.declared_by == CALLER.identity
    listed = await _call(evals, fixture, "campaigns_list")
    assert next(line for line in listed.structured["campaigns"] if line["id"] == campaign_id)["exploratory"] is False


async def test_withdrawing_a_declaration_without_saying_exploratory_is_refused(evals: MountedTool) -> None:
    fixture = off_loop(ops_fixture)
    campaign_id = await _declared_campaign(evals, fixture)
    before = _stored(fixture, campaign_id)

    outcome = await _call(evals, fixture, "campaign_update", campaign_id=campaign_id, declared_design=None)

    assert outcome.is_error and "exploratory: true" in outcome.text, outcome.text
    assert _stored(fixture, campaign_id) == before


async def test_withdrawing_a_declaration_saying_exploratory_leaves_the_campaign_exploratory(
    evals: MountedTool,
) -> None:
    fixture = off_loop(ops_fixture)
    campaign_id = await _declared_campaign(evals, fixture)

    outcome = await _call(
        evals, fixture, "campaign_update", campaign_id=campaign_id, declared_design=None, exploratory=True
    )

    assert not outcome.is_error, outcome.text
    assert CampaignLine.model_validate(outcome.structured).exploratory
    assert _stored(fixture, campaign_id).declared_design is None


@pytest.mark.parametrize(
    ("start", "arguments", "says"),
    [
        pytest.param(
            "exploratory",
            {"declared_design": "{design}", "exploratory": True},
            "both declares a design",
            id="a-design-and-exploratory",
        ),
        pytest.param(
            "declared",
            {"name": "renamed", "exploratory": True},
            "this update keeps it",
            id="exploratory-keeping-a-design",
        ),
        pytest.param("declared", {}, "names nothing to change", id="nothing"),
    ],
)
async def test_an_update_that_contradicts_itself_or_its_campaign_is_refused_and_writes_nothing(
    evals: MountedTool, start: str, arguments: dict[str, Any], says: str
) -> None:
    fixture = off_loop(ops_fixture)
    make = _exploratory_campaign if start == "exploratory" else _declared_campaign
    campaign_id = await make(evals, fixture)
    before = _stored(fixture, campaign_id)
    resolved = {key: (_design() if value == "{design}" else value) for key, value in arguments.items()}

    outcome = await _call(evals, fixture, "campaign_update", campaign_id=campaign_id, **resolved)

    assert outcome.is_error and says in outcome.text, outcome.text
    assert _stored(fixture, campaign_id) == before


async def test_an_update_changes_only_what_it_is_sent(evals: MountedTool) -> None:
    fixture = off_loop(ops_fixture)
    campaign_id = await _declared_campaign(evals, fixture)
    before = _stored(fixture, campaign_id)

    outcome = await _call(evals, fixture, "campaign_update", campaign_id=campaign_id, name="renamed")

    assert not outcome.is_error, outcome.text
    after = _stored(fixture, campaign_id)
    assert after.name == "renamed" and outcome.structured["name"] == "renamed"
    assert after.declared_design == before.declared_design, "an omitted declared_design is left as it is"
    assert after.description == before.description and after.run_ids == before.run_ids


async def test_an_exploratory_campaign_may_be_renamed_saying_it_is_exploratory(evals: MountedTool) -> None:
    fixture = off_loop(ops_fixture)
    campaign_id = await _exploratory_campaign(evals, fixture)

    outcome = await _call(
        evals, fixture, "campaign_update", campaign_id=campaign_id, description="still looking", exploratory=True
    )

    assert not outcome.is_error, outcome.text
    assert _stored(fixture, campaign_id).description == "still looking"


def test_the_python_update_holds_to_the_same_rule() -> None:
    fixture = off_loop(ops_fixture)
    eval_host = fixture.host.eval_host
    campaign = _python_create(fixture, declared_design=_design())

    def update(updates: dict[str, Any], **keywords: Any) -> EvalCampaign:
        return off_loop(
            update_campaign,
            eval_host.storage,
            campaign.id,
            TOYHOST_SCOPE,
            updates,
            updated_by="op",
            profile=eval_host.profile,
            **keywords,
        )

    with pytest.raises(ValidationFailedError, match="exploratory: true"):
        update({"declared_design": None})
    assert update({"declared_design": None}, exploratory=True).declared_design is None
    assert update({"declared_design": _design()}).declared_design is not None


# --- the action, and the refusal that names it ----------------------------------------------------------------------


async def test_campaign_update_is_a_write_with_a_help_page_and_an_example(evals: MountedTool) -> None:
    action = evals.action("campaign_update")
    assert action is not None and action.permission == "write"
    assert set(action.params.model_fields) == {
        "campaign_id",
        "name",
        "description",
        "declared_design",
        "exploratory",
        "control_from_run_id",
    }

    page = await evals.call(
        {"action": "help", "topic": "campaign_update"}, host=off_loop(ops_fixture).host, caller=CALLER
    )

    assert not page.is_error and "declared_design" in page.text and "exploratory" in page.text, page.text


async def test_the_control_refusal_on_an_exploratory_campaign_names_campaign_update(evals: MountedTool) -> None:
    fixture = off_loop(ops_fixture)
    campaign_id = await _exploratory_campaign(evals, fixture)
    eval_host = fixture.host.eval_host

    with pytest.raises(ValidationFailedError) as refused:
        off_loop(
            set_campaign_control,
            eval_host.storage,
            campaign_id,
            TOYHOST_SCOPE,
            fixture.campaign.run_ids[0],
            set_by="op",
            profile=eval_host.profile,
        )

    assert "campaign_update" in refused.value.remedy and "analysis.update_campaign" not in str(refused.value)
    assert evals.action("campaign_update") is not None, "the remedy names an action an operator can call"
