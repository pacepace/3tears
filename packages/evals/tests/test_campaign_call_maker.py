"""A recorded campaign call says whether a person or a model made it (#1178).

Pinned here:

- a model's call stores the model, and whom it acted for, and reads back with both, in the call and in the listing;
- a person's call is recorded only when the host attests the caller for that person: with no check, or a check that
  declines, it is refused (403) and nothing is written; an attested one is stored under the host's name for them;
- a model's call names its model, and only a model's call names a model or whom it acted for;
- a call that says neither (every call stored before the field) reads back with no maker, never as a person's.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from threetears.evals.actions import MountedTool, eval_catalogue, standard_tools
from threetears.evals.ops import campaigns_list
from threetears.evals.ops.analysis import CampaignChanges, campaign_update
from threetears.evals.ops.host import OpsHost
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture

PERSON = "ana@example.com"
MODEL = "anthropic/claude-opus"
MISSED_BAR = "extract_invoice_fields/field_accuracy"


def _attests(*, scope_id: str, caller: str, person: str) -> str | None:
    """The host's check: it vouches for :data:`CALLER` relaying anyone, under its own spelling of their name."""
    return person.lower() if caller == CALLER.identity else None


def _choice(attests: Any = _attests) -> tuple[OpsHost, OpsFixture, str]:
    """The toy corpus campaign as a choice (its first run's arm the control), and the control's variant key."""
    fixture = off_loop(ops_fixture)
    eval_host = fixture.host.eval_host
    off_loop(
        campaign_update,
        eval_host,
        fixture.campaign.id,
        CampaignChanges(control_from_run_id=fixture.campaign.run_ids[0]),
        TOYHOST_SCOPE,
        updated_by=CALLER.identity,
    )
    # The control, by the variant key its run's results carry (a call names an arm by label or variant key): the
    # candidate misses a bar, which a call on it would have to override, and that is not what this file is about.
    (held, *_) = off_loop(eval_host.storage.query_eval_results_by_run, fixture.campaign.run_ids[0], TOYHOST_SCOPE)
    arm = held.variant_key
    launch = fixture.host.launch
    host = replace(fixture.host, launch=replace(launch, eval_host=replace(eval_host, attests_person=attests)))
    return host, fixture, arm


def _evals() -> MountedTool:
    return {tool.name: tool for tool in eval_catalogue().mount_all(standard_tools())}["evals"]


async def _record(host: OpsHost, fixture: OpsFixture, arm: str, **params: Any) -> Any:
    return await _evals().call(
        {
            "action": "campaign_record",
            "campaign_id": fixture.campaign.id,
            "arm": arm,
            # The toy control misses its field-accuracy bar, which every call on it overrides by name.
            "bars_overridden": [MISSED_BAR],
            **params,
        },
        host=host,
        caller=CALLER,
    )


def _stored_call(host: OpsHost, fixture: OpsFixture) -> Any:
    campaign = off_loop(host.eval_host.storage.load_campaign, fixture.campaign.id, TOYHOST_SCOPE)
    assert campaign is not None
    return campaign.call


async def test_a_model_s_call_stores_the_model_and_whom_it_acted_for_and_reads_back_with_both() -> None:
    host, fixture, arm = _choice()

    outcome = await _record(
        host, fixture, arm, made_by="model", call_model=MODEL, on_behalf_of="brookstalley", call_reason="it reads best"
    )

    assert not outcome.is_error, outcome.text
    call = outcome.structured["call"]
    assert (call["made_by"], call["model"], call["on_behalf_of"], call["called_by"]) == (
        "model",
        MODEL,
        "brookstalley",
        CALLER.identity,
    )
    stored = _stored_call(host, fixture)
    assert (stored.made_by, stored.model, stored.on_behalf_of) == ("model", MODEL, "brookstalley")
    (line,) = [
        line
        for line in off_loop(campaigns_list, host.eval_host, TOYHOST_SCOPE).campaigns
        if line.id == fixture.campaign.id
    ]
    assert line.call is not None
    assert (line.call.made_by, line.call.model, line.call.on_behalf_of) == ("model", MODEL, "brookstalley")


async def test_an_attested_person_s_call_is_stored_under_the_host_s_name_for_them() -> None:
    host, fixture, arm = _choice()

    outcome = await _record(host, fixture, arm, made_by="person", person="Ana@Example.com")

    assert not outcome.is_error, outcome.text
    stored = _stored_call(host, fixture)
    assert (stored.made_by, stored.called_by, stored.model) == ("person", PERSON, None)


@pytest.mark.parametrize("attests", [None, lambda **_: None], ids=["no check", "declines"])
async def test_a_person_s_call_the_host_does_not_attest_is_refused_and_nothing_is_written(attests: Any) -> None:
    host, fixture, arm = _choice(attests)

    outcome = await _record(host, fixture, arm, made_by="person", person=PERSON)

    assert outcome.is_error and outcome.refusal is not None
    assert (outcome.refusal.kind, outcome.refusal.status) == ("refused", 403)
    assert _stored_call(host, fixture) is None


@pytest.mark.parametrize(
    "params",
    [
        {"made_by": "model"},
        {"made_by": "person", "person": PERSON, "call_model": MODEL},
        {"made_by": "person"},
        {"call_model": MODEL},
        {"person": PERSON},
        {"made_by": "model", "call_model": MODEL, "person": PERSON},
    ],
    ids=[
        "model unnamed",
        "person names a model",
        "person unnamed",
        "model without maker",
        "person without maker",
        "model names a person",
    ],
)
async def test_a_call_whose_maker_and_names_disagree_is_refused(params: dict[str, Any]) -> None:
    host, fixture, arm = _choice()

    outcome = await _record(host, fixture, arm, **params)

    assert outcome.is_error, params
    assert _stored_call(host, fixture) is None


async def test_a_call_that_says_neither_reads_back_with_no_maker_never_as_a_person_s() -> None:
    host, fixture, arm = _choice()

    outcome = await _record(host, fixture, arm, call_reason="recorded the way every call was before")

    assert not outcome.is_error, outcome.text
    assert outcome.structured["call"].get("made_by") is None
    stored = _stored_call(host, fixture)
    assert stored.made_by is None and "made_by" not in stored.model_dump(mode="json"), "stored as before the field"
    (line,) = [
        line
        for line in off_loop(campaigns_list, host.eval_host, TOYHOST_SCOPE).campaigns
        if line.id == fixture.campaign.id
    ]
    assert line.call is not None and line.call.made_by is None
