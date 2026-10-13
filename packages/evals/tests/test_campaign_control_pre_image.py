"""A campaign's control keeps its pre-image, so a change of identity predicate re-keys it instead of orphaning it.

A run survives an ``IDENTITY_VERSION`` bump because it stores ``variant_levers`` beside ``variant_key``. The control
is a variant key on a document kept forever, and it stored the key alone: after a bump every stored campaign's
control matched no new run of its own arm, and nothing could recover it. Pinned here:

- **Addressing a control from a run records its pre-image**: ``campaign_create`` and ``campaign_update`` with
  ``control_from_run_id``, and ``set_campaign_control``, store the run's lever map and the predicate beside the key;
  clearing the control clears all three.
- **A control addressed under another predicate names the same arm**: stored under ``IDENTITY_VERSION - 1`` with a
  key that predicate computed, it is re-keyed from its lever map, and the bundle's control arm, its declaration and
  its decision surface all name the arm this build's runs carry. The campaign keeps the key as declared.
- **A typed key under another predicate is disclosed, never silent**: it reads as ``stale`` with the remedy
  (``control_from_run_id`` on ``campaign_update``), in the bundle and in the code-only report, rather than as a
  campaign that declared no control.
- **The declaration holds the three fields together**: a typed key is dated by the build accepting it, a lever map
  stands only beside a key, and a map that does not digest to the key under this build's predicate is refused.

A predicate change is exercised as the identity tests exercise one: a document stamped ``IDENTITY_VERSION - 1``.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.actions import MountedTool, eval_catalogue, standard_tools
from threetears.evals.analysis import assemble_context_bundle, variant_key_of_run
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.campaigns import set_campaign_control
from threetears.evals.analysis.report.build import build_code_only_report
from threetears.evals.kernel import EvalCampaign, IDENTITY_VERSION
from threetears.evals.kernel.declaration import CampaignDesign, resolve_declared_control
from threetears.evals.kernel.identity import compute_variant_key, resolve_variant_identity
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.models import utc_now_iso
from threetears.evals.schema.values import SweepableValue
from packages.evals.tests.factories import minimal_declaration
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_design
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, TOYHOST_SUBJECT, OpsFixture, ops_fixture
from packages.evals.tests.off_loop import off_loop

#: The predicate before this build's, which every "declared under another predicate" document here is stamped with.
_EARLIER = IDENTITY_VERSION - 1


@pytest.fixture
def evals() -> MountedTool:
    return eval_catalogue().mount_all(standard_tools())[0]


def _design(**changes: Any) -> dict[str, Any]:
    design = toyhost_design().model_dump(mode="json", exclude={"declared_at", "declared_by"})
    design.update(changes)
    return design


async def _call(tool: MountedTool, fixture: OpsFixture, action: str, **arguments: Any) -> Any:
    outcome = await tool.call({"action": action, **arguments}, host=fixture.host, caller=CALLER)
    assert not outcome.is_error, outcome.text
    return outcome


def _stored(fixture: OpsFixture, campaign_id: str) -> EvalCampaign:
    campaign = fixture.host.eval_host.storage.load_campaign(campaign_id, TOYHOST_SCOPE)
    assert campaign is not None
    return campaign


def _levers_of(fixture: OpsFixture, run_id: str) -> dict[str, SweepableValue]:
    """``run_id``'s pre-image as a reader reads it: the map it recorded, else the one this build derives."""
    run = fixture.host.eval_host.storage.load_eval_run(run_id, TOYHOST_SCOPE)
    assert run is not None
    return resolve_variant_identity(run=run, profile=fixture.host.eval_host.profile).levers


def _key_of(fixture: OpsFixture, run_id: str) -> str:
    """The key ``run_id``'s observations carry under this build."""
    key = variant_key_of_run(fixture.host.eval_host.storage.query_eval_results_by_run(run_id, TOYHOST_SCOPE))
    assert key is not None
    return key


def _assert_addressed_from(design: CampaignDesign | None, fixture: OpsFixture, run_id: str) -> None:
    """``design``'s control is ``run_id``'s key, stored beside that run's lever map and the predicate."""
    assert design is not None
    assert design.control == _key_of(fixture, run_id)
    assert design.control_levers == _levers_of(fixture, run_id)
    assert design.control_identity_version == IDENTITY_VERSION
    assert compute_variant_key(design.control_levers) == design.control


def _older_predicates_key(levers: dict[str, SweepableValue]) -> str:
    """A key an earlier predicate computed over ``levers`` — any digest this build's arithmetic does not produce."""
    key = canonical_digest({"predicate": _EARLIER, "levers": {name: v.content_hash for name, v in levers.items()}})
    assert key != compute_variant_key(levers)
    return key


def _declared_under_an_earlier_predicate(fixture: OpsFixture, run_id: str, *, with_levers: bool) -> EvalCampaign:
    """The corpus campaign, its control declared from ``run_id``'s arm under ``IDENTITY_VERSION - 1``, stored."""
    levers = _levers_of(fixture, run_id)
    design = toyhost_design().model_copy(
        update={
            "control": _older_predicates_key(levers),
            "control_levers": levers if with_levers else None,
            "control_identity_version": _EARLIER,
        }
    )
    campaign = fixture.campaign.model_copy(
        update={"declared_design": CampaignDesign.model_validate(design.model_dump(mode="json"))}
    )
    fixture.host.eval_host.storage.save_campaign(campaign)
    return campaign


# --- Addressing a control from a run records its pre-image ---------------------------------------------------------


async def test_campaign_create_with_control_from_run_id_stores_the_runs_lever_map(evals: MountedTool) -> None:
    fixture = off_loop(ops_fixture)
    run_id = fixture.campaign.run_ids[0]

    created = await _call(
        evals,
        fixture,
        "campaign_create",
        name="addressed at birth",
        subject_id=TOYHOST_SUBJECT.subject_id,
        behavior=fixture.campaign.behavior,
        run_ids=list(fixture.campaign.run_ids),
        declared_design=_design(),
        control_from_run_id=run_id,
    )

    stored = off_loop(_stored, fixture, created.structured["id"])
    off_loop(_assert_addressed_from, stored.declared_design, fixture, run_id)


async def test_campaign_update_with_control_from_run_id_stores_it_and_a_null_clears_all_three(
    evals: MountedTool,
) -> None:
    fixture = off_loop(ops_fixture)
    first, second = fixture.campaign.run_ids[0], fixture.campaign.run_ids[1]
    created = await _call(
        evals,
        fixture,
        "campaign_create",
        name="addressed later",
        subject_id=TOYHOST_SUBJECT.subject_id,
        behavior=fixture.campaign.behavior,
        run_ids=[first, second],
        exploratory=True,
    )
    campaign_id = created.structured["id"]

    await _call(
        evals, fixture, "campaign_update", campaign_id=campaign_id, declared_design=_design(), control_from_run_id=first
    )
    declared = off_loop(_stored, fixture, campaign_id).declared_design
    await _call(evals, fixture, "campaign_update", campaign_id=campaign_id, control_from_run_id=second)
    renamed = off_loop(_stored, fixture, campaign_id).declared_design
    await _call(evals, fixture, "campaign_update", campaign_id=campaign_id, control_from_run_id=None)
    cleared = off_loop(_stored, fixture, campaign_id).declared_design

    off_loop(_assert_addressed_from, declared, fixture, first)
    off_loop(_assert_addressed_from, renamed, fixture, second)
    assert cleared is not None
    assert (cleared.control, cleared.control_levers, cleared.control_identity_version) == (None, None, None)


def test_set_campaign_control_stores_the_runs_lever_map_and_a_clear_clears_all_three() -> None:
    fixture = ops_fixture()
    eval_host = fixture.host.eval_host
    run_id = fixture.campaign.run_ids[0]
    eval_host.storage.save_campaign(fixture.campaign.model_copy(update={"declared_design": toyhost_design()}))

    designated = set_campaign_control(
        eval_host.storage, fixture.campaign.id, TOYHOST_SCOPE, run_id, set_by="op", profile=eval_host.profile
    )
    cleared = set_campaign_control(
        eval_host.storage, fixture.campaign.id, TOYHOST_SCOPE, None, set_by="op", profile=eval_host.profile
    )

    _assert_addressed_from(designated.declared_design, fixture, run_id)
    assert cleared.declared_design is not None
    assert (cleared.declared_design.control, cleared.declared_design.control_levers) == (None, None)
    assert cleared.declared_design.control_identity_version is None


def test_a_run_s_recorded_lever_map_is_what_the_control_keeps() -> None:
    fixture = ops_fixture()
    eval_host = fixture.host.eval_host
    run_id = fixture.campaign.run_ids[0]
    run = eval_host.storage.load_eval_run(run_id, TOYHOST_SCOPE)
    assert run is not None and run.variant_levers is None, "the corpus run was assembled without the launch"
    # Recorded as a launch records it — the map and its predicate together. Renamed on display only, which no key
    # hashes, so the map digests to the key the run's observations carry and the recorded one is told apart.
    recorded = {
        name: level.model_copy(update={"display": f"{level.display} (as launched)"})
        for name, level in _levers_of(fixture, run_id).items()
    }
    eval_host.storage.save_eval_run(
        run.model_copy(update={"variant_levers": recorded, "identity_version": IDENTITY_VERSION})
    )
    eval_host.storage.save_campaign(fixture.campaign.model_copy(update={"declared_design": toyhost_design()}))

    designated = set_campaign_control(
        eval_host.storage, fixture.campaign.id, TOYHOST_SCOPE, run_id, set_by="op", profile=eval_host.profile
    )

    assert designated.declared_design is not None
    assert designated.declared_design.control_levers == recorded
    assert designated.declared_design.control == _key_of(fixture, run_id)


# --- A control addressed under another predicate names the same arm ------------------------------------------------


def test_a_control_addressed_under_an_earlier_predicate_resolves_to_the_same_arm() -> None:
    fixture = ops_fixture()
    eval_host = fixture.host.eval_host
    run_id = fixture.campaign.run_ids[0]
    campaign = _declared_under_an_earlier_predicate(fixture, run_id, with_levers=True)
    assert campaign.declared_design is not None
    declared_key = campaign.declared_design.control

    bundle = assemble_context_bundle(campaign, storage=eval_host.storage, profile=eval_host.profile)

    arm = _key_of(fixture, run_id)
    assert resolve_declared_control(campaign.declared_design).source == "rekeyed"
    assert bundle.design.control_arm is not None and bundle.design.control_arm.variant_key == arm
    assert run_id in bundle.design.control_arm.run_ids
    assert bundle.design.control_excluded is None
    assert bundle.declared_design is not None
    assert bundle.declared_design.control == arm, "the bundle carries the control as its cells are keyed"
    assert bundle.declared_design.control_identity_version == IDENTITY_VERSION
    assert bundle_decision_surface(bundle).control_variant_key == arm
    stored = _stored(fixture, campaign.id).declared_design
    assert stored is not None and stored.control == declared_key, "a read writes nothing back onto the campaign"
    assert stored.control_identity_version == _EARLIER


def test_a_control_declared_under_this_predicate_is_read_as_stored() -> None:
    levers = {"model": SweepableValue.of("glm", display="GLM")}
    design = minimal_declaration(control=compute_variant_key(levers), control_levers=levers)

    resolved = resolve_declared_control(design)

    assert (resolved.variant_key, resolved.identity_version, resolved.source) == (
        design.control,
        IDENTITY_VERSION,
        "stored",
    )
    assert design.as_keyed_by_this_build() is design
    assert (
        resolve_declared_control(None).source == resolve_declared_control(minimal_declaration()).source == "undeclared"
    )


# --- A typed key under another predicate is disclosed, never silent ------------------------------------------------


def test_a_typed_control_under_an_earlier_predicate_is_disclosed_with_its_remedy() -> None:
    fixture = ops_fixture()
    eval_host = fixture.host.eval_host
    campaign = _declared_under_an_earlier_predicate(fixture, fixture.campaign.run_ids[0], with_levers=False)

    bundle = assemble_context_bundle(campaign, storage=eval_host.storage, profile=eval_host.profile)
    report = build_code_only_report(bundle, measures=eval_host.profile.measures, assembled_at=utc_now_iso())

    resolved = resolve_declared_control(campaign.declared_design)
    assert (resolved.variant_key, resolved.identity_version, resolved.source) == (None, _EARLIER, "stale")
    assert bundle.design.control_arm is None
    assert bundle.design.control_excluded == "stale", "a stale control is not a campaign that declared none"
    disclosure = bundle.design.stale_control_disclosure
    assert disclosure is not None and disclosure == resolved.disclosure
    assert f"v{_EARLIER}" in disclosure and f"v{IDENTITY_VERSION}" in disclosure
    assert "control_from_run_id on campaign_update" in disclosure
    assert disclosure in [getattr(block, "text", None) for block in report.blocks]


def test_a_control_nothing_stale_carries_no_stale_disclosure() -> None:
    fixture = ops_fixture()
    eval_host = fixture.host.eval_host
    campaign = _declared_under_an_earlier_predicate(fixture, fixture.campaign.run_ids[0], with_levers=True)

    bundle = assemble_context_bundle(campaign, storage=eval_host.storage, profile=eval_host.profile)

    assert bundle.design.stale_control_disclosure is None
    assert resolve_declared_control(campaign.declared_design).disclosure is None


# --- The declaration holds the three fields together ---------------------------------------------------------------


def test_a_typed_control_is_dated_by_the_build_accepting_it_and_a_stated_version_is_kept() -> None:
    key = "a" * 64

    typed = CampaignDesign.model_validate({**_design(), "control": key})
    echoed = CampaignDesign.model_validate({**_design(), "control": key, "control_identity_version": _EARLIER})

    assert typed.control_identity_version == IDENTITY_VERSION
    assert typed.control_levers is None
    assert echoed.control_identity_version == _EARLIER, "a declaration read back and echoed is not re-dated"


@pytest.mark.parametrize(
    ("fields", "says"),
    [
        pytest.param(
            {"control_levers": {"model": SweepableValue.of("glm", display="GLM").model_dump(mode="json")}},
            "declares no control",
            id="levers-without-a-control",
        ),
        pytest.param({"control_identity_version": IDENTITY_VERSION}, "declares no control", id="version-without-one"),
        pytest.param(
            {
                "control": "b" * 64,
                "control_levers": {"model": SweepableValue.of("glm", display="GLM").model_dump(mode="json")},
            },
            "does not digest to `control`",
            id="a-map-that-is-another-arms",
        ),
    ],
)
def test_a_pre_image_that_is_not_the_controls_is_refused(fields: dict[str, Any], says: str) -> None:
    with pytest.raises(ValidationError, match=says):
        CampaignDesign.model_validate({**_design(), **fields})


def test_a_map_recorded_under_an_earlier_predicate_is_not_held_to_this_ones_arithmetic() -> None:
    levers = {"model": SweepableValue.of("glm", display="GLM")}

    design = minimal_declaration(
        control=_older_predicates_key(levers), control_levers=levers, control_identity_version=_EARLIER
    )

    assert resolve_declared_control(design).variant_key == compute_variant_key(levers)
    assert design.as_keyed_by_this_build().control == compute_variant_key(levers)
