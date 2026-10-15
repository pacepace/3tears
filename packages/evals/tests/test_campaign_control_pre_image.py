"""A campaign's control keeps its pre-image, and names its arm by every key that names it under this build.

A run survives an ``IDENTITY_VERSION`` bump because it stores ``variant_levers`` beside ``variant_key``. The control
is a variant key on a document kept forever, and it stored the key alone. Stamped keys are never re-derived, so once
the predicate moves a control's arm carries two keys — the stored one on the cells stamped before, the re-keyed one
on the cells stamped after — and the control must name both. Pinned here:

- **Addressing a control from a run records its pre-image**: ``campaign_create`` and ``campaign_update`` with
  ``control_from_run_id``, and ``set_campaign_control``, store the run's lever map and the predicate beside the key;
  clearing the control clears all three.
- **A control matches its arm on either side of a predicate change**: re-keyed from its map, it matches the cells
  this build stamps; by its stored key, it matches the cells its own predicate stamped — a typed key included, and a
  control addressed from a run whose map does not reproduce its stamped key — so it is never stale while they are
  here. The campaign keeps the key as declared.
- **A campaign carrying both keys keeps ONE control arm**: the cells under this build's predicate. The cells under
  the earlier one are named as the control under that predicate, disclosed in the bundle, the code-only report and
  the time positions they ran at, and kept out of every contrast, guardrail, frontier boundary check and adoption
  check, and out of the control's numbers. A marginal row, read over the whole campaign, counts them at the
  control's level, as it counts every cell there.
- **Every reading agrees on which arm is the control**: the realized design decides it once, and the decision
  surface, the surface's row order, the time axis, the adoption check and the report's scoreboard read that.
- **A key nothing can carry is disclosed, never silent**: a typed key from another predicate that no cell carries
  reads as ``stale`` with the remedy, in the bundle and in the code-only report.
- **The pre-image is the server's**: a typed key is dated by the build accepting it; an update echoing a typed key
  keeps the stored date; a lever map or version sent with a value the server would not store is refused; and the
  declaration refuses a map without a key, or one that does not digest to it under this build's predicate.

A predicate change is exercised as the identity tests exercise one: documents stamped ``IDENTITY_VERSION - 1``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.actions import MountedTool, eval_catalogue, standard_tools
from threetears.evals.analysis import TableBlock, assemble_context_bundle, build_report, variant_key_of_run
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.campaigns import set_campaign_control, update_campaign
from threetears.evals.analysis.cells import variant_of_cell_ref
from threetears.evals.analysis.generator import generate_analysis
from threetears.evals.analysis.lenses.frontier import compute_frontier
from threetears.evals.analysis.references import cell_aliases
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.analysis.report.build import build_code_only_report
from threetears.evals.kernel import EvalCampaign, IDENTITY_VERSION
from threetears.evals.kernel.declaration import CampaignDesign, SweptAxis, resolve_declared_control
from threetears.evals.kernel.identity import compute_variant_key, resolve_variant_identity
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.models import utc_now_iso
from threetears.evals.schema.values import SweepableValue
from packages.evals.tests.factories import minimal_declaration
from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_AXIS, TOYHOST_NARROW, toyhost_design
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, TOYHOST_SUBJECT, OpsFixture, ops_fixture
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.toyhost_memo import MODEL, PROMPT, PROMPT_ID, FixturedClient, alias_at, cell_at, memo_payload

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


def _stamped_under_the_earlier_predicate(fixture: OpsFixture, run_id: str) -> str:
    """Re-store ``run_id``'s results as the earlier predicate stamped them, and return the key they now carry."""
    storage = fixture.host.eval_host.storage
    key = _older_predicates_key(_levers_of(fixture, run_id))
    for result in storage.query_eval_results_by_run(run_id, TOYHOST_SCOPE):
        storage.save_eval_result(result.model_copy(update={"variant_key": key, "identity_version": _EARLIER}))
    return key


def _assemble(fixture: OpsFixture, campaign: EvalCampaign) -> Any:
    eval_host = fixture.host.eval_host
    return assemble_context_bundle(campaign, storage=eval_host.storage, profile=eval_host.profile)


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


async def test_campaign_update_with_control_from_run_id_stores_it_once_and_a_null_clears_all_three(
    evals: MountedTool,
) -> None:
    """Declared before its runs, addressed once they joined, never re-pointed, and cleared once they are gone."""
    fixture = off_loop(ops_fixture)
    first, second = fixture.campaign.run_ids[0], fixture.campaign.run_ids[1]
    created = await _call(
        evals,
        fixture,
        "campaign_create",
        name="addressed later",
        subject_id=TOYHOST_SUBJECT.subject_id,
        behavior=fixture.campaign.behavior,
        run_ids=[],
        exploratory=True,
    )
    campaign_id = created.structured["id"]

    await _call(evals, fixture, "campaign_update", campaign_id=campaign_id, declared_design=_design())
    await _call(evals, fixture, "campaign_runs_add", campaign_id=campaign_id, run_ids=[first, second])
    await _call(evals, fixture, "campaign_update", campaign_id=campaign_id, control_from_run_id=first)
    declared = off_loop(_stored, fixture, campaign_id).declared_design
    refused = await _refused(evals, fixture, "campaign_update", campaign_id=campaign_id, control_from_run_id=second)
    kept = off_loop(_stored, fixture, campaign_id).declared_design
    await _call(evals, fixture, "campaign_runs_remove", campaign_id=campaign_id, run_ids=[first, second])
    await _call(evals, fixture, "campaign_update", campaign_id=campaign_id, control_from_run_id=None)
    cleared = off_loop(_stored, fixture, campaign_id).declared_design

    off_loop(_assert_addressed_from, declared, fixture, first)
    assert "re-pointing it" in refused and kept == declared, "a control is addressed once"
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
    # A control is cleared only once no run holds the campaign's design frozen.
    _before_its_runs(fixture, designated)
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


# --- A control matches its arm on either side of a predicate change ----------------------------------------------


def test_a_control_addressed_under_an_earlier_predicate_names_the_cells_this_build_stamps() -> None:
    fixture = ops_fixture()
    run_id = fixture.campaign.run_ids[0]
    campaign = _declared_under_an_earlier_predicate(fixture, run_id, with_levers=True)
    assert campaign.declared_design is not None
    declared_key = campaign.declared_design.control

    bundle = _assemble(fixture, campaign)

    arm = _key_of(fixture, run_id)
    resolved = resolve_declared_control(campaign.declared_design)
    assert (resolved.source, resolved.variant_keys) == ("rekeyed", [arm, declared_key])
    assert bundle.design.control_arm is not None and bundle.design.control_arm.variant_key == arm
    assert run_id in bundle.design.control_arm.run_ids
    assert bundle.design.control_excluded is None
    assert bundle.declared_design == campaign.declared_design, "the bundle carries the declaration as declared"
    assert bundle_decision_surface(bundle).control_variant_key == arm
    stored = _stored(fixture, campaign.id).declared_design
    assert stored is not None and stored.control == declared_key, "a read writes nothing back onto the campaign"
    assert stored.control_identity_version == _EARLIER


@pytest.mark.parametrize("with_levers", [False, True], ids=["typed", "addressed"])
def test_a_control_from_an_earlier_predicate_still_names_the_cells_that_predicate_stamped(with_levers: bool) -> None:
    fixture = ops_fixture()
    run_id = fixture.campaign.run_ids[0]
    old_key = _stamped_under_the_earlier_predicate(fixture, run_id)
    campaign = _declared_under_an_earlier_predicate(fixture, run_id, with_levers=with_levers)
    assert campaign.declared_design is not None and campaign.declared_design.control == old_key

    bundle = _assemble(fixture, campaign)

    assert bundle.design.control_excluded is None, "a key its own cells carry is the control, never stale"
    assert bundle.design.stale_control_disclosure is None
    assert bundle.design.control_arm is not None and bundle.design.control_arm.variant_key == old_key
    assert run_id in bundle.design.control_arm.run_ids
    assert bundle_decision_surface(bundle).control_variant_key == old_key


def test_a_control_addressed_from_a_run_whose_map_does_not_reproduce_its_key_names_that_runs_cells() -> None:
    fixture = ops_fixture()
    eval_host = fixture.host.eval_host
    run_id = fixture.campaign.run_ids[0]
    run = eval_host.storage.load_eval_run(run_id, TOYHOST_SCOPE)
    assert run is not None
    levers = _levers_of(fixture, run_id)
    # Launched under the earlier predicate: the map recorded with its date, the cells stamped with the key that
    # predicate digested it to — which this build's arithmetic does not reproduce.
    eval_host.storage.save_eval_run(run.model_copy(update={"variant_levers": levers, "identity_version": _EARLIER}))
    old_key = _stamped_under_the_earlier_predicate(fixture, run_id)
    eval_host.storage.save_campaign(fixture.campaign.model_copy(update={"declared_design": toyhost_design()}))

    designated = set_campaign_control(
        eval_host.storage, fixture.campaign.id, TOYHOST_SCOPE, run_id, set_by="op", profile=eval_host.profile
    )
    bundle = _assemble(fixture, designated)

    assert designated.declared_design is not None
    assert (designated.declared_design.control, designated.declared_design.control_levers) == (old_key, levers)
    assert designated.declared_design.control_identity_version == _EARLIER
    assert bundle.design.control_excluded is None, "the run named as the control is the control"
    assert bundle.design.control_arm is not None and bundle.design.control_arm.variant_key == old_key
    assert run_id in bundle.design.control_arm.run_ids


def test_a_control_declared_under_this_predicate_is_its_one_key() -> None:
    levers = {"model": SweepableValue.of("glm", display="GLM")}
    design = minimal_declaration(control=compute_variant_key(levers), control_levers=levers)

    resolved = resolve_declared_control(design)

    assert (resolved.variant_keys, resolved.identity_version, resolved.source) == (
        [design.control],
        IDENTITY_VERSION,
        "current",
    )
    assert resolved.stale_disclosure is None
    undeclared = resolve_declared_control(None)
    assert undeclared == resolve_declared_control(minimal_declaration())
    assert (undeclared.variant_keys, undeclared.source) == ([], "undeclared")


# --- Every reading agrees on which arm is the control --------------------------------------------------------------


def _toy_campaign_with_an_earlier_control(*, cells_stamped_earlier: bool) -> tuple[Any, str]:
    """The toy campaign, its narrow arm declared the control under the earlier predicate; its bundle and that arm's key.

    With ``cells_stamped_earlier`` the narrow arm's cells carry the earlier predicate's key, and the control is the
    typed key they carry; otherwise they carry this build's, and the control is addressed with its lever map, so the
    declaration's own key is one no cell carries.
    """
    fixture = ops_fixture()
    narrow = variant_of_cell_ref(cell_at(_assemble(fixture, fixture.campaign), TOYHOST_NARROW))
    assert narrow is not None
    narrow_runs = [run_id for run_id in fixture.campaign.run_ids if _key_of(fixture, run_id) == narrow]
    levers = _levers_of(fixture, narrow_runs[0])
    old_key = _older_predicates_key(levers)
    if cells_stamped_earlier:
        for run_id in narrow_runs:
            assert _stamped_under_the_earlier_predicate(fixture, run_id) == old_key
    assert fixture.campaign.declared_design is not None
    declared = CampaignDesign.model_validate(
        {
            **fixture.campaign.declared_design.model_dump(mode="json"),
            "control": old_key,
            "control_levers": None
            if cells_stamped_earlier
            else {name: level.model_dump(mode="json") for name, level in levers.items()},
            "control_identity_version": _EARLIER,
        }
    )
    bundle = _assemble(fixture, fixture.campaign.model_copy(update={"declared_design": declared}))
    return bundle, (old_key if cells_stamped_earlier else narrow)


@pytest.mark.parametrize("cells_stamped_earlier", [False, True], ids=["rekeyed", "stamped-earlier"])
def test_every_bundle_reading_names_the_arm_the_realized_design_matched(cells_stamped_earlier: bool) -> None:
    bundle, expected = _toy_campaign_with_an_earlier_control(cells_stamped_earlier=cells_stamped_earlier)

    assert bundle.design.control_arm is not None and bundle.design.control_arm.variant_key == expected
    surface = bundle_decision_surface(bundle)
    assert surface.control_variant_key == expected
    assert bundle.cell_measures[0].variant_key == expected, "the surface's rows lead with the control's cells"
    for position in bundle.time_axis.positions if bundle.time_axis is not None else []:
        if any(cell.variant_key == expected for cell in position.cells):
            assert position.cells[0].variant_key == expected, "each time position leads with the control too"


async def test_the_adoption_check_and_the_report_name_the_arm_the_realized_design_matched() -> None:
    bundle, expected = off_loop(_toy_campaign_with_an_earlier_control, cells_stamped_earlier=False)
    assert bundle.declared_design is not None and bundle.declared_design.control != expected
    payload = memo_payload(bundle)
    payload["decisions"] = [
        {
            "proposal": "Keep the chunk width the pipeline runs today.",
            "disposition": "adopted",
            "cells": [alias_at(bundle, TOYHOST_NARROW)],
            "confidence": "medium",
            "rests_on": [0],
            "revisit_when": "",
        }
    ]
    # Every reading unseparated: adopting any arm but the control is refused, so acceptance proves the check exempted
    # the arm the realized design matched — not the declaration's own key, which no cell here carries.
    families = [
        family.model_copy(
            update={"comparisons": [c.model_copy(update={"verdict": "not_separated"}) for c in family.comparisons]}
        )
        for family in bundle.multiple_comparisons.families
    ]
    bundle = bundle.model_copy(
        update={"multiple_comparisons": bundle.multiple_comparisons.model_copy(update={"families": families})}
    )
    client = FixturedClient(json.dumps(payload))

    analysis, _insights = await generate_analysis(
        bundle,
        prompt=PROMPT,
        model=MODEL,
        client=client,
        prompt_id=PROMPT_ID,
        bundle_assembled_at=utc_now_iso(),
        profile=toyhost_profile(),
    )

    assert len(client.calls) == 1, "adopting the control needs no separation"
    assert analysis.decision_surface.control_variant_key == expected
    report = build_report(analysis)
    (scoreboard,) = [block for block in report.blocks if isinstance(block, TableBlock) and block.name == "scoreboard"]
    assert sum(str(row["arm"]).startswith("▸ ") for row in scoreboard.values()) == 1
    assert report.bundle.design.control_arm is not None and report.bundle.design.control_arm.variant_key == expected


# --- A key nothing can carry is disclosed, never silent ------------------------------------------------------------


def test_a_typed_control_no_cell_carries_under_an_earlier_predicate_is_disclosed_with_its_remedy() -> None:
    fixture = ops_fixture()
    campaign = _declared_under_an_earlier_predicate(fixture, fixture.campaign.run_ids[0], with_levers=False)

    bundle = _assemble(fixture, campaign)
    report = build_code_only_report(bundle, assembled_at=utc_now_iso())

    resolved = resolve_declared_control(campaign.declared_design)
    assert campaign.declared_design is not None
    assert (resolved.variant_keys, resolved.identity_version, resolved.source) == (
        [campaign.declared_design.control],
        _EARLIER,
        "unrekeyable",
    )
    assert bundle.design.control_arm is None
    assert bundle.design.control_excluded == "stale", "a stale control is not a campaign that declared none"
    assert bundle_decision_surface(bundle).control_variant_key is None, "no reading names a control nothing carries"
    disclosure = bundle.design.stale_control_disclosure
    assert disclosure is not None and disclosure == resolved.stale_disclosure
    assert f"v{_EARLIER}" in disclosure and f"v{IDENTITY_VERSION}" in disclosure
    assert "control_from_run_id on campaign_update" in disclosure
    assert disclosure in [getattr(block, "text", None) for block in report.blocks]


def test_a_control_that_re_keys_carries_no_stale_disclosure() -> None:
    fixture = ops_fixture()
    campaign = _declared_under_an_earlier_predicate(fixture, fixture.campaign.run_ids[0], with_levers=True)

    bundle = _assemble(fixture, campaign)

    assert bundle.design.stale_control_disclosure is None
    assert resolve_declared_control(campaign.declared_design).stale_disclosure is None


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

    assert resolve_declared_control(design).variant_keys == [compute_variant_key(levers), design.control]


# --- The pre-image is the server's ---------------------------------------------------------------------------------


def _echoed(design: CampaignDesign | None, **changes: Any) -> dict[str, Any]:
    """``design`` as a caller echoes it back from a read, without the stamps the surface writes."""
    assert design is not None
    echoed = design.model_dump(mode="json", exclude={"declared_at", "declared_by"})
    echoed.update(changes)
    return echoed


async def _refused(tool: MountedTool, fixture: OpsFixture, action: str, **arguments: Any) -> str:
    outcome = await tool.call({"action": action, **arguments}, host=fixture.host, caller=CALLER)
    assert outcome.is_error, outcome.text
    return str(outcome.text)


def _before_its_runs(fixture: OpsFixture, campaign: EvalCampaign) -> EvalCampaign:
    """``campaign`` stored holding no run, as before its first one joined: the only state its design is written in.

    A design is frozen once a run has joined (``test_campaign_freeze.py``), so the server's handling of an echoed or
    hand-changed declaration is exercised before the runs; a reading of it puts them back.
    """
    emptied = campaign.model_copy(update={"run_ids": []})
    fixture.host.eval_host.storage.save_campaign(emptied)
    return emptied


async def test_an_echoed_typed_control_keeps_its_stored_date_so_it_stays_stale(evals: MountedTool) -> None:
    fixture = off_loop(ops_fixture)
    campaign = off_loop(_declared_under_an_earlier_predicate, fixture, fixture.campaign.run_ids[0], with_levers=False)
    campaign = off_loop(_before_its_runs, fixture, campaign)
    echoed = _echoed(campaign.declared_design)
    del echoed["control_levers"], echoed["control_identity_version"]

    await _call(evals, fixture, "campaign_update", campaign_id=campaign.id, declared_design=echoed)

    stored = off_loop(_stored, fixture, campaign.id)
    assert stored.declared_design is not None and campaign.declared_design is not None
    assert stored.declared_design.control == campaign.declared_design.control
    assert stored.declared_design.control_identity_version == _EARLIER, "an echo is not re-dated as current"
    with_its_runs = stored.model_copy(update={"run_ids": list(fixture.campaign.run_ids)})
    assert off_loop(_assemble, fixture, with_its_runs).design.control_excluded == "stale"


async def test_a_declaration_echoed_whole_keeps_its_pre_image(evals: MountedTool) -> None:
    fixture = off_loop(ops_fixture)
    campaign = off_loop(_declared_under_an_earlier_predicate, fixture, fixture.campaign.run_ids[0], with_levers=True)
    campaign = off_loop(_before_its_runs, fixture, campaign)

    await _call(
        evals, fixture, "campaign_update", campaign_id=campaign.id, declared_design=_echoed(campaign.declared_design)
    )

    stored = off_loop(_stored, fixture, campaign.id).declared_design
    assert stored is not None and campaign.declared_design is not None
    assert (stored.control, stored.control_levers, stored.control_identity_version) == (
        campaign.declared_design.control,
        campaign.declared_design.control_levers,
        _EARLIER,
    )


async def test_a_hand_changed_key_sent_with_the_old_pre_image_is_refused(evals: MountedTool) -> None:
    fixture = off_loop(ops_fixture)
    campaign = off_loop(_declared_under_an_earlier_predicate, fixture, fixture.campaign.run_ids[0], with_levers=True)
    campaign = off_loop(_before_its_runs, fixture, campaign)

    message = await _refused(
        evals,
        fixture,
        "campaign_update",
        campaign_id=campaign.id,
        declared_design=_echoed(campaign.declared_design, control="c" * 64),
    )

    assert "`control_identity_version` and `control_levers`" in message
    assert "control_from_run_id" in message, "the refusal names the way to record a lever map"
    stored = off_loop(_stored, fixture, campaign.id).declared_design
    assert stored == campaign.declared_design, "a refused update writes nothing"


async def test_a_hand_changed_key_alone_has_no_map_and_is_dated_current(evals: MountedTool) -> None:
    fixture = off_loop(ops_fixture)
    campaign = off_loop(_declared_under_an_earlier_predicate, fixture, fixture.campaign.run_ids[0], with_levers=True)
    campaign = off_loop(_before_its_runs, fixture, campaign)
    echoed = _echoed(campaign.declared_design, control="c" * 64)
    del echoed["control_levers"], echoed["control_identity_version"]

    await _call(evals, fixture, "campaign_update", campaign_id=campaign.id, declared_design=echoed)

    stored = off_loop(_stored, fixture, campaign.id).declared_design
    assert stored is not None
    assert (stored.control, stored.control_levers, stored.control_identity_version) == (
        "c" * 64,
        None,
        IDENTITY_VERSION,
    )


async def test_a_pre_image_sent_on_create_is_refused(evals: MountedTool) -> None:
    fixture = off_loop(ops_fixture)
    run_id = fixture.campaign.run_ids[0]
    levers = off_loop(_levers_of, fixture, run_id)

    message = await _refused(
        evals,
        fixture,
        "campaign_create",
        name="a typed pre-image",
        subject_id=TOYHOST_SUBJECT.subject_id,
        behavior=fixture.campaign.behavior,
        run_ids=list(fixture.campaign.run_ids),
        declared_design=_design(
            control=compute_variant_key(levers),
            control_levers={name: level.model_dump(mode="json") for name, level in levers.items()},
        ),
    )

    assert "`control_levers`" in message


def test_a_model_whose_key_was_changed_beside_its_stored_map_is_refused() -> None:
    fixture = ops_fixture()
    eval_host = fixture.host.eval_host
    campaign = _before_its_runs(
        fixture, _declared_under_an_earlier_predicate(fixture, fixture.campaign.run_ids[0], with_levers=True)
    )
    assert campaign.declared_design is not None
    hand_edited = campaign.declared_design.model_copy(update={"control": "c" * 64})

    with pytest.raises(ValidationFailedError, match="`control_levers`"):
        update_campaign(
            eval_host.storage,
            campaign.id,
            TOYHOST_SCOPE,
            {"declared_design": hand_edited},
            updated_by="op",
            profile=eval_host.profile,
        )


# --- A campaign carrying both of the control's keys keeps one control arm, and says so --------------------------


#: The day the control's re-keyed run ran — after the predicate changed — so the time axis has a position before it.
_AFTER_THE_CHANGE = "2026-03-15T09:30:00+00:00"


def _split_control_campaign() -> tuple[OpsFixture, EvalCampaign, str, str, str, str]:
    """The corpus campaign, its control's arm run twice: once under the earlier predicate, once under this one.

    The first control run's cells carry the stored key the earlier predicate stamped; a second run of the same
    arm, a day later, carries the key this build stamps. The control is declared under the earlier predicate with
    its lever map, so it matches both.

    Returns:
        The fixture, the stored campaign, the earlier run, the re-keyed run, the stored key and the re-keyed key.
    """
    fixture = ops_fixture()
    storage = fixture.host.eval_host.storage
    earlier_run = fixture.campaign.run_ids[0]
    run = storage.load_eval_run(earlier_run, TOYHOST_SCOPE)
    assert run is not None
    current_key = _key_of(fixture, earlier_run)
    rerun = run.model_copy(update={"id": f"{earlier_run}-rerun", "created_at": _AFTER_THE_CHANGE})
    storage.save_eval_run(rerun)
    for result in storage.query_eval_results_by_run(earlier_run, TOYHOST_SCOPE):
        storage.save_eval_result(result.model_copy(update={"id": f"{result.id}-rerun", "eval_run_id": rerun.id}))
    stored_key = _stamped_under_the_earlier_predicate(fixture, earlier_run)
    campaign = _declared_under_an_earlier_predicate(fixture, earlier_run, with_levers=True)
    campaign = campaign.model_copy(update={"run_ids": [*campaign.run_ids, rerun.id]})
    storage.save_campaign(campaign)
    return fixture, campaign, earlier_run, rerun.id, stored_key, current_key


def test_a_campaign_carrying_both_keys_keeps_one_control_arm_and_names_the_earlier_cells() -> None:
    fixture, campaign, earlier_run, rerun, stored_key, current_key = _split_control_campaign()

    bundle = _assemble(fixture, campaign)

    assert bundle.design.control_arm is not None
    assert (bundle.design.control_arm.variant_key, bundle.design.control_arm.run_ids) == (current_key, [rerun]), (
        "the control's numbers are the current predicate's cells only"
    )
    earlier = bundle.design.earlier_control
    assert earlier is not None
    assert (earlier.variant_key, earlier.identity_version, earlier.run_ids) == (stored_key, _EARLIER, [earlier_run])
    assert bundle.design.control_excluded is None
    assert bundle_decision_surface(bundle).earlier_control == earlier


def test_the_earlier_control_cells_are_in_no_contrast_and_no_guardrail() -> None:
    fixture, campaign, _earlier_run, _rerun, stored_key, _current_key = _split_control_campaign()

    bundle = _assemble(fixture, campaign)

    assert stored_key not in {contrast.variant_key for contrast in bundle.design.contrasts}
    assert bundle.design.contrasts, "the campaign's other arm is still contrasted"
    compared = {c.contrast.variant_key for f in bundle.multiple_comparisons.families for c in f.comparisons}
    assert compared and stored_key not in compared, "the control is never compared with itself under another digest"
    assert stored_key not in {check.contrast.variant_key for check in bundle.guardrails.checks}
    for subject in bundle.frontier.subjects:
        for point in subject.points:
            if point.variant_key == stored_key:
                assert point.boundary_checks == [] and point.disqualified_by == []


def test_the_earlier_control_point_is_no_contestant_on_the_frontier() -> None:
    fixture, campaign, _earlier_run, _rerun, stored_key, _current_key = _split_control_campaign()
    storage = fixture.host.eval_host.storage
    bundle = _assemble(fixture, campaign)
    runs = [run for run_id in campaign.run_ids if (run := storage.load_eval_run(run_id, TOYHOST_SCOPE)) is not None]
    results = [result for run in runs for result in storage.query_eval_results_by_run(run.id, TOYHOST_SCOPE)]
    assert bundle.design.control_arm is not None and bundle.design.earlier_control is not None

    # A bar every point clears, so the pick and its cost ties are read over every point. The earlier cells are the
    # control's own observations restamped, so as a contestant they would tie with the control on cost.
    frontier = compute_frontier(
        runs,
        results,
        bar=0.0,
        known_run_ids={run.id for run in runs},
        archived_run_ids=None,
        profile=fixture.host.eval_host.profile,
        control_variant_key=bundle.design.control_arm.variant_key,
        earlier_control=bundle.design.earlier_control,
    )

    for lens in (frontier, bundle.frontier):
        (earlier,) = [point for subject in lens.subjects for point in subject.points if point.variant_key == stored_key]
        assert earlier.earlier_control == f"the control under identity predicate v{_EARLIER}"
        assert (earlier.dominated, earlier.dominated_by, earlier.dominance) == (False, [], "untested")
        assert earlier.boundary_checks == [] and earlier.disqualified_by == []
        others = [point for subject in lens.subjects for point in subject.points if point.variant_key != stored_key]
        assert others and all(point.earlier_control is None for point in others)
        assert stored_key not in {rival.variant_key for point in others for rival in point.dominated_by}
    (subject,) = frontier.subjects
    assert subject.verdict is not None, "with every point clearing the bar there is a pick"
    assert subject.verdict.variant_key != stored_key, "the control under an earlier predicate is never the pick"
    assert stored_key not in {tie.variant_key for tie in subject.verdict.tied_with}


def test_the_split_is_disclosed_in_the_bundle_and_the_code_only_report() -> None:
    fixture, campaign, _earlier_run, _rerun, stored_key, _current_key = _split_control_campaign()

    bundle = _assemble(fixture, campaign)
    report = build_code_only_report(bundle, assembled_at=utc_now_iso())

    earlier = bundle.design.earlier_control
    assert earlier is not None
    assert f"v{_EARLIER}" in earlier.disclosure and f"v{IDENTITY_VERSION}" in earlier.disclosure
    assert f"The control's numbers are the v{IDENTITY_VERSION} cells only" in earlier.disclosure
    texts = [getattr(block, "text", None) for block in report.blocks]
    assert earlier.disclosure in texts
    (scoreboard,) = [block for block in report.blocks if isinstance(block, TableBlock) and block.name == "scoreboard"]
    assert sum(str(row["arm"]).startswith("▸ ") for row in scoreboard.values()) == 1, (
        "the earlier cells are never a second control"
    )
    assert bundle.design.control_arm is not None and stored_key != bundle.design.control_arm.variant_key


def test_a_marginal_row_pools_the_earlier_control_cells_at_the_controls_level() -> None:
    fixture, campaign, earlier_run, _rerun, _stored_key, _current_key = _split_control_campaign()
    storage = fixture.host.eval_host.storage
    # A declared axis no arm moved: its row is read over the whole campaign, a marginal comparison, beside the swept
    # axis's row, which is a contrast against the control.
    unmoved = SweptAxis(
        axis_id="retriever_top_k",
        values=[SweepableValue.of(level, display=str(level)) for level in (3, 5)],
        rationale="does a deeper retrieval find the field a shallow one misses",
    )
    declared = campaign.declared_design
    assert declared is not None
    campaign = campaign.model_copy(
        update={
            "declared_design": CampaignDesign.model_validate(
                declared.model_copy(update={"axes": [*declared.axes, unmoved]}).model_dump(mode="json")
            )
        }
    )
    storage.save_campaign(campaign)
    results = {run_id: len(storage.query_eval_results_by_run(run_id, TOYHOST_SCOPE)) for run_id in campaign.run_ids}

    bundle = _assemble(fixture, campaign)

    rows = {row.name: row for row in bundle.coverage}
    contrast, marginal = rows[TOYHOST_AXIS], rows[unmoved.axis_id]
    assert contrast.cohort_scope == "control_referenced"
    assert contrast.n == sum(results.values()) - results[earlier_run], "a contrast reads the control's current cells"
    assert marginal.cohort_scope == "campaign"
    assert marginal.n == sum(results.values()), "a marginal row counts every cell, the earlier control's among them"
    assert bundle.design.control_arm is not None
    (control_summary,) = [s for s in bundle.run_summaries if s.run_id in bundle.design.control_arm.run_ids]
    (earlier_summary,) = [s for s in bundle.run_summaries if s.run_id == earlier_run]
    assert earlier_summary.config[unmoved.axis_id] == control_summary.config[unmoved.axis_id]
    assert marginal.levels == [control_summary.config[unmoved.axis_id]], (
        "the earlier cells pool at the control's level, never a level of their own"
    )
    assert bundle.design.earlier_control is not None and bundle.design.earlier_control.run_ids == [earlier_run], (
        "the pool includes them, so the bundle still names them"
    )
    assert "a marginal row" in bundle.design.earlier_control.disclosure and "control's level" in (
        bundle.design.earlier_control.disclosure
    ), "the disclosure says what the row counts, not that the earlier cells are compared with nothing"


def test_a_time_position_before_the_change_names_its_control_rather_than_reading_as_having_none() -> None:
    fixture, campaign, _earlier_run, _rerun, stored_key, current_key = _split_control_campaign()

    bundle = _assemble(fixture, campaign)
    report = build_code_only_report(bundle, assembled_at=utc_now_iso())

    assert bundle.time_axis is not None
    before, after = bundle.time_axis.positions
    assert before.cells[0].variant_key == stored_key, "the earlier cells lead as the reference before the change"
    assert before.control_disclosure is not None
    assert f"control under identity predicate v{_EARLIER}" in before.control_disclosure
    assert after.cells[0].variant_key == current_key and after.control_disclosure is None
    assert before.control_disclosure in [getattr(block, "text", None) for block in report.blocks]


async def test_keeping_the_control_under_the_earlier_predicate_needs_no_separation() -> None:
    fixture, campaign, _earlier_run, rerun, stored_key, _current_key = off_loop(_split_control_campaign)
    bundle = off_loop(_assemble, fixture, campaign)
    # The toy memo names the narrow arm by its one run; read without the rerun's summary, that is the earlier cells.
    payload = memo_payload(
        bundle.model_copy(update={"run_summaries": [s for s in bundle.run_summaries if s.run_id != rerun]})
    )
    (decision,) = payload["decisions"]
    assert [variant_of_cell_ref(cell_aliases(bundle.cell_measures)[decision["cells"][0]])] == [stored_key]
    payload["decisions"] = [{**decision, "disposition": "adopted", "revisit_when": ""}]
    client = FixturedClient(json.dumps(payload))

    analysis, _insights = await generate_analysis(
        bundle,
        prompt=PROMPT,
        model=MODEL,
        client=client,
        prompt_id=PROMPT_ID,
        bundle_assembled_at=utc_now_iso(),
        profile=fixture.host.eval_host.profile,
    )

    assert len(client.calls) == 1, "the earlier cells are the control: adopting them is keeping it, never checked"
    assert analysis.decision_surface.earlier_control == bundle.design.earlier_control
    assert analysis.decision_surface.earlier_control is not None
    assert analysis.decision_surface.earlier_control.variant_key == stored_key
