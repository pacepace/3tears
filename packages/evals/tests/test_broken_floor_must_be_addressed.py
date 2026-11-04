"""A broken floor warns and never blocks, and the writer's view of the campaign says so first.

Owner ruling (one memo per campaign): code reports, the writer recommends, a person decides; floors only warn, loudly,
and the writer must address a broken one. Which floors broke is code's (``CampaignReadings.floors``), and the writer's
view (:func:`~threetears.evals.analysis.campaign_bundle.build_campaign_user_message`) lists the floors first, the
broken ones first among them, under a warning. The memo writer's refusal of a recommendation that left a broken floor
unaddressed retired with the memo writer.

Driven over the toy campaign, the narrow chunk width its control, with a floor code read broken (or held) on the wide
width.
"""

from __future__ import annotations

import json

from packages.evals.tests.aspect_support import with_control
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_NARROW, TOYHOST_WIDE, toyhost_campaign
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.toyhost_cells import arm_at, cell_at
from threetears.evals.analysis import CampaignBundle, assemble_context_bundle
from threetears.evals.analysis.campaign_bundle import build_campaign_user_message
from threetears.evals.analysis.campaign_readings import FloorState, FloorStatus
from threetears.evals.analysis.cells import variant_of_cell_ref


def _floor(
    bundle: CampaignBundle, key: str, *, level: int = TOYHOST_WIDE, status: FloorState = "broken"
) -> FloorStatus:
    variant = variant_of_cell_ref(cell_at(bundle, level))
    assert variant is not None
    return FloorStatus(
        floor=key,
        aspect=bundle.labels[0],
        arm=variant,
        arm_label=arm_at(bundle, level),
        status=status,
        reading=None,
        interval=(-30.0, -12.0) if status == "broken" else (-4.0, 6.0),
        reason="the wide width passed 12 points fewer trials than the narrow one, past the floor's 5",
    )


def _bundle(*statuses: str, status: FloorState = "broken") -> CampaignBundle:
    """The toy campaign, the narrow width its control, each floor key in ``statuses`` read ``status`` on the wide."""
    profile = toyhost_profile()
    campaign, storage = toyhost_campaign(profile=profile)
    bundle = assemble_context_bundle(campaign, storage=storage, profile=profile)
    campaign = with_control(campaign, variant_of_cell_ref(cell_at(bundle, TOYHOST_NARROW)))
    aspect = assemble_context_bundle(campaign, storage=storage, profile=profile)
    bare = campaign_bundle(aspect)
    return campaign_bundle(aspect, floors=[_floor(bare, key, status=status) for key in statuses])


def _key(bundle: CampaignBundle, kind: str = "rate_drop") -> str:
    return f"{bundle.labels[0]}/{kind}/*"


class TestTheWriterIsWarned:
    def test_a_broken_floor_comes_first_in_the_writer_s_view_under_a_warning(self) -> None:
        held, broken = _key(_bundle(), "cost_lower"), _key(_bundle())
        bare = _bundle()
        bundle = campaign_bundle(
            *bare.aspects, floors=[_floor(bare, held, status="held"), _floor(bare, broken, status="broken")]
        )
        view = json.loads(build_campaign_user_message(bundle).split("\n", 1)[1])
        # The arm's floors are rolled up by status, broken first.
        [arm] = view["floors"]
        assert list(arm) == ["arm", "broken", "held"]
        assert [floor["floor"] for state in ("broken", "held") for floor in arm[state]] == [broken, held]
        assert view["floor_warning"].startswith(f"1 floor(s) broken: {broken} on {arm_at(bundle, TOYHOST_WIDE)}")
        assert "`floors_overridden`" in view["floor_warning"]

    def test_with_no_floor_broken_there_is_no_warning(self) -> None:
        bundle = _bundle(_key(_bundle()), status="held")
        view = json.loads(build_campaign_user_message(bundle).split("\n", 1)[1])
        assert "floor_warning" not in view
