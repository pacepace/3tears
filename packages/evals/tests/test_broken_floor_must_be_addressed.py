"""A broken floor warns and never blocks, but a recommendation of the arm that broke it must address it.

Owner ruling (one memo per campaign): code reports, the writer recommends, a person decides; floors only warn, loudly,
and the writer must address a broken one. So a recommendation of an arm with a broken floor names each such floor in
``floors_overridden``, by its key, with why the breach is acceptable, and the generator refuses one that does not —
as it refuses every other memo fault, through the one repair round. It refuses as well a floor named there that the
arm did not break, one named twice, and one named with no reason. Which floors broke is code's
(``CampaignReadings.floors``); whether the reason is good is the reader's.

The writer is told, too: the floors come first in its view, the broken ones first among them, under a warning.

Driven through :func:`~threetears.evals.analysis.generator.generate_analysis` over the toy campaign, recommending the
wide chunk width (which improved on field accuracy, so it passes the separation check) with a floor code read broken
on it.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from packages.evals.tests.aspect_support import with_control
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_NARROW, TOYHOST_WIDE, toyhost_campaign
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.toyhost_memo import (
    MODEL,
    PROMPT,
    PROMPT_ID,
    FixturedClient,
    FixturedCompletion,
    arm_at,
    cell_at,
    memo_payload,
)
from threetears.evals.analysis import CampaignBundle, assemble_context_bundle, generate_analysis
from threetears.evals.analysis.campaign_bundle import build_campaign_user_message
from threetears.evals.analysis.campaign_readings import FloorState, FloorStatus
from threetears.evals.analysis.cells import variant_of_cell_ref
from threetears.evals.analysis.errors import SoundnessRefusal
from threetears.evals.kernel.campaign import EvalAnalysis
from threetears.evals.schema.models import utc_now_iso


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


def _recommending_wide(bundle: CampaignBundle, *overridden: dict[str, str]) -> dict[str, Any]:
    payload = memo_payload(bundle)
    payload["recommendation"][0]["arm"] = arm_at(bundle, TOYHOST_WIDE)
    payload["recommendation"][0]["floors_overridden"] = list(overridden)
    return payload


async def _generate(bundle: CampaignBundle, payload: dict[str, Any]) -> EvalAnalysis:
    analysis, _ = await generate_analysis(
        bundle,
        prompt=PROMPT,
        model=MODEL,
        client=FixturedClient(json.dumps(payload)),
        prompt_id=PROMPT_ID,
        bundle_assembled_at=utc_now_iso(),
        profile=toyhost_profile(),
    )
    return analysis


class TestABrokenFloorMustBeAddressed:
    async def test_recommending_an_arm_over_its_broken_floor_unaddressed_is_refused(self) -> None:
        key = _key(_bundle())
        bundle = _bundle(key)
        with pytest.raises(SoundnessRefusal) as refused:
            await _generate(bundle, _recommending_wide(bundle))
        message = str(refused.value)
        assert f"the recommendation names {arm_at(bundle, TOYHOST_WIDE)!r}, which broke floor(s) [{key!r}]" in message
        assert "A floor only warns, so the arm may still be recommended" in message
        assert "names each broken floor in `floors_overridden` with why breaking it is acceptable" in message
        # The refusal carries code's reason, so the repair can address the floor as code read it.
        assert "passed 12 points fewer trials" in message

    async def test_naming_the_broken_floor_and_why_is_kept(self) -> None:
        bundle = _bundle(_key(_bundle()))
        key = _key(bundle)
        why = "The accuracy gain is what this campaign is for, and the pass-rate loss is in cases we no longer serve."
        analysis = await _generate(bundle, _recommending_wide(bundle, {"floor": key, "why": why}))
        (override,) = analysis.document.recommendation[0].floors_overridden
        assert (override.floor, override.why) == (key, why)

    async def test_every_broken_floor_must_be_named_not_just_one(self) -> None:
        first, second = _key(_bundle()), _key(_bundle(), "cost_lower")
        bundle = _bundle(first, second)
        with pytest.raises(SoundnessRefusal, match=re.escape(f"broke floor(s) [{second!r}]")):
            await _generate(bundle, _recommending_wide(bundle, {"floor": first, "why": "Accuracy matters more."}))

    async def test_a_floor_named_with_no_reason_is_refused(self) -> None:
        bundle = _bundle(_key(_bundle()))
        with pytest.raises(SoundnessRefusal, match="gives no reason in `floors_overridden`"):
            await _generate(bundle, _recommending_wide(bundle, {"floor": _key(bundle), "why": "  "}))

    async def test_a_floor_named_twice_is_refused(self) -> None:
        bundle = _bundle(_key(_bundle()))
        override = {"floor": _key(bundle), "why": "Accuracy matters more."}
        with pytest.raises(SoundnessRefusal, match="more than once in `floors_overridden`"):
            await _generate(bundle, _recommending_wide(bundle, override, override))

    async def test_a_floor_the_arm_did_not_break_is_refused_in_floors_overridden(self) -> None:
        bundle = _bundle(_key(_bundle()), status="held")
        with pytest.raises(SoundnessRefusal, match=r"by floor, and code raised no such warning on .* \(it raised: none\)"):
            await _generate(bundle, _recommending_wide(bundle, {"floor": _key(bundle), "why": "Accuracy matters."}))

    async def test_a_floor_broken_on_another_arm_needs_no_word_when_that_arm_is_not_recommended(self) -> None:
        bundle = _bundle(_key(_bundle()))
        payload = memo_payload(bundle)  # recommends the narrow control, which no floor is read on
        analysis = await _generate(bundle, payload)
        assert analysis.document.recommendation[0].floors_overridden == []


class TestTheWriterIsWarned:
    def test_a_broken_floor_comes_first_in_the_writer_s_view_under_a_warning(self) -> None:
        held, broken = _key(_bundle(), "cost_lower"), _key(_bundle())
        bare = _bundle()
        bundle = campaign_bundle(
            *bare.aspects, floors=[_floor(bare, held, status="held"), _floor(bare, broken, status="broken")]
        )
        view = json.loads(build_campaign_user_message(bundle).split("\n", 1)[1])
        assert [floor["floor"] for floor in view["floors"]] == [broken, held]
        assert view["floor_warning"].startswith(f"1 floor(s) broken: {broken} on {arm_at(bundle, TOYHOST_WIDE)}")
        assert "`floors_overridden`" in view["floor_warning"]

    def test_with_no_floor_broken_there_is_no_warning(self) -> None:
        bundle = _bundle(_key(_bundle()), status="held")
        view = json.loads(build_campaign_user_message(bundle).split("\n", 1)[1])
        assert "floor_warning" not in view


class _Drafts(FixturedClient):
    """Hands back one prepared memo per call, in order: the first draft, then its repair."""

    def __init__(self, *bodies: str) -> None:
        super().__init__(bodies[0])
        self.bodies = list(bodies)

    async def generate(self, *, system: str, user: str, response_format: Any = None, tools: Any = None) -> Any:
        self.completion = FixturedCompletion(content=self.bodies[len(self.calls)])
        return await super().generate(system=system, user=user, response_format=response_format, tools=tools)


class TestTheRepairRoundAddressesTheFloor:
    async def test_a_recommendation_silent_on_its_broken_floor_is_refused_repaired_and_stored(self) -> None:
        key = _key(_bundle())
        bundle = _bundle(key)
        why = "Accuracy is what this campaign is for, and the trials it loses are in a queue we retire."
        client = _Drafts(
            json.dumps(_recommending_wide(bundle)),
            json.dumps(_recommending_wide(bundle, {"floor": key, "why": why})),
        )
        analysis, _ = await generate_analysis(
            bundle,
            prompt=PROMPT,
            model=MODEL,
            client=client,
            prompt_id=PROMPT_ID,
            bundle_assembled_at=utc_now_iso(),
            profile=toyhost_profile(),
        )
        assert len(client.calls) == 2 and analysis.generation.repair_attempts == 1
        refusal = analysis.generation.repaired_refusal
        assert refusal is not None and f"broke floor(s) [{key!r}]" in refusal
        assert refusal in client.calls[1]["user"], "the repair is told which floor it left out"
        (override,) = analysis.document.recommendation[0].floors_overridden
        assert override.floor == key


class TestAFloorOnAnotherAspect:
    async def test_a_floor_broken_on_the_second_aspect_must_be_named_too(self) -> None:
        first = _bundle().aspects[0]
        second = first.model_copy(update={"aspect": "second"})
        bare = campaign_bundle(first, second)
        floor = _floor(bare, "second/rate_drop/*").model_copy(update={"aspect": "second"})
        bundle = campaign_bundle(first, second, floors=[floor])
        payload = _recommending_wide(bundle)
        payload["aspect_summaries"].append({**payload["aspect_summaries"][0], "aspect": "second"})
        with pytest.raises(SoundnessRefusal, match=re.escape("broke floor(s) ['second/rate_drop/*']")):
            await _generate(bundle, payload)
        payload["recommendation"][0]["floors_overridden"] = [{"floor": "second/rate_drop/*", "why": "Accuracy wins."}]
        analysis = await _generate(bundle, payload)
        assert [aspect.label for aspect in analysis.aspects] == [first.aspect, "second"]
