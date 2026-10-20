"""The person's call as a campaign's page states it, and the page written to a directory.

The call line says every override the call stored, as stored: one the readings on the page no longer warn of is
marked so, never dropped. It says how many runs joined after the call. And a page whose aspect labels read alike as
ids writes each aspect's bundle to its own file.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.analysis import (
    AnalysisContextBundle,
    CampaignReadings,
    CampaignReport,
    build_campaign_report,
    build_code_only_report,
    campaign_readings,
    campaign_report_markdown,
)
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.declaration import CampaignCall, OverriddenBar, RateDrop

from packages.evals.tests.campaign_support import (
    ASSEMBLED_AT,
    CANDIDATE_LABEL,
    MeasuredAspect,
    arms_of,
    measured_aspect,
)

_MEASURED: dict[str, MeasuredAspect] = {}


async def _billing() -> MeasuredAspect:
    """Billing, where the candidate passes six cases to production's eighteen and breaks its 10-point floor."""
    if "billing" not in _MEASURED:
        _MEASURED["billing"] = await measured_aspect(
            "billing", production=18, candidate=6, floors=(RateDrop(max_points=10),)
        )
    return _MEASURED["billing"]


def _page(
    measured: list[MeasuredAspect], campaign: EvalCampaign, call: CampaignCall, readings: CampaignReadings
) -> CampaignReport:
    bundles: dict[str, AnalysisContextBundle] = {one.bundle.aspect: one.bundle for one in measured}
    reports = {
        label: build_code_only_report(bundle, assembled_at=ASSEMBLED_AT, campaign_name="support")
        for label, bundle in bundles.items()
    }
    return build_campaign_report(campaign, readings, reports, None, call=call)


def _call(measured: MeasuredAspect, **fields: Any) -> CampaignCall:
    return CampaignCall(
        arm=CANDIDATE_LABEL,
        variant_key=measured.candidate,
        called_at="2026-10-18T09:00:00+00:00",
        called_by="brooks",
        **fields,
    )


def _first_screen(page: CampaignReport) -> str:
    return campaign_report_markdown(page).split("\n## ", 1)[0]


async def test_an_override_the_page_no_longer_warns_of_is_said_as_stored_and_marked_so() -> None:
    """The call overrode billing's floor and a cost bar. On a page whose readings hold the floor and miss no bar,
    both overrides are still said, each marked as no longer warning."""
    billing = await _billing()
    campaign = billing.campaign
    read = campaign_readings(campaign, {"billing": billing.bundle}, arms=arms_of(billing))
    (floor,) = [status.floor for status in read.broken_floors()]
    call = _call(
        billing, floors_overridden=[floor], bars_overridden=[OverriddenBar(aspect="billing", measure="cost_usd")]
    )

    broken = _first_screen(_page([billing], campaign, call, read))
    assert "The call overrides a broken floor: no pass rate drops by more than 10 points on billing." in broken

    held = read.model_copy(update={"floors": [status.model_copy(update={"status": "held"}) for status in read.floors]})
    first = _first_screen(_page([billing], campaign, call, held))
    assert f"The call overrides the floor `{floor}` (no longer broken)." in first
    assert "The call overrides the bar on `cost_usd` on billing (no longer missed)." in first
    assert "does not name" not in first


async def test_the_call_line_counts_the_runs_added_since_the_call() -> None:
    billing = await _billing()
    campaign = billing.campaign
    read = campaign_readings(campaign, {"billing": billing.bundle}, arms=arms_of(billing))
    (floor,) = [status.floor for status in read.broken_floors()]
    first_run, *later = campaign.run_ids
    assert later, "the fixture's campaign holds more than one run"

    since = _first_screen(
        _page([billing], campaign, _call(billing, floors_overridden=[floor], run_ids=[first_run]), read)
    )
    assert f"Called: adopt {CANDIDATE_LABEL}, by brooks on 18 Oct 2026, over 1 run; {len(later)} added since." in since

    current = _call(billing, floors_overridden=[floor], run_ids=list(campaign.run_ids))
    assert f"Called: adopt {CANDIDATE_LABEL}, by brooks on 18 Oct 2026." in _first_screen(
        _page([billing], campaign, current, read)
    )


async def test_two_aspects_whose_labels_read_alike_as_ids_write_two_bundles(tmp_path: Any) -> None:
    """``Tone`` and ``tone`` are two aspects; their ids read alike, so the second is numbered, and each aspect's
    bundle is written to its own file and reads back as that aspect's."""
    upper = await measured_aspect("Tone", production=6, candidate=19)
    lower = await measured_aspect("tone", production=6, candidate=19)
    aspects = [*upper.campaign.aspects, *lower.campaign.aspects]
    campaign = upper.campaign.model_copy(update={"aspects": aspects})
    read = campaign_readings(campaign, {"Tone": upper.bundle, "tone": lower.bundle}, arms=arms_of(upper))
    page = _page([upper, lower], campaign, _call(upper), read)
    assert page.fold_ids == ["tone", "tone-2"]

    files = page.write(tmp_path)

    assert (files["bundle:Tone"].name, files["bundle:tone"].name) == ("bundle-tone.json", "bundle-tone-2.json")
    for label in ("Tone", "tone"):
        written = AnalysisContextBundle.from_json(files[f"bundle:{label}"].read_text(encoding="utf-8"))
        assert written.aspect == label
