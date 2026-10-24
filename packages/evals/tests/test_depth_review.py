"""The depth disclosure's edges: arms that share no case, depths that vary within one arm, several frontier subjects, a
model beside a measured arm, and a frozen campaign bundle of the old shape.

The common depth is the fewest runs any arm gave a case every arm shares (owner, 2026-10-24: compare at min(k)).
Every expected value is worked from the fixture's attempts.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from threetears.evals.analysis import build_campaign_report, build_code_only_report
from threetears.evals.analysis.campaign_bundle import CampaignBundle, writer_view
from threetears.evals.analysis.depth import Depth
from threetears.evals.analysis.depth_reading import aspect_depth
from threetears.evals.analysis.report.model import CampaignReport, DisclosureBlock, TableBlock
from threetears.evals.kernel.authored import AuthoredCampaignAnalysis
from threetears.evals.analysis.stats import INTERVAL_LEVEL, bounded_difference_interval
from threetears.evals.kernel.declaration import RateDrop
from threetears.evals.kernel.errors import RefusedInput
from packages.evals.tests.campaign_bundle_support import campaign_bundle, memo
from packages.evals.tests.depth_support import ONCE, PROFILE, TWICE, TwoArms

_ASSEMBLED_AT = "2026-10-24T00:00:00+00:00"

#: Three runs of every case, passing 3, 2, 1 and 3 of them.
_THRICE = {
    "tc1": (True, True, True),
    "tc2": (True, True, False),
    "tc3": (True, False, False),
    "tc4": (True, True, True),
}

#: The page's line on a memo that never says its arms ran to different depths, on character.
_UNSAID = (
    "The memo does not say the arms ran to different depths on character; the Depth table and the sentence under it "
    "say so."
)


def _page(arms: TwoArms, summary: str | None = None) -> CampaignReport:
    read = arms.readings()
    reports = {
        "character": build_code_only_report(
            arms.bundle, assembled_at=_ASSEMBLED_AT, campaign_name="depth", arm_labels=read.arm_labels()
        )
    }
    written = None
    if summary is not None:
        bundle = campaign_bundle(arms.bundle, arms=arms.arms, readings=read.readings)
        written = AuthoredCampaignAnalysis.model_validate(memo(bundle, summary=summary))
    key = arms.key_a if written is not None else None
    return build_campaign_report(arms.campaign, read, reports, written, recommended_variant_key=key)


def _lines(page: CampaignReport, section: str) -> list[str]:
    return [block.text for block in page.blocks if isinstance(block, DisclosureBlock) and block.section == section]


# =============================================================================
# Arms that share no case
# =============================================================================


def test_arms_that_share_no_case_are_said_to_compare_nothing_in_the_bundle_the_writer_view_and_the_page() -> None:
    # A ran tc1 and tc2 twice; B ran tc3 and tc4 once: a pool across case sets.
    arms = TwoArms({"tc1": (True, True), "tc2": (True, False)}, {"tc3": (True,), "tc4": (False,)}, k_a=2, k_b=2)
    subject = arms.bundle.frontier.subjects[0]

    assert subject.common_depth is None and subject.unmeasured_for == [arms.key_b]
    assert subject.shares_no_case
    view = writer_view(campaign_bundle(arms.bundle, arms=arms.arms, readings=arms.readings().readings))
    [entry] = view["aspects"][0]["depth"]["pass_k"]
    assert (entry["common_depth"], entry["shares_no_case"], entry["unmeasured_for"]) == (None, True, ["B"])
    page = _page(arms)
    assert "On character the arms share no case, so pass^k compares nothing between them." in _lines(page, "scoreboard")
    assert "On character, pass^2 is unmeasured for B, which ran no case twice." in _lines(page, "scoreboard")


# =============================================================================
# Depths that vary within one arm
# =============================================================================


def test_one_arm_short_on_one_case_sets_the_common_depth_and_the_planned_depth_is_still_measured() -> None:
    # Both planned 3. B ran tc4 only twice: the fewest runs any arm gave a shared case is 2.
    short = {**_THRICE, "tc4": (True, True)}
    arms = TwoArms(_THRICE, short, k_a=3, k_b=3)
    subject = arms.bundle.frontier.subjects[0]
    b = arms.point(arms.key_b)

    assert b.depth == Depth(cases=4, trials=11, repeats_min=2, repeats_median=3.0, repeats_max=3)
    # B reached 3 on three cases, so pass^3 is measured for it, over those three.
    assert (subject.k, subject.common_depth, subject.unmeasured_for) == (3, 2, [])
    assert (b.n_pass_cases, b.n_cases_short_of_k) == (3, 1)
    # At depth 2 over all four cases: C(3,2)/C(3,2) = 1, C(2,2)/C(3,2) = 1/3, 0, 1 on A, so 7/12; B the same but tc4's
    # two passes are C(2,2)/C(2,2) = 1, so 7/12 too.
    a = arms.point(arms.key_a)
    assert a.common_pass_hat_k == pytest.approx(7 / 12) and b.common_pass_hat_k == pytest.approx(7 / 12)
    assert (a.n_common_cases, a.n_cases_left_out) == (4, 0)
    assert not subject.shares_no_case
    page = _page(arms)
    assert (
        "On character, pass^k is compared at depth 2, the fewest runs any arm gave a shared case, over all runs."
        in _lines(page, "scoreboard")
    )


# =============================================================================
# The comparison, however it is worded
# =============================================================================


# =============================================================================
# Several subjects, several aspects
# =============================================================================


def test_every_frontier_subject_is_read_and_one_sharing_no_case_is_said_to() -> None:
    arms = TwoArms(TWICE, TWICE, k_a=2, k_b=2)
    [first] = arms.bundle.frontier.subjects
    second = first.model_copy(update={"subject_id": "subject-2", "subject_label": "second", "common_depth": None})
    bundle = arms.bundle.model_copy(
        update={"frontier": arms.bundle.frontier.model_copy(update={"subjects": [first, second]})}
    )

    depth = aspect_depth(bundle, {arms.key_a: "A", arms.key_b: "B"}.__getitem__)

    assert [subject.where for subject in depth.subjects] == ["character (Test subject)", "character (second)"]
    # The first subject shares every case twice; the second shares none.
    assert [subject.shares_no_case for subject in depth.subjects] == [False, True]
    assert depth.pass_k_lines() == [
        "On character (Test subject), pass^k is compared at depth 2, the fewest runs any arm gave a shared case, over "
        "all runs.",
        "On character (second) the arms share no case, so pass^k compares nothing between them.",
    ]


# =============================================================================
# The model, the memo's own depth words, and the old shape
# =============================================================================


def test_the_model_s_line_is_true_when_only_one_arm_falls_short() -> None:
    # Both planned 3: A ran every case three times, B twice, so only B is modelled.
    arms = TwoArms(_THRICE, TWICE, k_a=3, k_b=3, profile=replace(PROFILE, model_pass_hat_k=True))
    read = arms.readings()
    reports = {
        "character": build_code_only_report(
            arms.bundle, assembled_at=_ASSEMBLED_AT, campaign_name="depth", arm_labels=read.arm_labels()
        )
    }

    page = build_campaign_report(arms.campaign, read, reports, None)

    (table,) = [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "model_pass_k"]
    assert [row["arm"].value for row in table.rows] == ["B"]
    assert arms.point(arms.key_a).pass_hat_k is not None
    said = [line for line in _lines(page, "scoreboard") if line.startswith("Each figure here is model-based")]
    assert said and said[0].endswith("Its arm ran no case that deep, so nothing ranks on it.")


def test_a_frozen_campaign_bundle_or_frame_of_the_old_shape_is_refused_naming_why() -> None:
    arms = TwoArms(TWICE, ONCE, k_a=2, k_b=1)
    bundle = campaign_bundle(arms.bundle, arms=arms.arms, readings=arms.readings().readings)
    frozen = {**bundle.to_dict(), "schema_version": 1}
    frame = {**bundle.frame(), "schema_version": 1}

    with pytest.raises(RefusedInput, match="frozen by a build whose readings carried another shape") as refused:
        CampaignBundle.from_dict(frozen)
    assert "this build's shape (2)" in str(refused.value)
    with pytest.raises(RefusedInput, match="frozen by a build whose readings carried another shape"):
        CampaignBundle.from_frame(frame, bundle.aspects)


# =============================================================================
# Floors in the writer's view
# =============================================================================


def test_a_floor_states_its_worst_case_its_margin_and_how_far_it_is_short_after_each_pass_rate_verdict() -> None:
    arms = TwoArms(TWICE, ONCE, k_a=2, k_b=1, floors=[RateDrop(max_points=10.0)])
    read = arms.readings()
    [status] = read.floors
    # The floor reads the pass rate's own interval at 95%, B minus A over the paired shares.
    low, _ = bounded_difference_interval(
        [1.0, 0.5, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0], paired=True, value_range=(0.0, 1.0), confidence=INTERVAL_LEVEL
    ) or (0.0, 0.0)

    assert status.status == "unclear"
    assert (status.margin, status.worst_drop) == (10.0, pytest.approx(-100 * low))
    assert status.short_by == pytest.approx(-100 * low - 10.0)

    view = writer_view(campaign_bundle(arms.bundle, arms=arms.arms, readings=read.readings, floors=read.floors))

    keys = list(view)
    assert keys.index("pass_rate_verdicts") < keys.index("floors")
    assert view["pass_rate_verdicts"] == [{"aspect": "character", "arm": "B", "reading": "unclear", "change": 12.5}]
    [floor] = view["floors"]
    assert (floor["status"], floor["speaks_for"], floor["margin"]) == ("unclear", "the pass rate", 10.0)
    assert floor["short_by"] == pytest.approx(-100 * low - 10.0, rel=1e-3)


def test_a_floor_whose_worst_case_lies_within_its_margin_is_short_by_nothing() -> None:
    # Identical shares: the 95% interval on the paired difference is about ±74 points, inside a 90-point margin.
    arms = TwoArms(TWICE, TWICE, k_a=2, k_b=2, floors=[RateDrop(max_points=90.0)])
    [status] = arms.readings().floors

    assert status.margin == 90.0 and status.worst_drop == pytest.approx(73.96, abs=0.01)
    assert status.short_by is None
