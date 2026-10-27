"""Depth on every surface: the campaign page's tables and lines, the Methods' pairing line, the writer's view and the
frontier's text.

The page states depth in tables, each with one plain sentence under it, or in one sentence where that says all the
table would; the writer reads the same depths as fields and the sentence to say once. Code reads none of the memo's
prose for them.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from threetears.evals.analysis import (
    build_campaign_report,
    build_code_only_report,
    campaign_report_html,
    campaign_report_markdown,
)
from threetears.evals.analysis.campaign_bundle import writer_view
from threetears.evals.analysis.depth import (
    COMMON_DEPTH_ONE_EVERYWHERE,
    REPEATS_FLOOR,
    Depth,
    common_depth_sentence,
    repeats_sentence,
    unmeasured_sentence,
)
from threetears.evals.analysis.depth_reading import PAIRING_METHOD
from threetears.evals.analysis.gen_prompt import EVAL_ANALYSIS_GEN_DEFAULT
from threetears.evals.analysis.lens_text import frontier_text
from threetears.evals.analysis.report.campaign.depth_blocks import depth_provenance_blocks
from threetears.evals.analysis.report.model import CampaignReport, DisclosureBlock, TableBlock
from threetears.evals.analysis.style import lint_prose
from packages.evals.tests.style_support import lint_page
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.depth_support import ONCE, TWICE, TwoArms
from packages.evals.tests.report_shapes_support import Shape, s1_change, s2_choice, s3_measurement, s4_change

#: When each aspect's own page was assembled.
_ASSEMBLED_AT = "2026-10-24T00:00:00+00:00"

#: What the page says under the Depth table: A ran each case twice, B once.
_SAID = "A was run twice on character, B once; that tightens A's own estimate, not the comparison much."


def _page(arms: TwoArms) -> CampaignReport:
    read = arms.readings()
    reports = {
        "character": build_code_only_report(
            arms.bundle, assembled_at=_ASSEMBLED_AT, campaign_name="depth", arm_labels=read.arm_labels()
        )
    }
    return build_campaign_report(arms.campaign, read, reports, None)


def _table(page: CampaignReport, name: str) -> list[dict[str, object]]:
    (table,) = [block for block in page.blocks if isinstance(block, TableBlock) and block.name == name]
    return [{key: cell.value for key, cell in row.items()} for row in table.rows]


def _lines(page: CampaignReport, section: str) -> list[str]:
    return [block.text for block in page.blocks if isinstance(block, DisclosureBlock) and block.section == section]


def test_the_page_states_each_arm_s_depth_in_a_table_and_says_once_that_they_differ() -> None:
    page = _page(TwoArms(TWICE, ONCE, k_a=2, k_b=1))

    assert _table(page, "depth") == [
        {"aspect": "character", "arm": "A", "cases": 4, "trials": 8, "fewest": 2, "median": 2.0, "most": 2},
        {"aspect": "character", "arm": "B", "cases": 4, "trials": 4, "fewest": 1, "median": 1.0, "most": 1},
    ]
    assert _SAID in _lines(page, "scorecard")


def test_each_frontier_point_s_depth_sits_beside_it_and_the_common_depth_is_named() -> None:
    page = _page(TwoArms(TWICE, ONCE, k_a=2, k_b=2))

    # Planned 2: A's pass^1 at the common depth is 5/8 and B's 3/4, over all four cases each; B ran none twice.
    assert _table(page, "pass_k_depth") == [
        {"aspect": "character", "arm": "B", "planned": 2, "short": 4, "common": 1, "reading": 75.0, "left_out": 0},
        {"aspect": "character", "arm": "A", "planned": 2, "short": 0, "common": 1, "reading": 62.5, "left_out": 0},
    ]
    assert common_depth_sentence("character", 1, left_out=0) in _lines(page, "scorecard")


def test_pass_k_planned_and_compared_at_depth_1_is_one_sentence_and_no_table() -> None:
    page = _page(TwoArms(TWICE, ONCE, k_a=2, k_b=1))

    # Planned 1 and compared at 1 with no case left out: the table would print each arm's pass rate again.
    assert not [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "pass_k_depth"]
    assert common_depth_sentence("character", 1, left_out=0) in _lines(page, "scorecard")


def test_a_depth_some_arm_never_reached_is_said_to_be_unmeasured_for_it() -> None:
    page = _page(TwoArms(TWICE, ONCE, k_a=2, k_b=2))

    assert unmeasured_sentence("character", 2, ["B"]) in _lines(page, "scorecard")
    rows = {row["arm"]: row for row in _table(page, "pass_k_depth")}
    assert (rows["B"]["planned"], rows["B"]["short"], rows["B"]["common"]) == (2, 4, 1)


def test_the_precision_table_names_the_limiting_side_and_the_between_case_share() -> None:
    arms = TwoArms(TWICE, ONCE, k_a=2, k_b=1)
    page = _page(arms)
    [row] = arms.readings().readings
    assert row.precision is not None

    [shown] = _table(page, "precision")

    assert shown["arms"] == "A and B"
    assert shown["half_width"] == round(row.precision.half_width, 1)
    assert (shown["first"], shown["second"]) == (
        round(row.precision.half_widths[arms.key_a], 1),
        round(row.precision.half_widths[arms.key_b], 1),
    )
    assert shown["limited_by"] == {arms.key_a: "A", arms.key_b: "B"}[row.precision.limited_by or ""]
    # 10/19 of the difference's variance lies between cases.
    assert shown["between"] == 53
    # What extra repeats cannot lower is said once, in Design & provenance, not under the table.
    assert REPEATS_FLOOR not in _lines(page, "scorecard")
    assert _lines(page, "provenance").count(REPEATS_FLOOR) == 1


def test_the_methods_say_how_a_pass_rate_comparison_weighs_its_cases() -> None:
    assert PAIRING_METHOD in _lines(_page(TwoArms(TWICE, ONCE, k_a=2, k_b=1)), "methods")


def test_every_new_line_is_one_plain_sentence_and_the_page_passes_the_style_rules() -> None:
    page = _page(TwoArms(TWICE, ONCE, k_a=2, k_b=2))

    for sentence in (_SAID, PAIRING_METHOD, REPEATS_FLOOR, common_depth_sentence("character", 2, left_out=3)):
        assert lint_prose(sentence) == [], sentence
    assert lint_page(campaign_report_markdown(page), names=["A", "B", "▸ A"]) == []


def test_the_writer_reads_each_arm_s_depth_the_common_depth_and_what_limits_each_interval() -> None:
    arms = TwoArms(TWICE, ONCE, k_a=2, k_b=2)
    bundle = campaign_bundle(arms.bundle, arms=arms.arms, readings=arms.readings().readings)

    view = writer_view(bundle)
    [aspect] = view["aspects"]

    assert aspect["depth"] == {
        "arms": {
            "A": {"cases": 4, "trials": 8, "repeats_per_case": [2, 2.0, 2]},
            "B": {"cases": 4, "trials": 4, "repeats_per_case": [1, 1.0, 1]},
        },
        "differ": True,
        "said": _SAID,
        "pass_k": [
            {
                "where": "character",
                "k": 2,
                "common_depth": 1,
                "shares_no_case": False,
                "unmeasured_for": ["B"],
                "said": [common_depth_sentence("character", 1, left_out=0), unmeasured_sentence("character", 2, ["B"])],
            }
        ],
    }
    [reading] = aspect["readings"]
    assert set(reading["precision"]) == {"half_width", "own_half_width", "limited_by", "between_case_share"}
    assert set(reading["precision"]["own_half_width"]) == {"A", "B"}
    # What a between-case share means is defined once, for the whole view.
    assert view["precision_means"] == REPEATS_FLOOR


def test_the_writer_prompt_states_the_depth_rule() -> None:
    assert "`depth.said`" in EVAL_ANALYSIS_GEN_DEFAULT
    assert (
        "never compare arms on pass^k above a `depth.pass_k` `common_depth`, or at all where null (`shares_no_case`)"
        in EVAL_ANALYSIS_GEN_DEFAULT
    )


def test_the_frontier_text_states_each_contestant_s_depth_and_the_common_depth() -> None:
    text = frontier_text(TwoArms(TWICE, ONCE, k_a=2, k_b=2).bundle.frontier)

    assert common_depth_sentence("subject Test subject", 1, left_out=0) in text
    assert "ran each case twice over 4 case(s); pass^1 0.625 over 4 case(s) at the common depth" in text
    assert "ran each case once over 4 case(s), 4 short of k=2; pass^1 0.75 over 4 case(s) at the common depth" in text
    assert "On subject Test subject, pass^2 is unmeasured for haiku" in text


# =============================================================================
# A table that says no more than a sentence is that sentence (S1-S4)
# =============================================================================


def _tables(page: CampaignReport) -> dict[str, TableBlock]:
    return {block.name: block for block in page.blocks if isinstance(block, TableBlock)}


def test_a_campaign_run_alike_but_for_one_arm_says_its_depths_in_one_sentence_naming_that_arm() -> None:
    page = s1_change().page()

    # Every arm ran each case once on every aspect but v3 on tone, twice: one sentence, naming v3 on tone.
    assert "depth" not in _tables(page)
    said = (
        "Every arm ran each case once on every aspect but v3 twice on tone; extra repeats tighten an arm's own "
        "estimate, not the comparison much."
    )
    assert _lines(page, "scorecard").count(said) == 1
    # The aspect's own sentence is not said beside it.
    assert "v3 was run twice on tone, the others once" not in campaign_report_markdown(page)


def test_pass_k_at_a_common_depth_of_1_on_every_aspect_is_one_sentence_for_the_campaign() -> None:
    page = s1_change().page()

    assert "pass_k_depth" not in _tables(page)
    assert COMMON_DEPTH_ONE_EVERYWHERE in _lines(page, "scorecard")
    for aspect in ("billing", "tone", "refunds"):
        assert common_depth_sentence(aspect, 1, left_out=0) not in _lines(page, "scorecard")


def test_a_campaign_run_three_times_per_case_keeps_its_pass_k_and_precision_tables() -> None:
    page = s2_choice().page()
    tables = _tables(page)

    # Every arm ran each case three times: depth is one sentence, but pass^3 and the variance split are read.
    assert "Every arm ran each case 3 times on answers." in _lines(page, "scorecard")
    assert "depth" not in tables
    assert tables["pass_k_depth"].section == "scorecard" and len(tables["pass_k_depth"].rows) == 3
    assert common_depth_sentence("answers", 3, left_out=0) in _lines(page, "scorecard")
    assert tables["precision"].section == "scorecard"
    assert [column.key for column in tables["precision"].columns][-1] == "between"


def test_one_measured_arm_s_depth_is_one_sentence() -> None:
    page = s3_measurement().page()

    assert "depth" not in _tables(page)
    assert "solo ran each case twice on every aspect." in _lines(page, "scorecard")


def test_precision_read_with_every_arm_run_once_sits_under_the_methods_without_its_empty_column() -> None:
    page = s4_change().page()
    precision = _tables(page)["precision"]

    # No case repeated, so no comparison splits its noise: the between-case column is empty on every row.
    assert precision.section == "methods"
    assert "between" not in [column.key for column in precision.columns]
    assert all(set(row) == {column.key for column in precision.columns} for row in precision.rows)
    assert "Between cases" not in campaign_report_markdown(page)
    # Where a case repeated, the precision stays in the Scorecard with its between-case share.
    assert _tables(s1_change().page())["precision"].section == "scorecard"


@pytest.mark.parametrize("build", [s1_change, s2_choice], ids=["s1", "s2"])
def test_what_extra_repeats_cannot_lower_is_said_once_in_design_and_provenance(build: Callable[[], Shape]) -> None:
    page = build().page()

    assert _lines(page, "provenance").count(REPEATS_FLOOR) == 1
    assert campaign_report_markdown(page).count(REPEATS_FLOOR) == 1
    assert campaign_report_html(page).count(REPEATS_FLOOR) == 1


@pytest.mark.parametrize("build", [s3_measurement, s4_change], ids=["s3", "s4"])
def test_a_campaign_that_repeated_no_compared_case_says_nothing_of_repeats(build: Callable[[], Shape]) -> None:
    # S3 compares nothing; S4 ran every case once, so no comparison reads a between-case share.
    page = build().page()

    assert depth_provenance_blocks(page.readings) == []
    assert REPEATS_FLOOR not in campaign_report_markdown(page)


def test_the_writer_view_still_says_each_aspect_s_depth_where_the_page_says_it_once() -> None:
    shape = s1_change()
    bundle = campaign_bundle(*shape.bundles.values(), arms=shape.arms, readings=shape.readings.readings)

    said = {aspect["aspect"]: aspect["depth"].get("said") for aspect in writer_view(bundle)["aspects"]}

    assert said == {
        "billing": None,
        "tone": "v3 was run twice on tone, the others once; that tightens v3's own estimate, not the comparison much.",
        "refunds": None,
    }


def test_the_one_sentence_names_only_a_few_and_never_drops_a_figure() -> None:
    once = Depth(cases=20, trials=20, repeats_min=1, repeats_median=1.0, repeats_max=1)
    twice = Depth(cases=20, trials=40, repeats_min=2, repeats_median=2.0, repeats_max=2)
    ragged = Depth(cases=20, trials=30, repeats_min=1, repeats_median=1.5, repeats_max=2)
    short = Depth(cases=10, trials=10, repeats_min=1, repeats_median=1.0, repeats_max=1)
    alike = {"p": once, "q": once, "r": once}

    assert repeats_sentence({"x": alike, "y": {**alike, "r": twice}}) == (
        "Every arm ran each case once on every aspect but r twice on y; extra repeats tighten an arm's own "
        "estimate, not the comparison much."
    )
    assert repeats_sentence({"x": alike}) == "Every arm ran each case once on x."
    assert repeats_sentence({"x": {"p": twice}, "y": {"p": twice}}) == "p ran each case twice on every aspect."
    assert repeats_sentence({"x": {"p": twice}, "y": {"p": twice}, "z": {"p": once}}) == (
        "p ran each case twice on every aspect but once on z."
    )
    # A depth that varies within an arm, or arms of one aspect over different cases, keeps the table.
    assert repeats_sentence({"x": {**alike, "r": ragged}}) is None
    assert repeats_sentence({"x": {**alike, "r": short}}) is None
    # As many exceptions as alike pairs, or more than a few, keep it too.
    assert repeats_sentence({"x": {"p": once, "q": twice}}) is None
    many = {aspect: {"p": once, "q": twice} for aspect in "abcd"} | {"e": alike, "f": alike, "g": alike}
    assert repeats_sentence(many) is None
    for sentence in (
        COMMON_DEPTH_ONE_EVERYWHERE,
        repeats_sentence({"x": alike, "y": {**alike, "r": twice}}) or "",
        repeats_sentence({"x": {"p": twice}, "y": {"p": twice}, "z": {"p": once}}) or "",
    ):
        assert lint_prose(sentence) == [], sentence
