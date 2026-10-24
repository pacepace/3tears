"""Depth on every surface: the campaign page's tables and lines, the Methods' pairing line, the writer's view and the
frontier's text.

The page states depth in tables, each with one plain sentence under it; the writer reads the same depths as fields
and the sentence to say once. Code reads none of the memo's prose for them.
"""

from __future__ import annotations


from threetears.evals.analysis import build_campaign_report, build_code_only_report, campaign_report_markdown
from threetears.evals.analysis.campaign_bundle import writer_view
from threetears.evals.analysis.depth import REPEATS_FLOOR, common_depth_sentence, unmeasured_sentence
from threetears.evals.analysis.depth_reading import PAIRING_METHOD
from threetears.evals.analysis.gen_prompt import EVAL_ANALYSIS_GEN_DEFAULT
from threetears.evals.analysis.lens_text import frontier_text
from threetears.evals.analysis.report.model import CampaignReport, DisclosureBlock, TableBlock
from threetears.evals.analysis.style import lint_prose
from packages.evals.tests.style_support import lint_page
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.depth_support import ONCE, TWICE, TwoArms

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
    assert _SAID in _lines(page, "scoreboard")


def test_each_frontier_point_s_depth_sits_beside_it_and_the_common_depth_is_named() -> None:
    page = _page(TwoArms(TWICE, ONCE, k_a=2, k_b=1))

    # A's pass^1 is 5/8 and B's 3/4, over all four cases each.
    assert _table(page, "pass_k_depth") == [
        {"aspect": "character", "arm": "B", "planned": 1, "short": 0, "common": 1, "reading": 75.0, "left_out": 0},
        {"aspect": "character", "arm": "A", "planned": 1, "short": 0, "common": 1, "reading": 62.5, "left_out": 0},
    ]
    assert common_depth_sentence("character", 1, left_out=0) in _lines(page, "scoreboard")


def test_a_depth_some_arm_never_reached_is_said_to_be_unmeasured_for_it() -> None:
    page = _page(TwoArms(TWICE, ONCE, k_a=2, k_b=2))

    assert unmeasured_sentence("character", 2, ["B"]) in _lines(page, "scoreboard")
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
    assert REPEATS_FLOOR in _lines(page, "scoreboard")


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

    [aspect] = writer_view(bundle)["aspects"]

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
    assert set(reading["precision"]) == {"half_width", "own_half_width", "limited_by", "between_case_share", "means"}
    assert set(reading["precision"]["own_half_width"]) == {"A", "B"}
    assert reading["precision"]["means"] == REPEATS_FLOOR


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
