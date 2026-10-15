"""The report's marks: inline SVG with every value written as text, nothing that runs, and nothing drawn thin.

- **Every mark's value is text.** An outcome bar writes each arm's count of every state, an interval row its value,
  interval and cases, and an interval glyph sits in a row whose cells print its figures, so nothing is read off a
  length or a colour alone.
- **Nothing runs and nothing is fetched.** No ``<script>``, no ``on*=`` attribute, no URL — not even a
  ``url(#...)`` fill.
- **A row needs its cases.** An arm with fewer than ``MIN_CASES`` usable cases has no interval drawn, and a contrast
  under the floor is no row at all; the methods say once what was not drawn.
- **Light and dark.** The marks take their colours from the page's CSS variables, which are set for both; a glyph
  is ink alone, its verdict the word beside it.
- **A label is never cut.** An arm's label is the line above its bar, wrapped where it is long.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals.analysis import (
    ChartBlock,
    IntervalRow,
    MIN_CASES,
    OutcomeBar,
    OutcomeSegment,
    TableBlock,
    report_html,
)
from threetears.evals.analysis.report.svg import SVG_STYLE, chart_svg, interval_glyph_svg
from threetears.evals.analysis.style import GLOSSARY
from threetears.evals import Eval


def _bar_chart() -> ChartBlock:
    return ChartBlock(
        section="scoreboard",
        chart="outcome_bar",
        title="How each arm's trials ended",
        bars=[
            OutcomeBar(
                arm="model=a (control)",
                segments=[
                    OutcomeSegment(state="passed", label="passed", count=6),
                    OutcomeSegment(state="failed", label="failed: check", count=2),
                    OutcomeSegment(state="lost", label="lost: judge truncated", count=3),
                    OutcomeSegment(state="lost", label="lost: setup fault", count=1),
                ],
            ),
            OutcomeBar(arm="model=b", segments=[OutcomeSegment(state="passed", label="passed", count=12)]),
        ],
    )


def _interval_chart() -> ChartBlock:
    return ChartBlock(
        section="contrasts",
        chart="interval_rows",
        title="Accuracy: each arm minus the control",
        rows=[IntervalRow(label="model=b − model=a", value=0.25, low=0.05, high=0.45, n_cases=8, verdict="improved")],
        unit="",
        zero_line=True,
        margin=0.1,
    )


@pytest.mark.parametrize("block", [_bar_chart(), _interval_chart()], ids=["outcome_bar", "interval_rows"])
def test_nothing_in_a_chart_runs_or_is_fetched(block: ChartBlock) -> None:
    svg = chart_svg(block, chart_id="c1")
    assert "<script" not in svg
    assert not re.search(r"\son[a-z]+\s*=", svg)
    assert "url(" not in svg and "http" not in svg and "href" not in svg


def test_every_outcome_count_is_written_as_text() -> None:
    svg = chart_svg(_bar_chart(), chart_id="c1")
    texts = re.findall(r"<text[^>]*>(?:<title>.*?</title>)?([^<]*)</text>", svg)
    written = " ".join(texts)
    for count in ("passed 6", "failed: check 2", "lost: judge truncated 3", "lost: setup fault 1", "passed 12"):
        assert count in written, count
    assert "12 trials" in written
    # Each mark says what it is, in its title too.
    assert "<title>model=a (control): lost: judge truncated, 3 of the 12 trials it ran</title>" in svg


def test_an_interval_row_writes_its_value_interval_and_cases() -> None:
    svg = chart_svg(_interval_chart(), chart_id="c1")
    assert "0.25 [0.05, 0.45], n=8 cases, improved" in svg
    assert 'class="viz-zero"' in svg and 'class="viz-band"' in svg


def test_two_lost_causes_side_by_side_are_two_marks() -> None:
    svg = chart_svg(_bar_chart(), chart_id="c1")
    assert svg.count('class="viz-lost"') + svg.count('class="viz-lost viz-alt"') == 3  # two segments and the key
    assert 'class="viz-lost viz-alt"' in svg


def test_two_failure_causes_are_two_inks_of_one_hue_and_the_key_names_each() -> None:
    """A failed check and the judge's fail are two segments in the one failed hue, told apart in ink alone, each the
    same on every bar, and the key names each cause, so no third category reads as a state of its own."""
    bars = [
        OutcomeBar(
            arm="v1",
            segments=[
                OutcomeSegment(state="passed", label="passed", count=29),
                OutcomeSegment(state="failed", label="failed: check", count=26),
                OutcomeSegment(state="failed", label="failed: judge", count=5),
            ],
        ),
        OutcomeBar(
            arm="v2",
            segments=[
                OutcomeSegment(state="passed", label="passed", count=14),
                OutcomeSegment(state="failed", label="failed: judge", count=35),
                OutcomeSegment(state="failed", label="failed: check", count=11),
            ],
        ),
    ]
    svg = chart_svg(ChartBlock(section="scoreboard", chart="outcome_bar", title="Outcomes", bars=bars), chart_id="c1")
    inks = re.findall(r'<rect class="(viz-failed[^"]*)"[^>]*><title>(v\d): (failed: \w+),', svg)
    assert sorted(set(inks)) == [
        ("viz-failed", "v1", "failed: check"),
        ("viz-failed", "v2", "failed: check"),
        ("viz-failed viz-alt", "v1", "failed: judge"),
        ("viz-failed viz-alt", "v2", "failed: judge"),
    ]
    key = svg[: svg.index("v1</text>")]
    assert re.findall(r">(passed|failed: \w+|failed|lost)</text>", key) == ["passed", "failed: check", "failed: judge"]
    assert "--viz-failed-" not in SVG_STYLE and SVG_STYLE.count("stroke: var(--viz-failed)") == 2, "one hue, ink only"


def test_the_marks_are_coloured_by_css_variables_set_for_light_and_dark() -> None:
    assert "prefers-color-scheme: dark" in SVG_STYLE and ':root[data-theme="dark"]' in SVG_STYLE
    for variable in ("--viz-passed", "--viz-failed", "--viz-lost", "--viz-ink"):
        assert SVG_STYLE.count(f"{variable}:") == 3, variable
    assert "fill: var(--viz-passed)" in SVG_STYLE
    svg = chart_svg(_bar_chart(), chart_id="c1")
    assert "#" not in re.sub(r'id="[^"]*"|aria-labelledby="[^"]*"', "", svg).replace("&#x27;", ""), "no colour inline"


def test_a_row_under_the_case_floor_cannot_be_built_into_a_page_but_a_chart_names_its_values() -> None:
    headers, rows = _interval_chart().values_table()
    assert headers == ["Row", "Value", "Interval", "Cases", "Verdict"]
    assert rows == [["model=b − model=a", "0.25", "[0.05, 0.45]", "8", "improved"]]


async def _classify(case: Mapping[str, Any]) -> str:
    return "billing" if "refund" in case["text"] else "bug"


async def _never(case: Mapping[str, Any]) -> str:
    return "bug"


def _cases(n: int) -> list[dict[str, str]]:
    return [{"text": f"refund {index}", "label": "billing"} for index in range(n)]


async def test_a_campaign_with_enough_cases_draws_its_contrast_s_interval_in_the_row() -> None:
    ev = Eval(_cases(MIN_CASES + 3), {"never": _never, "classify": _classify}, ["label"])
    comparison = await ev.run(store=":memory:")
    charts = [block for block in comparison.report.blocks if isinstance(block, ChartBlock)]
    assert [block.chart for block in charts] == ["outcome_bar"], "the outcome bar is the page's one chart"
    (table,) = [b for b in comparison.report.blocks if isinstance(b, TableBlock) and b.name == "comparisons"]
    assert table.rows and all(row["change"].low is not None and row["change"].zero_line for row in table.rows)
    page = report_html(comparison.report)
    assert page.count('<svg class="viz"') == 1 and 'class="visually-hidden"' in page
    assert page.count('<svg class="viz-glyph"') == len(table.rows)
    # The scale's ends are written once under the table, not in every row.
    assert page.count('<p class="axis">Change drawn from ') == 1


def test_a_glyph_is_ink_on_one_scale_with_its_numbers_in_its_title() -> None:
    svg = interval_glyph_svg(-1.1, -4.2, 2.0, unit="pts", scale=(-7.0, 36.7), margin=5.0)
    assert svg.startswith('<svg class="viz-glyph" viewBox="0 0 120 14" role="img">')
    assert "<title>-1.1 pts, interval [-4.2, 2] pts, margin ±5 pts</title>" in svg
    for mark in ("viz-track", "viz-zero", "viz-band", "viz-whisker", "viz-point"):
        assert f'class="{mark}"' in svg, mark
    assert "<script" not in svg and "url(" not in svg and "#" not in svg and "href" not in svg
    # Ink alone: the point and the whisker take the page's ink, never a series colour.
    assert "svg.viz-glyph .viz-point { fill: var(--viz-ink); }" in SVG_STYLE
    assert "svg.viz-glyph .viz-whisker { stroke: var(--viz-ink)" in SVG_STYLE


def test_a_glyph_off_zero_draws_no_zero_line_and_no_band() -> None:
    svg = interval_glyph_svg(0.92, 0.85, 0.99, unit="s", scale=(0.8, 1.5))
    assert 'class="viz-zero"' not in svg and 'class="viz-band"' not in svg
    assert "<title>0.92 s, interval [0.85, 0.99] s</title>" in svg


def test_the_key_names_only_the_states_that_occur_and_defines_lost_on_hover() -> None:
    svg = chart_svg(_bar_chart(), chart_id="c1")
    assert f"<title>{GLOSSARY['lost']}</title>" in svg and ">lost</text>" in svg
    assert "lost to the test setup" not in svg
    clean = ChartBlock(
        section="scoreboard",
        chart="outcome_bar",
        title="t",
        bars=[OutcomeBar(arm="a", segments=[OutcomeSegment(state="passed", label="passed", count=3)])],
    )
    svg = chart_svg(clean, chart_id="c2")
    assert ">passed</text>" in svg and ">failed</text>" not in svg and ">lost</text>" not in svg


def test_a_long_arm_label_is_wrapped_above_its_bar_never_cut() -> None:
    label = "the cheaper candidate with a label long enough that a column of its own would have cut it short"
    block = ChartBlock(
        section="scoreboard",
        chart="outcome_bar",
        title="t",
        bars=[OutcomeBar(arm=label, segments=[OutcomeSegment(state="passed", label="passed", count=3)])],
    )
    svg = chart_svg(block, chart_id="c1")
    texts = re.findall(r"<text[^>]*>(?:<title>.*?</title>)?([^<]*)</text>", svg)
    assert "…" not in svg and " ".join(texts).count(label) == 1
    # The bar itself, not the key's swatch: it starts at the chart's left edge, under the label's line.
    (bar_y,) = re.findall(r'<rect class="viz-passed" x="8\.0" y="(\d+)" width="[\d.]+" height="16"', svg)
    (label_y,) = re.findall(r'<text x="8.0" y="([\d.]+)">the cheaper', svg)
    assert float(label_y) < float(bar_y), "the label is the line above the bar"


async def test_an_arm_under_the_case_floor_is_not_charted_and_the_caveat_says_so_once() -> None:
    ev = Eval(_cases(MIN_CASES - 1), {"never": _never, "classify": _classify}, ["label"])
    comparison = await ev.run(store=":memory:")
    charts = [block for block in comparison.report.blocks if isinstance(block, ChartBlock)]
    assert [block.chart for block in charts] == ["outcome_bar"]
    page = report_html(comparison.report)
    assert page.count("<svg") == 1
    # Under the floor on every arm, the one caveat under the scoreboard says so, and no "not charted" line repeats it.
    assert page.count(f"Under {MIN_CASES} independent cases a rate is shown as a count") == 1
    assert "Not charted" not in page
