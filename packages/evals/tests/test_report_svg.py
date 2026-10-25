"""The report's marks: the charts the kit draws from their data, and the interval glyph a table cell draws.

- **Every value is text.** An outcome bar is the kit's stacked bars, each arm's census in its tooltip as words and
  in the chart's table; an interval glyph sits in a row whose cells print its figures, so nothing is read off a
  length or a colour alone.
- **Nothing runs and nothing is fetched.** A glyph has no ``<script>``, no ``on*=`` attribute, no URL — not even a
  ``url(#...)`` fill; a chart is JSON the kit draws (``test_report_html_no_script``).
- **A row needs its cases.** An arm with fewer than ``MIN_CASES`` usable cases has no interval drawn, and a contrast
  under the floor is no row at all; the methods say once what was not drawn.
- **Light and dark.** The marks take their colours from the kit's tokens, which the kit defines for both; a glyph
  is ink alone, its verdict the word beside it; each outcome of an outcome bar is a colour of its own, the same on
  every bar, named in the legend.
- **A label is whole somewhere.** The kit cuts a label to its column; the whole label is the tooltip's first line
  and the table's row.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from importlib.resources import files
from typing import Any, get_args


from threetears.evals.analysis import (
    ChartBlock,
    IntervalRow,
    MIN_CASES,
    OutcomeBar,
    OutcomeSegment,
    TableBlock,
    report_html,
)
from threetears.evals.analysis.bundle.schema import FailCause
from threetears.evals.analysis.report.svg import GLYPH_STYLE, interval_glyph_svg
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


def _drawn(block: ChartBlock) -> tuple[str, dict[str, Any]]:
    drawn = block.kit_chart()
    assert drawn is not None
    return drawn


def test_an_outcome_bar_is_the_kit_s_stacked_bars_with_every_count_in_it() -> None:
    kind, data = _drawn(_bar_chart())
    assert kind == "bars" and data["fmt"] == "int" and data["domain"] == [0, 12]
    first, second = data["rows"]
    assert [segment["value"] for segment in first["segments"]] == [6, 2, 3, 1]
    assert first["label"] == "model=a (control)" and first["note"] == "12 trials"
    # The census is in each row's tooltip as words, so no count is read off a length or a colour alone.
    assert first["tip"] == [
        "model=a (control)",
        "12 trials",
        "passed 6",
        "failed: check 2",
        "lost: judge truncated 3",
        "lost: setup fault 1",
    ]
    assert second["segments"] == [{"value": 12, "color": "--s1"}]


def test_every_outcome_count_is_in_the_chart_s_table() -> None:
    headers, rows = _bar_chart().values_table()
    assert headers == ["Arm", "ran", "passed", "failed: check", "lost: judge truncated", "lost: setup fault"]
    assert rows == [["model=a (control)", "12", "6", "2", "3", "1"], ["model=b", "12", "12", "0", "0", "0"]]


def test_interval_rows_carry_no_level_so_they_read_as_their_table_alone() -> None:
    """The kit never draws an interval without its level (GENERATOR.md §5.2), and interval rows name none."""
    assert _interval_chart().kit_chart() is None


def test_two_lost_causes_side_by_side_are_two_colours_and_neither_is_a_series() -> None:
    _, data = _drawn(_bar_chart())
    lost = [segment["color"] for segment in data["rows"][0]["segments"][2:]]
    assert lost == ["--base", "--muted"]


def test_each_failure_cause_has_a_colour_of_its_own_the_same_on_every_bar_and_the_legend_names_each() -> None:
    """A failed check and the judge's fail are two segments in two colours, each the same on every bar whatever order
    the bar lists them in, and the legend names each cause, in the order they first occur."""
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
    _, data = _drawn(ChartBlock(section="scoreboard", chart="outcome_bar", title="Outcomes", bars=bars))
    legend = {item["label"]: item["color"] for item in data["legend"]}
    assert list(legend) == ["passed", "failed: check", "failed: judge"]
    assert len(set(legend.values())) == 3
    for bar, row in zip(bars, data["rows"], strict=True):
        assert [segment["color"] for segment in row["segments"]] == [legend[s.label] for s in bar.segments]


def test_five_failure_causes_are_five_colours() -> None:
    """Five failure causes exist (``FailCause``), so the bar has a colour for each, none of them passed's."""
    causes = ("check", "judge", "model failed", "turn budget", "output cap")
    assert len(causes) == len(get_args(FailCause))
    segments = [OutcomeSegment(state="failed", label=f"failed: {cause}", count=3) for cause in causes]
    block = ChartBlock(
        section="scoreboard", chart="outcome_bar", title="Outcomes", bars=[OutcomeBar(arm="v1", segments=segments)]
    )
    _, data = _drawn(block)
    colours = [segment["color"] for segment in data["rows"][0]["segments"]]
    assert "--s1" not in colours
    assert len(set(colours[:4])) == 4, "the first four causes each a colour of their own"


def _kit_roots() -> list[str]:
    """The kit stylesheet's three token blocks: light, dark by preference, dark by attribute."""
    kit = (files("threetears.evals.analysis.report") / "kit" / "kit.css").read_text(encoding="utf-8")
    return re.findall(r":root[^{]*\{([^}]*)\}", kit)


def test_the_marks_are_coloured_by_kit_tokens_set_for_light_and_dark() -> None:
    """Neither the glyph's stylesheet nor an outcome bar names a colour of its own: every token is one the kit sets in
    its light block and in both dark ones, so a chart follows the page's theme, a host's ``data-theme`` included."""
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b|rgba?\(|hsla?\(", GLYPH_STYLE), "a colour of the glyph's own"
    roots = _kit_roots()
    assert len(roots) == 3
    _, data = _drawn(_bar_chart())
    used = set(re.findall(r"var\((--[\w-]+)\)", GLYPH_STYLE)) | {item["color"] for item in data["legend"]}
    assert {"--s1", "--s2", "--base", "--ink"} <= used
    for variable in used:
        assert all(f"{variable}:" in root for root in roots), variable


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
    assert page.count('data-kind="bars"') == 1 and page.count('class="tbl chart-values"') == 1
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
    assert "svg.viz-glyph .viz-point { fill: var(--ink); }" in GLYPH_STYLE
    assert "svg.viz-glyph .viz-whisker { stroke: var(--ink)" in GLYPH_STYLE


def test_a_glyph_off_zero_draws_no_zero_line_and_no_band() -> None:
    svg = interval_glyph_svg(0.92, 0.85, 0.99, unit="s", scale=(0.8, 1.5))
    assert 'class="viz-zero"' not in svg and 'class="viz-band"' not in svg
    assert "<title>0.92 s, interval [0.85, 0.99] s</title>" in svg


def test_the_legend_names_only_the_states_that_occur() -> None:
    clean = ChartBlock(
        section="scoreboard",
        chart="outcome_bar",
        title="t",
        bars=[OutcomeBar(arm="a", segments=[OutcomeSegment(state="passed", label="passed", count=3)])],
    )
    _, data = _drawn(clean)
    assert [item["label"] for item in data["legend"]] == ["passed"]


def test_a_long_arm_label_is_whole_in_its_tooltip_and_its_table() -> None:
    """The kit fits a label to its column and cuts it there with an ellipsis; the whole label is the tooltip's first
    line and the table's row."""
    label = "the cheaper candidate with a label long enough that a column of its own would have cut it short"
    block = ChartBlock(
        section="scoreboard",
        chart="outcome_bar",
        title="t",
        bars=[OutcomeBar(arm=label, segments=[OutcomeSegment(state="passed", label="passed", count=3)])],
    )
    _, data = _drawn(block)
    assert data["rows"][0]["label"] == label and data["rows"][0]["tip"][0] == label
    assert block.values_table()[1][0][0] == label


async def test_an_arm_under_the_case_floor_is_not_charted_and_the_caveat_says_so_once() -> None:
    ev = Eval(_cases(MIN_CASES - 1), {"never": _never, "classify": _classify}, ["label"])
    comparison = await ev.run(store=":memory:")
    charts = [block for block in comparison.report.blocks if isinstance(block, ChartBlock)]
    assert [block.chart for block in charts] == ["outcome_bar"]
    page = report_html(comparison.report)
    assert page.count('data-kind="') == 1 and "<svg" not in page
    # Under the floor on every arm, the one caveat under the scoreboard says so, and no "not charted" line repeats it.
    assert page.count(f"Under {MIN_CASES} independent cases a rate is shown as a count") == 1
    assert "Not charted" not in page
