"""The report's marks: the charts the kit draws from their data, and the interval glyph a table cell draws.

- **Every value is text.** An outcome bar is the kit's stacked bars, each arm's census in its tooltip as words and
  in the chart's table; an interval glyph sits in a row whose cells print its figures, so nothing is read off a
  length or a colour alone.
- **Nothing runs and nothing is fetched.** A glyph has no ``<script>``, no ``on*=`` attribute, no URL — not even a
  ``url(#...)`` fill; a chart is JSON the kit draws (``test_report_html_no_script``).
- **A row needs its cases.** An arm with fewer than ``MIN_CASES`` usable cases has no interval drawn, and a contrast
  under the floor is no row at all; the methods say once what was not drawn.
- **Light and dark.** The marks take their colours from the kit's tokens, which the kit defines for both; a glyph
  is ink alone, its verdict the word beside it.
- **An outcome is never read off a colour alone.** Each outcome of an outcome bar is a look of its own, the same on
  every bar and named in the legend: passed the kit's good, solid; every failure cause the kit's bad under a texture
  of its own, so the five causes are five looks and none reads as a pass without colour; a loss a neutral. No outcome
  takes a series colour, since those name arms.
- **An arm is told apart on every chart.** Each arm carries its letter tag (P for the control, then A, B, C…), its
  colour by that tag (the control ink-gray, the others the series in order) and its short name, which the kit draws
  where the whole label does not fit; the whole label is the tooltip's first line and the table's row.
- **A cost axis reads.** A cost/quality chart draws cost per 1,000 replies, with the digits its ticks need, so a
  fraction of a cent per reply never prints as $0.000.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from importlib.resources import files
from typing import Any, get_args

import pytest

from threetears.evals.analysis import (
    ChartBlock,
    CostQualityPoint,
    IntervalRow,
    MIN_CASES,
    OutcomeBar,
    OutcomeSegment,
    TableBlock,
    report_html,
)
from threetears.evals.analysis.bundle.schema import FailCause
from threetears.evals.analysis.quantities import MONEY
from threetears.evals.analysis.report.svg import GLYPH_STYLE, interval_glyph_svg
from threetears.evals.analysis.report.words import arm_tags
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


#: Two arms whose whole labels share a long prefix, as the package names arms by their levels.
_CONTROL_ARM = "model=claude-haiku-5-5, reasoning=low (control)"
_CANDIDATE_ARMS = ("model=claude-haiku-5-5, reasoning=high", "model=claude-haiku-5-5, reasoning=medium")


def _tagged_bars() -> ChartBlock:
    """An outcome bar over a control and two candidates, each with its tag and short name, as the builder sets them."""
    passed = [OutcomeSegment(state="passed", label="passed", count=4)]
    return ChartBlock(
        section="scoreboard",
        chart="outcome_bar",
        title="Outcomes",
        bars=[
            OutcomeBar(arm=_CONTROL_ARM, segments=passed, tag="P", short_name="control"),
            OutcomeBar(arm=_CANDIDATE_ARMS[0], segments=passed, tag="A", short_name="reasoning high"),
            OutcomeBar(arm=_CANDIDATE_ARMS[1], segments=passed, tag="B", short_name="reasoning medium"),
        ],
    )


def _cost_chart(*costs: float, tags: bool = True) -> ChartBlock:
    """A cost/quality chart over arms costing ``costs`` per reply, in dollars, the first the control."""
    return ChartBlock(
        section="scorecard",
        chart="cost_quality",
        title="Cost and pass rate",
        unit=MONEY,
        points=[
            CostQualityPoint(
                arm=f"arm {index}",
                tag=("P" if index == 0 else chr(ord("A") + index - 1)) if tags else None,
                short_name=f"arm {index}",
                cost=cost,
                cost_low=cost * 0.9,
                cost_high=cost * 1.1,
                pass_rate=60 + index,
                pass_rate_low=50,
                pass_rate_high=80,
            )
            for index, cost in enumerate(costs)
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
    assert second["segments"] == [{"value": 12, "color": "--good"}]


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


def _look(item: Mapping[str, Any]) -> tuple[str, str | None]:
    """A segment's or a legend swatch's look: its colour token and its texture (None for solid)."""
    return item["color"], item.get("pattern")


def test_each_failure_cause_has_a_look_of_its_own_the_same_on_every_bar_and_the_legend_names_each() -> None:
    """A failed check and the judge's fail are two segments in two looks, each the same on every bar whatever order
    the bar lists them in, and the legend names each cause, in the order they first occur, in its look."""
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
    legend = {item["label"]: _look(item) for item in data["legend"]}
    assert list(legend) == ["passed", "failed: check", "failed: judge"]
    assert len(set(legend.values())) == 3
    for bar, row in zip(bars, data["rows"], strict=True):
        assert [_look(segment) for segment in row["segments"]] == [legend[s.label] for s in bar.segments]


#: The kit's series colours, which name arms and never an outcome (GENERATOR.md §5.2).
_SERIES = {"--s1", "--s2", "--s3", "--s4", "--s5"}


def test_five_failure_causes_are_five_looks() -> None:
    """Five failure causes exist (``FailCause``), so the bar has a look for each, all five distinct: each the kit's
    bad, never green and never a series colour, each under a texture, so no failure reads as a pass, solid, even
    without colour; none of them passed's or a loss's."""
    causes = ("check", "judge", "model failed", "turn budget", "output cap")
    assert len(causes) == len(get_args(FailCause))
    segments = [
        OutcomeSegment(state="passed", label="passed", count=3),
        *(OutcomeSegment(state="failed", label=f"failed: {cause}", count=3) for cause in causes),
        OutcomeSegment(state="lost", label="lost: judge truncated", count=3),
        OutcomeSegment(state="lost", label="lost: setup fault", count=3),
    ]
    block = ChartBlock(
        section="scoreboard", chart="outcome_bar", title="Outcomes", bars=[OutcomeBar(arm="v1", segments=segments)]
    )
    _, data = _drawn(block)
    looks = [_look(segment) for segment in data["rows"][0]["segments"]]
    passed, failed, lost = looks[0], looks[1:6], looks[6:]
    assert len(set(failed)) == len(causes), "each failure cause a look of its own"
    assert all(colour == "--bad" and pattern is not None for colour, pattern in failed)
    assert passed == ("--good", None)
    assert all(colour in {"--base", "--muted"} for colour, _ in lost) and len(set(lost)) == 2
    assert len(set(looks)) == len(looks), "passed, every failure and every loss, each a look of its own"
    assert not _SERIES & {colour for colour, _ in looks}
    assert [_look(item) for item in data["legend"]] == looks, "the legend shows each cause in its look"


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
    used |= {row["color"] for row in _drawn(_tagged_bars())[1]["rows"]}
    assert {"--good", "--bad", "--base", "--ink", "--ink-2", "--s1", "--s2"} <= used
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


@pytest.mark.parametrize("toward", [math.inf, -math.inf], ids=["up", "down"])
def test_a_glyph_on_a_twentieth_of_a_px_draws_the_same_at_its_one_ulp_neighbours(toward: float) -> None:
    """On a 4..116 scale a figure is its own x, so 44.45, 20.05 and 50.15 fall exactly between two tenths of a px:
    a computed figure's last bit (which differs between platforms) would pick the tenth. Each coordinate is rounded to
    twelve figures before it is written, so the neighbours draw the same glyph."""
    figures = (44.45, 20.05, 50.15)
    drawn = interval_glyph_svg(*figures, unit="", scale=(4.0, 116.0))
    moved = interval_glyph_svg(*(math.nextafter(figure, toward) for figure in figures), unit="", scale=(4.0, 116.0))

    assert moved == drawn
    assert 'cx="44.5"' in drawn and 'x1="20.1"' in drawn and 'x2="50.1"' in drawn, drawn


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


def test_each_arm_of_an_outcome_bar_carries_its_tag_its_colour_and_its_short_name() -> None:
    """GENERATOR.md §5.2 and §6.1: the control ink-gray and tagged P, the candidates the series in order and tagged A,
    B; each row's short name is what the kit draws where the whole label does not fit, and the whole label stays the
    tooltip's first line and the table's row."""
    block = _tagged_bars()
    _, data = _drawn(block)
    rows = data["rows"]
    assert [(row["tag"], row["color"]) for row in rows] == [("P", "--ink-2"), ("A", "--s1"), ("B", "--s2")]
    assert [row["short_label"] for row in rows] == ["control", "reasoning high", "reasoning medium"]
    assert [row["label"] for row in rows] == [row["tip"][0] for row in rows] == [_CONTROL_ARM, *_CANDIDATE_ARMS]
    assert [row[0] for row in block.values_table()[1]] == [_CONTROL_ARM, *_CANDIDATE_ARMS]


def test_no_candidate_is_tagged_p_however_many_there_are() -> None:
    """GENERATOR.md §6.1: P is the control's. Thirty candidates are lettered A to O, then Q to Z, then AA on, so the
    sixteenth is Q, never P; each candidate's row on the outcome bar is a series colour, never the control's
    ink-gray."""
    keys = ["control", *(f"candidate {index}" for index in range(30))]
    tags = arm_tags(keys, "control")
    assert tags["control"] == "P"
    lettered = [tags[key] for key in keys[1:]]
    assert "P" not in lettered and len(set(lettered)) == len(lettered)
    assert lettered[:17] == [*"ABCDEFGHIJKLMNOQR"] and lettered[25:] == ["AA", "AB", "AC", "AD", "AE"]
    assert arm_tags(keys[1:], None)["candidate 15"] == "Q"
    passed = [OutcomeSegment(state="passed", label="passed", count=4)]
    block = ChartBlock(
        section="scoreboard",
        chart="outcome_bar",
        title="Outcomes",
        bars=[OutcomeBar(arm=key, segments=passed, tag=tags[key]) for key in keys],
    )
    _, data = _drawn(block)
    colours = [row["color"] for row in data["rows"]]
    assert colours[0] == "--ink-2" and "--ink-2" not in colours[1:]


def test_each_arm_of_a_cost_chart_is_its_colour_and_its_mark_and_the_legend_names_its_tag() -> None:
    """One colour per arm for the page, plus a mark, so colour is never the only cue (GENERATOR.md §5.2): the control
    ink-gray and square, the candidates the series and shapes in order; each point labelled by its short name, by its
    tag alone on a phone, and the legend names each tag's arm."""
    _, data = _drawn(_cost_chart(0.00042, 0.00018, 0.0011))
    points = data["points"]
    assert [(p["color"], p["shape"]) for p in points] == [
        ("--ink-2", "square"),
        ("--s1", "circle"),
        ("--s2", "triangle"),
    ]
    assert [p["label"] for p in points] == ["arm 0", "arm 1", "arm 2"]
    assert [p["narrow_label"] for p in points] == ["P", "A", "B"]
    assert [(item["label"], item["color"], item["mark"]) for item in data["legend"]] == [
        ("P · arm 0", "--ink-2", "square"),
        ("A · arm 1", "--s1", "circle"),
        ("B · arm 2", "--s2", "triangle"),
    ]


def test_a_cost_chart_stored_before_arms_carried_tags_draws_as_it_did() -> None:
    _, data = _drawn(_cost_chart(0.00042, 0.00018, tags=False))
    assert "legend" not in data and all("color" not in p and "shape" not in p for p in data["points"])


def _nice_ticks(top: float, n: int = 5) -> list[float]:
    """The kit's nice ticks from zero to ``top`` (kit.js ``ticksRaw``): a step of 1, 2, 5 or 10 times a power of ten."""
    step = 10 ** math.floor(math.log10(top / n))
    err = top / n / step
    step *= 10 if err >= 7.5 else 5 if err >= 3.5 else 2 if err >= 1.5 else 1
    return [round(index * step, 10) for index in range(int(top / step + 1e-9) + 1)]


def _usd(value: float, spec: str) -> str:
    """A value as the kit's ``usd:N`` prints it."""
    return f"${value:,.{int(spec.split(':')[1])}f}"


@pytest.mark.parametrize("costs", [(0.00042, 0.00018), (0.002, 0.0002), (0.0000042, 0.0000018)])
def test_a_cost_axis_at_a_real_reply_cost_prints_distinct_ticks_none_reading_as_zero(costs: tuple[float, ...]) -> None:
    """A reply costs a fraction of a cent, so the chart draws cost per 1,000 replies (per 1,000,000 where a cost is
    under a cent even per 1,000, the one owner's ruler) and gives the axis the digits its ticks need: every tick past
    the origin prints a figure of its own, none of them zero."""
    _, data = _drawn(_cost_chart(*costs))
    count = 1000 if min(costs) * 1000 >= 0.01 else 1_000_000
    assert [p["x"] for p in data["points"]] == pytest.approx([cost * count for cost in costs])
    assert data["x_title"] == f"Cost per {count:,} replies ($)"
    spec = data["x_fmt"]
    assert re.fullmatch(r"usd:\d", spec)
    labels = [_usd(tick, spec) for tick in _nice_ticks(data["x_domain"][1])]
    assert len(labels) >= 3 and len(set(labels)) == len(labels), labels
    assert all(re.search(r"[1-9]", label) for label in labels[1:]), labels
