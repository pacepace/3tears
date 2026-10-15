"""The report's two charts: inline SVG with every value written as text, nothing that runs, and nothing drawn thin.

- **Every mark's value is text.** An outcome bar writes each arm's count of every state, and an interval row its
  value, interval and cases, so nothing is read off a length or a colour alone.
- **Nothing runs and nothing is fetched.** No ``<script>``, no ``on*=`` attribute, no URL — not even a
  ``url(#...)`` fill.
- **A row needs its cases.** An arm with fewer than ``MIN_CASES`` usable cases is not drawn, and a chart left with
  no row is not built at all; the scoreboard says once what was not charted.
- **Light and dark.** The marks take their colours from the page's CSS variables, which are set for both.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals.analysis import ChartBlock, IntervalRow, MIN_CASES, OutcomeBar, OutcomeSegment, report_html
from threetears.evals.analysis.report.svg import SVG_STYLE, chart_svg
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


async def test_a_campaign_with_enough_cases_draws_its_contrast_as_interval_rows() -> None:
    ev = Eval(_cases(MIN_CASES + 3), {"never": _never, "classify": _classify}, ["label"])
    comparison = await ev.run(store=":memory:")
    charts = [block for block in comparison.report.blocks if isinstance(block, ChartBlock)]
    assert [block.chart for block in charts] == ["outcome_bar", "interval_rows"]
    page = report_html(comparison.report)
    assert page.count("<svg") == 2 and 'class="visually-hidden"' in page


async def test_an_arm_under_the_case_floor_is_not_charted_and_the_caveat_says_so_once() -> None:
    ev = Eval(_cases(MIN_CASES - 1), {"never": _never, "classify": _classify}, ["label"])
    comparison = await ev.run(store=":memory:")
    charts = [block for block in comparison.report.blocks if isinstance(block, ChartBlock)]
    assert [block.chart for block in charts] == ["outcome_bar"]
    page = report_html(comparison.report)
    assert page.count("<svg") == 1
    # Under the floor on every arm, the one caveat under the scoreboard says so, and no "not charted" line repeats it.
    assert page.count(f"(needs {MIN_CASES})") == 1 and "Not charted" not in page
