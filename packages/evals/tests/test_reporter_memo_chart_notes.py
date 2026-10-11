"""The memo the reporter judge reads carries the prose code prints with a chart (#683).

A chart carries two kinds of words: the model's caption, and the compiler's footnote and disclosures
(``ChartIntent.footnote``, ``ChartIntent.disclosures``). The report prints both to every reader, and the judge
used to read only the first, so no eval scored the compiler's. These pin that the judge's input for an analysis
with a footnoted chart includes the footnote, beside that chart's caption, and that it is the same text the
report prints. What the judge makes of the words is the rubric's to say, not a check here.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.analysis import ChartBlock, render_memo_as_written
from threetears.evals.kernel.campaign import EvalAnalysis, Viz
from packages.evals.tests.chart_examples import EVERY_TYPE
from packages.evals.tests.report_support import toy_report
from threetears.evals.analysis.viz import chart_intent

#: The label the memo puts over the lines code prints with a chart.
_LABEL = "Printed with the chart:"


@pytest.fixture
async def toy() -> tuple[Any, EvalAnalysis, Any]:
    return await toy_report()


def _with_chart(analysis: EvalAnalysis, viz: Viz) -> EvalAnalysis:
    """The analysis, its first finding's compiled chart replaced by ``viz``."""
    first, *rest = analysis.resolutions
    return analysis.model_copy(update={"resolutions": [first.model_copy(update={"chart": viz}), *rest]})


def _footnoted() -> Viz:
    """A distribution whose label-only cohort cannot be placed on the axis, so its compiler writes a footnote."""
    return Viz(type="distribution", payload=EVERY_TYPE["distribution"], ref={})


def _chart_section(memo: str) -> str:
    """The memo from the first chart's line to the next blank-line-separated section that is not its notes."""
    return memo.split("\nChart (", 1)[1]


async def test_a_footnoted_charts_footnote_reaches_the_judge_beside_its_caption(toy) -> None:
    _, analysis, _ = toy
    footnote = chart_intent("distribution", EVERY_TYPE["distribution"]).footnote
    assert footnote.startswith("Individual runs not drawn for deepseek"), "the example is footnoted"

    memo = render_memo_as_written(_with_chart(analysis, _footnoted()))

    chart = _chart_section(memo)
    assert chart.startswith("delta_table): the wide chunk is the slower one"), "the model's caption is unchanged"
    notes = chart.split(f"\n\n{_LABEL}\n", 1)[1].split("\n\n", 1)[0].splitlines()
    assert notes[0] == f"- {footnote}", "the footnote is printed first, as the report prints it"


@pytest.mark.parametrize("viz_type", sorted(EVERY_TYPE))
async def test_the_memo_prints_every_line_the_report_prints_under_a_chart_in_its_order(toy, viz_type: str) -> None:
    """For every chart type: the footnote, then each disclosure — what ``serialize_md`` and ``serialize_html`` list."""
    _, analysis, _ = toy
    intent = chart_intent(viz_type, EVERY_TYPE[viz_type])
    printed = [line for line in (intent.footnote, *intent.disclosures) if line]

    memo = render_memo_as_written(_with_chart(analysis, Viz(type=viz_type, payload=EVERY_TYPE[viz_type], ref={})))

    if not printed:
        assert _LABEL not in memo
        return
    notes = _chart_section(memo).split(f"\n\n{_LABEL}\n", 1)[1].split("\n\n", 1)[0].splitlines()
    assert notes == [f"- {' '.join(line.splitlines()).strip()}" for line in printed]


async def test_a_chart_whose_compiler_wrote_nothing_prints_no_label(toy) -> None:
    """The toy's own delta table discloses nothing, so the judge reads its caption alone, as before."""
    _, analysis, report = toy
    (block,) = [block for block in report.blocks if isinstance(block, ChartBlock) and block.viz_type == "delta_table"]
    assert block.intent is not None and not block.intent.footnote and not block.intent.disclosures

    assert _LABEL not in render_memo_as_written(analysis)


async def test_a_chart_this_build_cannot_draw_prints_no_compiler_lines(toy) -> None:
    _, analysis, _ = toy
    memo = render_memo_as_written(_with_chart(analysis, Viz(type="distribution", payload={}, ref={})))

    assert "\nChart (delta_table): the wide chunk is the slower one" in memo
    assert _LABEL not in memo
