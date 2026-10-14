"""A classifier's per-label statistics are one table, every figure with its n, and no chart per label.

A classifier's code-only report once spent most of its length on a distribution "chart" per label and statistic
(eight for four labels, most reading 1 in every arm). Held here through the public report path: every label × arm ×
statistic the bundle holds is in the ``labels`` table with its n, stated as a count below ``MIN_CASES`` cases, and no
chart is drawn for a label.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from threetears.evals.analysis import (
    MIN_CASES,
    ChartBlock,
    DisclosureBlock,
    Report,
    TableBlock,
    inspect_campaign_bundle,
    report_markdown,
)
from threetears.evals.analysis.numbers import format_number
from threetears.evals.kernel.analysis_measures import MeasureSummary
from threetears.evals.kernel.metrics import classifier_label_of
from threetears.evals import Eval, Result
from packages.evals.tests.off_loop import off_loop

SCOPE = "compact-code-only-report"

CASES = [
    {"text": "a cat", "label": "animal"},
    {"text": "a dog", "label": "animal"},
    {"text": "a fir", "label": "plant"},
    {"text": "an oak", "label": "plant"},
]

CONTROL_ARM = "candidate=control (control)"
CANDIDATE_ARM = "candidate=candidate"


async def right(case: Mapping[str, Any]) -> str:
    return str(case["label"])


async def guess(case: Mapping[str, Any]) -> str:
    return "plant"


async def _classifier() -> Result:
    """Two arms over four cases twice: the control right every time, the candidate never saying ``animal``."""
    return await Eval(CASES, {"control": right, "candidate": guess}, ["label"], k=2).run(store=":memory:", scope=SCOPE)


def _table(report: Report, name: str) -> TableBlock | None:
    return next((block for block in report.blocks if isinstance(block, TableBlock) and block.name == name), None)


def _figure(summary: MeasureSummary) -> str:
    """A per-label figure as the table states it: a rate as a count under the case floor, else with its interval.

    The n names the cases it is over where they repeat — k=2 here — since those are the draws the interval counts.
    """
    cases = summary.n_independent
    n = f"(n={summary.n} over {cases} case{'' if cases == 1 else 's'})" if 0 < cases < summary.n else f"(n={summary.n})"
    if summary.rate is not None and cases < MIN_CASES:
        return f"{round(summary.rate * summary.n)} of {summary.n} {n}"
    if summary.rate is not None:
        assert summary.ci_low is not None and summary.ci_high is not None
        return f"{format_number(summary.rate)} [{format_number(summary.ci_low)}, {format_number(summary.ci_high)}] {n}"
    assert summary.mean is not None
    return f"{format_number(summary.mean)} {n}"


class TestPerLabelStatisticsAreOneTable:
    async def test_every_label_arm_and_statistic_is_in_it_with_its_interval_and_n(self) -> None:
        comparison = await _classifier()
        table = _table(comparison.report, "labels")
        assert table is not None
        assert [column.key for column in table.columns] == ["label", "arm", "precision", "recall", "f1"]

        # What the bundle holds, per label: one (precision, recall, F1) triple per arm.
        bundle = off_loop(inspect_campaign_bundle, comparison.host, comparison.campaign_id, SCOPE).bundle
        held: dict[str, Counter[tuple[str | None, ...]]] = {}
        for cell in bundle.cell_measures:
            by_label: dict[str, dict[str, str]] = {}
            for summary in cell.measures.measures:
                if (classifier := classifier_label_of(summary.name)) is not None:
                    by_label.setdefault(classifier[1], {})[classifier[0]] = _figure(summary)
            for label, figures in by_label.items():
                triple = tuple(figures.get(statistic) for statistic in ("precision", "recall", "f1"))
                held.setdefault(label, Counter())[triple] += 1
        assert set(held) == {"animal", "plant"}

        shown: dict[str, Counter[tuple[str | None, ...]]] = {}
        for row in table.values():
            triple = tuple(None if row[key] is None else str(row[key]) for key in ("precision", "recall", "f1"))
            shown.setdefault(str(row["label"]), Counter())[triple] += 1
        assert shown == held

        # Each label reads across both arms, the control first; label-major, so a label's arms sit together.
        assert [(row["label"], row["arm"]) for row in table.values()] == [
            ("animal", CONTROL_ARM),
            ("animal", CANDIDATE_ARM),
            ("plant", CONTROL_ARM),
            ("plant", CANDIDATE_ARM),
        ]
        assert table.total_rows == len(table.rows) == 4

    async def test_a_statistic_an_arm_has_none_of_is_a_dash_and_said_why(self) -> None:
        comparison = await _classifier()
        table = _table(comparison.report, "labels")
        assert table is not None
        (never_said,) = [row for row in table.values() if (row["label"], row["arm"]) == ("animal", CANDIDATE_ARM)]
        # Never predicted: no precision, and so no F1; recall is 0 of the 4 animals it met, a count on two cases.
        assert never_said["precision"] is None and never_said["f1"] is None
        assert never_said["recall"] == "0 of 4 (n=4 over 2 cases)"

        (said,) = [
            block.text
            for block in comparison.report.blocks
            if isinstance(block, DisclosureBlock) and block.text.startswith("Precision is counted over")
        ]
        assert "95% Wilson interval over the arm's cases" in said and "F1 has no interval by construction" in said
        assert f"below {MIN_CASES} cases a rate is stated as a count" in said
        assert "A label an arm never predicted has no precision" in said

        markdown = report_markdown(comparison.report)
        assert "**Per-label precision, recall and F1**" in markdown
        assert "| animal | candidate=candidate | — | 0 of 4 (n=4 over 2 cases) | — |" in markdown

    async def test_no_per_label_chart_and_no_notice_for_one(self) -> None:
        report = (await _classifier()).report
        charts = [block for block in report.blocks if isinstance(block, ChartBlock)]
        assert [chart.chart for chart in charts] == ["outcome_bar"], "two cases chart no rate, and no label is charted"
        assert not [
            block for block in report.blocks if isinstance(block, DisclosureBlock) and "classifier:" in block.text
        ]
        assert "**Chart: classifier:" not in report_markdown(report)
