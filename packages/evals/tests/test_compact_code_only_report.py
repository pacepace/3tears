"""The code-only report says each thing once, and keeps every number.

A classifier's code-only report used to spend most of its length on blocks that repeated themselves: a
distribution "chart" per label and statistic (eight for four labels, each its own table, most reading 1 in
every arm), a ``Shape`` column reading "unknown — interval only" in every row of every chart, a ``Run
notes`` column of em dashes, a provenance sentence the opening line already says, and a paragraph on what
an analysis would add. Each is held here through the public report path:

- **One per-label table.** Every label × arm × statistic the bundle holds is in the ``labels`` table, with
  its interval and its n, and no per-label chart block is emitted.
- **No empty Shape column** in Markdown or HTML, while the intent keeps the shape for a renderer — and a
  chart whose shape IS known for one group prints the column.
- **No empty Run notes column**, and the column back as soon as one row has a note.
- **No Status or Rests on finding column** in the arms table when every arm is unresolved and none rests
  on a finding — always so on a code-only report — and both back on an analysis report whose arms have a
  verdict and the findings it rests on.
- **One line** where the paragraph was, no provenance sentence on a code-only surface, and no second
  "no analysis was generated" in the byline.

Mutations that turn this file red: charting ``classifier:`` measures again in ``_chartable``; dropping
``_label_blocks`` from ``build_code_only_report``; printing ``intent.columns`` in either serializer;
keying ``notes`` unconditionally in ``_surface_blocks``; keying ``status`` or ``findings`` unconditionally
in ``_arm_blocks``; restoring the long ``NO_ANALYSIS`` or the byline's repeat of it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from threetears.evals.analysis import (
    NO_ANALYSIS,
    ChartBlock,
    DisclosureBlock,
    Report,
    TableBlock,
    build_report,
    inspect_campaign_bundle,
    report_html,
    report_markdown,
)
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.surface_table import SURFACE_PROVENANCE
from threetears.evals.analysis.viz.intent import chart_intent
from threetears.evals.analysis.viz.intents.distribution import SHAPE_UNKNOWN
from threetears.evals.kernel.analysis_measures import MeasureSummary
from threetears.evals.kernel.authored import AuthoredAnalysis
from threetears.evals.kernel.metrics import classifier_label_of
from threetears.evals.quick import Comparison, compare
from packages.evals.tests.test_surface_table import CANDIDATE, analysis, key, two_arm_surface

SCOPE = "compact-code-only-report"

CASES = [
    {"text": "a cat", "label": "animal"},
    {"text": "a dog", "label": "animal"},
    {"text": "a fir", "label": "plant"},
    {"text": "an oak", "label": "plant"},
]

CONTROL_ARM = "candidate=control (control)"
CANDIDATE_ARM = "candidate=candidate"


def _expected(case: Mapping[str, Any]) -> str:
    return str(case["label"])


async def right(case: Mapping[str, Any]) -> str:
    return str(case["label"])


async def guess(case: Mapping[str, Any]) -> str:
    return "plant"


async def _classifier() -> Comparison:
    """Two arms over four cases twice: the control right every time, the candidate never saying ``animal``."""
    return await compare(
        CASES, {"control": right, "candidate": guess}, expected=_expected, control="control", scope_id=SCOPE, k=2
    )


def _table(report: Report, name: str) -> TableBlock | None:
    return next((block for block in report.blocks if isinstance(block, TableBlock) and block.name == name), None)


def _figure(summary: MeasureSummary) -> str:
    """A per-label figure as the table states it: the rate with its interval, or F1's value, with its n.

    The n names the cases it is over where they repeat — k=2 here — since those are the draws the interval counts.
    """
    cases = summary.n_independent
    n = f"(n={summary.n} over {cases} case{'' if cases == 1 else 's'})" if 0 < cases < summary.n else f"(n={summary.n})"
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
        bundle = inspect_campaign_bundle(comparison.host, comparison.campaign_id, SCOPE).bundle
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
        for row in table.rows:
            triple = tuple(None if row[key] is None else str(row[key]) for key in ("precision", "recall", "f1"))
            shown.setdefault(str(row["label"]), Counter())[triple] += 1
        assert shown == held

        # Each label reads across both arms, the control first; label-major, so a label's arms sit together.
        assert [(row["label"], row["arm"]) for row in table.rows] == [
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
        (never_said,) = [row for row in table.rows if (row["label"], row["arm"]) == ("animal", CANDIDATE_ARM)]
        # Never predicted: no precision, and so no F1; recall is 0 of the 4 animals it met.
        assert never_said["precision"] is None and never_said["f1"] is None
        assert str(never_said["recall"]).startswith("0 [") and str(never_said["recall"]).endswith("(n=4 over 2 cases)")

        (said,) = [
            block.text
            for block in comparison.report.blocks
            if isinstance(block, DisclosureBlock) and block.text.startswith("Precision is counted over")
        ]
        assert "95% Wilson interval" in said and "F1 has no interval by construction" in said
        # k=2 repeats each case: the interval is over the cases, and says so.
        assert "Wilson interval over the cell's cases" in said
        assert "a case's repeats are not counted as independent" in said
        assert "A label an arm never predicted has no precision" in said

        markdown = report_markdown(comparison.report)
        assert "**Per-label precision, recall and F1**" in markdown
        assert "| animal | candidate=candidate | — | 0 [" in markdown

    async def test_no_per_label_chart_and_no_notice_for_one(self) -> None:
        report = (await _classifier()).report
        charts = [block for block in report.blocks if isinstance(block, ChartBlock)]
        titles = [chart.intent.title for chart in charts if chart.intent is not None]
        assert titles == ["Accuracy"], "a single reading keeps its distribution chart; per-label ones are the table's"
        assert not [
            block for block in report.blocks if isinstance(block, DisclosureBlock) and "classifier:" in block.text
        ]
        assert "**Chart: classifier:" not in report_markdown(report)


class TestNoColumnThatSaysNothing:
    async def test_an_interval_only_chart_prints_no_shape_column_and_its_intent_keeps_the_shape(self) -> None:
        # Five cases, the fewest a chart draws an interval band from: below that it draws the cases (#677).
        five = [*CASES, {"text": "a fern", "label": "plant"}]
        report = (
            await compare(
                five, {"control": right, "candidate": guess}, expected=_expected, control="control", scope_id=SCOPE, k=2
            )
        ).report
        (chart,) = [block for block in report.blocks if isinstance(block, ChartBlock)]
        assert chart.intent is not None
        assert {row["shape"] for row in chart.intent.rows} == {SHAPE_UNKNOWN}, "the intent still says it, per row"

        markdown = report_markdown(report)
        assert "| Shape |" not in markdown and SHAPE_UNKNOWN not in markdown
        assert "| Group | Mean | Low | High | Cases |" in markdown
        page = report_html(report)
        assert '<th scope="col">Shape</th>' not in page
        assert '<th scope="col">Group</th><th scope="col">Mean</th>' in page

    def test_a_chart_with_one_known_shape_prints_the_column_for_every_group(self) -> None:
        intent = chart_intent(
            "distribution",
            {
                "groups": [
                    {"label": "sampled", "samples": [1.0, 2.0, 3.0], "n": 3},
                    {
                        "label": "bounded",
                        "ci": {"low": 1.0, "high": 3.0, "mean": 2.0, "level": 0.95, "variability": "across runs"},
                        "n": 5,
                    },
                ],
                "unit": "s",
            },
        )
        report = Report.model_validate(
            {
                "basis": "code_only",
                "headline": "",
                "finding_count": 0,
                "source": {
                    "campaign_id": "c",
                    "scope_id": "s",
                    "subject_id": "x",
                    "subject_kind": "",
                    "behavior": "b",
                    "generated_at": "2026-10-09T00:00:00+00:00",
                    "bundle_fingerprint": "f",
                },
                "blocks": [
                    {"kind": "chart", "section": "surface", "viz_type": "distribution", "intent": intent.model_dump()}
                ],
            }
        )
        markdown = report_markdown(report)
        assert "| Shape |" in markdown and "3 samples" in markdown and SHAPE_UNKNOWN in markdown
        assert '<th scope="col">Shape</th>' in report_html(report)

    async def test_a_surface_with_no_run_notes_has_no_notes_column(self) -> None:
        report = (await _classifier()).report
        surface = _table(report, "surface")
        assert surface is not None
        assert [column.key for column in surface.columns] == ["arm", "replication"]
        assert "Run notes" not in report_markdown(report)

    def test_one_run_note_brings_the_column_back_for_every_row(self) -> None:
        quiet = _table(build_report(analysis(two_arm_surface())), "surface")
        assert quiet is not None and "notes" not in [column.key for column in quiet.columns]

        noted = two_arm_surface()
        candidate = next(cell for cell in noted.cells if cell.variant_key == key(CANDIDATE))
        candidate.short_runs = {"run-a": "measured 3 of 5"}
        table = _table(build_report(analysis(noted)), "surface")
        assert table is not None and table.columns[-1].key == "notes"
        assert sorted(row["notes"] or "" for row in table.rows) == ["", "short (run-a): measured 3 of 5"]


class TestTheArmsTableHasNoColumnNothingFilled:
    async def test_a_code_only_report_has_no_status_and_no_finding_column(self) -> None:
        report = (await _classifier()).report
        arms = _table(report, "arms")
        assert arms is not None
        assert [column.key for column in arms.columns] == ["arm", "levers"]
        assert [row["arm"] for row in arms.rows] == ["candidate=candidate", CONTROL_ARM]
        # With no status to order by, the caption names the order the rows are in, not one by status.
        assert arms.order == "by arm"

        markdown = report_markdown(report)
        assert "**Arms** (by arm)\n\n| Arm | Every lever it ran |\n" in markdown
        assert "| Status |" not in markdown and "Rests on finding" not in markdown and "unresolved" not in markdown
        page = report_html(report)
        assert '<th scope="col">Arm</th><th scope="col">Every lever it ran</th>' in page
        assert '<th scope="col">Status</th>' not in page and "Rests on finding" not in page
        assert "winner, then" not in markdown + page

    def test_an_analysis_report_whose_arms_have_verdicts_keeps_both_columns(self) -> None:
        report = build_report(analysis(two_arm_surface()))
        arms = _table(report, "arms")
        assert arms is not None
        assert [column.key for column in arms.columns] == ["arm", "status", "findings", "levers"]
        assert arms.order == "winner, then contradicted, then ruled out, then replaced incumbent, then unresolved"
        by_status = {row["status"]: row["findings"] for row in arms.rows}
        # The winner rests on the decision's finding; the incumbent it replaced rests on none, a dash.
        assert by_status == {"winner": "1", "replaced incumbent": None}

        markdown = report_markdown(report)
        assert "| Arm | Status | Rests on finding | Every lever it ran |" in markdown
        page = report_html(report)
        assert '<th scope="col">Status</th><th scope="col">Rests on finding</th>' in page

    def test_the_rule_is_the_tables_not_the_reports_basis(self) -> None:
        """An analysis that decided nothing about its arms has an arms table as bare as a code-only one."""
        decided = analysis(two_arm_surface())
        document = decided.document.model_dump()
        document["decisions"][0] |= {"disposition": "deferred", "revisit_when": "a k=5 pass"}
        report = build_report(analysis(two_arm_surface(), document=AuthoredAnalysis.model_validate(document)))
        arms = _table(report, "arms")
        assert arms is not None
        assert [column.key for column in arms.columns] == ["arm", "levers"] and arms.order == "by arm"


class TestTheOpeningIsOneLine:
    async def test_the_summary_is_one_sentence_and_the_surface_does_not_repeat_it(self) -> None:
        report = (await _classifier()).report
        assert (
            NO_ANALYSIS
            == "No analysis was generated: everything below was computed by code from the campaign's evidence."
        )
        first = report.blocks[0]
        assert isinstance(first, DisclosureBlock) and first.text == NO_ANALYSIS
        summary = report_markdown(report).split("## Summary\n\n", 1)[1].split("\n\n## ", 1)[0]
        assert summary == f"> {NO_ANALYSIS}"
        assert not [
            block for block in report.blocks if isinstance(block, DisclosureBlock) and block.text == SURFACE_PROVENANCE
        ]

    async def test_the_byline_does_not_repeat_it(self) -> None:
        report = (await _classifier()).report
        markdown, page = report_markdown(report), report_html(report)
        byline = markdown.splitlines()[2]
        assert byline.startswith("_Code-only report of campaign ") and "No analysis was generated" not in byline
        assert markdown.count("No analysis was generated") == 1
        assert page.count("No analysis was generated") == 1

    def test_an_analysis_report_keeps_the_provenance_sentence(self) -> None:
        report = build_report(analysis(two_arm_surface()))
        assert [
            block for block in report.blocks if isinstance(block, DisclosureBlock) and block.text == SURFACE_PROVENANCE
        ]
