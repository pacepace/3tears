"""A table's labels start with a capital, a figure's cell holds its line, and a written report names itself as its
analysis.

- **Labels.** The cases row a table by stratum opens each arm with prints ``Cases`` (its value, which a program
  reads the row by, stays ``cases``); an outcome bar's values table heads its columns ``Ran``, ``Passed``,
  ``Failed: check`` (its legend and tooltips keep the segments' words).
- **Held cells, by structure.** A table's cells of figures and of intervals are found from what the table declares of
  its columns (a figure's or an interval's glossary term, an interval column's key), never by matching their text: a
  figure with its interval beside it (``4.5 (3.8 to 5.0)``) holds its whole line (``figs``), and an interval in any
  form the page prints, a duration's among them (``340ms to 1.2s``), sets its ends in the ``iv`` span ``check.py``
  finds them by. A column that declares neither is left as it was, whatever its text.
- **A written report's evidence record** names the report as its analysis (``IN_THE_WRITTEN_REPORT``), never
  ``none written yet``; the code-only page still says none was written.
- **A result's line** names its stratum as the summary's stratum lines do: two ids that would print alike each print
  as their id in both.
"""

from __future__ import annotations

from typing import Any

from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import TOYHOST_SCOPE, generate, ops_fixture, settled
from packages.evals.tests.report_kit_support import report_of
from packages.evals.tests.test_quick_comparison import CASES, careful, hasty
from threetears.evals import Eval
from threetears.evals.analysis import ChartBlock, OutcomeBar, OutcomeSegment, report_html, report_markdown
from threetears.evals.analysis.report.build import IN_THE_WRITTEN_REPORT, NO_ANALYSIS
from threetears.evals.analysis.report.html_parts import table_html
from threetears.evals.analysis.report.md_parts import table_lines
from threetears.evals.analysis.report.model import TableBlock, TableCell, TableColumn
from threetears.evals.analysis.report.words import CASES_READING, table_label
from threetears.evals.ops import report_read

# --- labels ------------------------------------------------------------------------------------------------------------


def test_the_cases_reading_prints_as_a_label_and_keeps_its_value() -> None:
    assert table_label(CASES_READING) == "Cases"
    assert table_label("cases, then measures") == "cases, then measures", "only the whole reading is a label"


async def test_a_table_by_stratum_prints_its_cases_row_capitalised_and_a_program_reads_it_as_before() -> None:
    stratified = [{**case, "stratum": "plain" if index % 2 else "lookalike"} for index, case in enumerate(CASES)]
    result = await Eval(stratified, {"hasty": hasty, "careful": careful}, ["label"], k=1).run(store=":memory:")
    (strata,) = [block for block in result.report.blocks if isinstance(block, TableBlock) and block.name == "strata"]
    assert CASES_READING in {str(row["reading"].value) for row in strata.rows}
    at = [column.key for column in strata.columns].index("reading")
    printed = {line.split(" | ")[at].strip("| ") for line in table_lines(strata)[2:]}
    assert "Cases" in printed and CASES_READING not in printed
    assert result.strata(reading=CASES_READING), "a program reads the row by its value"
    assert "| cases |" not in report_markdown(result.report)


def test_an_outcome_bars_values_table_heads_its_columns_as_labels_and_its_segments_keep_their_words() -> None:
    chart = ChartBlock(
        section="scoreboard",
        chart="outcome_bar",
        title="How each arm's trials ended",
        bars=[
            OutcomeBar(
                arm="alpha",
                segments=[
                    OutcomeSegment(state="passed", label="passed", count=6),
                    OutcomeSegment(state="failed", label="failed: check", count=2),
                    OutcomeSegment(state="lost", label="lost: judge truncated", count=1),
                ],
            )
        ],
    )
    headers, rows = chart.values_table()
    assert headers == ["Arm", "Ran", "Passed", "Failed: check", "Lost: judge truncated"]
    assert rows == [["alpha", "9", "6", "2", "1"]]
    assert [segment.label for bar in chart.bars for segment in bar.segments] == [
        "passed",
        "failed: check",
        "lost: judge truncated",
    ]


# --- held cells --------------------------------------------------------------------------------------------------------


def _against() -> TableBlock:
    """The shape of an aspect page's "Against …" table: a reading, the control's and the arm's figure, each with its
    interval beside it, a change, an interval column, a duration among them, and a column of words holding "to"."""
    columns = [
        TableColumn(key="reading", header="Reading", term="reading"),
        TableColumn(key="control", header="▸ base", term="control"),
        TableColumn(key="arm", header="fast", term="arm mean"),
        TableColumn(key="interval", header="Interval", term="interval"),
        TableColumn(key="note", header="Note"),
    ]
    rows = [
        {
            "reading": TableCell(value="Score (judged, 1–5)"),
            "control": TableCell(value="4.5 (3.8 to 5.0)"),
            "arm": TableCell(value="4.1 (3.6 to 4.7)"),
            "interval": TableCell(value="-0.9 to +0.1"),
            "note": TableCell(value="5 to 7 retries"),
        },
        {
            "reading": TableCell(value="Latency p50"),
            "control": TableCell(value="420ms"),
            "arm": TableCell(value=None),
            "interval": TableCell(value="340ms to 1.2s"),
            "note": TableCell(value="no variation"),
        },
    ]
    return TableBlock(
        section="contrasts", name="comparisons", title="Against base", columns=columns, rows=rows, order="as built",
        total_rows=len(rows),
    )  # fmt: skip


def test_a_figure_cell_holds_its_line_and_sets_its_interval_in_the_span_check_py_finds() -> None:
    written = table_html(_against())
    assert '<td class="figs">4.5 (<span class="iv">3.8 to 5.0</span>)</td>' in written
    assert '<td class="figs">4.1 (<span class="iv">3.6 to 4.7</span>)</td>' in written
    assert '<td class="figs">420ms</td>' in written


def test_an_interval_cell_in_any_form_sets_its_ends_in_one_span() -> None:
    written = table_html(_against())
    assert '<td class="interval"><span class="iv">-0.9 to +0.1</span></td>' in written
    assert '<td class="interval"><span class="iv">340ms to 1.2s</span></td>' in written


def test_a_cell_stating_no_figure_and_a_column_declaring_neither_are_left_as_they_were() -> None:
    written = table_html(_against())
    assert "<td>5 to 7 retries</td>" in written, "a column that declares no figure is never read by its text"
    assert "<td>no variation</td>" in written
    assert '<td class="figs">—</td>' not in written, "an absent figure is a dash, held nowhere"


def test_the_stylesheet_holds_a_figure_cell_on_one_line() -> None:
    page = report_html(report_of([_against()]))
    assert "td .iv, td.num.split .aside, td.figs { white-space: nowrap; }" in page


# --- a written report's evidence record --------------------------------------------------------------------------------


def _read(fixture: Any, form: str) -> str:
    """The campaign's page in ``form``, a written report's with its evidence record (every trial's record)."""
    return off_loop(
        report_read, fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE, format=form, trials="all"
    ).body


async def test_a_written_reports_evidence_record_names_the_report_as_its_analysis() -> None:
    fixture = off_loop(ops_fixture)
    code_only = _read(fixture, "markdown")
    assert NO_ANALYSIS in code_only and IN_THE_WRITTEN_REPORT not in code_only

    (job,) = (await generate(fixture.host, fixture.campaign.id)).jobs
    status = await settled(fixture.host, job.job_id)
    assert (status.state, status.status) == ("completed", "stored"), status

    for form in ("markdown", "html"):
        written = _read(fixture, form)
        assert IN_THE_WRITTEN_REPORT in written, form
        assert NO_ANALYSIS not in written, form
