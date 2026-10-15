"""The words are served, not ported: a report's cells and a run summary's JSON carry every word a reader is shown.

A browser, a terminal and this package's serializers must print the same words, so no client re-implements the
number rule. Held here, field by field:

- **A report cell is its value and its text.** The value is what a program sorts by; the text is the number rule's
  (``render_cell``), derived on every read and never stored, and the Markdown and HTML print exactly it.
- **A run summary's JSON carries every line ``render()`` prints** — measure states, the spend cap and why it
  stopped included — and ``render()`` is those lines.
- **An unpriced amount reads ``unpriced``**, from the one public spelling of a spend.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.analysis import (
    UNPRICED,
    EvalSummary,
    MeasureLine,
    RoleSpend,
    RunMeasureSummary,
    TableBlock,
    TableCell,
    TableColumn,
    dollars_text,
    report_html,
    report_markdown,
)
from threetears.evals.analysis.confusion import ConfusionCount
from threetears.evals.analysis.report.figures import NO_SUCCESSFUL_RESULTS
from threetears.evals.analysis.report.model import table_row
from threetears.evals.analysis.quantities import render_cell
from threetears.evals.kernel import CONFUSION_CELL_MEASURE
from packages.evals.tests.g2_support import g2_report

# =============================================================================
# A report cell is its value and its text
# =============================================================================


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (None, "—"),
        (True, "yes"),
        (False, "no"),
        (3, "3"),
        (12345.678, "12346"),
        (0.000012345, "0.00001234"),
        ("a", "a"),
    ],
)
def test_a_cells_text_is_the_number_rules_spelling_of_its_value(value: Any, text: str) -> None:
    cell = TableCell(value=value)
    assert cell.value == value and type(cell.value) is type(value)
    assert cell.text == text == render_cell(value)


def test_a_cells_number_is_joined_to_its_unit() -> None:
    assert TableCell(value=1.42, unit="s").text == "1.42 s"
    assert TableCell(value=0.42, unit="%").text == "0.42%"


def test_a_cell_stores_its_value_never_its_text() -> None:
    """The text is derived on every read, so a stored word can never drift from the value it states."""
    assert TableCell(value=0.5).model_dump(mode="json") == {"value": 0.5, "unit": ""}
    with pytest.raises(ValueError, match="extra"):
        TableCell.model_validate({"value": 0.5, "text": "a lie"})


def _table(**update: Any) -> TableBlock:
    fields: dict[str, Any] = {
        "section": "contrasts",
        "name": "arms",
        "title": "Arms",
        "columns": [TableColumn(key="arm", header="Arm"), TableColumn(key="mean", header="Mean (ms)")],
        "rows": [table_row({"arm": "b", "mean": 1234.5}), table_row({"arm": "a", "mean": None})],
        "order": "as given",
        "total_rows": 2,
    }
    return TableBlock(**(fields | update))


class TestATableServesItsCells:
    def test_each_cell_carries_its_value_and_its_text(self) -> None:
        rows = _table().rows
        assert [[(cell.value, cell.text) for cell in row.values()] for row in rows] == [
            [("b", "b"), (1234.5, "1234")],
            [("a", "a"), (None, "—")],
        ]

    def test_values_reads_the_rows_as_values_alone(self) -> None:
        assert _table().values() == [{"arm": "b", "mean": 1234.5}, {"arm": "a", "mean": None}]

    async def test_the_markdown_and_html_print_the_served_text(self) -> None:
        report = await g2_report()
        report = report.model_copy(update={"blocks": [*report.blocks, _table()]})
        assert "| b | 1234 |" in report_markdown(report) and "| a | — |" in report_markdown(report)
        assert "<td>b</td><td>1234</td>" in report_html(report)


# =============================================================================
# A run summary's JSON carries every line render() prints
# =============================================================================


def _summary(**update: Any) -> EvalSummary:
    fields: dict[str, Any] = {
        "run_id": "r",
        "scope_id": "s",
        "template_id": None,
        "candidate_model": "m",
        "status": "completed",
        "k_runs": 2,
        "n_cases": 3,
        "n_results": 6,
        "n_scored": 6,
        "n_candidate_failed": 0,
        "n_excluded": 0,
        "measures": [],
        "confusion": [],
        "labels": [],
        "errors": [],
    }
    return EvalSummary(**(fields | update))


def _measure(name: str, **update: Any) -> RunMeasureSummary:
    fields: dict[str, Any] = {"name": name, "n": 4, "mean": 0.5, "minimum": 0.0, "maximum": 1.0}
    return RunMeasureSummary(**(fields | update))


class TestTheRunSummaryServesItsLines:
    def test_the_run_and_results_lines(self) -> None:
        summary = _summary(arm="fast", n_candidate_failed=1, n_scored=5)
        assert summary.headline == "run r completed: arm fast (model m) over 3 case(s) x k=2"
        assert summary.results_text == "6 result(s): 5 scored, 1 failed by the candidate"

    def test_why_it_stopped(self) -> None:
        assert _summary().stopped_text is None
        assert _summary(stopped_because="cost cap reached").stopped_text == "stopped: cost cap reached"

    def test_the_spend_cap(self) -> None:
        assert _summary().spend_cap_text is None
        candidate = RoleSpend(role="candidate", calls=3, cost_usd=0.01)
        assert _summary(max_cost_usd=2.5, spend=[candidate]).spend_cap_text == "spend cap: $2.50 for this run"
        assert _summary(max_cost_usd=2.5).spend_cap_text == (
            "spend cap: $2.50 for this run; it counts only spend a result reports, and no result reported any"
        )
        judge = RoleSpend(role="judge", calls=1, cost_usd=0.01)
        assert _summary(max_cost_usd_origin="uncapped", spend=[judge]).spend_cap_text is not None

    def test_a_measure_no_result_carries_is_counted_in_one_line_never_printed_alone(self) -> None:
        """The host team's read of G2: fifteen ``no result carries it`` lines buried the three that mattered."""
        uncarried = [_measure(f"unused_{index}", n=0, mean=None, minimum=None, maximum=None) for index in range(15)]
        summary = _summary(measures=[_measure("accuracy", mean=0.75), *uncarried])
        printed = summary.render()
        assert "no result carries it" not in printed
        assert printed.count("unused_") == 15, "named once each, in the one line"
        assert summary.uncarried_text is not None and summary.uncarried_text.startswith(
            "not shown, carried by no result: 15 measures ("
        )
        assert [line for line in printed.splitlines() if "carried by no result" in line] == [
            f"  {summary.uncarried_text}"
        ]
        assert _summary(measures=[_measure("accuracy")]).uncarried_text is None

    def test_every_measure_state(self) -> None:
        summary = _summary(
            measures=[
                _measure("accuracy", mean=0.75, guardrail=True),
                _measure("verdict", mean=None),
                _measure(CONFUSION_CELL_MEASURE, mean=None),
                _measure("latency_ms", n=0, mean=None, minimum=None, maximum=None, n_no_turn=2),
                _measure("cost_usd", n=0, mean=None, minimum=None, maximum=None, n_faulted=1),
                _measure("leaked", n=0, mean=None, minimum=None, maximum=None),
            ],
            confusion=[ConfusionCount(expected="a", predicted="b", count=2)],
        )
        assert [(line.name, line.state) for line in summary.measure_lines] == [
            ("accuracy", "measured"),
            ("verdict", "not_a_number"),
            (CONFUSION_CELL_MEASURE, "in_confusion_matrix"),
            ("latency_ms", "no_successful_results"),
            ("cost_usd", "all_excluded"),
            ("leaked", "not_carried"),
        ]
        texts = [line.text for line in summary.measure_lines]
        assert texts[0].startswith("accuracy: mean 0.75 (n=4, min 0, max 1); a guardrail")
        assert texts[3] == f"latency_ms: {NO_SUCCESSFUL_RESULTS}, left out: 2 refused or errored with no turn taken"
        assert summary.confusion_lines == ["a → b: 2"]

    def test_each_roles_spend_and_the_total(self) -> None:
        from threetears.evals.analysis import DimensionSummary

        judged = [
            DimensionSummary(name="helpful", scale="ordinal", n=2, mean=4.0, minimum=3.0, maximum=5.0, cannot_tell=0)
        ]
        spend = [
            RoleSpend(role="candidate", calls=3, cost_usd=0.0123),
            RoleSpend(role="judge", calls=2, cost_usd=None, unpriced_calls=2),
        ]
        summary = _summary(judged=judged, spend=spend)
        assert summary.spend_lines == [
            "candidate spend: $0.0123 over 3 call(s)",
            "judge spend: unpriced over 2 call(s): its client reported no price for them",
        ]
        assert summary.total_spend_text == (
            "total spend: unknown in full: $0.0123 over the priced calls, plus 2 call(s) unpriced"
        )
        assert summary.dimension_lines == ["helpful (judged 1-5): mean 4 (n=2, min 3, max 5)"]
        assert _summary().spend_lines == [] and _summary().total_spend_text is None

    def test_the_intent_and_the_self_judging_warning(self) -> None:
        summary = _summary(intent="Answer  the\nquestion.", intent_source="from f's docstring")
        assert summary.intent_text == "intent (from f's docstring): Answer the question."
        assert _summary().self_judging_warning is None
        assert _summary(judge_shares_candidate_model=["m"]).self_judging_warning.startswith("self-judging: ")  # type: ignore[union-attr]

    def test_render_is_the_served_lines_and_text_is_render(self) -> None:
        summary = _summary(
            stopped_because="cancelled by an operator",
            max_cost_usd=1.0,
            measures=[_measure("accuracy")],
            errors=["case 0: boom"],
        )
        assert summary.text == summary.render()
        assert summary.render().splitlines() == [
            summary.headline,
            f"  {summary.results_text}",
            f"  {summary.stopped_text}",
            f"  {summary.spend_cap_text}",
            f"  {summary.measure_lines[0].text}",
            "  error: case 0: boom",
        ]

    def test_the_json_carries_every_line_and_reads_back_the_same(self) -> None:
        summary = _summary(stopped_because="budget", max_cost_usd=1.0, measures=[_measure("accuracy")])
        document = summary.model_dump(mode="json")
        for field in (
            "headline",
            "results_text",
            "stopped_text",
            "spend_cap_text",
            "measure_lines",
            "confusion_lines",
            "label_lines",
            "intent_text",
            "dimension_lines",
            "self_judging_warning",
            "spend_lines",
            "total_spend_text",
            "goal_check_lines",
            "text",
        ):
            assert field in document, field
        assert document["measure_lines"] == [
            MeasureLine(name="accuracy", state="measured", text=summary.measure_lines[0].text).model_dump()
        ]
        assert document["stopped_because"] == "budget" and document["stopped_text"] == "stopped: budget"
        document["text"] = "a lie"
        assert EvalSummary.model_validate(document) == summary
        assert EvalSummary.model_validate_json(summary.model_dump_json()).text == summary.text

    def test_an_unknown_key_is_still_refused(self) -> None:
        with pytest.raises(ValueError, match="extra"):
            EvalSummary.model_validate(_summary().model_dump(mode="json") | {"surprise": 1})


# =============================================================================
# An unpriced amount reads "unpriced"
# =============================================================================


def test_an_unpriced_amount_reads_unpriced_never_zero() -> None:
    assert dollars_text(None) == UNPRICED == "unpriced"
    assert dollars_text(0) == "$0"
