"""The words are served, not ported: a report's and a run summary's JSON carry every word a reader is shown.

A browser, a terminal and this package's serializers must print the same words, so no client re-implements the
number rule, the section headings or a finding's number. Held here, field by field:

- **A report cell is its value and its text.** The value is what a client sorts and draws by; the text is the
  number rule's (``render_cell``). A chart block serves its values-as-drawn table the same way, its columns
  already chosen for printing.
- **A report carries its title and byline, each block its section's title and its finding's number** — and the
  Markdown and HTML print exactly those.
- **The served words are derived, never stored.** A report's JSON reads back equal, what a document said for a
  served word is discarded and re-derived, and the published schema requires every one.
- **A run summary's JSON carries every line ``render()`` prints** — measure states, the spend cap and why it
  stopped included — and ``render()`` is those lines.
- **An unpriced amount reads ``unpriced``**, from the one public spelling of a spend.
"""

from __future__ import annotations

import json
from typing import Any

import jsonschema
import pytest

from threetears.evals.analysis import (
    REPORT_VERSION,
    SECTION_TITLES,
    UNPRICED,
    ChartBlock,
    DisclosureBlock,
    EvalSummary,
    RoleSpend,
    MeasureLine,
    Report,
    RunMeasureSummary,
    TableBlock,
    TableCell,
    TableColumn,
    TextBlock,
    dollars_text,
    published_report_schema,
    report_html,
    report_json_schema,
    report_markdown,
)
from threetears.evals.analysis.confusion import ConfusionCount
from threetears.evals.analysis.report import SCHEMA_ID
from threetears.evals.analysis.report.model import table_row
from threetears.evals.analysis.surface_table import NO_SUCCESSFUL_RESULTS
from threetears.evals.analysis.viz.intent import chart_intent
from threetears.evals.analysis.viz.intents.distribution import SHAPE_UNKNOWN
from threetears.evals.analysis.viz.quantities import render_cell
from threetears.evals.kernel import CONFUSION_CELL_MEASURE
from packages.evals.tests.report_support import minimal_report

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
    assert cell.model_dump(mode="json") == {"value": value, "unit": "", "text": text}


def test_a_cells_number_is_joined_to_its_unit() -> None:
    assert TableCell(value=1.42, unit="s").text == "1.42 s"
    assert TableCell(value=0.42, unit="%").text == "0.42%"


def test_a_cells_text_is_rederived_on_read_and_what_the_document_said_is_discarded() -> None:
    assert TableCell.model_validate({"value": 0.5, "text": "a lie"}).text == "0.5"


def _table(**update: Any) -> TableBlock:
    fields: dict[str, Any] = {
        "section": "arms",
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
        dumped = _table().model_dump(mode="json")
        assert dumped["rows"] == [
            {"arm": {"value": "b", "unit": "", "text": "b"}, "mean": {"value": 1234.5, "unit": "", "text": "1234"}},
            {"arm": {"value": "a", "unit": "", "text": "a"}, "mean": {"value": None, "unit": "", "text": "—"}},
        ]

    def test_values_reads_the_rows_as_values_alone(self) -> None:
        assert _table().values() == [{"arm": "b", "mean": 1234.5}, {"arm": "a", "mean": None}]

    def test_the_markdown_and_html_print_the_served_text(self) -> None:
        report = minimal_report(blocks=[*minimal_report().model_dump()["blocks"], _table().model_dump()])
        assert "| b | 1234 |" in report_markdown(report) and "| a | — |" in report_markdown(report)
        assert "<td>b</td><td>1234</td>" in report_html(report)


# =============================================================================
# A chart block serves its values-as-drawn table, its columns already chosen
# =============================================================================


def _distribution(groups: list[dict[str, Any]]) -> ChartBlock:
    return ChartBlock(
        section="surface",
        viz_type="distribution",
        intent=chart_intent("distribution", {"groups": groups, "unit": "s"}),
    )


def _bounded(label: str, mean: float) -> dict[str, Any]:
    return {
        "label": label,
        "ci": {"low": mean - 1, "high": mean + 1, "mean": mean, "level": 0.95, "variability": "across runs"},
        "n": 5,
    }


class TestAChartServesItsTable:
    def test_a_shape_no_row_knows_arrives_left_out_and_the_intent_keeps_it(self) -> None:
        block = _distribution([_bounded("a", 2.0), _bounded("b", 3.5)])
        assert block.intent is not None and {row["shape"] for row in block.intent.rows} == {SHAPE_UNKNOWN}
        assert [column.key for column in block.columns] == ["label", "mean", "low", "high", "n"]
        assert all(set(row) == {"label", "mean", "low", "high", "n"} for row in block.rows)
        assert "shape" not in json.dumps(block.model_dump(mode="json")["columns"])

    def test_one_known_shape_keeps_the_column_for_every_row(self) -> None:
        block = _distribution([{"label": "sampled", "samples": [1.0, 2.0, 3.0], "n": 3}, _bounded("bounded", 2.0)])
        assert block.columns[-1] == TableColumn(key="shape", header="Shape")
        assert block.rows[1]["shape"].text == SHAPE_UNKNOWN

    def test_each_chart_cell_is_the_intents_value_and_its_text(self) -> None:
        block = _distribution([_bounded("a", 2.25), _bounded("b", 3.5)])
        assert block.intent is not None
        first = block.rows[0]
        assert first["label"] == TableCell(value="a")
        assert first["mean"].value == block.intent.rows[0]["mean"] and first["mean"].text == "2.25"
        assert [column.header for column in block.columns][:2] == [block.intent.columns[0].header, "Mean (s)"]

    def test_a_chart_that_cannot_be_drawn_serves_no_table(self) -> None:
        block = ChartBlock(section="findings", finding=0, viz_type="breakdown", error="the payload names no parts")
        assert block.columns == [] and block.rows == []


# =============================================================================
# Every block carries its section's title and its finding's number
# =============================================================================


@pytest.mark.parametrize("section", list(SECTION_TITLES))
def test_every_block_carries_its_sections_title(section: Any) -> None:
    block = DisclosureBlock(section=section, source="generation", text="t")
    assert block.section_title == SECTION_TITLES[section]
    assert block.model_dump(mode="json")["section_title"] == SECTION_TITLES[section]


def test_a_block_carries_the_number_a_reader_counts_its_finding_by() -> None:
    assert TextBlock(section="findings", role="finding_title", finding=2, body="t").finding_number == 3
    assert TextBlock(section="summary", role="summary", body="t").finding_number is None


@pytest.mark.parametrize(
    "block",
    [
        TextBlock(section="findings", role="finding_body", finding=0, rests_on=[2, 0], body="t"),
        DisclosureBlock(section="methods", source="generation", text="t", rests_on=[2, 0]),
        ChartBlock(section="findings", rests_on=[2, 0], viz_type="frontier", error="cannot be drawn"),
    ],
    ids=["text", "disclosure", "chart"],
)
def test_a_block_carries_the_numbers_a_reader_counts_the_findings_it_rests_on_by(block: Any) -> None:
    """Counted from 1 the way ``finding_number`` is, in ``rests_on``'s order, which itself stays 0-based."""
    assert block.rests_on == [2, 0]
    assert block.rests_on_numbers == [3, 1]
    assert block.model_dump(mode="json")["rests_on_numbers"] == [3, 1]


def test_a_block_resting_on_nothing_carries_no_numbers() -> None:
    assert TextBlock(section="summary", role="summary", body="t").rests_on_numbers == []


def test_the_serializers_print_the_served_rests_on_numbers() -> None:
    report = minimal_report(
        finding_count=3,
        blocks=[{"kind": "text", "section": "decisions", "role": "decision", "body": "adopt b", "rests_on": [2, 0]}],
    )
    assert report.blocks[0].rests_on_numbers == [3, 1]
    assert "Rests on finding 3, 1." in report_markdown(report)
    assert "Rests on finding 3, 1." in report_html(report)


def test_the_serializers_print_the_served_section_titles_and_finding_numbers() -> None:
    report = minimal_report(
        finding_count=2,
        blocks=[
            {"kind": "text", "section": "findings", "role": "finding_title", "finding": 1, "body": "second"},
            {"kind": "disclosure", "section": "methods", "source": "generation", "text": "made by code"},
        ],
    )
    markdown, page = report_markdown(report), report_html(report)
    assert "### 2. second" in markdown and "<h3>2. second</h3>" in page
    for block in report.blocks:
        assert f"## {block.section_title}" in markdown and f"<h2>{block.section_title}</h2>" in page


# =============================================================================
# The report carries its title and byline
# =============================================================================


def _source(**update: Any) -> dict[str, Any]:
    return minimal_report().source.model_dump() | update


class TestTheReportCarriesItsTitleAndByline:
    def test_an_analysis_report_is_titled_by_its_headline(self) -> None:
        assert minimal_report(headline="the wide chunk is slower").title == "the wide chunk is slower"

    def test_an_analysis_report_names_its_campaign_when_it_has_its_name(self) -> None:
        report = minimal_report(headline="the wide chunk is slower", source=_source(campaign_name="chunk sweep"))
        assert report.title == "chunk sweep: the wide chunk is slower"

    def test_a_blank_headline_says_so(self) -> None:
        assert minimal_report(headline="  ").title == "(blank headline)"
        assert minimal_report(headline="", source=_source(campaign_name="sweep")).title == "sweep: (blank headline)"

    def test_a_code_only_report_names_its_campaign_by_name_else_id(self) -> None:
        code_only = {
            "basis": "code_only",
            "headline": "",
            "finding_count": 0,
            "source": _source(analysis_id=None, generator_model=None),
            "blocks": [],
        }
        assert Report.model_validate(code_only).title == "Campaign c: its evidence, with no analysis"
        named = code_only | {"source": code_only["source"] | {"campaign_name": "sweep"}}
        assert Report.model_validate(named).title == "Campaign sweep: its evidence, with no analysis"
        assert Report.model_validate(named).byline.startswith("Code-only report of campaign c — b; computed")

    def test_an_analysis_reports_byline_names_the_analysis_campaign_and_model(self) -> None:
        assert minimal_report().byline == ("Analysis a of campaign c — b; generated 2026-10-05T00:00:00+00:00 by m.")

    def test_the_serializers_print_the_served_title_and_byline(self) -> None:
        report = minimal_report(source=_source(campaign_name="sweep"))
        assert report_markdown(report).startswith(f"# {report.title}\n\n_{report.byline}_\n")
        page = report_html(report)
        assert f"<title>{report.title}</title>" in page and f'<p class="source">{report.byline}</p>' in page


# =============================================================================
# The served words round-trip, and the schema requires them
# =============================================================================


def _served_report() -> Report:
    return minimal_report(
        source=_source(campaign_name="sweep"),
        blocks=[
            {"kind": "text", "section": "findings", "role": "finding_title", "finding": 0, "body": "t"},
            _table(section="findings", finding=0).model_dump(),
            _distribution([_bounded("a", 2.0), _bounded("b", 3.5)]).model_dump(),
        ],
    )


class TestTheServedWordsRoundTrip:
    def test_the_json_reads_back_as_the_same_report(self) -> None:
        report = _served_report()
        document = json.loads(report.to_canonical_json())
        assert document["title"] == "sweep: h" and document["byline"] == report.byline
        assert [block["section_title"] for block in document["blocks"]] == ["Findings", "Findings", "Decision surface"]
        assert Report.model_validate(document) == report
        assert Report.model_validate_json(report.to_canonical_json()).to_canonical_json() == report.to_canonical_json()

    def test_what_a_document_says_for_a_served_word_is_rederived(self) -> None:
        report = _served_report()
        document = json.loads(report.to_canonical_json())
        document["title"] = "another title"
        document["blocks"][0]["finding_number"] = 9
        document["blocks"][0]["rests_on_numbers"] = [9]
        document["blocks"][1]["rows"][0]["mean"]["text"] = "a lie"
        assert Report.model_validate(document) == report

    def test_the_published_schema_validates_the_served_report(self) -> None:
        jsonschema.Draft202012Validator(published_report_schema()).validate(
            json.loads(_served_report().to_canonical_json())
        )

    @pytest.mark.parametrize(
        "drop",
        [
            lambda d: d.pop("title"),
            lambda d: d.pop("byline"),
            lambda d: d["blocks"][0].pop("section_title"),
            lambda d: d["blocks"][0].pop("finding_number"),
            lambda d: d["blocks"][0].pop("rests_on_numbers"),
            lambda d: d["blocks"][1]["rows"][0]["mean"].pop("text"),
            lambda d: d["blocks"][2].pop("columns"),
            lambda d: d["blocks"][2]["rows"][0]["label"].pop("text"),
        ],
        ids=[
            "title",
            "byline",
            "section_title",
            "finding_number",
            "rests_on_numbers",
            "table-cell-text",
            "chart-columns",
            "chart-cell-text",
        ],
    )
    def test_the_schema_requires_every_served_word(self, drop: Any) -> None:
        document = json.loads(_served_report().to_canonical_json())
        drop(document)
        assert not jsonschema.Draft202012Validator(published_report_schema()).is_valid(document)

    def test_the_schema_is_identified_by_the_report_version(self) -> None:
        schema = published_report_schema()
        assert schema == report_json_schema()
        assert schema["$id"] == SCHEMA_ID == f"urn:threetears-evals:report:v{REPORT_VERSION}"
        assert schema["properties"]["report_version"]["const"] == REPORT_VERSION == 9


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
