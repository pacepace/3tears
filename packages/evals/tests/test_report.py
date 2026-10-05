"""The ``Report``: one document an analysis is read through, its published schema, and its serializers.

Driven end to end on the toy host: its corpus campaign is generated through the host (a fixtured writer
standing in for the model, the memo carrying a chart) and read back as a report through the service —
the one call that replaces the four-endpoint assembly. Then:

- **It serializes three ways.** JSON (the model's own dump, which the published schema validates),
  Markdown and HTML, each carrying the author's words, code's numbers and every chart's values as drawn.
- **The schema is published and true.** ``schema.json`` is the model's schema, byte for byte, and it
  validates the toy report's JSON.
- **Who wrote what stays legible.** An author's caveat is attached to its finding as written; what code
  adds is a disclosure; a stored chart that cannot be drawn says why.
- **Every refusal fires** — a block linked to a finding the report does not hold, a table showing more
  rows than it has or a value no column shows, a chart block that is drawn and failed or neither, a
  finding's words that name no finding.

The HTML's no-script property has its own file, ``test_report_html_no_script.py``.
"""

from __future__ import annotations

import json
from typing import Any

import jsonschema
import pytest

from threetears.evals.analysis import (
    ChartBlock,
    DisclosureBlock,
    Report,
    TableBlock,
    TableColumn,
    TextBlock,
    analysis_report,
    build_report,
    finding_chart_intent,
    published_report_schema,
    report_html,
    report_json_schema,
    report_markdown,
)
from threetears.evals.analysis.report import SCHEMA_PATH
from threetears.evals.contracts.campaign import EvalAnalysis, Viz
from threetears.evals.contracts.errors import NotFoundError
from threetears.evals.contracts.host import EvalHost
from packages.evals.tests.report_support import minimal_report, toy_report


@pytest.fixture
async def toy() -> tuple[EvalHost, EvalAnalysis, Report]:
    return await toy_report()


# =============================================================================
# Done when: the toy host's analysis serializes three ways, and the schema validates the JSON
# =============================================================================


class TestTheToyAnalysisSerializesThreeWays:
    async def test_json_validates_against_the_published_schema(self, toy: tuple[Any, Any, Report]) -> None:
        _, _, report = toy
        document = json.loads(report.to_canonical_json())

        jsonschema.Draft202012Validator(published_report_schema()).validate(document)
        assert Report.model_validate(document) == report

    async def test_the_schema_refuses_json_that_is_not_a_report(self, toy: tuple[Any, Any, Report]) -> None:
        """A schema that accepted anything would validate the report for nothing."""
        _, _, report = toy
        document = json.loads(report.to_canonical_json())
        document["blocks"][0]["kind"] = "banner"

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(published_report_schema()).validate(document)

    async def test_markdown_carries_the_words_the_numbers_and_the_chart_values(
        self, toy: tuple[Any, EvalAnalysis, Report]
    ) -> None:
        _, analysis, report = toy
        markdown = report_markdown(report)
        finding = analysis.document.findings[0]

        assert markdown.startswith(f"# {analysis.document.headline}\n")
        assert f"### 1. {finding.title}" in markdown
        assert "**Evidence**" in markdown and "| Arm | Measure | Value | n | Spread |" in markdown
        chart = _chart(report)
        assert f"**Chart: {chart.title}** (delta_table)" in markdown
        assert "the wide chunk is the slower one" in markdown
        assert "| " + " | ".join(column.header for column in chart.columns) + " |" in markdown
        assert f"> **Caveat** (Kind: sampling): {finding.caveats[0].text}" in markdown
        assert "## Methods and disclosures" in markdown

    async def test_html_carries_the_same_and_embeds_each_charts_intent(self, toy: tuple[Any, Any, Report]) -> None:
        import html as html_module

        _, _, report = toy
        page = report_html(report)
        chart = _chart(report)

        assert page.startswith("<!doctype html>")
        assert f'data-chart-type="{chart.type}"' in page
        embedded = page.split('data-chart-intent="', 1)[1].split('"', 1)[0]
        assert json.loads(html_module.unescape(embedded)) == chart.model_dump(mode="json")
        assert '<th scope="col">Metric</th>' in page


def _chart(report: Report) -> Any:
    (block,) = [block for block in report.blocks if isinstance(block, ChartBlock)]
    assert block.intent is not None, block.error
    return block.intent


# =============================================================================
# The schema is published, and is the model's
# =============================================================================


def test_the_published_schema_is_the_models_schema() -> None:
    """``schema.json`` is generated, committed and held here; regenerate it from ``report_json_schema()``."""
    assert published_report_schema() == report_json_schema(), (
        f"{SCHEMA_PATH.name} is stale: write json.dumps(report_json_schema(), indent=2, sort_keys=True) to it"
    )


def test_the_published_schema_is_a_valid_draft_2020_12_schema() -> None:
    jsonschema.Draft202012Validator.check_schema(published_report_schema())
    assert published_report_schema()["$schema"] == "https://json-schema.org/draft/2020-12/schema"


# =============================================================================
# What the toy report holds, and who wrote it
# =============================================================================


class TestTheToyReportsContent:
    async def test_the_sections_are_in_reading_order(self, toy: tuple[Any, Any, Report]) -> None:
        _, _, report = toy
        order = ["summary", "questions", "decisions", "findings", "arms", "surface", "next", "methods"]
        seen = list(dict.fromkeys(block.section for block in report.blocks))
        assert seen == [section for section in order if section in seen]
        assert set(seen) == set(order)

    async def test_an_authors_caveat_is_attached_to_its_finding_as_written(
        self, toy: tuple[Any, EvalAnalysis, Report]
    ) -> None:
        _, analysis, report = toy
        (caveat,) = [block for block in report.blocks if isinstance(block, TextBlock) and block.role == "caveat"]
        written = analysis.document.findings[0].caveats[0]

        assert (caveat.section, caveat.finding, caveat.body) == ("findings", 0, written.text)
        assert [(fact.name, fact.value) for fact in caveat.facts] == [("Kind", written.kind)]

    async def test_the_evidence_table_is_codes_numbers_at_the_finding(
        self, toy: tuple[Any, EvalAnalysis, Report]
    ) -> None:
        _, analysis, report = toy
        (table,) = [block for block in report.blocks if isinstance(block, TableBlock) and block.name == "evidence"]
        evidence = analysis.resolutions[0].evidence

        assert table.finding == 0
        assert [row["value"] for row in table.rows] == [row.value for row in evidence]
        assert [row["n"] for row in table.rows] == [row.n for row in evidence]

    async def test_the_chart_is_its_intent_never_a_renderers_spec(self, toy: tuple[Any, Any, Report]) -> None:
        _, _, report = toy
        (block,) = [block for block in report.blocks if isinstance(block, ChartBlock)]
        dumped = block.model_dump(mode="json")

        assert block.finding == 0 and block.error == ""
        assert "spec" not in json.dumps(dumped["intent"]["encodings"])
        assert "$schema" not in json.dumps(dumped)

    async def test_the_arm_and_surface_tables_are_in_the_report(self, toy: tuple[Any, Any, Report]) -> None:
        _, _, report = toy
        names = [block.name for block in report.blocks if isinstance(block, TableBlock)]
        assert "arms" in names and "surface" in names

    async def test_the_decision_rests_on_its_finding(self, toy: tuple[Any, Any, Report]) -> None:
        _, _, report = toy
        (decision,) = [block for block in report.blocks if isinstance(block, TextBlock) and block.role == "decision"]
        assert decision.rests_on == [0]
        assert ("Disposition", "adopted") in [(fact.name, fact.value) for fact in decision.facts]

    async def test_the_methods_appendix_says_how_the_analysis_was_generated(
        self, toy: tuple[Any, EvalAnalysis, Report]
    ) -> None:
        _, analysis, report = toy
        methods = [block for block in report.blocks if block.section == "methods"]
        assert [block.source for block in methods if isinstance(block, DisclosureBlock)] == ["generation"]
        assert analysis.generation.bundle_fingerprint in methods[0].text  # type: ignore[union-attr]

    async def test_a_stored_chart_this_build_cannot_draw_says_why(self, toy: tuple[Any, EvalAnalysis, Report]) -> None:
        """Served as the reason, never omitted — a missing chart reads as one that never was."""
        _, analysis, _ = toy
        resolution = analysis.resolutions[0]
        broken = resolution.model_copy(
            update={"chart": Viz(type="delta_table", payload={"rows": []}, ref=resolution.chart.ref)}  # type: ignore[union-attr]
        )
        report = build_report(analysis.model_copy(update={"resolutions": [broken]}))
        (block,) = [block for block in report.blocks if isinstance(block, ChartBlock)]

        assert block.intent is None and "delta_table" in block.error
        assert "cannot be drawn" in report_markdown(report)

    async def test_a_date_time_axis_discloses_why_it_is_not_builds(self, toy: tuple[Any, EvalAnalysis, Report]) -> None:
        from packages.evals.tests.test_viz_refs import time_axis

        _, analysis, _ = toy
        builds = time_axis()
        days = builds.model_copy(
            update={
                "basis": "date",
                "release_label": None,
                "basis_reason": "1 of 2 runs recorded no app_version (run-x)",
            }
        )
        surface = analysis.decision_surface.model_copy(update={"time_axis": days})
        report = build_report(analysis.model_copy(update={"decision_surface": surface}))

        assert any(
            isinstance(block, DisclosureBlock)
            and block.source == "time_axis"
            and block.text == "The time axis is UTC days, not builds: 1 of 2 runs recorded no app_version (run-x)."
            for block in report.blocks
        )

    async def test_the_service_refuses_an_analysis_it_does_not_hold(self, toy: tuple[EvalHost, Any, Any]) -> None:
        host, analysis, _ = toy
        with pytest.raises(NotFoundError):
            analysis_report(host.storage, "no-such-analysis", analysis.scope_id)


class TestOneFindingsChart:
    """``finding_chart_intent``: one finding's chart decided for a host's own renderer, or refused by name."""

    async def test_the_service_decides_the_findings_chart(self, toy: tuple[EvalHost, EvalAnalysis, Any]) -> None:
        host, analysis, report = toy
        intent = finding_chart_intent(host.storage, analysis.id, analysis.scope_id, "0")

        (block,) = [block for block in report.blocks if isinstance(block, ChartBlock)]
        assert intent == block.intent

    @pytest.mark.parametrize("finding_id", ["1", "-1", "first"])
    async def test_a_finding_the_analysis_does_not_hold_is_refused(
        self, toy: tuple[EvalHost, EvalAnalysis, Any], finding_id: str
    ) -> None:
        host, analysis, _ = toy
        with pytest.raises(NotFoundError, match="finding"):
            finding_chart_intent(host.storage, analysis.id, analysis.scope_id, finding_id)

    async def test_a_finding_with_no_chart_is_refused(self, toy: tuple[EvalHost, EvalAnalysis, Any]) -> None:
        host, analysis, _ = toy
        bare = analysis.resolutions[0].model_copy(update={"chart": None})
        host.storage.save_analysis(analysis.model_copy(update={"resolutions": [bare]}))
        with pytest.raises(NotFoundError, match="chart for finding"):
            finding_chart_intent(host.storage, analysis.id, analysis.scope_id, "0")

    async def test_a_stored_chart_this_build_cannot_decide_is_refused_with_why(
        self, toy: tuple[EvalHost, EvalAnalysis, Any]
    ) -> None:
        """Refused rather than answered empty: an empty answer would read as "no chart here"."""
        host, analysis, _ = toy
        resolution = analysis.resolutions[0]
        broken = resolution.model_copy(
            update={"chart": Viz(type="delta_table", payload={"rows": []}, ref=resolution.chart.ref)}  # type: ignore[union-attr]
        )
        host.storage.save_analysis(analysis.model_copy(update={"resolutions": [broken]}))
        with pytest.raises(NotFoundError, match="drawable chart for finding"):
            finding_chart_intent(host.storage, analysis.id, analysis.scope_id, "0")


# =============================================================================
# The report's own refusals
# =============================================================================


class TestTheReportsRefusals:
    def test_the_fixture_passes(self) -> None:
        assert minimal_report().finding_count == 1

    @pytest.mark.parametrize(
        "block",
        [
            {"kind": "text", "section": "findings", "role": "finding_body", "finding": 1, "body": "t"},
            {"kind": "disclosure", "section": "arms", "source": "arms", "text": "t", "rests_on": [3]},
        ],
        ids=["finding", "rests_on"],
    )
    def test_a_block_linked_to_a_finding_the_report_does_not_hold_is_refused(self, block: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match=r"is linked to finding position\(s\) \[\d\], and the report holds 1"):
            minimal_report(blocks=[block])

    def test_a_table_showing_more_rows_than_it_has_is_refused(self) -> None:
        with pytest.raises(ValueError, match="shows 2 rows but says it has 1"):
            TableBlock(
                section="arms",
                name="arms",
                title="Arms",
                columns=[TableColumn(key="arm", header="Arm")],
                rows=[{"arm": "a"}, {"arm": "b"}],
                order="as given",
                total_rows=1,
            )

    def test_a_truncated_table_says_so(self) -> None:
        table = TableBlock(
            section="arms",
            name="arms",
            title="Arms",
            columns=[TableColumn(key="arm", header="Arm")],
            rows=[{"arm": "a"}],
            order="as given",
            total_rows=4,
        )
        assert "_1 of 4 rows shown._" in report_markdown(minimal_report(blocks=[table]))

    def test_a_row_stating_a_value_no_column_shows_is_refused(self) -> None:
        with pytest.raises(ValueError, match="has rows keyed hidden, which no column shows"):
            TableBlock(
                section="arms",
                name="arms",
                title="Arms",
                columns=[TableColumn(key="arm", header="Arm")],
                rows=[{"arm": "a", "hidden": 1}],
                order="as given",
                total_rows=1,
            )

    @pytest.mark.parametrize("error", ["", "   "])
    def test_a_chart_block_with_neither_an_intent_nor_a_reason_is_refused(self, error: str) -> None:
        with pytest.raises(ValueError, match="exactly one of an intent and the reason it cannot be drawn"):
            ChartBlock(section="findings", finding=0, viz_type="breakdown", error=error)

    def test_a_chart_block_with_both_an_intent_and_a_reason_is_refused(self) -> None:
        from packages.evals.tests.chart_examples import EVERY_TYPE
        from threetears.evals.analysis.viz import chart_intent

        intent = chart_intent("breakdown", EVERY_TYPE["breakdown"])
        with pytest.raises(ValueError, match="exactly one of an intent and the reason it cannot be drawn"):
            ChartBlock(section="findings", finding=0, viz_type="breakdown", intent=intent, error="it broke")

    def test_a_chart_block_whose_intent_is_another_type_is_refused(self) -> None:
        from packages.evals.tests.chart_examples import EVERY_TYPE
        from threetears.evals.analysis.viz import chart_intent

        intent = chart_intent("breakdown", EVERY_TYPE["breakdown"])
        with pytest.raises(ValueError, match="a frontier chart block carries a breakdown intent"):
            ChartBlock(section="findings", finding=0, viz_type="frontier", intent=intent)

    @pytest.mark.parametrize("role", ["finding_title", "finding_body", "caveat", "carried_forward"])
    def test_a_findings_own_words_that_name_no_finding_are_refused(self, role: str) -> None:
        with pytest.raises(ValueError, match=f"a {role} block belongs to a finding and names none"):
            TextBlock(section="findings", role=role, body="t")  # type: ignore[arg-type]

    def test_a_summary_belongs_to_no_finding(self) -> None:
        assert TextBlock(section="summary", role="summary", body="t").finding is None
