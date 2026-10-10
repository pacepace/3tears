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
    set_campaign_control,
    NO_ANALYSIS,
    REPORT_VERSION,
    ChartBlock,
    DisclosureBlock,
    Report,
    TableBlock,
    TableColumn,
    TextBlock,
    analysis_report,
    build_report,
    campaign_report,
    finding_chart_intent,
    published_report_schema,
    report_html,
    report_json_schema,
    report_markdown,
)
from threetears.evals.analysis.references import resolve_reading
from threetears.evals.analysis.report import SCHEMA_PATH
from threetears.evals.contracts.campaign import EvalAnalysis, Viz
from threetears.evals.contracts.errors import NotFoundError
from threetears.evals.contracts.host import EvalHost
from threetears.evals.run import set_analysis_archived
from packages.evals.tests.report_support import minimal_report, toy_campaign_host, toy_report


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
        assert "**Evidence**" in markdown and "| Arm | Measure | Value | Cases | Spread |" in markdown
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
    (block,) = [block for block in report.blocks if isinstance(block, ChartBlock) and block.section == "findings"]
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
        order = ["summary", "questions", "decisions", "guardrails", "findings", "arms", "surface", "next", "methods"]
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
        assert [row["cases"] for row in table.rows] == [row.n_cases for row in evidence]

    async def test_the_delta_tables_n_is_the_smaller_arms_cases_never_its_observations(
        self, toy: tuple[Any, EvalAnalysis, Report]
    ) -> None:
        _, analysis, _ = toy
        chart = analysis.resolutions[0].chart
        assert chart is not None and chart.type == "delta_table"
        surface = analysis.decision_surface
        for row, ref in zip(chart.payload["rows"], chart.ref["measures"], strict=True):
            readings = [
                resolve_reading(surface, chart.ref[side], ref["measure_id"], ref["reading"])
                for side in ("a_cell", "b_cell")
            ]
            assert all(reading.n_cases is not None and reading.n_cases < reading.n for reading in readings), (
                "the toy repeats each case"
            )
            assert row["n"] == min(reading.n_cases or 0 for reading in readings)

    async def test_the_chart_is_its_intent_never_a_renderers_spec(self, toy: tuple[Any, Any, Report]) -> None:
        _, _, report = toy
        (block,) = [block for block in report.blocks if isinstance(block, ChartBlock) and block.section == "findings"]
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
        assert ("Disposition", "deferred") in [(fact.name, fact.value) for fact in decision.facts]

    async def test_the_methods_appendix_says_how_the_analysis_was_generated(
        self, toy: tuple[Any, EvalAnalysis, Report]
    ) -> None:
        _, analysis, report = toy
        methods = [block for block in report.blocks if block.section == "methods"]
        # Then the count of coverage levers no finding names (#631), which every analysis with a coverage map states.
        assert [block.source for block in methods if isinstance(block, DisclosureBlock)] == ["generation", "surface"]
        assert analysis.generation.bundle_fingerprint in methods[0].text  # type: ignore[union-attr]
        assert "lever(s) in the coverage map" in methods[1].text  # type: ignore[union-attr]

    async def test_a_stored_chart_this_build_cannot_draw_says_why(self, toy: tuple[Any, EvalAnalysis, Report]) -> None:
        """Served as the reason, never omitted — a missing chart reads as one that never was."""
        _, analysis, _ = toy
        resolution = analysis.resolutions[0]
        broken = resolution.model_copy(
            update={"chart": Viz(type="delta_table", payload={"rows": []}, ref=resolution.chart.ref)}  # type: ignore[union-attr]
        )
        report = build_report(analysis.model_copy(update={"resolutions": [broken]}))
        (block,) = [block for block in report.blocks if isinstance(block, ChartBlock) and block.section == "findings"]

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

        (block,) = [block for block in report.blocks if isinstance(block, ChartBlock) and block.section == "findings"]
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


# =============================================================================
# The basis: a report of an analysis, or of the evidence alone — and the shape refuses a mix
# =============================================================================


def _code_only(**update: Any) -> Report:
    """A code-only report that passes, with ``update`` applied over its fields."""
    fields: dict[str, Any] = {
        "basis": "code_only",
        "headline": "",
        "finding_count": 0,
        "source": {
            "campaign_id": "c",
            "scope_id": "s",
            "subject_id": "x",
            "subject_kind": "",
            "behavior": "b",
            "generated_at": "2026-10-05T00:00:00+00:00",
            "bundle_fingerprint": "f",
        },
        "blocks": [{"kind": "disclosure", "section": "summary", "source": "generation", "text": "no analysis"}],
    }
    return Report.model_validate(fields | update)


class TestTheBasisIsRefusedWhenTheReportDisagreesWithIt:
    def test_the_code_only_fixture_passes(self) -> None:
        assert _code_only().basis == "code_only"

    @pytest.mark.parametrize("missing", ["analysis_id", "generator_model"])
    def test_an_analysis_report_names_its_analysis_and_its_writer(self, missing: str) -> None:
        source = minimal_report().source.model_dump() | {missing: None}
        with pytest.raises(ValueError, match="an analysis report names its analysis_id and generator_model"):
            minimal_report(source=source)

    @pytest.mark.parametrize("named", [{"analysis_id": "a"}, {"generator_model": "m"}])
    def test_a_code_only_report_names_no_analysis(self, named: dict[str, str]) -> None:
        source = _code_only().source.model_dump() | named
        with pytest.raises(ValueError, match="a code-only report renders no analysis"):
            _code_only(source=source)

    def test_a_code_only_report_carries_no_headline(self) -> None:
        with pytest.raises(ValueError, match="carries no headline"):
            _code_only(headline="the wide chunk is slower")

    def test_a_code_only_report_counts_no_findings(self) -> None:
        with pytest.raises(ValueError, match="has no findings, and this one counts 1"):
            _code_only(finding_count=1)

    def test_a_code_only_report_holds_no_authors_words(self) -> None:
        blocks = [{"kind": "text", "section": "summary", "role": "summary", "body": "words"}]
        with pytest.raises(ValueError, match=r"blocks \[0\] are text blocks"):
            _code_only(blocks=blocks)

    def test_the_schema_holds_the_version(self) -> None:
        document = json.loads(_code_only().to_canonical_json())
        assert document["report_version"] == REPORT_VERSION == 6
        document["report_version"] = 5
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(published_report_schema()).validate(document)


# =============================================================================
# The schema states every cross-field rule JSON Schema can express; the model holds the three it cannot
# =============================================================================


def _schema_refuses(document: dict[str, Any]) -> bool:
    return not jsonschema.Draft202012Validator(published_report_schema()).is_valid(document)


def _model_refuses(document: dict[str, Any]) -> bool:
    try:
        Report.model_validate(document)
    except ValueError:
        return True
    return False


def _analysis_document() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(minimal_report().to_canonical_json())
    return document


def _code_only_document() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(_code_only().to_canonical_json())
    return document


def _with(document: dict[str, Any], change: Any) -> dict[str, Any]:
    change(document)
    return document


def _chart_block(**update: Any) -> dict[str, Any]:
    return {"kind": "chart", "section": "methods", "finding": None, "rests_on": [], "viz_type": "frontier"} | update


_TABLE = {
    "kind": "table",
    "section": "methods",
    "finding": None,
    "rests_on": [],
    "name": "evidence",
    "title": "Evidence",
    "columns": [{"key": "arm", "header": "Arm"}],
    "rows": [{"arm": "a"}, {"arm": "b"}],
    "order": "as listed",
    "total_rows": 2,
}


class TestTheSchemaHoldsTheCrossFieldRulesItCanState:
    """A host validating against ``schema.json`` alone is refused what the model refuses, but for three rules."""

    def test_the_conforming_documents_pass_both(self) -> None:
        for document in (_analysis_document(), _code_only_document()):
            assert not _schema_refuses(document) and not _model_refuses(document)

    @pytest.mark.parametrize(
        "document",
        [
            _with(_code_only_document(), lambda d: d["source"].update(analysis_id="a")),
            _with(_code_only_document(), lambda d: d["source"].update(generator_model="m")),
            _with(_code_only_document(), lambda d: d.update(headline="the wide chunk is slower")),
            _with(_code_only_document(), lambda d: d.update(finding_count=3)),
            _with(
                _code_only_document(),
                lambda d: d["blocks"].append(
                    {
                        "kind": "text",
                        "section": "summary",
                        "finding": None,
                        "rests_on": [],
                        "role": "summary",
                        "body": "words",
                        "facts": [],
                    }
                ),
            ),
            _with(_code_only_document(), lambda d: d["blocks"][0].update(finding=7)),
            _with(_code_only_document(), lambda d: d["blocks"][0].update(rests_on=[99])),
            _with(_analysis_document(), lambda d: d["source"].update(analysis_id=None)),
            _with(_analysis_document(), lambda d: d["source"].update(generator_model=None)),
            _with(_analysis_document(), lambda d: d["blocks"][0].update(finding=None)),
            _with(_code_only_document(), lambda d: d["blocks"].append(_chart_block(intent=None, error=""))),
            _with(_code_only_document(), lambda d: d["blocks"].append(_chart_block(intent=None, error="   "))),
        ],
        ids=[
            "code-only-names-an-analysis",
            "code-only-names-a-model",
            "code-only-has-a-headline",
            "code-only-counts-findings",
            "code-only-holds-a-text-block",
            "no-findings-but-a-block-names-one",
            "no-findings-but-a-block-rests-on-one",
            "analysis-names-no-analysis",
            "analysis-names-no-model",
            "a-findings-title-names-no-finding",
            "a-chart-with-neither-intent-nor-error",
            "a-chart-whose-error-is-blank",
        ],
    )
    def test_each_rule_the_schema_can_state_refuses_in_the_schema_and_the_model(self, document: dict[str, Any]) -> None:
        assert _model_refuses(document), "the case must be one the model refuses"
        assert _schema_refuses(document)

    async def test_a_chart_with_both_an_intent_and_an_error_or_another_types_intent_is_refused(
        self, toy: tuple[Any, Any, Report]
    ) -> None:
        _, _, report = toy
        drawn = json.loads(report.to_canonical_json())
        index = next(i for i, block in enumerate(drawn["blocks"]) if block["kind"] == "chart")
        for change in ({"error": "it cannot be drawn"}, {"viz_type": "frontier"}):
            document = json.loads(report.to_canonical_json())
            document["blocks"][index].update(change)
            assert _model_refuses(document) and _schema_refuses(document), change
        assert not _schema_refuses(drawn)

    @pytest.mark.parametrize(
        "document",
        [
            _with(_analysis_document(), lambda d: d["blocks"][0].update(finding=5)),
            _with(_code_only_document(), lambda d: d["blocks"].append(_TABLE | {"total_rows": 0})),
            _with(_code_only_document(), lambda d: d["blocks"].append(_TABLE | {"rows": [{"stray": "x"}]})),
        ],
        ids=["a-position-past-finding_count", "total_rows-below-rows-shown", "a-row-keyed-by-no-column"],
    )
    def test_the_three_rules_that_compare_siblings_are_the_models_alone(self, document: dict[str, Any]) -> None:
        """What the README and the model's docstring name as the schema's superset, pinned so it cannot grow."""
        assert _model_refuses(document)
        assert not _schema_refuses(document)


# =============================================================================
# The code-only report: the toy campaign, with no analysis generated
# =============================================================================


@pytest.fixture
def code_only() -> tuple[EvalHost, Report]:
    host, campaign = toy_campaign_host()
    return host, campaign_report(host, campaign.id, campaign.scope_id)


class TestACampaignWithNoAnalysisIsReportedFromItsEvidence:
    def test_it_is_code_only_and_says_so_first(self, code_only: tuple[EvalHost, Report]) -> None:
        _, report = code_only
        assert report.basis == "code_only" and report.headline == "" and report.finding_count == 0
        first = report.blocks[0]
        assert isinstance(first, DisclosureBlock) and (first.section, first.source, first.text) == (
            "summary",
            "generation",
            NO_ANALYSIS,
        )

    def test_it_holds_no_authors_words(self, code_only: tuple[EvalHost, Report]) -> None:
        _, report = code_only
        assert not [block for block in report.blocks if isinstance(block, TextBlock)]

    def test_it_lays_out_the_arms_and_the_surface_through_the_analysis_reports_builders(
        self, code_only: tuple[EvalHost, Report]
    ) -> None:
        """Every arm is unresolved and rests on no finding — a verdict is a decision's, and nothing decided — so
        the arm table has neither a status nor a finding column, and its rows are in the arms' order alone."""
        _, report = code_only
        tables = {block.name: block for block in report.blocks if isinstance(block, TableBlock)}
        assert {"arms", "surface"} <= set(tables)
        arms = tables["arms"]
        assert arms.rows and [column.key for column in arms.columns] == ["arm", "levers"]
        assert all(set(row) == {"arm", "levers"} for row in arms.rows)
        assert arms.order == "by arm"
        assert tables["surface"].rows

    def test_it_draws_a_distribution_per_measure_the_surface_can_draw(self, code_only: tuple[EvalHost, Report]) -> None:
        host, report = code_only
        charts = [block for block in report.blocks if isinstance(block, ChartBlock)]
        assert charts and all(chart.viz_type == "distribution" and chart.finding is None for chart in charts)
        assert all(chart.intent is not None for chart in charts)
        drawn_or_disclosed = {chart.intent.title for chart in charts if chart.intent is not None} | {
            block.text for block in report.blocks if isinstance(block, DisclosureBlock) and block.source == "chart"
        }
        assert any(text.startswith("Turn time") for text in drawn_or_disclosed), drawn_or_disclosed
        titles = [chart.intent.title for chart in charts if chart.intent is not None]
        assert len(set(titles)) == len(titles), titles

    def test_it_ends_on_how_it_was_computed(self, code_only: tuple[EvalHost, Report]) -> None:
        _, report = code_only
        last = report.blocks[-1]
        assert isinstance(last, DisclosureBlock) and last.source == "generation"
        assert report.source.bundle_fingerprint in last.text and "No model was called" in last.text

    def test_it_serializes_three_ways_and_the_schema_validates_it(self, code_only: tuple[EvalHost, Report]) -> None:
        _, report = code_only
        document = json.loads(report.to_canonical_json())
        jsonschema.Draft202012Validator(published_report_schema()).validate(document)
        assert Report.model_validate(document) == report

        markdown = report_markdown(report)
        # Titled by the campaign's name; the byline under it still names the campaign by its id.
        assert report.source.campaign_name
        assert markdown.startswith(f"# Campaign {report.source.campaign_name}: its evidence, with no analysis\n")
        assert f"Code-only report of campaign {report.source.campaign_id}" in markdown.splitlines()[2]
        # Said once, by the summary's line, and not again by the byline above it.
        assert "No analysis was generated" not in markdown.splitlines()[2]
        assert markdown.count("No analysis was generated") == 1 and "(blank headline)" not in markdown
        page = report_html(report)
        assert 'data-basis="code_only"' in page and 'data-chart-type="distribution"' in page


# =============================================================================
# The resolver: the campaign's newest unarchived analysis, else its evidence
# =============================================================================


async def test_a_campaign_with_an_analysis_is_reported_through_it(toy: tuple[EvalHost, EvalAnalysis, Report]) -> None:
    host, analysis, report = toy
    assert campaign_report(host, analysis.campaign_id, analysis.scope_id) == report
    assert report.basis == "analysis" and report.source.analysis_id == analysis.id


async def test_an_archived_analysis_is_not_the_campaigns_report(toy: tuple[EvalHost, EvalAnalysis, Report]) -> None:
    """An archive says the analysis was wrong, so the campaign falls back to its evidence — stated, not silent."""
    host, analysis, _ = toy
    set_analysis_archived(host.storage, analysis.id, analysis.scope_id, archived=True, reason="shown false")

    report = campaign_report(host, analysis.campaign_id, analysis.scope_id)
    assert report.basis == "code_only" and report.source.analysis_id is None
    # The archived analysis is still readable by its id.
    assert analysis_report(host.storage, analysis.id, analysis.scope_id).source.analysis_id == analysis.id


def test_the_resolver_refuses_a_campaign_the_scope_does_not_hold() -> None:
    host, campaign = toy_campaign_host()
    with pytest.raises(NotFoundError):
        campaign_report(host, "nope", campaign.scope_id)


def test_a_code_only_report_states_the_contrasts_code_tested_against_the_control() -> None:
    """With a control declared, the bundle's Holm-corrected contrasts are a table, each family's correction disclosed."""
    host, campaign = toy_campaign_host()
    set_campaign_control(
        host.storage, campaign.id, campaign.scope_id, campaign.run_ids[0], set_by="t", profile=host.profile
    )

    report = campaign_report(host, campaign.id, campaign.scope_id)
    (comparisons,) = [block for block in report.blocks if isinstance(block, TableBlock) and block.name == "comparisons"]
    # The question in the words it was asked, never its id.
    asked = next(q.text for q in campaign.declared_design.questions if q.id == "q-chunk-width")
    assert comparisons.rows and {row["question"] for row in comparisons.rows} == {asked}
    by_reading = {row["reading"]: row["verdict"] for row in comparisons.rows}
    assert by_reading["Field accuracy"] == "improved on the control"
    assert by_reading["Turn time"] == "regressed from the control"
    assert any(
        isinstance(block, DisclosureBlock) and block.source == "comparisons" and "Holm" in block.text
        for block in report.blocks
    )
    (arms,) = [block for block in report.blocks if isinstance(block, TableBlock) and block.name == "arms"]
    assert sum(str(row["arm"]).endswith("(control)") for row in arms.rows) == 1


# =============================================================================
# The comparative tables compile to charts that lead them (#643)
# =============================================================================


def _surface_order(report: Report) -> tuple[list[int], int]:
    """Where the surface's compiled charts sit, and where its table does, in block order."""
    charts = [
        index
        for index, block in enumerate(report.blocks)
        if isinstance(block, ChartBlock) and block.section == "surface"
    ]
    (table,) = [
        index for index, block in enumerate(report.blocks) if isinstance(block, TableBlock) and block.name == "surface"
    ]
    return charts, table


class TestTheComparativeTablesCompileToCharts:
    async def test_a_stored_analysis_draws_its_surface_before_the_surface_table(
        self, toy: tuple[Any, EvalAnalysis, Report]
    ) -> None:
        _, _, report = toy
        charts, table = _surface_order(report)
        assert charts, "the decision surface compiled to no chart"
        assert max(charts) < table, "the surface charts follow the table they draw"
        for index in charts:
            block = report.blocks[index]
            assert isinstance(block, ChartBlock) and block.intent is not None and len(block.intent.rows) >= 2
        markdown = report_markdown(report)
        assert markdown.index("**Chart: ") < markdown.index("**Decision surface**")
        page = report_html(report)
        assert page.index("data-chart-intent") < page.index("Decision surface")

    async def test_a_three_arm_code_only_report_draws_a_chart_before_the_surface_table(self) -> None:
        from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_NARROW, TOYHOST_WIDE, toyhost_campaign
        from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
        from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
        from packages.evals.tests.test_viz_timeseries import DAY_ONE, timeseries_batches
        from threetears.evals.analysis.bundle import assemble_context_bundle
        from threetears.evals.analysis.report.build import build_code_only_report

        host = toyhost_profile()
        campaign, _ = toyhost_campaign(profile=host)
        runs, results = timeseries_batches([(DAY_ONE, None)], profile=host, levels=(TOYHOST_NARROW, 512, TOYHOST_WIDE))
        bundle = assemble_context_bundle(
            campaign.model_copy(update={"run_ids": [run.id for run in runs]}),
            storage=ToyhostStorage(runs, results),
            profile=host,
        )
        assert len(bundle.cell_measures) == 3
        report = build_code_only_report(bundle, measures=host.measures, assembled_at="2026-10-10T00:00:00+00:00")

        charts, table = _surface_order(report)
        assert charts and max(charts) < table
        drawn = report.blocks[charts[-1]]
        assert isinstance(drawn, ChartBlock) and drawn.intent is not None
        assert len(drawn.intent.rows) == 3, "the chart compares all three arms"

    async def test_a_finding_without_a_drawn_chart_compiles_its_evidence_ahead_of_the_table(
        self, toy: tuple[Any, EvalAnalysis, Report]
    ) -> None:
        _, analysis, _ = toy
        resolution = analysis.resolutions[0]
        cited = {
            (row.measure_id, row.reading): {r.cell_ref for r in resolution.evidence if r.measure_id == row.measure_id}
            for row in resolution.evidence
        }
        assert any(len(cells) >= 2 for cells in cited.values()), "the toy evidence compares no arms"
        report = build_report(
            analysis.model_copy(update={"resolutions": [resolution.model_copy(update={"chart": None})]})
        )

        finding = [block for block in report.blocks if block.section == "findings" and block.finding == 0]
        kinds = [type(block).__name__ for block in finding]
        assert "ChartBlock" in kinds and kinds.index("ChartBlock") < kinds.index("TableBlock")

    async def test_a_finding_whose_authored_chart_drew_gets_no_second_chart(
        self, toy: tuple[Any, EvalAnalysis, Report]
    ) -> None:
        _, _, report = toy
        charts = [block for block in report.blocks if isinstance(block, ChartBlock) and block.section == "findings"]
        assert [block.viz_type for block in charts] == ["delta_table"]
