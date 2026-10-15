"""The ``Report``: the one page a campaign is read through, its two serializers, and the refusals its shape holds.

Driven end to end on the toy host: its corpus campaign is generated through the host (a fixtured writer standing in
for the model) and read back as a report through the service. Then:

- **It serializes two ways, from one block list.** Markdown and HTML each carry the title line, the basis line, the
  verdict, the scoreboard, the author's words, code's numbers and every chart's values; the machine form is the
  bundle the page was computed from.
- **Who wrote what stays legible.** An author's caveat is attached to its finding as written; what code adds is a
  disclosure; the memo is labelled as written by its model, and every number on the page is code's.
- **A campaign with no analysis is reported from its evidence**, with no author's words at all, and an archived
  analysis is not the campaign's report.
- **Every refusal fires** — a block linked to a finding the report does not hold, a table showing more rows than it
  has or a row keying other columns, a chart drawing the other kind's marks, a finding's words that name no finding,
  a basis the report disagrees with.

The HTML's no-script property has its own file, ``test_report_html_no_script.py``.
"""

from __future__ import annotations

import html
from typing import Any

import pytest

from threetears.evals.analysis import (
    NO_ANALYSIS,
    SECTION_TITLES,
    AnalysisContextBundle,
    ChartBlock,
    DisclosureBlock,
    IntervalRow,
    Report,
    TableBlock,
    TableCell,
    TableColumn,
    TextBlock,
    analysis_report,
    build_code_only_report,
    campaign_report,
    report_html,
    report_markdown,
    set_campaign_control,
)
from threetears.evals.analysis.report.build import contrast_rows
from threetears.evals.analysis.report.figures import measure_rulers
from threetears.evals.analysis.report.model import table_row
from threetears.evals.kernel.campaign import EvalAnalysis
from threetears.evals.kernel.errors import NotFoundError
from threetears.evals.kernel.host import EvalHost
from threetears.evals.run import set_analysis_archived
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle
from packages.evals.tests.report_support import toy_campaign_host, toy_report
from packages.evals.tests.off_loop import off_loop


@pytest.fixture
async def toy() -> tuple[EvalHost, EvalAnalysis, Report]:
    return await toy_report()


# =============================================================================
# The toy host's analysis, laid out and serialized
# =============================================================================


class TestTheToyAnalysisSerializesTwoWays:
    async def test_markdown_carries_the_title_the_basis_the_words_the_numbers_and_the_chart_values(
        self, toy: tuple[Any, EvalAnalysis, Report]
    ) -> None:
        _, analysis, report = toy
        markdown = report_markdown(report)
        finding = analysis.document.findings[0]
        lines = markdown.splitlines()

        # The declared question is the title; the arms are line 2; who wrote the memo is in the methods.
        assert report.source.campaign_name == "chunk width sweep"
        # The question as declared, its first letter capitalized: a title starts with a capital.
        assert report.title == "Is the wider chunk worth what it costs to retrieve?"
        assert lines[0] == f"# {report.title}"
        assert lines[1].startswith(" vs ".join(sorted(["1024tok", "256tok"]))), "each arm once, then the date"
        assert f"| Analysis | {analysis.generation.generator_model}, prompt " in markdown
        # A finding leads with its headline, unnumbered: nothing on the page cites a finding by number.
        assert f"**{finding.title}**" in markdown and "**F1." not in markdown
        assert "**Readings**" in markdown and "| Arm | Turn time | Cases |" in markdown
        (chart,) = _finding_charts(report)
        assert f"**{chart.title}**" in markdown
        assert "| " + " | ".join(chart.values_table()[0]) + " |" in markdown
        assert f"Caveat (sampling): {finding.caveats[0].text}" in markdown
        assert "## Methods" in markdown

    async def test_html_carries_the_same_and_draws_each_chart_inline(self, toy: tuple[Any, Any, Report]) -> None:
        _, _, report = toy
        page = report_html(report)
        (chart,) = _finding_charts(report)

        assert page.startswith("<!doctype html>")
        assert page.index("<header>") < page.index(f"<h1>{_escaped(report.title)}</h1>") < page.index("</header>")
        assert 'data-chart="interval_rows"' in page and 'data-chart="outcome_bar"' in page
        assert f"<caption>{_escaped(chart.title)}</caption>" in page
        assert page.count("<svg") == page.count('<table class="visually-hidden">')


def _escaped(text: str) -> str:
    return html.escape(text, quote=True)


def _finding_charts(report: Report) -> list[ChartBlock]:
    return [block for block in report.blocks if isinstance(block, ChartBlock) and block.section == "memo"]


# =============================================================================
# What the toy report holds, and who wrote it
# =============================================================================


class TestTheToyReportsContent:
    async def test_the_sections_are_in_reading_order(self, toy: tuple[Any, Any, Report]) -> None:
        _, _, report = toy
        order = list(SECTION_TITLES)
        seen = list(dict.fromkeys(block.section for block in report.blocks))
        assert seen == [section for section in order if section in seen]
        # No decision rule is declared, and the glossary is the serializers', drawn from the blocks' terms.
        assert set(seen) == set(order) - {"decision", "glossary"}

    async def test_an_authors_caveat_is_attached_to_its_finding_as_written(
        self, toy: tuple[Any, EvalAnalysis, Report]
    ) -> None:
        _, analysis, report = toy
        (caveat,) = [block for block in report.blocks if isinstance(block, TextBlock) and block.role == "caveat"]
        written = analysis.document.findings[0].caveats[0]

        assert (caveat.section, caveat.finding, caveat.body) == ("memo", 0, written.text)
        assert [(fact.name, fact.value) for fact in caveat.facts] == [("Kind", written.kind)]

    async def test_the_evidence_table_is_codes_numbers_at_the_finding(
        self, toy: tuple[Any, EvalAnalysis, Report]
    ) -> None:
        _, analysis, report = toy
        (table,) = [block for block in report.blocks if isinstance(block, TableBlock) and block.name == "evidence"]
        evidence = analysis.resolutions[0].evidence

        assert table.finding == 0
        # Code's numbers, stated on each measure's one ruler for the report (a latency in s where the surface is).
        rulers = measure_rulers(analysis.decision_surface)
        stated = [row.value * (rulers[row.measure_id].factor if row.reading == "measure" else 1.0) for row in evidence]
        # One row per arm, one column per reading: the toy's finding reads one measure on each arm.
        assert [row["reading_0"] for row in table.values()] == pytest.approx(stated)
        assert any(cell["reading_0"].unit == "s" for cell in table.rows), (
            "the toy's turn time reads in s on its surface"
        )
        assert [row["cases"] for row in table.values()] == [row.n_cases for row in evidence]

    async def test_a_finding_s_evidence_is_drawn_as_interval_rows_on_the_same_ruler(
        self, toy: tuple[Any, EvalAnalysis, Report]
    ) -> None:
        _, analysis, report = toy
        (chart,) = _finding_charts(report)
        (table,) = [block for block in report.blocks if isinstance(block, TableBlock) and block.name == "evidence"]

        assert chart.finding == 0 and chart.chart == "interval_rows" and chart.unit == "s"
        assert [row.value for row in chart.rows] == pytest.approx([row["reading_0"] for row in table.values()])
        assert [row.n_cases for row in chart.rows] == [row.n_cases for row in analysis.resolutions[0].evidence]
        # The chart follows the evidence table it draws, inside its finding.
        assert report.blocks.index(table) < report.blocks.index(chart)

    async def test_the_recommendation_rests_on_its_finding(self, toy: tuple[Any, Any, Report]) -> None:
        _, _, report = toy
        (decision,) = [b for b in report.blocks if isinstance(b, TextBlock) and b.role == "recommendation"]
        assert decision.rests_on == [0]
        assert ("Disposition", "deferred") in [(fact.name, fact.value) for fact in decision.facts]

    async def test_the_memo_is_labelled_as_its_writer_s_and_the_methods_say_how_it_was_generated(
        self, toy: tuple[Any, EvalAnalysis, Report]
    ) -> None:
        _, analysis, report = toy
        generation = analysis.generation
        (table, *notes) = [block for block in report.blocks if block.section == "methods"]
        assert isinstance(table, TableBlock) and table.name == "methods"
        rows = {str(row["item"]): str(row["value"]) for row in table.values()}
        assert generation.generator_model in rows["Analysis"] and generation.prompt_id in rows["Analysis"]
        assert generation.bundle_fingerprint in rows["Bundle"]
        # The toy corpus records no served model, so the methods say which model answered is unconfirmed.
        texts = [block.text for block in notes if isinstance(block, DisclosureBlock)]
        assert any(text.startswith("Model identity unconfirmed:") for text in texts)
        assert all(block.fold == "Disclosures" for block in notes)
        # The memo is never introduced: the page does not describe itself.
        assert not [b for b in report.blocks if isinstance(b, DisclosureBlock) and "Written by" in b.text]

    async def test_the_coverage_map_is_joined_to_the_findings_and_counted(self, toy: tuple[Any, Any, Report]) -> None:
        _, _, report = toy
        (coverage,) = [block for block in report.blocks if isinstance(block, TableBlock) and block.name == "coverage"]
        (headline,) = [b.body for b in report.blocks if isinstance(b, TextBlock) and b.role == "finding_title"]
        # A finding is cited by the headline the page prints for it, never by a number.
        assert coverage.section == "memo" and coverage.values()[0]["findings"] == headline.rstrip(".")
        unnamed = [row for row in coverage.values() if row["findings"] == "no finding"]
        said = [b for b in report.blocks if isinstance(b, DisclosureBlock) and b.text.startswith("Levers no finding")]
        assert len(said) == (1 if unnamed else 0)

    async def test_the_service_refuses_an_analysis_it_does_not_hold(self, toy: tuple[EvalHost, Any, Any]) -> None:
        host, analysis, _ = toy
        with pytest.raises(NotFoundError):
            off_loop(analysis_report, host.storage, "no-such-analysis", analysis.scope_id)


# =============================================================================
# The report's own refusals
# =============================================================================


@pytest.fixture(scope="module")
def bundle() -> AnalysisContextBundle:
    return toyhost_bundle()


def _report(bundle: AnalysisContextBundle, **update: Any) -> Report:
    """A one-finding analysis report that passes, with ``update`` applied over its fields."""
    fields: dict[str, Any] = {
        "basis": "analysis",
        "title": "t",
        "finding_count": 1,
        "source": {
            "analysis_id": "a",
            "campaign_id": "c",
            "scope_id": "s",
            "behavior": "b",
            "generated_at": "2026-10-05T00:00:00+00:00",
            "generator_model": "m",
            "bundle_fingerprint": "f",
        },
        "blocks": [{"kind": "text", "section": "memo", "role": "finding_title", "finding": 0, "body": "t"}],
        "bundle": bundle,
    }
    return Report.model_validate(fields | update)


def _code_only(bundle: AnalysisContextBundle, **update: Any) -> Report:
    """A code-only report that passes, with ``update`` applied over its fields."""
    fields: dict[str, Any] = {
        "basis": "code_only",
        "title": "t",
        "finding_count": 0,
        "source": {
            "campaign_id": "c",
            "scope_id": "s",
            "behavior": "b",
            "generated_at": "2026-10-05T00:00:00+00:00",
            "bundle_fingerprint": "f",
        },
        "blocks": [{"kind": "disclosure", "section": "memo", "source": "memo", "text": "no analysis"}],
        "bundle": bundle,
    }
    return Report.model_validate(fields | update)


def _table(**update: Any) -> TableBlock:
    fields: dict[str, Any] = {
        "section": "contrasts",
        "name": "arms",
        "title": "Arms",
        "columns": [TableColumn(key="arm", header="Arm")],
        "rows": [table_row({"arm": "a"})],
        "order": "as given",
        "total_rows": 1,
    }
    return TableBlock(**(fields | update))


class TestTheReportsRefusals:
    def test_the_fixture_passes(self, bundle: AnalysisContextBundle) -> None:
        assert _report(bundle).finding_count == 1

    @pytest.mark.parametrize(
        "block",
        [
            {"kind": "text", "section": "memo", "role": "finding_body", "finding": 1, "body": "t"},
            {"kind": "disclosure", "section": "memo", "source": "memo", "text": "t", "rests_on": [3]},
        ],
        ids=["finding", "rests_on"],
    )
    def test_a_block_linked_to_a_finding_the_report_does_not_hold_is_refused(
        self, bundle: AnalysisContextBundle, block: dict[str, Any]
    ) -> None:
        with pytest.raises(ValueError, match=r"is linked to finding position\(s\) \[\d\], and the report holds 1"):
            _report(bundle, blocks=[block])

    def test_a_table_showing_more_rows_than_it_has_is_refused(self) -> None:
        with pytest.raises(ValueError, match="shows 2 rows but says it has 1"):
            _table(rows=[table_row({"arm": "a"}), table_row({"arm": "b"})])

    def test_a_truncated_table_says_so(self, bundle: AnalysisContextBundle) -> None:
        markdown = report_markdown(_report(bundle, blocks=[_table(total_rows=4)], finding_count=0))
        assert "_1 of 4 rows shown._" in markdown

    def test_a_row_stating_a_value_no_column_shows_is_refused(self) -> None:
        with pytest.raises(ValueError, match="has rows that do not key exactly its columns: hidden"):
            _table(rows=[table_row({"arm": "a", "hidden": 1})])

    def test_a_row_missing_a_column_is_refused(self) -> None:
        """A row keys every column, so a client prints each row as served and never invents a missing cell's text."""
        with pytest.raises(ValueError, match="has rows that do not key exactly its columns: levers"):
            _table(columns=[TableColumn(key="arm", header="Arm"), TableColumn(key="levers", header="Levers")])

    def test_an_outcome_bar_with_no_bar_is_refused(self) -> None:
        with pytest.raises(ValueError, match="an outcome bar draws one bar per arm and no interval row"):
            ChartBlock(section="scoreboard", chart="outcome_bar", title="t")

    def test_an_interval_chart_with_no_row_is_refused(self) -> None:
        with pytest.raises(ValueError, match="an interval chart draws at least one row and no outcome bar"):
            ChartBlock(section="contrasts", chart="interval_rows", title="t")

    def test_an_interval_that_runs_backwards_is_refused(self) -> None:
        with pytest.raises(ValueError, match="its interval runs from 2.0 down to 1.0"):
            IntervalRow(label="a", value=1.5, low=2.0, high=1.0, n_cases=5)

    @pytest.mark.parametrize("role", ["finding_title", "finding_body", "caveat", "carried_forward"])
    def test_a_findings_own_words_that_name_no_finding_are_refused(self, role: str) -> None:
        with pytest.raises(ValueError, match=f"a {role} block belongs to a finding and names none"):
            TextBlock(section="memo", role=role, body="t")  # type: ignore[arg-type]

    def test_a_recommendation_belongs_to_no_finding(self) -> None:
        assert TextBlock(section="memo", role="recommendation", body="t").finding is None

    @pytest.mark.parametrize("value", ["words", True, None])
    def test_a_unit_qualifies_only_a_number(self, value: Any) -> None:
        with pytest.raises(ValueError, match="qualifies a number"):
            TableCell(value=value, unit="s")


class TestTheBasisIsRefusedWhenTheReportDisagreesWithIt:
    def test_the_code_only_fixture_passes(self, bundle: AnalysisContextBundle) -> None:
        assert _code_only(bundle).basis == "code_only"

    @pytest.mark.parametrize("missing", ["analysis_id", "generator_model"])
    def test_an_analysis_report_names_its_analysis_and_its_writer(
        self, bundle: AnalysisContextBundle, missing: str
    ) -> None:
        source = _report(bundle).source.model_dump() | {missing: None}
        with pytest.raises(ValueError, match="an analysis report names its analysis_id and generator_model"):
            _report(bundle, source=source)

    @pytest.mark.parametrize("named", [{"analysis_id": "a"}, {"generator_model": "m"}])
    def test_a_code_only_report_names_no_analysis(self, bundle: AnalysisContextBundle, named: dict[str, str]) -> None:
        source = _code_only(bundle).source.model_dump() | named
        with pytest.raises(ValueError, match="a code-only report renders no analysis"):
            _code_only(bundle, source=source)

    def test_a_code_only_report_counts_no_findings(self, bundle: AnalysisContextBundle) -> None:
        with pytest.raises(ValueError, match="has no findings, and this one counts 1"):
            _code_only(bundle, finding_count=1)

    def test_a_code_only_report_holds_no_authors_words(self, bundle: AnalysisContextBundle) -> None:
        blocks = [{"kind": "text", "section": "memo", "role": "answer", "body": "words"}]
        with pytest.raises(ValueError, match=r"blocks \[0\] are text blocks"):
            _code_only(bundle, blocks=blocks)


# =============================================================================
# A campaign with no analysis is reported from its evidence
# =============================================================================


@pytest.fixture
def code_only() -> tuple[EvalHost, Report]:
    host, campaign = toy_campaign_host()
    return host, campaign_report(host, campaign.id, campaign.scope_id)


class TestACampaignWithNoAnalysisIsReportedFromItsEvidence:
    def test_it_is_code_only_and_its_memo_section_says_so(self, code_only: tuple[EvalHost, Report]) -> None:
        _, report = code_only
        assert report.basis == "code_only" and report.finding_count == 0
        (methods,) = [b for b in report.blocks if isinstance(b, TableBlock) and b.name == "methods"]
        assert {"item": "Analysis", "value": NO_ANALYSIS} in methods.values()
        assert "analysis" not in report.title

    def test_it_holds_no_authors_words(self, code_only: tuple[EvalHost, Report]) -> None:
        _, report = code_only
        assert not [block for block in report.blocks if isinstance(block, TextBlock)]

    def test_it_counts_every_arm_s_trials_on_its_first_screen(self, code_only: tuple[EvalHost, Report]) -> None:
        _, report = code_only
        (scoreboard,) = [b for b in report.blocks if isinstance(b, TableBlock) and b.name == "scoreboard"]
        assert len(scoreboard.rows) == len(report.bundle.scoreboard) == 2
        assert sum(row["ran"] for row in scoreboard.values()) == len(report.bundle.trials)
        charts = [block for block in report.blocks if isinstance(block, ChartBlock) and block.section == "scoreboard"]
        assert [chart.chart for chart in charts] == ["outcome_bar"]

    def test_it_ends_on_how_it_was_computed(self, code_only: tuple[EvalHost, Report]) -> None:
        _, report = code_only
        (methods, *notes) = [block for block in report.blocks if block.section == "methods"]
        assert isinstance(methods, TableBlock) and all(block.fold == "Disclosures" for block in notes)
        (bundle,) = [row["value"] for row in methods.values() if row["item"] == "Bundle"]
        assert report.source.bundle_fingerprint in str(bundle) and "bundle.json beside this page" in str(bundle)

    def test_it_serializes_as_one_page_and_its_markdown(self, code_only: tuple[EvalHost, Report]) -> None:
        _, report = code_only
        markdown = report_markdown(report)
        assert report.source.campaign_name
        assert markdown.startswith(f"# {report.title}\n")
        assert markdown.count(NO_ANALYSIS) == 1
        page = report_html(report)
        assert 'data-basis="code_only"' in page and 'data-chart="outcome_bar"' in page

    def test_a_time_axis_withheld_says_why(self, code_only: tuple[EvalHost, Report]) -> None:
        _, report = code_only
        withheld = "every run shared one build and one UTC day"
        rebuilt = build_code_only_report(
            report.bundle.model_copy(update={"time_axis_withheld": withheld}),
            assembled_at="2026-10-10T00:00:00+00:00",
        )
        assert any(
            isinstance(block, DisclosureBlock) and block.section == "methods" and block.text == withheld
            for block in rebuilt.blocks
        )


# =============================================================================
# The resolver: the campaign's newest unarchived analysis, else its evidence
# =============================================================================


async def test_a_campaign_with_an_analysis_is_reported_through_it(toy: tuple[EvalHost, EvalAnalysis, Report]) -> None:
    host, analysis, report = toy
    resolved = off_loop(campaign_report, host, analysis.campaign_id, analysis.scope_id)
    assert report_html(resolved) == report_html(report)
    assert report.basis == "analysis" and report.source.analysis_id == analysis.id


async def test_an_archived_analysis_is_not_the_campaigns_report(toy: tuple[EvalHost, EvalAnalysis, Report]) -> None:
    """An archive says the analysis was wrong, so the campaign falls back to its evidence — stated, not silent."""
    host, analysis, _ = toy
    off_loop(set_analysis_archived, host.storage, analysis.id, analysis.scope_id, archived=True, reason="shown false")

    report = off_loop(campaign_report, host, analysis.campaign_id, analysis.scope_id)
    assert report.basis == "code_only" and report.source.analysis_id is None
    # The archived analysis is still readable by its id.
    assert off_loop(analysis_report, host.storage, analysis.id, analysis.scope_id).source.analysis_id == analysis.id


def test_the_resolver_refuses_a_campaign_the_scope_does_not_hold() -> None:
    host, campaign = toy_campaign_host()
    with pytest.raises(NotFoundError):
        campaign_report(host, "nope", campaign.scope_id)


def _controlled() -> tuple[Any, Report]:
    host, campaign = toy_campaign_host()
    set_campaign_control(
        host.storage, campaign.id, campaign.scope_id, campaign.run_ids[0], set_by="t", profile=host.profile
    )
    return campaign, campaign_report(host, campaign.id, campaign.scope_id)


def test_a_code_only_report_states_the_contrasts_code_tested_against_the_control() -> None:
    """With a control declared, the bundle's Holm-corrected contrasts are a table, its family's correction disclosed."""
    campaign, report = _controlled()
    tables = {block.name: block for block in report.blocks if isinstance(block, TableBlock)}
    comparisons, statistics = tables["comparisons"], tables["comparison_statistics"]
    assert statistics.details and len(statistics.rows) == len(comparisons.rows)
    # The page names the question by its number; a program reads it in the words it was asked, never its id.
    asked = next(q.text for q in campaign.declared_design.questions if q.id == "q-chunk-width")
    assert comparisons.rows and {row["question"] for row in statistics.values()} == {"Q1"}
    assert {row["question"] for row in contrast_rows(report.bundle)} == {asked}
    by_reading = {row["reading"]: row["verdict"] for row in comparisons.values()}
    assert by_reading["Field accuracy"] == "improved on the control"
    assert by_reading["Turn time"] == "regressed from the control"
    assert any(
        isinstance(block, DisclosureBlock) and block.source == "comparisons" and "Holm" in block.text
        for block in report.blocks
    )
    (scoreboard,) = [b for b in report.blocks if isinstance(b, TableBlock) and b.name == "scoreboard"]
    assert sum(str(row["arm"]).startswith("▸ ") for row in scoreboard.values()) == 1
    # Each reading tested on enough cases is drawn as interval rows of the difference, a zero line beneath it.
    drawn = [block for block in report.blocks if isinstance(block, ChartBlock) and block.section == "contrasts"]
    assert {block.title.removeprefix("Change in ") for block in drawn} >= {"Field accuracy", "Turn time"}
    assert all(block.zero_line and block.chart == "interval_rows" for block in drawn)


def test_a_contrast_states_its_figures_on_the_measure_s_one_ruler() -> None:
    """One figure, one ruler, with its unit: a latency the surface states in s is never ``1420`` beside it."""
    _, report = _controlled()
    (statistics,) = [b for b in report.blocks if isinstance(b, TableBlock) and b.name == "comparison_statistics"]
    (turn_time,) = [row for row in statistics.rows if row["reading"].value == "Turn time"]
    assert [turn_time[key].text for key in ("control_mean", "arm_mean")] == ["0.92 s", "1.42 s"]
    assert (turn_time["arm_mean"].value, turn_time["arm_mean"].unit) == (1.42, "s")
    (comparisons,) = [b for b in report.blocks if isinstance(b, TableBlock) and b.name == "comparisons"]
    (shown,) = [row for row in comparisons.rows if row["reading"].value == "Turn time"]
    # The level every interval holds at is said once, in the column's header, never in each cell.
    assert (shown["change"].text, shown["interval"].text) == ("0.5 s", "[0.4498, 0.5502] s")
    assert next(column.header for column in comparisons.columns if column.key == "interval") == "Interval (99%)"
    # A program's row has no header, so its interval says its level.
    (row,) = [row for row in contrast_rows(report.bundle) if row["reading"] == "Turn time"]
    assert row["change"] == "0.5 s [0.4498, 0.5502] s at 99%"
    # A share on 0–1 is stated as a percent, its cell restated onto that ruler.
    (accuracy,) = [row for row in statistics.rows if row["reading"].value == "Field accuracy"]
    assert (accuracy["arm_mean"].text, accuracy["arm_mean"].unit) == ("91%", "%")
