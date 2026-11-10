"""The ``Report``: the one page a campaign is read through, its two serializers, and the refusals its shape holds.

Driven end to end on the toy host: its corpus campaign is generated through the host (a fixtured writer standing in
for the model) and read back as a report through the service. Then:

- **It serializes two ways, from one block list.** Markdown and HTML each carry the title line, the basis line, the
  verdict, the scoreboard, code's numbers and every chart's values; the machine form is the bundle the page was
  computed from.
- **Every word on it is code's.** What code adds is a disclosure, the writer is named in the methods, and a block of
  an author's words is no block a page holds.
- **A campaign with no analysis is reported from its evidence**, and an archived analysis is not the campaign's
  report.
- **Every refusal fires** — a block linked to a finding, which no page holds, a table showing more rows than it has
  or a row keying other columns, a chart drawing the other kind's marks.

The HTML's no-script property has its own file, ``test_report_html_no_script.py``.
"""

from __future__ import annotations

import html
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import (
    NO_ANALYSIS,
    SECTION_TITLES,
    AnalysisContextBundle,
    aspect_view,
    ChartBlock,
    DisclosureBlock,
    IntervalRow,
    Report,
    TableBlock,
    TableCell,
    TableColumn,
    analysis_report,
    build_code_only_report,
    campaign_report,
    report_html,
    report_markdown,
    set_campaign_control,
)
from threetears.evals.analysis.report.build import contrast_rows
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
    async def test_markdown_carries_the_title_the_question_the_arms_and_the_writer_in_the_methods(
        self, toy: tuple[Any, EvalAnalysis, Report]
    ) -> None:
        _, analysis, report = toy
        markdown = report_markdown(report)
        lines = markdown.splitlines()

        # The heading is the campaign's name; the declared question is quoted whole on its
        # own line under it, then the arms and the aspect; who wrote the analysis is in the methods.
        assert report.source.campaign_name == "chunk width sweep"
        assert report.title == "chunk width sweep"
        # The title is the first line: nothing stands above it.
        top = lines.index(f"# {report.title}")
        assert top == 0
        assert lines[top + 1] == '"Is the wider chunk worth what it costs to retrieve?"', (
            "the question, under the title"
        )
        assert lines[top + 3].startswith(" vs ".join(sorted(["1024tok", "256tok"]))), "each arm once, then the date"
        assert f"| Analysis | {analysis.generation.generator_model}, prompt " in markdown
        assert "## Methods" in markdown

    async def test_html_carries_the_same_and_draws_the_chart(self, toy: tuple[Any, Any, Report]) -> None:
        _, _, report = toy
        page = report_html(report)

        assert page.startswith("<!doctype html>")
        assert page.index("<header>") < page.index(f"<h1>{_escaped(report.title)}</h1>") < page.index("</header>")
        header = page[page.index("<header>") + len("<header>") : page.index("</header>")]
        assert header.lstrip().startswith("<h1>") and 'role="alert"' not in page
        # The outcome bar is the page's one chart, beside its table.
        assert 'data-figure="outcome_bar"' in page and 'data-figure="interval_rows"' not in page
        assert page.count('data-kind="') == page.count('class="tbl chart-values"') == 1


def _escaped(text: str) -> str:
    return html.escape(text, quote=True)


# =============================================================================
# What the toy report holds, and who wrote it
# =============================================================================


class TestTheToyReportsContent:
    async def test_the_sections_are_in_reading_order(self, toy: tuple[Any, Any, Report]) -> None:
        _, _, report = toy
        order = list(SECTION_TITLES)
        seen = list(dict.fromkeys(block.section for block in report.blocks))
        assert seen == [section for section in order if section in seen]
        # No decision rule is declared, the glossary is the serializers', drawn from the blocks' terms, and code's page
        # lays out no writer's words.
        assert set(seen) == set(order) - {"decision", "glossary", "memo"}

    async def test_the_methods_name_the_analysis_s_writer_and_say_how_it_was_generated(
        self, toy: tuple[Any, EvalAnalysis, Report]
    ) -> None:
        _, analysis, report = toy
        generation = analysis.generation
        (table, *notes) = [block for block in report.blocks if block.section == "methods"]
        assert isinstance(table, TableBlock) and table.name == "methods"
        rows = {str(row["item"]): str(row["value"]) for row in table.values()}
        assert generation.generator_model in rows["Analysis"] and generation.prompt_id in rows["Analysis"]
        # The bundle served beside the page is the aspect's, so the page states its fingerprint, not the campaign's: in
        # a cell of its own, never in a sentence.
        assert analysis.aspects[0].bundle_fingerprint in rows["Fingerprint"]
        assert generation.bundle_fingerprint not in rows["Fingerprint"]
        # The toy corpus records no served model, so the methods say which model answered is unconfirmed.
        texts = [block.text for block in notes if isinstance(block, DisclosureBlock)]
        assert any(text.startswith("Model identity unconfirmed:") for text in texts)
        # After the table: the engine's disclosures, in their fold.
        assert {block.fold for block in notes} == {"Disclosures"}
        # The writer is never introduced: the page does not describe itself.
        assert not [b for b in report.blocks if isinstance(b, DisclosureBlock) and "Written by" in b.text]

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
    """A report that passes, with ``update`` applied over its fields."""
    fields: dict[str, Any] = {
        "title": "t",
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
        assert [block.kind for block in _report(bundle).blocks] == ["disclosure"]

    @pytest.mark.parametrize(
        "block",
        [
            {"kind": "disclosure", "section": "memo", "source": "memo", "text": "t", "finding": 1},
            {"kind": "disclosure", "section": "memo", "source": "memo", "text": "t", "rests_on": [3]},
        ],
        ids=["finding", "rests_on"],
    )
    def test_a_block_naming_a_finding_field_it_no_longer_has_is_refused(
        self, bundle: AnalysisContextBundle, block: dict[str, Any]
    ) -> None:
        with pytest.raises(ValidationError, match="extra_forbidden"):
            _report(bundle, blocks=[block])

    def test_a_table_showing_more_rows_than_it_has_is_refused(self) -> None:
        with pytest.raises(ValueError, match="shows 2 rows but says it has 1"):
            _table(rows=[table_row({"arm": "a"}), table_row({"arm": "b"})])

    def test_a_truncated_table_says_so(self, bundle: AnalysisContextBundle) -> None:
        markdown = report_markdown(_report(bundle, blocks=[_table(total_rows=4)]))
        assert "_1 of 4 rows shown._" in markdown

    def test_a_row_stating_a_value_no_column_shows_is_refused(self) -> None:
        with pytest.raises(ValueError, match="has rows that do not key exactly its columns: hidden"):
            _table(rows=[table_row({"arm": "a", "hidden": "one"})])

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

    @pytest.mark.parametrize("value", ["words", True, None])
    def test_a_unit_qualifies_only_a_number(self, value: Any) -> None:
        with pytest.raises(ValueError, match="qualifies a number"):
            TableCell(value=value, unit="s")


class TestAPageHoldsNoAuthorsWords:
    def test_a_block_of_an_authors_words_is_refused(self, bundle: AnalysisContextBundle) -> None:
        blocks = [{"kind": "text", "section": "memo", "role": "answer", "body": "words"}]
        with pytest.raises(ValueError, match="'text' found using 'kind' does not match any of the expected tags"):
            _report(bundle, blocks=blocks)


# =============================================================================
# A campaign with no analysis is reported from its evidence
# =============================================================================


@pytest.fixture
def code_only() -> tuple[EvalHost, Report]:
    """The campaign's one aspect's page, as the campaign's code-only page folds it."""
    host, campaign = toy_campaign_host()
    (fold,) = campaign_report(host, campaign.id, campaign.scope_id).aspects
    return host, fold.report


class TestACampaignWithNoAnalysisIsReportedFromItsEvidence:
    def test_it_is_code_only_and_its_memo_section_says_so(self, code_only: tuple[EvalHost, Report]) -> None:
        _, report = code_only
        (methods,) = [b for b in report.blocks if isinstance(b, TableBlock) and b.name == "methods"]
        assert {"item": "Analysis", "value": NO_ANALYSIS} in methods.values()
        assert "analysis" not in report.title

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
        (evidence,) = [row["value"] for row in methods.values() if row["item"] == "Evidence"]
        (fingerprint,) = [row["value"] for row in methods.values() if row["item"] == "Fingerprint"]
        assert report.source.bundle_fingerprint in str(fingerprint) and "its JSON beside this page" in str(evidence)

    def test_it_serializes_as_one_page_and_its_markdown(self, code_only: tuple[EvalHost, Report]) -> None:
        _, report = code_only
        markdown = report_markdown(report)
        assert report.source.campaign_name
        assert markdown.startswith(f"# {report.title}\n")
        assert markdown.count(NO_ANALYSIS) == 1
        page = report_html(report)
        assert 'data-basis="code_only"' in page and 'data-figure="outcome_bar"' in page

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
    """The campaign's newest analysis is its report, through either door: a memo, a retired writer's, is stood in for
    by code's page naming it (its own page is laid out by its id over what was frozen with it, not over the evidence
    now), and its aspect's page is code's, over the bundle frozen with it."""
    host, analysis, report = toy
    resolved = off_loop(campaign_report, host, analysis.campaign_id, analysis.scope_id)
    stored = off_loop(analysis_report, host.storage, analysis.id, analysis.scope_id)
    assert (resolved.basis, resolved.unread_analysis_id) == ("code_only", analysis.id)
    assert (stored.basis, stored.unread_analysis_id) == ("code_only", analysis.id)
    assert report.source.bundle_fingerprint == analysis.aspects[0].bundle_fingerprint


async def test_an_archived_analysis_is_not_the_campaigns_report(toy: tuple[EvalHost, EvalAnalysis, Report]) -> None:
    """An archive says the analysis was wrong, so the campaign falls back to its evidence — stated, not silent."""
    host, analysis, _ = toy
    off_loop(set_analysis_archived, host.storage, analysis.id, analysis.scope_id, archived=True, reason="shown false")

    report = off_loop(campaign_report, host, analysis.campaign_id, analysis.scope_id)
    assert report.basis == "code_only" and report.analysis_id is None
    assert report.unread_analysis_id is None, "the campaign has no report standing in for the archived analysis"
    # The archived analysis is still readable by its id.
    assert off_loop(analysis_report, host.storage, analysis.id, analysis.scope_id).unread_analysis_id == analysis.id


async def test_a_lever_no_comparison_exercised_is_said_in_words_never_by_its_path(
    toy: tuple[EvalHost, EvalAnalysis, Report],
) -> None:
    """A research model the runs never exercised is said in words under Not tested, never as its lever's path."""
    _, analysis, _ = toy
    lever = "persona.tool_configs.research.research_model"
    unexercised = {"factor": lever, "role": "research", "results": 1, "arms": ["candidate"]}
    (aspect,) = analysis.aspects
    frozen = analysis.model_copy(
        update={
            "aspects": [aspect.model_copy(update={"bundle": {**aspect.bundle, "unexercised_factors": [unexercised]}})]
        }
    )
    (held,) = frozen.aspects
    report = build_code_only_report(
        AnalysisContextBundle.from_dict(dict(held.bundle)), assembled_at=analysis.generation.bundle_assembled_at
    )
    markdown = report_markdown(report)
    assert lever not in markdown and "research model: one research call was made, on " in markdown


def test_the_resolver_refuses_a_campaign_the_scope_does_not_hold() -> None:
    host, campaign = toy_campaign_host()
    with pytest.raises(NotFoundError):
        campaign_report(host, "nope", campaign.scope_id)


def _controlled() -> tuple[Any, Report]:
    """The toy campaign with its control declared beside its question, before the runs.

    Not named afterwards through ``set_campaign_control``: a control chosen once the results exist is dated then,
    and the page names it as set after its readings were visible (``test_campaign_aspects.py``).
    """
    host, campaign = toy_campaign_host()
    named = set_campaign_control(
        host.storage, campaign.id, campaign.scope_id, campaign.run_ids[0], set_by="t", profile=host.profile
    )
    assert named.control is not None
    host.storage.save_campaign(named.model_copy(update={"arms": [named.control.model_copy(update={"named_at": None})]}))
    (fold,) = campaign_report(host, campaign.id, campaign.scope_id).aspects
    return campaign, fold.report


def test_a_code_only_report_states_the_contrasts_code_tested_against_the_control() -> None:
    """With a control declared, the bundle's Holm-corrected contrasts are a table, its family's correction disclosed."""
    campaign, report = _controlled()
    tables = {block.name: block for block in report.blocks if isinstance(block, TableBlock)}
    comparisons, statistics = tables["comparisons"], tables["comparison_statistics"]
    assert statistics.details and len(statistics.rows) == len(comparisons.rows)
    # The page names the question by its number; a program reads it in the words it was asked, never its id.
    declared = aspect_view(campaign).declared
    assert declared is not None
    asked = declared.question
    assert comparisons.rows and {row["question"] for row in statistics.values()} == {"Q1"}
    assert {row["question"] for row in contrast_rows(report.bundle)} == {asked}
    by_reading = {row["reading"]: row["verdict"] for row in comparisons.values()}
    # The campaign's page folds this page, and names each arm by the campaign's label for it.
    assert by_reading["Field accuracy"] == "1024tok better than control"
    assert by_reading["Turn time"] == "1024tok worse than control (lower is better)"
    assert any(
        isinstance(block, DisclosureBlock) and block.source == "comparisons" and "Holm" in block.text
        for block in report.blocks
    )
    (scoreboard,) = [b for b in report.blocks if isinstance(b, TableBlock) and b.name == "scoreboard"]
    assert sum(str(row["arm"]).startswith("▸ ") for row in scoreboard.values()) == 1
    # Each reading tested on enough cases carries the interval on its change in its own row, read against zero;
    # no chart of the differences stands apart from the table.
    changes = {str(row["reading"].value): row["change"] for row in comparisons.rows}
    assert {reading for reading, cell in changes.items() if cell.low is not None} >= {"Field accuracy", "Turn time"}
    assert all(cell.zero_line for cell in changes.values() if cell.low is not None)
    assert not [block for block in report.blocks if isinstance(block, ChartBlock) and block.section == "contrasts"]


def test_a_contrast_states_its_figures_on_the_measure_s_one_ruler() -> None:
    """One figure, one ruler, with its unit: a latency the surface states in s is never ``1420`` beside it."""
    _, report = _controlled()
    (statistics,) = [b for b in report.blocks if isinstance(b, TableBlock) and b.name == "comparison_statistics"]
    (turn_time,) = [row for row in statistics.rows if row["reading"].value == "Turn time"]
    assert [turn_time[key].text for key in ("control_mean", "arm_mean")] == ["0.92 s", "1.42 s"]
    assert (turn_time["arm_mean"].value, turn_time["arm_mean"].unit) == (1.42, "s")
    (comparisons,) = [b for b in report.blocks if isinstance(b, TableBlock) and b.name == "comparisons"]
    (shown,) = [row for row in comparisons.rows if row["reading"].value == "Turn time"]
    # What every interval guarantees is said once, in the column's header, never in each cell: the family's joint 95%
    # over its six members, the question's pass rate among them (owner, 2026-11-06); each interval's own 1 − α/6 is the
    # family's disclosure's and the glossary's (owner ruling, level in headings).
    # A change reads signed, and it and its interval's ends at two significant figures of the set, as a count does.
    assert (shown["change"].text, shown["interval"].text) == ("+0.50 s", "+0.45 to +0.55 s")
    assert (
        next(column.header for column in comparisons.columns if column.key == "interval")
        == "95% interval (adjusted for 6 comparisons)"
    )
    # A program's row has no header, so its interval says its level.
    (row,) = [row for row in contrast_rows(report.bundle) if row["reading"] == "Turn time"]
    assert row["change"] == "+0.50 s (+0.45 to +0.55 s at 99.2%)"
    # A share on 0–1 is stated as a percent, its cell restated onto that ruler, at the tenth its interval sets.
    (accuracy,) = [row for row in statistics.rows if row["reading"].value == "Field accuracy"]
    assert (accuracy["arm_mean"].text, accuracy["arm_mean"].unit) == ("91.0%", "%")
