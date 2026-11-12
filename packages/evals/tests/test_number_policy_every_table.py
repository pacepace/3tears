"""The number policy held on every table that states a measure, as a production page read them apart.

A host's fraction (``value_range=(0, 1)``, no unit) read ``0.3`` on a measurement's scorecard while its aspect's page
read ``30%`` (rule 4: a rate carries its "%"); a measure read ``0.10`` in the scorecard's matrix and ``0.1`` in the
fold under it (rule 1: one precision on the page); a chart of two arms compared on a corrected contrast headed its
interval ``99.69%`` where every other heading states the joint guarantee; a tooltip read ``85% (80 to 89%)``, its
unit twice, and a glyph's title dropped the sign its cell shows (rule 3); and a mean's spread read ``$3.33 ±
$0.088``, finer than the mean beside it (rule 1).

Rule 3, as the coordinator ruled it for every surface: a figure with its interval carries the unit on the figure
alone (``91% (81 to 96)``), and an interval standing alone carries it once, after its high end (``73 to 92%``).
"""

from __future__ import annotations

import copy
import html
import re

from packages.evals.tests.campaign_bundle_support import arms_of, campaign_bundle
from packages.evals.tests.campaign_page_support import kind_page_of
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.report_doc_support import converted_doc, snapshot_of
from threetears.evals import Eval
from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle, campaign_report_markdown
from threetears.evals.analysis.campaign_readings import MeasureReading
from threetears.evals.analysis.report.blocks import TableBlock, TableCell
from threetears.evals.analysis.report.charts import BarRow, BarsChart
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.markdown import doc_page_markdown
from threetears.evals.analysis.report.figure_keys import format_figure, resolve
from threetears.evals.analysis.report.figures import SHARE, UNITLESS, reply_cost_ruler
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.html_parts import grid_html
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.kernel.metrics import MetricDescriptor
from threetears.evals.kernel.report_doc import ReportDoc
from threetears.evals.quick import measure

#: The host measure the variants add: a share of research deliveries, as a host declares one.
_FRACTION = "research_failed_rate"


def _with_measure(
    bundle: AnalysisContextBundle,
    *,
    value_range: tuple[float, float] | None,
    mean: float,
    interval: tuple[float, float],
) -> AnalysisContextBundle:
    """The bundle with one more host measure on every cell, declared with no unit on ``value_range``, at ``mean`` and
    ``interval``: its summary is the cell's cost summary renamed, so it is shaped as a numeric summary is."""
    described = MetricDescriptor(
        name=_FRACTION,
        reader_name="Failed research rate",
        data_type="numeric",
        family="research_conclusion",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        population="all_observed",
        unit=None,
        description="Share of the research deliveries that produced nothing usable.",
        higher_is_better=False,
        value_range=value_range,
        merit_axis="reliability",
    )
    cells = []
    for cell in bundle.cell_measures:
        held = cell.measures.measures
        (cost,) = [summary for summary in held if summary.name == "cost_usd"]
        added = cost.model_copy(
            update={
                "name": _FRACTION,
                "mean": mean,
                "ci_low": interval[0],
                "ci_high": interval[1],
                "higher_is_better": False,
                "population": "all_observed",
            }
        )
        cells.append(
            cell.model_copy(update={"measures": cell.measures.model_copy(update={"measures": [*held, added]})})
        )
    return bundle.model_copy(
        update={"measure_catalog": {**bundle.measure_catalog, _FRACTION: described}, "cell_measures": cells}
    )


async def _page_rows(
    *, value_range: tuple[float, float] | None, mean: float, interval: tuple[float, float]
) -> list[str]:
    """The measurement page's rows for the added measure, as Markdown: its scorecard's matrix and fold, then its
    aspect's All cases."""
    page = await kind_page_of(
        "solo", bundle=lambda bundle: _with_measure(bundle, value_range=value_range, mean=mean, interval=interval)
    )
    return [line for line in campaign_report_markdown(page).splitlines() if "Failed research rate" in line]


async def test_a_host_fraction_reads_as_a_rate_on_the_scorecard() -> None:
    """A fraction on 0 to 1 that declares no unit is a share: the scorecard's matrix and its fold read it in percent,
    its interval with the "%" once, as its aspect's page and the written page read it."""
    matrix, fold, *_ = await _page_rows(value_range=(0.0, 1.0), mean=0.3, interval=(0.0, 0.7))
    assert matrix == "| solo | Failed research rate | 30% | 0 to 70% | 20 |"
    assert fold == "| solo | alpha | Failed research rate | 30% | 0 to 70% | 20 |"


async def test_a_measure_reads_at_one_precision_in_the_matrix_and_its_fold() -> None:
    """A measure in its own unit reads at its measure's one precision in the matrix and the fold alike: ``0.10``
    in both, never ``0.1`` in the fold."""
    matrix, fold, *_ = await _page_rows(value_range=None, mean=0.1, interval=(0.0, 0.38))
    assert matrix == "| solo | Failed research rate | 0.10 | 0 to 0.38 | 20 |"
    assert fold == "| solo | alpha | Failed research rate | 0.10 | 0 to 0.38 | 20 |"


#: Kairo's written page: the third prototype's document on its own snapshot.
_KAIRO = "prototype3/kairo"


def test_two_arms_compared_on_a_corrected_contrast_head_their_interval_with_its_joint_guarantee() -> None:
    """A chart of two arms compared overall reads its row on the paired contrast's corrected interval: its head and
    its values table state the family's joint guarantee, never the per-interval ``99.69%``."""
    doc = copy.deepcopy(converted_doc(_KAIRO))
    evidence = doc["claims"][3]["evidence"][1]
    evidence["query"].update(
        {
            "measures": ["production_replicating_cost"],
            "arms": ["qwen3.8-flash", "September"],
            "strata": ["overall"],
            "transform": "per_1k",
        }
    )
    evidence["caption"] = "Two arms' cost, compared."
    held = ReportInput.model_validate(snapshot_of(_KAIRO).report_input)
    lines = doc_page_markdown(build(ReportDoc.model_validate(doc), held, strict=True)).splitlines()
    at = lines.index("Two arms' cost, compared.")
    head = next(line for line in reversed(lines[:at]) if line.startswith("| Row |"))
    assert head == "| Row | Value | 95% interval (adjusted for 16 comparisons) |"
    assert not any("99.69% interval" in line for line in lines)


async def test_rule_3_on_a_page_table_an_interval_alone_ends_on_its_unit_and_beside_its_figure_carries_none() -> None:
    """A page table: the scorecard's interval column states the interval alone, its "%" once after its high end
    (``0 to 70%``); the aspect's All cases writes it beside its figure, which carries the "%" (``30% (0 to 70)``)."""
    matrix, _fold, cases = await _page_rows(value_range=(0.0, 1.0), mean=0.3, interval=(0.0, 0.7))
    assert matrix.endswith("| 30% | 0 to 70% | 20 |")
    assert cases == "| Failed research rate | 30% (0 to 70) | 20 |"


def test_rule_3_in_a_prose_key_an_interval_alone_ends_on_its_unit() -> None:
    """A prose key: an interval cited alone writes its unit once, after its high end (``27 to 28%``)."""
    held = ReportInput.model_validate(snapshot_of(_KAIRO).report_input)
    figure = resolve("cost.uplift_range", held)
    assert format_figure(figure, ("pct",), forms=PageForms.of(held)) == "27 to 28%"


def test_rule_3_in_a_chart_s_tooltip_the_interval_beside_its_figure_carries_no_unit() -> None:
    """A chart's texts: the kit's tooltip joins a row's value and its ends as they are, ``85% (80 to 89)``, so the
    ends carry no unit; the values table states the interval alone in its column, ``80 to 89%``."""
    chart = BarsChart(rows=[BarRow(label="a", value=85.0, lo=80.0, hi=89.0)], domain=(0, 100), fmt="pct:0", level="95%")
    (row,) = chart.spelled().rows or []
    assert f"{row.value_text} ({row.lo_text} to {row.hi_text})" == "85% (80 to 89)"
    _, body = chart.table([])
    assert body == [["a", "85%", "80 to 89%"]]


def test_rule_3_on_a_chart_read_in_points_its_values_table_names_the_unit() -> None:
    """A change chart whose format names no unit, read in points: its values table writes ``pts`` on the change and
    once after the interval's high end, never ``+10 / -29 to +58`` under a head that names no unit."""
    chart = BarsChart(
        rows=[BarRow(label="a", value=10.0, lo=-29.0, hi=58.0, value_text="+10", lo_text="-29", hi_text="+58")],
        domain=(-30, 60),
        fmt="+num:0",
        level="95%",
    )
    _, body = chart.table([], read_at="pts")
    assert body == [["a", "+10 pts", "-29 to +58 pts"]]


def _glyph_titles(cell: TableCell, header: str) -> list[str]:
    """The titles of the glyphs a one-cell table draws, as its static page writes them."""
    block = TableBlock(
        section="scorecard",
        name="scorecard",
        title="By aspect",
        shared_scale=True,
        columns=[{"key": "aspect", "header": "Aspect"}, {"key": "arm:v2", "header": header}],  # type: ignore[list-item]
        order="aspect by aspect",
        rows=[{"aspect": TableCell(value="classify"), "arm:v2": cell}],
        total_rows=1,
    )
    return [html.unescape(title) for title in re.findall(r"<title>([^<]*)</title>", grid_html(block))]


def test_rule_3_and_signs_in_a_static_evidence_title() -> None:
    """A static evidence title: a change's figure carries its sign, as its cell does (``+0.2``, ``+10 pts``), and its
    unit once, on the figure, the interval beside it bare."""
    forms = PageForms.off_page()
    score = forms.ruled_change_cell(UNITLESS, 0.2, interval=(-1.2, 1.6), places=1)
    assert _glyph_titles(score, "v2") == ["+0.2, interval -1.2 to +1.6"]
    rate = forms.ruled_change_cell(SHARE, 0.1, interval=(-0.45, 0.6), places=0)
    assert _glyph_titles(rate, "v2 (pts)") == ["+10 pts, interval -45 to +60"]


def test_a_mean_s_spread_reads_at_the_mean_s_precision() -> None:
    """A mean ± its standard error: money's spread to the decimals the mean beside it reads to, ``$3.33 ± $0.09``,
    never ``$0.088``; a spread too small for them takes the figures that keep it off zero."""
    ruler = reply_cost_ruler([0.00333, 0.00543])
    forms = PageForms.off_page()
    assert forms.ruled_level(ruler, 0.00333).text == "$3.33"
    assert forms.ruled_spread(ruler, 0.000088, mean=0.00333).text == "$0.09"
    assert forms.ruled_spread(ruler, 0.0000012, mean=0.00333).text == "$0.0012"


def test_a_cost_per_reply_reads_on_the_ruler_the_campaign_s_tables_state_it_on() -> None:
    """One money measure, one ruler on the page: a cost of two cents a reply reads per 1,000 replies on an aspect's
    page (``$20``), as the scorecard and the written page state it, never ``$0.020`` beside them."""
    ruler = reply_cost_ruler([0.02, 0.03])
    assert ruler.scaled("Production cost") == "Production cost, per 1,000 replies"
    assert PageForms.off_page().ruled_level(ruler, 0.02).text == "$20"


def _kairo_chart(block: str, query: dict[str, object]) -> list[str]:
    """The values table of one chart drawn on Kairo's written page from ``query``, as Markdown rows."""
    doc = copy.deepcopy(converted_doc(_KAIRO))
    evidence = doc["claims"][3]["evidence"][1]
    evidence["block"] = block
    evidence["query"].update(query)
    evidence["caption"] = "The chart drawn here."
    held = ReportInput.model_validate(snapshot_of(_KAIRO).report_input)
    lines = doc_page_markdown(build(ReportDoc.model_validate(doc), held, strict=True)).splitlines()
    at = lines.index("The chart drawn here.")
    head = max(index for index in range(at) if lines[index].startswith("| ") and lines[index + 1].startswith("|---"))
    return [line for line in lines[head + 2 : at] if line.startswith("| ")]


def test_a_change_chart_over_contrasts_names_its_unit_in_its_values_table() -> None:
    """Diverging bars over the contrasts, a change in points under a ``+num`` format: the values table writes ``pts``
    on each change and once after its interval, never ``-8 | -13 to -2``."""
    rows = _kairo_chart(
        "diverging_bars",
        {"source": "contrasts", "aspect": "character adherence (v2)", "measures": ["passed"], "arms": [], "strata": []},
    )
    assert rows and all(" pts | " in row for row in rows), rows
    assert any("| -8 pts | -13 to -2 pts |" in row for row in rows), rows


def test_a_change_chart_over_strata_names_its_unit_in_its_values_table() -> None:
    """Diverging bars over strata: each stratum's change reads ``-23 pts``, never a bare ``-23``."""
    rows = _kairo_chart(
        "diverging_bars",
        {
            "source": "strata",
            "aspect": "",
            "measures": ["persona.holds_character"],
            "arms": ["qwen3.8-flash"],
            "strata": ["off_lane", "machinery"],
        },
    )
    assert rows[0].startswith("| off lane | -23 pts |"), rows


def _readings_only(**described: MetricDescriptor) -> ReportInput:
    """The toy measurement's report input, with each measure ``described`` read by code alone: a reading on the
    campaign's one arm that no cell's summary carries, its descriptor in the aspect's catalog."""
    profile = toyhost_profile()
    campaign, storage = toyhost_campaign(profile=profile)
    aspect = assemble_context_bundle(campaign, storage=storage, profile=profile)
    arm = arms_of([aspect])[0]
    bundle = campaign_bundle(aspect, arms=[arm])
    (one,) = bundle.aspects
    read = [
        MeasureReading(
            aspect=bundle.labels[0],
            arm=arm.variant_key,
            arm_label=arm.label or arm.variant_key,
            measure=name,
            value=0.3,
            interval=(0.0, 0.7),
            cases=12,
        )
        for name in described
    ]
    return report_input(
        bundle.model_copy(
            update={
                "aspects": [one.model_copy(update={"measure_catalog": {**one.measure_catalog, **described}})],
                "readings": bundle.readings.model_copy(update={"measures": [*bundle.readings.measures, *read]}),
            }
        )
    )


def _described(name: str, data_type: str, value_range: tuple[float, float] | None) -> MetricDescriptor:
    return MetricDescriptor(
        name=name,
        reader_name=name,
        data_type=data_type,  # type: ignore[arg-type]
        family="research_conclusion",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        population="all_observed",
        unit=None,
        description="A share the host declares with no unit.",
        value_range=value_range,
    )


def test_a_measure_read_by_code_alone_takes_its_unit_from_its_descriptor() -> None:
    """A measurement's code readings carry figures no cell does: such a measure's unit is its descriptor's, so a
    fraction on 0 to 1 enters the report input as a share and every written table reads it in percent."""
    held = _readings_only(research_failed_rate=_described("research_failed_rate", "numeric", (0.0, 1.0)))
    assert held.measures["research_failed_rate"].unit == "share"


def test_a_boolean_with_no_unit_is_a_share() -> None:
    """A boolean that declares no unit and no better end (no check) is a share of what held: the report input
    states it so, as the aspect page reads it."""
    held = _readings_only(went_offstage=_described("went_offstage", "boolean", None))
    assert held.measures["went_offstage"].unit == "share"


@measure(
    reader_name="Fields found", higher_is_better=True, merit_axis="quality", value_range=(0.0, 1.0), population="scored"
)
def fields_found(case: dict[str, str], answer: str) -> float:
    """The share of a case's four fields its answer found: none, a quarter, a half or three quarters."""
    return (int(case["text"].split()[-1]) % 4) / 4


async def _answer(case: dict[str, str]) -> str:
    return "found"


async def test_a_fraction_grader_reads_in_percent_on_the_page_and_in_the_console() -> None:
    """A quick eval's 0-to-1 grader is a share: its console line reads it in percent with its trials named, as its
    page's tables read it, never ``mean 0.375``."""
    cases = [{"text": f"case {index}"} for index in range(12)]
    result = await Eval(cases, {"solo": _answer}, [fields_found], k=1).run(store=":memory:", scope="share")
    console = [line.strip() for line in result.render().splitlines() if line.strip().startswith("Fields found")]
    assert console == ["Fields found: 38% (12 trials)"], console
    # The full print's run-summary line too, which reads the measure off the summary's own text.
    full = [line.strip() for line in result.render(full=True).splitlines() if line.strip().startswith("Fields found:")]
    assert full == ["Fields found: 38% (12 trials)"], full
    page = [line for line in campaign_report_markdown(result.page).splitlines() if "| Fields found |" in line]
    assert page and all("| 38% |" in line or "| 38% (" in line for line in page), page
