"""Every arm's figures across a campaign's aspects, each aspect weighed alike, and the cost/quality chart drawn from
them.

- **The figures are the hand mean.** On S1 each arm's pass rate, each candidate's change and each cost per reply
  across aspects is the plain mean of the aspects' own figures, and the scorecard's equal-weight row prints them as
  the matrix prints its own cells.
- **The weighting is stated in the JSON**: the basis, each aspect's weight (summing to one), and that the intervals
  take the aspects as independent.
- **A figure any aspect lacks is withheld, naming it**, never a mean over the rest.
- **The chart is drawn where the campaign compares more than one pair of figures** (S1, and S2 on its one aspect),
  never for a measurement (S3) or one candidate on one aspect (S4); it prints every figure as a table too.
- **A frozen campaign bundle of shape 2 still reads.** A reporter case or a stored analysis frozen before the
  aggregate existed is read as this shape, gaining the same figures, its fingerprint the one it recorded, so the
  labels frozen on it stand.
"""

from __future__ import annotations

import json
import math
from html.parser import HTMLParser
from typing import Any

import pytest

from threetears.evals.analysis import (
    assemble_campaign_bundle,
    campaign_report,
    campaign_report_html,
    campaign_report_markdown,
    serialize_report,
)
from threetears.evals.analysis.campaign_bundle import (
    CAMPAIGN_BUNDLE_SHAPE,
    CampaignBundle,
    build_campaign_bundle,
    build_campaign_user_message,
    campaign_evidence_digest,
)
from threetears.evals.analysis.campaign_readings import AggregateWeighting
from threetears.evals.analysis.report.campaign.aggregate import EQUAL_WEIGHT_ROW
from threetears.evals.analysis.report.model import CampaignReport, ChartBlock, DisclosureBlock, TableBlock
from threetears.evals.analysis.reporter_bank import frozen_case_receipt
from threetears.evals.analysis.reporter_kind import (
    judge_case_material,
    rebuild_bundle,
    render_case_material,
    reporter_case_of,
    reporter_case_payload,
)
from threetears.evals.kernel.errors import RefusedInput
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.report_support import stored_memo_analysis
from packages.evals.tests.ops_support import TOYHOST_SCOPE, OpsFixture
from packages.evals.tests.report_shapes_support import (
    SHAPES,
    ArmTrials,
    AspectTrials,
    Shape,
    s1_change,
    s2_choice,
    s3_measurement,
    s4_change,
    shape_of,
)
from packages.evals.tests.reporter_case_support import (
    GROUNDED,
    freeze_corpus_campaign,
    reporter_fixture,
    reporter_template,
)
from packages.evals.tests.style_support import lint_page


#: How a page says, once, how its figures across aspects were read.
WEIGHTING_LINE = "A figure across aspects weighs every aspect alike, and its interval takes the aspects as independent."


def _charts(page: CampaignReport) -> list[ChartBlock]:
    return [block for block in page.blocks if isinstance(block, ChartBlock) and block.chart == "cost_quality"]


def _lines(page: CampaignReport) -> list[str]:
    return [block.text for block in page.blocks if isinstance(block, DisclosureBlock)]


def _matrix(page: CampaignReport) -> TableBlock:
    return next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "scorecard")


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


# =============================================================================
# The figures
# =============================================================================


def test_s1_s_figures_across_aspects_are_the_plain_mean_of_each_aspect_s() -> None:
    shape = s1_change()
    aggregate = shape.readings.aggregate
    assert aggregate is not None
    labels = [aspect.label for aspect in shape.campaign.aspects]
    # Each arm's pass rate is its usable trials passed, aspect by aspect, as the matrix and scoreboard count them.
    rates = {
        arm.label: [
            100 * board.passed / board.usable
            for label in labels
            for board in shape.bundles[label].scoreboard
            if board.variant_key == arm.variant_key
        ]
        for arm in shape.arms
    }
    assert rates == {
        "production": [90.0, 50.0, 0.0],
        "v2": [30.0, 90.0, 25.0],
        "v3": [85.0, 50.0, 0.0],
        "v4": [95.0, 40.0, 10.0],
    }
    held = {arm.arm_label: arm for arm in aggregate.arms}
    assert list(held) == ["production", "v2", "v3", "v4"]
    for label, values in rates.items():
        assert held[label].pass_rate.value == pytest.approx(_mean(values))
        low, high = held[label].pass_rate.interval or (math.nan, math.nan)
        assert 0.0 <= low < _mean(values) < high <= 100.0
    # Each candidate's change is the mean of its rows' changes, the matrix's own figures.
    for label in ("v2", "v3", "v4"):
        changes = [row.change for row in shape.readings.readings if row.arm_label == label]
        assert held[label].change is not None
        assert held[label].change.value == pytest.approx(_mean(changes))
    assert held["v2"].change is not None and held["v2"].change.value == pytest.approx((-60 + 40 + 25) / 3)
    assert held["production"].change is None
    # A reply costs v4 a tenth of a cent on every aspect and every other arm a fifth.
    assert {label: arm.cost.value for label, arm in held.items()} == pytest.approx(
        {"production": 0.002, "v2": 0.002, "v3": 0.002, "v4": 0.001}
    )
    assert (held["v4"].cheaper_on, held["v4"].dearer_on) == (labels, [])
    assert (held["v2"].cheaper_on, held["v2"].dearer_on) == ([], [])
    assert held["production"].cheaper_on is None


def test_the_equal_weight_row_prints_each_arm_s_figure_as_the_matrix_prints_its_cells() -> None:
    shape = s1_change()
    page = shape.page()
    *_, across = _matrix(page).rows
    assert across["aspect"].text == EQUAL_WEIGHT_ROW
    by_arm = {arm.label: across[f"arm:{arm.variant_key}"] for arm in shape.arms}
    assert {label: cell.text for label, cell in by_arm.items()} == {
        "production": "46.7",
        "v2": "1.7",
        "v3": "-1.7",
        "v4": "1.7",
    }
    # Each figure's interval is drawn beside it; a change's against zero.
    assert all(cell.low is not None and cell.high is not None for cell in by_arm.values())
    assert by_arm["v2"].zero_line and not by_arm["production"].zero_line
    assert f"| {EQUAL_WEIGHT_ROW} | — | 46.7 | 1.7 | -1.7 | 1.7 |" in campaign_report_markdown(page)
    # The weighting is said once, in the methods.
    said = [
        block.section for block in page.blocks if isinstance(block, DisclosureBlock) and block.text == WEIGHTING_LINE
    ]
    assert said == ["methods"]


@pytest.mark.parametrize("name", ["s1", "s2"])
def test_the_weighting_is_stated_in_the_json_its_weights_summing_to_one(name: str) -> None:
    shape = SHAPES[name]()
    document = json.loads(serialize_report(shape.page(), "json"))
    weighting = document["readings"]["aggregate"]["weighting"]
    labels = [aspect.label for aspect in shape.campaign.aspects]
    assert weighting["basis"] == "equal" and weighting["intervals"] == "aspects_independent"
    assert list(weighting["weights"]) == labels
    assert math.fsum(weighting["weights"].values()) == pytest.approx(1.0)
    assert list(weighting["weights"].values()) == pytest.approx([1 / len(labels)] * len(labels))
    with pytest.raises(ValueError, match="sum to 1"):
        AggregateWeighting(weights={label: 0.5 for label in labels} | {"another": 0.25})


def _priced_on_two(*, unpriced_on: str | None = None, missing_on: str | None = None) -> Shape:
    """S1's first two aspects, v4 unpriced on one of them, or not run there."""
    aspects = [
        AspectTrials(
            "billing",
            [ArmTrials("production", passing=18), ArmTrials("v2", passing=6), ArmTrials("v4", passing=19, cost=0.001)],
        ),
        AspectTrials(
            "tone",
            [ArmTrials("production", passing=10), ArmTrials("v2", passing=18), ArmTrials("v4", passing=10, cost=0.001)],
        ),
    ]
    edited = []
    for aspect in aspects:
        arms = list(aspect.arms)
        if aspect.label == unpriced_on:
            arms = [ArmTrials(arm.label, passing=arm.passing, cost=None) if arm.label == "v4" else arm for arm in arms]
        if aspect.label == missing_on:
            arms = [arm for arm in arms if arm.label != "v4"]
        edited.append(AspectTrials(aspect.label, arms))
    return shape_of("withheld", edited, control=True)


def test_an_arm_unpriced_on_one_aspect_has_its_cost_withheld_naming_it_and_is_left_off_the_chart() -> None:
    shape = _priced_on_two(unpriced_on="tone")
    aggregate = shape.readings.aggregate
    assert aggregate is not None
    v4 = aggregate.arm(shape.key("v4"))
    assert v4 is not None
    assert (v4.cost.value, v4.cost.interval, v4.cost.missing) == (None, None, ["tone"])
    # Its pass rate and change are read on both aspects, and its cost is compared only where both are priced.
    assert v4.pass_rate.value is not None and v4.change is not None and v4.change.value is not None
    assert (v4.cheaper_on, v4.dearer_on) == (["billing"], [])
    page = shape.page()
    (chart,) = _charts(page)
    assert [point.arm for point in chart.points] == ["production", "v2"]
    assert "v4 is left off the chart: it has no cost per reply on tone." in _lines(page)


def test_a_candidate_absent_from_one_aspect_has_its_change_withheld_naming_it() -> None:
    shape = _priced_on_two(missing_on="tone")
    v4 = shape.readings.aggregate.arm(shape.key("v4")) if shape.readings.aggregate else None
    assert v4 is not None and v4.change is not None
    assert (v4.change.value, v4.change.missing) == (None, ["tone"])
    assert (v4.pass_rate.missing, v4.cost.missing) == (["tone"], ["tone"])
    page = shape.page()
    *_, across = _matrix(page).rows
    assert across[f"arm:{shape.key('v4')}"].text == "— (withheld)"
    lines = _lines(page)
    assert "v4's change across aspects is withheld: it has none on tone." in lines
    assert "v4 is left off the chart: it has no cost per reply on tone." in lines
    assert "v4 is left off the chart: it has no pass rate on tone." in lines


# =============================================================================
# The chart
# =============================================================================


def test_s2_draws_its_arms_on_one_aspect_with_that_aspect_s_own_figures_and_no_row() -> None:
    shape = s2_choice()
    page = shape.page()
    (chart,) = _charts(page)
    by_key = {entry.variant_key: entry for entry in shape.bundles["answers"].scoreboard}
    assert [point.arm for point in chart.points] == ["alpha", "beta", "gamma"]
    for point in chart.points:
        entry = by_key[shape.key(point.arm)]
        assert point.pass_rate == pytest.approx(100 * entry.passed / entry.usable)
        assert point.cost == pytest.approx(entry.reply_cost_usd or math.nan)
        assert point.pass_rate_low <= point.pass_rate <= point.pass_rate_high
    assert chart.title == "Cost and pass rate" and chart.unit == "usd"
    # One aspect: the matrix carries no row across aspects, and nothing says how aspects are weighed.
    assert EQUAL_WEIGHT_ROW not in {row["aspect"].text for row in _matrix(page).rows}
    assert WEIGHTING_LINE not in _lines(page)
    drawn = chart.kit_chart()
    assert drawn is not None and drawn[0] == "scatter_frontier"
    assert [point["label"] for point in drawn[1]["points"]] == ["alpha", "beta", "gamma"]  # type: ignore[index,union-attr]
    html = campaign_report_html(page)
    assert '<figure class="fig" data-figure="cost_quality">' in html and 'data-kind="scatter_frontier"' in html
    assert '<div class="tbl chart-values"><table><caption>Cost and pass rate</caption>' in html
    markdown = campaign_report_markdown(page)
    # The table reads cost per 1,000 replies, as the chart's axis and the scoreboard do: $0.004 a reply is $4.
    assert "| Arm | Cost per 1,000 replies ($) | Interval ($) | Pass rate (%) | Interval (%) |" in markdown
    assert "| alpha | 4 | [4, 4] | 88.3 |" in markdown


def test_s1_draws_every_arm_across_its_aspects() -> None:
    page = s1_change().page()
    (chart,) = _charts(page)
    assert chart.title == "Cost and pass rate, all aspects (equal weight)"
    assert [(point.arm, round(point.pass_rate, 1)) for point in chart.points] == [
        ("production", 46.7),
        ("v2", 48.3),
        ("v3", 45.0),
        ("v4", 48.3),
    ]
    drawn = chart.kit_chart()
    assert drawn is not None
    assert [point["label"] for point in drawn[1]["points"]] == ["production", "v2", "v3", "v4"]  # type: ignore[index,union-attr]
    html = campaign_report_html(page)
    assert all(f"<td>{arm}</td>" in html for arm in ("production", "v2", "v3", "v4"))


def _sparse() -> Shape:
    """A campaign whose second aspect lacks an arm: ``billing`` runs production, v2 and v3, ``tone`` production and v3,
    so v3 is the second candidate the campaign lists but the first ``tone`` does."""
    return shape_of(
        "sparse",
        [
            AspectTrials(
                "billing",
                [ArmTrials("production", passing=18), ArmTrials("v2", passing=6), ArmTrials("v3", passing=17)],
            ),
            AspectTrials("tone", [ArmTrials("production", passing=10), ArmTrials("v3", passing=12)]),
        ],
        control=True,
    )


#: The campaigns whose pages tag their arms: dense ones, and one whose aspect lacks an arm.
_TAGGED = {"s1": SHAPES["s1"], "s2": SHAPES["s2"], "sparse": _sparse}


@pytest.mark.parametrize("name", sorted(_TAGGED))
def test_an_arm_is_one_tag_and_one_colour_on_every_chart_of_the_campaign_page(name: str) -> None:
    """GENERATOR.md §5.2: one colour per arm for the whole page. The cost/quality chart and every aspect's outcome bar
    on the campaign page tag an arm alike (the control P, the others A, B, C in the order the campaign's aspects list
    them), so each draws it in one colour; the control, where there is one, ink-gray. An aspect that lacks an arm
    shifts no other arm's tag: on the sparse campaign v3 is B on ``tone`` too, as on ``billing`` and the chart, where
    v2, run on one aspect, is withheld."""
    shape = _TAGGED[name]()
    page = shape.page()
    (chart,) = _charts(page)
    tags = {point.arm: point.tag for point in chart.points}
    assert len(set(tags.values())) == len(tags) and None not in tags.values()
    assert ("P" in tags.values()) == (shape.readings.control is not None)
    colours = {point["label"]: point["color"] for point in chart.kit_chart()[1]["points"]}  # type: ignore[index]
    bars = [
        bar
        for aspect in page.aspects
        for block in aspect.report.blocks
        if isinstance(block, ChartBlock) and block.chart == "outcome_bar"
        for bar in block.bars
    ]
    assert bars, "every aspect's page draws its outcome bar"
    worn: dict[str, set[str | None]] = {}
    for bar in bars:
        worn.setdefault(bar.arm, set()).add(bar.tag)
    assert all(len(held) == 1 for held in worn.values()), worn
    assert len({tag for held in worn.values() for tag in held}) == len(worn), worn
    for bar in bars:
        if bar.arm not in tags:
            continue
        assert bar.tag == tags[bar.arm], bar.arm
        row = next(row for row in _outcome_rows(page) if row["label"] == bar.arm)
        assert row["color"] == colours[bar.short_name or bar.arm]
    if shape.readings.control is not None:
        assert colours[shape.readings.control_label] == "--ink-2"
    assert set(worn) == set(tags) | ({"v2"} if name == "sparse" else set()), (worn, tags)
    if name == "sparse":
        assert sorted(tags) == ["production", "v3"] and worn == {"production": {"P"}, "v2": {"A"}, "v3": {"B"}}


class _ChartData(HTMLParser):
    """Every ``div.chart`` of a page's HTML as the kit reads it: its ``data-kind`` and its ``data-chart`` JSON."""

    def __init__(self) -> None:
        super().__init__()
        self.charts: list[tuple[str, dict[str, Any]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        held = dict(attrs)
        if tag == "div" and "data-chart" in held:
            self.charts.append((held.get("data-kind") or "", json.loads(held["data-chart"] or "")))


def test_the_service_page_draws_a_shared_arm_in_one_tag_and_one_colour_on_every_chart_though_an_aspect_lacks_an_arm() -> (
    None
):
    """The page the service hands out (``campaign_report``, the door every caller reads) tags arms across the whole
    campaign, not per aspect: on the sparse campaign v3 is B and one colour on ``billing``, on ``tone`` (where it is
    the first candidate) and on the cost/quality chart."""
    shape = _sparse()
    page = campaign_report(shape.host(), shape.campaign.id, shape.campaign.scope_id)
    drawn = _ChartData()
    drawn.feed(campaign_report_html(page))
    bars = [data for kind, data in drawn.charts if kind == "bars"]
    (cost,) = (data for kind, data in drawn.charts if kind == "scatter_frontier")
    assert len(bars) == 2, "billing and tone each draw an outcome bar"
    dressed: dict[str, set[tuple[str, str]]] = {}
    for data in [*bars, cost]:
        for entry in data.get("rows", data.get("points", [])):
            dressed.setdefault(entry["label"], set()).add((entry.get("tag") or entry["narrow_label"], entry["color"]))
    assert dressed["production"] == {("P", "--ink-2")}, dressed
    assert dressed["v3"] == {("B", "--s2")}, dressed
    assert "v3" in {entry["label"] for entry in cost["points"]}, "the cost chart draws v3 too"


def _outcome_rows(page: CampaignReport) -> list[dict[str, Any]]:
    """Every drawn row of every aspect's outcome bar on the campaign page."""
    return [
        row
        for aspect in page.aspects
        for block in aspect.report.blocks
        if isinstance(block, ChartBlock) and block.chart == "outcome_bar"
        for row in block.kit_chart()[1]["rows"]  # type: ignore[index,union-attr]
    ]


@pytest.mark.parametrize("build", [s3_measurement, s4_change], ids=["s3", "s4"])
def test_a_measurement_and_one_candidate_on_one_aspect_draw_no_chart(build: Any) -> None:
    shape = build()
    page = shape.page()
    assert _charts(page) == []
    assert WEIGHTING_LINE not in _lines(page)
    assert '<figure class="fig" data-figure="cost_quality">' not in campaign_report_html(page)
    if shape.readings.kind == "measurement":
        assert shape.readings.aggregate is None


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_every_line_the_aggregate_writes_keeps_the_style_rules(name: str) -> None:
    shape = SHAPES[name]()
    markdown = campaign_report_markdown(shape.page())
    labels = [arm.label for arm in shape.arms]
    names = [*labels, *(f"▸ {label}" for label in labels), *{reading.name for reading in shape.readings.measures}]
    said = [line for line in markdown.splitlines() if "left off the chart" in line or "across aspects" in line]
    assert lint_page("\n\n".join(said), names=names) == []
    withheld = _priced_on_two(missing_on="tone").page()
    lines = [line for line in _lines(withheld) if "withheld" in line or "left off" in line]
    assert lines and lint_page("\n\n".join(lines), names=["v4"]) == []


# =============================================================================
# A campaign bundle frozen at shape 2
# =============================================================================


def _at_shape_2(document: dict[str, Any]) -> dict[str, Any]:
    """A campaign bundle or frame as a build at shape 2 wrote it: no figures across aspects in its readings."""
    readings = {key: value for key, value in document["readings"].items() if key != "aggregate"}
    return {**document, "schema_version": 2, "readings": readings}


def test_a_bundle_and_a_frame_frozen_at_shape_2_read_as_this_shape_fingerprinting_alike() -> None:
    shape = s1_change()
    bundle = build_campaign_bundle(
        shape.campaign, list(shape.bundles.values()), arms=shape.arms, readings=shape.readings
    )
    assert CAMPAIGN_BUNDLE_SHAPE == 3 and bundle.readings.aggregate is not None
    frozen = _at_shape_2(bundle.to_dict())
    # The figures across aspects add no evidence, so the shape-2 document fingerprints as this shape's bundle does.
    assert campaign_evidence_digest(frozen) == bundle.fingerprint() == campaign_evidence_digest(bundle.to_dict())
    read = CampaignBundle.from_dict(frozen)
    assert read.schema_version == CAMPAIGN_BUNDLE_SHAPE
    assert read.readings == bundle.readings and read.fingerprint() == bundle.fingerprint()
    stored = CampaignBundle.from_frame(_at_shape_2(bundle.frame()), bundle.aspects)
    assert stored.readings.aggregate == bundle.readings.aggregate
    # A shape before 2 is still refused, naming this build's.
    with pytest.raises(RefusedInput, match=r"this build's shape \(3\) or one it reads as that \(2\)"):
        CampaignBundle.from_dict({**frozen, "schema_version": 1})


async def _analysis_id(fixture: OpsFixture) -> str:
    """A memo analysis of the corpus campaign, stored: a reporter case pins a recorded memo, which the memo writer
    writes (a generation through the host is the report planner's)."""
    analysis, _ = await stored_memo_analysis(fixture.host.eval_host, fixture.campaign)
    return analysis.id


async def test_a_reporter_case_frozen_at_shape_2_still_generates_calibrates_and_keeps_its_labels() -> None:
    fixture = off_loop(reporter_fixture)
    analysis_id = await _analysis_id(fixture)
    off_loop(_a_case_frozen_at_shape_2_reads, fixture, analysis_id)


def _a_case_frozen_at_shape_2_reads(fixture: OpsFixture, analysis_id: str) -> None:
    """Freeze the corpus campaign with its memo and a label, store it again as shape 2 froze it, and read it back."""
    storage = fixture.host.eval_host.storage
    labels = [{"dimension": GROUNDED, "direction": "high", "quote": "every number traces to a run"}]
    receipt = freeze_corpus_campaign(fixture, recorded_analysis_id=analysis_id, labels=labels)
    (stored,) = storage.query_test_cases(TOYHOST_SCOPE, template_id=reporter_template().id)
    case = reporter_case_of(stored)
    assert case is not None and case.recorded_memo
    # As a build at shape 2 froze it: the same case, its bundle without the figures across aspects.
    before = case.model_copy(update={"bundle": _at_shape_2(case.bundle)})
    storage.save_test_case(stored.model_copy(update={"host_payload": reporter_case_payload(before)}))
    (frozen,) = storage.query_test_cases(TOYHOST_SCOPE, template_id=reporter_template().id)
    held = reporter_case_of(frozen)
    assert held is not None and held.bundle["schema_version"] == 2

    # It rebuilds, its recorded fingerprint verifying, with the figures read off what it froze.
    rebuilt = rebuild_bundle(held)
    today = assemble_campaign_bundle(fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE)
    assert rebuilt.fingerprint() == held.bundle_fingerprint == receipt.bundle_fingerprint
    assert rebuilt.readings.aggregate == today.readings.aggregate
    # A generator is sent what it would be sent over today's bundle; the recorded memo is judged as frozen.
    assert render_case_material(held) == build_campaign_user_message(today)
    assert held.writer_message is not None and held.writer_message in judge_case_material(held, generated_over=None)
    # The bank still lists it, labels and all, and the same labels frozen again are this live case.
    assert [label.dimension for label in frozen_case_receipt(frozen).labels] == [GROUNDED]
    again = freeze_corpus_campaign(fixture, recorded_analysis_id=analysis_id, labels=labels)
    assert again.test_case_id == frozen.id
