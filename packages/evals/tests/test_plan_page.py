"""The page a report plan draws: built from the plan and its report input alone, in GENERATOR §2 order.

Held over every r1–r5 anchor and current run, and the Kairo reference plan (``references/r5-kairo-round2.plan.json``),
each read against its own input. The expected side is always read from somewhere else than the page: the figure keys
(:func:`~threetears.evals.analysis.report.figure_keys.resolve`), the rows a query selects
(:func:`~threetears.evals.analysis.report_rows.query_rows`), the gate check, code's plan facts, GENERATOR.md's own
words, and the reference page's source, whose landmarks are parsed, never typed in.

- **Resolvers.** Every forest, bars, diverging bars, lines and examples chart of every plan resolves to its kind,
  drawing what its query selects: a reading's change, interval and margin; a contrast's delta on its scale and a
  caption naming each asked comparison the input holds none of; a figure from zero, a summary chart in the options
  table's order with what fails the gates faded and the recommendation bold; each arm less the control per stratum,
  bold by the descriptive 95% rule; a line per arm over the slices; a card per selected example; small multiples a
  panel per arm or per slice, as the plan asks. No note prints a signed zero. A kind no resolver draws yet (P4-21, P4-22) is refused, naming its kind and source; a whole page here
  leaves those charts out by kind.
- **Tables.** The options table is the plan's rows and code's column families, each cell what its figure key reads,
  the recommended (or leaned) row set apart; a column outside the families is refused. The arms key shows each arm
  once, only where the plan shows it.
- **Confidence, decision and answers.** The confidence reads GENERATOR §2.5's form words, the planner's clauses,
  then code's checked lines; the slots are titled by mode and kind (GENERATOR §2.5); each answer names its warning in
  code's words.
- **The page.** Its parts lie in GENERATOR §2 order and a part the plan drops is not drawn; its Markdown prints each
  chart as its values table; its static page runs and loads nothing, every chart its table, and is the default page
  less exactly the script, the fonts and the chart boxes; markup in plan prose and input names is text.
- **The reference outline.** The reference plan's page has the landmarks the reference page's source has, for the
  parts this page draws (the explore part and Full results are filled elsewhere).
"""

from __future__ import annotations

import functools
import math
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.fixtures.report_plans import dataset
from packages.evals.tests.hostile_page_support import HOSTILE, HOSTILE_LINE
from packages.evals.tests.page_contract_support import what_runs
from packages.evals.tests.report_kit_support import KIT
from packages.evals.tests.static_page_support import assert_static, less_what_static_leaves_out
from threetears.evals.analysis import (
    BarsChart,
    ChartBlock,
    ExampleMetaArm,
    ExampleMetaText,
    ExamplesChart,
    ForestChart,
    LinesChart,
    PlannedPage,
    TableBlock,
    planned_page_html,
    planned_page_markdown,
)
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.figure_keys import format_figure, render, resolve
from threetears.evals.analysis.report.md_parts import markdown_table
from threetears.evals.analysis.report.plan_page.build import plan_page
from threetears.evals.analysis.report.words import arm_tags
from threetears.evals.analysis.report_decision import surviving_options
from threetears.evals.analysis.report_gates import gate_check
from threetears.evals.analysis.report_input import (
    Contrast,
    Example,
    PreferenceRow,
    Reading,
    ReportInput,
    Stat,
    Stratum,
    Task,
)
from threetears.evals.analysis.report_plan_facts import plan_facts
from threetears.evals.analysis.report_rows import query_aspect, query_rows
from threetears.evals.kernel.report_plan import ChartSpec, ReportPlan, WarningAnswer, WarningRef

#: The Kairo reference plan, and the source of the page drawn by hand from it.
REFERENCE_PLAN = KIT / "references" / "r5-kairo-round2.plan.json"
REFERENCE_SOURCE = KIT / "references" / "r5-kairo-round2.src.html"
GENERATOR = KIT / "GENERATOR.md"

#: The kinds whose resolvers are later chunks' (P4-21 and P4-22). A whole page built here leaves a chart of one out,
#: by kind; the dispatcher refuses one, which a test below holds.
_LATER_KINDS = frozenset(
    {
        "gate_matrix",
        "gate_scatter",
        "lattice",
        "dot_interval",
        "range_bars",
        "scatter_frontier",
        "heatmap",
        "diff_heatmap",
        "confusion",
        "table",
    }
)

#: A signed zero, which the package's number rule never writes for a change it holds digits of.
_SIGNED_ZERO = re.compile(r"(?<![\d.])[+-]0(?![\d.])")


def _plan_paths() -> list[tuple[str, str, Path]]:
    found = []
    for name in ("r1", "r2", "r3", "r4", "r5"):
        found += [(name, "anchor", dataset.HERE / name / "anchor.plan.json"), (name, "run", dataset.current_run(name))]
    return [*found, ("r5", "reference", REFERENCE_PLAN)]


_PLANS = _plan_paths()
_IDS = [f"{name}-{label}" for name, label, _ in _PLANS]


@functools.cache
def _input(name: str) -> ReportInput:
    return dataset.report_input_of(name)


@functools.cache
def _plan(path: Path) -> ReportPlan:
    return dataset.load_plan(path)


def _drawable(plan: ReportPlan) -> ReportPlan:
    """The plan less every chart of a kind a later chunk resolves: the page builder never re-checks a plan."""
    return plan.model_copy(
        update={
            "summary": [chart for chart in plan.summary if chart.kind not in _LATER_KINDS],
            "sections": [section for section in plan.sections if section.chart.kind not in _LATER_KINDS],
        }
    )


#: The plans whose page draws a chart of this chunk's kinds: a static page's test needs a figure to hold.
_CHARTED = [
    (name, label, path)
    for name, label, path in _PLANS
    if _drawable(_plan(path)).summary or _drawable(_plan(path)).sections
]
_CHARTED_IDS = [f"{name}-{label}" for name, label, _ in _CHARTED]


@functools.cache
def _page(name: str, path: Path) -> PlannedPage:
    return plan_page(_drawable(_plan(path)), _input(name))


def _charts(plan: ReportPlan, page: PlannedPage) -> list[tuple[ChartSpec, ChartBlock, list[str], bool]]:
    """Each drawn chart beside the stored chart it resolves, its notes, and whether it is a summary chart."""
    drawable = _drawable(plan)
    pairs: list[tuple[ChartSpec, ChartBlock, list[str], bool]] = []
    for spec, chart in zip(drawable.summary, page.summary, strict=True):
        assert isinstance(chart.chart, ChartBlock)
        pairs.append((spec, chart.chart, chart.notes, True))
    for section, drawn in zip(drawable.sections, page.sections, strict=True):
        assert isinstance(drawn.chart.chart, ChartBlock)
        pairs.append((section.chart, drawn.chart.chart, drawn.chart.notes, False))
    return pairs


def _charts_of(kind: str, source: str | None = None) -> list[Any]:
    """Every (input, plan, spec, block, notes, summary) of one kind (and source) across the plans, as parameters."""
    found = []
    for name, label, path in _PLANS:
        plan = _plan(path)
        for spec, block, notes, summary in _charts(plan, _page(name, path)):
            if spec.kind == kind and (source is None or spec.query.source == source):
                found.append(pytest.param(name, plan, spec, block, notes, summary, id=f"{name}-{label}-{spec.title}"))
    assert found, f"no plan draws a {kind} chart over {source}: the test would hold nothing"
    return found


def _scale_factor(key: str, report_input: ReportInput) -> float:
    """What a figure key's value is multiplied by to be drawn: a share as a percent, money per reply per 1,000."""
    figure = resolve(key, report_input)
    return {"share": 100.0, "usd": 1000.0}.get(figure.scale, 1.0)


# --- resolvers ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "label", "path"), _PLANS, ids=_IDS)
def test_every_chart_of_this_chunks_kinds_resolves_to_its_own_kind_with_values_to_read(
    name: str, label: str, path: Path
) -> None:
    plan = _plan(path)
    for spec, block, _, _ in _charts(plan, _page(name, path)):
        assert block.chart == spec.kind
        drawn = block.kit_chart()
        assert drawn is not None and drawn[0] == spec.kind
        _, rows = block.values_table()
        assert rows, f"{spec.kind} over {spec.query.source} drew nothing"


def test_a_chart_kind_no_resolver_draws_is_refused_naming_its_kind_and_source() -> None:
    plan = _plan(dataset.HERE / "r1" / "anchor.plan.json")
    later = next(section.chart for section in plan.sections if section.chart.kind in _LATER_KINDS)
    with pytest.raises(UnresolvableReference) as refused:
        plan_page(plan, _input("r1"))
    assert repr(later.kind) in str(refused.value) and repr(later.query.source) in str(refused.value)


@pytest.mark.parametrize(("name", "plan", "spec", "block", "notes", "summary"), _charts_of("forest", "readings"))
def test_a_forest_of_readings_draws_each_reading_its_interval_and_the_margin_line(
    name: str, plan: ReportPlan, spec: ChartSpec, block: ChartBlock, notes: list[str], summary: bool
) -> None:
    report_input = _input(name)
    data = block.data
    assert isinstance(data, ForestChart)
    readings = {row.arm: row.item for row in query_rows(spec.query, report_input)}
    assert sorted(row.id or "" for row in data.rows) == sorted(readings)
    for row in data.rows:
        reading = readings[row.id or ""]
        assert isinstance(reading, Reading) and reading.within_interval is not None
        low, high = reading.within_interval
        assert (row.value, row.lo, row.hi) == (reading.change, low, high)
        assert bool(row.strong) == (high < 0 or low > 0), "bold exactly where the interval is clear of zero"
    assert data.level == spec.level
    if spec.domain:
        assert list(data.domain) == spec.domain
    if spec.reference:
        margin = resolve(spec.reference[0].key, report_input).value
        assert data.reference is not None and data.reference.kind == "margin"
        assert isinstance(margin, float) and data.reference.value == -margin


@pytest.mark.parametrize(("name", "plan", "spec", "block", "notes", "summary"), _charts_of("forest", "contrasts"))
def test_a_forest_of_contrasts_draws_each_delta_on_its_scale_and_names_what_is_not_in_the_evidence(
    name: str, plan: ReportPlan, spec: ChartSpec, block: ChartBlock, notes: list[str], summary: bool
) -> None:
    report_input = _input(name)
    data = block.data
    assert isinstance(data, ForestChart)
    contrasts = [row.item for row in query_rows(spec.query, report_input) if isinstance(row.item, Contrast)]
    drawn = {(row.id, row.value) for row in data.rows}
    expected = set()
    for contrast in contrasts:
        assert contrast.delta is not None
        factor = _scale_factor(f"contrast.{contrast.arm}.{contrast.measure}.delta", report_input)
        expected.add((contrast.arm, contrast.delta * factor))
    assert {(arm, round(value, 9)) for arm, value in drawn} == {(arm, round(value, 9)) for arm, value in expected}
    verdicts = {contrast.verdict for contrast in contrasts}
    assert any(row.strong for row in data.rows) == bool(verdicts & {"regressed", "improved"})
    assert (len({contrast.measure for contrast in contrasts}) > 1) == all(row.group for row in data.rows)
    control = next((arm.id for arm in report_input.arms if arm.role == "control"), None)
    held = {(contrast.arm, contrast.measure) for contrast in contrasts}
    asked_arms = spec.query.arms or [arm.id for arm in report_input.arms if arm.id != control]
    missing = [
        (arm, key) for key in spec.query.measures for arm in asked_arms if arm != control and (arm, key) not in held
    ]
    caption = [note for note in notes if note.startswith("Not in the evidence")]
    assert len(caption) == (1 if missing else 0)
    names = {fact.id: fact.name for fact in plan_facts(report_input).arms}
    for arm, _ in missing:
        assert names[arm] in caption[0]


@pytest.mark.parametrize(("name", "plan", "spec", "block", "notes", "summary"), _charts_of("bars"))
def test_bars_draw_each_figure_from_zero_and_a_summary_follows_the_options_table(
    name: str, plan: ReportPlan, spec: ChartSpec, block: ChartBlock, notes: list[str], summary: bool
) -> None:
    report_input = _input(name)
    data = block.data
    assert isinstance(data, BarsChart) and data.domain is not None
    rows = [row for panel in data.panels or [] for row in panel.rows or []] + list(data.rows or [])
    assert rows
    assert data.domain[0] <= 0 <= data.domain[1] or spec.domain, "bars start at zero"
    if spec.query.source in ("measures", "configs"):
        for row in rows:
            key = spec.query.measures[0]
            stat = report_input.measures[key].per_arm[row.id or ""]
            factor = _scale_factor(f"measure.{key}.{row.id}.mean", report_input) if stat.mean is not None else 1.0
            assert stat.mean is None or math.isclose(row.value or 0.0, stat.mean * factor)
    if summary:
        order = [option.option for option in plan.options.rows]
        ids = [row.id for row in rows if row.id in order]
        assert ids == [arm for arm in order if arm in ids], "a summary chart's rows follow the options table"
        survivors = set(surviving_options(report_input))
        recommended = next((option.option for option in plan.options.rows if option.role == "recommended"), None)
        for row in rows:
            assert bool(row.muted) == (row.id not in survivors)
            assert bool(row.strong) == (row.id == recommended)


def _slice_value(stat: Stat) -> float | None:
    return float(stat.mean) if stat.mean is not None else None


@pytest.mark.parametrize(("name", "plan", "spec", "block", "notes", "summary"), _charts_of("diverging_bars", "strata"))
def test_diverging_bars_of_strata_draw_each_arm_less_the_control_bold_by_the_descriptive_rule(
    name: str, plan: ReportPlan, spec: ChartSpec, block: ChartBlock, notes: list[str], summary: bool
) -> None:
    report_input = _input(name)
    data = block.data
    assert isinstance(data, BarsChart) and data.domain is not None
    assert data.domain[0] <= 0 <= data.domain[1]
    control = next(arm.id for arm in report_input.arms if arm.role == "control")
    key = spec.query.measures[0]
    factor = _scale_factor(f"judged.{key}.{control}", report_input) if key in report_input.judged else 1.0
    slices = [row.item for row in query_rows(spec.query, report_input) if isinstance(row.item, Stratum)]
    arms = [arm for arm in spec.query.arms or [arm.id for arm in report_input.arms] if arm != control]
    drawn = [row for panel in data.panels or [] for row in panel.rows or []] + list(data.rows or [])
    assert len(drawn) == len(slices) * len(arms)
    by_arm = {arm: [row for row in drawn if row.id == arm] for arm in arms}
    for arm, rows in by_arm.items():
        assert len(rows) == len(slices)
        for item, row in zip(slices, rows, strict=True):
            mine, theirs = item.per_arm[arm][key], item.per_arm.get(control, {}).get(key)
            if mine.change is not None:
                expected = mine.change * factor
            else:
                assert theirs is not None
                expected = ((_slice_value(mine) or 0.0) - (_slice_value(theirs) or 0.0)) * factor
            assert math.isclose(row.value or 0.0, expected, abs_tol=1e-9)
            if mine.sem is not None and theirs is not None and theirs.sem is not None:
                beyond = abs(expected) > 1.96 * math.hypot(mine.sem, theirs.sem) * factor
                assert (bool(row.strong), bool(row.muted)) == (beyond, not beyond)
    if any(row.strong or row.muted for row in drawn):
        assert any("uncorrected" in note for note in notes), "a descriptive rule's bold says it is uncorrected"
    if spec.query.small_multiples_by == "arm" and len(arms) > 1:
        assert data.panels is not None and [panel.tag for panel in data.panels] == [
            arm_tags([arm.id for arm in report_input.arms], control)[arm] for arm in by_arm
        ]


@pytest.mark.parametrize(
    ("name", "plan", "spec", "block", "notes", "summary"), _charts_of("diverging_bars", "human_pref")
)
def test_diverging_bars_of_human_preference_sit_about_the_no_preference_point(
    name: str, plan: ReportPlan, spec: ChartSpec, block: ChartBlock, notes: list[str], summary: bool
) -> None:
    report_input = _input(name)
    data = block.data
    assert isinstance(data, BarsChart) and data.reference is not None and data.domain is not None
    assert data.reference.kind == "no_preference" and data.domain[0] < data.reference.value < data.domain[1]
    slices = [row.item for row in query_rows(spec.query, report_input) if isinstance(row.item, PreferenceRow)]
    assert [(row.id, row.value) for row in data.rows or []] == [(item.id, item.prefers_control) for item in slices]


@pytest.mark.parametrize(("name", "plan", "spec", "block", "notes", "summary"), _charts_of("lines"))
def test_lines_draw_a_line_per_arm_over_the_slices_in_their_order(
    name: str, plan: ReportPlan, spec: ChartSpec, block: ChartBlock, notes: list[str], summary: bool
) -> None:
    report_input = _input(name)
    data = block.data
    assert isinstance(data, LinesChart)
    slices = [row.item for row in query_rows(spec.query, report_input) if isinstance(row.item, Stratum | Task)]
    assert data.categories == [item.name or item.id for item in slices]
    key = spec.query.measures[0]
    arms = spec.query.arms or [arm.id for arm in report_input.arms]
    assert len(data.series) == len(arms)
    ids = {fact.name: fact.id for fact in plan_facts(report_input).arms}
    for series in data.series:
        arm = ids[series.end_label or ""]
        assert series.values == [float(item.per_arm[arm][key].mean or 0.0) for item in slices]


@pytest.mark.parametrize(("name", "plan", "spec", "block", "notes", "summary"), _charts_of("examples"))
def test_example_cards_are_the_examples_the_query_selects_in_its_order(
    name: str, plan: ReportPlan, spec: ChartSpec, block: ChartBlock, notes: list[str], summary: bool
) -> None:
    report_input = _input(name)
    data = block.data
    assert isinstance(data, ExamplesChart)
    examples = [row.item for row in query_rows(spec.query, report_input) if isinstance(row.item, Example)]
    assert len(data.cards) == len(examples)
    control = next((arm.id for arm in report_input.arms if arm.role == "control"), None)
    tags = arm_tags([arm.id for arm in report_input.arms], control)
    for card, example in zip(data.cards, examples, strict=True):
        meta = card.meta or []
        first = meta[0]
        assert isinstance(first, ExampleMetaText) and first.mono and first.text == (example.trial or example.case)
        chips = [item for item in meta if isinstance(item, ExampleMetaArm)]
        assert [chip.tag for chip in chips] == ([tags[example.arm]] if example.arm in tags else [])
        assert card.reason == (example.reason or "No judge reason recorded.")
        assert card.excerpt == (example.excerpt or None)


@pytest.mark.parametrize(("name", "label", "path"), _PLANS, ids=_IDS)
def test_no_chart_note_or_tooltip_prints_a_signed_zero(name: str, label: str, path: Path) -> None:
    page = _page(name, path)
    for chart in [*page.summary, *(section.chart for section in page.sections)]:
        assert isinstance(chart.chart, ChartBlock)
        texts = [*chart.notes, *_strings(chart.chart.kit_chart())]
        assert not [text for text in texts if _SIGNED_ZERO.search(text)]


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for key, item in value.items() if key in ("note", "tip", "label") for text in _strings(item)] + [
            text for item in value.values() if isinstance(item, dict | list | tuple) for text in _strings(item)
        ]
    if isinstance(value, list | tuple):
        return [text for item in value for text in _strings(item)]
    return []


# --- the options table and the arms key ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "label", "path"), _PLANS, ids=_IDS)
def test_the_options_table_is_the_plans_rows_under_codes_column_families_its_choice_set_apart(
    name: str, label: str, path: Path
) -> None:
    plan, page = _plan(path), _page(name, path)
    table = page.options.table
    assert [column.key for column in table.columns] == ["option", "role", *plan.options.columns]
    names = {fact.id: fact.name for fact in plan_facts(_input(name)).arms}
    assert [row["option"].value for row in table.rows] == [
        names.get(row.option, row.option) for row in plan.options.rows
    ]
    chosen = [index for index, row in enumerate(plan.options.rows) if row.role in ("recommended", "lean")]
    assert page.options.highlighted == (chosen[0] if chosen else None)
    html = planned_page_html(page)
    assert html.count('<tr class="hl">') == (1 if chosen else 0)


def _flagged(cell: Any) -> str | None:
    return cell.flag


@pytest.mark.parametrize(("name", "label", "path"), _PLANS, ids=_IDS)
def test_each_options_cell_reads_what_its_figure_key_or_the_gate_check_reads(name: str, label: str, path: Path) -> None:
    report_input = _input(name)
    plan, page = _plan(path), _page(name, path)
    aspect = query_aspect("", report_input)
    for option, row in zip(plan.options.rows, page.options.table.rows, strict=True):
        arm = option.option
        for column in plan.options.columns:
            cell = row[column]
            if column == "pass_rate":
                assert cell.value == format_figure(resolve(f"pass_rate.{arm}", report_input))
            elif column == "reading.change" and cell.value is not None:
                assert cell.value == format_figure(resolve(f"reading.{arm}.change", report_input))
                assert cell.flag == format_figure(resolve(f"reading.{arm}.interval", report_input))
            elif column == "reading.verdict":
                reading = next((one for one in aspect.readings if one.arm == arm), None) if aspect else None
                if reading is not None and reading.within_interval is not None:
                    low, high = reading.within_interval
                    assert cell.status == ("bad" if high < 0 else "good" if low > 0 else "neutral")
            elif column.startswith("judged:"):
                assert cell.value == format_figure(
                    resolve(f"judged.{column.removeprefix('judged:')}.{arm}", report_input)
                )
            elif column == "gate":
                status = gate_check(report_input, arm).status
                assert cell.status == {"pass": "good", "borderline": "warn", "fail": "bad"}[status]
            elif column == "cost_per_1k":
                assert cell.value == format_figure(resolve(f"cost_per_1k.{arm}", report_input), ("usd",))


def test_an_options_column_outside_the_families_is_refused() -> None:
    plan = _plan(REFERENCE_PLAN)
    unknown = plan.model_copy(update={"options": plan.options.model_copy(update={"columns": ["latency_by_moon"]})})
    with pytest.raises(UnresolvableReference, match="latency_by_moon"):
        plan_page(unknown, _input("r5"))


@pytest.mark.parametrize(("name", "label", "path"), _PLANS, ids=_IDS)
def test_the_arms_key_shows_each_arm_once_by_its_tag_and_only_where_the_plan_shows_it(
    name: str, label: str, path: Path
) -> None:
    report_input, plan, page = _input(name), _plan(path), _page(name, path)
    html = planned_page_html(page)
    if not plan.show_arms_key:
        assert page.arms_key is None and 'data-part="arms"' not in html
        return
    assert page.arms_key is not None and 'data-part="arms"' in html
    control = next((arm.id for arm in report_input.arms if arm.role == "control"), None)
    tags = arm_tags([arm.id for arm in report_input.arms], control)
    rows = page.arms_key.table.rows
    assert [row["arm"].tag for row in rows] == [tags[arm.id] for arm in report_input.arms]
    for arm, row in zip(report_input.arms, rows, strict=True):
        shown = str(row["settings"].value)
        assert all(str(value) in shown for value in arm.settings.values())


# --- confidence, decision and answers -------------------------------------------------------------------------------


def _generator_forms() -> dict[str, str]:
    """GENERATOR §2.5's three confidence forms, verbatim, in its order: high, high if immaterial, low."""
    forms = re.findall(r'^\s*- \*\*"([^"]+)"\*\*:', GENERATOR.read_text(encoding="utf-8"), flags=re.MULTILINE)
    assert len(forms) == 3, forms
    return dict(zip(("high", "high_if_immaterial", "low"), forms, strict=True))


def _generator_slots(mode: str) -> list[str]:
    """GENERATOR §2.5's slot titles for a mode, each less its parenthetical."""
    text = GENERATOR.read_text(encoding="utf-8")
    found = re.search(rf"{mode} mode: (.+?)\. (?:Which|\*\*)", text)
    assert found is not None
    return [re.sub(r" \(.*\)$", "", title) for title in found.group(1).split(" · ")]


@pytest.mark.parametrize(("name", "label", "path"), _PLANS, ids=_IDS)
def test_the_confidence_reads_the_generators_form_words_the_planners_clauses_then_codes_checked_lines(
    name: str, label: str, path: Path
) -> None:
    report_input, plan, page = _input(name), _plan(path), _page(name, path)
    confidence = page.confidence
    form = plan.decision.confidence.form
    assert (confidence.form, confidence.words) == (form, _generator_forms()[form])
    names = {fact.id: fact.name for fact in plan_facts(report_input).arms}
    assert [(clause.caveat, clause.text) for clause in confidence.clauses] == [
        (clause.id, render(clause.material_if, report_input, names)) for clause in plan.decision.confidence.caveats
    ]
    assert confidence.checked == [caveat.checked for caveat in plan_facts(report_input).caveats if caveat.checked]
    html = planned_page_html(page)
    box = re.search(r'<div class="(flag|callout)" data-part="confidence"', html)
    assert box is not None and box.group(1) == ("callout" if form == "high" else "flag")
    assert html.count('<li class="checked">') == len(confidence.checked)


@pytest.mark.parametrize(("name", "label", "path"), _PLANS, ids=_IDS)
def test_the_decision_slots_are_titled_by_mode_and_kind_and_set_the_call_apart(
    name: str, label: str, path: Path
) -> None:
    report_input, plan, page = _input(name), _plan(path), _page(name, path)
    titles = [slot.title for slot in page.decision]
    change = any(arm.role == "control" for arm in report_input.arms)
    if plan.mode == "tradeoff":
        assert titles == _generator_slots("Tradeoff")
        set_apart = "our_lean"
    else:
        expected = _generator_slots("Decide")
        if change:
            # The candidates against a control are named so, as the reference page names them.
            expected[1] = expected[1].replace("the others", "the candidates")
        assert titles == expected
        set_apart = "recommendation" if plan.mode == "decide" else ""
    assert [slot.key for slot in page.decision if slot.highlighted] == ([set_apart] if set_apart else [])
    assert planned_page_html(page).count('<div class="rec"') == (1 if set_apart else 0)


def test_each_answer_names_its_warning_in_codes_words_beside_the_planners_answer() -> None:
    report_input = _input("r5")
    plan = _plan(REFERENCE_PLAN)
    aspect = report_input.aspects[0]
    candidate = next(arm.id for arm in report_input.arms if arm.role != "control")
    floor = report_input.floors[0].floor if report_input.floors else "floor"
    warnings = [
        WarningRef(kind="floor_broken", aspect=aspect.id, name=floor),
        WarningRef(kind="floor_unclear", aspect=aspect.id, name=floor),
        WarningRef(kind="behind_on", aspect=aspect.id, name=candidate),
        WarningRef(kind="bar_missed", aspect=aspect.id, name="persona.holds_character"),
    ]
    answered = plan.model_copy(
        update={
            "answers": [
                WarningAnswer(warning=warning, text=f"It stands over {{arm:{candidate}}} here.") for warning in warnings
            ]
        }
    )
    page = plan_page(answered, report_input)
    said = [answer.warning for answer in page.answers]
    assert len(set(said)) == len(warnings), "one template per warning kind"
    reader = plan_facts(report_input).names["aspect"][aspect.id]
    assert all(reader in line for line in said)
    names = {fact.id: fact.name for fact in plan_facts(report_input).arms}
    assert names[candidate] in said[2]
    assert [answer.text for answer in page.answers] == [
        render(f"It stands over {{arm:{candidate}}} here.", report_input, names)
    ] * len(warnings)
    assert 'data-part="answers"' in planned_page_html(page)


# --- the page -------------------------------------------------------------------------------------------------------

#: GENERATOR §2's parts, as the page marks them, in the order the page lays them out.
_PAGE_ORDER = (
    'class="mast"',
    "<h1>",
    'class="lede"',
    'data-part="flag"',
    'data-part="confidence"',
    'data-part="decision"',
    'data-part="answers"',
    'data-table="options"',
    'data-part="summary"',
    'data-part="arms"',
    "data-section=",
    'class="method"',
)


@pytest.mark.parametrize(("name", "label", "path"), _PLANS, ids=_IDS)
def test_the_page_lays_its_parts_out_in_generator_order_and_draws_none_the_plan_drops(
    name: str, label: str, path: Path
) -> None:
    plan, page = _drawable(_plan(path)), _page(name, path)
    html = planned_page_html(page)
    present = {
        'data-part="flag"': bool(plan.flag),
        'data-part="answers"': bool(plan.answers),
        'data-part="summary"': bool(plan.summary),
        'data-part="arms"': plan.show_arms_key,
        "data-section=": bool(plan.sections),
    }
    places = []
    for marker in _PAGE_ORDER:
        drawn = marker in html
        assert drawn == present.get(marker, True), f"{marker} drawn: {drawn}"
        if drawn:
            places.append(html.index(marker))
    assert places == sorted(places)
    assert html.count("data-section=") == len(plan.sections)
    assert [section.id for section in page.sections] == [section.id for section in plan.sections]


def test_a_page_whose_plan_drops_its_flag_summary_and_arms_key_draws_none_of_them() -> None:
    plan = _drawable(_plan(dataset.HERE / "r4" / "anchor.plan.json"))
    assert plan.flag and plan.summary and plan.show_arms_key
    dropped = plan.model_copy(update={"flag": [], "summary": [], "show_arms_key": False})
    html = planned_page_html(plan_page(dropped, _input("r4")))
    assert not [part for part in ('data-part="flag"', 'data-part="summary"', 'data-part="arms"') if part in html]


@pytest.mark.parametrize(("name", "label", "path"), _PLANS, ids=_IDS)
def test_the_markdown_prints_each_chart_as_its_values_table(name: str, label: str, path: Path) -> None:
    page = _page(name, path)
    markdown = planned_page_markdown(page)
    for chart in [*page.summary, *(section.chart for section in page.sections)]:
        block = chart.chart
        assert isinstance(block, ChartBlock | TableBlock)
        if isinstance(block, ChartBlock):
            headers, rows = block.values_table()
            assert "\n".join(markdown_table(headers, rows)) in markdown
    assert "\n".join(markdown_table([column.header for column in page.options.table.columns], [])[:2]) in markdown


@pytest.mark.parametrize(("name", "label", "path"), _CHARTED, ids=_CHARTED_IDS)
def test_the_static_page_runs_and_loads_nothing_and_is_the_default_page_less_script_fonts_and_chart_boxes(
    name: str, label: str, path: Path
) -> None:
    page = _page(name, path)
    static = planned_page_html(page, static=True)
    assert_static(static)
    assert static == less_what_static_leaves_out(planned_page_html(page))
    assert planned_page_html(page, static=False) == planned_page_html(page)


def test_the_appendix_is_laid_out_closed_after_method() -> None:
    page = _page("r5", REFERENCE_PLAN)
    markup = '<section id="aspect-x"><p>folded</p></section>'
    html = planned_page_html(page, appendix=[markup])
    assert html.index('class="method"') < html.index('data-part="appendix"') < html.index(markup)
    assert '<details class="section" data-part="appendix">' in html


def _hostile_input(report_input: ReportInput) -> ReportInput:
    """The input with markup in every name it gives a reader: campaign, arms, aspects, strata, tasks, labels, cards."""
    data = report_input.model_dump(by_alias=True, exclude_none=True)
    data["meta"]["campaign"] = HOSTILE_LINE
    for index, arm in enumerate(data["arms"]):
        arm.update(name=f"{HOSTILE_LINE} {index}")
        arm.pop("short_name", None)
    for part in ("aspects", "strata", "tasks"):
        for item in data.get(part, []):
            item["name"] = HOSTILE_LINE
    for table in ("measures", "judged"):
        for figures in data[table].values():
            figures["label"] = HOSTILE_LINE
    for example in data.get("examples", []):
        example.update(reason=HOSTILE, excerpt=HOSTILE, input=HOSTILE)
    return ReportInput.model_validate(data)


#: The plan fields that hold the planner's prose.
_PROSE_FIELDS = frozenset(
    {"headline", "lede", "title", "text", "material_if", "eyebrow", "takeaway", "caption", "term", "recomputes"}
)


def _hostile_prose(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: HOSTILE if key in _PROSE_FIELDS and isinstance(item, str) and item else _hostile_prose(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_hostile_prose(item) for item in value]
    return value


@pytest.mark.parametrize(
    ("name", "path"), [("r5", REFERENCE_PLAN), ("r4", dataset.HERE / "r4" / "anchor.plan.json")], ids=["r5", "r4"]
)
def test_markup_in_plan_prose_and_input_names_is_text_on_the_page(name: str, path: Path) -> None:
    plan = _drawable(_plan(path))
    data = _hostile_prose(plan.model_dump(by_alias=True))
    data["masthead"]["system"] = HOSTILE_LINE
    data["lede"] = HOSTILE.split("<iframe")[0]  # the lede holds two sentences at most
    hostile = plan_page(ReportPlan.model_validate(data), _hostile_input(_input(name)))
    page = planned_page_html(hostile)
    assert what_runs(page) == []
    assert "<script>alert" not in page and "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert_static(planned_page_html(hostile, static=True))
    markdown = planned_page_markdown(hostile)
    assert HOSTILE.split("<")[1] in markdown, "the words survive in the Markdown"


# --- the reference outline ------------------------------------------------------------------------------------------


class _Outline(HTMLParser):
    """A page's landmarks, in order: the masthead, the headline, the lede, each box above the decision, the decision's
    titles, the options table, the summary's chart kinds, each section's eyebrow, chart kinds and whether it holds a
    table of its own, each part's title, and Method. Each landmark carries the part it sits under."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.landmarks: list[tuple[str | None, tuple[Any, ...]]] = []
        self._stack: list[tuple[str, set[str]]] = []
        self._part: str | None = None
        self._text: list[str] | None = None
        self._reading: str = ""
        self._decision: list[str] | None = None
        self._summary: list[str] | None = None
        self._section: dict[str, Any] | None = None
        self._options_seen = False

    def _inside(self, css: str) -> bool:
        return any(css in classes for _, classes in self._stack)

    def _inside_tag(self, tag: str) -> bool:
        return any(held == tag for held, _ in self._stack)

    def _add(self, *landmark: Any) -> None:
        self.landmarks.append((self._part, landmark))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        held = dict(attrs)
        classes = set((held.get("class") or "").split())
        if tag in ("meta", "link", "br", "input"):
            return
        if "mast" in classes:
            self._add("mast")
        elif tag == "h1":
            self._add("h1")
        elif tag == "p" and "lede" in classes:
            self._add("lede")
        elif ("flag" in classes or "callout" in classes) and self._inside("bluf"):
            self._add("box")
        elif "decision" in classes:
            self._decision = []
        elif tag == "span" and "k" in classes and self._decision is not None:
            self._start_text("k")
        elif "grid2" in classes:
            self._summary = []
        elif "chart" in classes and tag == "div":
            kind = held.get("data-kind") or ""
            if self._summary is not None:
                self._summary.append(kind)
            elif self._section is not None:
                self._section["kinds"].append(kind)
        elif tag == "table" and not self._inside_tag("figure"):
            if self._inside("bluf") and not self._inside("grid2") and not self._options_seen:
                self._options_seen = True
                self._add("options")
            elif self._section is not None:
                self._section["table"] = True
        elif tag == "section":
            self._section = {"eyebrow": "", "kinds": [], "table": False}
        elif tag == "span" and "eyebrow" in classes and self._section is not None:
            self._start_text("eyebrow")
        elif tag == "h2" and self._inside("part"):
            self._start_text("part")
        elif tag == "dl" and "method" in classes:
            self.landmarks.append((None, ("method",)))
        self._stack.append((tag, classes))

    def _start_text(self, reading: str) -> None:
        self._text, self._reading = [], reading

    def handle_data(self, data: str) -> None:
        if self._text is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._text is not None and tag in ("span", "h2"):
            said = " ".join("".join(self._text).split())
            if self._reading == "k" and self._decision is not None:
                self._decision.append(said)
            elif self._reading == "eyebrow" and self._section is not None:
                self._section["eyebrow"] = said
            elif self._reading == "part":
                self._part = said
                self._add("part", said)
            self._text = None
        while self._stack:
            open_tag, classes = self._stack.pop()
            if open_tag == tag:
                break
        else:
            return
        if "decision" in classes and self._decision is not None:
            self._add("decision", tuple(self._decision))
            self._decision = None
        elif "grid2" in classes and self._summary is not None:
            self._add("summary", tuple(self._summary))
            self._summary = None
        elif tag == "section" and self._section is not None:
            section = self._section
            if section["eyebrow"] or section["kinds"] or section["table"]:
                self._add("section", section["eyebrow"], tuple(section["kinds"]), section["table"])
            self._section = None


def _outline(page: str, excluded: set[str]) -> list[tuple[Any, ...]]:
    parser = _Outline()
    parser.feed(page)
    parser.close()
    return [landmark for part, landmark in parser.landmarks if part not in excluded]


def test_the_reference_plans_page_has_the_reference_pages_outline() -> None:
    plan = _plan(REFERENCE_PLAN)
    # The explore part and Full results are filled by their own parts of the builder; Method, which closes Full results
    # on the reference page, is held wherever it sits.
    excluded = {explore.title for explore in plan.explore} | {"Full results"}
    reference = _outline(REFERENCE_SOURCE.read_text(encoding="utf-8"), excluded)
    ours = _outline(planned_page_html(plan_page(plan, _input("r5"))), excluded)
    assert reference == ours
    assert ("section", "The arms", (), True) in reference, "the outline must reach the arms key to test it"


def _section_with(plan: ReportPlan, section_id: str, by: str) -> ReportPlan:
    """The plan with one section's chart in small multiples by ``by``, and no other chart."""
    section = next(section for section in plan.sections if section.id == section_id)
    query = section.chart.query.model_copy(update={"small_multiples_by": by})
    chart = section.chart.model_copy(update={"query": query})
    return plan.model_copy(update={"summary": [], "sections": [section.model_copy(update={"chart": chart})]})


@pytest.mark.parametrize(
    ("name", "path", "section_id", "by"),
    [
        ("r5", REFERENCE_PLAN, "strata", "task"),
        ("r5", REFERENCE_PLAN, "strata", "arm"),
        ("r2", dataset.HERE / "r2" / "anchor.plan.json", "tasks", "arm"),
        ("r2", dataset.HERE / "r2" / "anchor.plan.json", "tasks", "task"),
    ],
    ids=["diverging-by-slice", "diverging-by-arm", "bars-by-arm", "bars-by-task"],
)
def test_small_multiples_are_a_panel_per_arm_or_per_slice_as_the_plan_asks(
    name: str, path: Path, section_id: str, by: str
) -> None:
    report_input = _input(name)
    plan = _section_with(_plan(path), section_id, by)
    (section,) = plan_page(plan, report_input).sections
    data = section.chart.chart.data if isinstance(section.chart.chart, ChartBlock) else None
    assert isinstance(data, BarsChart) and data.panels is not None and data.rows is None
    spec = plan.sections[0].chart
    slices = [row.item for row in query_rows(spec.query, report_input) if isinstance(row.item, Stratum | Task)]
    names = {fact.id: fact.name for fact in plan_facts(report_input).arms}
    drawn = {row.id for panel in data.panels for row in panel.rows or []}
    if by == "arm":
        order = spec.query.arms or list(names)
        assert [panel.title for panel in data.panels] == [names[arm] for arm in order if arm in drawn]
        assert all(len(panel.rows or []) == len(slices) for panel in data.panels)
    else:
        assert [panel.title for panel in data.panels] == [item.name or item.id.replace("_", " ") for item in slices]
        assert all({row.id for row in panel.rows or []} <= set(names) for panel in data.panels)
