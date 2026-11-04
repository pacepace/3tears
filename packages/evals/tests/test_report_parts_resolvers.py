"""The chart resolvers the report document's page and the analyst's blocks draw with, held on every chart a plan
stores and on charts redrawn from them, each against its own input.

Held over every r1–r5 anchor and current run, and the Kairo reference plan (``references/r5-kairo-round2.plan.json``),
each drawn through :func:`~threetears.evals.analysis.report.parts.resolve.resolve_chart` under the plan's look
(``plan_charts_support.drawn_charts``). The expected side is always read from somewhere else than the resolver: the
figure keys (:func:`~threetears.evals.analysis.report.figure_keys.resolve`), the rows a query selects
(:func:`~threetears.evals.analysis.report_rows.query_rows`) and code's facts, never typed in.

- **Every chart resolves** to its kind, drawing what its query selects: a reading's change, interval and margin; a
  contrast's delta on its scale and a caption naming each asked comparison the input holds none of; a figure from
  zero or on a log axis; each arm less the control per stratum, bold by the descriptive 95% rule; a line per arm over
  the slices; a card per selected example; small multiples a panel per arm or per slice, as the chart asks. No note
  or tooltip prints a signed zero. A chart its kind's resolver cannot draw from the input is refused, naming its kind
  and source, never drawn empty.
- **Every (kind, source) pair** the resolvers draw that no committed plan draws is held on an anchor's chart redrawn
  as that kind and source, its expected side computed from the input; and each arm less the control is refused on an
  input with no control.
- **The Full results tables** (:func:`~threetears.evals.analysis.report.parts.full_results.every_table`), over each
  r1–r5 input and the Kairo reference plan's look: every arm by measure, each figure in one column; every corrected
  comparison, its change and interval the input's on the measure's scale; every slice's figures; every gate at every
  place it is read, an arm failing (or borderline on) a row exactly where the gate check fails it (or holds it
  borderline) somewhere; every configuration. A pill says its status in words.

These tests held the planned page until it retired (P4-5); the resolvers they hold survive it.
"""

from __future__ import annotations

import functools
import math
import re
from pathlib import Path
from typing import Any, get_args

import pytest

from packages.evals.tests.fixtures.report_plans import dataset
from packages.evals.tests.plan_charts_support import anchor_look, drawn_charts
from packages.evals.tests.report_kit_support import KIT
from threetears.evals.analysis import (
    BarsChart,
    ChartBlock,
    ExampleMetaArm,
    ExampleMetaText,
    ExamplesChart,
    ForestChart,
    LinesChart,
)
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.blocks import TableBlock
from threetears.evals.analysis.report.parts.full_results import every_table
from threetears.evals.analysis.report_gates import gate_check
from threetears.evals.analysis.report.figure_keys import resolve
from threetears.evals.analysis.report.words import arm_tags
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
from threetears.evals.analysis.report_rows import query_rows
from threetears.evals.kernel.report_plan import ChartSpec, ReportPlan

#: The Kairo reference plan.
REFERENCE_PLAN = KIT / "references" / "r5-kairo-round2.plan.json"

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


@functools.cache
def _charts(name: str, path: Path) -> list[tuple[ChartSpec, ChartBlock, list[str], bool]]:
    """Each chart the plan stores, drawn on its input, beside the stored chart, its notes, and whether it is a summary
    chart."""
    plan = _plan(path)
    pairs: list[tuple[ChartSpec, ChartBlock, list[str], bool]] = []
    for index, (spec, drawn) in enumerate(drawn_charts(plan, _input(name))):
        assert isinstance(drawn.chart, ChartBlock), f"{spec.kind} over {spec.query.source} drew a table"
        pairs.append((spec, drawn.chart, drawn.notes, index < len(plan.summary)))
    return pairs


def _charts_of(kind: str, source: str | None = None) -> list[Any]:
    """Every (input, plan, spec, block, notes, summary) of one kind (and source) across the plans, as parameters."""
    found = []
    for name, label, path in _PLANS:
        plan = _plan(path)
        for spec, block, notes, summary in _charts(name, path):
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
def test_every_chart_a_plan_stores_resolves_to_its_own_kind_with_values_to_read(
    name: str, label: str, path: Path
) -> None:
    for spec, block, _, _ in _charts(name, path):
        assert block.chart == spec.kind
        drawn = block.kit_chart()
        assert drawn is not None and drawn[0] == spec.kind
        _, rows = block.values_table()
        assert rows, f"{spec.kind} over {spec.query.source} drew nothing"


#: Every chart kind a plan stores but the plain table, which draws any source.
_DRAWN_KINDS = sorted(set(get_args(ChartSpec.model_fields["kind"].annotation)) - {"table"})


def _undrawable_source(kind: str) -> str:
    """A source the plan checks refuse for ``kind``, so its resolver never draws it: confusion axes for every kind
    but the confusion matrix, which reads only those, and target-matrix cells for it and for example cards."""
    return "cells" if kind in ("confusion", "examples") else "confusion"


@pytest.mark.parametrize("kind", _DRAWN_KINDS)
def test_a_chart_its_kind_cannot_draw_is_refused_naming_its_kind_and_source(kind: str) -> None:
    """A stored chart over a source its kind does not draw (a plan the checks would refuse, read without them) is a
    typed refusal naming both, never a chart drawn empty or a crash. r3 holds confusion axes and cells, so the source
    has rows to draw."""
    source = _undrawable_source(kind)
    plan = _plan(dataset.HERE / "r3" / "anchor.plan.json")
    section = plan.sections[0]
    query = section.chart.query.model_copy(update={"source": source})
    chart = section.chart.model_copy(update={"kind": kind, "query": query})
    refused_plan = plan.model_copy(update={"summary": [], "sections": [section.model_copy(update={"chart": chart})]})
    with pytest.raises(UnresolvableReference) as refused:
        drawn_charts(refused_plan, _input("r3"))
    assert repr(kind) in str(refused.value) and repr(source) in str(refused.value)


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
def test_bars_draw_each_figure_from_zero_or_a_log_axis(
    name: str, plan: ReportPlan, spec: ChartSpec, block: ChartBlock, notes: list[str], summary: bool
) -> None:
    report_input = _input(name)
    data = block.data
    assert isinstance(data, BarsChart) and data.domain is not None
    rows = [row for panel in data.panels or [] for row in panel.rows or []] + list(data.rows or [])
    assert rows
    values = [row.value for row in rows if row.value is not None]
    if data.log:
        # A magnitude spanning more than ten times is drawn on a log axis, from its low end: the decade under the least.
        assert 0 < data.domain[0] <= min(values) and max(values) / min(values) > 10, "a log axis only where it spans"
        assert any("log scale" in note for note in notes), "a caption says the axis is logarithmic"
    else:
        assert data.domain[0] <= 0 <= data.domain[1] or spec.domain, "bars start at zero"
    if spec.query.source in ("measures", "configs"):
        for row in rows:
            key = spec.query.measures[0]
            stat = report_input.measures[key].per_arm[row.id or ""]
            factor = _scale_factor(f"measure.{key}.{row.id}.mean", report_input) if stat.mean is not None else 1.0
            assert stat.mean is None or math.isclose(row.value or 0.0, stat.mean * factor)


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
    for _, block, notes, _ in _charts(name, path):
        texts = [*notes, *_strings(block.kit_chart())]
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


# --- small multiples ----------------------------------------------------------------------------------------------


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
    ((_, drawn),) = drawn_charts(plan, report_input)
    data = drawn.chart.data if isinstance(drawn.chart, ChartBlock) else None
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


# --- every (kind, source) pair a resolver draws, beyond the ones a committed plan draws ---------------------------
#
# Each test takes an anchor's own chart, sets the kind and source, and computes what it should draw from the input
# directly: never typed in, never through the resolver's own helpers.


def _redrawn(plan: ReportPlan, chart: ChartSpec, *, kind: str, level: str = "", **query: Any) -> ReportPlan:
    """The plan with one section only, drawing ``chart`` as ``kind`` with its query updated, its level set and its
    stored axis and reference line cleared (they were the old kind's)."""
    spec = chart.model_copy(
        update={
            "kind": kind,
            "query": chart.query.model_copy(update=query),
            "level": level,
            "domain": [],
            "reference": [],
        }
    )
    shell = plan.sections[0].model_copy(update={"id": "redrawn", "chart": spec})
    return plan.model_copy(update={"summary": [], "sections": [shell]})


def _drawn(plan: ReportPlan, report_input: ReportInput) -> tuple[Any, list[str]]:
    ((_, drawn),) = drawn_charts(plan, report_input)
    block = drawn.chart
    assert isinstance(block, ChartBlock)
    return block.data, drawn.notes


def _bars_of(data: Any) -> list[Any]:
    assert isinstance(data, BarsChart)
    return [row for panel in data.panels or [] for row in panel.rows or []] + list(data.rows or [])


def _section(plan: ReportPlan, section_id: str) -> ChartSpec:
    return next(section.chart for section in plan.sections if section.id == section_id)


def _with_control(report_input: ReportInput, arm: str) -> ReportInput:
    """The input with ``arm`` as its control."""
    data = report_input.model_dump(by_alias=True, exclude_none=True)
    for held in data["arms"]:
        if held["id"] == arm:
            held["role"] = "control"
    return ReportInput.model_validate(data)


def test_diverging_bars_over_configs_draw_each_levers_effect_averaged_over_the_other_factors() -> None:
    report_input = _input("r2")
    plan = _plan(dataset.HERE / "r2" / "anchor.plan.json")
    chart = _section(plan, "levers")
    (key,) = chart.query.measures
    data, notes = _drawn(_redrawn(plan, chart, kind="diverging_bars"), report_input)
    value = {arm: stat.mean for arm, stat in report_input.measures[key].per_arm.items() if stat.mean is not None}
    factor = _scale_factor(f"measure.{key}.{report_input.configs[0].arm}.mean", report_input)
    expected: dict[tuple[str, str], float] = {}
    for lever in report_input.factors:
        base = lever.levels[0].id
        for level in lever.levels[1:]:
            effects = []
            for config in report_input.configs:
                if config.levels[lever.id] != level.id or config.arm not in value:
                    continue
                twins = [
                    other.arm
                    for other in report_input.configs
                    if other.levels[lever.id] == base
                    and all(other.levels[name] == held for name, held in config.levels.items() if name != lever.id)
                ]
                if twins and twins[0] in value:
                    effects.append(value[config.arm] - value[twins[0]])
            if effects:
                expected[(lever.name or lever.id, level.name or level.id)] = sum(effects) / len(effects) * factor
    rows = _bars_of(data)
    assert {(row.group, row.label): row.value for row in rows} == pytest.approx(expected)
    assert len(rows) == len(expected) and expected
    assert any("descriptive" in note for note in notes)


@pytest.mark.parametrize(
    ("name", "path", "where", "source", "measures"),
    [
        ("r5", REFERENCE_PLAN, "summary", "measures", None),
        ("r4", dataset.HERE / "r4" / "anchor.plan.json", "summary", "judged", ["persona"]),
    ],
    ids=["measures", "judged"],
)
def test_diverging_bars_over_measures_and_judged_draw_each_candidate_less_the_control(
    name: str, path: Path, where: str, source: str, measures: list[str] | None
) -> None:
    report_input = _input(name)
    plan = _plan(path)
    chart = next(spec for spec in plan.summary if spec.kind == "bars")
    update: dict[str, Any] = {"source": source, "arms": [], "transform": "arm_minus_control"}
    if measures is not None:
        update["measures"] = measures
    redrawn = _redrawn(plan, chart, kind="diverging_bars", **update)
    data, _ = _drawn(redrawn, report_input)
    (key,) = redrawn.sections[0].chart.query.measures
    table = report_input.measures if source == "measures" else report_input.judged
    control = next(arm.id for arm in report_input.arms if arm.role == "control")
    family = "measure" if source == "measures" else "judged"
    path_of = f"measure.{key}.{control}.mean" if family == "measure" else f"judged.{key}.{control}"
    factor = _scale_factor(path_of, report_input)
    theirs = table[key].per_arm[control].mean
    assert theirs is not None
    expected = {
        arm: ((stat.mean or 0.0) - theirs) * factor
        for arm, stat in table[key].per_arm.items()
        if arm != control and stat.mean is not None
    }
    rows = _bars_of(data)
    assert {row.id: row.value for row in rows} == pytest.approx(expected)
    assert data.domain[0] <= 0 <= data.domain[1]


def test_diverging_bars_over_tasks_draw_each_arm_less_the_control_in_each_task() -> None:
    plan = _plan(dataset.HERE / "r2" / "anchor.plan.json")
    chart = _section(plan, "tasks")
    control = chart.query.arms[0]
    report_input = _with_control(_input("r2"), control)
    redrawn = _redrawn(plan, chart, kind="diverging_bars", small_multiples_by="none", transform="arm_minus_control")
    data, _ = _drawn(redrawn, report_input)
    (key,) = chart.query.measures
    expected = {}
    for task in report_input.tasks:
        for arm in chart.query.arms[1:]:
            mine, theirs = task.per_arm[arm][key].mean, task.per_arm[control][key].mean
            assert mine is not None and theirs is not None
            expected[(arm, task.name)] = mine - theirs
    assert {(row.id, row.label): row.value for row in _bars_of(data)} == pytest.approx(expected)


@pytest.mark.parametrize(("source", "section_id"), [("tasks", "tasks"), ("measures", "levers")])
def test_a_chart_of_each_arm_less_the_control_is_refused_where_the_input_names_no_control(
    source: str, section_id: str
) -> None:
    report_input = _input("r2")
    assert not [arm for arm in report_input.arms if arm.role == "control"]
    plan = _plan(dataset.HERE / "r2" / "anchor.plan.json")
    redrawn = _redrawn(plan, _section(plan, section_id), kind="diverging_bars", source=source)
    with pytest.raises(UnresolvableReference, match="no control") as refused:
        drawn_charts(redrawn, report_input)
    assert "'diverging_bars'" in str(refused.value) and repr(source) in str(refused.value)


def test_diverging_bars_over_readings_bold_what_is_separated_and_mute_the_rest() -> None:
    report_input = _input("r5")
    plan = _plan(REFERENCE_PLAN)
    chart = _section(plan, "question")
    data, _ = _drawn(_redrawn(plan, chart, kind="diverging_bars", level=chart.level), report_input)
    readings = {row.arm: row.item for row in query_rows(chart.query, report_input)}
    rows = _bars_of(data)
    assert sorted(row.id for row in rows) == sorted(readings)
    for row in rows:
        reading = readings[row.id]
        assert isinstance(reading, Reading) and reading.within_interval is not None
        low, high = reading.within_interval
        assert (row.value, row.lo, row.hi) == (reading.change, low, high)
        beyond = high < 0 or low > 0
        assert (bool(row.strong), bool(row.muted)) == (beyond, not beyond)


def test_diverging_bars_over_contrasts_bold_the_directed_verdicts_and_mute_the_rest() -> None:
    report_input = _input("r5")
    plan = _plan(REFERENCE_PLAN)
    chart = next(spec for spec in plan.summary if spec.kind == "forest")
    data, _ = _drawn(_redrawn(plan, chart, kind="diverging_bars", level=chart.level), report_input)
    contrasts = [row.item for row in query_rows(chart.query, report_input) if isinstance(row.item, Contrast)]
    expected = []
    for contrast in contrasts:
        assert contrast.delta is not None
        factor = _scale_factor(f"contrast.{contrast.arm}.{contrast.measure}.delta", report_input)
        expected.append((contrast.arm, contrast.delta * factor, contrast.verdict in ("regressed", "improved")))
    drawn = [(row.id, row.value, bool(row.strong)) for row in _bars_of(data)]
    assert sorted(drawn) == pytest.approx(sorted(expected))
    assert all(bool(row.muted) != bool(row.strong) for row in _bars_of(data))


def test_bars_over_a_judged_dimension_draw_each_arms_figure_from_zero() -> None:
    report_input = _input("r4")
    plan = _plan(dataset.HERE / "r4" / "anchor.plan.json")
    chart = next(spec for spec in plan.summary if spec.kind == "bars")
    data, _ = _drawn(_redrawn(plan, chart, kind="bars", source="judged", measures=["persona"]), report_input)
    factor = _scale_factor(f"judged.persona.{report_input.arms[0].id}", report_input)
    expected = {arm: (stat.mean or 0.0) * factor for arm, stat in report_input.judged["persona"].per_arm.items()}
    assert {row.id: row.value for row in _bars_of(data)} == pytest.approx(expected)
    assert data.domain[0] == 0


@pytest.mark.parametrize("figure", ["passed", "reply_cost_usd"])
def test_bars_over_the_scoreboard_draw_each_arms_figure_money_per_reply_per_thousand(figure: str) -> None:
    report_input = _input("r5")
    plan = _plan(REFERENCE_PLAN)
    chart = next(spec for spec in plan.summary if spec.kind == "bars")
    data, _ = _drawn(_redrawn(plan, chart, kind="bars", source="scoreboard", measures=[figure]), report_input)
    expected = {
        arm: getattr(row, figure) * _scale_factor(f"scoreboard.{arm}.{figure}", report_input)
        for arm, row in report_input.scoreboard.items()
    }
    assert {row.id: row.value for row in _bars_of(data)} == pytest.approx(expected)


def test_bars_over_strata_draw_each_arms_figure_in_each_stratum_grouped_by_stratum() -> None:
    report_input = _input("r5")
    plan = _plan(REFERENCE_PLAN)
    chart = _section(plan, "strata")
    data, _ = _drawn(_redrawn(plan, chart, kind="bars", transform="none", small_multiples_by="none"), report_input)
    (key,) = chart.query.measures
    control = next(arm.id for arm in report_input.arms if arm.role == "control")
    factor = _scale_factor(f"judged.{key}.{control}", report_input)
    kept = sorted(report_input.strata, key=lambda item: -(item.n_cases or 0))[: chart.query.limit]
    expected = {
        (item.name or item.id.replace("_", " "), arm): (figures[key].mean or 0.0) * factor
        for item in kept
        for arm, figures in item.per_arm.items()
        if key in figures
    }
    assert {(row.group, row.id): row.value for row in _bars_of(data)} == pytest.approx(expected)


def test_bars_over_human_preference_draw_the_share_preferring_the_control_in_each_slice() -> None:
    report_input = _input("r4")
    plan = _plan(dataset.HERE / "r4" / "anchor.plan.json")
    data, _ = _drawn(_redrawn(plan, _section(plan, "users"), kind="bars"), report_input)
    assert report_input.human_pref is not None
    assert [(row.id, row.value) for row in _bars_of(data)] == [
        (item.id, item.prefers_control) for item in report_input.human_pref.rows
    ]


def test_lines_over_tasks_draw_a_line_per_arm_over_the_tasks_in_their_order() -> None:
    report_input = _input("r2")
    plan = _plan(dataset.HERE / "r2" / "anchor.plan.json")
    chart = _section(plan, "tasks")
    data, _ = _drawn(_redrawn(plan, chart, kind="lines", small_multiples_by="none"), report_input)
    assert isinstance(data, LinesChart)
    (key,) = chart.query.measures
    assert data.categories == [task.name or task.id for task in report_input.tasks]
    names = {fact.name: fact.id for fact in plan_facts(report_input).arms}
    assert [names[series.end_label or ""] for series in data.series] == list(chart.query.arms)
    for series in data.series:
        arm = names[series.end_label or ""]
        assert series.values == [task.per_arm[arm][key].mean for task in report_input.tasks]


def test_lines_over_a_measure_with_bins_draw_each_arms_share_in_each_bin() -> None:
    report_input = _input("r2")
    plan = _plan(dataset.HERE / "r2" / "anchor.plan.json")
    key = next(key for key, measure in report_input.measures.items() if measure.bins)
    chart = next(spec for spec in plan.summary if spec.kind == "bars")
    data, _ = _drawn(_redrawn(plan, chart, kind="lines", measures=[key]), report_input)
    assert isinstance(data, LinesChart)
    bins = report_input.measures[key].bins or []
    assert data.categories == [level.name or level.id for level in bins]
    names = {fact.name: fact.id for fact in plan_facts(report_input).arms}
    for series in data.series:
        assert series.values == report_input.measures[key].per_arm[names[series.end_label or ""]].shares
    assert len(data.series) == len(chart.query.arms)


def test_lines_over_a_judged_dimension_draw_the_lead_aspect_then_each_aspect_it_holds_figures_on() -> None:
    base = _input("r4")
    lead = next(aspect for aspect in base.aspects if aspect.asked)
    other = next(aspect for aspect in base.aspects if aspect.id != lead.id)
    data_in = base.model_dump(by_alias=True, exclude_none=True)
    persona = data_in["judged"]["persona"]
    persona["by_aspect"] = {other.id: {arm: {"mean": stat["mean"] - 5} for arm, stat in persona["per_arm"].items()}}
    report_input = ReportInput.model_validate(data_in)
    plan = _plan(dataset.HERE / "r4" / "anchor.plan.json")
    chart = next(spec for spec in plan.summary if spec.kind == "bars")
    data, _ = _drawn(_redrawn(plan, chart, kind="lines", source="judged", measures=["persona"]), report_input)
    assert isinstance(data, LinesChart)
    assert data.categories == [lead.name or lead.id, other.name or other.id], "only aspects the figures sit on"
    names = {fact.name: fact.id for fact in plan_facts(report_input).arms}
    for series in data.series:
        mean = base.judged["persona"].per_arm[names[series.end_label or ""]].mean
        assert mean is not None and series.values == pytest.approx([mean, mean - 5])


# --- the Full results tables --------------------------------------------------------------------------------------

#: Each input beside the plan whose look its tables are drawn under: the r1–r5 anchors and the Kairo reference.
_TABLED = [
    *((name, dataset.HERE / name / "anchor.plan.json") for name in ("r1", "r2", "r3", "r4", "r5")),
    ("r5", REFERENCE_PLAN),
]
_TABLED_IDS = ["r1", "r2", "r3", "r4", "r5", "r5-reference"]

#: A number as a cell prints it: its sign, digits and decimal point, its thousands unseparated.
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


@functools.cache
def _tables(name: str, path: Path) -> tuple[TableBlock, ...]:
    report_input = _input(name)
    return tuple(every_table(report_input, anchor_look(_plan(path), report_input)))


def _full(name: str, path: Path, table: str) -> TableBlock:
    return next(one for one in _tables(name, path) if one.name == table)


def _holding(table: str) -> list[tuple[str, Path]]:
    return [(name, path) for name, path in _TABLED if any(one.name == table for one in _tables(name, path))]


def _number(text: Any) -> float:
    found = _NUMBER.search(str(text).replace(",", ""))
    assert found is not None, f"no number in {text!r}"
    return float(found.group(0))


@pytest.mark.parametrize(("name", "path"), _TABLED, ids=_TABLED_IDS)
def test_every_pill_in_the_full_results_tables_says_its_status_in_words(name: str, path: Path) -> None:
    pills = [cell for table in _tables(name, path) for row in table.rows for cell in row.values() if cell.status]
    for cell in pills:
        assert cell.value is not None and re.search(r"[A-Za-z]{3}", cell.text), (
            f"a {cell.status} pill reads {cell.text!r}"
        )


@pytest.mark.parametrize(("name", "path"), _TABLED, ids=_TABLED_IDS)
def test_arms_by_measure_is_every_arm_and_each_figure_it_holds(name: str, path: Path) -> None:
    report_input = _input(name)
    table = _full(name, path, "arms_by_measure")
    names = {arm.id: (arm.short_name or arm.name or arm.id) for arm in report_input.arms}
    assert [row["arm"].value for row in table.rows] == [names[arm.id] for arm in report_input.arms]
    for arm, row in zip(report_input.arms, table.rows, strict=True):
        for key, judged in report_input.judged.items():
            stat = judged.per_arm.get(arm.id)
            if f"judged:{key}" in row and stat is not None and stat.mean is not None:
                scale = 100.0 if judged.scale == "pass_fail" and float(stat.mean) <= 1 else 1.0
                assert math.isclose(_number(row[f"judged:{key}"].value), float(stat.mean) * scale, rel_tol=0.01)


@pytest.mark.parametrize(("name", "path"), _TABLED, ids=_TABLED_IDS)
def test_arms_by_measure_states_each_figure_in_one_column(name: str, path: Path) -> None:
    headers = [column.header for column in _full(name, path, "arms_by_measure").columns]
    assert len(headers) == len(set(headers)), headers


def test_r2_arms_by_measure_reads_its_pass_rate_off_its_measure_once() -> None:
    """r2's arms count no passes and its input holds a ``pass_rate`` measure: the measure's column is the one."""
    keys = [column.key for column in _full("r2", dataset.HERE / "r2" / "anchor.plan.json", "arms_by_measure").columns]
    assert "pass_rate" not in keys and keys.count("measure:pass_rate") == 1, keys


@pytest.mark.parametrize(("name", "path"), _holding("contrasts"))
def test_every_corrected_comparison_is_a_row_its_change_the_input_s_on_its_scale(name: str, path: Path) -> None:
    report_input = _input(name)
    table = _full(name, path, "contrasts")
    assert len(table.rows) == len(report_input.contrasts)
    for contrast, row in zip(report_input.contrasts, table.rows, strict=True):
        if contrast.delta is None:
            continue
        drawn = _number(row["change"].value)
        # The scale is the drawn one: a rate in points, money per reply per 1,000 replies, any other figure as it is.
        assert any(
            math.isclose(drawn, float(contrast.delta) * factor, rel_tol=0.01, abs_tol=0.051)
            for factor in (1.0, 100.0, 1000.0)
        ), f"{contrast.arm} on {contrast.measure}: {row['change'].value} is not {contrast.delta} on any scale"
        assert (row["verdict"].status == "bad") == (contrast.verdict == "regressed")


@pytest.mark.parametrize(("name", "path"), _holding("gate_matrix_full"))
def test_the_gate_matrix_fails_an_arm_on_a_row_exactly_where_the_gate_check_fails_it(name: str, path: Path) -> None:
    report_input = _input(name)
    table = _full(name, path, "gate_matrix_full")
    worst = {"good": 0, "warn": 1, "bad": 2}
    for arm in report_input.arms:
        statuses = [row[f"arm:{arm.id}"].status for row in table.rows]
        drawn = max(worst[status or "good"] for status in statuses)
        assert drawn == {"pass": 0, "borderline": 1, "fail": 2}[gate_check(report_input, arm.id).status], arm.id
        figures = [row[f"arm:{arm.id}"].flag for row in table.rows]
        assert all(figures), "every cell states its figure, or that there is none"


@pytest.mark.parametrize(("name", "path"), _TABLED, ids=_TABLED_IDS)
def test_each_slice_table_is_a_row_per_slice_and_measure_it_holds(name: str, path: Path) -> None:
    report_input = _input(name)
    held = {
        "strata": report_input.strata,
        "tasks": report_input.tasks,
        "cells": list(report_input.cells.values()),
    }
    for table in _tables(name, path):
        if table.name not in held:
            continue
        slices = held[table.name]
        expected = sum(len({key for figures in one.per_arm.values() for key in figures}) for one in slices)
        assert len(table.rows) == expected
        cases = [one.n_cases for one in slices for _ in {key for figures in one.per_arm.values() for key in figures}]
        assert [row["cases"].value for row in table.rows] == cases


@pytest.mark.parametrize(("name", "path"), _holding("configs"))
def test_every_configuration_is_its_arm_s_level_of_each_factor(name: str, path: Path) -> None:
    report_input = _input(name)
    table = _full(name, path, "configs")
    assert len(table.rows) == len(report_input.configs)
    for config, row in zip(report_input.configs, table.rows, strict=True):
        for factor in report_input.factors:
            level = next(one for one in factor.levels if one.id == config.levels[factor.id])
            assert row[f"factor:{factor.id}"].value == (level.name or level.id)
