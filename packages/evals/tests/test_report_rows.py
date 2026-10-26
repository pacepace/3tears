"""What a chart query selects from the report input (``threetears.evals.analysis.report_rows``).

Contracts held here, over the r1–r5 fixture inputs (``fixtures/report_plans/rN/input.json``):

- each source's rows are the input's own records, labelled, with the arm, key and interval the module's table names,
  in the input's order, and every filter (arms, measures, strata, aspect) keeps what it says;
- :func:`query_aspect` is the named aspect, else the one asked aspect (the only one asked, else the only one);
- the limit: strata keep the largest by cases, largest first, ties in the input's order and an uncounted stratum
  last; examples keep the first in the input's order, so the Kairo plan's examples query gives T15, T44, T213 and
  T215; a limit on any other source keeps every row;
- every chart of every anchor plan selects rows, and a forest's rows each carry an interval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.fixtures.report_plans import dataset
from packages.evals.tests.report_kit_support import KIT
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_rows import SourceRow, query_aspect, query_rows
from threetears.evals.kernel.report_plan import ChartQuery

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "report_plans"
ASPECT = "character adherence (v2)"
FLOOR = "character adherence (v2)/rate_drop/*"


def _query(source: str, **fields: Any) -> ChartQuery:
    base: dict[str, Any] = {
        "source": source,
        "aspect": "",
        "measures": [],
        "arms": [],
        "strata": [],
        "limit": 0,
        "transform": "none",
        "small_multiples_by": "none",
    }
    return ChartQuery.model_validate(base | fields)


def _data(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((FIXTURES / name / "input.json").read_text(encoding="utf-8"))
    return loaded


def _labels(rows: list[SourceRow]) -> list[str]:
    return [row.label for row in rows]


def _rows(name: str, source: str, **fields: Any) -> list[SourceRow]:
    return query_rows(_query(source, **fields), dataset.report_input_of(name))


# --- the aspect a query reads --------------------------------------------------------------------------------------


def test_the_aspect_a_query_reads_is_the_named_one_else_the_one_asked() -> None:
    r4 = dataset.report_input_of("r4")
    named, asked = query_aspect("facts", r4), query_aspect("", r4)
    assert named is not None and named.id == "facts"
    assert asked is not None and asked.id == "persona", "r4 asks about persona alone of its four aspects"
    assert query_aspect("no such aspect", dataset.report_input_of("r5")) is None
    assert query_aspect("", dataset.report_input_of("r1")) is None, "r1 holds no aspect"


def test_with_none_named_and_no_one_asked_the_query_reads_no_aspect_unless_the_input_holds_one() -> None:
    data = _data("r4")
    data["aspects"][1]["asked"] = True
    assert query_aspect("", ReportInput.model_validate(data)) is None, "two asked aspects: no one aspect"
    data = _data("r5")
    data["aspects"][0]["asked"] = None
    only = query_aspect("", ReportInput.model_validate(data))
    assert only is not None and only.id == ASPECT, "the only aspect, asked or not"


# --- each source ---------------------------------------------------------------------------------------------------


def test_readings_rows_are_the_query_aspects_readings_with_their_intervals() -> None:
    r5 = dataset.report_input_of("r5")
    rows = query_rows(_query("readings"), r5)
    readings = r5.aspects[0].readings
    assert _labels(rows) == ["qwen3.8-flash", "1017"]
    assert [row.item for row in rows] == readings
    assert [row.arm for row in rows] == ["qwen3.8-flash", "1017"]
    assert [row.interval for row in rows] == [reading.within_interval for reading in readings]
    assert rows[0].interval == (-10.79, -3.381)
    assert _labels(query_rows(_query("readings", arms=["1017"]), r5)) == ["1017"]
    assert _labels(query_rows(_query("readings", aspect=ASPECT), r5)) == ["qwen3.8-flash", "1017"]
    assert _rows("r4", "readings") == [], "r4's asked aspect carries no reading"
    assert _rows("r1", "readings") == [], "r1 holds no aspect to read"


def test_contrasts_rows_keep_the_querys_arms_measures_and_aspect() -> None:
    r5 = _rows("r5", "contrasts", measures=["persona.holds_character"])
    assert _labels(r5) == ["qwen3.8-flash/persona.holds_character", "1017/persona.holds_character"]
    assert [(row.arm, row.key) for row in r5] == [
        ("qwen3.8-flash", "persona.holds_character"),
        ("1017", "persona.holds_character"),
    ]
    assert all(row.interval is not None for row in r5)
    assert _labels(_rows("r4", "contrasts", arms=["f16"])) == ["f16/persona", "f16/task", "f16/facts", "f16/specul"]
    assert len(_rows("r4", "contrasts")) == 12
    assert _rows("r4", "contrasts", aspect="persona") == [], "r4's contrasts name no aspect"
    data = _data("r4")
    data["contrasts"][0]["aspect"] = "persona"
    on_persona = query_rows(_query("contrasts", aspect="persona"), ReportInput.model_validate(data))
    assert _labels(on_persona) == ["f64/persona"]


def test_measures_and_judged_rows_are_each_arms_figure_on_each_kept_key() -> None:
    r4 = _rows("r4", "measures", measures=["cost_per_1k"])
    assert _labels(r4) == ["cost_per_1k/prod", "cost_per_1k/f64", "cost_per_1k/f16", "cost_per_1k/cor"]
    assert [(row.arm, row.key, row.interval) for row in r4][0] == ("prod", "cost_per_1k", None)
    report_input = dataset.report_input_of("r5")
    cost = query_rows(_query("measures", measures=["production_replicating_cost"]), report_input)
    per_arm = report_input.measures["production_replicating_cost"].per_arm
    assert [row.item for row in cost] == list(per_arm.values())
    assert [row.interval for row in cost] == [stat.ci for stat in per_arm.values()]
    assert all(row.interval is not None for row in cost)
    judged = _rows("r5", "judged", measures=["persona.holds_character"], arms=["1017", "September"])
    assert _labels(judged) == ["persona.holds_character/September", "persona.holds_character/1017"], "input order"
    assert len(_rows("r5", "judged")) == 4 * 3
    assert _rows("r1", "judged") == [], "r1's judged dimensions carry no per-arm figure"


def test_a_figure_row_reads_the_query_aspects_figures_where_the_measure_holds_them() -> None:
    data = _data("r4")
    data["measures"]["task"]["by_aspect"] = {"facts": {"f64": {"mean": 0.5}}}
    report_input = ReportInput.model_validate(data)
    on_facts = query_rows(_query("measures", aspect="facts", measures=["task", "specul"]), report_input)
    assert _labels(on_facts) == ["task/f64", "specul/prod", "specul/f64", "specul/f16", "specul/cor"], (
        "task reads its facts figures; specul holds none on facts and reads its own"
    )
    assert on_facts[0].item == report_input.measures["task"].by_aspect["facts"]["f64"]


def test_configs_rows_read_measures_and_judged_for_the_arms_that_ran_a_config() -> None:
    assert len(_rows("r3", "configs", measures=["balanced_accuracy"])) == 9
    rows = _rows("r2", "configs", measures=["pass_rate"], arms=["atlas-m.off.r0", "nano-xs.high.r1"])
    assert _labels(rows) == ["pass_rate/atlas-m.off.r0", "pass_rate/nano-xs.high.r1"]
    data = _data("r4")
    data["configs"] = [{"arm": "f64", "levels": {}}, {"arm": "cor", "levels": {}}]
    grid = query_rows(_query("configs", measures=["task", "persona"]), ReportInput.model_validate(data))
    assert _labels(grid) == ["task/f64", "task/cor", "persona/f64", "persona/cor"], "measures, then judged"


def test_scoreboard_rows_read_the_named_aspects_own_scoreboard_where_it_has_one() -> None:
    r1 = _rows("r1", "scoreboard")
    report_input = dataset.report_input_of("r1")
    assert _labels(r1) == ["atlas-l", "atlas-m", "brisk-s", "open-70b"]
    assert [row.item for row in r1] == list(report_input.scoreboard.values())
    assert all(row.interval is None and row.arm == row.label for row in r1)
    assert _labels(_rows("r1", "scoreboard", arms=["brisk-s"])) == ["brisk-s"]
    data = _data("r4")
    data["aspects"][1]["scoreboard"] = {"prod": {"ran": 3}}
    report_input = ReportInput.model_validate(data)
    assert _labels(query_rows(_query("scoreboard", aspect="task"), report_input)) == ["prod"]
    assert len(query_rows(_query("scoreboard", aspect="facts"), report_input)) == 4, "facts has none of its own"


def test_floors_rows_keep_the_querys_arms_floors_and_aspect() -> None:
    rows = _rows("r5", "floors")
    assert _labels(rows) == [f"{FLOOR}/qwen3.8-flash", f"{FLOOR}/1017"]
    assert [(row.arm, row.key) for row in rows] == [("qwen3.8-flash", FLOOR), ("1017", FLOOR)]
    assert all(row.interval is not None for row in rows)
    assert len(_rows("r5", "floors", aspect=ASPECT, measures=[FLOOR])) == 2
    assert _rows("r5", "floors", measures=["another floor"]) == []
    assert _rows("r5", "floors", arms=["September"]) == [], "the control holds no floor read"


def test_strata_and_tasks_rows_are_the_kept_slices_with_a_figure_for_a_kept_arm() -> None:
    report_input = dataset.report_input_of("r1")
    tasks = query_rows(_query("tasks", strata=["persona", "billing"]), report_input)
    assert _labels(tasks) == ["billing", "persona"], "the input's order"
    assert tasks[0].item is report_input.tasks[0]
    assert all(row.arm == "" and row.interval is None for row in tasks)
    assert len(_rows("r5", "strata")) == 26
    depth = _rows("r4", "strata", arms=["prod"])
    assert _labels(depth) == ["depth:1-2", "depth:3-5", "depth:6-10", "depth:11+"], "prod runs no scenario stratum"


def test_examples_rows_keep_the_querys_arms_strata_and_aspect() -> None:
    assert _labels(_rows("r2", "examples")) == ["TQ-0418", "TQ-0772", "TQ-1103", "TQ-0051"], "a case when no trial"
    assert _labels(_rows("r2", "examples", arms=["nano-xs.low.r1"])) == ["TQ-0051"]
    off_lane = _rows("r5", "examples", strata=["off_lane"])
    assert [(row.label, row.arm) for row in off_lane] == [("T15", "1017"), ("T44", "qwen3.8-flash")]
    data = _data("r5")
    data["examples"][0]["arm"] = None
    data["examples"][1]["aspect"] = "another aspect"
    data["examples"][2]["aspect"] = ASPECT
    report_input = ReportInput.model_validate(data)
    kept = query_rows(_query("examples", aspect=ASPECT, arms=["September"]), report_input)
    assert _labels(kept) == ["T15"], "an example of no arm is kept; one of another aspect, or another arm, is not"
    assert kept[0].arm == ""
    on_aspect = query_rows(_query("examples", aspect=ASPECT, strata=["off_lane", "machinery"]), report_input)
    assert _labels(on_aspect) == ["T15", "T5", "T7"], "an example naming no aspect is kept, one on another is not"


def test_cells_and_confusion_rows_keep_the_querys_cells_and_axes() -> None:
    report_input = dataset.report_input_of("r3")
    cells = query_rows(_query("cells"), report_input)
    assert _labels(cells) == list(report_input.cells)
    assert [row.item for row in cells] == list(report_input.cells.values())
    assert _labels(query_rows(_query("cells", strata=["simple|news"]), report_input)) == ["simple|news"]
    axes = query_rows(_query("confusion"), report_input)
    assert [(row.label, row.key) for row in axes] == [("complexity", "complexity"), ("domain", "domain")]
    assert axes[1].item is report_input.confusion["domain"]
    assert _labels(query_rows(_query("confusion", measures=["domain"]), report_input)) == ["domain"]
    assert _rows("r5", "cells") == [] and _rows("r5", "confusion") == []


def test_human_pref_rows_are_its_slices_with_the_interval_about_each_share() -> None:
    rows = _rows("r4", "human_pref")
    assert _labels(rows) == ["short", "long"]
    assert [row.arm for row in rows] == ["f64", "f64"], "the arm rated against the control"
    assert rows[0].interval == pytest.approx((52.0 - 8.9, 52.0 + 8.9))
    assert rows[1].interval == pytest.approx((61.0 - 8.7, 61.0 + 8.7))
    data = _data("r4")
    data["human_pref"]["rows"][0]["half_width"] = None
    unsure = query_rows(_query("human_pref"), ReportInput.model_validate(data))
    assert unsure[0].interval is None
    assert _rows("r1", "human_pref") == []


def test_human_pref_rows_keep_the_querys_arms_as_every_source_of_one_arm_does() -> None:
    report_input = dataset.report_input_of("r4")
    preference = report_input.human_pref
    assert preference is not None
    every = _rows("r4", "human_pref")
    assert _rows("r4", "human_pref", arms=[preference.arm]) == every
    assert _rows("r4", "human_pref", arms=[preference.control, preference.arm]) == every
    others = [arm.id for arm in report_input.arms if arm.id != preference.arm]
    assert others, "the input must hold an arm the preference does not rate"
    assert _rows("r4", "human_pref", arms=others) == [], "no slice is of an arm the query leaves out"


def test_every_stored_human_pref_query_selects_what_it_did_without_its_arms() -> None:
    """The filter moves no stored chart: each committed plan's preference query keeps its rated arm."""
    plans = [FIXTURES / name / "anchor.plan.json" for name in ("r1", "r2", "r3", "r4", "r5")]
    plans += dataset.run_plans()
    held = 0
    for path in plans:
        plan = dataset.load_plan(path)
        report_input = dataset.report_input_of(dataset.plan_input(path) if "runs" in path.parts else path.parent.name)
        for chart in [*plan.summary, *(section.chart for section in plan.sections)]:
            if chart.query.source != "human_pref":
                continue
            held += 1
            unfiltered = chart.query.model_copy(update={"arms": []})
            assert query_rows(chart.query, report_input) == query_rows(unfiltered, report_input), path
    assert held, "no stored plan reads human preference: the test would hold nothing"


# --- the limit -----------------------------------------------------------------------------------------------------


def test_a_strata_limit_keeps_the_largest_by_cases_largest_first_ties_in_the_inputs_order() -> None:
    assert _labels(_rows("r5", "strata", limit=12)) == [
        "off_lane",
        "machinery",
        "about",
        "superlative",
        "small_talk",
        "multi",
        "emotional",
        "lineage",
        "pushback",
        "rejection",
        "short",
        "story",
    ], "five strata of ten cases tie for the last five places and keep the input's order"
    assert _labels(_rows("r5", "strata", strata=["multi", "about", "slang"], limit=2)) == ["about", "multi"]
    data = _data("r5")
    data["strata"].reverse()
    reordered = query_rows(_query("strata", limit=3), ReportInput.model_validate(data))
    assert _labels(reordered) == ["off_lane", "machinery", "about"], "largest first, whatever the input's order"
    data = _data("r5")
    data["strata"][0]["n_cases"] = None
    uncounted = query_rows(_query("strata", limit=2), ReportInput.model_validate(data))
    assert _labels(uncounted) == ["machinery", "about"], "a stratum without a count ranks as the smallest"


def test_an_examples_limit_keeps_the_first_in_the_inputs_order() -> None:
    assert _labels(_rows("r5", "examples", limit=3)) == ["T15", "T44", "T5"]
    assert len(_rows("r5", "examples")) == 12


def test_the_kairo_plans_examples_query_selects_t15_t44_t213_and_t215() -> None:
    plan = dataset.load_plan(KIT / "references" / "r5-kairo-round2.plan.json")
    (query,) = [section.chart.query for section in plan.sections if section.chart.query.source == "examples"]
    assert query.limit == 4
    assert _labels(query_rows(query, dataset.report_input_of("r5"))) == ["T15", "T44", "T213", "T215"]


def test_a_limit_on_any_other_source_keeps_every_row() -> None:
    assert len(_rows("r1", "tasks", limit=2)) == 5
    assert len(_rows("r1", "scoreboard", limit=1)) == 4
    assert len(_rows("r4", "contrasts", limit=3)) == 12
    assert len(_rows("r3", "cells", limit=2)) == 9


# --- every anchor plan's charts --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["r1", "r2", "r3", "r4", "r5"])
def test_every_chart_of_an_anchor_plan_selects_rows_and_a_forests_carry_intervals(name: str) -> None:
    plan = dataset.load_plan(FIXTURES / name / "anchor.plan.json")
    report_input = dataset.report_input_of(name)
    charts = [*plan.summary, *(section.chart for section in plan.sections)]
    assert charts
    for chart in charts:
        rows = query_rows(chart.query, report_input)
        assert rows, (chart.kind, chart.query.source)
        if chart.kind == "forest":
            assert all(row.interval is not None for row in rows), chart.query
