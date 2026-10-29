"""Five defects a production report page showed, each held on a committed fixture.

- The examples table's Details column reads a case's input as key and value pairs, never as raw JSON, cut or whole.
- The weakest margin's worst case is read on the multiplicity-corrected interval where the input holds one, and the
  figure ``weakest.interval`` (and the planner's facts) say which interval that is.
- A judged dimension no judge tier covers still carries the scale it was recorded on, so a pass/fail one reads as a
  pass rate and never as a 1 to 5 mean.
- A goal check with no declared name is named the same everywhere: a reader name is the measure's printed name.
- An options table's cost cell never leaves a blank: the arm's cost where the page holds one, else ``unpriced``.
"""

from __future__ import annotations

import functools

import pytest

from packages.evals.tests.fixtures.report_plans import dataset
from packages.evals.tests.report_shapes_support import SHAPES
from threetears.evals.analysis.campaign_bundle import build_campaign_bundle
from threetears.evals.analysis.report.block_spec import block_look
from threetears.evals.analysis.report.charts import ExamplesChart
from threetears.evals.analysis.report.figure_keys import reader_name, render, resolve
from threetears.evals.analysis.report.plan_page.build import plan_page
from threetears.evals.analysis.report.plan_page.charting import measure_words
from threetears.evals.analysis.report.plan_page.tables import arms_table
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.analysis.report_margins import weakest_margin
from threetears.evals.analysis.report_plan_facts import fact_arms, plan_facts
from threetears.evals.kernel.numbers import UNPRICED

#: A world as a case gave it, cut short by the snapshot: its last value ends mid-word, its last pair mid-list.
_WORLD = (
    '{"city": "Oslo", "guests": 4, "dietary": ["vegan", "no nuts"], "notes": "The table by the window, away from the '
    'kitchen door, and a high chair if there is one to spare for the youngest of the guests", "tail": {"a": 1'
)


@functools.cache
def _r5() -> ReportInput:
    return dataset.report_input_of("r5")


# --- 1. the examples' details ---------------------------------------------------------------------------------------


def test_a_cut_world_reads_as_key_and_value_pairs_with_long_values_cut_at_a_word() -> None:
    from threetears.evals.analysis.report.plan_page.resolve_examples import readable_fields

    fields = readable_fields("Input", _WORLD)
    assert [field.term for field in fields] == ["Input · city", "Input · guests", "Input · dietary", "Input · notes"]
    assert [field.text for field in fields[:3]] == ["Oslo", "4", "vegan, no nuts"]
    notes = fields[3].text
    assert notes.endswith("…") and len(notes) <= 81
    assert notes.rstrip("…").rsplit(" ", 1)[0] in _WORLD  # cut at a word, never mid-word
    assert not any(mark in field.term + field.text for field in fields for mark in '{}"')


def test_a_plain_input_stays_one_field_as_written() -> None:
    from threetears.evals.analysis.report.plan_page.resolve_examples import readable_fields

    (field,) = readable_fields("Input", "Where can I park?")
    assert (field.term, field.text) == ("Input", "Where can I park?")


def test_a_text_opening_on_a_brace_that_is_not_json_stays_one_field_and_an_empty_object_gives_none() -> None:
    from threetears.evals.analysis.report.plan_page.resolve_examples import readable_fields

    for text in ("{customer} asked for a refund", "{", '{"cut'):
        (field,) = readable_fields("Input", text)
        assert (field.term, field.text) == ("Input", text), "the input never vanishes from its card"
    assert readable_fields("Input", "{}") == []


def test_the_examples_table_details_column_holds_no_json() -> None:
    held = _r5()
    examples = [
        example.model_copy(update={"input": _WORLD, "expected": '{"table": "A4"}'}) for example in held.examples
    ]
    moved = held.model_copy(update={"examples": examples})
    for name, label, path in _plans_with_examples():
        page = plan_page(dataset.load_plan(path), moved)
        blocks = [*(one.chart for one in page.summary), *(section.chart.chart for section in page.sections)]
        charts = [block.data for block in blocks if isinstance(block.data, ExamplesChart)]
        assert charts, (name, label)
        for chart in charts:
            headers, rows = chart.table([])
            details = [row[headers.index("Details")] for row in rows]
            assert details and all("Input · city: Oslo" in text and "Expected · table: A4" in text for text in details)
            assert not any("{" in text or '"' in text for text in details)


def _plans_with_examples() -> list[tuple[str, str, object]]:
    found = []
    for name in ("r1", "r2", "r3", "r4", "r5"):
        for path in (dataset.HERE / name / "anchor.plan.json", dataset.current_run(name)):
            plan = dataset.load_plan(path)
            drawn = [chart for chart in plan.summary if chart.kind == "examples"]
            drawn += [section.chart for section in plan.sections if section.chart.kind == "examples"]
            if drawn and name == "r5":
                found.append((name, path.stem, path))
    return found


# --- 2. the weakest margin on the corrected interval ----------------------------------------------------------------


def test_the_weakest_margins_worst_case_is_read_on_the_corrected_interval_and_says_so() -> None:
    held = _r5()
    margin = weakest_margin(held)
    assert margin is not None and margin.kind == "margin"
    reading = next(r for a in held.aspects for r in a.readings if r.arm == margin.arm)
    contrast = next(c for c in held.contrasts if c.kind == "trial" and c.arm == margin.arm)
    assert reading.within_interval is not None and contrast.interval is not None
    plain_worst, corrected_worst = float(reading.within_interval[0]), float(contrast.interval[0]) * 100
    assert corrected_worst < plain_worst  # the corrected interval reaches further (10.8 against 12.6 on Kairo)
    assert margin.corrected and margin.value == pytest.approx(corrected_worst)
    said = render("{weakest.interval}", held, {})
    assert said.startswith("the multiplicity-corrected 99.69% interval")
    assert "weakest.interval" in plan_facts(held).weakest.keys  # type: ignore[union-attr]
    assert "read on the multiplicity-corrected" in plan_facts(held).weakest.said  # type: ignore[union-attr]


def test_a_margin_with_no_corrected_interval_says_it_is_read_on_the_plain_one() -> None:
    held = _r5()
    bare = held.model_copy(update={"contrasts": [c for c in held.contrasts if c.kind != "trial"]})
    margin = weakest_margin(bare)
    assert margin is not None and not margin.corrected
    assert resolve("weakest.interval", bare).value == "the plain 95% interval, uncorrected"


# --- 3. a recorded scale, with no judge tier to say it ---------------------------------------------------------------


def test_a_dimension_no_judge_tier_covers_keeps_the_scale_it_was_recorded_on() -> None:
    shape = SHAPES["s3"]()
    built = build_campaign_bundle(
        shape.campaign, list(shape.bundles.values()), arms=shape.arms, readings=shape.readings
    )
    with_tiers = report_input(built)
    for aspect in shape.bundles.values():
        aspect.judge_evidence_tiers = []
    bare = build_campaign_bundle(shape.campaign, list(shape.bundles.values()), arms=shape.arms, readings=shape.readings)
    without = report_input(bare)
    assert {key: held.scale for key, held in with_tiers.judged.items()} == {
        "answer.clear": "ordinal",
        "answer.correct": "pass_fail",
    }
    assert {key: held.scale for key, held in without.judged.items()} == {
        key: held.scale for key, held in with_tiers.judged.items()
    }


# --- 4. a header is the measure's printed name -----------------------------------------------------------------------


def test_a_goal_check_with_no_declared_name_is_named_alike_by_every_reader() -> None:
    held = _r5()
    key = next(key for key in held.measures if key.startswith("goal_state:"))
    check = key.removeprefix("goal_state:")
    labelled = held.model_copy(
        update={"measures": {**held.measures, key: held.measures[key].model_copy(update={"label": check})}}
    )
    printed = measure_words(labelled, key)
    assert printed == f"`{check}`"
    assert reader_name("measure", key, labelled) == printed


# --- 5. a cost cell is a cost or "unpriced" ------------------------------------------------------------------------


def _cost_cells(held: ReportInput) -> dict[str, str | None]:
    table = arms_table(held, block_look(held, fact_arms(held)), ())
    assert any(column.key == "cost_per_1k" for column in table.columns)
    return {arm.id: row["cost_per_1k"].value for arm, row in zip(held.arms, table.rows, strict=True)}  # type: ignore[misc]


def test_an_arm_the_cost_source_lacks_but_the_scoreboard_prices_shows_its_cost_and_one_unknown_says_unpriced() -> None:
    held = _r5()
    first, second, third = (arm.id for arm in held.arms)
    source = "production_replicating_cost"
    measure = held.measures[source]
    rows = dict(held.scoreboard)
    rows[third] = rows[third].model_copy(update={"reply_cost_usd": None, "pricing": None})
    moved = held.model_copy(
        update={
            "measures": {
                **held.measures,
                source: measure.model_copy(
                    update={"per_arm": {arm: stat for arm, stat in measure.per_arm.items() if arm == first}}
                ),
            },
            "scoreboard": rows,
        }
    )
    cells = _cost_cells(moved)
    assert cells[first] is not None and cells[first].startswith("$")
    assert cells[second] is not None and cells[second].startswith("$")  # the scoreboard's cost per reply, same form
    assert cells[third] == UNPRICED
