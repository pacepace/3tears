"""The analyst's fact tools: what each lists is the frozen input's own record, every figure by the key that names it.

Contracts held here (``threetears.evals.analysis.analyst.facts``, ``ops.analyst_facts``, ``actions.analyst_facts``):

- on every fixture snapshot (r1 to r5, the misfits, Kairo round 2), each tool's rows are the input's own records in
  its order, and each figure's value is the input's field it names: ``campaign_overview`` its arms, aspects,
  measures, criteria, caveats, late criteria, prior insights and reader names; ``readings_list`` every reading, pair,
  floor and bar by aspect; ``contrasts_list`` every contrast, its confounds by caveat id, and the families;
  ``measure_read`` an arm's statistics overall, on an aspect, and within each stratum, task and cell, with a slice's
  ``vs_control`` its mean minus the control's there; ``examples_list`` the frozen pool with each trial's stored
  result; ``caveats_list`` every caveat, its materiality, what it touches and the figures it rests on;
- a figure the input holds on more than one aspect is keyed by its aspect: a contrast's arm and measure, a stratum
  id, a contrast family, a measure read on two aspects; a floor or bar on an aspect the input holds no record for is
  listed by its words, with no figure and a note;
- paging: following ``next_offset`` from the first page reads every row once, in order, and ``total`` counts them;
  filters keep only what they name;
- a filter naming nothing the snapshot holds (an arm, an aspect, a measure key, a stratum, a failure class), a slice
  kind the snapshot holds none of, and ``vs_control`` where it cannot be read, are refused as the call, the remedy
  naming what to read instead (``campaign_overview`` for an id);
- each figure resolving to its own value, the text budget and determinism are held for every action by
  ``tests/test_analyst_contract.py``, through ``analyst_support.ANALYST_CALLS``, which calls these tools with the
  filters each input supports.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cache
from typing import Any

import pytest

from packages.evals.tests.analyst_support import ANALYST_CALLS, KAIRO, fixture_snapshots
from packages.evals.tests.fixtures.report_plans.misfits import MISFITS, NAMES
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture
from threetears.evals.actions import ActionOutcome, MountedTool, analyst_tools, analyst_toolset, eval_catalogue
from threetears.evals.analysis.analyst.cited import CitedFigure
from threetears.evals.analysis.analyst.facts import (
    campaign_overview,
    caveats_list,
    contrasts_list,
    examples_list,
    measure_read,
    readings_list,
)
from threetears.evals.analysis.analyst.snapshot import frozen_input
from threetears.evals.analysis.report.figure_keys import reader_names, resolve
from threetears.evals.analysis.report_decision import caveat_materiality, report_caveats, unsettled_arms
from threetears.evals.analysis.report_input import ReportInput, Stat
from threetears.evals.kernel import EvalReportSnapshot
from threetears.evals.kernel.errors import RefusedInput

#: Every fixture snapshot, by name.
FIXTURES = (*NAMES, *MISFITS, KAIRO)

#: The meta fields the grammar keys.
META_FIELDS = (
    "campaign",
    "question",
    "behavior",
    "kind",
    "kind_means",
    "design_shape",
    "run_date",
    "provenance",
    "volume_per_month",
)

#: The scoreboard's counts and costs, as its keys name them.
COUNTS = ("ran", "usable", "passed", "cases", "lost", "failed", "reply_cost_usd", "test_cost_usd")


@cache
def _snapshots() -> dict[str, EvalReportSnapshot]:
    return fixture_snapshots(TOYHOST_SCOPE)


def _held(name: str) -> tuple[EvalReportSnapshot, ReportInput]:
    snapshot = _snapshots()[name]
    return snapshot, frozen_input(snapshot)


def _stat(figure: CitedFigure) -> str:
    """The statistic or field a figure's key ends in."""
    return figure.key.rsplit(".", 1)[1]


def _by_stat(figures: Sequence[CitedFigure]) -> dict[str, Any]:
    return {_stat(figure): figure.value for figure in figures}


def _pair(values: tuple[float, float] | None) -> tuple[float, float] | None:
    return (float(values[0]), float(values[1])) if values is not None else None


def _control(held: ReportInput) -> str | None:
    return next((arm.id for arm in held.arms if arm.role == "control"), None)


def _two_aspects() -> ReportInput:
    """A control and one candidate on two aspects, each holding the same contrast, stratum id, a contrast family, and a
    measure read on both."""
    stratum: dict[str, Any] = {"id": "short", "n_cases": 10}
    figures = {"base": {"m": {"mean": 0.8}}, "a": {"m": {"mean": 0.7}}}
    return ReportInput.model_validate(
        {
            "schema": "report-input/1",
            "meta": {"campaign": "Two", "kind": "change"},
            "arms": [
                {"id": "base", "role": "control", "settings": {}},
                {"id": "a", "role": "challenger", "settings": {}},
            ],
            "aspects": [{"id": "one", "asked": True, "readings": []}, {"id": "two", "readings": []}],
            "scoreboard": {},
            "measures": {
                "m": {
                    "grade": "pass_fail",
                    "higher_is_better": True,
                    "per_arm": {"base": {"mean": 0.9}, "a": {"mean": 0.85}},
                    "by_aspect": {"two": {"base": {"mean": 0.6}, "a": {"mean": 0.5}}},
                }
            },
            "judged": {},
            "contrasts": [
                {"arm": "a", "measure": "m", "aspect": "one", "verdict": "regressed", "delta": -0.05},
                {"arm": "a", "measure": "m", "aspect": "two", "verdict": "improved", "delta": 0.03},
            ],
            "contrast_families": [{"aspect": "one", "correction": "holm"}, {"aspect": "two", "correction": "none"}],
            "strata": [
                {**stratum, "aspect": "one", "name": "Short on one", "per_arm": figures},
                {**stratum, "aspect": "two", "name": "Short on two", "per_arm": figures},
            ],
            "caveats": {},
        }
    )


# =============================================================================
# campaign_overview
# =============================================================================


@pytest.mark.parametrize("name", FIXTURES)
def test_the_overview_lists_the_inputs_own_arms_aspects_measures_criteria_and_caveats(name: str) -> None:
    snapshot, held = _held(name)
    overview = campaign_overview(held, snapshot.id)

    assert overview.snapshot == snapshot.id
    stated_meta = {
        field: value
        for field in META_FIELDS
        if (value := getattr(held.meta, field)) is not None and (not isinstance(value, str) or value.strip())
    }
    assert {_stat(figure): figure.value for figure in overview.meta} == stated_meta
    assert overview.n_candidates.value == sum(arm.role == "challenger" for arm in held.arms)

    assert [row.arm.value for row in overview.arms] == [arm.id for arm in held.arms]
    for row, arm in zip(overview.arms, held.arms, strict=True):
        assert (row.role, row.short_name, row.served_model) == (arm.role, arm.short_name, arm.served_model)
        assert row.settings == {setting: str(value) for setting, value in arm.settings.items() if value is not None}
        board = held.scoreboard.get(arm.id)
        if board is None:
            assert row.scoreboard == []
            continue
        stated = {count: getattr(board, count) for count in COUNTS if getattr(board, count) is not None}
        expected = {count: sum(value.values()) if isinstance(value, dict) else value for count, value in stated.items()}
        assert _by_stat(row.scoreboard) == expected

    assert [row.aspect for row in overview.aspects] == [aspect.id for aspect in held.aspects]
    for aspect_row, aspect in zip(overview.aspects, held.aspects, strict=True):
        assert (aspect_row.asked, aspect_row.graded_by) == (aspect.asked, aspect.graded_by)
        assert (aspect_row.margin.value if aspect_row.margin else None) == aspect.margin
        assert ("readings" in aspect_row.holds) == bool(aspect.readings)

    assert [(row.key, row.source) for row in overview.measures] == [
        *((key, "measure") for key in held.measures),
        *((key, "judged") for key in held.judged),
    ]
    assert [(row.id, row.measure, row.threshold.value) for row in overview.criteria] == [
        (criterion.id, criterion.measure, float(criterion.threshold)) for criterion in held.criteria
    ]
    assert [(ref.id, ref.kind, ref.reach) for ref in overview.caveats] == [
        (caveat.id, caveat.kind, caveat.reach) for caveat in report_caveats(held)
    ]
    assert overview.late_criteria == held.caveats.late_criteria
    assert overview.prior_insights == held.prior_insights
    assert overview.names == {kind: dict(names) for kind, names in reader_names(held).items()}
    assert [figure.key for figure in overview.cells] == [
        f"cells.{cell.row}.{cell.column}.name" for cell in held.cells.values()
    ]


def test_the_overview_names_each_measures_slices_and_the_key_form_that_reads_it() -> None:
    _snapshot, r1 = _held("r1")
    rows = {row.key: row for row in campaign_overview(r1, "s").measures}
    assert all(row.slices == ["task"] for row in rows.values() if row.source == "judged"), "r1 judges by task alone"
    assert rows["latency_s"].key_form == "measure.latency_s.<arm>.<statistic>"

    two = {row.key: row for row in campaign_overview(_two_aspects(), "s").measures}
    assert two["m"].aspects == ["one", "two"]
    assert two["m"].key_form == "aspects.<aspect>.measure.m.<arm>.<statistic>"
    assert two["m"].slices == ["arm", "stratum"]


def test_the_overview_cites_when_the_late_criteria_were_set_where_they_share_one_date() -> None:
    _snapshot, r5 = _held("r5")
    overview = campaign_overview(r5, "s")
    assert {late.set_at for late in r5.caveats.late_criteria} == {"25 Oct 2026"}
    assert overview.late_set_at is not None and overview.late_set_at.value == "25 Oct 2026"
    assert campaign_overview(_held("r4")[1], "s").late_set_at is None, "the control: r4 sets nothing late"


# =============================================================================
# readings_list
# =============================================================================


@pytest.mark.parametrize("name", FIXTURES)
def test_readings_list_every_reading_pair_floor_and_bar_by_its_aspect(name: str) -> None:
    _snapshot, held = _held(name)
    listing = readings_list(held, limit=200)
    groups = {group.aspect: group for group in listing.aspects}
    total = 0
    for aspect in held.aspects:
        floors = [read for read in held.floors if read.aspect == aspect.id]
        bars = [read for read in held.bars if read.aspect == aspect.id]
        if not (aspect.readings or aspect.pairs or floors or bars):
            assert aspect.id not in groups
            continue
        group = groups[aspect.id]
        assert group.note is None
        assert [row.arm.value for row in group.readings] == [reading.arm for reading in aspect.readings]
        for row, reading in zip(group.readings, aspect.readings, strict=True):
            figures = _by_stat(row.figures)
            assert figures["change"] == reading.change and figures["verdict"] == reading.reading
            assert figures.get("interval") == _pair(reading.within_interval)
            assert all(figure.key.startswith(f"aspects.{aspect.id}.reading.{reading.arm}.") for figure in row.figures)
        assert [[arm.value for arm in row.arms] for row in group.pairs] == [list(pair.arms) for pair in aspect.pairs]
        assert [_by_stat(row.figures)["change"] for row in group.pairs] == [pair.change for pair in aspect.pairs]
        assert [(row.floor, row.arm, row.status) for row in group.floors] == [
            (r.floor, r.arm, r.status) for r in floors
        ]
        for row, read in zip(group.floors, floors, strict=True):
            assert _by_stat(row.figures).get("interval") == _pair(read.interval)
        assert [(row.measure, row.arm, row.status) for row in group.bars] == [
            (r.measure, r.arm, r.status) for r in bars
        ]
        total += len(aspect.readings) + len(aspect.pairs) + len(floors) + len(bars)
    held_ids = {aspect.id for aspect in held.aspects}
    unheld = [read for read in [*held.floors, *held.bars] if read.aspect not in held_ids]
    for read in unheld:
        group = groups[read.aspect]
        assert group.note is not None and "no figure key reads" in group.note
        assert all(row.figures == [] for row in [*group.floors, *group.bars])
    assert listing.total == total + len(unheld)
    assert listing.next_offset is None


def test_a_bar_on_an_aspect_the_input_holds_no_record_for_is_listed_by_its_words_alone() -> None:
    _snapshot, held = _held("r1 reading only bars")
    assert held.aspects == [] and held.bars, "the control: bars on an aspect the input does not hold"
    (group,) = readings_list(held).aspects
    assert [(row.measure, row.arm, row.status) for row in group.bars] == [
        (read.measure, read.arm, read.status) for read in held.bars
    ]
    assert group.margin is None and group.family == []


def test_readings_list_keeps_only_the_arm_and_aspect_named() -> None:
    _snapshot, held = _held(KAIRO)
    candidate = held.aspects[0].readings[-1].arm
    narrowed = readings_list(held, arm_filter=candidate)
    for group in narrowed.aspects:
        assert {row.arm.value for row in group.readings} == {candidate}
        assert {row.arm for row in group.floors} <= {candidate}
    assert readings_list(held, aspect_filter=held.aspects[0].id) == readings_list(held)


# =============================================================================
# contrasts_list
# =============================================================================


@pytest.mark.parametrize("name", FIXTURES)
def test_contrasts_list_every_contrast_its_confounds_by_caveat_id_and_the_families(name: str) -> None:
    _snapshot, held = _held(name)
    listing = contrasts_list(held, limit=200)
    caveat_ids = {caveat.id for caveat in report_caveats(held)}
    assert listing.total == len(held.contrasts)
    assert [(row.arm.value, row.measure, row.aspect) for row in listing.contrasts] == [
        (contrast.arm, contrast.measure, contrast.aspect) for contrast in held.contrasts
    ]
    for row, contrast in zip(listing.contrasts, held.contrasts, strict=True):
        figures = _by_stat(row.figures)
        assert (figures.get("delta"), figures["verdict"]) == (contrast.delta, contrast.verdict)
        assert figures.get("interval") == _pair(contrast.interval)
        for named, dimension in zip(row.confounds, contrast.confounds, strict=True):
            assert named in {f"confound:{dimension}", f"apparatus:{dimension}"} & caveat_ids or named == dimension
    assert [family.aspect for family in listing.families] == [family.aspect for family in held.contrast_families]
    for row, family in zip(listing.families, held.contrast_families, strict=True):
        assert _by_stat(row.figures).get("correction") == family.correction


def test_a_kairo_contrasts_confound_is_named_by_its_caveat() -> None:
    _snapshot, held = _held(KAIRO)
    confounded = [row for row in contrasts_list(held).contrasts if row.confounds]
    assert confounded, "the control: Kairo's 1017 contrasts carry a confound"
    assert {name for row in confounded for name in row.confounds} == {"confound:observed:reasoning_ratio"}


def test_contrasts_list_keeps_only_the_arm_measure_and_aspect_named() -> None:
    _snapshot, held = _held(KAIRO)
    first = held.contrasts[0]
    narrowed = contrasts_list(held, arm_filter=first.arm, measure_filter=first.measure)
    assert [(row.arm.value, row.measure) for row in narrowed.contrasts] == [
        (c.arm, c.measure) for c in held.contrasts if (c.arm, c.measure) == (first.arm, first.measure)
    ]
    on_aspect = contrasts_list(held, aspect_filter=first.aspect)
    assert on_aspect.total == sum(c.aspect == first.aspect for c in held.contrasts)


# =============================================================================
# Held on more than one aspect
# =============================================================================


def test_a_figure_held_on_two_aspects_is_keyed_by_its_aspect_and_reads_that_aspects_record() -> None:
    held = _two_aspects()

    contrasts = contrasts_list(held).contrasts
    for row, contrast in zip(contrasts, held.contrasts, strict=True):
        delta = next(figure for figure in row.figures if _stat(figure) == "delta")
        assert delta.key == f"aspects.{contrast.aspect}.contrast.a.m.delta"
        assert delta.value == contrast.delta == resolve(delta.key, held).value
    families = contrasts_list(held).families
    assert [_by_stat(row.figures)["correction"] for row in families] == ["holm", "none"]
    assert all(figure.key.startswith(f"aspects.{row.aspect}.family.") for row in families for figure in row.figures)

    strata = measure_read(held, "m", by="stratum", statistics=["mean"])
    assert [(row.aspect, row.slice, row.arm.value) for row in strata.rows] == [
        ("one", "short", "base"),
        ("one", "short", "a"),
        ("two", "short", "base"),
        ("two", "short", "a"),
    ]
    assert [slice_row.name.value for slice_row in strata.slices if slice_row.name] == ["Short on one", "Short on two"]
    assert all(
        figure.key.startswith(f"aspects.{row.aspect}.strata.short.") for row in strata.rows for figure in row.figures
    )

    by_arm = measure_read(held, "m", statistics=["mean"])
    assert [(row.aspect, row.arm.value, row.figures[0].value) for row in by_arm.rows] == [
        ("one", "base", 0.9),
        ("one", "a", 0.85),
        ("two", "base", 0.6),
        ("two", "a", 0.5),
    ]
    on_two = measure_read(held, "m", statistics=["mean"], aspect_filter="two")
    assert [(row.arm.value, row.figures[0].key) for row in on_two.rows] == [
        ("base", "aspects.two.measure.m.base.mean"),
        ("a", "aspects.two.measure.m.a.mean"),
    ]


# =============================================================================
# measure_read
# =============================================================================


def _expected_rows(held: ReportInput, key: str, by: str) -> list[tuple[str | None, str, Stat]]:
    """The input's own (slice, arm, figure) records a read by ``by`` lists, in its order."""
    arms = [arm.id for arm in held.arms]
    if by == "arm":
        table = held.measures.get(key) or held.judged[key]
        return [(None, arm, table.per_arm[arm]) for arm in arms if arm in table.per_arm]
    slices: list[tuple[str, dict[str, dict[str, Stat]]]]
    if by == "stratum":
        slices = [(stratum.id, stratum.per_arm) for stratum in held.strata]
    elif by == "task":
        slices = [(task.id, task.per_arm) for task in held.tasks]
    else:
        slices = [(f"{cell.row}.{cell.column}", cell.per_arm) for cell in held.cells.values()]
    return [(at, arm, per_arm[arm][key]) for at, per_arm in slices for arm in arms if key in per_arm.get(arm, {})]


def _all_measure_reads() -> list[tuple[str, str, str]]:
    reads: list[tuple[str, str, str]] = []
    for name in FIXTURES:
        _snapshot, held = _held(name)
        for call in ANALYST_CALLS["measure_read"](held):
            if set(call) == {"measure_key", "by"}:
                reads.append((name, call["measure_key"], call["by"]))
    return reads


@pytest.mark.parametrize(("name", "key", "by"), _all_measure_reads())
def test_measure_read_lists_each_arms_figure_in_each_slice_as_the_input_holds_it(name: str, key: str, by: str) -> None:
    _snapshot, held = _held(name)
    read = measure_read(held, key, by=by, statistics=["mean", "ci", "n", "n_cases"], limit=200)  # type: ignore[arg-type]
    expected = _expected_rows(held, key, by)
    assert [(row.slice, row.arm.value) for row in read.rows] == [(at, arm) for at, arm, _stat in expected]
    assert read.total == len(expected) and read.next_offset is None
    for row, (_at, _arm, stat) in zip(read.rows, expected, strict=True):
        stated = {"mean": stat.mean, "ci": _pair(stat.ci), "n": stat.n, "n_cases": stat.n_cases}
        assert _by_stat(row.figures) == {statistic: value for statistic, value in stated.items() if value is not None}
    slices = list(dict.fromkeys(at for at, _arm, _stat in expected if at is not None))
    assert [slice_row.slice for slice_row in read.slices] == slices


def test_vs_control_is_a_candidates_mean_minus_the_controls_in_each_stratum_and_the_control_has_none() -> None:
    _snapshot, held = _held(KAIRO)
    control = _control(held)
    key = held.primary_judged
    assert control is not None and key is not None
    read = measure_read(held, key, by="stratum", statistics=["vs_control"], limit=200)
    assert read.rows
    for row in read.rows:
        stratum = next(s for s in held.strata if s.id == row.slice)
        if row.arm.value == control:
            assert row.figures == []
            continue
        arm = str(row.arm.value)
        mine, theirs = stratum.per_arm[arm][key].mean, stratum.per_arm[control][key].mean
        assert mine is not None and theirs is not None
        (figure,) = row.figures
        assert figure.derived and figure.value == pytest.approx(mine - theirs)


def test_measure_read_keeps_only_the_arm_named() -> None:
    _snapshot, held = _held("r2")
    arm = held.arms[3].id
    read = measure_read(held, "pass_rate", by="task", arm_filter=arm)
    assert {row.arm.value for row in read.rows} == {arm}
    assert read.total == len(held.tasks)


# =============================================================================
# examples_list
# =============================================================================


@pytest.mark.parametrize("name", FIXTURES)
def test_examples_list_the_frozen_pool_with_each_trials_stored_result(name: str) -> None:
    snapshot, held = _held(name)
    listing = examples_list(held, snapshot.example_result_ids, limit=200)
    assert listing.total == len(held.examples)
    assert [(row.trial, row.case, row.stratum, row.aspect, row.outcome) for row in listing.examples] == [
        (example.trial, example.case, example.stratum, example.aspect, example.outcome) for example in held.examples
    ]
    for row, example in zip(listing.examples, held.examples, strict=True):
        assert (row.arm.value if row.arm else None) == example.arm
        assert row.excerpt == example.excerpt and row.reason == example.reason
        assert row.repeat == (str(example.repeat) if example.repeat is not None else None)
        assert row.result_id == snapshot.example_result_ids.get(example.trial or "")


def test_kairos_examples_name_their_stored_results_and_filter_by_arm_stratum_and_outcome() -> None:
    snapshot, held = _held(KAIRO)
    first = examples_list(held, snapshot.example_result_ids)
    assert all(row.result_id is not None for row in first.examples), "the bundle names every shown trial's result"
    example = held.examples[0]
    assert example.arm is not None and example.stratum is not None and example.outcome is not None
    by_arm = examples_list(held, snapshot.example_result_ids, arm_filter=example.arm, limit=200)
    assert by_arm.total == sum(e.arm in (None, example.arm) for e in held.examples)
    by_stratum = examples_list(held, snapshot.example_result_ids, stratum_filter=example.stratum, limit=200)
    assert {row.stratum for row in by_stratum.examples} == {example.stratum}
    by_outcome = examples_list(held, snapshot.example_result_ids, outcome_filter=example.outcome, limit=200)
    assert by_outcome.total == sum(e.outcome == example.outcome for e in held.examples)


# =============================================================================
# caveats_list
# =============================================================================


@pytest.mark.parametrize("name", FIXTURES)
def test_caveats_list_every_caveat_its_materiality_and_what_it_touches(name: str) -> None:
    _snapshot, held = _held(name)
    rows = caveats_list(held).caveats
    materiality = caveat_materiality(held)
    assert [(row.id, row.kind, row.reach) for row in rows] == [
        (caveat.id, caveat.kind, caveat.reach) for caveat in report_caveats(held)
    ]
    for row in rows:
        assert (row.immaterial, row.checked) == (materiality[row.id].immaterial, materiality[row.id].checked)
        head, _, rest = row.id.partition(":")
        if head == "confound":
            naming = [c for c in held.contrasts if rest in c.confounds]
            assert row.touches_arms == list(dict.fromkeys(c.arm for c in naming))
            assert row.touches_measures == list(dict.fromkeys(c.measure for c in naming))
            assert row.says == held.caveats.confounds[rest]
        if head == "short_run":
            run = next(r for r in held.caveats.short_runs if r.run == rest)
            assert row.touches_arms == ([run.arm] if run.arm else [arm.id for arm in held.arms])
            assert row.says == run.note
        if head == "unsettled":
            assert row.touches_arms == list(unsettled_arms(held)[row.id]) and row.touches_aspects == [rest]
        if row.id == "latency":
            assert [(figure.key, figure.value) for figure in row.figures] == [("caveats.latency", held.caveats.latency)]
        if row.kind == "late" and len({late.set_at for late in held.caveats.late_criteria}) == 1:
            assert ("late_criteria.set_at", held.caveats.late_criteria[0].set_at) in {
                (figure.key, figure.value) for figure in row.figures
            }


def test_a_late_floor_touches_the_candidates_it_was_read_on_and_a_late_control_the_control() -> None:
    _snapshot, held = _held("r5")
    rows = {row.id: row for row in caveats_list(held).caveats}
    floor = next(late for late in held.caveats.late_criteria if late.criterion == "floor")
    late_floor = rows[f"late:floor:{floor.name}"]
    reads = [read for read in held.floors if read.floor == floor.name]
    assert reads, "the control: r5 reads its late floor"
    assert late_floor.touches_arms == [read.arm for read in reads]
    assert late_floor.touches_aspects == list(dict.fromkeys(read.aspect for read in reads))
    assert rows["late:control"].touches_arms == [_control(held)]
    assert rows["late:aspect"].touches_aspects == [aspect.id for aspect in held.aspects if aspect.asked]


# =============================================================================
# Paging
# =============================================================================


def _rows_of(listing: Any) -> list[Any]:
    if hasattr(listing, "contrasts"):
        return list(listing.contrasts)
    if hasattr(listing, "examples"):
        return list(listing.examples)
    if hasattr(listing, "rows"):
        return list(listing.rows)
    return [row for group in listing.aspects for row in [*group.readings, *group.pairs, *group.floors, *group.bars]]


@pytest.mark.parametrize(
    ("name", "listing"),
    [
        (KAIRO, lambda held, results, **page: readings_list(held, **page)),
        (KAIRO, lambda held, results, **page: contrasts_list(held, **page)),
        (KAIRO, lambda held, results, **page: examples_list(held, results, **page)),
        (KAIRO, lambda held, results, **page: measure_read(held, held.primary_judged, by="stratum", **page)),
        ("r2", lambda held, results, **page: measure_read(held, "pass_rate", by="task", **page)),
    ],
    ids=["readings", "contrasts", "examples", "measure by stratum", "measure by task"],
)
def test_following_next_offset_from_the_first_page_reads_every_row_once_in_order(
    name: str, listing: Callable[..., Any]
) -> None:
    snapshot, held = _held(name)
    whole = listing(held, snapshot.example_result_ids, limit=200, offset=0)
    assert whole.next_offset is None
    size = max(1, whole.total // 3)
    pages: list[Any] = []
    offset: int | None = 0
    while offset is not None:
        page = listing(held, snapshot.example_result_ids, limit=size, offset=offset)
        assert (page.total, page.offset) == (whole.total, offset)
        pages += _rows_of(page)
        offset = page.next_offset
    assert whole.total > size, "the control: more than one page"
    assert pages == _rows_of(whole) and len(pages) == whole.total


# =============================================================================
# Through the analyst's tool: wiring and refusals
# =============================================================================


def _analyst_tool() -> MountedTool:
    (spec,) = analyst_tools()
    return eval_catalogue().mount(spec)


def _seeded() -> OpsFixture:
    fixture = ops_fixture()
    for snapshot in _snapshots().values():
        fixture.host.eval_host.storage.save_report_snapshot(snapshot)
    return fixture


async def _call(fixture: OpsFixture, arguments: dict[str, Any]) -> ActionOutcome:
    return await _analyst_tool().call(arguments, host=fixture.host, caller=CALLER)


def test_the_fact_tools_are_in_the_analysts_and_the_editors_toolsets() -> None:
    names = {"campaign_overview", "readings_list", "contrasts_list", "measure_read", "examples_list", "caveats_list"}
    assert names <= set(analyst_toolset())
    tool = _analyst_tool()
    assert {action.workflow for action in tool.actions if action.name in names} == {"Investigate: facts"}
    assert {action.permission for action in tool.actions if action.name in names} == {"read"}


async def test_a_tool_reads_its_snapshot_and_examples_carry_the_stored_results_the_snapshot_froze() -> None:
    fixture = off_loop(_seeded)
    snapshot = _snapshots()[KAIRO]
    outcome = await _call(fixture, {"action": "examples_list", "snapshot_id": snapshot.id, "limit": 3})
    assert outcome.structured is not None, outcome.text
    assert [row["result_id"] for row in outcome.structured["examples"]] == [
        snapshot.example_result_ids[example.trial or ""] for example in frozen_input(snapshot).examples[:3]
    ]
    overview = await _call(fixture, {"action": "campaign_overview", "snapshot_id": snapshot.id})
    assert overview.structured is not None and overview.structured["snapshot"] == snapshot.id
    assert overview.text.startswith("campaign overview\n")


@pytest.mark.parametrize(
    ("arguments", "refusal"),
    [
        ({"action": "readings_list", "arm_filter": "no-such-arm"}, "arm_filter 'no-such-arm' names nothing"),
        ({"action": "readings_list", "aspect_filter": "no-such-aspect"}, "aspect_filter 'no-such-aspect' names"),
        ({"action": "contrasts_list", "measure_filter": "no-such-measure"}, "measure_filter 'no-such-measure' names"),
        ({"action": "measure_read", "measure_key": "no-such-measure"}, "names no measure or judged dimension"),
        ({"action": "measure_read", "measure_key": "cost_usd", "arm_filter": "x"}, "arm_filter 'x' names nothing"),
        ({"action": "examples_list", "stratum_filter": "no-such-stratum"}, "stratum_filter 'no-such-stratum' names"),
    ],
    ids=[
        "unknown arm",
        "unknown aspect",
        "unknown contrast measure",
        "unknown measure key",
        "unknown measure arm",
        "unknown stratum",
    ],
)
async def test_a_filter_naming_nothing_the_snapshot_holds_is_refused_pointing_to_the_overview(
    arguments: dict[str, Any], refusal: str
) -> None:
    fixture = off_loop(_seeded)
    outcome = await _call(fixture, {**arguments, "snapshot_id": _snapshots()["r5"].id})
    assert outcome.ending == "refused" and outcome.refusal is not None
    assert outcome.refusal.kind == "validation_failed"
    assert refusal in outcome.text and "campaign_overview lists" in outcome.text


@pytest.mark.parametrize(
    ("name", "arguments", "refusal", "remedy"),
    [
        ("r5", {"measure_key": "cost_usd", "by": "task"}, "holds no tasks", "campaign_overview lists"),
        ("r5", {"measure_key": "cost_usd", "statistics": ["vs_control"]}, "by arm there is no slice", "contrasts_list"),
        ("r1", {"measure_key": "resolves", "by": "task", "statistics": ["vs_control"]}, "has no control", "drop vs_"),
    ],
    ids=["no tasks", "vs_control by arm", "vs_control with no control"],
)
async def test_a_read_the_snapshot_cannot_answer_is_refused_naming_what_to_read_instead(
    name: str, arguments: dict[str, Any], refusal: str, remedy: str
) -> None:
    fixture = off_loop(_seeded)
    outcome = await _call(fixture, {"action": "measure_read", **arguments, "snapshot_id": _snapshots()[name].id})
    assert outcome.refusal is not None and outcome.refusal.kind == "validation_failed", outcome.text
    assert refusal in outcome.text and remedy in outcome.text


def test_tasks_are_not_narrowed_by_an_aspect_which_no_task_is_held_on() -> None:
    document = _two_aspects().model_dump(mode="json", by_alias=True)
    document["tasks"] = [{"id": "t", "per_arm": {"base": {"m": {"mean": 0.9}}, "a": {"m": {"mean": 0.8}}}}]
    held = ReportInput.model_validate(document)
    assert [row.arm.value for row in measure_read(held, "m", by="task").rows] == ["base", "a"]
    with pytest.raises(RefusedInput, match="tasks are not held per aspect") as refused:
        measure_read(held, "m", by="task", aspect_filter="one")
    assert refused.value.remedy == "omit aspect_filter"


async def test_an_unknown_failure_class_is_refused_naming_the_ones_held() -> None:
    fixture = off_loop(_seeded)
    snapshot = _snapshots()[KAIRO]
    outcome = await _call(fixture, {"action": "examples_list", "snapshot_id": snapshot.id, "outcome_filter": "nope"})
    assert outcome.refusal is not None and outcome.refusal.kind == "validation_failed"
    held = {example.outcome for example in frozen_input(snapshot).examples if example.outcome}
    assert all(outcome_name in outcome.text for outcome_name in held)
