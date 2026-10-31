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
  id, a contrast family, a measure read on two aspects; a contrast family naming no aspect among several is keyed by
  its place in the input's families; a floor or bar on an aspect the input holds no record for is keyed by the aspect
  its read names, with a note that the aspect has no margin or family;
- the overview cites each arm's settings as the input holds them, each aspect's cases and repeats, each arm's depth
  where an aspect's arms ran it to different depths, and each late criterion's own date;
- paging: following ``next_offset`` from the first page reads every row once, in order, and ``total`` counts them;
  filters keep only what they name;
- a filter naming nothing the snapshot holds (an arm, an aspect, a measure key, a stratum, a failure class), a slice
  kind the snapshot holds none of, and ``vs_control`` where it cannot be read, are refused as the call, the remedy
  naming what to read instead (``campaign_overview`` for an id);
- each figure resolving to its own value, the text budget and determinism are held for every action by
  ``tests/test_analyst_contract.py``, through ``analyst_support.ANALYST_CALLS``, which calls these tools with the
  filters each input supports; here the budget is held on the shape campaigns s1 to s3 as well (s1's three aspects
  are the widest rows), where a page the caller does not size holds as many rows as fit one result and following
  ``next_offset`` from it reads every row once.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cache
from typing import Any

import pytest

from packages.evals.tests.analyst_support import ANALYST_CALLS, KAIRO, TEXT_BUDGET, fixture_snapshots
from packages.evals.tests.fixtures.report_plans.misfits import MISFITS, NAMES
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture
from packages.evals.tests.report_shapes_support import SHAPES
from threetears.evals.actions import ActionOutcome, MountedTool, analyst_tools, eval_catalogue
from threetears.evals.analysis.analyst.cited import CitedFigure
from threetears.evals.analysis.analyst.facts import (
    campaign_overview,
    caveats_list,
    contrasts_list,
    examples_list,
    measure_read,
    readings_list,
)
from threetears.evals.analysis.analyst.snapshot import frozen_input, report_snapshot
from threetears.evals.analysis.campaign_bundle import build_campaign_bundle
from threetears.evals.analysis.report.figure_keys import reader_names, resolve
from threetears.evals.analysis.report_decision import (
    caveat_materiality,
    implied_caveats,
    report_caveats,
    surviving_options,
    unsettled_arms,
)
from threetears.evals.analysis.report_margins import CriterionFigure, criterion_figures
from threetears.evals.analysis.report_input import Aspect, Criterion, ReportInput, Stat
from threetears.evals.analysis.report_input_build import report_input
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


#: The shape campaigns the budget is held on as well: s1 holds three aspects, s2 a choice, s3 a measurement.
SHAPE_NAMES = ("s1", "s2", "s3")


@cache
def _shape_inputs() -> dict[str, ReportInput]:
    """Each shape campaign's report input, built from its campaign bundle."""
    built: dict[str, ReportInput] = {}
    for name in SHAPE_NAMES:
        shape = SHAPES[name]()
        bundle = build_campaign_bundle(
            shape.campaign, list(shape.bundles.values()), arms=shape.arms, readings=shape.readings
        )
        built[name] = report_input(bundle)
    return built


def _setting_values(held: ReportInput, arm_id: str) -> list[tuple[str, Any]]:
    """Each setting an arm states, by its key, and its value as the input holds it."""
    arm = next(arm for arm in held.arms if arm.id == arm_id)
    return [
        (f"settings.{arm.id}.{setting}", value if isinstance(value, str) else float(value))
        for setting, value in arm.settings.items()
        if value is not None
    ]


def _expected_depth(held: ReportInput, aspect: Aspect) -> tuple[float | None, float | None, list[tuple[str, Any]]]:
    """An aspect's one count of cases and of repeats (the control's, else the arms' where they agree), and each arm's
    depth figures where its arms ran it to different depths."""
    depth = aspect.depth or {}
    control = _control(held)
    picked = [depth[control]] if control in depth else list(depth.values())
    cases = {float(d.cases) for d in picked}
    repeats = {float(d.repeats_per_case[1]) for d in picked}
    differ = len({(d.cases, d.trials, tuple(d.repeats_per_case)) for d in depth.values()}) > 1
    per_arm = [
        (f"aspects.{aspect.id}.depth.{arm}.{field}", value)
        for arm, d in depth.items()
        for field, value in (("cases", d.cases), ("trials", d.trials), ("repeats", d.repeats_per_case[1]))
    ]
    return (
        cases.pop() if len(cases) == 1 else None,
        repeats.pop() if len(repeats) == 1 else None,
        per_arm if differ else [],
    )


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


def test_the_overview_cites_each_aspects_pass_rule_by_its_keys() -> None:
    """Kairo's pass rule reads off the overview with every figure under an `aspects.<aspect>.` key; an input frozen
    before the rule existed (r4, four aspects) states none, and its overview leaves the row out."""
    snapshot, held = _held(KAIRO)
    (row,) = campaign_overview(held, snapshot.id).aspects
    rule = row.pass_rule
    assert rule is not None
    keyed = f"aspects.{row.aspect}"
    assert [figure.key for figure in (rule.rule, rule.threshold, rule.no_criterion)] == [
        f"{keyed}.pass_rule",
        f"{keyed}.pass_threshold",
        f"{keyed}.no_criterion",
    ]
    assert (rule.threshold.reads, rule.no_criterion.reads) == ("3", "0")
    assert rule.sentence is not None and rule.sentence.key == f"{keyed}.description"
    assert rule.criteria == ["goal_checks", "rubric"] and row.graded_by == "judged"
    _, r4 = _held("r4")
    assert r4.aspects and all(aspect.pass_rule is None for aspect in r4.aspects)
    assert all("pass_rule" not in aspect for aspect in campaign_overview(r4, "s").model_dump()["aspects"])


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
        assert [(figure.key, figure.value) for figure in row.settings] == _setting_values(held, arm.id)
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
        cases, repeats, per_arm = _expected_depth(held, aspect)
        assert (aspect_row.cases.value if aspect_row.cases else None) == cases
        assert (aspect_row.repeats.value if aspect_row.repeats else None) == repeats
        assert [(figure.key, figure.value) for figure in aspect_row.depth] == per_arm
        rule = aspect.pass_rule
        if rule is None:
            assert aspect_row.pass_rule is None
            continue
        assert aspect_row.pass_rule is not None
        row_rule = aspect_row.pass_rule
        assert (row_rule.rule.value, row_rule.threshold.value, row_rule.no_criterion.value) == (
            rule.label,
            rule.threshold,
            rule.no_criterion,
        )
        assert (row_rule.criteria, row_rule.dimensions, row_rule.boundary, row_rule.checks) == (
            rule.criteria,
            rule.dimensions,
            rule.boundary,
            rule.checks,
        )

    assert [(row.key, row.source) for row in overview.measures] == [
        *((key, "measure") for key in held.measures),
        *((key, "judged") for key in held.judged),
    ]
    assert [(row.id, row.measure, row.threshold.value) for row in overview.criteria] == [
        (criterion.id, criterion.measure, float(criterion.threshold)) for criterion in held.criteria
    ]
    assert [(ref.id, ref.kind, ref.reach, ref.source) for ref in overview.caveats] == [
        *((caveat.id, caveat.kind, caveat.reach, "stated") for caveat in report_caveats(held)),
        *((caveat.id, caveat.kind, caveat.reach, "implied") for caveat in implied_caveats(held)),
    ]
    assert [(row.criterion, row.name, row.set_at.value) for row in overview.late_criteria] == [
        (late.criterion, late.name, late.set_at) for late in held.caveats.late_criteria
    ]
    assert overview.prior_insights == held.prior_insights
    assert overview.names == {kind: dict(names) for kind, names in reader_names(held).items()}
    assert [figure.key for figure in overview.cells] == [
        f"cells.{cell.row}.{cell.column}.name" for cell in held.cells.values()
    ]


#: A judged dimension's judge fields, by the key's last word and the input's attribute.
JUDGE_FIELDS = (
    ("tier", "tier"),
    ("model", "judge_model"),
    ("calibration", "calibration"),
    ("calibration_ratings", "calibration_ratings"),
    ("agreement", "agreement"),
)


def _judge_figures(held: ReportInput, dimension: str) -> list[tuple[str, Any]]:
    """A judged dimension's judge figures as the input states them, by key: a text, or a number as a float."""
    judged = held.judged[dimension]
    stated = [(field, getattr(judged, attribute)) for field, attribute in JUDGE_FIELDS]
    return [
        (f"judged.{dimension}.{field}", value if isinstance(value, str) else float(value))
        for field, value in stated
        if value is not None and (not isinstance(value, str) or value.strip())
    ]


@pytest.mark.parametrize("name", FIXTURES)
def test_the_overview_and_the_uncalibrated_judges_caveat_cite_each_judges_tier_calibration_and_agreement(
    name: str,
) -> None:
    _snapshot, held = _held(name)
    rows = {row.key: row for row in campaign_overview(held, "s").measures}
    for key, row in rows.items():
        expected = _judge_figures(held, key) if row.source == "judged" else []
        assert [(figure.key, figure.value) for figure in row.judge] == expected, key
    caveat = next((row for row in caveats_list(held).caveats if row.id == "judge_uncalibrated"), None)
    if caveat is not None:
        cited = [(figure.key, figure.value) for figure in caveat.figures]
        assert cited == [figure for dimension in held.judged for figure in _judge_figures(held, dimension)]


def test_a_judges_figures_are_cited_where_the_fixtures_state_them() -> None:
    """The control for the walk above: some fixture states a tier and an agreement, and one carries the caveat."""
    stated = [_judge_figures(held, key) for _, held in map(_held, FIXTURES) for key in held.judged]
    keys = {key.rsplit(".", 1)[1] for figures in stated for key, _ in figures}
    assert {"tier", "calibration"} <= keys
    assert any(
        row.id == "judge_uncalibrated" and row.figures
        for _, held in map(_held, FIXTURES)
        for row in caveats_list(held).caveats
    )


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


def test_the_overview_cites_each_late_criterions_own_date_where_they_differ() -> None:
    document = _held("r5")[1].model_dump(mode="json", by_alias=True)
    document["caveats"]["late_criteria"][-1]["set_at"] = "26 Oct 2026"
    held = ReportInput.model_validate(document)
    overview = campaign_overview(held, "s")
    assert overview.late_set_at is None, "the control: no one date"
    assert [row.set_at.value for row in overview.late_criteria] == [l.set_at for l in held.caveats.late_criteria]
    late_ids = {
        f"late:{row.set_at.key.removeprefix('late_criteria.').removesuffix('.set_at')}"
        for row in overview.late_criteria
    }
    assert late_ids == {ref.id for ref in overview.caveats if ref.kind == "late"}


def test_the_overview_cites_s1s_depth_by_aspect_and_each_arms_where_one_ran_twice() -> None:
    held = _shape_inputs()["s1"]
    overview = campaign_overview(held, "s")
    assert overview.cases is None and overview.repeats is None, "the control: s1 asks three aspects"
    rows = {row.aspect: row for row in overview.aspects}
    assert all(row.cases is not None and row.cases.value == 20 for row in rows.values())
    assert rows["billing"].depth == [], "every arm ran billing once per case"
    tone = next(aspect for aspect in held.aspects if aspect.id == "tone")
    assert tone.depth is not None
    assert {figure.key for figure in rows["tone"].depth} >= {"aspects.tone.depth.v3.repeats"}
    v3 = next(figure for figure in rows["tone"].depth if figure.key == "aspects.tone.depth.v3.repeats")
    assert v3.value == tone.depth["v3"].repeats_per_case[1] == 2


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
        assert group.note is not None and "no record for aspect" in group.note
        assert (group.margin, group.family) == (None, [])
        for row in [*group.floors, *group.bars]:
            assert row.figures and all(figure.key.startswith(f"aspects.{read.aspect}.") for figure in row.figures)
            assert _by_stat(row.figures)["status"] == row.status
    assert listing.total == total + len(unheld)
    assert listing.next_offset is None


def test_a_bar_on_an_aspect_the_input_holds_no_record_for_is_keyed_by_the_aspect_its_read_names() -> None:
    _snapshot, held = _held("r1 reading only bars")
    assert held.aspects == [] and held.bars, "the control: bars on an aspect the input does not hold"
    (group,) = readings_list(held).aspects
    assert [(row.measure, row.arm, row.status) for row in group.bars] == [
        (read.measure, read.arm, read.status) for read in held.bars
    ]
    for row, read in zip(group.bars, held.bars, strict=True):
        assert _by_stat(row.figures) == {"status": read.status, "bar": float(read.bar)}
        assert all(
            figure.key.startswith(f"aspects.{read.aspect}.bar.{read.measure}.{read.arm}.") for figure in row.figures
        )
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


def test_a_family_naming_no_aspect_among_several_is_keyed_by_its_place_and_each_reads_its_own() -> None:
    document = _two_aspects().model_dump(mode="json", by_alias=True)
    document["contrast_families"] = [
        {"aspect": "one", "correction": "holm"},
        {"correction": "none", "family_size": 2},
        {"aspect": "two", "correction": "bonferroni"},
        {"aspect": "two", "correction": "none"},
    ]
    document["aspects"][1]["readings"] = [{"arm": "a", "reading": "within_margin", "change": -1.0}]
    held = ReportInput.model_validate(document)
    families = contrasts_list(held).families
    assert [[figure.key.rsplit(".", 1)[0] for figure in row.figures][0] for row in families] == [
        "aspects.one.family",
        "families.1",
        "families.2",
        "families.3",
    ]
    assert [_by_stat(row.figures)["correction"] for row in families] == [f.correction for f in held.contrast_families]
    (group,) = readings_list(held, limit=200).aspects
    assert group.aspect == "two"
    assert {figure.key.rsplit(".", 1)[0] for figure in group.family} == {"families.2", "families.3"}


def test_a_kairo_contrasts_confound_is_named_by_its_caveat() -> None:
    _snapshot, held = _held(KAIRO)
    confounded = [row for row in contrasts_list(held).contrasts if row.confounds]
    assert confounded, "the control: Kairo's contrasts carry a confound"
    named = {name for row in confounded for name in row.confounds}
    caveats = {caveat.id: caveat.kind for caveat in report_caveats(held)}
    assert all(caveats.get(name) in ("confound", "apparatus") for name in named), "each is a caveat the input carries"
    assert {name.partition(":")[2] for name in named} == {
        dimension for contrast in held.contrasts for dimension in contrast.confounds
    }, "each confound a contrast names, by its caveat"


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


#: The statistics a figure stores, in the grammar's order: what a read naming none reads, each where stated.
STORED = ("mean", "ci", "half_width", "sem", "n", "n_cases", "p50", "p95", "change")

#: The statistics that are an interval on a mean.
INTERVALS = frozenset({"ci", "half_width"})


def _stored(stat: Stat) -> dict[str, Any]:
    """Every statistic one figure states, as a cited figure reads it."""
    values = {statistic: getattr(stat, statistic) for statistic in STORED}
    return {
        statistic: (_pair(value) if statistic == "ci" else value)
        for statistic, value in values.items()
        if value is not None
    }


def _slice_stats(held: ReportInput, key: str) -> dict[str, list[Stat]]:
    """Every figure the input holds on one measure, by the way it is read, each way holding one."""
    table = held.measures.get(key) or held.judged[key]
    by = {
        "arm": [*table.per_arm.values(), *(stat for arms in table.by_aspect.values() for stat in arms.values())],
        "stratum": [figures[key] for s in held.strata for figures in s.per_arm.values() if key in figures],
        "task": [figures[key] for t in held.tasks for figures in t.per_arm.values() if key in figures],
        "cell": [figures[key] for c in held.cells.values() for figures in c.per_arm.values() if key in figures],
    }
    return {way: stats for way, stats in by.items() if stats}


def _named_stats(stats: Sequence[Stat]) -> list[str]:
    return [statistic for statistic in STORED if any(getattr(stat, statistic) is not None for stat in stats)]


@pytest.mark.parametrize(("name", "key", "by"), _all_measure_reads())
def test_a_read_naming_no_statistics_reads_every_one_held_its_interval_among_them(name: str, key: str, by: str) -> None:
    _snapshot, held = _held(name)
    read = measure_read(held, key, by=by, limit=200)  # type: ignore[arg-type]
    expected = _expected_rows(held, key, by)
    assert [(row.slice, row.arm.value) for row in read.rows] == [(at, arm) for at, arm, _stat in expected]
    for row, (_at, _arm, stat) in zip(read.rows, expected, strict=True):
        assert _by_stat(row.figures) == _stored(stat)
    assert read.held == _named_stats(_slice_stats(held, key)[by])
    assert read.not_shown == []
    table = held.measures.get(key) or held.judged[key]
    assert read.level == table.level


def _interval_reads() -> list[tuple[str, str, str]]:
    """Every fixture read whose figures hold an interval on the mean."""
    found: list[tuple[str, str, str]] = []
    for name in FIXTURES:
        _snapshot, held = _held(name)
        for key in [*held.measures, *held.judged]:
            for by, stats in _slice_stats(held, key).items():
                if INTERVALS & set(_named_stats(stats)):
                    found.append((name, key, by))
    return found


def test_an_interval_on_a_slice_is_read_without_asking_and_a_narrower_read_names_it_as_not_shown() -> None:
    reads = _interval_reads()
    assert {name for name, _key, _by in reads} >= {"r1", "r3", "r4", "r5"}, "the control: intervals held by slice"
    for name, key, by in reads:
        _snapshot, held = _held(name)
        whole = measure_read(held, key, by=by, limit=200)  # type: ignore[arg-type]
        intervals = INTERVALS & set(whole.held)
        assert intervals and any(intervals & set(_by_stat(row.figures)) for row in whole.rows), (name, key, by)
        narrow = measure_read(held, key, by=by, statistics=["mean"], limit=200)  # type: ignore[arg-type]
        assert set(narrow.not_shown) == set(whole.held) - {"mean"} and intervals <= set(narrow.not_shown)
        assert all(set(_by_stat(row.figures)) <= {"mean"} for row in narrow.rows)


def test_a_measure_held_only_within_slices_is_refused_by_arm_naming_where_it_is_held() -> None:
    refused = 0
    for name in FIXTURES:
        _snapshot, held = _held(name)
        for key in [*held.measures, *held.judged]:
            ways = _slice_stats(held, key)
            if "arm" in ways or not ways:
                continue
            with pytest.raises(RefusedInput, match="holds no figure read by arm") as caught:
                measure_read(held, key)
            assert list(caught.value.valid_values or ()) == list(ways)
            refused += 1
    assert refused, "the control: a fixture holds a measure only within its slices"


@pytest.mark.parametrize("name", FIXTURES)
def test_the_overview_says_which_statistics_each_measure_holds_and_no_arm_lacks_one_held_per_slice(name: str) -> None:
    _snapshot, held = _held(name)
    for row in campaign_overview(held, "s").measures:
        ways = _slice_stats(held, row.key)
        assert row.statistics == [f"{way}: {', '.join(_named_stats(stats))}" for way, stats in ways.items()]
        assert row.slices == list(ways)
        slices = [
            *(s.per_arm for s in held.strata),
            *(t.per_arm for t in held.tasks),
            *(c.per_arm for c in held.cells.values()),
        ]
        table = held.measures.get(row.key) or held.judged[row.key]
        holding = set(table.per_arm) | {arm for arms in table.by_aspect.values() for arm in arms}
        holding |= {arm for per_arm in slices for arm, figures in per_arm.items() if row.key in figures}
        assert row.arms_without == [arm.id for arm in held.arms if arm.id not in holding]


def test_a_measure_held_per_cell_lists_no_arm_without_it() -> None:
    """The r3 discoverability fault: accuracy held per cell read as missing on every arm."""
    held = next(held for _, held in map(_held, FIXTURES) if held.cells)
    per_cell = [
        row
        for row in campaign_overview(held, "s").measures
        if row.slices == ["cell"] and INTERVALS & set(_named_stats(_slice_stats(held, row.key)["cell"]))
    ]
    assert per_cell, "the control: a measure held only per cell, with an interval"
    for row in per_cell:
        assert row.arms_without == []
        assert any(interval in row.statistics[0] for interval in INTERVALS)


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


def test_kairos_pool_holds_more_than_one_failure_of_a_slice_and_a_filter_pages_them() -> None:
    snapshot, held = _held(KAIRO)
    slices: dict[tuple[str | None, str | None], list[str | None]] = {}
    for example in held.examples:
        slices.setdefault((example.arm, example.stratum), []).append(example.trial)
    (arm, stratum), trials = max(slices.items(), key=lambda item: len(item[1]))
    assert len(trials) > 1, "the pool holds more than one failure of some arm and stratum"
    assert arm is not None and stratum is not None
    pages: list[str | None] = []
    offset: int | None = 0
    while offset is not None:
        page = examples_list(
            held, snapshot.example_result_ids, arm_filter=arm, stratum_filter=stratum, limit=1, offset=offset
        )
        pages += [row.trial for row in page.examples]
        offset = page.next_offset
    assert pages == [e.trial for e in held.examples if e.arm in (None, arm) and e.stratum == stratum]
    assert set(trials) <= set(pages)


# =============================================================================
# caveats_list
# =============================================================================


@pytest.mark.parametrize("name", FIXTURES)
def test_caveats_list_every_caveat_its_materiality_and_what_it_touches(name: str) -> None:
    _snapshot, held = _held(name)
    listed = caveats_list(held).caveats
    materiality = caveat_materiality(held)
    stated = report_caveats(held)
    rows = listed[: len(stated)]
    assert [(row.id, row.kind, row.reach, row.source) for row in listed] == [
        *((caveat.id, caveat.kind, caveat.reach, "stated") for caveat in stated),
        *((caveat.id, caveat.kind, caveat.reach, "implied") for caveat in implied_caveats(held)),
    ]
    assert all(row.matters_if is None for row in rows)
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
        if row.kind == "late":
            late = next(
                late
                for late in held.caveats.late_criteria
                if row.id == f"late:{late.criterion}" + (f":{late.name}" if late.name else "")
            )
            assert (f"late_criteria.{row.id.removeprefix('late:')}.set_at", late.set_at) in {
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


#: The fewest candidates a pick among is the best of many (``best_of_n``): the rule the caveat states.
MANY = 3


def _undescribed(held: ReportInput) -> list[str]:
    """The judged dimensions whose judge the input does not describe: no model, tier or calibration."""
    tiers = held.caveats.judge_tiers
    dimensions = [*held.judged, *(name for name in tiers if name not in held.judged)]
    return [
        dimension
        for dimension in dimensions
        if tiers.get(dimension) is None
        and (
            (judged := held.judged.get(dimension)) is None
            or (judged.tier, judged.calibration, judged.judge_model) == (None, None, None)
        )
    ]


def _borders(criterion: Criterion, figure: CriterionFigure) -> bool:
    """Whether a figure clears its criterion's line on its point estimate while its interval reaches past the line."""
    if figure.ci is not None:
        low, high = figure.ci
    elif figure.half_width is not None:
        low, high = figure.value - figure.half_width, figure.value + figure.half_width
    else:
        return False
    line = float(criterion.threshold)
    if criterion.direction == "higher_is_better":
        return figure.value >= line and low < line
    return figure.value <= line and high > line


def _expected_implied(held: ReportInput) -> dict[str, tuple[list[str], list[str]]]:
    """The implied caveats an input's holdings raise, by id, each with the arms and measures it touches, read off the
    input (the judge's partial calibration is held by its own test below)."""
    expected: dict[str, tuple[list[str], list[str]]] = {}
    undescribed = _undescribed(held)
    if undescribed:
        expected["judge_unstated"] = ([], undescribed)
    if len(held.arms) > 1 and _control(held) is None:
        expected["no_control"] = ([], [])
    margins = any(a.margin is not None or any(r.margin is not None for r in a.readings) for a in held.aspects)
    deciding = any(criterion.kind in ("gate", "guardrail", "floor") for criterion in held.criteria)
    if not deciding and not margins and not held.floors:
        expected["nothing_declared"] = ([], [])
    survivors = surviving_options(held)
    for criterion in held.criteria:
        if criterion.kind not in ("gate", "guardrail"):
            continue
        higher = criterion.direction == "higher_is_better"
        cleared = [
            arm
            for arm in survivors
            for figure in criterion_figures(held, criterion, arm)
            if figure.ci is None
            and figure.half_width is None
            and (figure.value >= criterion.threshold if higher else figure.value <= criterion.threshold)
        ]
        if cleared:
            expected[f"point_estimate:{criterion.id}"] = (list(dict.fromkeys(cleared)), [criterion.measure])
        bordering = [
            arm
            for arm in survivors
            for figure in criterion_figures(held, criterion, arm)
            if _borders(criterion, figure)
        ]
        if bordering:
            expected[f"borderline:{criterion.id}"] = (list(dict.fromkeys(bordering)), [criterion.measure])
    candidates = [arm.id for arm in held.arms if arm.role == "challenger"]
    if len(candidates) >= MANY:
        expected["best_of_n"] = (candidates, [])
    return expected


@pytest.mark.parametrize("name", FIXTURES)
def test_caveats_list_raises_what_the_snapshot_implies_each_with_when_it_matters(name: str) -> None:
    _snapshot, held = _held(name)
    implied = [row for row in caveats_list(held).caveats if row.source == "implied"]
    assert {row.id: (row.touches_arms, row.touches_measures) for row in implied} == _expected_implied(held)
    stated = {caveat.id for caveat in report_caveats(held)}
    for row in implied:
        assert row.id not in stated and row.matters_if and not row.immaterial and row.checked is None
        assert row.reach in ("report", "scoped")


def test_every_implied_kind_is_raised_by_some_fixture() -> None:
    """The control for the walk above: each implied kind but a partly calibrated judge is raised on the fixtures."""
    kinds = {row.kind for _, held in map(_held, FIXTURES) for row in caveats_list(held).caveats}
    assert {"judge_unstated", "no_control", "nothing_declared", "point_estimate", "borderline", "best_of_n"} <= kinds


def test_a_gate_cleared_on_a_point_estimate_cites_its_threshold_and_each_clearance() -> None:
    raised = 0
    for _, held in map(_held, FIXTURES):
        for row in caveats_list(held).caveats:
            if row.kind != "point_estimate":
                continue
            criterion = row.id.partition(":")[2]
            keys = [figure.key for figure in row.figures]
            assert keys[0] == f"criteria.{criterion}.threshold"
            assert keys[1:] and all(
                key.startswith(f"criteria.{criterion}.") and key.endswith(".gap") for key in keys[1:]
            )
            assert all(figure.value >= 0 for figure in row.figures[1:]), "a clearance is signed toward passing"
            assert all(any(key.startswith(f"criteria.{criterion}.{arm}.") for key in keys) for arm in row.touches_arms)
            assert all(
                any(key.startswith(f"criteria.{criterion}.{arm}.") for arm in row.touches_arms) for key in keys[1:]
            )
            raised += 1
    assert raised, "the control: a fixture's survivor clears a gate on a point estimate"


def test_a_gate_a_survivor_clears_on_a_borderline_figure_is_a_caveat_citing_its_threshold_and_each_clearance() -> None:
    """r1's pick cleared its resolve gate on its point estimate with an interval reaching below the line, and its
    analyst found no caveat id to carry that with: ``borderline:<criterion>`` is it, citing the line and each gap."""
    raised = 0
    for _, held in map(_held, FIXTURES):
        for row in caveats_list(held).caveats:
            if row.kind != "borderline":
                continue
            criterion = row.id.partition(":")[2]
            keys = [figure.key for figure in row.figures]
            assert row.source == "implied" and row.reach == "report" and row.matters_if
            assert keys[0] == f"criteria.{criterion}.threshold"
            assert keys[1:] and all(
                key.startswith(f"criteria.{criterion}.") and key.endswith(".gap") for key in keys[1:]
            )
            assert all(figure.value >= 0 for figure in row.figures[1:]), "the figure clears the line"
            assert all(any(key.startswith(f"criteria.{criterion}.{arm}.") for key in keys) for arm in row.touches_arms)
            raised += 1
    assert raised, "the control: a fixture's survivor clears a gate on a borderline figure"


def test_a_judge_stated_uncalibrated_on_some_dimensions_is_raised_where_the_whole_judge_is_not() -> None:
    _snapshot, held = _held(KAIRO)
    assert "judge_uncalibrated" in {caveat.id for caveat in report_caveats(held)}, "the control: every one is"
    calibrated = next(iter(held.judged))
    judged = {**held.judged, calibrated: held.judged[calibrated].model_copy(update={"calibration": "calibrated"})}
    tiers = {**held.caveats.judge_tiers, calibrated: "calibrated"}
    partly = held.model_copy(
        update={"judged": judged, "caveats": held.caveats.model_copy(update={"judge_tiers": tiers})}
    )
    assert "judge_uncalibrated" not in {caveat.id for caveat in report_caveats(partly)}
    rows = {row.id: row for row in caveats_list(partly).caveats}
    row = rows["judge_partly_uncalibrated"]
    assert row.touches_measures == [dimension for dimension in held.judged if dimension != calibrated]
    assert row.source == "implied" and row.matters_if
    assert {figure.key for figure in row.figures} <= {
        f"judged.{dimension}.{field}" for dimension in row.touches_measures for field, _ in JUDGE_FIELDS
    }
    assert "judge_partly_uncalibrated" not in {row.id for row in caveats_list(held).caveats}


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


def test_the_fact_tools_are_on_the_analysts_and_the_editors_cuts() -> None:
    names = {"campaign_overview", "readings_list", "contrasts_list", "measure_read", "examples_list", "caveats_list"}
    for role in ("analyst", "editor"):
        (spec,) = analyst_tools(writes=False, role=role)
        assert names <= {action.name for action in eval_catalogue().mount(spec).actions}, role
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


# =============================================================================
# The budget on the shape campaigns, and a page sized to one result
# =============================================================================

#: The fact tools, each held to the budget at its default page.
FACT_TOOLS = ("campaign_overview", "readings_list", "contrasts_list", "measure_read", "examples_list", "caveats_list")


@cache
def _shape_snapshots() -> dict[str, EvalReportSnapshot]:
    return {
        name: report_snapshot(held, scope_id=TOYHOST_SCOPE, campaign_id=None, source="report_input", created_by="me")
        for name, held in _shape_inputs().items()
    }


def _seeded_shapes() -> OpsFixture:
    fixture = ops_fixture()
    for snapshot in _shape_snapshots().values():
        fixture.host.eval_host.storage.save_report_snapshot(snapshot)
    return fixture


def _figure_nodes(node: Any) -> list[dict[str, Any]]:
    """Every cited figure in a result's JSON."""
    if isinstance(node, dict):
        if {"key", "reads", "value", "kind"} <= node.keys():
            return [node]
        return [figure for item in node.values() for figure in _figure_nodes(item)]
    if isinstance(node, list):
        return [figure for item in node for figure in _figure_nodes(item)]
    return []


def _calls(action: str, held: ReportInput) -> list[dict[str, Any]]:
    """How a fact tool is called on an input: its argument sets in ``ANALYST_CALLS``, else with none."""
    build = ANALYST_CALLS.get(action)
    return build(held) if build is not None else [{}]


@pytest.mark.parametrize("name", SHAPE_NAMES)
async def test_each_fact_tool_keeps_its_default_page_within_the_budget_on_the_shape_campaigns(name: str) -> None:
    fixture = off_loop(_seeded_shapes)
    snapshot = _shape_snapshots()[name]
    held = frozen_input(snapshot)
    for action in FACT_TOOLS:
        for arguments in _calls(action, held):
            outcome = await _call(fixture, {**arguments, "action": action, "snapshot_id": snapshot.id})
            where = f"{action} {arguments} on {name}"
            assert outcome.structured is not None, f"{where}: {outcome.text}"
            assert len(outcome.text) <= TEXT_BUDGET, f"{where}: {len(outcome.text)} characters"
            for figure in _figure_nodes(outcome.structured):
                value = resolve(figure["key"], held).value
                assert figure["value"] == (list(value) if isinstance(value, tuple) else value), figure["key"]


def _listed_rows(action: str, structured: dict[str, Any]) -> list[Any]:
    if action == "contrasts_list":
        return list(structured["contrasts"])
    return [
        row
        for group in structured["aspects"]
        for kind in ("readings", "pairs", "floors", "bars")
        for row in group[kind]
    ]


@pytest.mark.parametrize("action", ["readings_list", "contrasts_list"])
async def test_a_page_sized_to_one_result_carries_the_rest_to_the_next_and_every_row_is_read_once(action: str) -> None:
    fixture = off_loop(_seeded_shapes)
    snapshot = _shape_snapshots()["s1"]
    whole = await _call(fixture, {"action": action, "snapshot_id": snapshot.id, "limit": 200})
    assert whole.structured is not None and "next_offset" not in whole.structured and "next" not in whole.structured
    assert len(whole.text) > TEXT_BUDGET, "the control: every s1 row at once runs past one result"
    rows: list[Any] = []
    arguments: dict[str, Any] | None = {"action": action, "snapshot_id": snapshot.id}
    pages = 0
    while arguments is not None:
        page = await _call(fixture, arguments)
        assert page.structured is not None, page.text
        assert len(page.text) <= TEXT_BUDGET and page.structured["total"] == whole.structured["total"]
        rows += _listed_rows(action, page.structured)
        arguments = page.structured.get("next")
        pages += 1
    assert pages > 1
    assert rows == _listed_rows(action, whole.structured)


async def test_a_limit_the_caller_names_is_the_pages_size_as_named() -> None:
    fixture = off_loop(_seeded_shapes)
    snapshot = _shape_snapshots()["s1"]
    held = frozen_input(snapshot)
    named = await _call(fixture, {"action": "contrasts_list", "snapshot_id": snapshot.id, "limit": 50})
    assert named.structured is not None
    assert len(named.structured["contrasts"]) == len(held.contrasts) and "next_offset" not in named.structured
    small = await _call(fixture, {"action": "contrasts_list", "snapshot_id": snapshot.id, "limit": 2})
    assert small.structured is not None and len(small.structured["contrasts"]) == 2
