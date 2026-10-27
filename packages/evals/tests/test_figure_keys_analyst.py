"""The figure-key families the report analyst cites: every figure a tool returns is a key, so the grammar keys them all.

Contracts held here (``threetears.evals.analysis.report.figure_keys``, the analyst's W1 grammar extensions):

- each family reads the input's own field: a contrast on one aspect, a measure's and a judged dimension's figure on
  one aspect, any statistic of a judged figure, an aspect's scoreboard, a pair reading, a floor, a bar, an aspect's
  contrast family and flip margin, an arm's figure in a stratum and in a target-matrix cell, a criterion's figure for
  an arm at each place it reads. Every expectation is read off the input the key resolves against (r1 to r5, the Kairo
  bundle's input, and the shape campaigns where neither holds the data), never typed;
- the derived values (``vs_control``, a floor's ``held_above`` and ``broken_at``, a loss share on any measure, a
  criterion figure's ``gap``) are computed from the input's point figures;
- an unqualified key whose figure the input holds on more than one aspect (a contrast's arm and measure, a stratum id,
  a contrast family) is refused, naming the ``aspects.<aspect>.`` keys that read each one;
- the keys the analyst's fact tools cite where no other key reads the input's field: a floor or bar on an aspect the
  input holds no record for (keyed by the aspect its read names, with no aspect record invented), each of an arm's
  settings as the input holds it (a number or text), one aspect's cases and repeats and each arm's depth on it, each
  late criterion's own date, and any contrast family by its place in the input's families;
- the grammar lists every family, each with an example rendered on its made-up input.
"""

from __future__ import annotations

from typing import Any

import pytest

from packages.evals.tests.fixtures.report_plans.dataset import kairo_bundle, report_input_of
from packages.evals.tests.fixtures.report_plans.misfits import only_bars, reference_data
from packages.evals.tests.report_shapes_support import SHAPES
from threetears.evals.analysis.campaign_bundle import build_campaign_bundle
from threetears.evals.analysis.errors import UnresolvableReference, UnstatedFigure
from threetears.evals.analysis.report.figure_keys import (
    figure_key_grammar,
    format_figure,
    key_subjects,
    resolve,
)
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.analysis.report_decision import report_caveats
from threetears.evals.analysis.report_margins import criterion_figures, flip_margin, weakest_margin

FIXTURES = ("r1", "r2", "r3", "r4", "r5")


@pytest.fixture(scope="module")
def kairo() -> ReportInput:
    """The Kairo round 2 input as the bundle builds it: its contrasts, family and strata name their aspect."""
    return report_input(kairo_bundle())


@pytest.fixture(scope="module")
def shapes() -> dict[str, ReportInput]:
    """The shape campaigns' inputs: s1 holds three aspects, s2 a choice's pairs and bars, s3 a judged by-aspect."""
    built: dict[str, ReportInput] = {}
    for name, make in SHAPES.items():
        shape = make()
        bundle = build_campaign_bundle(
            shape.campaign, list(shape.bundles.values()), arms=shape.arms, readings=shape.readings
        )
        built[name] = report_input(bundle)
    return built


def _value(key: str, report_input: ReportInput) -> Any:
    return resolve(key, report_input).value


def _lead(report_input: ReportInput) -> str:
    asked = [aspect.id for aspect in report_input.aspects if aspect.asked]
    return asked[0] if asked else report_input.aspects[0].id


def _control(report_input: ReportInput) -> str | None:
    return next((arm.id for arm in report_input.arms if arm.role == "control"), None)


def _two_aspects(**parts: Any) -> ReportInput:
    """A control and one candidate on two aspects, each holding the same contrast, stratum id and a contrast family."""
    stratum: dict[str, Any] = {"id": "short", "n_cases": 10}
    figures = {"base": {"m": {"mean": 0.8}}, "a": {"m": {"mean": 0.7}}}
    base: dict[str, Any] = {
        "schema": "report-input/1",
        "meta": {"campaign": "Two", "kind": "change"},
        "arms": [{"id": "base", "role": "control", "settings": {}}, {"id": "a", "role": "challenger", "settings": {}}],
        "aspects": [{"id": "one", "asked": True, "readings": []}, {"id": "two", "readings": []}],
        "scoreboard": {},
        "measures": {"m": {"grade": "pass_fail", "higher_is_better": True, "per_arm": {}}},
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
    return ReportInput.model_validate({**base, **parts})


# --- G1: a contrast on one aspect ----------------------------------------------------------------------------------


class TestAContrastOnOneAspect:
    def test_each_kairo_contrast_reads_by_its_aspect(self, kairo: ReportInput) -> None:
        assert all(contrast.aspect for contrast in kairo.contrasts)
        for contrast in kairo.contrasts:
            prefix = f"aspects.{contrast.aspect}.contrast.{contrast.arm}.{contrast.measure}"
            if contrast.delta is not None:
                assert _value(f"{prefix}.delta", kairo) == contrast.delta
            if contrast.interval is not None:
                assert _value(f"{prefix}.interval", kairo) == tuple(contrast.interval)
            assert _value(f"{prefix}.verdict", kairo) == contrast.verdict
            assert _value(f"{prefix}.level", kairo) == contrast.level
            assert _value(f"{prefix}.materiality", kairo) == contrast.materiality

    @pytest.mark.parametrize("name", ["r4", "r5"])
    def test_the_unqualified_form_reads_level_materiality_and_reason_where_stated(self, name: str) -> None:
        report_input = report_input_of(name)
        for contrast in report_input.contrasts:
            prefix = f"contrast.{contrast.arm}.{contrast.measure}"
            for field in ("level", "materiality", "reason"):
                stated = getattr(contrast, field)
                if stated:
                    assert _value(f"{prefix}.{field}", report_input) == stated
                else:
                    with pytest.raises(UnstatedFigure, match="does not state"):
                        resolve(f"{prefix}.{field}", report_input)

    def test_a_contrast_held_on_two_aspects_is_refused_naming_both(self) -> None:
        report_input = _two_aspects()
        with pytest.raises(UnresolvableReference) as refused:
            resolve("contrast.a.m.delta", report_input)
        message = str(refused.value)
        for contrast in report_input.contrasts:
            assert f"`aspects.{contrast.aspect}.contrast.a.m.delta`" in message
            assert _value(f"aspects.{contrast.aspect}.contrast.a.m.delta", report_input) == contrast.delta

    def test_every_s1_contrast_is_held_on_each_aspect_and_read_only_by_it(self, shapes: dict[str, ReportInput]) -> None:
        report_input = shapes["s1"]
        for contrast in report_input.contrasts:
            held = [c.aspect for c in report_input.contrasts if (c.arm, c.measure) == (contrast.arm, contrast.measure)]
            key = f"contrast.{contrast.arm}.{contrast.measure}.verdict"
            if len(held) > 1:
                with pytest.raises(UnresolvableReference, match=f"holds {len(held)} times"):
                    resolve(key, report_input)
            qualified = f"aspects.{contrast.aspect}.contrast.{contrast.arm}.{contrast.measure}.verdict"
            assert _value(qualified, report_input) == contrast.verdict


# --- G2 and G3: a measure's and a judged dimension's figures -------------------------------------------------------


class TestFiguresOnOneAspect:
    def test_a_measure_reads_by_aspect_and_per_arm_on_the_lead(self, shapes: dict[str, ReportInput]) -> None:
        report_input = shapes["s1"]
        lead = _lead(report_input)
        checked = 0
        for key, measure in report_input.measures.items():
            for aspect in report_input.aspects:
                figures = measure.per_arm if aspect.id == lead else measure.by_aspect.get(aspect.id, {})
                for arm, stat in figures.items():
                    if stat.mean is not None:
                        assert _value(f"aspects.{aspect.id}.measure.{key}.{arm}.mean", report_input) == stat.mean
                        checked += 1
        assert checked and any(measure.by_aspect for measure in report_input.measures.values())

    @pytest.mark.parametrize("name", ["r4", "r5"])
    def test_on_one_aspect_inputs_the_lead_reads_the_by_arm_map(self, name: str) -> None:
        report_input = report_input_of(name)
        lead = _lead(report_input)
        for key, measure in report_input.measures.items():
            for arm, stat in measure.per_arm.items():
                if stat.mean is not None:
                    assert _value(f"aspects.{lead}.measure.{key}.{arm}.mean", report_input) == stat.mean

    def test_an_aspect_with_no_figures_on_a_measure_is_refused(self) -> None:
        report_input = report_input_of("r4")
        other = next(aspect.id for aspect in report_input.aspects if aspect.id != _lead(report_input))
        key = next(iter(report_input.measures))
        arm = next(iter(report_input.measures[key].per_arm))
        with pytest.raises(UnresolvableReference, match="names no figure"):
            resolve(f"aspects.{other}.measure.{key}.{arm}.mean", report_input)

    @pytest.mark.parametrize("name", FIXTURES)
    def test_any_statistic_of_a_judged_figure_reads_its_stat(self, name: str) -> None:
        report_input = report_input_of(name)
        for dimension, judged in report_input.judged.items():
            for arm, stat in judged.per_arm.items():
                for statistic in ("mean", "n", "n_cases", "sem", "half_width"):
                    stated = getattr(stat, statistic)
                    if stated is not None:
                        assert _value(f"judged.{dimension}.{arm}.{statistic}", report_input) == stated
                if stat.mean is not None:
                    shown = format_figure(resolve(f"judged.{dimension}.{arm}.mean", report_input))
                    assert shown == format_figure(resolve(f"judged.{dimension}.{arm}", report_input))

    def test_a_judged_dimension_reads_by_aspect(self, shapes: dict[str, ReportInput]) -> None:
        report_input = shapes["s3"]
        lead = _lead(report_input)
        assert any(judged.by_aspect for judged in report_input.judged.values())
        for dimension, judged in report_input.judged.items():
            for aspect in report_input.aspects:
                figures = judged.per_arm if aspect.id == lead else judged.by_aspect.get(aspect.id, {})
                for arm, stat in figures.items():
                    assert _value(f"aspects.{aspect.id}.judged.{dimension}.{arm}.mean", report_input) == stat.mean
                    assert _value(f"aspects.{aspect.id}.judged.{dimension}.{arm}.n", report_input) == stat.n


# --- G4: an aspect's scoreboard ------------------------------------------------------------------------------------


class TestAnAspectsScoreboard:
    def test_each_aspect_reads_its_own_and_the_lead_the_inputs(self, shapes: dict[str, ReportInput]) -> None:
        report_input = shapes["s1"]
        lead = _lead(report_input)
        assert any(aspect.scoreboard for aspect in report_input.aspects)
        for aspect in report_input.aspects:
            board = aspect.scoreboard if aspect.scoreboard is not None else report_input.scoreboard
            assert aspect.scoreboard is not None or aspect.id == lead
            for arm, row in board.items():
                for statistic in ("ran", "usable", "passed", "cases"):
                    if getattr(row, statistic) is not None:
                        key = f"aspects.{aspect.id}.scoreboard.{arm}.{statistic}"
                        assert _value(key, report_input) == getattr(row, statistic)

    def test_r5s_asked_aspect_reads_the_inputs_scoreboard(self) -> None:
        report_input = report_input_of("r5")
        lead = _lead(report_input)
        for arm, row in report_input.scoreboard.items():
            assert row.ran is not None
            assert _value(f"aspects.{lead}.scoreboard.{arm}.ran", report_input) == row.ran


# --- G5, G6, G7: pairs, floors, bars -------------------------------------------------------------------------------


class TestPairsFloorsAndBars:
    def test_each_pair_reads_its_change_interval_and_arm_ahead(self, shapes: dict[str, ReportInput]) -> None:
        report_input = shapes["s2"]
        pairs = [(aspect.id, pair) for aspect in report_input.aspects for pair in aspect.pairs]
        assert pairs and any(pair.ahead for _, pair in pairs) and any(pair.ahead is None for _, pair in pairs)
        for aspect, pair in pairs:
            prefix = f"aspects.{aspect}.pair.{pair.arms[0]}.{pair.arms[1]}"
            assert _value(f"{prefix}.change", report_input) == pair.change
            assert pair.interval is not None
            assert _value(f"{prefix}.interval", report_input) == tuple(pair.interval)
            assert _value(f"{prefix}.low", report_input) == pair.interval[0]
            assert _value(f"{prefix}.confidence", report_input) == pair.confidence
            assert _value(f"{prefix}.why", report_input) == pair.why
            if pair.ahead is None:
                with pytest.raises(UnstatedFigure, match="does not state"):
                    resolve(f"{prefix}.ahead", report_input)
            else:
                ahead = resolve(f"{prefix}.ahead", report_input)
                assert (ahead.kind, ahead.value) == ("arm", pair.ahead)

    def test_a_pair_names_both_arms_it_reads(self, shapes: dict[str, ReportInput]) -> None:
        report_input = shapes["s2"]
        aspect = next(aspect for aspect in report_input.aspects if aspect.pairs)
        first, second = aspect.pairs[0].arms
        subjects = key_subjects(f"aspects.{aspect.id}.pair.{first}.{second}.change", report_input)
        assert (subjects.arms, subjects.aspects, subjects.against_control) == ((first, second), (aspect.id,), False)

    @pytest.mark.parametrize("source", ["r5", "kairo"])
    def test_each_floor_reads_its_fields_and_the_margins_it_holds_and_breaks_at(
        self, source: str, kairo: ReportInput
    ) -> None:
        report_input = kairo if source == "kairo" else report_input_of(source)
        assert report_input.floors
        for read in report_input.floors:
            prefix = f"aspects.{read.aspect}.floor.{read.floor}.{read.arm}"
            assert _value(f"{prefix}.status", report_input) == read.status
            assert _value(f"{prefix}.worst_drop", report_input) == read.worst_drop
            assert _value(f"{prefix}.margin", report_input) == read.margin
            assert read.interval is not None
            assert _value(f"{prefix}.interval", report_input) == tuple(read.interval)
            assert _value(f"{prefix}.held_above", report_input) == -read.interval[0]
            assert _value(f"{prefix}.broken_at", report_input) == -read.interval[1]

    @pytest.mark.parametrize("name", ["s2", "s3"])
    def test_each_bar_reads_its_value_line_status_and_interval(self, name: str, shapes: dict[str, ReportInput]) -> None:
        report_input = shapes[name]
        assert report_input.bars
        for read in report_input.bars:
            prefix = f"aspects.{read.aspect}.bar.{read.measure}.{read.arm}"
            assert _value(f"{prefix}.bar", report_input) == read.bar
            assert _value(f"{prefix}.status", report_input) == read.status
            if read.value is not None:
                assert _value(f"{prefix}.value", report_input) == read.value
            if read.interval is not None:
                assert _value(f"{prefix}.interval", report_input) == tuple(read.interval)


# --- G8, G9: an aspect's contrast family and flip margins ----------------------------------------------------------


class TestAnAspectsFamilyAndFlipMargins:
    def test_kairos_one_family_reads_unqualified_and_by_its_aspect(self, kairo: ReportInput) -> None:
        (family,) = kairo.contrast_families
        assert family.aspect is not None
        for field in ("correction", "disclosure"):
            assert _value(f"aspects.{family.aspect}.family.{field}", kairo) == getattr(family, field)
            assert _value(f"family.{field}", kairo) == getattr(family, field)

    def test_several_families_refuse_the_unqualified_form_naming_each(self, shapes: dict[str, ReportInput]) -> None:
        report_input = shapes["s1"]
        assert len(report_input.contrast_families) > 1
        with pytest.raises(UnresolvableReference) as refused:
            resolve("family.correction", report_input)
        for family in report_input.contrast_families:
            assert f"`aspects.{family.aspect}.family.correction`" in str(refused.value)
            assert _value(f"aspects.{family.aspect}.family.correction", report_input) == family.correction

    @pytest.mark.parametrize("source", ["r5", "s1"])
    def test_each_readings_flip_margin_reads_on_its_aspect(self, source: str, shapes: dict[str, ReportInput]) -> None:
        report_input = shapes.get(source) or report_input_of(source)
        checked = 0
        for aspect in report_input.aspects:
            for reading in aspect.readings:
                flip = flip_margin(reading)
                if flip is not None:
                    assert _value(f"aspects.{aspect.id}.flip_margin.{reading.arm}", report_input) == flip
                    checked += 1
        assert checked

    def test_the_asked_aspects_flip_margin_is_the_unqualified_one(self) -> None:
        report_input = report_input_of("r5")
        lead = _lead(report_input)
        for reading in report_input.aspects[0].readings:
            key = f"aspects.{lead}.flip_margin.{reading.arm}"
            assert _value(key, report_input) == _value(f"flip_margin.{reading.arm}", report_input)


# --- G10, G11, G12: strata -----------------------------------------------------------------------------------------


class TestStrata:
    @pytest.mark.parametrize("name", ["r4", "r5"])
    def test_an_arms_figure_in_a_stratum_reads_the_strata_stat(self, name: str) -> None:
        report_input = report_input_of(name)
        for stratum in report_input.strata:
            for arm, figures in stratum.per_arm.items():
                for key, stat in figures.items():
                    path = f"strata.{stratum.id}.{arm}.{key}.mean"
                    if stat.mean is None:
                        with pytest.raises(UnstatedFigure, match="does not state"):
                            resolve(path, report_input)
                    else:
                        assert _value(path, report_input) == stat.mean

    def test_kairos_strata_read_on_their_aspect(self, kairo: ReportInput) -> None:
        for stratum in kairo.strata:
            assert stratum.aspect is not None
            assert _value(f"aspects.{stratum.aspect}.strata.{stratum.id}.n_cases", kairo) == stratum.n_cases
            for arm, figures in stratum.per_arm.items():
                for key, stat in figures.items():
                    qualified = f"aspects.{stratum.aspect}.strata.{stratum.id}.{arm}.{key}.mean"
                    assert _value(qualified, kairo) == stat.mean

    @pytest.mark.parametrize("name", ["r4", "r5"])
    def test_vs_control_is_the_arms_mean_minus_the_controls_in_the_stratum(self, name: str) -> None:
        report_input = report_input_of(name)
        control = _control(report_input)
        assert control is not None
        checked = 0
        for stratum in report_input.strata:
            for arm, figures in stratum.per_arm.items():
                for key, stat in figures.items():
                    theirs = stratum.per_arm.get(control, {}).get(key)
                    if arm == control or theirs is None or theirs.mean is None or stat.mean is None:
                        continue
                    figure = resolve(f"strata.{stratum.id}.{arm}.{key}.vs_control", report_input)
                    assert figure.value == pytest.approx(stat.mean - theirs.mean)
                    assert figure.change
                    checked += 1
        assert checked

    def test_vs_control_names_the_control_as_a_subject(self) -> None:
        report_input = report_input_of("r5")
        stratum = report_input.strata[0]
        control = _control(report_input)
        arm = next(arm for arm in stratum.per_arm if arm != control)
        key = next(iter(stratum.per_arm[arm]))
        assert key_subjects(f"strata.{stratum.id}.{arm}.{key}.vs_control", report_input).against_control
        assert not key_subjects(f"strata.{stratum.id}.{arm}.{key}.mean", report_input).against_control

    def test_vs_control_is_refused_for_the_control_and_without_one(self) -> None:
        report_input = report_input_of("r5")
        control = _control(report_input)
        stratum = report_input.strata[0]
        key = next(iter(stratum.per_arm[str(control)]))
        with pytest.raises(UnresolvableReference, match="names the control itself"):
            resolve(f"strata.{stratum.id}.{control}.{key}.vs_control", report_input)
        r1 = report_input_of("r1")
        task = r1.tasks[0]
        arm, figures = next(iter(task.per_arm.items()))
        with pytest.raises(UnresolvableReference, match="this campaign has none"):
            resolve(f"task.{task.id}.{arm}.{next(iter(figures))}.vs_control", r1)

    def test_a_loss_share_on_the_primary_dimension_named_is_the_loss_share(self) -> None:
        report_input = report_input_of("r5")
        primary = report_input.primary_judged
        assert primary is not None
        for stratum in report_input.strata:
            for candidate in (arm.id for arm in report_input.arms if arm.role == "challenger"):
                named = _value(f"strata.{stratum.id}.loss_share.{candidate}.{primary}", report_input)
                assert named == pytest.approx(_value(f"strata.{stratum.id}.loss_share.{candidate}", report_input))

    def test_a_loss_share_on_any_dimension_is_its_weighted_part_of_the_loss(self) -> None:
        """Every candidate and dimension other than the primary that lost overall: each stratum's part of it."""
        report_input = report_input_of("r5")
        control = _control(report_input)
        assert control is not None
        checked = 0
        for candidate in (arm.id for arm in report_input.arms if arm.role == "challenger"):
            for dimension in (key for key in report_input.judged if key != report_input.primary_judged):
                losses = {}
                for stratum in report_input.strata:
                    theirs, mine = stratum.per_arm[control][dimension].mean, stratum.per_arm[candidate][dimension].mean
                    assert theirs is not None and mine is not None and stratum.n_cases is not None
                    losses[stratum.id] = stratum.n_cases * (theirs - mine)
                total = sum(losses.values())
                for stratum_id, loss in losses.items():
                    key = f"strata.{stratum_id}.loss_share.{candidate}.{dimension}"
                    if total <= 0:
                        with pytest.raises(UnresolvableReference, match="lost nothing overall"):
                            resolve(key, report_input)
                        continue
                    assert _value(key, report_input) == pytest.approx(loss / total)
                    checked += 1
        assert checked

    def test_a_loss_where_lower_is_better_runs_the_other_way(self) -> None:
        figures = {"base": {"ms": {"mean": 100}}, "a": {"ms": {"mean": 130}}}
        other = {"base": {"ms": {"mean": 100}}, "a": {"ms": {"mean": 110}}}
        report_input = _two_aspects(
            measures={"ms": {"unit": "ms", "higher_is_better": False, "per_arm": {}}},
            contrasts=[],
            strata=[{"id": "x", "n_cases": 10, "per_arm": figures}, {"id": "y", "n_cases": 10, "per_arm": other}],
        )
        losses = [10 * (130 - 100), 10 * (110 - 100)]
        assert _value("strata.x.loss_share.a.ms", report_input) == pytest.approx(losses[0] / sum(losses))

    def test_a_loss_on_a_measure_of_no_stated_direction_is_refused(self) -> None:
        figures = {"base": {"n": {"mean": 3}}, "a": {"n": {"mean": 4}}}
        report_input = _two_aspects(
            measures={"n": {"per_arm": {}}}, contrasts=[], strata=[{"id": "x", "n_cases": 1, "per_arm": figures}]
        )
        with pytest.raises(UnresolvableReference, match="better direction the input does not state"):
            resolve("strata.x.loss_share.a.n", report_input)

    def test_a_stratum_id_held_on_two_aspects_is_refused_naming_each(self) -> None:
        report_input = _two_aspects()
        for tail in ("name", "a.m.mean", "loss_share.a.m"):
            with pytest.raises(UnresolvableReference) as refused:
                resolve(f"strata.short.{tail}", report_input)
            message = str(refused.value)
            assert "reads 2 ways" not in message
            for stratum in report_input.strata:
                assert f"`aspects.{stratum.aspect}.strata.short.{tail}`" in message
        for stratum in report_input.strata:
            assert _value(f"aspects.{stratum.aspect}.strata.short.name", report_input) == stratum.name
            mean = stratum.per_arm["a"]["m"].mean
            assert _value(f"aspects.{stratum.aspect}.strata.short.a.m.mean", report_input) == mean

    def test_a_qualified_loss_share_reads_the_aspects_own_strata(self) -> None:
        report_input = _two_aspects()
        for stratum in report_input.strata:
            assert _value(f"aspects.{stratum.aspect}.strata.short.loss_share.a.m", report_input) == pytest.approx(1.0)


# --- G13: target-matrix cells --------------------------------------------------------------------------------------


class TestCells:
    def test_each_r3_cell_reads_its_name_cases_and_figures(self) -> None:
        report_input = report_input_of("r3")
        matrix = report_input.target_matrix
        assert matrix is not None and report_input.cells
        rows = {level.id: level.name or level.id for level in matrix.rows.levels}
        columns = {level.id: level.name or level.id for level in matrix.columns.levels}
        for cell in report_input.cells.values():
            spelled = f"{cell.row}.{cell.column}"
            assert _value(f"cells.{spelled}.name", report_input) == f"{rows[cell.row]} × {columns[cell.column]}"
            if cell.n_cases is not None:
                assert _value(f"cells.{spelled}.n_cases", report_input) == cell.n_cases
            for arm, figures in cell.per_arm.items():
                for key, stat in figures.items():
                    assert _value(f"cells.{spelled}.{arm}.{key}.mean", report_input) == stat.mean

    def test_a_cell_with_a_control_reads_vs_control(self) -> None:
        figures = {"base": {"m": {"mean": 0.9}}, "a": {"m": {"mean": 0.75}}}
        report_input = _two_aspects(
            target_matrix={
                "rows": {"id": "r", "levels": [{"id": "lo"}]},
                "columns": {"id": "c", "levels": [{"id": "x"}]},
            },
            cells={"lo|x": {"row": "lo", "column": "x", "per_arm": figures}},
        )
        expected = figures["a"]["m"]["mean"] - figures["base"]["m"]["mean"]
        assert _value("cells.lo.x.a.m.vs_control", report_input) == pytest.approx(expected)


# --- G14: a criterion's figures ------------------------------------------------------------------------------------


class TestCriterionFigures:
    @pytest.mark.parametrize("name", FIXTURES)
    def test_each_criterion_reads_each_arms_figure_at_each_place(self, name: str) -> None:
        report_input = report_input_of(name)
        for criterion in report_input.criteria:
            threshold = resolve(f"criteria.{criterion.id}.threshold", report_input)
            factor = 100.0 if threshold.scale == "share" else 1.0
            for arm in report_input.arms:
                figures = criterion_figures(report_input, criterion, arm.id)
                for figure in figures:
                    place = figure.task if figure.cell is None else _spelled(report_input, figure.cell)
                    key = f"criteria.{criterion.id}.{arm.id}" + (f".{place}" if place is not None else "")
                    single = f"criteria.{criterion.id}.{arm.id}"
                    gap = figure.value - float(criterion.threshold)
                    if criterion.direction == "lower_is_better":
                        gap = -gap
                    assert _value(f"{key}.value", report_input) == figure.value
                    assert _value(f"{key}.gap", report_input) == pytest.approx(gap * factor)
                    if len(figures) == 1:
                        assert _value(f"{single}.value", report_input) == figure.value
                if len(figures) > 1:
                    with pytest.raises(UnresolvableReference, match=f"the criterion reads {len(figures)}"):
                        resolve(f"criteria.{criterion.id}.{arm.id}.value", report_input)

    @pytest.mark.parametrize("name", ["r1", "r2", "r3", "r4"])
    def test_the_weakest_criterion_margins_gap_is_its_figures_gap(self, name: str) -> None:
        report_input = report_input_of(name)
        found = weakest_margin(report_input)
        assert found is not None and found.kind == "criterion"
        place = found.task if found.cell is None else _spelled(report_input, found.cell)
        key = f"criteria.{found.source}.{found.arm}" + (f".{place}" if place is not None else "")
        assert format_figure(resolve(f"{key}.gap", report_input)) == format_figure(
            resolve("weakest.margin", report_input)
        )


def _spelled(report_input: ReportInput, cell_id: str) -> str:
    cell = report_input.cells[cell_id]
    return f"{cell.row}.{cell.column}"


# --- the keys the fact tools cite where no other key reads the field ------------------------------------------------


class TestAFloorOrBarOnAnAspectWithNoRecord:
    def test_each_bar_is_keyed_by_the_aspect_its_read_names_and_no_aspect_is_invented(self) -> None:
        report_input = ReportInput.model_validate(only_bars())
        assert report_input.aspects == [] and report_input.bars, "the control: bars on an aspect the input lacks"
        for read in report_input.bars:
            prefix = f"aspects.{read.aspect}.bar.{read.measure}.{read.arm}"
            assert _value(f"{prefix}.bar", report_input) == read.bar
            assert _value(f"{prefix}.status", report_input) == read.status
            with pytest.raises(UnstatedFigure, match="does not state"):
                resolve(f"{prefix}.value", report_input)
            subjects = key_subjects(f"{prefix}.status", report_input)
            assert (subjects.arms, subjects.aspects) == ((read.arm,), (read.aspect,))
        aspect = report_input.bars[0].aspect
        with pytest.raises(UnresolvableReference, match="names no figure"):
            resolve(f"aspects.{aspect}.name", report_input)

    def test_each_floor_on_an_aspect_with_no_record_reads_its_own_fields(self) -> None:
        data = reference_data("r5")
        for floor in data["floors"]:
            floor["aspect"] = "unrecorded"
        report_input = ReportInput.model_validate(data)
        assert "unrecorded" not in {aspect.id for aspect in report_input.aspects}
        for read in report_input.floors:
            prefix = f"aspects.unrecorded.floor.{read.floor}.{read.arm}"
            assert _value(f"{prefix}.status", report_input) == read.status
            assert read.interval is not None
            assert _value(f"{prefix}.interval", report_input) == tuple(read.interval)
            assert _value(f"{prefix}.held_above", report_input) == -read.interval[0]


class TestAnArmsSettings:
    @pytest.mark.parametrize("name", FIXTURES)
    def test_each_setting_reads_as_the_input_holds_it_a_number_or_its_text(self, name: str) -> None:
        report_input = report_input_of(name)
        kinds: set[str] = set()
        for arm in report_input.arms:
            for setting, held in arm.settings.items():
                key = f"settings.{arm.id}.{setting}"
                if held is None:
                    with pytest.raises(UnstatedFigure, match="does not state"):
                        resolve(key, report_input)
                    continue
                figure = resolve(key, report_input)
                kinds.add(figure.kind)
                if isinstance(held, str):
                    assert (figure.kind, figure.value) == ("text", held)
                else:
                    assert (figure.kind, figure.value) == ("number", float(held))
                assert key_subjects(key, report_input).arms == (arm.id,)
        if name == "r2":
            assert kinds == {"number", "text"}, "the control: r2's retry is a number, its reasoning text"


def _depth(report_input: ReportInput, aspect: str, arm: str) -> tuple[int, int, float]:
    held = next(a for a in report_input.aspects if a.id == aspect).depth
    assert held is not None
    return held[arm].cases, held[arm].trials, float(held[arm].repeats_per_case[1])


class TestAnAspectsDepth:
    @pytest.mark.parametrize("source", ["s1", "s3", "r5"])
    def test_each_arms_depth_and_the_aspects_one_count_read_the_inputs_depth(
        self, source: str, shapes: dict[str, ReportInput]
    ) -> None:
        report_input = shapes.get(source) or report_input_of(source)
        control = _control(report_input)
        checked = 0
        for aspect in report_input.aspects:
            assert aspect.depth
            for arm in aspect.depth:
                cases, trials, repeats = _depth(report_input, aspect.id, arm)
                prefix = f"aspects.{aspect.id}.depth.{arm}"
                assert _value(f"{prefix}.cases", report_input) == cases
                assert _value(f"{prefix}.trials", report_input) == trials
                assert _value(f"{prefix}.repeats", report_input) == repeats
                assert key_subjects(f"{prefix}.cases", report_input).aspects == (aspect.id,)
                checked += 1
            by = control if control in aspect.depth else next(iter(aspect.depth))
            cases, _trials, repeats = _depth(report_input, aspect.id, str(by))
            assert _value(f"aspects.{aspect.id}.cases", report_input) == cases
            assert _value(f"aspects.{aspect.id}.repeats", report_input) == repeats
        assert checked

    def test_s1s_tone_reads_the_controls_repeats_where_a_candidate_ran_twice(
        self, shapes: dict[str, ReportInput]
    ) -> None:
        report_input = shapes["s1"]
        tone = next(aspect for aspect in report_input.aspects if aspect.id == "tone")
        assert tone.depth is not None and len({d.repeats_per_case[1] for d in tone.depth.values()}) > 1
        control = _control(report_input)
        assert _value("aspects.tone.repeats", report_input) == tone.depth[str(control)].repeats_per_case[1]
        with pytest.raises(UnresolvableReference, match="reads the one asked aspect"):
            resolve("repeats", report_input)

    def test_arms_that_disagree_with_no_control_are_refused_naming_each_arms_key(
        self, shapes: dict[str, ReportInput]
    ) -> None:
        data = shapes["s2"].model_dump(mode="json", by_alias=True)
        (aspect,) = data["aspects"]
        arm = next(iter(aspect["depth"]))
        aspect["depth"][arm]["repeats_per_case"] = [1, 1.5, 2]
        report_input = ReportInput.model_validate(data)
        assert _control(report_input) is None
        with pytest.raises(UnstatedFigure, match="does not state") as refused:
            resolve(f"aspects.{aspect['id']}.repeats", report_input)
        assert f"`aspects.{aspect['id']}.depth.<arm>.repeats`" in str(refused.value)
        assert _value(f"aspects.{aspect['id']}.depth.{arm}.repeats", report_input) == 1.5
        assert _value(f"aspects.{aspect['id']}.cases", report_input) == aspect["depth"][arm]["cases"]

    def test_the_unqualified_repeats_names_the_asked_aspect_as_its_subject(self) -> None:
        report_input = report_input_of("r5")
        asked = [aspect.id for aspect in report_input.aspects if aspect.asked]
        assert key_subjects("repeats", report_input).aspects == tuple(asked)
        assert key_subjects("cases", report_input).aspects == tuple(asked)


class TestEachLateCriterionsDate:
    def test_each_late_criterion_reads_its_own_date_by_the_name_its_caveat_carries(self) -> None:
        report_input = report_input_of("r5")
        late_ids = {caveat.id for caveat in report_caveats(report_input) if caveat.kind == "late"}
        assert report_input.caveats.late_criteria and any(late.name for late in report_input.caveats.late_criteria)
        for late in report_input.caveats.late_criteria:
            named = f"{late.criterion}:{late.name}" if late.name else late.criterion
            assert f"late:{named}" in late_ids
            assert _value(f"late_criteria.{named}.set_at", report_input) == late.set_at

    def test_dates_that_differ_refuse_the_one_date_naming_each_criterions_key(self) -> None:
        data = reference_data("r5")
        data["caveats"]["late_criteria"][-1]["set_at"] = "26 Oct 2026"
        report_input = ReportInput.model_validate(data)
        with pytest.raises(UnresolvableReference, match="2 different times") as refused:
            resolve("late_criteria.set_at", report_input)
        for late in report_input.caveats.late_criteria:
            named = f"{late.criterion}:{late.name}" if late.name else late.criterion
            assert f"`late_criteria.{named}.set_at`" in str(refused.value)
            assert _value(f"late_criteria.{named}.set_at", report_input) == late.set_at


class TestAFamilyByItsPlace:
    def test_r4s_one_family_naming_no_aspect_reads_by_its_place_and_unqualified(self) -> None:
        report_input = report_input_of("r4")
        (family,) = report_input.contrast_families
        assert family.aspect is None
        for field in ("correction", "alpha", "family_size", "disclosure"):
            stated = getattr(family, field)
            if stated is not None:
                assert _value(f"families.0.{field}", report_input) == stated == _value(f"family.{field}", report_input)

    def test_a_family_naming_no_aspect_beside_another_is_named_by_its_place(self) -> None:
        report_input = _two_aspects(
            contrast_families=[{"aspect": "one", "correction": "holm"}, {"correction": "none", "family_size": 3}]
        )
        with pytest.raises(UnresolvableReference) as refused:
            resolve("family.correction", report_input)
        assert "`aspects.one.family.correction`" in str(refused.value)
        assert "`families.1.correction`" in str(refused.value)
        assert _value("families.1.correction", report_input) == "none"
        assert _value("families.1.family_size", report_input) == 3
        assert _value("families.0.correction", report_input) == _value("aspects.one.family.correction", report_input)

    def test_two_families_on_one_aspect_are_refused_by_it_naming_each_place(self) -> None:
        report_input = _two_aspects(
            contrast_families=[{"aspect": "one", "correction": "holm"}, {"aspect": "one", "correction": "none"}]
        )
        with pytest.raises(UnresolvableReference, match="holds 2 on it") as refused:
            resolve("aspects.one.family.correction", report_input)
        assert "`families.0.correction`" in str(refused.value) and "`families.1.correction`" in str(refused.value)

    def test_each_s1_family_reads_by_its_place_as_by_its_aspect(self, shapes: dict[str, ReportInput]) -> None:
        report_input = shapes["s1"]
        for index, family in enumerate(report_input.contrast_families):
            assert _value(f"families.{index}.correction", report_input) == family.correction
            assert _value(f"aspects.{family.aspect}.family.correction", report_input) == family.correction


# --- a figure says whether code computed it ----------------------------------------------------------------------


class TestAFigureSaysWhetherItIsDerived:
    def test_r5s_derived_figures_say_so_and_its_copied_ones_do_not(self) -> None:
        report_input = report_input_of("r5")
        control = _control(report_input)
        aspect = report_input.aspects[0]
        reading = aspect.readings[0]
        floor = report_input.floors[0]
        stratum = report_input.strata[0]
        arm = next(arm for arm in stratum.per_arm if arm != control)
        key = next(iter(stratum.per_arm[arm]))
        contrast = report_input.contrasts[0]
        derived = [
            f"cost.uplift.{reading.arm}",
            "cost.uplift_range",
            f"strata.{stratum.id}.loss_share",
            f"strata.{stratum.id}.loss_share.{reading.arm}.{report_input.primary_judged}",
            f"flip_margin.{reading.arm}",
            f"aspects.{aspect.id}.flip_margin.{reading.arm}",
            f"strata.{stratum.id}.{arm}.{key}.vs_control",
            f"aspects.{floor.aspect}.floor.{floor.floor}.{floor.arm}.held_above",
            f"aspects.{floor.aspect}.floor.{floor.floor}.{floor.arm}.broken_at",
            "weakest.margin",
        ]
        copied = [
            f"reading.{reading.arm}.change",
            f"contrast.{contrast.arm}.{contrast.measure}.verdict",
            f"strata.{stratum.id}.{arm}.{key}.mean",
            f"aspects.{floor.aspect}.floor.{floor.floor}.{floor.arm}.worst_drop",
            f"judged.{key}.{arm}" if key in report_input.judged else f"pass_rate.{control}",
            "meta.campaign",
            f"arm:{reading.arm}",
        ]
        assert [path for path in derived if not resolve(path, report_input).derived] == []
        assert [path for path in copied if resolve(path, report_input).derived] == []

    @pytest.mark.parametrize("name", ["r1", "r2", "r3"])
    def test_a_criterion_figures_gap_is_derived_and_its_value_is_not(self, name: str) -> None:
        report_input = report_input_of(name)
        criterion = report_input.criteria[0]
        arm = next(arm.id for arm in report_input.arms if criterion_figures(report_input, criterion, arm.id))
        figure = criterion_figures(report_input, criterion, arm)[0]
        place = figure.task if figure.cell is None else _spelled(report_input, figure.cell)
        key = f"criteria.{criterion.id}.{arm}" + (f".{place}" if place is not None else "")
        assert resolve(f"{key}.gap", report_input).derived
        assert not resolve(f"{key}.value", report_input).derived
        assert not resolve(f"criteria.{criterion.id}.threshold", report_input).derived


# --- the grammar ---------------------------------------------------------------------------------------------------


class TestTheGrammarTeachesTheAnalystsFamilies:
    def test_it_lists_every_family(self) -> None:
        grammar = figure_key_grammar()
        for form in (
            "aspects.<aspect>.contrast.<arm>.<measure>.<statistic>",
            "aspects.<aspect>.measure.<measure>.<arm>.<statistic>",
            "judged.<dimension>.<arm>.<statistic>",
            "aspects.<aspect>.judged.<dimension>.<arm>.<statistic>",
            "aspects.<aspect>.scoreboard.<arm>.<statistic>",
            "aspects.<aspect>.pair.<arm>.<arm>.<statistic>",
            "aspects.<aspect>.floor.<floor>.<arm>.<field>",
            "aspects.<aspect>.bar.<measure>.<arm>.<field>",
            "aspects.<aspect>.family.<field>",
            "aspects.<aspect>.flip_margin.<candidate>",
            "strata.<stratum>.<arm>.<measure>.<statistic>",
            "aspects.<aspect>.strata.<stratum>.<field>",
            "aspects.<aspect>.strata.<stratum>.<arm>.<measure>.<statistic>",
            "strata.<stratum>.loss_share.<candidate>.<measure>",
            "aspects.<aspect>.strata.<stratum>.loss_share",
            "aspects.<aspect>.strata.<stratum>.loss_share.<candidate>",
            "aspects.<aspect>.strata.<stratum>.loss_share.<candidate>.<measure>",
            "cells.<cell>.<field>",
            "cells.<cell>.<arm>.<measure>.<statistic>",
            "criteria.<criterion>.<arm>.<field>",
            "criteria.<criterion>.<arm>.<place>.<field>",
            "settings.<arm>.<setting>",
            "aspects.<aspect>.cases",
            "aspects.<aspect>.repeats",
            "aspects.<aspect>.depth.<arm>.<field>",
            "late_criteria.<late>.set_at",
            "families.<family>.<field>",
            "calls.unanswered.<arm>",
            "calls.unanswered.<arm>.<field>",
            "aspects.<aspect>.calls.unanswered.<arm>",
            "aspects.<aspect>.calls.unanswered.<arm>.<field>",
        ):
            assert f"- `{form}`:" in grammar, form

    def test_it_marks_what_is_derived(self) -> None:
        lines = {line.split("`:", 1)[0]: line for line in figure_key_grammar().splitlines() if line.startswith("- `")}
        for form, derived in (
            ("- `task.<task>.<arm>.<measure>.<statistic>", "`vs_control` derived"),
            ("- `strata.<stratum>.<arm>.<measure>.<statistic>", "`vs_control` derived"),
            ("- `cells.<cell>.<arm>.<measure>.<statistic>", "`vs_control` derived"),
            ("- `aspects.<aspect>.floor.<floor>.<arm>.<field>", "`held_above`, `broken_at` derived"),
            ("- `criteria.<criterion>.<arm>.<field>", "`gap` derived"),
            ("- `aspects.<aspect>.flip_margin.<candidate>", "Derived: descriptive, no interval."),
            ("- `strata.<stratum>.loss_share.<candidate>.<measure>", "Derived: descriptive, no interval."),
        ):
            assert derived in lines[form], form

    def test_it_teaches_the_aspect_qualification_and_the_cell_spelling(self) -> None:
        grammar = figure_key_grammar()
        assert "names the `aspects.<aspect>.` keys to write instead" in grammar
        assert "(`<row>.<column>`)" in grammar

    def test_each_family_it_adds_shows_an_example_with_its_reading(self) -> None:
        """The grammar renders each example on its made-up input, and raises where one no longer resolves; so a line
        carrying ``reads "…"`` shows code's own reading of a key that resolves."""
        lines = {line.split("`:", 1)[0]: line for line in figure_key_grammar().splitlines() if line.startswith("- `")}
        for form in (
            "aspects.<aspect>.contrast.<arm>.<measure>.<statistic>",
            "aspects.<aspect>.measure.<measure>.<arm>.<statistic>",
            "judged.<dimension>.<arm>.<statistic>",
            "aspects.<aspect>.scoreboard.<arm>.<statistic>",
            "aspects.<aspect>.pair.<arm>.<arm>.<statistic>",
            "aspects.<aspect>.floor.<floor>.<arm>.<field>",
            "aspects.<aspect>.bar.<measure>.<arm>.<field>",
            "aspects.<aspect>.family.<field>",
            "aspects.<aspect>.flip_margin.<candidate>",
            "strata.<stratum>.<arm>.<measure>.<statistic>",
            "strata.<stratum>.loss_share.<candidate>.<measure>",
            "cells.<cell>.<field>",
            "cells.<cell>.<arm>.<measure>.<statistic>",
            "criteria.<criterion>.<arm>.<field>",
            "criteria.<criterion>.<arm>.<place>.<field>",
            "settings.<arm>.<setting>",
            "aspects.<aspect>.cases",
            "aspects.<aspect>.repeats",
            "aspects.<aspect>.depth.<arm>.<field>",
            "late_criteria.<late>.set_at",
            "families.<family>.<field>",
            "calls.unanswered.<arm>",
            "calls.unanswered.<arm>.<field>",
        ):
            line = lines[f"- `{form}"]
            assert '` reads "' in line, form
            key = line.rsplit(" `{", 1)[1].split("}` reads", 1)[0]
            assert key.split("|")[0].count(".") >= form.count("."), form
