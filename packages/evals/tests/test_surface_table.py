"""The decision surface laid out once — every rule the two renders used to derive for themselves.

`build_surface_table()` turns the frozen per-cell facts (`EvalAnalysis.decision_surface`) into the table
both report surfaces render: the row order, which measures stand in as cost and latency columns,
the one unit each column is stated in, the verdict under each bar, the replication sentence, the
run notes, the bars no cell could be read against, and the two states that are not a table at all.
Each class below pins one of those rules against the deriver itself, so no surface's rendering
can be what makes it pass.

Cost and latency are named as the engine names them — ``production_replicating_cost`` (the only
measure carrying the cost axis) and ``total_ms`` — because a fixture that invents a
``cost_usd``/``latency_ms`` pair is testing a campaign the engine never columns.

The builders here are module-level so another render suite can stand on the same fixture.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.evals.analysis.arms import arm_table
from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.surface_table import (
    NO_CELLS,
    SURFACE_PROVENANCE,
    SurfaceColumn,
    SurfaceRow,
    SurfaceTable,
    SurfaceValue,
    build_surface_table,
)
from threetears.evals.contracts.analysis_measures import BarAdjudication, BarVerdict, MeasureCollection, MeasureSummary
from threetears.evals.contracts.authored import NO_CHART, AuthoredAnalysis
from threetears.evals.contracts.campaign import EvalAnalysis, GenerationProvenance, VariantIndexEntry
from threetears.evals.contracts.declaration import CampaignDesign, ControlDeclaration, SweptAxis
from threetears.evals.contracts.host.values import SweepableValue
from threetears.evals.contracts.identity import compute_variant_key
from threetears.evals.contracts.surface import CellFacts, DecisionSurface, MeasureFacts


AXIS = "candidate_model"
CANDIDATE = "model-b-rev2"
INCUMBENT = "model-b"
RIG = "a" * 64
RIG_B = "b" * 64
COST = "production_replicating_cost"
LATENCY = "total_ms"


def level(model: str) -> SweepableValue:
    """One level of the candidate-model axis."""
    return SweepableValue.of(model, display=model)


def key(model: str) -> str:
    """The variant key of the arm that ran ``model``."""
    return compute_variant_key({AXIS: level(model)})


def entry(model: str) -> VariantIndexEntry:
    """The index entry generation freezes for the arm that ran ``model``."""
    levers = {AXIS: level(model)}
    return VariantIndexEntry(variant_key=compute_variant_key(levers), levers=levers)


def summary(name: str, mean: float, sem: float | None = None) -> MeasureSummary:
    """A numeric measure summary — a mean, and a spread when given."""
    return MeasureSummary(
        population="scored", name=name, attribution_scope="end_to_end", n=6, n_independent=2, mean=mean, sem=sem
    )


def cell(model: str | None = None, *, variant: str | None = None, rig: str = RIG, **overrides) -> CellFacts:
    """One cell, 6 observations over 2 cases × 3 by default, carrying cost and latency."""
    defaults = {
        "variant_key": variant or key(model),
        "apparatus_class_id": rig,
        "run_ids": [f"run-{(model or variant)[:6]}-{rig[:2]}"],
        "n_observations": 6,
        "n_cases": 2,
        "repeats_per_case_min": 3,
        "repeats_per_case_max": 3,
        "measures": MeasureCollection(measures=[summary(COST, 0.0123), summary(LATENCY, 41250.0)]),
    }
    return CellFacts(**{**defaults, **overrides})


def verdict(of: CellFacts, value: float | None, sem: float | None, cleared: bool | None) -> BarVerdict:
    """A bar verdict on the cell ``of``."""
    return BarVerdict(
        variant_key=of.variant_key,
        apparatus_class_id=of.apparatus_class_id,
        run_ids=of.run_ids,
        value=value,
        sem=sem,
        n=0 if value is None else 6,
        n_independent=0 if value is None else 2,
        cleared=cleared,
    )


def measures() -> dict[str, MeasureFacts]:
    """The frozen catalogue: production's cost and latency measures, and one quality measure."""
    return {
        COST: MeasureFacts(unit="usd", merit_axis="cost", higher_is_better=False),
        LATENCY: MeasureFacts(unit="ms", merit_axis="latency", higher_is_better=False),
        "delivered_items": MeasureFacts(unit=None, merit_axis="quality", higher_is_better=True),
    }


def two_arm_surface() -> DecisionSurface:
    """Two arms under one rig, one adjudicated bar and one bar no cell could be read against."""
    candidate, incumbent = cell(CANDIDATE), cell(INCUMBENT)
    return DecisionSurface(
        control_variant_key=key(INCUMBENT),
        cells=sorted([candidate, incumbent], key=lambda c: (c.variant_key, c.apparatus_class_id)),
        bars=[
            BarAdjudication(
                measure_id="delivered_items",
                threshold=3,
                direction="higher_is_better",
                source="declared",
                state="adjudicated",
                verdicts=sorted(
                    [verdict(candidate, 3.5, 0.25, True), verdict(incumbent, 2.0, None, False)],
                    key=lambda v: (v.variant_key, v.apparatus_class_id),
                ),
            ),
            BarAdjudication(
                measure_id="grounding_rate",
                threshold=0.9,
                direction="higher_is_better",
                source="declared",
                state="names_no_stored_measure",
                reason="no non-faulted member result carried a value under this name",
            ),
        ],
        measures=measures(),
    )


def analysis(surface: DecisionSurface, **overrides) -> EvalAnalysis:
    """An analysis over the two-arm campaign, the incumbent declared as the control."""
    defaults = {
        "campaign_id": "camp-17s",
        "subject_id": "ent-maple",
        "subject_kind": "agent",
        "behavior": "conversation",
        "generation": GenerationProvenance(
            prompt_id="eval_analysis_gen",
            prompt_version="v1",
            generator_model="anthropic/claude-opus",
            bundle_fingerprint="sha256:abc",
            generated_at="2026-09-22T00:00:00+00:00",
            token_cost=0.0,
            bundle_assembled_at="2026-01-01T00:00:00+00:00",
            repair_attempts=0,
            repaired_refusal=None,
            cell_model_version=1,
            user_message_digest="sha256:message",
        ),
        "document": AuthoredAnalysis.model_validate(
            {
                "headline": "Move to the newer build.",
                "summary": "",
                "findings": [
                    {
                        "title": "rev2 holds tone at k=3.",
                        "body": "",
                        "confidence": "high",
                        "axes": [AXIS],
                        "evidence": [],
                        "chart": {"type": NO_CHART, "cells": [], "measures": [], "axis": "", "note": "", "caption": ""},
                        "caveats": [],
                        "invalidates": [],
                        "durable": "",
                    }
                ],
                "decisions": [
                    {
                        "proposal": "Move to the newer build.",
                        "disposition": "adopted",
                        "cells": [cell_ref(key(CANDIDATE), RIG)],
                        "confidence": "high",
                        "rests_on": [0],
                        "revisit_when": "",
                    }
                ],
                "questions": [],
                "next": [],
            }
        ),
        "design_snapshot": CampaignDesign(
            axes=[SweptAxis(axis_id=AXIS, values=[level(CANDIDATE), level(INCUMBENT)])],
            control=key(INCUMBENT),
            controls=ControlDeclaration(stimulus="controlled", apparatus="commissioned"),
        ),
        "variant_index": [entry(CANDIDATE), entry(INCUMBENT)],
        "decision_surface": surface,
    }
    return EvalAnalysis(**{"scope_id": "uni-1", **defaults, **overrides})


def _row(table: SurfaceTable, model: str) -> SurfaceRow:
    """The one row of the arm that ran ``model``."""
    (found,) = [row for row in table.rows if row.variant_key == key(model)]
    return found


def _value(table: SurfaceTable, row: SurfaceRow, measure_id: str) -> SurfaceValue | None:
    """The row's value under the column read on ``measure_id``."""
    (index,) = [i for i, column in enumerate(table.columns) if column.measure_id == measure_id]
    return row.values[index]


class TestTheStateThatIsNotATable:
    """A surface with no cells — a sentence, never an empty table."""

    def test_a_surface_with_no_cells_says_so_and_still_lists_its_unread_bars(self) -> None:
        surface = two_arm_surface()
        surface.cells = []
        surface.bars = [bar for bar in surface.bars if bar.state != "adjudicated"]
        table = build_surface_table(analysis(surface))
        assert table.state == "no_cells"
        assert table.disclosure == NO_CELLS
        assert table.rows == [] and table.columns == []
        assert [bar.measure_id for bar in table.unadjudicated_bars] == ["grounding_rate"]

    def test_a_measured_surface_carries_no_disclosure(self) -> None:
        table = build_surface_table(analysis(two_arm_surface()))
        assert table.state == "measured"
        assert table.disclosure is None


class TestTheProvenanceSentence:
    """Served on the table, so the renders showing it print one sentence rather than keeping copies."""

    def test_a_table_that_states_numbers_carries_it(self) -> None:
        assert build_surface_table(analysis(two_arm_surface())).provenance == SURFACE_PROVENANCE

    def test_a_table_with_no_cells_carries_it_above_its_disclosure(self) -> None:
        surface = two_arm_surface()
        surface.cells = []
        assert build_surface_table(analysis(surface)).provenance == SURFACE_PROVENANCE

    def test_it_is_in_the_served_json(self) -> None:
        served = build_surface_table(analysis(two_arm_surface())).model_dump(mode="json")
        assert served["provenance"] == SURFACE_PROVENANCE


class TestRowOrder:
    """The control's cells first, then every other by (variant_key, apparatus_class_id)."""

    def test_the_control_leads_even_when_its_key_sorts_last(self) -> None:
        # Only exercised when the control's key sorts LAST — a deriver that merely sorted by key
        # would then put it second, which is the whole thing this rule exists to prevent.
        last, first = max((CANDIDATE, INCUMBENT), key=key), min((CANDIDATE, INCUMBENT), key=key)
        surface = two_arm_surface()
        surface.control_variant_key = key(last)
        table = build_surface_table(analysis(surface))
        assert [row.variant_key for row in table.rows] == [key(last), key(first)]
        assert [row.is_control for row in table.rows] == [True, False]

    def test_without_a_control_the_rows_follow_the_cell_key(self) -> None:
        surface = two_arm_surface()
        surface.control_variant_key = None
        surface.cells.reverse()
        table = build_surface_table(analysis(surface))
        assert [row.variant_key for row in table.rows] == sorted(key(m) for m in (CANDIDATE, INCUMBENT))
        assert not any(row.is_control for row in table.rows)

    def test_two_rigs_of_one_arm_order_by_rig(self) -> None:
        surface = DecisionSurface(cells=[cell(CANDIDATE, rig=RIG_B), cell(CANDIDATE, rig=RIG)], measures=measures())
        table = build_surface_table(analysis(surface))
        assert [row.apparatus_class_id for row in table.rows] == [RIG, RIG_B]


class TestWhatARowCarriesForItsLabel:
    """The arm table's own levels — so a surface names the row with the labeller it names that arm with."""

    def test_the_levels_are_the_arm_tables_levels(self) -> None:
        subject = analysis(two_arm_surface())
        arms = {row.variant_key: row for row in arm_table(subject).rows}
        for row in build_surface_table(subject).rows:
            assert row.placed
            assert row.levels == arms[row.variant_key].levels
            assert row.levels_unavailable == arms[row.variant_key].levels_unavailable

    def test_a_swept_member_names_the_arm_as_the_arm_table_does(self) -> None:
        # `named_levers` folds a resolved surface into the knob it swept; an arm table built from it
        # and a surface built from `levers` would name the same arm two ways.
        levers = {"tool_config": level("surface-hash")}
        swept = VariantIndexEntry(
            variant_key=compute_variant_key(levers),
            levers=levers,
            swept={"planner.max_rounds": level("2")},
            folded=["tool_config"],
        )
        subject = analysis(
            DecisionSurface(cells=[cell(variant=swept.variant_key)], measures=measures()), variant_index=[swept]
        )
        (row,) = build_surface_table(subject).rows
        assert [lv.axis_id for lv in row.levels] == ["planner.max_rounds"]
        assert row.levels == next(r for r in arm_table(subject).rows if r.variant_key == swept.variant_key).levels

    def test_levels_unavailable_travels_with_its_reason(self) -> None:
        index = [VariantIndexEntry(variant_key="c" * 64, levers={}, levels_unavailable="minted under predicate v3")]
        subject = analysis(DecisionSurface(cells=[cell(variant="c" * 64)], measures=measures()), variant_index=index)
        (row,) = build_surface_table(subject).rows
        assert row.placed and row.levels == []
        assert row.levels_unavailable == "minted under predicate v3"

    def test_a_variant_the_index_lacks_is_unplaced_with_no_borrowed_levels(self) -> None:
        surface = two_arm_surface()
        surface.cells.append(cell(variant="d" * 64))
        stray = next(row for row in build_surface_table(analysis(surface)).rows if row.variant_key == "d" * 64)
        assert stray.placed is False
        assert (stray.levels, stray.levels_unavailable) == ([], None)

    def test_a_control_the_index_lacks_is_unplaced_though_the_arm_table_gives_it_a_row(self) -> None:
        subject = analysis(two_arm_surface(), variant_index=[entry(CANDIDATE)])
        assert any(row.variant_key == key(INCUMBENT) for row in arm_table(subject).rows)
        control = build_surface_table(subject).rows[0]
        assert control.variant_key == key(INCUMBENT) and control.is_control
        assert control.placed is False


class TestTheRigDisambiguator:
    """Set only when one arm was measured under more than one rig — the suffix marks ambiguity."""

    def test_two_rigs_of_one_arm_are_told_apart_and_a_one_rig_arm_is_not_marked(self) -> None:
        surface = DecisionSurface(
            cells=[cell(CANDIDATE, rig=RIG), cell(CANDIDATE, rig=RIG_B), cell(INCUMBENT)], measures=measures()
        )
        table = build_surface_table(analysis(surface))
        candidates = [row for row in table.rows if row.variant_key == key(CANDIDATE)]
        assert [row.rig for row in candidates] == [RIG[:12], RIG_B[:12]]
        assert _row(table, INCUMBENT).rig is None


class TestReplication:
    """`{n} obs over {cases} cases × {repeats}` — cases lead, since they are the independent draws."""

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            ({}, "6 obs over 2 cases × 3"),
            ({"n_observations": 5, "repeats_per_case_min": 2, "repeats_per_case_max": 3}, "5 obs over 2 cases × 2–3"),
            ({"n_cases": None, "repeats_per_case_min": None, "repeats_per_case_max": None}, "6 obs, cases unrecorded"),
            ({"repeats_per_case_min": None, "repeats_per_case_max": None}, "6 obs over 2 cases"),
            ({"n_infra_excluded": 2}, "6 obs over 2 cases × 3, 2 faulted, excluded"),
        ],
    )
    def test_the_replication_sentence(self, overrides, expected) -> None:
        surface = DecisionSurface(cells=[cell(CANDIDATE, **overrides)], measures=measures())
        assert build_surface_table(analysis(surface)).rows[0].replication == expected


class TestRunNotes:
    """A short or incomplete cell is flagged, and every sentence behind the flag is carried in order."""

    def test_flags_and_their_notes(self) -> None:
        surface = two_arm_surface()
        candidate = next(c for c in surface.cells if c.variant_key == key(CANDIDATE))
        candidate.short_runs = {"run-z": "measured 11 of 15 planned observations", "run-a": "measured 3 of 5"}
        candidate.incomplete_runs = {"run-failed-1": "failed"}
        table = build_surface_table(analysis(surface))
        row = _row(table, CANDIDATE)
        assert row.flags == ["short", "incomplete"]
        assert [(note.kind, note.run_id, note.text) for note in row.run_notes] == [
            ("short", "run-a", "measured 3 of 5"),
            ("short", "run-z", "measured 11 of 15 planned observations"),
            ("incomplete", "run-failed-1", "status failed"),
        ]
        assert (_row(table, INCUMBENT).flags, _row(table, INCUMBENT).run_notes) == ([], [])


class TestBarColumns:
    """Every adjudicated bar is a column; every other bar is listed below, never a column of misses."""

    def test_an_adjudicated_bar_is_a_column_with_its_direction_and_threshold(self) -> None:
        table = build_surface_table(analysis(two_arm_surface()))
        (bar,) = [column for column in table.columns if column.kind == "bar"]
        assert (bar.measure_id, bar.direction, bar.source, bar.threshold, bar.unit) == (
            "delivered_items",
            "higher_is_better",
            "declared",
            3,
            "",
        )
        assert bar.header == "delivered_items ≥ 3"

    def test_a_registered_lower_is_better_bar_states_its_threshold_on_the_columns_ruler(self) -> None:
        surface = two_arm_surface()
        surface.bars.append(
            BarAdjudication(
                measure_id=LATENCY,
                threshold=45000,
                direction="lower_is_better",
                source="registered",
                state="adjudicated",
                verdicts=[verdict(c, 41250.0, 900.0, True) for c in surface.cells],
            )
        )
        table = build_surface_table(analysis(surface))
        bar = next(column for column in table.columns if column.kind == "bar" and column.measure_id == LATENCY)
        # 45000 ms is 45 s, and the cells under it are restated in seconds with it.
        assert (bar.threshold, bar.unit, bar.header) == (45, "s", f"{LATENCY} ≤ 45 s (registered)")
        index = table.columns.index(bar)
        served = table.rows[0].values[index]
        assert (served.value, served.sem, served.text) == (41.25, 0.9, "41.25 ± 0.9 (n=6)")

    def test_the_threshold_alone_can_move_the_ruler(self) -> None:
        # Every verdict value is under a second, but the threshold is 1.5 s: the column is ONE ruler,
        # so the header and the cells under it move to seconds together.
        surface = two_arm_surface()
        surface.bars = [
            BarAdjudication(
                measure_id=LATENCY,
                threshold=1500,
                direction="lower_is_better",
                source="declared",
                state="adjudicated",
                verdicts=[verdict(c, 900.0, None, True) for c in surface.cells],
            )
        ]
        table = build_surface_table(analysis(surface))
        assert table.columns[0].header == f"{LATENCY} ≤ 1.5 s"
        assert table.rows[0].values[0].text == "0.9 (n=6)"

    @pytest.mark.parametrize(
        ("cleared", "word", "said"),
        [(True, "clears", "clears"), (False, "misses", "misses"), (None, "no_data", "no data")],
    )
    def test_the_verdict_word(self, cleared, word, said) -> None:
        surface = two_arm_surface()
        surface.bars[0].verdicts = [verdict(c, None if cleared is None else 3.5, None, cleared) for c in surface.cells]
        value = build_surface_table(analysis(surface)).rows[0].values[0]
        assert value.verdict == word
        assert value.verdict_word == said
        assert value.model_dump(mode="json")["verdict_word"] == said

    def test_a_merit_value_carries_no_verdict_word(self) -> None:
        table = build_surface_table(analysis(two_arm_surface()))
        merit = [
            value for column, value in zip(table.columns, table.rows[0].values, strict=True) if column.kind == "merit"
        ]
        assert merit and all(value is not None and value.verdict_word is None for value in merit)

    def test_a_cell_with_no_observation_reads_no_data_never_misses(self) -> None:
        surface = two_arm_surface()
        surface.bars[0].verdicts = [verdict(c, None, None, None) for c in surface.cells]
        value = build_surface_table(analysis(surface)).rows[0].values[0]
        assert (value.value, value.sem, value.n, value.verdict, value.text) == (None, None, 0, "no_data", "— (n=0)")

    def test_the_value_spread_and_sample_are_spelled_once(self) -> None:
        table = build_surface_table(analysis(two_arm_surface()))
        assert _value(table, _row(table, CANDIDATE), "delivered_items").text == "3.5 ± 0.25 (n=6)"
        # sem None → no "±" at all, never a dash standing in for a spread.
        assert _value(table, _row(table, INCUMBENT), "delivered_items").text == "2 (n=6)"

    def test_a_cell_the_server_wrote_no_verdict_for_is_absent_not_no_data(self) -> None:
        surface = two_arm_surface()
        surface.bars[0].verdicts = [v for v in surface.bars[0].verdicts if v.variant_key == key(CANDIDATE)]
        table = build_surface_table(analysis(surface))
        assert _value(table, _row(table, INCUMBENT), "delivered_items") is None
        assert _value(table, _row(table, CANDIDATE), "delivered_items") is not None

    @pytest.mark.parametrize(
        ("state", "reason", "listed"),
        [
            ("names_no_stored_measure", "nothing carried it", "nothing carried it"),
            ("not_numeric", "the measure is categorical", "the measure is categorical"),
            ("not_numeric", None, "not_numeric"),
        ],
    )
    def test_a_bar_with_no_verdict_is_listed_with_its_reason_and_is_not_a_column(self, state, reason, listed) -> None:
        surface = two_arm_surface()
        surface.bars = [
            surface.bars[0],
            BarAdjudication(
                measure_id="tone",
                threshold=1,
                direction="higher_is_better",
                source="registered",
                state=state,
                reason=reason,
            ),
        ]
        table = build_surface_table(analysis(surface))
        assert "tone" not in [column.measure_id for column in table.columns]
        (bar,) = table.unadjudicated_bars
        assert (bar.measure_id, bar.source, bar.state, bar.reason) == ("tone", "registered", state, listed)


class TestCostAndLatencyColumns:
    """Chosen by each measure's FROZEN merit axis — not by name — and only when a cell carries one."""

    def test_cost_then_latency_after_the_bars_each_with_its_unit_once(self) -> None:
        table = build_surface_table(analysis(two_arm_surface()))
        assert [(c.kind, c.measure_id, c.axis, c.unit, c.header) for c in table.columns] == [
            ("bar", "delivered_items", None, "", "delivered_items ≥ 3"),
            ("merit", COST, "cost", "usd", f"{COST} (usd)"),
            ("merit", LATENCY, "latency", "s", f"{LATENCY} (s)"),
        ]
        row = table.rows[0]
        # No sem below n=2 in the fixture's summaries, so no "±" — but the sample is always stated.
        assert [v.text for v in row.values[1:]] == ["0.0123 (n=6)", "41.25 (n=6)"]
        assert [v.verdict for v in row.values[1:]] == [None, None]

    def test_a_cost_and_a_latency_cell_state_their_spread_and_sample_as_a_bar_does(self) -> None:
        # A mean alone cannot be weighed: every cost and latency cell carries its sem and n in the
        # spelling a bar cell uses, restated on the column's ruler, each read off the summary behind
        # the mean — n=5 and n=4 here, deliberately not the cell's 6 observations.
        surface = two_arm_surface()
        for c in surface.cells:
            c.measures = MeasureCollection(
                measures=[
                    MeasureSummary(
                        population="scored",
                        name=COST,
                        attribution_scope="end_to_end",
                        n=5,
                        n_independent=2,
                        mean=0.0123,
                        sem=0.0021,
                    ),
                    MeasureSummary(
                        population="scored",
                        name=LATENCY,
                        attribution_scope="end_to_end",
                        n=4,
                        n_independent=2,
                        mean=41250.0,
                        sem=900.0,
                    ),
                ]
            )
        table = build_surface_table(analysis(surface))
        cost, latency = _value(table, table.rows[0], COST), _value(table, table.rows[0], LATENCY)
        assert (cost.value, cost.sem, cost.n, cost.text) == (0.0123, 0.0021, 5, "0.0123 ± 0.0021 (n=5)")
        assert (latency.value, latency.sem, latency.n, latency.text) == (41.25, 0.9, 4, "41.25 ± 0.9 (n=4)")

    @pytest.mark.parametrize(
        ("means", "unit", "shown"),
        [((900.0, 1200.0), "s", ["0.9", "1.2"]), ((450.0, 450.0), "ms", ["450", "450"])],
    )
    def test_one_ruler_per_column_never_one_per_value(self, means, unit, shown) -> None:
        # 900 ms beside 1200 ms: the largest picks seconds and the smaller is restated in seconds
        # too — "900 ms" over "1.2 s" is two rulers. At 450 ms nothing reaches a second, so ms stays.
        surface = two_arm_surface()
        for c, mean in zip(surface.cells, means, strict=True):
            c.measures = MeasureCollection(measures=[summary(LATENCY, mean, sem=mean / 10)])
        table = build_surface_table(analysis(surface))
        latency = next(column for column in table.columns if column.measure_id == LATENCY)
        assert (latency.unit, latency.header) == (unit, f"{LATENCY} ({unit})")
        values = [_value(table, row, LATENCY) for row in table.rows]
        assert sorted(v.text.split(" ± ")[0] for v in values) == shown
        # The spread rides the same ruler as the mean it qualifies.
        assert all(v.sem == pytest.approx(v.value / 10) for v in values)

    def test_the_axis_decides_not_the_name(self) -> None:
        surface = two_arm_surface()
        surface.measures["provider_spend"] = MeasureFacts(unit="usd", merit_axis="cost", higher_is_better=False)
        surface.measures[COST] = MeasureFacts(unit="usd", merit_axis="quality", higher_is_better=False)
        surface.cells[0].measures.measures.append(summary("provider_spend", 0.5))
        merit = [c.measure_id for c in build_surface_table(analysis(surface)).columns if c.kind == "merit"]
        assert merit == ["provider_spend", LATENCY]

    def test_a_quality_measure_is_never_a_merit_column(self) -> None:
        surface = two_arm_surface()
        surface.cells[0].measures.measures.append(summary("delivered_items", 3.5))
        columns = build_surface_table(analysis(surface)).columns
        assert [c.kind for c in columns if c.measure_id == "delivered_items"] == ["bar"]

    def test_two_measures_on_one_axis_order_by_name(self) -> None:
        surface = two_arm_surface()
        surface.measures["a_cost"] = MeasureFacts(unit="usd", merit_axis="cost", higher_is_better=False)
        surface.cells[0].measures.measures.append(summary("a_cost", 0.5))
        merit = [c.measure_id for c in build_surface_table(analysis(surface)).columns if c.kind == "merit"]
        assert merit == ["a_cost", COST, LATENCY]

    def test_a_cell_lacking_the_measure_has_no_value_there(self) -> None:
        surface = two_arm_surface()
        lacking = next(c for c in surface.cells if c.variant_key == key(CANDIDATE))
        lacking.measures = MeasureCollection(measures=[summary(LATENCY, 100.0)])
        table = build_surface_table(analysis(surface))
        assert _value(table, _row(table, CANDIDATE), COST) is None
        assert _value(table, _row(table, INCUMBENT), COST).text == "0.0123 (n=6)"

    def test_a_merit_measure_no_cell_carries_gets_no_column(self) -> None:
        surface = two_arm_surface()
        surface.measures["tokens_usd"] = MeasureFacts(unit="usd", merit_axis="cost", higher_is_better=False)
        assert "tokens_usd" not in [c.measure_id for c in build_surface_table(analysis(surface)).columns]


class TestTheTableRefusesAnInconsistentShape:
    """Every guard on the served models fires on the shape it forbids."""

    def _bar(self, **overrides) -> dict:
        return {
            "kind": "bar",
            "measure_id": "m",
            "header": "m ≥ 1",
            "direction": "higher_is_better",
            "source": "declared",
            "threshold": 1.0,
        } | overrides

    def _merit(self, **overrides) -> dict:
        return {"kind": "merit", "measure_id": "c", "header": "c", "axis": "cost"} | overrides

    def test_the_valid_columns_construct(self) -> None:
        assert SurfaceColumn(**self._bar()).kind == "bar"
        assert SurfaceColumn(**self._merit()).kind == "merit"

    @pytest.mark.parametrize(
        "overrides",
        [{"direction": None}, {"source": None}, {"threshold": None}, {"axis": "cost"}],
    )
    def test_a_bar_column_without_its_facts_or_with_an_axis_is_refused(self, overrides) -> None:
        with pytest.raises(ValidationError, match="bar column"):
            SurfaceColumn(**self._bar(**overrides))

    @pytest.mark.parametrize(
        "overrides",
        [{"axis": None}, {"direction": "lower_is_better"}, {"source": "declared"}, {"threshold": 1.0}],
    )
    def test_a_merit_column_without_its_axis_or_with_a_bar_fact_is_refused(self, overrides) -> None:
        with pytest.raises(ValidationError, match="merit column"):
            SurfaceColumn(**self._merit(**overrides))

    def _row(self, values: list) -> SurfaceRow:
        return SurfaceRow(variant_key="v", apparatus_class_id="r", placed=True, replication="6 obs", values=values)

    def test_the_valid_table_constructs(self) -> None:
        table = SurfaceTable(
            state="measured",
            columns=[SurfaceColumn(**self._bar()), SurfaceColumn(**self._merit())],
            rows=[self._row([SurfaceValue(n=1, verdict="clears", text="1 (n=1)"), None])],
        )
        assert table.state == "measured"

    @pytest.mark.parametrize(
        ("state", "disclosure"),
        [("measured", "something"), ("no_cells", None)],
    )
    def test_a_disclosure_exactly_when_the_state_is_not_measured(self, state, disclosure) -> None:
        with pytest.raises(ValidationError, match="disclosure"):
            SurfaceTable(state=state, disclosure=disclosure, rows=[self._row([])] if state == "measured" else [])

    def test_rows_exactly_when_measured(self) -> None:
        with pytest.raises(ValidationError, match="rows exactly"):
            SurfaceTable(state="measured")
        with pytest.raises(ValidationError, match="rows exactly"):
            SurfaceTable(state="no_cells", disclosure=NO_CELLS, rows=[self._row([])])

    def test_a_row_whose_values_do_not_fit_the_columns_is_refused(self) -> None:
        with pytest.raises(ValidationError, match=r"states 0 value\(s\) under 1 column"):
            SurfaceTable(state="measured", columns=[SurfaceColumn(**self._merit())], rows=[self._row([])])

    def test_a_verdict_under_a_merit_column_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="verdict exactly where"):
            SurfaceTable(
                state="measured",
                columns=[SurfaceColumn(**self._merit())],
                rows=[self._row([SurfaceValue(n=1, verdict="clears", text="1")])],
            )

    def test_a_bar_value_without_a_verdict_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="verdict exactly where"):
            SurfaceTable(
                state="measured",
                columns=[SurfaceColumn(**self._bar())],
                rows=[self._row([SurfaceValue(n=1, text="1 (n=1)")])],
            )


class TestTheDeriverWritesNothingBack:
    """Derived on every read — the stored analysis is unchanged by a derivation."""

    def test_the_analysis_is_unchanged(self) -> None:
        subject = analysis(two_arm_surface())
        before = subject.model_dump(mode="json")
        build_surface_table(subject)
        assert subject.model_dump(mode="json") == before

    def test_the_served_shape_round_trips(self) -> None:
        table = build_surface_table(analysis(two_arm_surface()))
        assert SurfaceTable.model_validate(table.model_dump(mode="json")) == table
