"""The decision surface's table, laid out for a campaign that names no host.

:func:`threetears.evals.analysis.surface_table.build_surface_table` turns an analysis's frozen per-cell
facts into the one table every surface renders. No toy-driving test reaches it, because it reads a
stored :class:`~threetears.evals.kernel.campaign.EvalAnalysis` rather than a bundle. The substitute
evidence is an analysis built by hand for an invoice extractor swept over two models. It has no
adapter import and no host vocabulary.

Pinned here: the states that are not a table, the row order, the served row label, the one ruler
per column, the verdict words, the replication sentence, and the one number spelling.
"""

from __future__ import annotations


from threetears.evals.analysis.arms import short_digest
from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.surface_table import (
    NO_CELLS,
    SurfaceTable,
    build_surface_table,
)
from threetears.evals.kernel.analysis_measures import BarAdjudication, BarVerdict, MeasureCollection, MeasureSummary
from threetears.evals.kernel.authored import NO_CHART, AuthoredAnalysis
from threetears.evals.kernel.campaign import EvalAnalysis, GenerationProvenance, VariantIndexEntry
from threetears.evals.kernel.declaration import CampaignDesign, ControlDeclaration, SweptAxis
from threetears.evals.schema.values import SweepableValue
from threetears.evals.kernel.identity import compute_variant_key
from threetears.evals.kernel.surface import CellFacts, DecisionSurface, MeasureFacts


AXIS = "extractor_model"
BASELINE = "model-a"
CHALLENGER = "model-b"
RIG = "1" * 64
RIG_2 = "2" * 64
COST = "cost_usd"
LATENCY = "extract_ms"
FIELDS = "fields_correct"


def _level(model: str) -> SweepableValue:
    return SweepableValue.of(model, display=model)


def _key(model: str) -> str:
    return compute_variant_key({AXIS: _level(model)})


def _entry(model: str) -> VariantIndexEntry:
    return VariantIndexEntry(variant_key=_key(model), levers={AXIS: _level(model)})


def _summary(name: str, mean: float) -> MeasureSummary:
    return MeasureSummary(
        population="scored", name=name, attribution_scope="end_to_end", n=6, n_independent=2, mean=mean, sem=mean / 10
    )


def _cell(variant_key: str, *, rig: str = RIG, latency: float = 900.0) -> CellFacts:
    return CellFacts(
        variant_key=variant_key,
        apparatus_class_id=rig,
        run_ids=[f"run-{variant_key[:6]}-{rig[:2]}"],
        n_observations=6,
        n_cases=2,
        repeats_per_case_min=3,
        repeats_per_case_max=3,
        measures=MeasureCollection(measures=[_summary(COST, 0.004), _summary(LATENCY, latency)]),
    )


def _verdict(cell: CellFacts, value: float | None, cleared: bool | None) -> BarVerdict:
    return BarVerdict(
        variant_key=cell.variant_key,
        apparatus_class_id=cell.apparatus_class_id,
        run_ids=cell.run_ids,
        value=value,
        sem=None if value is None else 250.4,
        n=0 if value is None else 6,
        n_independent=0 if value is None else 2,
        # A decided verdict is decided on an interval; the bounds only need to exist here.
        ci_low=None if value is None or cleared is None else value - 500.0,
        ci_high=None if value is None or cleared is None else value + 500.0,
        cleared=cleared,
    )


def _surface(cells: list[CellFacts], verdicts: dict[str, tuple[float | None, bool | None]]) -> DecisionSurface:
    return DecisionSurface(
        control_variant_key=_key(BASELINE),
        cells=sorted(cells, key=lambda c: (c.variant_key, c.apparatus_class_id)),
        bars=[
            BarAdjudication(
                measure_id=FIELDS,
                threshold=10000,
                direction="higher_is_better",
                source="declared",
                state="adjudicated",
                verdicts=sorted(
                    (_verdict(c, *verdicts[c.variant_key]) for c in cells),
                    key=lambda v: (v.variant_key, v.apparatus_class_id),
                ),
            ),
            BarAdjudication(
                measure_id="totals_match",
                threshold=0.95,
                direction="higher_is_better",
                source="declared",
                state="names_no_stored_measure",
                reason="no result carried this measure",
            ),
        ],
        measures={
            COST: MeasureFacts(unit="usd", merit_axis="cost", higher_is_better=False),
            LATENCY: MeasureFacts(unit="ms", merit_axis="latency", higher_is_better=False),
            FIELDS: MeasureFacts(unit=None, merit_axis="quality", higher_is_better=True),
        },
    )


def analysis_over(surface: DecisionSurface, *, index: list[VariantIndexEntry] | None = None) -> EvalAnalysis:
    return EvalAnalysis(
        scope_id="uni-1",
        campaign_id="camp-invoices",
        subject_id="invoice-extractor",
        subject_kind="extractor_config",
        behavior="extraction",
        generation=GenerationProvenance(
            prompt_id="gen",
            prompt_version="v1",
            generator_model="gen/model",
            bundle_fingerprint="sha256:0",
            generated_at="2026-01-01T00:00:00+00:00",
            token_cost=0.0,
            bundle_assembled_at="2026-01-01T00:00:00+00:00",
            repair_attempts=0,
            repaired_refusal=None,
            cell_model_version=1,
            user_message_digest="sha256:message",
        ),
        document=AuthoredAnalysis.model_validate(
            {
                "headline": "Adopt model-b.",
                "summary": "",
                "findings": [
                    {
                        "title": "model-b extracts more fields.",
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
                        "proposal": "Adopt model-b.",
                        "disposition": "adopted",
                        "cells": [cell_ref(_key(CHALLENGER), RIG)],
                        "confidence": "high",
                        "rests_on": [0],
                        "revisit_when": "",
                    }
                ],
                "questions": [],
                "next": [],
            }
        ),
        design_snapshot=CampaignDesign(
            axes=[SweptAxis(axis_id=AXIS, values=[_level(BASELINE), _level(CHALLENGER)])],
            control=_key(BASELINE),
            held_fixed=ControlDeclaration(stimulus="controlled", apparatus="commissioned"),
        ),
        variant_index=index if index is not None else [_entry(BASELINE), _entry(CHALLENGER)],
        decision_surface=surface,
    )


def two_arm_table() -> SurfaceTable:
    baseline, challenger = _cell(_key(BASELINE), latency=900.0), _cell(_key(CHALLENGER), latency=1200.0)
    verdicts = {baseline.variant_key: (12345.6, True), challenger.variant_key: (None, None)}
    return build_surface_table(analysis_over(_surface([baseline, challenger], verdicts)))


class TestTheStatesThatAreNotATable:
    def test_a_surface_with_no_cells_says_so_and_still_lists_the_bar_it_could_not_read(self) -> None:
        table = build_surface_table(analysis_over(_surface([], {})))

        assert (table.state, table.disclosure) == ("no_cells", NO_CELLS)
        assert [bar.measure_id for bar in table.unadjudicated_bars] == ["totals_match"]


class TestTheRows:
    def test_the_control_leads_whatever_its_key_sorts_as(self) -> None:
        table = two_arm_table()

        assert table.rows[0].variant_key == _key(BASELINE)
        assert [row.is_control for row in table.rows] == [True, False]

    def test_a_row_is_served_the_arm_tables_label(self) -> None:
        table = two_arm_table()

        assert [row.label for row in table.rows] == [f"{AXIS}={BASELINE}", f"{AXIS}={CHALLENGER}"]
        assert "label" in table.rows[0].model_dump(mode="json")

    def test_an_arm_the_index_lacks_is_labelled_unplaced_not_given_a_neighbours_name(self) -> None:
        cells = [_cell(_key(BASELINE)), _cell(_key(CHALLENGER))]
        verdicts = {c.variant_key: (1.0, False) for c in cells}
        table = build_surface_table(analysis_over(_surface(cells, verdicts), index=[_entry(BASELINE)]))

        unplaced = next(row for row in table.rows if row.variant_key == _key(CHALLENGER))
        assert (unplaced.placed, unplaced.levels) == (False, [])
        assert unplaced.label.startswith(f"unplaced ({short_digest(_key(CHALLENGER))})")

    def test_only_an_arm_measured_under_two_rigs_names_its_rig(self) -> None:
        cells = [_cell(_key(BASELINE)), _cell(_key(CHALLENGER), rig=RIG), _cell(_key(CHALLENGER), rig=RIG_2)]
        verdicts = {c.variant_key: (1.0, False) for c in cells}
        table = build_surface_table(analysis_over(_surface(cells, verdicts)))

        assert [row.label for row in table.rows] == [
            f"{AXIS}={BASELINE}",
            f"{AXIS}={CHALLENGER} @ rig {short_digest(RIG)}",
            f"{AXIS}={CHALLENGER} @ rig {short_digest(RIG_2)}",
        ]

    def test_the_replication_sentence_leads_with_the_cases(self) -> None:
        assert two_arm_table().rows[0].replication == "6 obs over 2 cases × 3"


class TestTheColumns:
    def test_bars_first_then_cost_then_latency_each_on_one_ruler(self) -> None:
        table = two_arm_table()

        assert [(c.kind, c.measure_id, c.unit) for c in table.columns] == [
            ("bar", FIELDS, ""),
            ("merit", COST, "usd"),
            ("merit", LATENCY, "s"),
        ]
        # 900 ms beside 1200 ms: the column is stated in seconds, the smaller value included.
        assert [row.values[2].text for row in table.rows] == ["0.9 ± 0.09 (n=6)", "1.2 ± 0.12 (n=6)"]

    def test_a_bar_that_names_no_stored_measure_is_listed_not_drawn(self) -> None:
        table = two_arm_table()

        assert "totals_match" not in [c.measure_id for c in table.columns]
        assert [(b.measure_id, b.state) for b in table.unadjudicated_bars] == [
            ("totals_match", "names_no_stored_measure")
        ]


class TestTheValuesUnderABar:
    def test_a_verdict_is_a_word_and_a_cell_with_no_observation_is_no_data(self) -> None:
        table = two_arm_table()

        assert [row.values[0].verdict for row in table.rows] == ["clears", "no_data"]

    def test_a_large_value_is_spelled_whole_beside_its_spread_and_sample(self) -> None:
        """The one number rule: 12345.6 reads ``12346``, never with an exponent or a separator."""
        assert two_arm_table().rows[0].values[0].text == "12346 ± 250.4 (n=6)"

    def test_the_header_states_the_threshold_by_the_same_rule(self) -> None:
        assert two_arm_table().columns[0].header == f"{FIELDS} ≥ 10000"
