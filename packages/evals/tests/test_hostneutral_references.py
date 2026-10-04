"""The reference contract's remaining hunks, over a decision surface with no host vocabulary.

The subject is an invoice extractor swept over two models under one rig. Every surface, cell and
index entry is built here by hand. Nothing is imported from a host adapter, and the toy corpus is
not used, because none of these modules reads a bundle.

Each class names the engine file and the property it pins. The shared suites
(``test_references.py``, ``test_viz_refs.py``) already carry most of the contract. This file holds
the cases a mutation left green there.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import errors, viz_refs
from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.contracts.host import MeasureRegistry
from threetears.evals.analysis.viz_refs import build_viz_payload, cell_arm_labels, reference_from_chart
from threetears.evals.contracts.analysis_measures import MeasureCollection, MeasureSummary
from threetears.evals.contracts.authored import Chart, MeasureRef
from threetears.evals.contracts.campaign import VariantIndexEntry
from threetears.evals.contracts.host.values import SweepableValue
from threetears.evals.contracts.identity import compute_variant_key
from threetears.evals.contracts.surface import (
    CellFacts,
    DecisionSurface,
    JudgedDimensionFacts,
    JudgedReading,
    MeasureFacts,
)


AXIS = "extractor_model"
RIG = "r" * 64
RIG_2 = "s" * 64


def _level(model: str) -> SweepableValue:
    return SweepableValue.of(model, display=model)


def _key(model: str) -> str:
    return compute_variant_key({AXIS: _level(model)})


BASELINE, CHALLENGER = _key("model-a"), _key("model-b")
INDEX = [
    VariantIndexEntry(variant_key=BASELINE, levers={AXIS: _level("model-a")}),
    VariantIndexEntry(variant_key=CHALLENGER, levers={AXIS: _level("model-b")}),
]


def _summary(name: str, mean: float, *, scope: str = "end_to_end") -> MeasureSummary:
    return MeasureSummary(
        name=name,
        attribution_scope=scope,
        higher_is_better=False,
        n=6,
        n_independent=6,
        mean=mean,
        sem=mean / 20,
        ci_low=mean * 0.9,
        ci_high=mean * 1.1,
    )


def _cell(variant_key: str, *, rig: str = RIG, scale: float = 1.0) -> CellFacts:
    return CellFacts(
        variant_key=variant_key,
        apparatus_class_id=rig,
        run_ids=[f"run-{variant_key[:6]}-{rig[:2]}"],
        n_observations=6,
        measures=MeasureCollection(
            measures=[
                _summary("extract_ms", 900.0 * scale),
                _summary("parse_ms", 300.0 * scale, scope="subsystem"),
            ]
        ),
        judged=[JudgedReading(dimension="extraction.layout_kept", mean=4.0, sem=0.2, n=6, n_independent=6)],
    )


def _surface(cells: list[CellFacts] | None = None) -> DecisionSurface:
    cells = cells if cells is not None else [_cell(BASELINE), _cell(CHALLENGER, scale=1.2)]
    return DecisionSurface(
        control_variant_key=BASELINE,
        cells=sorted(cells, key=lambda c: (c.variant_key, c.apparatus_class_id)),
        measures={
            "extract_ms": MeasureFacts(unit="ms", merit_axis="latency", higher_is_better=False),
            "parse_ms": MeasureFacts(unit="ms", merit_axis=None, higher_is_better=False),
        },
        dimensions={"extraction.layout_kept": JudgedDimensionFacts(higher_is_better=True, value_range=(1.0, 5.0))},
    )


# --- errors.py -------------------------------------------------------------------------------------


class TestTheRefusalIsPublishedFromTheOneImportPath:
    """``UnresolvableReference`` is in ``errors.__all__``."""

    def test_the_reference_refusal_is_part_of_the_published_contract(self) -> None:
        assert "UnresolvableReference" in errors.__all__
        assert errors.UnresolvableReference is UnresolvableReference


# --- references.py / viz_refs.py ------------------------------------------------------------------


class TestAnEmptyCellNamesTheCellsThatExist:
    """A chart naming a cell the surface does not hold is refused listing the surface's cells, so the repair can pick one."""

    @staticmethod
    def _null_result(cells: list[str], measure_id: str = "extract_ms") -> Chart:
        return Chart(
            type="null_result",
            cells=cells,
            measures=[MeasureRef(measure_id=measure_id, reading="measure")],
            axis="",
            note="the parser never ran long enough to matter",
            caption="",
        )

    def _build(self, authored: Chart) -> dict[str, Any]:
        # A host declaring no measures of its own: the null-result chart reads no catalogue.
        return build_viz_payload(reference_from_chart(authored), _surface(), INDEX, measures=MeasureRegistry([]))

    def test_an_empty_cell_ref_is_refused_with_every_cell_on_the_surface(self) -> None:
        with pytest.raises(UnresolvableReference) as raised:
            self._build(self._null_result(["", cell_ref(CHALLENGER, RIG)]))

        message = str(raised.value)
        assert "the cells are: " in message
        assert cell_ref(BASELINE, RIG) in message and cell_ref(CHALLENGER, RIG) in message

    def test_a_bad_measure_names_the_measures_not_the_cells(self) -> None:
        """The control: a bad measure has nothing to do with which cells exist."""
        with pytest.raises(UnresolvableReference) as raised:
            self._build(self._null_result([cell_ref(BASELINE, RIG), cell_ref(CHALLENGER, RIG)], measure_id=""))

        assert "the cells are" not in str(raised.value)
        assert "Its measures are: extract_ms, parse_ms" in str(raised.value)


class TestEveryCellIsNamedByItsArm:
    """The cell labeller is public, so the memo's render names an arm as the charts do."""

    def test_the_labeller_is_part_of_the_published_contract(self) -> None:
        assert "cell_arm_labels" in viz_refs.__all__

    def test_each_cell_is_named_by_its_levels(self) -> None:
        labels = cell_arm_labels(_surface(), INDEX)

        assert labels == {
            cell_ref(BASELINE, RIG): f"{AXIS}=model-a",
            cell_ref(CHALLENGER, RIG): f"{AXIS}=model-b",
        }

    def test_an_arm_under_two_rigs_carries_its_rig_in_every_label(self) -> None:
        cells = [_cell(BASELINE), _cell(BASELINE, rig=RIG_2), _cell(CHALLENGER)]
        labels = cell_arm_labels(_surface(cells), INDEX)

        assert labels[cell_ref(BASELINE, RIG)].endswith(" @ rig rrrrrrrrrrrr")
        assert labels[cell_ref(BASELINE, RIG_2)].endswith(" @ rig ssssssssssss")
        assert labels[cell_ref(CHALLENGER, RIG)] == f"{AXIS}=model-b"


# --- surface.py ------------------------------------------------------------------------------------


def _surface_document(**overrides: Any) -> dict[str, Any]:
    return {**_surface().model_dump(mode="json"), **overrides}


class TestTheDecisionSurfaceIsAStoredContract:
    """What the surface carries survives a stored round-trip."""

    def test_a_cell_names_at_least_one_member_run(self) -> None:
        with pytest.raises(ValidationError, match=r"run_ids"):
            CellFacts(variant_key=BASELINE, apparatus_class_id=RIG, run_ids=[], n_observations=1)

    def test_the_facts_survive_a_round_trip(self) -> None:
        again = DecisionSurface.model_validate(_surface().model_dump(mode="json"))

        # Read field by field: comparing two instances built through the same dump would agree even if
        # the dump dropped a field on both sides.
        facts = again.measures["extract_ms"]
        assert (facts.unit, facts.merit_axis, facts.higher_is_better) == ("ms", "latency", False)
        assert (
            again.dimensions["extraction.layout_kept"].higher_is_better,
            again.dimensions["extraction.layout_kept"].value_range,
        ) == (
            True,
            (1.0, 5.0),
        )
        assert [c.judged[0].dimension for c in again.cells] == ["extraction.layout_kept", "extraction.layout_kept"]

    def test_a_judged_dimension_is_read_higher_is_better_unless_declared_otherwise(self) -> None:
        assert JudgedDimensionFacts().higher_is_better is True
        assert JudgedDimensionFacts(higher_is_better=False).higher_is_better is False

    def test_a_surface_stored_before_judged_references_loads_with_no_dimension_facts(self) -> None:
        document = _surface_document()
        document.pop("dimensions")
        assert DecisionSurface.model_validate(document).dimensions == {}

    def test_a_surface_stored_before_measure_facts_loads_with_none(self) -> None:
        document = _surface_document()
        document.pop("measures")
        assert DecisionSurface.model_validate(document).measures == {}
