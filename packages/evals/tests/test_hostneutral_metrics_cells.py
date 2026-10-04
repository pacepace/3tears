"""Metric description and cell naming, over inputs that carry no host vocabulary.

Every measure name here is either one the engine's own registry seeds (``total_ms`` and its
parts) or one invented for this file (``parse_page_ms``), and every cell is a bare model instance.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.evals.analysis.cells import Cell, NextExperiment
from threetears.evals.contracts.host import MeasureRegistry
from threetears.evals.contracts.metrics import describe_measure, describe_reported_measure

#: A host that declares no measures of its own: everything described here is the engine's core.
_NO_HOST_MEASURES = MeasureRegistry([])


class TestAReportedMeasureIsDescribedTheWayItsSummaryWas:
    """A phase timing that reached a report resolves as a phase, not as an unclassified name."""

    def test_a_phase_key_reads_as_measured_wall_clock(self) -> None:
        # The bare lookup has no phase branch — a phase key is indistinguishable by name from a
        # run-summary statistic — so this is the state the reported-measure resolution exists for.
        assert describe_measure("parse_page_ms", _NO_HOST_MEASURES).family is None
        described = describe_reported_measure("parse_page_ms", _NO_HOST_MEASURES)
        assert described.family == "mechanical"
        assert described.unit == "ms"
        assert described.attribution_scope == "subsystem"
        assert "parse page phase" in described.description

    def test_a_seeded_measure_keeps_its_seeded_description(self) -> None:
        assert describe_reported_measure("total_ms", _NO_HOST_MEASURES) == describe_measure(
            "total_ms", _NO_HOST_MEASURES
        )
        assert describe_reported_measure("total_ms", _NO_HOST_MEASURES).merit_axis == "latency"


_SIX = [f"obs-{i}" for i in range(6)]


def _missing(error: ValidationError) -> set[str]:
    """The fields a validation error reports as absent — the refusal a retired spelling must meet."""
    return {".".join(map(str, e["loc"])) for e in error.errors() if e["type"] == "missing"}


class TestACellIsReadOnlyByItsCountsName:
    """A cell's pooled count is ``n_observations``; the retired ``k`` spelling is not read as it.

    Each refusal is asserted as the canonical field reported MISSING, on input that would otherwise
    construct — so a read alias that accepted ``k`` would build the cell and the assertion would fail,
    rather than the cell failing some other check for an unrelated reason.
    """

    def test_the_count_is_read_by_its_name(self) -> None:
        cell = Cell(
            variant_key="v-1", apparatus_class_id="rig-1", provenance="declared", n_observations=6, observation_ids=_SIX
        )
        assert cell.n_observations == 6

    def test_k_is_not_a_spelling_of_the_count(self) -> None:
        with pytest.raises(ValidationError) as refused:
            Cell(variant_key="v-1", apparatus_class_id="rig-1", provenance="declared", k=6, observation_ids=_SIX)
        assert _missing(refused.value) == {"n_observations"}

    def test_a_next_experiment_reads_its_counts_only_by_their_names(self) -> None:
        named = NextExperiment(
            variant_key="v-1", dimension="rig.width", n_observations_now=3, n_observations_if_recorded=9
        )
        assert (named.n_observations_now, named.n_observations_if_recorded) == (3, 9)
        with pytest.raises(ValidationError) as refused:
            NextExperiment(variant_key="v-1", dimension="rig.width", k_now=3, k_if_recorded=9)
        assert _missing(refused.value) == {"n_observations_now", "n_observations_if_recorded"}
