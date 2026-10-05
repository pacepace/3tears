"""The one resolver every code-filled number goes through — what it reads, and what it refuses.

Each refusal is asserted per forbidden shape, because a guard that only ever sees valid input has
proved it rejects nothing. The judged interval is recomputed here from the t multiplier rather than
compared against a literal, so a change to the method fails loudly instead of re-pinning.
"""

from __future__ import annotations

import pytest

from threetears.evals.analysis import stats
from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.errors import SoundnessRefusal, UnresolvableReference
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.references import (
    require_cell,
    resolve_reading,
)
from threetears.evals.analysis.stats import INTERVAL_LEVEL, t_critical_two_sided
from threetears.evals.contracts.analysis_measures import MeasureCollection, MeasureSummary
from threetears.evals.contracts.surface import (
    CellFacts,
    DecisionSurface,
    JudgedDimensionFacts,
    JudgedReading,
    MeasureFacts,
)


_REF = cell_ref("v1", "rig")


def _surface(*, n_independent: int = 4) -> DecisionSurface:
    cell = CellFacts(
        variant_key="v1",
        apparatus_class_id="rig",
        run_ids=["r1"],
        n_observations=4,
        measures=MeasureCollection(
            measures=[
                MeasureSummary(
                    population="scored",
                    name="total_ms",
                    attribution_scope="end_to_end",
                    higher_is_better=False,
                    n=4,
                    n_independent=n_independent,
                    mean=1200.0,
                    p50=1100.0,
                    p95=1900.0,
                    sem=50.0,
                    ci_low=1040.9,
                    ci_high=1359.1,
                ),
                MeasureSummary(
                    population="scored",
                    name="stop_reason",
                    attribution_scope="end_to_end",
                    n=4,
                    categories={"end_turn": 3, "max_tokens": 1},
                ),
                MeasureSummary(
                    population="scored",
                    name="meanless",
                    attribution_scope="end_to_end",
                    higher_is_better=True,
                    n=1,
                    p50=3.0,
                ),
            ]
        ),
        judged=[JudgedReading(dimension="reply.grounding", mean=4.0, sem=0.25, n=4, n_independent=4)],
    )
    return DecisionSurface(
        cells=[cell],
        measures={"total_ms": MeasureFacts(unit="ms", merit_axis="latency", higher_is_better=False)},
        dimensions={"reply.grounding": JudgedDimensionFacts(higher_is_better=True, value_range=(1.0, 5.0))},
    )


def test_a_measure_reads_its_mean_basis_and_frozen_facts():
    reading = resolve_reading(_surface(), _REF, "total_ms")

    assert (reading.mean, reading.n, reading.sem) == (1200.0, 4, 50.0)
    assert (reading.ci_low, reading.ci_high) == (1040.9, 1359.1)
    assert (reading.unit, reading.merit_axis, reading.higher_is_better) == ("ms", "latency", False)
    assert reading.dispersion == "sem 50; 95% CI [1041, 1359]"


def test_clustered_observations_are_said_to_narrow_the_interval():
    reading = resolve_reading(_surface(n_independent=2), _REF, "total_ms")

    assert reading.dispersion.endswith("; 4 obs over 2 cases, interval too narrow")


def test_a_judged_reading_takes_a_t_interval_and_the_quality_axis():
    reading = resolve_reading(_surface(), _REF, "reply.grounding", "judged")

    half = t_critical_two_sided(INTERVAL_LEVEL, 3) * 0.25
    assert reading.ci_low == pytest.approx(4.0 - half)
    assert reading.ci_high == pytest.approx(4.0 + half)
    assert (reading.merit_axis, reading.higher_is_better, reading.unit) == ("quality", True, None)


def test_an_unresolvable_reference_is_a_repairable_refusal():
    assert issubclass(UnresolvableReference, SoundnessRefusal)


def test_an_unknown_cell_names_the_cells_that_exist():
    with pytest.raises(UnresolvableReference, match=r"v1:rig"):
        require_cell(_surface(), "nope:rig")


def test_a_judged_name_read_as_a_measure_says_which_namespace_it_is_in():
    with pytest.raises(UnresolvableReference, match=r"IS a judged dimension .* set reading to 'judged'"):
        resolve_reading(_surface(), _REF, "reply.grounding")


def test_a_measure_name_read_as_judged_says_which_namespace_it_is_in():
    with pytest.raises(UnresolvableReference, match=r"IS a measure .* set reading to 'measure'"):
        resolve_reading(_surface(), _REF, "total_ms", "judged")


def test_a_categorical_measure_has_no_point_estimate():
    with pytest.raises(UnresolvableReference, match=r"categorical"):
        resolve_reading(_surface(), _REF, "stop_reason")


def test_a_numeric_measure_with_no_mean_is_refused():
    with pytest.raises(UnresolvableReference, match=r"no mean"):
        resolve_reading(_surface(), _REF, "meanless")


def test_a_caller_that_bypasses_the_reference_models_is_a_programming_error_not_a_refusal():
    """The repair round cannot correct a kind no model output can reach the resolver with."""
    with pytest.raises(ValueError, match=r"neither 'measure' nor 'judged'") as raised:
        resolve_reading(_surface(), _REF, "total_ms", "telemetry")  # type: ignore[arg-type]
    assert not isinstance(raised.value, SoundnessRefusal)


class TestAnEntryTheSurfaceGuaranteesIsReadNotDefaulted:
    """Every name a cell carries has a facts entry; a surface missing one is broken, not the model's to fix.

    A default here would state a figure with no unit, on no axis, or call a judged dimension of
    unknown polarity higher-is-better — and a merit claim would then be refused as "on no axis of
    merit", pointing the repair round at the wrong fix.
    """

    def test_a_measure_with_no_facts_entry_is_a_hard_error_naming_it(self):
        broken = _surface().model_copy(update={"measures": {}})
        with pytest.raises(RuntimeError, match=r"measure 'total_ms'.*no `measures` entry") as raised:
            resolve_reading(broken, _REF, "total_ms")
        assert not isinstance(raised.value, SoundnessRefusal), "a repair round would meet the same surface"

    def test_a_judged_dimension_with_no_facts_entry_is_a_hard_error_naming_it(self):
        broken = _surface().model_copy(update={"dimensions": {}})
        with pytest.raises(RuntimeError, match=r"judged dimension 'reply.grounding'.*no `dimensions` entry") as raised:
            resolve_reading(broken, _REF, "reply.grounding", "judged")
        assert not isinstance(raised.value, SoundnessRefusal)

    def test_a_lower_is_better_dimension_reads_its_own_polarity(self):
        """The positive half: the entry is read, so a declared polarity reaches the reading."""
        declared = _surface().model_copy(
            update={
                "dimensions": {"reply.grounding": JudgedDimensionFacts(higher_is_better=False, value_range=(1.0, 5.0))}
            }
        )
        assert resolve_reading(declared, _REF, "reply.grounding", "judged").higher_is_better is False


class TestOneLevelComputesAndLabelsEveryInterval:
    """The level an interval is captioned with is the level its width was computed at — one constant.

    Moving `stats.INTERVAL_LEVEL` must move the judged width AND the caption together; a module that
    kept its own copy of either would leave one of them behind, which is the mislabelled interval.
    """

    def test_a_judged_interval_and_its_caption_follow_the_one_constant(self, monkeypatch):
        monkeypatch.setattr(stats, "INTERVAL_LEVEL", 0.8)
        reading = resolve_reading(_surface(), _REF, "reply.grounding", "judged")

        half = t_critical_two_sided(0.8, 3) * 0.25
        assert (reading.ci_low, reading.ci_high) == (pytest.approx(4.0 - half), pytest.approx(4.0 + half))
        assert "; 80% CI [" in reading.dispersion

    def test_a_measure_is_captioned_at_the_one_constant(self, monkeypatch):
        monkeypatch.setattr(stats, "INTERVAL_LEVEL", 0.8)
        assert "; 80% CI [" in resolve_reading(_surface(), _REF, "total_ms").dispersion


def test_a_value_and_its_interval_are_spelled_by_one_rule_on_one_line():
    """An arm-table cell prints the value and the resolver's dispersion side by side.

    Under two formatters a 12345.6 ms latency read ``1.235e+04`` in the value slot beside an interval
    spelled ``12,346`` — one magnitude, two spellings, on one line.
    """
    surface = _surface()
    [latency] = [m for m in surface.cells[0].measures.measures if m.name == "total_ms"]
    half = stats.ci_half_width(1234.0, 4)
    surface.cells[0].measures.measures[0] = latency.model_copy(
        update={"mean": 12345.6, "sem": 1234.0, "ci_low": 12345.6 - half, "ci_high": 12345.6 + half}
    )
    reading = resolve_reading(surface, _REF, "total_ms")

    # The value and its interval, side by side as a report prints them, in one spelling.
    cell = f"{format_number(reading.mean)} (n={reading.n}, {reading.dispersion})"

    assert cell == f"12346 (n=4, sem 1234; 95% CI [{round(12345.6 - half)}, {round(12345.6 + half)}])"
