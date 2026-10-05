"""A difference below a measure's declared materiality threshold is labelled immaterial — wherever one is stated.

``MetricDescriptor.materiality_threshold`` is the host's word for "smaller than this is not worth acting
on". It is read by one predicate, :func:`~threetears.evals.contracts.materiality`, at the two places the
engine states a difference:

* **the analysis bundle**, where every :class:`~threetears.evals.analysis.MeasureMovement` between two
  levels of a lever carries ``materiality`` — the input a generator's caveats are written from;
* **the decision surface**, which freezes each measure's threshold beside its facts, so a delta table
  drawn from it labels a row ``immaterial`` in its values table and names it in a disclosure.

Silence is conservative: a measure that declared no threshold has every difference material.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import ValidationError

from threetears.evals.analysis.bundle import _movement, cell_measure_facts
from threetears.evals.analysis.viz.compiler import compile_chart
from threetears.evals.contracts import MetricDescriptor, materiality
from threetears.evals.contracts.analysis_measures import MeasureSummary
from threetears.evals.contracts.host import MeasureRegistry
from threetears.evals.contracts.surface import MeasureFacts
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle
from packages.evals.tests.fixtures.toyhost.kind import FIELD_ACCURACY
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES, toyhost_profile
from packages.evals.tests.test_viz_refs import _measure_facts, build, chart, ref, surface


def _latency(threshold: float | None) -> MetricDescriptor:
    return MetricDescriptor(
        name="total_ms",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description="Wall-clock.",
        higher_is_better=False,
        unit="ms",
        materiality_threshold=threshold,
    )


def _summary(mean: float) -> MeasureSummary:
    return MeasureSummary(
        name="total_ms",
        attribution_scope="end_to_end",
        higher_is_better=False,
        population="all_observed",
        n=20,
        mean=mean,
        sem=1.0,
    )


# --- the predicate -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("threshold", "delta", "expected"),
    [
        (50.0, 49.9, "immaterial"),
        (50.0, -49.9, "immaterial"),
        (50.0, 50.0, "material"),
        (50.0, -120.0, "material"),
        (None, 0.0001, "material"),
    ],
)
def test_the_predicate_reads_the_threshold_and_ignores_the_sign(
    threshold: float | None, delta: float, expected: str
) -> None:
    assert materiality(threshold, delta) == expected


def test_a_negative_threshold_is_refused() -> None:
    with pytest.raises(ValidationError):
        _latency(-1.0)
    with pytest.raises(ValidationError):
        MeasureFacts(materiality_threshold=-1.0)


# --- the bundle labels a movement ---------------------------------------------------------------------


def test_a_movement_that_clears_its_noise_and_not_its_threshold_is_immaterial() -> None:
    """The two readings are independent: improved against noise, and still too small to act on."""
    movement = _movement(_latency(50.0), _summary(1000.0), _summary(970.0))

    assert movement.direction == "improved"
    assert movement.materiality == "immaterial"


def test_a_movement_on_a_measure_with_no_threshold_is_material() -> None:
    assert _movement(_latency(None), _summary(1000.0), _summary(999.0)).materiality == "material"


def test_a_movement_at_or_beyond_the_threshold_is_material() -> None:
    assert _movement(_latency(50.0), _summary(1000.0), _summary(900.0)).materiality == "material"


# --- the decision surface carries the threshold, and a delta drawn from it is labelled ------------------


def test_the_decision_surface_freezes_each_measures_threshold() -> None:
    measures = tuple(
        descriptor.model_copy(update={"materiality_threshold": 0.02})
        if descriptor.name == FIELD_ACCURACY
        else descriptor
        for descriptor in TOYHOST_MEASURES
    )
    profile = replace(
        toyhost_profile(), measures=MeasureRegistry(measures, families=toyhost_profile().measures.families)
    )

    facts = cell_measure_facts(toyhost_bundle(profile=profile))

    assert facts[FIELD_ACCURACY].materiality_threshold == 0.02
    assert facts[FIELD_ACCURACY].population == "scored"


def _delta_table(facts: dict[str, MeasureFacts]):
    payload = build(chart("delta_table", [ref("A"), ref("B")], ["pass_rate", "total_ms"]), surface(facts=facts))
    return payload, compile_chart("delta_table", payload)


def test_a_delta_below_its_threshold_is_labelled_immaterial_on_the_surface() -> None:
    """``total_ms`` moves 200ms between A and B, under a 250ms threshold; ``pass_rate`` moves 0.2, over 0.05."""
    facts = _measure_facts()
    facts["total_ms"] = facts["total_ms"].model_copy(update={"materiality_threshold": 250.0})
    facts["pass_rate"] = facts["pass_rate"].model_copy(update={"materiality_threshold": 0.05})

    payload, compiled = _delta_table(facts)

    rows = {row["metric"]: row for row in payload["rows"]}
    assert (rows["total_ms"]["materiality"], rows["pass_rate"]["materiality"]) == ("immaterial", "material")
    table = {row["metric"]: row for row in compiled.rows}
    assert table["total_ms"]["delta"].endswith("(immaterial)")
    assert not table["pass_rate"]["delta"].endswith("(immaterial)")
    (disclosure,) = [d for d in compiled.disclosures if "materiality threshold" in d]
    assert "1 of 2 changes" in disclosure and "total_ms" in disclosure


def test_a_surface_with_no_thresholds_labels_nothing_immaterial() -> None:
    payload, compiled = _delta_table(_measure_facts())

    assert {row["materiality"] for row in payload["rows"]} == {"material"}
    assert not any("materiality threshold" in d for d in compiled.disclosures)
